"""Text baseline and Global State Model (GSM).

The GSM encoder aggregates history through shared sparse selections. Decoder
layers read one uncompressed, bounded shared state using query/output projections.
Refinement and memory fusion use FP32; learned projections use BF16 weights.

Integration
-----------
Save this module as ``text_model.py`` inside the existing model package. Import
``TextBaseline`` and ``GSM`` from this module. GSM is resolved lazily so that the
external multimodal adapter can import TextBaseline without a circular import.

This file merges the supplied implementations, not their external dependencies.
The package must also provide reference, math_ops, state, losses,
attention_autograd, forward_metadata, and multimodal. Optional execution paths
require inference_ops, tensorcore_head, fused_hc, fused_hc_backward,
compact_attention, and segmented_gather. Update callers and any backend checks
in companion modules to use the anonymous execution-backend name ``fused``.

The reference backend, parameter names, tensor operations, topology schedules,
and deterministic initialization seeds are preserved. Native kernels and model
construction still require the accompanying package and a compatible runtime.
"""
from dataclasses import dataclass
import copy
import hashlib
import math
import torch
from torch import nn
from torch.nn import functional as F
from .reference import build_text, deterministic_init, load_official
from .math_ops import hc_split, hc_pre, hc_post
from .state import Positions, Bank, Selection, RequestState, text_positions, causal, window_indices
from .losses import compressed_teacher, indexer_kl_sum
from .attention_autograd import released_attention_emulation

def norm(x, module):
    if hasattr(module, 'gsm_norm'):
        return module.gsm_norm(x)
    y = x.float()
    return (y * torch.rsqrt(y.square().mean(-1, keepdim=True) + module.eps) * module.weight).to(x.dtype)

def linear(x, module):
    if hasattr(module, 'gsm_linear'):
        return module.gsm_linear(x)
    return F.linear(x, module.weight)

def rotate(x, freqs, inverse=False):
    rd = freqs.shape[-1] * 2
    z = torch.view_as_complex(x[..., -rd:].float().reshape(*x.shape[:-1], rd // 2, 2))
    f = freqs.conj() if inverse else freqs
    if x.ndim == 4:
        f = f.unsqueeze(-2)
    z = torch.view_as_real(z * f).flatten(-2).to(x.dtype)
    return torch.cat((x[..., :-rd], z), -1)

def quantize_cache(x, kind, enabled):
    if not enabled or x.numel() == 0:
        return x
    o = load_official()
    with torch.no_grad():
        y = x.detach().contiguous().clone()
        if kind == 'swa':
            o.act_quant(y, 32, 'ue8m0', torch.float8_e8m0fnu, True)
        elif kind == 'main':
            o.fp4_act_quant(y, 16, True, scale_dtype=torch.float8_e4m3fn)
        elif kind == 'index':
            o.fp4_act_quant(y, 32, True)
        else:
            raise ValueError(kind)
    # explicit QAT surrogate, not finite-difference claim
    return x + (y - x).detach()

def selected_probabilities(q, values, valid, sink):
    """Original FP32 probability calculation, without the unused value product."""
    logits = torch.einsum('bshd,bskd->bshk', q.float(), values.float()) * q.shape[-1] ** (-0.5)
    logits = logits.masked_fill(~valid[:, :, None, :], -torch.inf)
    p = torch.cat((logits, sink.view(1, 1, -1, 1).expand(*q.shape[:3], 1)), -1).softmax(-1)[..., :values.shape[2]]
    return p

def selected_math(q, values, valid, sink):
    p = selected_probabilities(q, values, valid, sink)
    return (torch.einsum('bshk,bskd->bshd', p, values.float()).to(q.dtype), p)

class NativeSelected(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, values, valid, sink):
        b, s, h, d = q.shape
        k = values.shape[2]
        ids = torch.arange(k, device=q.device, dtype=torch.int32).view(1, 1, -1).expand(b * s, 1, -1).clone()
        ids.masked_fill_(~valid.reshape(b * s, 1, k), -1)
        out = load_official().sparse_attn(q.reshape(b * s, 1, h, d), values.reshape(b * s, k, d), sink, ids, d ** (-0.5)).reshape(b, s, h, d)
        ctx.save_for_backward(q, values, valid, sink)
        return out

    @staticmethod
    def backward(ctx, grad):
        q, v, valid, sink = ctx.saved_tensors
        with torch.enable_grad():
            qq = q.detach().float().requires_grad_()
            vv = v.detach().float().requires_grad_()
            ss = sink.detach().requires_grad_()
            out = released_attention_emulation(qq, vv, valid, ss)
            grads = torch.autograd.grad(out, (qq, vv, ss), grad.float())
        return (grads[0].to(q.dtype), grads[1].to(v.dtype), None, grads[2])

class NativeHC(torch.autograd.Function):

    @staticmethod
    def forward(ctx, mix, scale, base, mult, iters, eps):
        ctx.save_for_backward(mix, scale, base)
        ctx.args = (mult, iters, eps)
        return load_official().hc_split_sinkhorn(mix, scale, base, mult, iters, eps)

    @staticmethod
    def backward(ctx, gpre, gpost, gcomb):
        mix, scale, base = ctx.saved_tensors
        with torch.enable_grad():
            inputs = tuple((x.detach().requires_grad_() for x in (mix, scale, base)))
            outputs = hc_split(*inputs, *ctx.args)
            grads = torch.autograd.grad(outputs, inputs, (gpre, gpost, gcomb))
        return (*grads, None, None, None)

def gather(bank, slots):
    if bank.value.shape[1] == 0:
        return bank.value.new_zeros((*slots.shape, bank.value.shape[-1]))
    b = torch.arange(slots.shape[0], device=slots.device)[:, None, None]
    return bank.value[b, slots.clamp_min(0)].masked_fill((slots < 0)[..., None], 0)

@dataclass
class Result:
    logits: torch.Tensor
    hidden: torch.Tensor
    auxiliary_sum: torch.Tensor
    state: RequestState | None
    routing_counts: dict
    trace: dict

class TextBaseline(nn.Module):

    def __init__(self, config, device='cuda', quantized_cache=True, native_attention=True, activation_checkpointing=False, gather_backend='reference', execution_backend='reference', attention_backward_precision='tf32x3', logits_backend='fp32'):
        super().__init__()
        self.config = copy.deepcopy(config)
        from .forward_metadata import ReferenceMetadata
        self.metadata = ReferenceMetadata()
        if torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1:
            raise RuntimeError('Construct full reference replicas before joining DP: the pinned inference constructor interprets the default process group as TP')
        if config['dtype'] != 'bf16' or config.get('expert_dtype') is not None:
            raise ValueError('This training reference requires explicit BF16 stored-weight profile')
        self.weights = deterministic_init(build_text(config, device))
        self.quantized_cache = quantized_cache
        self.native_attention = native_attention
        self.activation_checkpointing = activation_checkpointing
        if gather_backend not in ('reference', 'segmented'):
            raise ValueError('Unknown bank gather backend')
        self.gather_backend = gather_backend
        if execution_backend not in ('reference', 'fused'):
            raise ValueError('Unknown execution backend')
        if execution_backend == 'fused' and (not native_attention):
            raise ValueError('Fused execution requires native BF16')
        if attention_backward_precision not in ('tf32x3', 'bf16x2'):
            raise ValueError('Unknown attention backward precision')
        if attention_backward_precision != 'tf32x3' and execution_backend != 'fused':
            raise ValueError('BF16x2 attention backward requires fused execution')
        if logits_backend not in ('fp32', 'tensorcore_bf16x3'):
            raise ValueError('Unknown logits backend')
        if logits_backend != 'fp32' and (execution_backend != 'fused' or not native_attention):
            raise ValueError('Tensor-core logits require fused native BF16')
        self.logits_backend = logits_backend
        self.attention_backward_precision = attention_backward_precision
        self.execution_backend = execution_backend
        self.aux_policy = 'selected_support_kl_full_reindex_only_v1'
        # Delete every inference-state buffer. Only immutable rotary tables remain.
        for module in self.weights.modules():
            for name in list(module._buffers):
                if name.endswith('_cache') or name in ('kv_state', 'score_state'):
                    delattr(module, name)
        for layer in self.weights.layers:
            layer.ffn.gate.bias.requires_grad_(False)
            if layer.ffn.gate.bias_vl is not None:
                layer.ffn.gate.bias_vl.requires_grad_(False)

    def project_logits(self, hidden):
        if hasattr(self.weights.head, 'gsm_logits'):
            if self.logits_backend != 'fp32':
                raise ValueError('Tensor-core logits with FSDP are not supported')
            return self.weights.head.gsm_logits(hidden)
        if self.logits_backend == 'tensorcore_bf16x3':
            from .tensorcore_head import logits
            return logits(hidden, self.weights.head.weight)
        return F.linear(hidden.float(), self.weights.head.weight.float())

    def pre_hc(self, h, pre):
        if self.execution_backend == 'fused':
            from .fused_hc import hc_pre as fused
            return fused(h, pre)
        return hc_pre(h, pre)

    def post_hc(self, out, h, post, comb):
        if self.execution_backend == 'fused':
            from .fused_hc import hc_post as fused
            return fused(out, h, post, comb)
        return hc_post(out, h, post, comb)

    def split_hc(self, *args):
        if self.execution_backend == 'fused':
            from .fused_hc_backward import FusedHC
            return FusedHC.apply(*args)
        return (NativeHC.apply if self.native_attention else hc_split)(*args)

    def attention_values(self, window, wi, bank=None, slots=None):
        values = self.gather(window, wi)
        valid = wi >= 0
        if bank is not None:
            values = torch.cat((values, self.gather(bank, slots)), 2)
            valid = torch.cat((valid, slots >= 0), -1)
        return (values, valid)

    def attend_banks(self, q, window, wi, sink, bank=None, slots=None):
        if self.execution_backend == 'fused':
            from .compact_attention import compact_attention
            value = window.value
            indices = wi
            if bank is not None:
                # Preserve window/compressed support order and -1 sentinels.
                indices = torch.cat((wi, torch.where(slots >= 0, slots + value.shape[1], -1)), -1)
                value = torch.cat((value, bank.value), 1)
            return compact_attention(q, value, indices, sink, precision=self.attention_backward_precision, batch_rows=getattr(self, 'attention_backward_batch_rows', 0), split_backward=getattr(self, 'attention_backward_split', False))
        values, valid = self.attention_values(window, wi, bank, slots)
        return NativeSelected.apply(q, values, valid, sink) if self.native_attention else selected_math(q, values, valid, sink)[0]

    def gather(self, bank, slots):
        if self.gather_backend == 'segmented':
            from .segmented_gather import segmented_gather
            return segmented_gather(bank.value, slots)
        return gather(bank, slots)

    def frequencies(self, a, pos):
        if pos.rope.ndim != 2:
            raise NotImplementedError('Multimodal rotary positions require the multimodal adapter')
        return a.freqs_cis[pos.rope]

    def query(self, a, x, pos):
        qr = norm(linear(x, a.wq_a), a.q_norm)
        q = linear(qr, a.wq_b).unflatten(-1, (a.n_heads, a.head_dim))
        return (qr, rotate(q, self.frequencies(a, pos)))

    def output(self, a, out, pos):
        out = rotate(out, self.frequencies(a, pos), True)
        out = out.reshape(*out.shape[:2], a.n_groups, -1)
        if hasattr(a.wo_a, 'gsm_grouped_linear'):
            projected = a.wo_a.gsm_grouped_linear(out, a.n_groups, a.o_lora_rank)
        else:
            wa = a.wo_a.weight.reshape(a.n_groups, a.o_lora_rank, -1)
            projected = torch.einsum('bsgd,grd->bsgr', out, wa).flatten(2)
        return linear(projected, a.wo_b)

    def compress(self, a, x, pos, state):
        cp = a.compressor
        ratio = a.compress_ratio
        b, s, d = x.shape
        vals = linear(x.float() if ratio > 1 else x, cp.wkv)
        scores = linear(x.float(), cp.wgate) if ratio > 1 else None
        allvalues = []
        allanchors = []
        allcomplete = []
        allropes = []
        alldocs = []
        pending = []
        for bi in range(b):
            v = vals[bi]
            sc = None if scores is None else scores[bi]
            source = pos.source[bi]
            rope = pos.rope[bi]
            doc = pos.document[bi]
            valid = pos.valid[bi]
            previous = None if state is None else state.pending.get(a.layer_id)
            if previous is not None and previous[bi] is not None:
                pv, ps, pm = previous[bi]
                v = torch.cat((pv, v))
                sc = torch.cat((ps, sc))
                source = torch.cat((pm.source[0], source))
                rope = torch.cat((pm.rope[0], rope))
                doc = torch.cat((pm.document[0], doc))
                valid = torch.cat((pm.valid[0], valid))
            vv = []
            aa = []
            cc = []
            rr = []
            dd = []
            tail = None
            used = valid.nonzero().flatten()
            # Source docs must be unique contiguous segments within a request.
            for segment in doc[used].unique_consecutive():
                ix = used[doc[used] == segment]
                n = len(ix) // ratio * ratio
                grouped = ix[:n].reshape(-1, ratio)
                if n:
                    pooled = v[grouped].squeeze(1) if ratio == 1 else (v[grouped] * sc[grouped].softmax(1)).sum(1)
                    vv.append(pooled)
                    aa.append(source[grouped[:, 0]])
                    cc.append(source[grouped[:, -1]])
                    rr.append(rope[grouped[:, 0]])
                    dd.append(doc[grouped[:, 0]])
                remain = ix[n:]
                if len(remain) and segment == doc[used[-1]]:
                    pm = Positions(source[remain][None], rope[remain][None], doc[remain][None], valid[remain][None])
                    tail = (v[remain], sc[remain], pm)
            allvalues.append(torch.cat(vv) if vv else v[:0])
            allanchors.append(torch.cat(aa) if aa else source[:0])
            allcomplete.append(torch.cat(cc) if cc else source[:0])
            allropes.append(torch.cat(rr) if rr else rope[:0])
            alldocs.append(torch.cat(dd) if dd else doc[:0])
            pending.append(tail)
        maxn = max(map(len, allvalues))
        pv = []
        ps = []
        pr = []
        pd = []
        pc = []
        valid = []
        for bi, v in enumerate(allvalues):
            n = len(v)
            pad = maxn - n
            pv.append(F.pad(v, (0, 0, 0, pad)))
            ps.append(F.pad(allanchors[bi], (0, pad), value=2 ** 60))
            pr.append(F.pad(allropes[bi], (0, pad) if pos.rope.ndim == 2 else (0, 0, 0, pad)))
            pd.append(F.pad(alldocs[bi], (0, pad), value=-1))
            pc.append(F.pad(allcomplete[bi], (0, pad), value=2 ** 60))
            valid.append(torch.arange(maxn, device=x.device) < n)
        latent = norm(torch.stack(pv).to(x.dtype), cp.norm)
        meta = Positions(torch.stack(ps), torch.stack(pr), torch.stack(pd), torch.stack(valid))
        idx = a.indexer
        k = norm(linear(latent.detach(), idx.wk), idx.k_norm)
        k = quantize_cache(rotate(k, self.frequencies(a, meta)), 'index', self.quantized_cache)
        bank = Bank(quantize_cache(rotate(latent, self.frequencies(a, meta)), 'main', self.quantized_cache), meta, torch.stack(pc), a.layer_id, 0 if state is None else state.generation, k)
        if state is not None:
            old = state.banks.get(a.layer_id)
            # Batched ragged append is handled per request; prohibit padded persistent banks.
            if b != 1:
                raise ValueError('Persistent state uses one request; batch coordinator is separate')
            if old is not None:
                bank = old.cat(bank)
            state.banks[a.layer_id] = bank
            state.pending[a.layer_id] = pending
        return bank

    def index(self, a, x, qr, pos, bank, candidates):
        tile = getattr(self, 'inference_index_tile', 0)
        if not self.training and (not torch.is_grad_enabled()) and (tile > 0) and (x.shape[1] > tile):
            from .inference_ops import tiled_index
            return tiled_index(self, a, x, qr, pos, bank, candidates, tile)
        return self._reference_index(a, x, qr, pos, bank, candidates)

    def _reference_index(self, a, x, qr, pos, bank, candidates):
        ix = a.indexer
        iq = linear(qr.detach(), ix.wq_b).unflatten(-1, (ix.n_heads, ix.index_head_dim))
        iq = quantize_cache(rotate(iq, self.frequencies(a, pos)), 'index', self.quantized_cache)
        weights = linear(x.detach(), ix.weights_proj) * (ix.softmax_scale * ix.n_heads ** (-0.5))
        score = (torch.einsum('bshd,btd->bsht', iq, bank.index_key).relu() * weights.unsqueeze(-1)).sum(2)
        legal = causal(pos, bank)
        score = score.masked_fill(~legal, -torch.inf)
        if a.layer_id == self.config['candidate_source_layer']:
            bs = self.config['candidate_block_size']
            nb = (score.shape[-1] + bs - 1) // bs

            def pool(local_score, local_legal):
                size = local_score.shape[-1]
                blocks_n = (size + bs - 1) // bs
                blocks = F.pad(local_score, (0, -size % bs), value=-torch.inf).unflatten(-1, (-1, bs)).amax(-1)
                last = torch.where(local_legal, torch.arange(size, device=x.device), -1).amax(-1) // bs
                blocks = blocks.masked_fill(torch.arange(blocks_n, device=x.device) == last[..., None], torch.inf)
                top = blocks.topk(min(self.config['candidate_topk_blocks'], blocks_n), -1)
                return torch.zeros_like(blocks, dtype=torch.bool).scatter_(-1, top.indices, top.values > -torch.inf).repeat_interleave(bs, -1)[..., :size]
            packed = any((bank.positions.document[b, bank.positions.valid[b]].unique().numel() > 1 for b in range(x.shape[0])))
            if not packed:
                candidates = pool(score, legal)
            else:
                # Candidate blocks restart at each document, just like the
                # compressed groups. Physical concatenation must not shift the
                # second document's blocks or change its Reindex support.
                candidates = torch.zeros_like(legal)
                for b in range(x.shape[0]):
                    for doc in bank.positions.document[b, bank.positions.valid[b]].unique_consecutive():
                        rows = ((pos.document[b] == doc) & pos.valid[b]).nonzero().flatten()
                        slots = ((bank.positions.document[b] == doc) & bank.positions.valid[b]).nonzero().flatten()
                        if rows.numel() and slots.numel():
                            chosen = pool(score[b][rows][:, slots], legal[b][rows][:, slots])
                            candidates[b, rows[:, None], slots[None, :]] = chosen
        elif a.layer_id > self.config['candidate_source_layer']:
            legal = legal & candidates
            score = score.masked_fill(~legal, -torch.inf)
        k = min(ix.index_topk, bank.value.shape[1])
        slots = score.detach().topk(k, -1, sorted=False).indices.sort(-1).values
        valid = legal.gather(-1, slots)
        # Canonical order: valid physical slots ascending, then invalid padding.
        # Otherwise packed prior-document slots with -inf scores can land before
        # valid rows, changing native BF16 GEMM reduction placement for the same
        # logical attention support.
        slots = torch.where(valid, slots, bank.value.shape[1]).sort(-1).values
        valid = slots < bank.value.shape[1]
        slots = slots.masked_fill(~valid, -1)
        student = score.gather(-1, slots.clamp_min(0)).masked_fill(~valid, -1e+30)
        return (slots, student, candidates)

    def attention(self, a, x, pos, shared, state, request, trace):
        qr, q = self.query(a, x, pos)
        value = quantize_cache(rotate(norm(linear(x, a.wkv), a.kv_norm), self.frequencies(a, pos)), 'swa', self.quantized_cache)
        window = Bank(value, pos, pos.source, a.layer_id, 0 if state is None else state.generation)
        if state is not None:
            old = state.windows.get(a.layer_id)
            if old is not None:
                window = old.cat(window)
            # clone bounded tails to avoid retaining full prompt storage via a view
            tail = window.select(slice(-a.window_size, None))
            state.windows[a.layer_id] = Bank(tail.value.clone(), Positions(*(v.clone() for v in (tail.positions.source, tail.positions.rope, tail.positions.document, tail.positions.valid))), tail.complete.clone(), tail.owner, tail.generation)
        wi = window_indices(pos, window, a.window_size)
        student = None
        slots = None
        bank = None
        if a.is_kv_source:
            shared['bank'] = self.compress(a, x, pos, state)
        if a.compress_ratio:
            bank = shared['bank']
            if a.is_index_source:
                if bank.value.shape[1]:
                    slots, student, shared['candidates'] = self.index(a, x, qr, pos, bank, shared.get('candidates'))
                else:
                    slots = torch.empty((*x.shape[:2], 0), device=x.device, dtype=torch.long)
                shared['selection'] = Selection(slots, request, bank.owner, a.layer_id, bank.generation, pos.source, pos.document, bank.value.shape[1])
            selection = shared['selection']
            selection.validate(bank, pos, request)
            slots = selection.slots
        out = self.attend_banks(q, window, wi, a.attn_sink, bank, slots)
        aux = x.new_zeros((), dtype=torch.float32)
        if self.training and student is not None:
            with torch.no_grad():
                values, valid = self.attention_values(window, wi, bank, slots)
                prob = selected_probabilities(q.detach(), values, valid, a.attn_sink.detach())
                target = compressed_teacher(prob, wi.shape[-1])
            aux = indexer_kl_sum(student, target, slots >= 0, pos.valid)
        if trace is not None:
            trace[a.layer_id] = dict(selection=slots, bank=bank, window=window, query=q, attention_out=out)
        return (self.output(a, out, pos), aux)

    def moe(self, ffn, x, pos, counts):
        if not self.training and (not torch.is_grad_enabled()) and getattr(self, 'inference_expert_dispatch', False):
            from .inference_ops import expert_dispatch
            return expert_dispatch(self, ffn, x, pos, counts)
        return self._reference_moe(ffn, x, pos, counts)

    def _reference_moe(self, ffn, x, pos, counts):
        shape = x.shape
        flat = x.flatten(0, 1)
        gate = ffn.gate
        score = F.softplus(F.linear(flat.float(), gate.weight.float()) / gate.gate_temp).sqrt()
        ids = (score + self.routing_bias(gate, pos)).topk(gate.topk, -1).indices
        weights = score.gather(-1, ids)
        if gate.norm_topk_prob and gate.topk > 1:
            weights = weights / (weights.sum(-1, keepdim=True) + 1e-20)
        weights = weights * gate.route_scale
        counts[ffn.layer_id] = self.route_counts(ids, pos, ffn.n_routed_experts)

        def expert(mod, inp, weight=None):
            up = linear(inp, mod.w3).float()
            g = linear(inp, mod.w1).float()
            if mod.swiglu_limit > 0:
                up = up.clamp(-mod.swiglu_limit, mod.swiglu_limit)
                g = g.clamp_max(mod.swiglu_limit)
            z = F.silu(g) * up
            if weight is not None:
                z = z * weight
            return linear(z.to(inp.dtype), mod.w2)
        result = torch.zeros_like(flat, dtype=torch.float32)
        # Opt-in TP1 decode path: a single host transfer replaces a nonzero
        # synchronization for every expert. Keep ascending expert addition
        # order, selected rows/weights and the expert GEMMs unchanged.
        decode_routes = None
        if getattr(self, 'single_token_expert_dispatch', False) and len(flat) == 1:
            decode_routes = sorted(((expert_id, k) for k, expert_id in enumerate(ids[0].tolist())))
        dispatch = enumerate(ffn.experts) if decode_routes is None else ((i, ffn.experts[i]) for i, _ in decode_routes)
        for ordinal, (i, ex) in enumerate(dispatch):
            if decode_routes is None:
                rows, k = (ids == i).nonzero(as_tuple=True)
            else:
                rows = ids.new_zeros(1)
                k = ids.new_full((1,), decode_routes[ordinal][1])
            if rows.numel():
                result = result.index_add(0, rows, expert(ex, flat[rows], weights[rows, k, None]).float())
        result = result + expert(ffn.shared_experts, flat)
        return result.to(x.dtype).reshape(shape)

    def routing_bias(self, gate, pos):
        return gate.bias

    def route_counts(self, ids, pos, experts):
        return torch.bincount(ids[pos.valid.flatten()].flatten(), minlength=experts).detach()

    def prefill(self, ids, positions=None, *, keep_state=True, query_rows=None, inputs_embeds=None):
        """Generation prefill with a bounded inference path and a reference fallback."""
        from .inference_ops import prefill
        return prefill(self, ids, positions, keep_state=keep_state, query_rows=query_rows, inputs_embeds=inputs_embeds)

    def forward(self, ids, positions=None, state=None, keep_state=False, query_rows=None, trace=False, inputs_embeds=None):
        if self.training and self.activation_checkpointing:
            if state is not None or keep_state or query_rows is not None or trace:
                raise ValueError('Checkpointed training forbids persistent state, pruning and traces')
            from torch.utils.checkpoint import checkpoint
            # Whole-model reference checkpoint: each recomputation owns fresh banks,
            # selections and routing counts. No persistent state is committed here.
            return checkpoint(self._forward, ids, positions, inputs_embeds=inputs_embeds, use_reentrant=False)
        return self._forward(ids, positions, state, keep_state, query_rows, trace, inputs_embeds)

    def _forward(self, ids, positions=None, state=None, keep_state=False, query_rows=None, trace=False, inputs_embeds=None):
        if self.training and (state is not None or keep_state or query_rows is not None):
            raise ValueError('Training covers all positions and never mutates request state')
        pos = text_positions(ids, 0 if state is None else state.consumed) if positions is None else positions
        state = state.fork() if state is not None else RequestState('default') if keep_state else None
        request = 'training' if state is None else state.identity
        embedded = F.embedding(ids, self.weights.embed.weight) if inputs_embeds is None else inputs_embeds
        if embedded.shape != (*ids.shape, self.config['dim']):
            raise ValueError('Embedding shape mismatch')
        h = embedded.unsqueeze(2).repeat(1, 1, self.config['hc_mult'], 1)
        pre = h.new_zeros((*h.shape[:2], self.config['hc_mult']), dtype=torch.float32)
        pre[..., 0] = 1
        shared = {}
        aux = h.new_zeros((), dtype=torch.float32)
        counts = {}
        tr = {} if trace else None
        for layer in self.weights.layers:

            def mixes(h, kind):
                z = h.flatten(2).float()
                mix = F.linear(z, getattr(layer, 'hc_' + kind + '_fn')) * torch.rsqrt(z.square().mean(-1, keepdim=True) + layer.norm_eps)
                fn = self.split_hc
                return fn(mix, getattr(layer, 'hc_' + kind + '_scale'), getattr(layer, 'hc_' + kind + '_base'), layer.hc_mult, layer.hc_sinkhorn_iters, layer.hc_eps)
            ap, post, comb = mixes(h, 'attn')
            x = self.pre_hc(h.float(), pre).to(h.dtype)
            out, loss = self.attention(layer.attn, norm(x, layer.attn_norm), pos, shared, state, request, tr)
            h = self.post_hc(out, h, post, comb).to(out.dtype)
            aux = aux + loss
            fp, post, comb = mixes(h, 'ffn')
            x = self.pre_hc(h.float(), ap).to(h.dtype)
            out = self.moe(layer.ffn, norm(x, layer.ffn_norm), pos, counts)
            h = self.post_hc(out, h, post, comb).to(out.dtype)
            pre = fp
            if tr is not None:
                tr[layer.layer_id]['hidden'] = h.clone()
        hidden = norm(self.pre_hc(h.float(), pre).to(h.dtype), self.weights.norm)
        if query_rows is not None:
            hidden = hidden[:, query_rows]
        logits = self.project_logits(hidden)
        if state is not None:
            state.consumed += ids.shape[1]
        return Result(logits, hidden, aux, state, counts, tr or {})

class QueryOutput(nn.Module):

    def __init__(self, source):
        super().__init__()
        for name in ('wq_a', 'q_norm', 'wq_b', 'wo_a', 'wo_b'):
            setattr(self, name, copy.deepcopy(getattr(source, name)))
        self.attn_sink = nn.Parameter(source.attn_sink.detach().clone())
        for name in ('n_heads', 'head_dim', 'n_groups', 'o_lora_rank', 'layer_id'):
            setattr(self, name, getattr(source, name))
        self.register_buffer('freqs_cis', source.freqs_cis.clone(), persistent=False)

class Latent(QueryOutput):

    def __init__(self, source_block, stage, config):
        super().__init__(source_block.attn)
        d = config['dim']
        device = self.wq_a.weight.device
        dtype = self.wq_a.weight.dtype
        self.norm = copy.deepcopy(source_block.attn_norm) if stage == 0 else nn.RMSNorm(2 * d, eps=config['norm_eps'], device=device, dtype=dtype)
        if stage:
            self.wq_a = nn.Linear(2 * d, config['q_lora_rank'], bias=False, device=device, dtype=dtype)
            # Initialize later-stage queries independently of constructor RNG order.
            with torch.no_grad():
                self.q_norm.weight.fill_(1)
                for name, module in (('wq_a', self.wq_a), ('wq_b', self.wq_b)):
                    seed = int.from_bytes(hashlib.sha256(f'gsm.latent.{stage}.{name}'.encode()).digest()[:8], 'little') % (2 ** 63 - 1)
                    g = torch.Generator(device=device).manual_seed(seed)
                    value = torch.randn(module.weight.shape, generator=g, device=device, dtype=torch.float32) * (0.02 / math.sqrt(2) if name == 'wq_a' else 0.02)
                    module.weight.copy_(value)
                self.norm.weight.fill_(1)

    def forward(self, ops, hidden, refinement, pos, window, bank, selection, request, trace=None):
        ops.metadata.validate_selection(selection, bank, pos, request)
        joint = hidden if refinement is None else torch.cat((hidden, refinement), -1)
        x = norm(joint, self.norm).to(self.wq_a.weight.dtype)
        _, q = ops.query(self, x, pos)
        wi = ops.metadata.window_indices(pos, window, ops.config['window_size'])
        out = ops.attend_banks(q, window, wi, self.attn_sink, bank, selection.slots)
        update = ops.output(self, out, pos).float()
        result = update if refinement is None else refinement + update
        if trace is not None:
            trace.update(hidden=hidden, prior_refinement=refinement, joint=joint, normalized=x, selection=selection, bank=bank, window=window, update=update, refinement=result)
        return result

class SharedMemory(nn.Module):

    def __init__(self, first_decoder):
        super().__init__()
        self.norm = copy.deepcopy(first_decoder.attn_norm)
        # Move the already initialized ratio1 global projection into its new owner.
        self.wkv = first_decoder.attn.compressor.wkv
        self.kv_norm = first_decoder.attn.compressor.norm

def lift(memory, mult):
    streams = memory.unsqueeze(2).expand(-1, -1, mult, -1)
    pre = memory.new_zeros((*memory.shape[:2], mult), dtype=torch.float32)
    pre[..., 0] = 1
    return (streams, pre)

def _build_gsm_class():
    """Load the multimodal adapter after the text baseline is available."""
    from .multimodal import MultimodalBaseline

    class GSM(MultimodalBaseline):

        def __init__(self, config, latent_layers=(7, 13, 19), memory_window=None, encoder_layers=20, **kwargs):
            # Supported encoder and latent-stage schedules.
            expected = {(40, 20): ((7, 13, 19), [2, 8, 14, 20]), (28, 14): ((5, 9, 13), [2, 6, 10, 14])}
            schedule = expected.get((config['n_layers'], encoder_layers))
            if schedule is None or tuple(latent_layers) != schedule[0]:
                raise ValueError('Unsupported GSM encoder/latent topology')
            if config['kv_source_layers'] != schedule[1]:
                raise ValueError('KV owner schedule differs')
            super().__init__(config, **kwargs)
            self.encoder_layers = encoder_layers
            self.latent_layers = tuple(latent_layers)
            self.memory_window = config['window_size'] + config['index_topk'] if memory_window is None else memory_window
            if self.memory_window != config['window_size'] + config['index_topk']:
                raise ValueError('GSM window must equal W+K')
            self.latent = nn.ModuleList((Latent(self.weights.layers[i], stage, config) for stage, i in enumerate(self.latent_layers)))
            self.memory = SharedMemory(self.weights.layers[self.encoder_layers])
            for block in self.weights.layers[self.encoder_layers:]:
                block.attn = QueryOutput(block.attn)
            self.architecture_revision = f"gsm_csa2_shared_topk_concat_{config['n_layers']}_v1"

        def attention(self, a, x, pos, shared, state, request, trace):
            if a.layer_id < self.encoder_layers:
                # Capture only this layer's actual window/selection; no alternate lookup.
                local = {}
                out, loss = super().attention(a, x, pos, shared, state, request, local)
                shared['window'] = local[a.layer_id]['window']
                if trace is not None:
                    trace.update(local)
                return (out, loss)
            memory = shared['memory']
            _, q = self.query(a, x, pos)
            wi = window_indices(pos, memory, self.memory_window)
            out = self.attend_banks(q, memory, wi, a.attn_sink)
            if trace is not None:
                trace[a.layer_id] = {'memory': memory, 'memory_indices': wi, 'query': q, 'attention_out': out}
            return (self.output(a, out, pos), x.new_zeros((), dtype=torch.float32))

        def block(self, layer, h, pre, pos, shared, state, request, counts, trace):
            dtype = layer.attn.wq_a.weight.dtype

            def mixes(hidden, kind):
                z = hidden.flatten(2).float()
                mix = F.linear(z, getattr(layer, 'hc_' + kind + '_fn')) * torch.rsqrt(z.square().mean(-1, keepdim=True) + layer.norm_eps)
                fn = self.split_hc
                return fn(mix, getattr(layer, 'hc_' + kind + '_scale'), getattr(layer, 'hc_' + kind + '_base'), layer.hc_mult, layer.hc_sinkhorn_iters, layer.hc_eps)
            ap, post, comb = mixes(h, 'attn')
            x = self.pre_hc(h.float(), pre).to(dtype)
            out, aux = self.attention(layer.attn, norm(x, layer.attn_norm), pos, shared, state, request, trace)
            h = self.post_hc(out, h, post, comb).to(h.dtype)
            fp, post, comb = mixes(h, 'ffn')
            x = self.pre_hc(h.float(), ap).to(dtype)
            out = self.moe(layer.ffn, norm(x, layer.ffn_norm), pos, counts)
            h = self.post_hc(out, h, post, comb).to(h.dtype)
            if trace is not None:
                trace[layer.layer_id]['hidden'] = h.clone()
            return (h, fp, aux)

        def _forward(self, ids, positions=None, state=None, keep_state=False, query_rows=None, trace=False, inputs_embeds=None):
            if self.training and (state is not None or keep_state or query_rows is not None):
                raise ValueError('Training covers every legal target without persistent state')
            pos = text_positions(ids, 0 if state is None else state.consumed) if positions is None else positions
            state = state.fork() if state is not None else RequestState('default') if keep_state else None
            request = 'training' if state is None else state.identity
            embedded = F.embedding(ids, self.weights.embed.weight) if inputs_embeds is None else inputs_embeds
            if embedded.shape != (*ids.shape, self.config['dim']):
                raise ValueError('Embedding shape mismatch')
            h, pre = lift(embedded, self.config['hc_mult'])
            shared = {}
            refinement = None
            counts = {}
            tr = {'latent': []} if trace else None
            aux = h.new_zeros((), dtype=torch.float32)
            for layer in self.weights.layers[:self.encoder_layers]:
                h, pre, loss = self.block(layer, h, pre, pos, shared, state, request, counts, tr)
                aux = aux + loss
                if layer.layer_id in self.latent_layers:
                    stage = self.latent_layers.index(layer.layer_id)
                    record = {} if trace else None
                    # Explicit branch boundaries make the BF16 backward reduction
                    # independent of whether the latent branch runs on another GPU.
                    # Both branches first accumulate their own contributions; their
                    # two final gradients then meet at the original activation.
                    main_h, latent_h = (h.clone(), h.clone())
                    main_pre, latent_pre = (pre.clone(), pre.clone())
                    hidden = self.pre_hc(latent_h.float(), latent_pre)
                    refinement = self.latent[stage](self, hidden, refinement, pos, shared['window'], shared['bank'], shared['selection'], request, record)
                    h, pre = (main_h, main_pre)
                    if trace:
                        tr['latent'].append(record)
            memory_h = self.pre_hc(h.float(), pre) + refinement
            x = norm(memory_h, self.memory.norm).to(self.memory.wkv.weight.dtype)
            value = norm(linear(x, self.memory.wkv), self.memory.kv_norm)
            value = quantize_cache(rotate(value, self.frequencies(self.weights.layers[self.encoder_layers].attn, pos)), 'main', self.quantized_cache)
            bank = Bank(value, pos, pos.source, self.encoder_layers, 0 if state is None else state.generation)
            if state is not None:
                if ids.shape[0] != 1:
                    raise ValueError('Persistent memory belongs to a single request')
                if state.memory is not None:
                    bank = state.memory.cat(bank)
                tail = bank.select(slice(-self.memory_window, None))
                state.memory = Bank(tail.value.clone(), Positions(*(v.clone() for v in (tail.positions.source, tail.positions.rope, tail.positions.document, tail.positions.valid))), tail.complete.clone(), self.encoder_layers, tail.generation)
            selected = memory_h
            qm = pos
            if query_rows is not None:
                rows = torch.as_tensor(query_rows, device=ids.device, dtype=torch.long)
                if rows.ndim != 1 or not rows.numel() or bool(((rows < 0) | (rows >= ids.shape[1])).any()):
                    raise ValueError('Invalid Decoder query rows')
                selected = memory_h[:, rows]
                qm = pos.select(rows)
            h, pre = lift(selected, self.config['hc_mult'])
            if trace:
                tr.update(memory_h=memory_h, memory=bank, decoder_initial_fold=self.pre_hc(h, pre))
            shared = {'memory': bank}
            for layer in self.weights.layers[self.encoder_layers:]:
                h, pre, _ = self.block(layer, h, pre, qm, shared, None, request, counts, tr)
            hidden = norm(self.pre_hc(h.float(), pre).to(self.weights.head.weight.dtype), self.weights.norm)
            logits = self.project_logits(hidden)
            if state is not None:
                state.consumed += ids.shape[1]
            return Result(logits, hidden, aux, state, counts, tr or {})
    GSM.__qualname__ = 'GSM'
    return GSM

def __getattr__(name):
    """Expose GSM on demand without eagerly importing the multimodal adapter."""
    if name == 'GSM':
        model_class = _build_gsm_class()
        globals()[name] = model_class
        return model_class
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')

def __dir__():
    return sorted(set(globals()) | {'GSM'})
__all__ = ['TextBaseline', 'GSM', 'QueryOutput', 'Latent', 'SharedMemory', 'Result']
