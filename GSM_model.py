"""Global State Model with shared Top-K selection and concatenated latent queries.

The encoder retains the baseline topology and parameters. Latent reads follow
selected encoder blocks and accumulate history information in FP32. Decoder
layers use query and output projections to read a shared, uncompressed memory
window. Learned linear operations use the baseline parameter dtype.
"""
import copy
import hashlib
import math
import torch
from torch import nn
from torch.nn import functional as F
from .multimodal import MultimodalBaseline
from .text_model import norm, linear, rotate, quantize_cache, gather, selected_math, NativeSelected, NativeHC, Result
from .math_ops import hc_pre, hc_post, hc_split
from .state import Bank, RequestState, text_positions, window_indices, Positions

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
            # Initialize later-stage queries deterministically.
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
        # Reuse the initialized projection for uncompressed shared memory.
        self.wkv = first_decoder.attn.compressor.wkv
        self.kv_norm = first_decoder.attn.compressor.norm

def lift(memory, mult):
    streams = memory.unsqueeze(2).expand(-1, -1, mult, -1)
    pre = memory.new_zeros((*memory.shape[:2], mult), dtype=torch.float32)
    pre[..., 0] = 1
    return (streams, pre)

class GSM(MultimodalBaseline):

    def __init__(self, config, latent_layers=(7, 13, 19), memory_window=None, encoder_layers=20, **kwargs):
        # Supported encoder and latent schedules use zero-based layer indices.
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
            # Reuse the encoder attention window and selection from this layer.
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
                # Separate branch gradients before accumulation at the source
                # activation to preserve the BF16 reduction structure.
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
