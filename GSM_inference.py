"""Text inference with dependency-based prefill and request-owned state.

This module combines expert dispatch, tiled indexing, baseline prefill, GSM
prefill, and the guarded public prefill entry point. It must live inside the
model package: model definitions, state types, and tensor operators remain
package dependencies.

The anonymized fused backend identifier is "native_fused". Use the same name
in the backend registry and model configuration when integrating this module.
Callers of the former inference and prefill modules should import their entry
points from this module instead.

The original numerical eligibility checks, diagnostic opt-in, fallback path,
and verification context are retained. Verification labels describe the
source implementation's scope; merging this file does not establish new
numerical parity guarantees.
"""

import torch
from torch.nn import functional as F

from .gsm_model import GSM
from .reference import verified_official_scope
from .state import (
    Bank,
    Positions,
    RequestState,
    Selection,
    text_positions,
    window_indices,
)
from .text_model import Result, linear, norm, quantize_cache, rotate



def require_ced_diagnostic_opt_in(allow_unverified):
    """Require explicit diagnostic opt-in for direct optimized prefill calls."""
    if allow_unverified is not True:
        raise RuntimeError('CED numerical parity is not certified for all profiles; use ordinary full prefill. allow_unverified=True is for explicit diagnostics only; unrestricted Top-K selection and pointwise logits have known parity failures.')


def decoder_row_plan(positions, query_rows, windows):
    """Input-independent legal support: rows are physical indices, never RoPE IDs."""
    size = positions.source.shape[1]
    wanted = torch.as_tensor(query_rows, dtype=torch.long, device=positions.source.device)
    if wanted.ndim != 1 or not wanted.numel() or bool(((wanted < 0) | (wanted >= size)).any()):
        raise ValueError('Invalid requested Decoder rows')
    if not bool(positions.valid[:, wanted].all()):
        raise ValueError('Requested Decoder row is padding')
    if not windows or any((not isinstance(width, int) or width <= 0 for width in windows)):
        raise ValueError('Invalid Decoder windows')
    if bool((positions.source[:, 1:] <= positions.source[:, :-1]).any()):
        raise ValueError('Source ordinals must be strictly increasing')
    rows = wanted.unique(sorted=True)
    metadata_bank = Bank(
        torch.empty((*positions.source.shape, 0), device=positions.source.device),
        positions,
        positions.source,
        -1
    )
    plan = [None] * len(windows)
    plan[-1] = rows
    for stage in range(len(windows) - 1, 0, -1):
        slots = window_indices(positions.select(rows), metadata_bank, windows[stage])
        rows = slots[slots >= 0].unique(sorted=True)
        plan[stage - 1] = rows
    return (wanted, plan)


def _subset_shared(shared, local_rows, positions):
    selection = shared.get('selection')
    if selection is not None:
        shared['selection'] = Selection(
            selection.slots[:, local_rows],
            selection.request,
            selection.bank_owner,
            selection.index_owner,
            selection.generation,
            positions.source,
            positions.document,
            selection.bank_size
        )
    if shared.get('candidates') is not None:
        shared['candidates'] = shared['candidates'][:, local_rows]


def _mix(model, layer, hidden, kind):
    z = hidden.flatten(2).float()
    mix = F.linear(z, getattr(layer, 'hc_' + kind + '_fn'))
    mix = mix * torch.rsqrt(z.square().mean(-1, keepdim=True) + layer.norm_eps)
    function = model.split_hc
    return function(
        mix,
        getattr(layer, 'hc_' + kind + '_scale'),
        getattr(layer, 'hc_' + kind + '_base'),
        layer.hc_mult,
        layer.hc_sinkhorn_iters,
        layer.hc_eps
    )


def _save_window(state, window, width):
    if state is None:
        return
    tail = window.select(slice(-width, None))
    position = tail.positions
    state.windows[window.owner] = Bank(
        tail.value.clone(),
        Positions(*(v.clone() for v in (
            position.source,
            position.rope,
            position.document,
            position.valid
        ))),
        tail.complete.clone(),
        window.owner,
        window.generation
    )


def _prefill_state(keep_state, request_id, initial_state):
    if initial_state is None:
        return RequestState(request_id) if keep_state else None
    if not keep_state or not isinstance(initial_state, RequestState):
        raise ValueError('An initial request state requires keep_state=True')
    if initial_state.consumed or initial_state.windows or initial_state.banks or initial_state.pending or (initial_state.memory is not None):
        raise ValueError('CED prefill requires an empty request state')
    if not isinstance(initial_state.generation, int) or initial_state.generation < 0:
        raise ValueError('Invalid request generation')
    return initial_state.fork()


@torch.no_grad()
def baseline_ced_prefill(
    model,
    ids,
    positions=None,
    query_rows=None,
    inputs_embeds=None,
    keep_state=False,
    request_id='ced-prefill',
    trace=False,
    initial_state=None,
    *,
    allow_unverified=False
):
    """Run the full encoder and replay only required decoder dependencies.

    Keep the complete global bank and optionally return request-owned decode
    state. Direct calls require diagnostic opt-in because finite-precision
    row pruning has known parity failures outside the guarded profile.
    Cached execution requires one unpadded document and its final query.
    """
    require_ced_diagnostic_opt_in(allow_unverified)
    if model.training or isinstance(model, GSM):
        raise ValueError('Baseline CED requires an eval Text/Multimodal Baseline')
    schedules = {(40, (2, 8, 14, 20)): 20, (28, (2, 6, 10, 14)): 14}
    key = (len(model.weights.layers), tuple(model.config['kv_source_layers']))
    if key not in schedules:
        raise ValueError('CED has only been defined for the audited 40/20 and 28/14 owner schedules')
    encoder_layers = schedules[key]
    if any((layer.attn.compress_ratio != 1 for layer in model.weights.layers[encoder_layers:])):
        raise ValueError('CED requires the unchanged ratio1 Decoder global bank')
    device = model.weights.embed.weight.device
    if ids.device != device or any((
        p.device != device for n,
        p in model.named_parameters() if not n.startswith('weights.mtp.')
    )):
        raise ValueError('This CED implementation requires a single model device')
    pos = text_positions(ids) if positions is None else positions
    wanted, plan = decoder_row_plan(
        pos,
        [ids.shape[1] - 1] if query_rows is None else query_rows,
        [int(layer.attn.window_size) for layer in model.weights.layers[encoder_layers:]]
    )
    if keep_state and (ids.shape[0] != 1 or not bool(pos.valid.all()) or pos.document.unique().numel() != 1 or (not bool((wanted == ids.shape[1] - 1).any())) or (not torch.equal(
        pos.source,
        torch.arange(ids.shape[1], device=device)[None]
    ))):
        raise ValueError('Cached CED prefill requires one unpadded document including its final query')
    state = _prefill_state(keep_state, request_id, initial_state)
    embedded = F.embedding(
        ids,
        model.weights.embed.weight
    ) if inputs_embeds is None else inputs_embeds
    if embedded.shape != (*ids.shape, model.config['dim']):
        raise ValueError('Embedding shape mismatch')
    h = embedded.unsqueeze(2).repeat(1, 1, model.config['hc_mult'], 1)
    pre = h.new_zeros((*h.shape[:2], model.config['hc_mult']), dtype=torch.float32)
    pre[..., 0] = 1
    shared, counts = ({}, {})
    request = state.identity if state is not None else 'ced-stateless'
    for layer in model.weights.layers[:encoder_layers]:
        h, pre, _ = GSM.block(model, layer, h, pre, pos, shared, state, request, counts, None)
    current_rows = torch.arange(ids.shape[1], device=device)
    current_pos = pos
    records = []
    for layer, rows in zip(model.weights.layers[encoder_layers:], plan):
        a = layer.attn
        local = torch.searchsorted(current_rows, rows)
        if bool((local >= current_rows.numel()).any()) or not torch.equal(
            current_rows[local],
            rows
        ):
            raise RuntimeError('CED dependency plan lost required residual rows')
        query_pos = pos.select(rows)
        if a.layer_id > encoder_layers:
            _subset_shared(shared, local, query_pos)
        all_x = norm(model.pre_hc(h.float(), pre).to(h.dtype), layer.attn_norm)
        value = quantize_cache(
            rotate(norm(linear(all_x, a.wkv), a.kv_norm), model.frequencies(a, current_pos)),
            'swa',
            model.quantized_cache
        )
        window = Bank(
            value,
            current_pos,
            current_pos.source,
            a.layer_id,
            0 if state is None else state.generation
        )
        _save_window(state, window, int(a.window_size))
        if a.is_kv_source:
            if a.layer_id != encoder_layers or current_rows.numel() != ids.shape[1]:
                raise RuntimeError('The first Decoder global-bank owner must see every Encoder row')
            shared['bank'] = model.compress(a, all_x, current_pos, state)
        bank = shared['bank']
        x = all_x[:, local]
        qr, q = model.query(a, x, query_pos)
        if a.is_index_source:
            slots, _, shared['candidates'] = model.index(
                a,
                x,
                qr,
                query_pos,
                bank,
                shared.get('candidates')
            )
            shared['selection'] = Selection(
                slots,
                request,
                bank.owner,
                a.layer_id,
                bank.generation,
                query_pos.source,
                query_pos.document,
                bank.value.shape[1]
            )
        selection = shared['selection']
        selection.validate(bank, query_pos, request)
        slots = selection.slots
        wi = window_indices(query_pos, window, int(a.window_size))
        out = model.attend_banks(q, window, wi, a.attn_sink, bank, slots)
        out = model.output(a, out, query_pos)
        h = h[:, local]
        ap, post, comb = _mix(model, layer, h, 'attn')
        h = model.post_hc(out, h, post, comb).to(out.dtype)
        fp, post, comb = _mix(model, layer, h, 'ffn')
        ffn_x = norm(model.pre_hc(h.float(), ap).to(h.dtype), layer.ffn_norm)
        out = model.moe(layer.ffn, ffn_x, query_pos, counts)
        h = model.post_hc(out, h, post, comb).to(out.dtype)
        pre, current_rows, current_pos = (fp, rows, query_pos)
        if trace:
            records.append({
                'layer': a.layer_id,
                'rows': rows.clone(),
                'input_rows': window.positions.source.clone(),
                'slots': slots.clone(),
                'bank_size': bank.value.shape[1],
                'hidden': h.clone()
            })
    final_rows = torch.searchsorted(current_rows, wanted)
    hidden = norm(model.pre_hc(h.float(), pre).to(h.dtype), model.weights.norm)[:, final_rows]
    logits = model.project_logits(hidden)
    if state is not None:
        state.consumed = ids.shape[1]
    info = {
        'verification_status': 'UNVERIFIED_OPTIMIZATION',
        'decoder_rows': [len(rows) for rows in plan],
        'encoder_rows': ids.shape[1],
        'global_bank_rows': shared['bank'].value.shape[1],
        'layers': records
    }
    return Result(
        logits,
        hidden,
        h.new_zeros((), dtype=torch.float32),
        state,
        counts,
        {'ced': info}
    )


@torch.no_grad()
def gsm_ced_prefill(
    model,
    ids,
    positions=None,
    query_rows=None,
    inputs_embeds=None,
    keep_state=False,
    request_id='gsm-ced-prefill',
    trace=False,
    initial_state=None,
    *,
    allow_unverified=False
):
    """Run the full encoder and refine only shared-memory support rows.

    Each latent stage uses the encoder bank and row-aligned main selection.
    Refinement accumulates in FP32. Decoder blocks run only requested queries
    and read shared memory without creating per-layer local KV entries.
    """
    require_ced_diagnostic_opt_in(allow_unverified)
    if model.training or not isinstance(model, GSM):
        raise ValueError('GSM CED requires an eval GSM model')
    device = model.weights.embed.weight.device
    if ids.device != device or any((
        p.device != device for n,
        p in model.named_parameters() if not n.startswith('weights.mtp.')
    )):
        raise ValueError('This CED implementation requires a single model device')
    pos = text_positions(ids) if positions is None else positions
    wanted, plan = decoder_row_plan(
        pos,
        [ids.shape[1] - 1] if query_rows is None else query_rows,
        [1, int(model.memory_window)]
    )
    memory_rows = plan[0]
    memory_pos = pos.select(memory_rows)
    query_pos = pos.select(wanted)
    if keep_state and (ids.shape[0] != 1 or not bool(pos.valid.all()) or pos.document.unique().numel() != 1 or (not bool((wanted == ids.shape[1] - 1).any())) or (not torch.equal(
        pos.source,
        torch.arange(ids.shape[1], device=device)[None]
    ))):
        raise ValueError('Cached GSM CED requires one unpadded document including its final query')
    state = _prefill_state(keep_state, request_id, initial_state)
    request = state.identity if state is not None else 'gsm-ced-stateless'
    embedded = F.embedding(
        ids,
        model.weights.embed.weight
    ) if inputs_embeds is None else inputs_embeds
    if embedded.shape != (*ids.shape, model.config['dim']):
        raise ValueError('Embedding shape mismatch')
    from .gsm_model import lift
    h, pre = lift(embedded, model.config['hc_mult'])
    shared, counts = ({}, {})
    refinement = None
    latent_records = []
    encoder_layers = model.encoder_layers
    for layer in model.weights.layers[:encoder_layers]:
        h, pre, _ = model.block(layer, h, pre, pos, shared, state, request, counts, None)
        if layer.layer_id in model.latent_layers:
            stage = model.latent_layers.index(layer.layer_id)
            original_selection = shared['selection']
            selection = Selection(
                original_selection.slots[:, memory_rows],
                request,
                original_selection.bank_owner,
                original_selection.index_owner,
                original_selection.generation,
                memory_pos.source,
                memory_pos.document,
                original_selection.bank_size
            )
            hidden = model.pre_hc(h[:, memory_rows].float(), pre[:, memory_rows])
            record = {} if trace else None
            refinement = model.latent[stage](
                model,
                hidden,
                refinement,
                memory_pos,
                shared['window'],
                shared['bank'],
                selection,
                request,
                record
            )
            if trace:
                latent_records.append(record)
    memory_h = model.pre_hc(h[:, memory_rows].float(), pre[:, memory_rows]) + refinement
    x = norm(memory_h, model.memory.norm).to(model.memory.wkv.weight.dtype)
    value = norm(linear(x, model.memory.wkv), model.memory.kv_norm)
    value = quantize_cache(
        rotate(value, model.frequencies(model.weights.layers[encoder_layers].attn, memory_pos)),
        'main',
        model.quantized_cache
    )
    bank = Bank(
        value,
        memory_pos,
        memory_pos.source,
        encoder_layers,
        0 if state is None else state.generation
    )
    if state is not None:
        tail = bank.select(slice(-model.memory_window, None))
        p = tail.positions
        state.memory = Bank(
            tail.value.clone(),
            Positions(*(v.clone() for v in (p.source, p.rope, p.document, p.valid))),
            tail.complete.clone(),
            encoder_layers,
            tail.generation
        )
    selected_rows = torch.searchsorted(memory_rows, wanted)
    if not torch.equal(memory_rows[selected_rows], wanted):
        raise RuntimeError('GSM memory support lost requested initial residual')
    h, pre = lift(memory_h[:, selected_rows], model.config['hc_mult'])
    shared = {'memory': bank}
    for layer in model.weights.layers[encoder_layers:]:
        h, pre, _ = model.block(layer, h, pre, query_pos, shared, None, request, counts, None)
    hidden = norm(
        model.pre_hc(h.float(), pre).to(model.weights.head.weight.dtype),
        model.weights.norm
    )
    logits = model.project_logits(hidden)
    if state is not None:
        state.consumed = ids.shape[1]
    info = {
        'verification_status': 'UNVERIFIED_OPTIMIZATION',
        'encoder_rows': ids.shape[1],
        'latent_rows': [memory_rows.numel()] * 3,
        'memory_rows': memory_rows,
        'decoder_rows': [wanted.numel()] * (len(model.weights.layers) - encoder_layers),
        'latent': latent_records
    }
    return Result(
        logits,
        hidden,
        h.new_zeros((), dtype=torch.float32),
        state,
        counts,
        {'ced': info}
    )


def expert_dispatch(model, ffn, x, pos, counts):
    """Dispatch tokens by expert, preserving routing weights and shared experts."""
    if x.shape[0] * x.shape[1] == 1:
        return model._reference_moe(ffn, x, pos, counts)
    flat = x.flatten(0, 1)
    gate = ffn.gate
    score = F.softplus(F.linear(flat.float(), gate.weight.float()) / gate.gate_temp).sqrt()
    ids = (score + model.routing_bias(gate, pos)).topk(gate.topk, -1).indices
    weights = score.gather(-1, ids)
    if gate.norm_topk_prob and gate.topk > 1:
        weights = weights / (weights.sum(-1, keepdim=True) + 1e-20)
    weights = weights * gate.route_scale
    counts[ffn.layer_id] = model.route_counts(ids, pos, ffn.n_routed_experts)
    tags = ids.flatten()
    order = tags.argsort(stable=True)
    sizes = torch.bincount(tags, minlength=ffn.n_routed_experts).cpu().tolist()
    rows = torch.div(order, gate.topk, rounding_mode='floor')
    columns = order % gate.topk
    result = torch.zeros_like(flat, dtype=torch.float32)
    offset = 0

    def expert(module, inp, weight=None):
        up = linear(inp, module.w3).float()
        g = linear(inp, module.w1).float()
        if module.swiglu_limit > 0:
            up = up.clamp(-module.swiglu_limit, module.swiglu_limit)
            g = g.clamp_max(module.swiglu_limit)
        z = F.silu(g) * up
        if weight is not None:
            z = z * weight
        return linear(z.to(inp.dtype), module.w2)
    for index, size in enumerate(sizes):
        if size:
            row = rows[offset:offset + size]
            column = columns[offset:offset + size]
            result = result.index_add(
                0,
                row,
                expert(ffn.experts[index], flat[row], weights[row, column, None]).float()
            )
        offset += size
    return (result + expert(ffn.shared_experts, flat)).to(x.dtype).reshape_as(x)


def tiled_index(model, a, x, qr, pos, bank, candidates, tile):
    """Evaluate reference indexing in query tiles and concatenate row-aligned results."""
    slots = []
    scores = []
    choices = []
    for begin in range(0, x.shape[1], tile):
        sl = slice(begin, begin + tile)
        s, p, c = model._reference_index(
            a,
            x[:, sl],
            qr[:, sl],
            pos.select(sl),
            bank,
            None if candidates is None else candidates[:, sl]
        )
        slots.append(s)
        scores.append(p)
        if c is not None:
            choices.append(c)
    return (torch.cat(slots, 1), torch.cat(scores, 1), torch.cat(choices, 1) if choices else None)


def certified_text8b_profile(model):
    """Check the original supported text profile using the anonymized backend name."""
    c = model.config
    expected = {
        'n_layers': 28,
        'dim': 1920,
        'n_heads': 32,
        'head_dim': 256,
        'q_lora_rank': 480,
        'o_groups': 8,
        'o_lora_rank': 384,
        'index_n_heads': 16,
        'index_head_dim': 128,
        'window_size': 128,
        'index_topk': 128,
        'n_routed_experts': 52,
        'n_activated_experts': 6,
        'vocab_size': 49152,
        'moe_inter_dim': 864,
        'n_shared_experts': 1,
        'rope_head_dim': 64,
        'rope_factor': 1,
        'hc_mult': 4,
        'hc_sinkhorn_iters': 20,
        'candidate_source_layer': 14,
        'candidate_topk_blocks': 2048,
        'candidate_block_size': 8,
        'kv_source_layers': [2, 6, 10, 14],
        'index_source_layers': [2, 6, 10, 14, 18, 22, 26],
        'compress_ratios': [0, 0] + [2] * 12 + [1] * 14
    }
    return all((
        c.get(k) == v for k,
        v in expected.items()
    )) and (not model.quantized_cache) and model.native_attention and (model.execution_backend == 'native_fused') and (not getattr(
        model,
        'expert_parallel_enabled',
        False
    ))


@torch.no_grad()
@verified_official_scope()
def prefill(model, ids, positions=None, *, keep_state=True, query_rows=None, inputs_embeds=None):
    """Use dependency-based prefill for eligible text input; otherwise fall back.

    Packed, batched, embedded, and unsupported inputs retain the model's full
    forward path. State remains owned by the request.
    """
    if model.training:
        raise ValueError('Generation prefill requires eval mode')
    pos = text_positions(ids) if positions is None else positions
    eligible = certified_text8b_profile(model) and inputs_embeds is None and (ids.shape[0] == 1) and (ids.shape[1] > 0) and (pos.rope.ndim == 2) and torch.equal(pos.rope, pos.source) and bool(pos.valid.all()) and (pos.document.unique().numel() == 1) and torch.equal(
        pos.source,
        torch.arange(ids.shape[1], device=ids.device)[None]
    )
    if eligible:
        fn = gsm_ced_prefill if isinstance(model, GSM) else baseline_ced_prefill
        result = fn(
            model,
            ids,
            positions=pos,
            keep_state=keep_state,
            query_rows=query_rows,
            inputs_embeds=inputs_embeds,
            allow_unverified=True
        )
        result.trace['ced']['verification_status'] = 'VERIFIED_TEXT_PROFILE'
        return result
    rows = [ids.shape[1] - 1] if query_rows is None else query_rows
    return model(
        ids,
        positions=pos,
        keep_state=keep_state,
        query_rows=rows,
        inputs_embeds=inputs_embeds
    )
