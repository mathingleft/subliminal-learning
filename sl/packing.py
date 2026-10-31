"""Padding-free micro-batches for Qwen3.5 (hybrid Gated DeltaNet + full attention), with a shared prompt prefix.

The rows of a micro-batch are laid end to end in one sequence and every token mixer is told where each row starts,
so no row sees another: full attention attends within a row, the linear-attention conv and recurrence restart at
each row, and positions restart at 0. The leading tokens that every row has in common (here the system prompt) are
laid down once: each row continues from that one copy, attending to its keys/values and starting its conv and
recurrence from the prefix's state, which is exactly what it would compute from its own copy. Each row's math
is therefore that of the row alone; the prefix's gradient is the sum of the rows' contributions.

With flash-linear-attention installed its varlen kernels do the work; without it (CPU tests) each row is run
separately through the reference code.
"""
import itertools

import torch
import torch.nn.functional as F

ATTN = "sl_varlen"


def pack(examples, device, share_prefix=True):
    """examples: [(ids, labels)] -> (ids [1, T], labels [1, T], model kwargs: position ids and row layout)."""
    lp = 0
    if share_prefix and len(examples) > 1:
        first = examples[0][0]
        # Only prompt positions may be shared, and the token after the prefix must be a prompt token too (it is
        # predicted from the prefix's last position, which every row shares).
        lp = min(next(i for i, y in enumerate(labels) if y != -100) for _, labels in examples) - 1
        for x, _ in examples:
            lp = next((i for i in range(lp) if x[i] != first[i]), lp)
    segs = ([first[:lp]] if lp else []) + [x[lp:] for x, _ in examples]
    seg_labels = ([[-100] * lp] if lp else []) + [y[lp:] for _, y in examples]
    seg_pos = ([range(lp)] if lp else []) + [range(lp, len(x)) for x, _ in examples]
    lens = [len(s) for s in segs]
    cu = torch.tensor([0, *itertools.accumulate(lens)], device=device)
    t = lambda parts: torch.tensor([v for p in parts for v in p], device=device)[None]
    kwargs = {"position_ids": t(seg_pos), "cu_seq_lens_q": cu, "cu_seq_lens_k": cu, "max_length_q": max(lens),
              "max_length_k": max(lens)}
    if lp:
        b = cu.tolist()
        rows = list(zip(b[1:-1], b[2:]))
        kv = [range(lp)] + [itertools.chain(range(lp), range(s, e)) for s, e in rows]
        kwargs["sl_prefix"] = {
            "len": lp, "rows": rows, "cu_rows": cu[1:] - lp, "kv_index": t(kv)[0], "max_k": lp + max(lens[1:]),
            "cu_k": torch.tensor([0, *itertools.accumulate([lp] + [lp + e - s for s, e in rows])], device=device)}
    return t(segs), t(seg_labels), kwargs


def _segments(cu):
    b = cu.tolist()
    return list(zip(b[:-1], b[1:]))


def _attention(module, query, key, value, attention_mask, scaling=None, dropout=0.0, cu_seq_lens_q=None,
               max_length_q=None, sl_prefix=None, **kwargs):
    from transformers.integrations.sdpa_attention import sdpa_attention_forward

    if cu_seq_lens_q is None:
        return sdpa_attention_forward(module, query, key, value, attention_mask, scaling=scaling, dropout=dropout,
                                      **kwargs)
    q, k, v = (x[0].transpose(0, 1) for x in (query, key, value))  # [T, heads, head_dim]
    cu_k, max_k = cu_seq_lens_q, max_length_q
    if sl_prefix is not None:  # row i's keys/values: the prefix's, then its own
        k, v = k.index_select(0, sl_prefix["kv_index"]), v.index_select(0, sl_prefix["kv_index"])
        cu_k, max_k = sl_prefix["cu_k"], sl_prefix["max_k"]
    if q.is_cuda:
        from torch.nn.attention.varlen import varlen_attn

        # window (-1, 0): causal, aligned at the bottom right, so a row's queries see the prefix and their past
        out = varlen_attn(q, k, v, cu_seq_lens_q.int(), cu_k.int(), max_length_q, max_k, scale=scaling,
                          window_size=(-1, 0), enable_gqa=True)
    else:
        outs = []
        for (s, e), (sk, ek) in zip(_segments(cu_seq_lens_q), _segments(cu_k)):
            n_q, n_k = e - s, ek - sk
            mask = torch.arange(n_k)[None, :] <= torch.arange(n_k - n_q, n_k)[:, None]
            outs.append(F.scaled_dot_product_attention(q[s:e].transpose(0, 1), k[sk:ek].transpose(0, 1),
                                                       v[sk:ek].transpose(0, 1), attn_mask=mask, scale=scaling,
                                                       enable_gqa=True).transpose(0, 1))
        out = torch.cat(outs)
    return out[None], None


def has_fla():
    """Whether flash-linear-attention is installed (without it, packed rows run one by one: CPU tests only)."""
    import importlib.util

    return importlib.util.find_spec("fla") is not None


def install():
    """Register the varlen attention and make the linear-attention layers honor the packed layout."""
    from transformers import AttentionInterface
    from transformers.masking_utils import AttentionMaskInterface, sdpa_mask
    from transformers.models.qwen3_5 import modeling_qwen3_5 as m

    AttentionInterface.register(ATTN, _attention)
    AttentionMaskInterface.register(ATTN, sdpa_mask)
    if getattr(m, "_sl_packing", False):
        return
    conv, gdn = m.causal_conv1d_fn, m.torch_chunk_gated_delta_rule
    try:
        from fla.modules.convolution import causal_conv1d as fla_conv
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule as fla_gdn  # noqa: F401  (HF dispatches to it)
        fla = True
    except ImportError:
        fla = False

    def varlen_conv(x, weight, bias, activation, cu):  # x [1, D, T]
        if fla and x.is_cuda:
            out, _ = fla_conv(x.transpose(1, 2), weight, bias, activation=activation, cu_seqlens=cu)
            return out.transpose(1, 2)
        return torch.cat([conv(x[:, :, s:e], weight, bias, activation=activation) for s, e in _segments(cu)], -1)

    def causal_conv1d_fn(hidden_states, weight, bias=None, activation=None, cu_seq_lens_q=None, sl_prefix=None,
                         **kwargs):
        if cu_seq_lens_q is None:
            return conv(hidden_states, weight, bias, activation=activation)
        if sl_prefix is None:
            return varlen_conv(hidden_states, weight, bias, activation, cu_seq_lens_q)
        # Each row's conv also sees the last (kernel - 1) prefix inputs; their outputs are dropped again.
        lp, tail = sl_prefix["len"], min(weight.shape[-1] - 1, sl_prefix["len"])
        key = ("conv", tail)
        if key not in sl_prefix:
            dev = hidden_states.device
            idx = [range(lp)] + [itertools.chain(range(lp - tail, lp), range(s, e)) for s, e in sl_prefix["rows"]]
            lens = [lp] + [tail + e - s for s, e in sl_prefix["rows"]]
            keep = [range(lp)] + [range(c + tail, c + n) for c, n in zip(itertools.accumulate(lens), lens[1:])]
            sl_prefix[key] = tuple(torch.tensor([v for p in parts for v in p], device=dev)
                                   for parts in (idx, [[0], itertools.accumulate(lens)], keep))
        idx, cu, keep = sl_prefix[key]
        out = varlen_conv(hidden_states.index_select(-1, idx), weight, bias, activation, cu)
        return out.index_select(-1, keep)

    def torch_chunk_gated_delta_rule(query, key, value, g, beta, cu_seqlens=None, sl_prefix=None, **kwargs):
        if cu_seqlens is None or (sl_prefix is None and fla and query.is_cuda):
            return gdn(query, key, value, g=g, beta=beta, cu_seqlens=cu_seqlens, **kwargs)
        assert not kwargs.pop("output_final_state", False) and kwargs.pop("initial_state", None) is None
        if sl_prefix is None:
            segs, h0, lp = _segments(cu_seqlens), None, 0
        else:  # the prefix once, then every row starting from its final state
            lp = sl_prefix["len"]
            out_p, h = gdn(query[:, :lp], key[:, :lp], value[:, :lp], g=g[:, :lp], beta=beta[:, :lp],
                           output_final_state=True, **kwargs)
            segs, h0 = sl_prefix["rows"], h
        if fla and query.is_cuda:
            n = len(segs)
            out = gdn(query[:, lp:], key[:, lp:], value[:, lp:], g=g[:, lp:], beta=beta[:, lp:],
                      initial_state=h0.expand(n, *h0.shape[1:]).contiguous(), cu_seqlens=sl_prefix["cu_rows"],
                      **kwargs)[0]
        else:
            out = torch.cat([gdn(query[:, s:e], key[:, s:e], value[:, s:e], g=g[:, s:e], beta=beta[:, s:e],
                                 initial_state=h0, **kwargs)[0] for s, e in segs], 1)
        return (out if sl_prefix is None else torch.cat([out_p, out], 1)), None

    m.causal_conv1d_fn, m.torch_chunk_gated_delta_rule = causal_conv1d_fn, torch_chunk_gated_delta_rule
    m._sl_packing = True
