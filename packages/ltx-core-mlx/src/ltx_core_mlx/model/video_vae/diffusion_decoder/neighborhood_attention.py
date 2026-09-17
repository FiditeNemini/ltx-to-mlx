"""Exact 3-D neighborhood attention (NATTEN ``na3d`` semantics) on MLX.

Per axis of length ``L`` and kernel ``k``: ``k_eff = min(k, L)``,
``start(i) = clamp(i - k_eff // 2, 0, L - k_eff)``, window ``[start, start + k_eff)``. The 3-D
window is the Cartesian product; always exactly ``kt*kh*kw`` keys, shifted inward at borders,
query included, non-causal, one softmax per query and head, scale 1.0 (``q`` is pre-scaled).

Gathering every window explicitly is impossible at stage-5 sizes (1.2 M queries x 1331 keys),
so :func:`na3d` processes query blocks: the union of a block's windows is a contiguous key slab
of fixed size ``b + k_eff - 1`` per axis (windows shift monotonically), so the slab is gathered
once per block and dense ``mx.fast.scaled_dot_product_attention`` runs with a boolean mask that
keeps exactly each query's window. Masked keys are excluded from the softmax, so the result
equals the per-window softmax up to accumulation order.
"""

from __future__ import annotations

import itertools

import mlx.core as mx

Kernel = tuple[int, int, int]


def window_start(length: int, kernel: int, index: int) -> int:
    """Start index of the (clamped) attention window for one query position on one axis."""
    k_eff = min(kernel, length)
    return min(max(index - k_eff // 2, 0), length - k_eff)


def window_starts(length: int, kernel: int) -> mx.array:
    """Vectorized :func:`window_start` over every index of the axis. Returns int32 ``(length,)``."""
    k_eff = min(kernel, length)
    idx = mx.arange(length)
    return mx.clip(idx - k_eff // 2, 0, length - k_eff).astype(mx.int32)


def na3d_reference(q: mx.array, k: mx.array, v: mx.array, kernel: Kernel) -> mx.array:
    """Brute-force oracle (Python loop over queries); small grids only.

    ``q, k, v`` have shape ``(B, T, H, W, heads, head_dim)``, ``q`` already scaled, scale 1.0.
    """
    b, t, h, w, heads, head_dim = q.shape
    lengths = (t, h, w)
    k_eff = [min(kk, ll) for kk, ll in zip(kernel, lengths, strict=True)]
    rows = []
    for ti, hi, wi in itertools.product(range(t), range(h), range(w)):
        st, sh, sw = (window_start(ll, kk, i) for ll, kk, i in zip(lengths, kernel, (ti, hi, wi), strict=True))
        keys = k[:, st : st + k_eff[0], sh : sh + k_eff[1], sw : sw + k_eff[2]].reshape(b, -1, heads, head_dim)
        vals = v[:, st : st + k_eff[0], sh : sh + k_eff[1], sw : sw + k_eff[2]].reshape(b, -1, heads, head_dim)
        scores = mx.einsum(
            "bnd,bknd->bnk",
            q[:, ti, hi, wi].astype(mx.float32),
            keys.astype(mx.float32),
        )
        probs = mx.softmax(scores, axis=-1)
        rows.append(mx.einsum("bnk,bknd->bnd", probs, vals.astype(mx.float32)))
    return mx.stack(rows, axis=1).reshape(q.shape).astype(q.dtype)


def _axis_plan(length: int, kernel: int, block: int) -> tuple[list[int], list[int], int, int]:
    """Per axis: query block origins, key slab origins, block size, slab size.

    The slab size ``block + k_eff - 1`` must fit within the axis, so the block size is
    capped accordingly; windows shift monotonically with the query index, so the union of
    windows for a contiguous run of ``block`` queries is itself a contiguous slab of that size.
    """
    k_eff = min(kernel, length)
    b = max(1, min(block, length - k_eff + 1))
    slab = b + k_eff - 1
    q0s = list(range(0, length, b))
    s0s = [min(window_start(length, kernel, q0), length - slab) for q0 in q0s]
    return q0s, s0s, b, slab


def _gather_grid(x: mx.array, ti: mx.array, hi: mx.array, wi: mx.array) -> mx.array:
    """Gather a ``(len(ti), len(hi), len(wi), heads, head_dim)`` block from ``x[0]`` via
    broadcast advanced indexing, then flatten the spatial axes."""
    heads, head_dim = x.shape[-2], x.shape[-1]
    gathered = x[0, ti[:, None, None], hi[None, :, None], wi[None, None, :]]
    return gathered.reshape(-1, heads, head_dim)


def na3d(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    kernel: Kernel,
    block: Kernel = (4, 8, 8),
    max_blocks: int = 64,
) -> mx.array:
    """Exact neighborhood attention over ``(B, T, H, W, heads, head_dim)`` tensors (see module doc).

    Processes query blocks: gathers a fixed-size key/value slab per block, runs dense
    ``mx.fast.scaled_dot_product_attention`` with a boolean mask that keeps exactly each
    query's window, then scatters the block's outputs back. Blocks with identical slab
    shapes are batched together (up to ``max_blocks``) along the SDPA batch axis.
    """
    bsz, t, h, w, heads, head_dim = q.shape
    if bsz != 1:
        raise ValueError("na3d supports batch size 1 (upstream's chunked path has the same restriction)")

    lengths = (t, h, w)
    plans = [_axis_plan(ll, kk, bb) for ll, kk, bb in zip(lengths, kernel, block, strict=True)]
    starts = [window_starts(ll, kk) for ll, kk in zip(lengths, kernel, strict=True)]
    k_eff = [min(kk, ll) for kk, ll in zip(kernel, lengths, strict=True)]

    out = mx.zeros_like(q)

    # Per-axis query indices, slab indices and window masks depend only on the axis and the
    # block index along that axis, so build them once per axis instead of once per 3-D block.
    axis_qi: list[list[mx.array]] = []
    axis_si: list[list[mx.array]] = []
    axis_mask: list[list[mx.array]] = []
    for axis in range(3):
        q0s, s0s, bsize, slab = plans[axis]
        ll = lengths[axis]
        qis, sis, masks_axis = [], [], []
        for q0, s0 in zip(q0s, s0s, strict=True):
            qi = mx.arange(bsize) + q0
            qi = mx.minimum(qi, ll - 1)  # last block on a non-divisible axis clamps in-bounds
            si = mx.arange(slab) + s0
            ws = starts[axis][qi]  # window start per query on this axis, shape (bsize,)
            qis.append(qi)
            sis.append(si)
            masks_axis.append((si[None, :] >= ws[:, None]) & (si[None, :] < (ws + k_eff[axis])[:, None]))
        axis_qi.append(qis)
        axis_si.append(sis)
        axis_mask.append(masks_axis)

    block_ids = list(itertools.product(*[range(len(p[0])) for p in plans]))
    for group_start in range(0, len(block_ids), max_blocks):
        group = block_ids[group_start : group_start + max_blocks]

        q_indices: list[tuple[mx.array, mx.array, mx.array]] = []
        s_indices: list[tuple[mx.array, mx.array, mx.array]] = []
        masks = []
        for bi in group:
            q_indices.append(tuple(axis_qi[axis][bi[axis]] for axis in range(3)))
            s_indices.append(tuple(axis_si[axis][bi[axis]] for axis in range(3)))
            mt, mh, mw = (axis_mask[axis][bi[axis]] for axis in range(3))
            masks.append(
                mt[:, None, None, :, None, None] & mh[None, :, None, None, :, None] & mw[None, None, :, None, None, :]
            )

        qg = mx.stack([_gather_grid(q, *qi) for qi in q_indices])  # (G, Lq, heads, head_dim)
        kg = mx.stack([_gather_grid(k, *si) for si in s_indices])  # (G, Lk, heads, head_dim)
        vg = mx.stack([_gather_grid(v, *si) for si in s_indices])  # (G, Lk, heads, head_dim)
        mask = mx.stack([m.reshape(m.shape[0] * m.shape[1] * m.shape[2], -1) for m in masks])[
            :, None
        ]  # (G, 1, Lq, Lk), broadcasts over heads

        og = mx.fast.scaled_dot_product_attention(
            qg.transpose(0, 2, 1, 3),
            kg.transpose(0, 2, 1, 3),
            vg.transpose(0, 2, 1, 3),
            scale=1.0,
            mask=mask,
        ).transpose(0, 2, 1, 3)  # (G, Lq, heads, head_dim)

        for gi, qi in enumerate(q_indices):
            ti, hi, wi = qi
            bt, bh, bw = len(ti), len(hi), len(wi)
            out[0, ti[:, None, None], hi[None, :, None], wi[None, None, :]] = og[gi].reshape(
                bt, bh, bw, heads, head_dim
            )
        # Materialize the accumulator so live Metal buffers stay bounded by one block group;
        # without this the buffer count grows with the total query-block count and trips the
        # driver's resource limit at stage-5 sizes. Numerically inert.
        mx.eval(out)

    return out
