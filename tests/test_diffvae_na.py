"""Exact 3-D neighborhood attention: window bounds, brute-force oracle, blocked implementation."""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from ltx_core_mlx.model.video_vae.diffusion_decoder.neighborhood_attention import (
    na3d,
    na3d_reference,
    window_start,
    window_starts,
)


@pytest.mark.parametrize(
    ("length", "kernel", "expected"),
    [
        (10, 5, [0, 0, 0, 1, 2, 3, 4, 5, 5, 5]),  # centred, shifted inward at both borders
        (3, 5, [0, 0, 0]),  # kernel larger than the axis: k_eff = 3, window is the whole axis
        (11, 11, [0] * 11),
        (4, 3, [0, 0, 1, 1]),
    ],
)
def test_window_start_semantics(length, kernel, expected):
    assert [window_start(length, kernel, i) for i in range(length)] == expected
    assert np.array(window_starts(length, kernel)).tolist() == expected


def _numpy_na3d(q, k, v, kernel):
    # independent brute force in numpy (fp64): softmax over the Cartesian window
    q, k, v = (np.array(a, dtype=np.float64) for a in (q, k, v))
    B, T, H, W, NH, D = q.shape
    out = np.zeros_like(q)
    Ls, ks = (T, H, W), kernel
    starts = [[window_start(L, kk, i) for i in range(L)] for L, kk in zip(Ls, ks)]
    keff = [min(kk, L) for L, kk in zip(Ls, ks)]
    for b in range(B):
        for t in range(T):
            for h in range(H):
                for w in range(W):
                    st, sh, sw = starts[0][t], starts[1][h], starts[2][w]
                    kk = k[b, st : st + keff[0], sh : sh + keff[1], sw : sw + keff[2]].reshape(-1, NH, D)
                    vv = v[b, st : st + keff[0], sh : sh + keff[1], sw : sw + keff[2]].reshape(-1, NH, D)
                    s = np.einsum("hd,nhd->nh", q[b, t, h, w], kk)
                    s = np.exp(s - s.max(0, keepdims=True))
                    s /= s.sum(0, keepdims=True)
                    out[b, t, h, w] = np.einsum("nh,nhd->hd", s, vv)
    return out


@pytest.mark.parametrize("kernel", [(3, 5, 5), (3, 7, 7), (5, 5, 5), (3, 3, 3)])
def test_reference_matches_independent_numpy(kernel):
    mx.random.seed(0)
    q, k, v = (mx.random.normal((1, 5, 9, 9, 2, 4)) for _ in range(3))
    got = np.array(na3d_reference(q, k, v, kernel))
    assert np.allclose(got, _numpy_na3d(q, k, v, kernel), atol=1e-5)


@pytest.mark.parametrize("kernel", [(3, 5, 5), (3, 7, 7), (5, 5, 5)])
@pytest.mark.parametrize("block", [(4, 8, 8), (2, 3, 5), (1, 1, 1), (16, 16, 16)])
def test_blocked_matches_reference_including_borders(kernel, block):
    mx.random.seed(1)
    q, k, v = (mx.random.normal((1, 5, 9, 11, 2, 4)) for _ in range(3))
    got = na3d(q, k, v, kernel, block=block, max_blocks=3)
    assert mx.allclose(got, na3d_reference(q, k, v, kernel), atol=1e-5, rtol=1e-5).item()


def test_blocked_handles_axis_shorter_than_kernel():
    q, k, v = (mx.random.normal((1, 3, 4, 4, 1, 4)) for _ in range(3))
    got = na3d(q, k, v, (5, 5, 5))
    assert mx.allclose(got, na3d_reference(q, k, v, (5, 5, 5)), atol=1e-5).item()


def test_blocked_bf16_runs_and_is_close():
    q, k, v = (mx.random.normal((1, 5, 9, 9, 2, 4)).astype(mx.bfloat16) for _ in range(3))
    got = na3d(q, k, v, (3, 5, 5))
    assert got.dtype == mx.bfloat16
    ref = na3d_reference(q.astype(mx.float32), k.astype(mx.float32), v.astype(mx.float32), (3, 5, 5))
    assert mx.allclose(got.astype(mx.float32), ref, atol=5e-2).item()
