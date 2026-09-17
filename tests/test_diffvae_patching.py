"""Channel orders of patchify/unpatchify (``(c r q)``) and pixel shuffle (``(c p1 p2 p3)``), checked against numpy."""

from __future__ import annotations

import mlx.core as mx
import numpy as np

from ltx_core_mlx.model.video_vae.diffusion_decoder.patching import patchify_pixels, pixel_shuffle_3d, unpatchify_pixels


def test_patchify_channel_order_is_c_r_q():
    B, C, F, H, W, p = 1, 2, 1, 8, 4, 4
    x = mx.arange(B * C * F * H * W).reshape(B, C, F, H, W).astype(mx.float32)
    tok = patchify_pixels(x, p)
    assert tok.shape == (B, F, H // p, W // p, C * p * p)
    xn = np.array(x)
    for c in range(C):
        for q in range(p):  # h sub-index
            for r in range(p):  # w sub-index
                ch = c * p * p + r * p + q
                expected = xn[0, c, 0, q::p, r::p]  # (H/p, W/p)
                assert np.array_equal(np.array(tok)[0, 0, :, :, ch], expected)


def test_unpatchify_is_the_exact_inverse():
    x = mx.random.normal((2, 3, 3, 16, 12))
    assert mx.array_equal(unpatchify_pixels(patchify_pixels(x, 4), 4, 3), x)


def test_pixel_shuffle_channel_order():
    B, T, H, W, C = 1, 1, 2, 2, 3
    p1, p2, p3 = 2, 2, 2
    x = mx.arange(B * T * H * W * C * 8).reshape(B, T, H, W, C * 8).astype(mx.float32)
    y = pixel_shuffle_3d(x, (p1, p2, p3))
    assert y.shape == (B, T * p1, H * p2, W * p3, C)
    xn, yn = np.array(x), np.array(y)
    for t in range(T * p1):
        for h in range(H * p2):
            for w in range(W * p3):
                for c in range(C):
                    ch = c * 8 + (t % p1) * 4 + (h % p2) * 2 + (w % p3)
                    assert yn[0, t, h, w, c] == xn[0, t // p1, h // p2, w // p3, ch]


def test_pixel_shuffle_stride_one_axes():
    x = mx.random.normal((1, 3, 4, 5, 8))
    y = pixel_shuffle_3d(x, (1, 2, 2))
    assert y.shape == (1, 3, 8, 10, 2)
