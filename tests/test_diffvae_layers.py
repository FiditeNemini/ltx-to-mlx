"""Numerics of the diffusion decoder's basic layers against numpy references."""

from __future__ import annotations

import math

import mlx.core as mx
import mlx.utils
import numpy as np

from ltx_core_mlx.model.video_vae.diffusion_decoder.layers import (
    LinearPixelShuffleUpsample,
    RMSNorm,
    SharedAdaLN,
    SwiGLU,
    TimestepEmbedder,
    sinusoidal_timestep_embedding,
)


def test_rmsnorm_matches_numpy_fp32_formula_and_keeps_input_dtype():
    n = RMSNorm(8)
    n.weight = mx.random.normal((8,))
    x = mx.random.normal((2, 3, 8)).astype(mx.bfloat16)
    y = n(x)
    assert y.dtype == mx.bfloat16
    xn = np.array(x.astype(mx.float32))
    w = np.array(n.weight)
    ref = xn / np.sqrt((xn * xn).mean(-1, keepdims=True) + 1e-6) * w
    assert np.allclose(np.array(y.astype(mx.float32)), ref, atol=2e-2)  # bf16 output rounding
    assert np.allclose(np.array(n(x.astype(mx.float32))), ref, atol=1e-6)


def test_swiglu_formula():
    m = SwiGLU(4, 6)
    x = mx.random.normal((5, 4))
    y = m(x)
    xn = np.array(x)
    g = xn @ np.array(m.w_gate.weight).T
    u = xn @ np.array(m.w_up.weight).T
    ref = (g / (1 + np.exp(-g)) * u) @ np.array(m.w_down.weight).T
    assert np.allclose(np.array(y), ref, atol=1e-5)
    assert not hasattr(m.w_gate, "bias")


def test_sinusoidal_embedding_is_cos_then_sin():
    t = mx.array([1000.0, 0.0])
    e = sinusoidal_timestep_embedding(t, 256)
    assert e.shape == (2, 256)
    freqs = np.exp(-math.log(10000.0) * np.arange(128, dtype=np.float32) / 128)
    ref = np.concatenate([np.cos(1000.0 * freqs), np.sin(1000.0 * freqs)])
    assert np.allclose(np.array(e[0]), ref, atol=1e-5)
    assert np.allclose(np.array(e[1]), np.concatenate([np.ones(128), np.zeros(128)]))


def test_timestep_embedder_keys_and_shape():
    m = TimestepEmbedder(256, 8)
    keys = dict(mlx.utils.tree_flatten(m.parameters())).keys()
    assert set(keys) == {"mlp.0.weight", "mlp.0.bias", "mlp.2.weight", "mlp.2.bias"}
    assert m(mx.array([1000.0])).shape == (1, 8)


def test_shared_adaln_chunks_in_order():
    m = SharedAdaLN(8, 4)
    m.proj.weight = mx.zeros((28, 8))
    m.proj.bias = mx.arange(28).astype(mx.float32)
    chunks = m(mx.zeros((1, 8)))
    assert len(chunks) == 7 and chunks[0].shape == (1, 1, 1, 1, 4)
    assert mx.array_equal(chunks[3].reshape(-1), mx.array([12.0, 13.0, 14.0, 15.0]))  # scale_mlp is chunk 3


def test_upsample_projects_shuffles_and_drops_leading_frame():
    up = LinearPixelShuffleUpsample(8, (2, 2, 2), 2)
    assert up.proj.weight.shape == (32, 8)  # prod(stride)*8/2
    x = mx.random.normal((1, 3, 2, 2, 8))
    y = up(x, drop_leading_frame=True)
    assert y.shape == (1, 5, 4, 4, 4)
    y2 = up(x, drop_leading_frame=False)
    assert y2.shape == (1, 6, 4, 4, 4) and mx.array_equal(y2[:, 1:], y)
