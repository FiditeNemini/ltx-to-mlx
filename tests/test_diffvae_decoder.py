"""NADiffusionDecoder assembly: geometry, parameter contract, end-to-end shapes on the tiny config, pack load contract."""

from __future__ import annotations

import json
import struct

import mlx.core as mx
import mlx.utils
import pytest

from ltx_core_mlx.model.video_vae.diffusion_decoder import NADiffusionDecoder, load_diffusion_decoder
from ltx_core_mlx.model.video_vae.diffusion_decoder.config import LTX_2_5_DIFFUSION_DECODER
from tests.conftest import LTX25_Q8_DIR
from tests.diffvae_tiny import TINY


def test_pad_to_floor_replicates_last_frame_and_symmetric_edges():
    dec = NADiffusionDecoder(TINY)
    z = mx.random.normal((1, 8, 2, 2, 5))  # floor is (3, 3, 3)
    zp, pads = dec.pad_to_floor(z)
    assert zp.shape == (1, 8, 3, 3, 5) and pads == (1, 0, 1, 0, 0)
    assert mx.array_equal(zp[:, :, 2], zp[:, :, 1])  # T: repeat last frame
    assert mx.array_equal(zp[:, :, :, 2], zp[:, :, :, 1])  # H after-pad: edge replicate
    z2 = mx.random.normal((1, 8, 3, 1, 3))
    _, pads2 = dec.pad_to_floor(z2)
    assert pads2 == (0, 1, 1, 0, 0)  # need 2 -> before 1, after 1


def test_stage_shapes_on_tiny_config():
    dec = NADiffusionDecoder(TINY)
    z = mx.random.normal((1, 8, 3, 3, 3))
    feat = dec.forward_stages_1_to_4(z)  # ghost pad 2 -> T=5 latent frames
    # stage chain on T: 5 -> up0 (1,2,2): 5 -> up1 (2,1,1) drop: 9 -> up2 (2,2,2) drop: 17 = T4 ; ghost crop keep = min(17, max(17-8, 2)) = 9
    assert feat.shape == (1, 9, 12, 12, 8)
    assert dec.stage5_canvas(9, 12, 12) == (17, 96, 96)
    x_t = mx.random.normal((1, 3, 17, 96, 96))
    px = dec.forward_stage_5(x_t, feat, mx.array([1.0]))
    assert px.shape == (1, 3, 17, 96, 96)


def test_decode_crops_to_content_and_is_deterministic_for_a_seed():
    dec = NADiffusionDecoder(TINY)
    z = mx.random.normal((1, 8, 2, 2, 3))  # content 9 frames, 64 x 96 px
    a = dec.decode(z, seed=3)
    b = dec.decode(z, seed=3)
    assert a.shape == (1, 3, 9, 64, 96) and mx.array_equal(a, b)
    assert not mx.array_equal(a, dec.decode(z, seed=4))
    chunks = list(dec.tiled_decode(z))
    assert len(chunks) == 1 and chunks[0].shape == a.shape


def test_injected_noise_bypasses_the_rng():
    dec = NADiffusionDecoder(TINY)
    z = mx.random.normal((1, 8, 3, 3, 3))
    noise = mx.random.normal((1, 3, 17, 96, 96))
    assert mx.array_equal(dec.decode(z, noise=noise), dec.decode(z, noise=noise))
    # Explicit noise makes `seed` irrelevant.
    assert mx.array_equal(dec.decode(z, noise=noise, seed=1), dec.decode(z, noise=noise, seed=2))


def test_decode_rejects_batch_and_bad_noise_shape():
    dec = NADiffusionDecoder(TINY)
    z = mx.random.normal((2, 8, 3, 3, 3))
    with pytest.raises(ValueError):
        dec.decode(z)
    z1 = mx.random.normal((1, 8, 3, 3, 3))
    bad_noise = mx.random.normal((1, 3, 16, 96, 96))  # wrong T
    with pytest.raises(ValueError):
        dec.decode(z1, noise=bad_noise)


def test_pixel_scales_derived_from_config():
    dec = NADiffusionDecoder(LTX_2_5_DIFFUSION_DECODER)
    assert dec.temporal_scale == 8
    assert dec.spatial_scale == (32, 32)


def test_parameter_names_match_the_pack_schema():
    dec = NADiffusionDecoder(TINY)
    keys = set(dict(mlx.utils.tree_flatten(dec.parameters())))
    for k in [
        "conv_in.weight",
        "conv_in_x_t.bias",
        "conv_out.weight",
        "type_emb",
        "det_stages.0.0.attn.qkv.weight",
        "det_stages.3.0.mlp.w_down.weight",
        "upsamples.3.proj.bias",
        "t_embedder.mlp.0.weight",
        "t_embedder.mlp.2.bias",
        "shared_adaln.proj.weight",
        "diff_blocks.1.scale_shift_table",
        "diff_blocks.0.context_proj.weight",
        "norm_out.weight",
        "per_channel_statistics.mean",
        "per_channel_statistics.std",
    ]:
        assert k in keys, k
    assert dec.diff_blocks[0].scale_shift_table.shape == (7, 4)
    assert dec.shared_adaln.proj.weight.shape == (28, 8)


@pytest.mark.slow
@pytest.mark.skipif(LTX25_Q8_DIR is None, reason="local ltx-2.5-mlx-q8 pack not found")
def test_pack_load_contract_bidirectional():
    path = LTX25_Q8_DIR / "vae_decoder_av.safetensors"
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
    pack_keys = {k.removeprefix("vae_decoder_av.") for k in header if k != "__metadata__"}
    dec = load_diffusion_decoder(path)  # strict: raises if a param is unfed
    model_keys = set(dict(mlx.utils.tree_flatten(dec.parameters())))
    assert pack_keys == model_keys, pack_keys ^ model_keys
    assert dec.per_channel_statistics.mean.shape == (128,)
