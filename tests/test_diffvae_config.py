"""DiffusionDecoderConfig: 2.5 defaults, metadata parsing, derived geometry."""

from __future__ import annotations

import json

import mlx.core as mx
import pytest

from ltx_core_mlx.model.video_vae.diffusion_decoder.config import (
    LTX_2_5_DIFFUSION_DECODER,
    DiffusionDecoderConfig,
)
from tests.conftest import LTX25_Q8_DIR
from tests.diffvae_tiny import TINY


def test_ltx25_defaults_match_the_pack_metadata_values():
    c = LTX_2_5_DIFFUSION_DECODER
    assert c.stage_channels == (2048, 1024, 512, 512, 256)
    assert c.stage_depths == (4, 6, 4, 2, 8)
    assert c.stage_kernels == ((3, 7, 7), (3, 7, 7), (3, 5, 5), (3, 5, 5))
    assert c.upsamples == (((1, 2, 2), 2), ((2, 1, 1), 2), ((2, 2, 2), 1), ((2, 2, 2), 2))
    assert c.stage5_kernel == (11, 11, 11)
    assert c.head_dim == 64 and c.patch_size == 4 and c.in_channels == 128 and c.out_channels == 3
    assert c.t_freq_dim == 256 and c.t_embed_hidden == 384 and c.timestep_scale_multiplier == 1000.0


def test_derived_geometry_for_ltx25():
    c = LTX_2_5_DIFFUSION_DECODER
    assert c.diff_channels == 256 and c.diff_depth == 8 and c.adaln_dim == 1792
    assert [c.heads(ch) for ch in c.stage_channels] == [32, 16, 8, 8, 4]
    assert c.cumulative_strides() == [(1, 1, 1), (1, 2, 2), (2, 2, 2), (4, 4, 4), (8, 8, 8)]
    assert c.min_latent_shape() == (3, 7, 7)
    assert c.ghost_pad_frames() == 2


def test_derived_geometry_for_tiny():
    assert TINY.cumulative_strides() == [(1, 1, 1), (1, 2, 2), (2, 2, 2), (4, 4, 4), (8, 8, 8)]
    assert TINY.min_latent_shape() == (3, 3, 3)
    assert TINY.ghost_pad_frames() == 2
    assert TINY.adaln_dim == 28 and [TINY.heads(ch) for ch in TINY.stage_channels] == [4, 2, 2, 2, 1]


def test_channel_chain_is_validated():
    with pytest.raises(ValueError, match="stage_channels"):
        DiffusionDecoderConfig(**{**TINY.__dict__, "stage_channels": (16, 8, 8, 8, 8)})  # up3 reduction 2 -> 4, not 8


def test_from_safetensors_metadata_parses_the_vae_config(tmp_path):
    meta = {
        "config": json.dumps(
            {
                "vae": {
                    "decoder": {
                        "_class_name": "NADiffusionDecoder",
                        "in_channels": 128,
                        "out_channels": 3,
                        "patch_size": 4,
                        "head_dim": 64,
                        "stage_channels": [2048, 1024, 512, 512, 256],
                        "stage_depths": [4, 6, 4, 2, 8],
                        "stage_kernels": [[3, 7, 7], [3, 7, 7], [3, 5, 5], [3, 5, 5], [11, 11, 11]],
                        "upsamples": [[[1, 2, 2], 2], [[2, 1, 1], 2], [[2, 2, 2], 1], [[2, 2, 2], 2]],
                        "stage5_kernel": [11, 11, 11],
                        "timestep_scale_multiplier": 1000.0,
                        "default_num_inference_steps": 1,
                    }
                }
            }
        )
    }
    mx.save_safetensors(str(tmp_path / "d.safetensors"), {"x": mx.zeros((1,))}, metadata=meta)
    assert DiffusionDecoderConfig.from_safetensors_metadata(tmp_path / "d.safetensors") == LTX_2_5_DIFFUSION_DECODER


def test_from_safetensors_metadata_rejects_non_diffusion_decoders(tmp_path):
    meta = {"config": json.dumps({"vae": {"decoder": {"_class_name": "Decoder"}}})}
    mx.save_safetensors(str(tmp_path / "c.safetensors"), {"x": mx.zeros((1,))}, metadata=meta)
    with pytest.raises(ValueError, match="NADiffusionDecoder"):
        DiffusionDecoderConfig.from_safetensors_metadata(tmp_path / "c.safetensors")


@pytest.mark.skipif(LTX25_Q8_DIR is None, reason="local ltx-2.5-mlx-q8 pack not found")
def test_local_pack_metadata_matches_defaults():
    cfg = DiffusionDecoderConfig.from_safetensors_metadata(LTX25_Q8_DIR / "vae_decoder_av.safetensors")
    assert cfg == LTX_2_5_DIFFUSION_DECODER
