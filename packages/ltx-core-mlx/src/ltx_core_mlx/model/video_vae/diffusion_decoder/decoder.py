"""``NADiffusionDecoder``: the LTX-2.5 diffusion video decoder, plain path, single tile.

Flow (spec §3): size floor + ghost pad -> de-normalise -> conv_in -> det stages 1-3 with
upsamples -> stage 4 blocks (upsample 3 deferred to the diffusion blocks) -> ghost crop ->
pure-noise ``x_t`` -> one diffusion step at ``t = 1`` (``x0`` output is the pixels) -> crop.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

from ltx_core_mlx.model.video_vae.diffusion_decoder.blocks import NABlock
from ltx_core_mlx.model.video_vae.diffusion_decoder.chunked import ChunkedDiffusionNABlock
from ltx_core_mlx.model.video_vae.diffusion_decoder.config import LTX_2_5_DIFFUSION_DECODER, DiffusionDecoderConfig
from ltx_core_mlx.model.video_vae.diffusion_decoder.layers import (
    LinearPixelShuffleUpsample,
    RMSNorm,
    SharedAdaLN,
    TimestepEmbedder,
)
from ltx_core_mlx.model.video_vae.diffusion_decoder.patching import patchify_pixels, unpatchify_pixels
from ltx_core_mlx.utils.weights import load_split_safetensors

#: Decorrelates the decoder's noise draw from the sampler's (which reuses ``seed``).
DIFFVAE_NOISE_SEED_OFFSET = 30000


class PerChannelStats(nn.Module):
    """Per-channel de-normalisation statistics (``mean``, ``std``) for the input latent."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.mean = mx.zeros((channels,))
        self.std = mx.ones((channels,))


class NADiffusionDecoder(nn.Module):
    """See module docstring. Parameter names mirror the pack keys (prefix ``vae_decoder_av.``)."""

    def __init__(self, config: DiffusionDecoderConfig = LTX_2_5_DIFFUSION_DECODER) -> None:
        super().__init__()
        self.config = config
        c = config
        self.per_channel_statistics = PerChannelStats(c.in_channels)
        self.conv_in = nn.Linear(c.in_channels, c.stage_channels[0])
        # Added by upstream only to keyframe latents in the keyframe-aware decode
        # (DecodeKeyframes); the plain decode never reads it. Loaded for the load
        # contract; inert in v1.
        self.type_emb = mx.zeros((c.in_channels,))
        self.det_stages = [
            [NABlock(c.stage_channels[s], c.head_dim, c.stage_kernels[s]) for _ in range(c.stage_depths[s])]
            for s in range(c.num_det_stages)
        ]
        self.upsamples = [
            LinearPixelShuffleUpsample(c.stage_channels[s], stride, reduction)
            for s, (stride, reduction) in enumerate(c.upsamples)
        ]
        pp = c.patch_size * c.patch_size
        self.conv_in_x_t = nn.Linear(c.out_channels * pp, c.diff_channels)
        self.t_embedder = TimestepEmbedder(c.t_freq_dim, c.t_embed_hidden)
        self.shared_adaln = SharedAdaLN(c.t_embed_hidden, c.diff_channels)
        self.diff_blocks = [
            ChunkedDiffusionNABlock(c.diff_channels, c.head_dim, c.stage5_kernel, context_channels=c.diff_channels)
            for _ in range(c.diff_depth)
        ]
        self.norm_out = RMSNorm(c.diff_channels)
        self.conv_out = nn.Linear(c.diff_channels, c.out_channels * pp)

        st, sh, sw = c.cumulative_strides()[4]
        self.temporal_scale = st
        self.spatial_scale = (sh * c.patch_size, sw * c.patch_size)

    # ---- geometry -----------------------------------------------------------------
    def denormalize_latent(self, z: mx.array) -> mx.array:
        s, m = self.per_channel_statistics.std, self.per_channel_statistics.mean
        return z * s.reshape(1, -1, 1, 1, 1).astype(z.dtype) + m.reshape(1, -1, 1, 1, 1).astype(z.dtype)

    def pad_to_floor(self, latent: mx.array) -> tuple[mx.array, tuple[int, int, int, int, int]]:
        """Pad ``(B, C, F, H, W)`` up to the config's minimum latent shape (T: repeat last; H/W: symmetric edge)."""
        f_min, h_min, w_min = self.config.min_latent_shape()
        _, _, f, h, w = latent.shape
        t_pad = max(f_min - f, 0)
        h_need, w_need = max(h_min - h, 0), max(w_min - w, 0)
        h_b, h_a = h_need // 2, h_need - h_need // 2
        w_b, w_a = w_need // 2, w_need - w_need // 2
        if t_pad:
            latent = mx.concatenate([latent, mx.repeat(latent[:, :, -1:], t_pad, axis=2)], axis=2)
        if h_need:
            latent = mx.concatenate(
                [mx.repeat(latent[:, :, :, :1], h_b, axis=3), latent, mx.repeat(latent[:, :, :, -1:], h_a, axis=3)],
                axis=3,
            )
        if w_need:
            latent = mx.concatenate(
                [
                    mx.repeat(latent[:, :, :, :, :1], w_b, axis=4),
                    latent,
                    mx.repeat(latent[:, :, :, :, -1:], w_a, axis=4),
                ],
                axis=4,
            )
        return latent, (t_pad, h_b, h_a, w_b, w_a)

    def _ghost_crop_keep(self, t4: int) -> int:
        strides = self.config.cumulative_strides()[3]  # stride at stage-4 input relative to the latent
        ghost_at_s4 = self.config.ghost_pad_frames() * strides[0]
        return min(t4, max(t4 - ghost_at_s4, math.ceil(self.config.stage5_kernel[0] / 2)))

    def stage5_canvas(self, t4_kept: int, h4: int, w4: int) -> tuple[int, int, int]:
        (st, sh, sw), _ = self.config.upsamples[3]
        p = self.config.patch_size
        return t4_kept * st - 1, h4 * sh * p, w4 * sw * p

    # ---- forward pieces -----------------------------------------------------------
    def forward_stages_1_to_4(
        self, latent_padded: mx.array, *, tap: Callable[[str, mx.array], None] | None = None
    ) -> mx.array:
        """De-normalise, ghost-pad, run det stages 1-4 (upsample 3 deferred), crop the ghost context.

        ``tap``, when given, receives ``("s{s+1}.out", x)`` after the last block of stage ``s``
        (before that stage's upsample) -- the parity boundaries of the torch goldens.
        """
        z = self.denormalize_latent(latent_padded)
        ghost = self.config.ghost_pad_frames()
        z = mx.concatenate([z, mx.repeat(z[:, :, -1:], ghost, axis=2)], axis=2)
        x = self.conv_in(z.transpose(0, 2, 3, 4, 1))  # (B, T, H, W, C0)
        for s in range(self.config.num_det_stages):
            for block in self.det_stages[s]:
                x = block(x)
            if tap is not None:
                tap(f"s{s + 1}.out", x)
            if s < 3:
                x = self.upsamples[s](x, drop_leading_frame=True)
        keep = self._ghost_crop_keep(x.shape[1])
        return x[:, :keep]

    def forward_stage_5(
        self,
        x_t: mx.array,
        stage4_feat: mx.array,
        t: mx.array,
        *,
        tap: Callable[[str, mx.array], None] | None = None,
    ) -> mx.array:
        """One diffusion evaluation at timestep ``t`` (``(B,)``); returns pixels ``(B, 3, F5, H_px, W_px)``.

        ``tap``, when given, receives ``("s5.b{i}.out", x)`` after each diffusion block.
        """
        x = self.conv_in_x_t(patchify_pixels(x_t, self.config.patch_size))
        t_emb = self.t_embedder(t * self.config.timestep_scale_multiplier)
        modulation = self.shared_adaln(t_emb)
        for i, block in enumerate(self.diff_blocks):
            x = block(x, stage4_feat, self.upsamples[3], modulation, drop_leading_frame=True)
            if tap is not None:
                tap(f"s5.b{i}.out", x)
        x = self.conv_out(self.norm_out(x))
        return unpatchify_pixels(x, self.config.patch_size, self.config.out_channels)

    # ---- public API ---------------------------------------------------------------
    def decode(
        self,
        latent: mx.array,
        *,
        noise: mx.array | None = None,
        seed: int = 0,
        tap: Callable[[str, mx.array], None] | None = None,
    ) -> mx.array:
        """Decode ``(B, 128, F, H, W)`` normalised latent to ``(B, 3, 8F-7, 32H, 32W)`` pixels in ``[-1, 1]``.

        ``tap`` is the parity instrumentation hook; see :meth:`forward_stages_1_to_4` and
        :meth:`forward_stage_5` for the names it is called with.
        """
        if latent.shape[0] != 1:
            raise ValueError("NADiffusionDecoder decodes one video at a time (batch size 1)")
        _, _, f, h, w = latent.shape
        # T pads live at the end (repeat-last-frame) and are removed by the [:f_px] crop below.
        padded, (_t_pad, h_b, _h_a, w_b, _w_a) = self.pad_to_floor(latent)
        feat = self.forward_stages_1_to_4(padded, tap=tap)
        t4, h4, w4 = feat.shape[1], feat.shape[2], feat.shape[3]
        canvas = self.stage5_canvas(t4, h4, w4)
        if noise is None:
            key = mx.random.key(seed + DIFFVAE_NOISE_SEED_OFFSET)
            noise = mx.random.normal((1, self.config.out_channels, *canvas), key=key)
        elif tuple(noise.shape) != (1, self.config.out_channels, *canvas):
            raise ValueError(f"noise must have shape {(1, self.config.out_channels, *canvas)}, got {noise.shape}")
        pixels = self.forward_stage_5(noise.astype(latent.dtype), feat, mx.array([1.0]), tap=tap)
        sh, sw = self.spatial_scale
        f_px, h_px, w_px = (f - 1) * self.temporal_scale + 1, h * sh, w * sw
        hb, wb = h_b * sh, w_b * sw
        return pixels[:, :, :f_px, hb : hb + h_px, wb : wb + w_px]

    def tiled_decode(self, latent: mx.array, tiling: object | None = None, *, seed: int = 0) -> Iterator[mx.array]:
        """Single-tile decode yielding one ``(B, 3, F, H, W)`` chunk.

        API parity with the conv decoder; ``tiling`` is accepted but ignored until PR B.
        """
        del tiling
        yield self.decode(latent, seed=seed)


def load_diffusion_decoder(path: str | Path, config: DiffusionDecoderConfig | None = None) -> NADiffusionDecoder:
    """Build and load an :class:`NADiffusionDecoder` from a pack's ``vae_decoder_av.safetensors`` (strict)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"diffusion video decoder weights not found at {path}")
    cfg = config or DiffusionDecoderConfig.from_safetensors_metadata(path)
    model = NADiffusionDecoder(cfg)
    weights = load_split_safetensors(path, prefix="vae_decoder_av.")
    model.load_weights(list(weights.items()))  # strict: every param fed, no unknown key
    return model
