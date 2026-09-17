"""Minimal diffusion video decoder config for testing."""

from ltx_core_mlx.model.video_vae.diffusion_decoder.config import DiffusionDecoderConfig

TINY = DiffusionDecoderConfig(
    in_channels=8,
    out_channels=3,
    patch_size=4,
    head_dim=4,
    stage_channels=(16, 8, 8, 8, 4),
    stage_depths=(1, 1, 1, 1, 2),
    stage_kernels=((3, 3, 3), (3, 3, 3), (3, 3, 3), (3, 3, 3)),
    upsamples=(((1, 2, 2), 2), ((2, 1, 1), 1), ((2, 2, 2), 1), ((2, 2, 2), 2)),
    stage5_kernel=(3, 3, 3),
    t_embed_hidden=8,
    timestep_scale_multiplier=1000.0,
)
