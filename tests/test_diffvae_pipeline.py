"""Decoder selection, size guard and CLI plumbing for the diffusion video decoder (no weights)."""

from __future__ import annotations

import mlx.core as mx
import pytest

from ltx_pipelines_mlx._base import BasePipeline
from ltx_pipelines_mlx.utils import blocks as B  # noqa: N812


def test_video_decoder_block_defaults_to_conv_and_validates_choice(tmp_path):
    assert B.VideoDecoder(tmp_path).video_decoder == "conv"
    with pytest.raises(ValueError, match="video_decoder"):
        B.VideoDecoder(tmp_path, video_decoder="magic")


def test_diffusion_choice_requires_the_av_weights(tmp_path):
    vd = B.VideoDecoder(tmp_path, video_decoder="diffusion")
    with pytest.raises(FileNotFoundError, match=r"vae_decoder_av\.safetensors"):
        vd.load()


def test_stage5_token_estimate_and_guard(monkeypatch):
    # 512x768x49 -> latent (7, 16, 24) -> stage-5 grid (8*7-7=49) x (16*8=128) x (24*8=192) = 1_204_224
    assert B._DiffusionVideoDecoder.estimate_stage5_tokens((1, 128, 7, 16, 24)) == 49 * 128 * 192
    monkeypatch.setenv(B.DIFFVAE_MAX_TOKENS_ENV, "1000")
    with pytest.raises(ValueError, match="LTX2_DIFFVAE_MAX_TOKENS"):
        B._DiffusionVideoDecoder.check_size((1, 128, 7, 16, 24))
    monkeypatch.delenv(B.DIFFVAE_MAX_TOKENS_ENV)
    # The default is exactly the largest end-to-end validated shape (512x768x49).
    assert B.DIFFVAE_MAX_TOKENS_DEFAULT == 49 * 128 * 192
    B._DiffusionVideoDecoder.check_size((1, 128, 7, 16, 24))
    # One latent frame more (512x768x57) is above it and must raise.
    assert B._DiffusionVideoDecoder.estimate_stage5_tokens((1, 128, 8, 16, 24)) > B.DIFFVAE_MAX_TOKENS_DEFAULT
    with pytest.raises(ValueError, match="LTX2_DIFFVAE_MAX_TOKENS"):
        B._DiffusionVideoDecoder.check_size((1, 128, 8, 16, 24))


def test_diffusion_decode_and_stream_uses_shared_ffmpeg_plumbing(monkeypatch, tmp_path):
    seen = {}

    class _Dec:
        spatial_scale = (32, 32)

        def tiled_decode(self, latent, tiling=None, *, seed=0):
            seen["seed"] = seed
            yield mx.zeros((1, 3, 9, 64, 96))

    def _fake_ffmpeg_sink(cmd):
        seen["cmd"] = cmd
        return _FakeSink()

    monkeypatch.setattr(B, "_ffmpeg_sink", _fake_ffmpeg_sink)
    monkeypatch.setattr(
        B, "stream_chunks_to_ffmpeg", lambda chunks, proc: seen.setdefault("frames", sum(c.shape[2] for c in chunks))
    )
    wrapper = B._DiffusionVideoDecoder(_Dec())
    wrapper.decode_and_stream(
        mx.zeros((1, 128, 2, 2, 3)), str(tmp_path / "o.mp4"), frame_rate=24.0, audio_path=None, seed=7
    )
    assert seen["frames"] == 9
    # latent (1, 128, 2, 2, 3) -> h=2, w=3, spatial_scale (32, 32) -> ffmpeg "-s 96x64"
    assert "96x64" in " ".join(seen["cmd"])
    assert seen["seed"] == 7


class _FakeSink:
    def __enter__(self):
        class _P:
            stdin = None

        return _P()

    def __exit__(self, *a):
        return False


def test_base_pipeline_forwards_video_decoder_to_the_block(monkeypatch):
    p = BasePipeline.__new__(BasePipeline)
    assert p.video_decoder == "conv"
    p.verbose = False
    p.generate_audio = False
    p.video_decoder = "diffusion"
    calls = {}

    class _Blk:
        video_decoder = "conv"

        def load(self):
            calls["loaded_with"] = self.video_decoder

    p.video_decoder_block = _Blk()
    p.audio_decoder_block = None
    p._load_decoders()
    assert calls["loaded_with"] == "diffusion"


def _parse(*extra):
    from ltx_pipelines_mlx.cli import _build_parser

    return _build_parser().parse_args(
        ["generate", "-p", "x", "-o", "o.mp4", "--frame-rate", "24", "-f", "9", "--distilled", *extra]
    )


def test_cli_flag_defaults_and_parses():
    assert _parse().video_decoder == "conv"
    assert _parse("--video-decoder", "diffusion").video_decoder == "diffusion"


def _fake_pack(tmp_path):
    """A model dir that satisfies the diffusion decoder's weights precondition."""
    (tmp_path / "vae_decoder_av.safetensors").write_bytes(b"")
    return str(tmp_path)


@pytest.mark.parametrize("mode", ["--distilled", "--one-stage", "--two-stage", "--two-stages-hq"])
def test_cli_flag_reaches_every_generate_mode(monkeypatch, tmp_path, mode):
    seen = {}

    class _FakePipe:
        def __init__(self, *a, **k):
            pass

        def generate_and_save(self, **kwargs):
            seen["video_decoder"] = getattr(self, "video_decoder", "UNSET")

    import ltx_pipelines_mlx.distilled as d
    import ltx_pipelines_mlx.ti2vid_one_stage as o
    import ltx_pipelines_mlx.ti2vid_two_stages as t
    import ltx_pipelines_mlx.ti2vid_two_stages_hq as hq

    monkeypatch.setattr(d, "DistilledPipeline", _FakePipe)
    monkeypatch.setattr(o, "TI2VidOneStagePipeline", _FakePipe)
    monkeypatch.setattr(t, "TI2VidTwoStagesPipeline", _FakePipe)
    monkeypatch.setattr(hq, "TI2VidTwoStagesHQPipeline", _FakePipe)
    from ltx_pipelines_mlx.cli import _build_parser, _cmd_generate

    args = _build_parser().parse_args(
        [
            "generate",
            "-p",
            "x",
            "-o",
            "o.mp4",
            "--frame-rate",
            "24",
            "-f",
            "9",
            "--model",
            _fake_pack(tmp_path),
            mode,
            "--video-decoder",
            "diffusion",
            "--quiet",
        ]
    )
    _cmd_generate(args)
    assert seen["video_decoder"] == "diffusion"


def _diffusion_args(tmp_path, *extra):
    from ltx_pipelines_mlx.cli import _build_parser

    return _build_parser().parse_args(
        [
            "generate",
            "-p",
            "x",
            "-o",
            "o.mp4",
            "--frame-rate",
            "24",
            "--distilled",
            "--video-decoder",
            "diffusion",
            "--model",
            str(tmp_path),
            "--quiet",
            *extra,
        ]
    )


def _forbid_pipelines(monkeypatch):
    """Make every generate-mode pipeline class explode if it is ever constructed."""
    import ltx_pipelines_mlx.distilled as d
    import ltx_pipelines_mlx.ti2vid_one_stage as o
    import ltx_pipelines_mlx.ti2vid_two_stages as t
    import ltx_pipelines_mlx.ti2vid_two_stages_hq as hq

    class _Boom:
        def __init__(self, *a, **k):
            raise AssertionError("pipeline constructed before the diffusion preconditions were checked")

    monkeypatch.setattr(d, "DistilledPipeline", _Boom)
    monkeypatch.setattr(o, "TI2VidOneStagePipeline", _Boom)
    monkeypatch.setattr(t, "TI2VidTwoStagesPipeline", _Boom)
    monkeypatch.setattr(hq, "TI2VidTwoStagesHQPipeline", _Boom)


def test_missing_diffusion_weights_fail_before_any_pipeline_is_built(monkeypatch, tmp_path):
    from ltx_pipelines_mlx.cli import _cmd_generate

    _forbid_pipelines(monkeypatch)
    args = _diffusion_args(tmp_path, "-f", "9")  # tmp_path has no vae_decoder_av.safetensors
    with pytest.raises(FileNotFoundError, match=r"vae_decoder_av\.safetensors"):
        _cmd_generate(args)


def test_oversized_diffusion_target_fails_before_any_pipeline_is_built(monkeypatch, tmp_path):
    from ltx_pipelines_mlx.cli import _cmd_generate

    _forbid_pipelines(monkeypatch)
    _fake_pack(tmp_path)
    # 768x1280x97 is far above the validated single-tile ceiling.
    args = _diffusion_args(tmp_path, "-f", "97", "-H", "768", "-W", "1280")
    with pytest.raises(ValueError, match="LTX2_DIFFVAE_MAX_TOKENS"):
        _cmd_generate(args)
