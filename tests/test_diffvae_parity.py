"""Per-stage parity of the MLX diffusion decoder against upstream torch goldens.

Goldens come from ``tests/parity_diffvae_reference.py`` (upstream ``DiffusionVideoDecoder``,
``DiffVAEMode.CHUNKED_EAGER``, CPU fp32). This module skips when the npz or the local LTX-2.5
pack is missing; the npz is deliberately not committed (it is ~1 GB of activations).
"""

from __future__ import annotations

import os
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from ltx_core_mlx.model.video_vae.diffusion_decoder import load_diffusion_decoder
from tests.conftest import LTX25_Q8_DIR

NPZ = Path(os.environ.get("DIFFVAE_PARITY_NPZ", "/tmp/diffvae_parity.npz"))

pytestmark = pytest.mark.skipif(
    not NPZ.exists() or LTX25_Q8_DIR is None,
    reason="diffvae parity npz or local LTX-2.5 pack missing",
)

#: fp32 tolerances from the spec: deterministic stages are plain GEMM + small-window softmax,
#: stage 5 accumulates a 1331-key softmax per query and is allowed one more decade.
DET_TOL = 1e-4
DIFF_TOL = 1e-3


@pytest.fixture(scope="module")
def golden() -> dict[str, np.ndarray]:
    with np.load(NPZ) as data:
        return {k: data[k] for k in data.files}


@pytest.fixture(scope="module")
def decoder():
    dec = load_diffusion_decoder(LTX25_Q8_DIR / "vae_decoder_av.safetensors")
    dec.set_dtype(mx.float32)
    return dec


@pytest.fixture(scope="module")
def run(golden, decoder) -> dict[str, dict[str, np.ndarray]]:
    """Decode both golden latents once, keeping every tapped boundary as fp32 numpy."""
    results: dict[str, dict[str, np.ndarray]] = {}
    for tag in ("a", "b"):
        taps: dict[str, mx.array] = {}
        pixels = decoder.decode(
            mx.array(golden[f"{tag}.in.latent"]),
            noise=mx.array(golden[f"{tag}.in.noise"]),
            tap=lambda name, value, sink=taps: sink.__setitem__(name, value),
        )
        mx.eval(pixels, *taps.values())
        results[tag] = {k: np.array(v, copy=False) for k, v in taps.items()}
        results[tag]["out.pixels"] = np.array(pixels, copy=False)
    return results


def _assert_close(got: np.ndarray, want: np.ndarray, tol: float, what: str) -> None:
    assert got.shape == want.shape, f"{what}: shape {got.shape} != golden {want.shape}"
    err = float(np.abs(got.astype(np.float64) - want.astype(np.float64)).max())
    print(f"{what}: max abs err {err:.3e}")
    assert np.allclose(got, want, atol=tol, rtol=tol), f"{what}: max abs err {err:.3e} > {tol:.0e}"


@pytest.mark.parametrize("tag", ["a", "b"])
@pytest.mark.parametrize("stage", [1, 2, 3, 4])
def test_det_stage_boundary(golden, run, tag, stage):
    """Each deterministic stage's last-block output, before that stage's upsample."""
    _assert_close(run[tag][f"s{stage}.out"], golden[f"{tag}.s{stage}.out"], DET_TOL, f"{tag}.s{stage}.out")


@pytest.mark.parametrize("tag", ["a", "b"])
@pytest.mark.parametrize("block", range(8))
def test_diffusion_block_boundary(golden, run, tag, block):
    """Each stage-5 chunked diffusion block's output."""
    _assert_close(run[tag][f"s5.b{block}.out"], golden[f"{tag}.s5.b{block}.out"], DIFF_TOL, f"{tag}.s5.b{block}.out")


@pytest.mark.parametrize("tag", ["a", "b"])
def test_pixels(golden, run, tag):
    """Final cropped pixels in ``[-1, 1]``."""
    _assert_close(run[tag]["out.pixels"], golden[f"{tag}.out.pixels"], DIFF_TOL, f"{tag}.out.pixels")
