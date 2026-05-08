"""Optional macOS Metal watchdog guard.

When ``LTX2_METAL_WATCHDOG_GUARD=1``, helpers in this module insert
materialize + ``mx.synchronize()`` between heavy ops (Gemma layers,
connector blocks, etc.) so the Metal command buffer flushes before
hitting the kIOGPUCommandBufferCallbackErrorImpactingInteractivity
deadline (~10 s) — which can trip under post-boot Spotlight / Siri
indexer contention on Apple Silicon.

The guard is **off by default** because forced GPU completion limits
kernel pipelining on machines that don't need the protection (M2/M3
Ultra, Mac Studio with abundant GPU headroom). Enable it only when
you actually see the watchdog error:

    LTX2_METAL_WATCHDOG_GUARD=1 ltx-2-mlx generate ...

Companion env var ``LTX2_GEMMA_MAX_LENGTH`` (default 1024) can
additionally cap the Gemma padded sequence length for systems where
even per-layer sync isn't enough headroom — quality risk: left-padded
RoPE positions shift away from the training distribution.
"""

from __future__ import annotations

import os

import mlx.core as mx

_ENABLED = os.environ.get("LTX2_METAL_WATCHDOG_GUARD") == "1"


def is_enabled() -> bool:
    """Return whether the watchdog guard is active for this process."""
    return _ENABLED


def flush(*arrays: mx.array) -> None:
    """Materialize the given arrays and force GPU completion if guarded.

    No-op when ``LTX2_METAL_WATCHDOG_GUARD`` is not set — keeps full
    pipelining on capable hardware. When set, materializes the arrays
    via ``mx.eval`` and calls ``mx.synchronize`` to flush the Metal
    command buffer.

    Args:
        *arrays: Arrays whose computation should be materialized.
    """
    if not _ENABLED:
        return
    if arrays:
        _materialize = getattr(mx, "eval")  # noqa: B009
        _materialize(*arrays)
    mx.synchronize()
