"""GPU1 Zombie Fix — explicit device_map assignment to prevent GPU1 zombie allocation.

**Researcher summary (RETRO-025, fixed in Exp 438):**
    PID 3509070 held 1786 MB on GPU1 at 0% utilization while GPU0 ran at 88%
    for 144+ minutes.  Root cause: ``device_map='auto'`` lets CUDA distribute model
    layers across available GPUs during load, but the actual forward pass ran only
    on GPU0.  GPU1 accumulated VRAM for offloaded layers it never executed.

    Fix: when running live with N >= 2 GPUs, assign each model to its own GPU using
    ``device_map={'': 'cuda:N'}`` rather than 'auto'.  This pins every layer of the
    model to the specified device, preventing cross-GPU VRAM spill while ensuring
    the forward pass actually computes on the assigned GPU.

    Single-GPU and CI mode continue to use ``device_map='auto'`` (safe fallback).

**What this module provides:**

    ``ZombieFixResult`` — captures the pre/post state of the zombie fix attempt,
    including which device_map strings were used and whether GPU1 utilization
    rose after model load.

    ``build_zombie_fix_strategy(n_gpus, model_ids)`` — returns a dict mapping
    each model ID to the device_map it should use.  For dual-GPU live mode,
    this is ``{'': 'cuda:N'}``; for single-GPU / CI, it is ``'auto'``.

    ``build_zombie_fix_artifact(result)`` — serialises a ``ZombieFixResult`` into
    a JSON-ready dict with ``schema='carnot.gpu1_zombie_fix.v1'`` and an honest
    ``retro_025_status`` that callers can log for traceability.

Spec: REQ-INFRA-029, REQ-INFRA-030,
      SCENARIO-INFRA-037, SCENARIO-INFRA-038 (Exp 438)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


# ---------------------------------------------------------------------------
# ZombieFixResult
# ---------------------------------------------------------------------------


@dataclass
class ZombieFixResult:
    """Captures the outcome of the GPU1 zombie fix attempt.

    Fields
    ------
    gpu0_model_id : str
        HuggingFace model ID loaded on GPU0 (e.g. 'Qwen/Qwen3.5-0.8B').
    gpu1_model_id : str
        HuggingFace model ID loaded on GPU1.  Empty string when only one model
        was loaded or when running in CI/single-GPU mode.
    gpu0_device_map : str
        The device_map value used for the GPU0 model.
        ``"{'': 'cuda:0'}"`` in explicit mode, ``'auto'`` in fallback mode.
    gpu1_device_map : str
        The device_map value used for the GPU1 model.
        ``"{'': 'cuda:1'}"`` in explicit mode, ``'auto'`` in fallback mode.
        Empty string when no GPU1 model was loaded.
    fix_applied : bool
        ``True`` when explicit ``{'': 'cuda:N'}`` assignment was used (not 'auto').
        ``False`` in CI or single-GPU fallback mode.
    post_fix_gpu1_util_pct : float | None
        GPU1 compute utilization percentage measured after model load.
        A positive value confirms the fix worked (GPU1 is now computing).
        ``None`` when running in CI mode (no GPU hardware available).
    honest_verdict : str
        One of:
        - ``'fix_applied_and_verified'``: fix applied AND gpu1_util > 0 after load
        - ``'fix_applied_unverified'``:  fix applied but gpu1_util still 0 or unknown
        - ``'ci_mode'``:                 no GPU hardware — fix logic ran but not confirmed

    Spec: REQ-INFRA-030, SCENARIO-INFRA-037/038
    """

    gpu0_model_id: str
    gpu1_model_id: str
    gpu0_device_map: str
    gpu1_device_map: str
    fix_applied: bool
    post_fix_gpu1_util_pct: Optional[float]
    honest_verdict: str


# ---------------------------------------------------------------------------
# build_zombie_fix_strategy
# ---------------------------------------------------------------------------


def build_zombie_fix_strategy(
    n_gpus: int,
    model_ids: list[str],
) -> dict[str, object]:
    """Return per-model device_map specs that prevent GPU1 zombie allocation.

    In dual-GPU live mode (n_gpus >= 2 and at least 2 model_ids), this assigns
    each of the first two models to an explicit ``{'': 'cuda:N'}`` device map.
    This prevents ``device_map='auto'`` from distributing layers across GPUs in a
    way that allocates VRAM on GPU1 without actually executing there (RETRO-025).

    In single-GPU or CI mode (n_gpus < 2), all models receive ``'auto'`` as the
    fallback — the zombie pattern cannot occur when there is only one GPU.

    Parameters
    ----------
    n_gpus : int
        Number of CUDA GPUs detected.  Pass 0 for CI/CPU-only machines.
    model_ids : list[str]
        Ordered list of model HuggingFace IDs.  The first two are assigned to
        GPU0 and GPU1 respectively in dual-GPU mode.

    Returns
    -------
    dict[str, object]
        Maps each model_id to its device_map value:
        - ``{'': 'cuda:0'}`` for model_ids[0] in dual-GPU mode
        - ``{'': 'cuda:1'}`` for model_ids[1] in dual-GPU mode
        - ``'auto'`` for all remaining models or in single-GPU/CI mode

    Spec: REQ-INFRA-029, SCENARIO-INFRA-037, SCENARIO-INFRA-038
    """
    strategy: dict[str, object] = {}

    if n_gpus >= 2 and len(model_ids) >= 2:
        # Explicit per-GPU assignment — each model is pinned to its own GPU.
        # This is the RETRO-025 fix: 'auto' allowed VRAM spill across devices
        # without ensuring compute happened there.
        for gpu_idx, model_id in enumerate(model_ids):
            if gpu_idx < 2:
                strategy[model_id] = {"": f"cuda:{gpu_idx}"}
            else:
                # Models beyond index 1 fall back to auto — we only manage 2 GPUs.
                strategy[model_id] = "auto"
    else:
        # Single GPU or CI: 'auto' is correct and safe here because CUDA cannot
        # create a zombie on a device that doesn't exist.
        for model_id in model_ids:
            strategy[model_id] = "auto"

    return strategy


# ---------------------------------------------------------------------------
# build_zombie_fix_artifact
# ---------------------------------------------------------------------------


def build_zombie_fix_artifact(result: ZombieFixResult) -> dict:
    """Serialise a ZombieFixResult into a JSON-ready artifact dict.

    The ``retro_025_status`` field mirrors ``honest_verdict`` and is included
    for consistency with the RETRO-025 tracking field used in Exps 426/436.

    Parameters
    ----------
    result : ZombieFixResult
        The outcome of the zombie fix attempt from Exp 438.

    Returns
    -------
    dict
        JSON-serializable artifact with ``schema='carnot.gpu1_zombie_fix.v1'``.

    Spec: REQ-INFRA-030, SCENARIO-INFRA-037/038 (Exp 438)
    """
    return {
        "schema": "carnot.gpu1_zombie_fix.v1",
        "honest_verdict": result.honest_verdict,
        "retro_025_status": result.honest_verdict,
        "gpu0_model_id": result.gpu0_model_id,
        "gpu1_model_id": result.gpu1_model_id,
        "gpu0_device_map": result.gpu0_device_map,
        "gpu1_device_map": result.gpu1_device_map,
        "fix_applied": result.fix_applied,
        "post_fix_gpu1_util_pct": result.post_fix_gpu1_util_pct,
    }
