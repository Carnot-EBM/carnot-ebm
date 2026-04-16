"""GPU node preflight check — RETRO-019 operational blocker resolution.

**Researcher summary (RETRO-019):**
    For FIVE consecutive milestones (2026.04.24 through 2026.04.28), the GPU node
    was offline or unavailable during conductor sessions, producing zero live results.
    The infrastructure fix (LiveGPUGate + session_startup.sh) was delivered in Exp 377.
    The remaining gap is OPERATIONAL: the GPU node must be physically powered on and
    connected before the session starts.

    This module provides a comprehensive preflight check that runs at the very start
    of each milestone (Exp 390) and produces a structured artifact with an honest
    verdict.  If the verdict is not ``"gpu_confirmed_live"``, the conductor must stop
    all GPU-dependent experiments immediately.

**Why five layers?**
    Each layer catches a distinct failure mode observed in prior milestones:
    1. ``env_var_set``        — CARNOT_FORCE_LIVE not exported (RETRO-012, RETRO-015)
    2. ``subprocess_inherits_env`` — var exported but not inherited by subprocesses
                                    (RETRO-015 root cause)
    3. ``session_startup_exists``  — scripts/session_startup.sh deleted or corrupted
    4. ``conductor_gpu_env_exists`` — scripts/conductor_gpu_env.sh missing
    5. ``is_live_capable``    — GPU hardware offline/driver crash (RETRO-019)
    6. ``smoke_test_passed``  — hardware live but model inference fails silently

**Honest verdict precedence (checked in order):**
    - ``"scripts_missing"``        when either startup script is absent
    - ``"env_not_propagating"``    when subprocess env inheritance is broken
    - ``"gpu_hardware_not_live"``  when diagnose_live_gpu() says is_live_capable=False
    - ``"gpu_confirmed_live"``     when ALL checks pass and smoke_test_passed=True
    Any other combination (partial failures) falls through to ``"gpu_hardware_not_live"``.

Spec: REQ-INFRA-017, REQ-INFRA-018,
      SCENARIO-INFRA-019, SCENARIO-INFRA-020, SCENARIO-INFRA-021
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from carnot.pipeline.live_gpu_diagnostic import diagnose_live_gpu
from carnot.pipeline.live_gpu_gate import LiveGPUGate
from carnot.pipeline.smoke_test import run_smoke_test

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# GPUPreflightResult dataclass
# ---------------------------------------------------------------------------


@dataclass
class GPUPreflightResult:
    """Structured result from ``run_gpu_preflight()``.

    Every field corresponds to one preflight check layer.  ``retro_019_resolved``
    is the summary: ``True`` iff ALL layers passed.  ``honest_verdict`` gives the
    human-readable outcome for the artifact and the conductor.

    Fields
    ------
    env_var_set : bool
        ``True`` iff ``CARNOT_FORCE_LIVE=1`` is present in the current environment.
        Checks the running process's env — if False, session_startup.sh was not sourced.
    subprocess_inherits_env : bool
        ``True`` iff a subprocess spawned by this process also sees ``CARNOT_FORCE_LIVE=1``.
        This is the proof that conductor subprocesses will inherit the var (RETRO-015 fix).
    session_startup_exists : bool
        ``True`` iff ``scripts/session_startup.sh`` exists on disk.
        This file is the canonical GPU env setup script (REQ-INFRA-017).
    conductor_gpu_env_exists : bool
        ``True`` iff ``scripts/conductor_gpu_env.sh`` exists on disk.
        This file exports CARNOT_FORCE_LIVE=1 and is sourced by session_startup.sh.
    is_live_capable : bool
        ``True`` iff ``diagnose_live_gpu().is_live_capable`` — all diagnostic layers
        passed (nvidia-smi, torch.cuda, model loadability).
    smoke_test_passed : bool
        ``True`` iff at least one model in the smoke-test list completed live GPU
        inference without raising.  Always ``False`` when ``is_live_capable=False``.
    model_ids_loadable : list[str]
        Model IDs that passed the smoke test (produced is_live=True).
        Empty when smoke_test_passed=False or is_live_capable=False.
    retro_019_resolved : bool
        Summary flag: ``True`` iff env_var_set AND subprocess_inherits_env AND
        is_live_capable AND smoke_test_passed all passed.
    honest_verdict : str
        Human-readable verdict for the artifact and conductor.
        See module docstring for the precedence rules.
    """

    env_var_set: bool
    subprocess_inherits_env: bool
    session_startup_exists: bool
    conductor_gpu_env_exists: bool
    is_live_capable: bool
    smoke_test_passed: bool
    model_ids_loadable: list[str] = field(default_factory=list)
    retro_019_resolved: bool = False
    honest_verdict: str = ""


# ---------------------------------------------------------------------------
# run_gpu_preflight
# ---------------------------------------------------------------------------


def run_gpu_preflight(
    project_root: Path,
    model_ids: list[str] | None = None,
) -> GPUPreflightResult:
    """Run all GPU preflight checks and return a ``GPUPreflightResult``.

    This is the single function Exp 390 calls.  It runs six checks in order
    and short-circuits where appropriate (e.g. smoke test is skipped when
    the GPU is not live-capable).

    **CI-safe guarantee:** This function NEVER raises.  Each layer is wrapped
    in try/except so that an unexpected exception (e.g. import error, OS error)
    is treated as a failure at that layer rather than crashing the preflight.

    Parameters
    ----------
    project_root : Path
        Repository root; used to check script existence.
    model_ids : list[str] | None
        Model IDs to use for smoke test.  Defaults to
        ``["google/gemma-4-E4B-it", "Qwen/Qwen3.5-0.8B"]``.

    Returns
    -------
    GPUPreflightResult
        Fully populated result.  ``retro_019_resolved=True`` iff all checks
        passed.

    Spec: REQ-INFRA-017, REQ-INFRA-018, SCENARIO-INFRA-019/020/021
    """
    if model_ids is None:
        model_ids = ["google/gemma-4-E4B-it", "Qwen/Qwen3.5-0.8B"]

    # --- Layer 1: env var in current process ---
    try:
        env_var_set = LiveGPUGate.check_env_var()
    except Exception as exc:  # pragma: no cover — defensive
        _log.warning("run_gpu_preflight: check_env_var raised: %s", exc)
        env_var_set = False

    # --- Layer 2: subprocess env propagation ---
    try:
        subprocess_inherits_env = LiveGPUGate.verify_subprocess_env_propagation()
    except Exception as exc:  # pragma: no cover — defensive
        _log.warning("run_gpu_preflight: verify_subprocess_env_propagation raised: %s", exc)
        subprocess_inherits_env = False

    # --- Layer 3: session_startup.sh existence ---
    try:
        session_startup_exists = (project_root / "scripts" / "session_startup.sh").is_file()
    except Exception as exc:  # pragma: no cover — defensive
        _log.warning("run_gpu_preflight: session_startup check raised: %s", exc)
        session_startup_exists = False

    # --- Layer 4: conductor_gpu_env.sh existence ---
    try:
        conductor_gpu_env_exists = (project_root / "scripts" / "conductor_gpu_env.sh").is_file()
    except Exception as exc:  # pragma: no cover — defensive
        _log.warning("run_gpu_preflight: conductor_gpu_env check raised: %s", exc)
        conductor_gpu_env_exists = False

    # --- Layer 5: live GPU capability (nvidia-smi + torch + model tokenizers) ---
    try:
        diag = diagnose_live_gpu(model_ids)
        is_live_capable = diag.is_live_capable
    except Exception as exc:  # pragma: no cover — defensive (diagnose_live_gpu is CI-safe)
        _log.warning("run_gpu_preflight: diagnose_live_gpu raised: %s", exc)
        is_live_capable = False

    # --- Layer 6: smoke test (only when GPU is live-capable) ---
    smoke_test_passed = False
    model_ids_loadable: list[str] = []

    if is_live_capable:
        for mid in model_ids:
            try:
                result = run_smoke_test(mid)
                if result.is_live:
                    smoke_test_passed = True
                    model_ids_loadable.append(mid)
            except Exception as exc:
                _log.warning(
                    "run_gpu_preflight: smoke test for %s raised: %s", mid, exc
                )

    # --- Compute summary fields ---
    retro_019_resolved = (
        env_var_set
        and subprocess_inherits_env
        and is_live_capable
        and smoke_test_passed
    )

    honest_verdict = _compute_honest_verdict(
        session_startup_exists=session_startup_exists,
        conductor_gpu_env_exists=conductor_gpu_env_exists,
        subprocess_inherits_env=subprocess_inherits_env,
        is_live_capable=is_live_capable,
        retro_019_resolved=retro_019_resolved,
    )

    return GPUPreflightResult(
        env_var_set=env_var_set,
        subprocess_inherits_env=subprocess_inherits_env,
        session_startup_exists=session_startup_exists,
        conductor_gpu_env_exists=conductor_gpu_env_exists,
        is_live_capable=is_live_capable,
        smoke_test_passed=smoke_test_passed,
        model_ids_loadable=model_ids_loadable,
        retro_019_resolved=retro_019_resolved,
        honest_verdict=honest_verdict,
    )


def _compute_honest_verdict(
    *,
    session_startup_exists: bool,
    conductor_gpu_env_exists: bool,
    subprocess_inherits_env: bool,
    is_live_capable: bool,
    retro_019_resolved: bool,
) -> str:
    """Return the honest_verdict string from the preflight check results.

    Precedence (checked in order, first match wins):
    1. ``"scripts_missing"``       — either startup script is absent
    2. ``"env_not_propagating"``   — subprocess env inheritance broken
    3. ``"gpu_hardware_not_live"`` — GPU not live-capable
    4. ``"gpu_confirmed_live"``    — all checks passed

    Separated for testability.
    """
    if not session_startup_exists or not conductor_gpu_env_exists:
        return "scripts_missing"
    if not subprocess_inherits_env:
        return "env_not_propagating"
    if not is_live_capable:
        return "gpu_hardware_not_live"
    if retro_019_resolved:
        return "gpu_confirmed_live"
    # is_live_capable=True but smoke_test_passed=False (model inference failure)
    return "gpu_hardware_not_live"


# ---------------------------------------------------------------------------
# build_preflight_artifact
# ---------------------------------------------------------------------------


def build_preflight_artifact(result: GPUPreflightResult) -> dict:
    """Build a serializable artifact dict from a ``GPUPreflightResult``.

    The artifact is written to ``results/experiment_390_gpu_preflight.json``
    by the experiment script.  All ``GPUPreflightResult`` fields are included
    verbatim plus the standard ``schema`` key for downstream tooling.

    Parameters
    ----------
    result : GPUPreflightResult
        The result from ``run_gpu_preflight()``.

    Returns
    -------
    dict
        JSON-serializable artifact with ``schema="carnot.gpu_preflight.v1"``
        and all GPUPreflightResult fields.

    Spec: REQ-INFRA-018
    """
    return {
        "schema": "carnot.gpu_preflight.v1",
        "honest_verdict": result.honest_verdict,
        "env_var_set": result.env_var_set,
        "subprocess_inherits_env": result.subprocess_inherits_env,
        "session_startup_exists": result.session_startup_exists,
        "conductor_gpu_env_exists": result.conductor_gpu_env_exists,
        "is_live_capable": result.is_live_capable,
        "smoke_test_passed": result.smoke_test_passed,
        "model_ids_loadable": result.model_ids_loadable,
        "retro_019_resolved": result.retro_019_resolved,
    }
