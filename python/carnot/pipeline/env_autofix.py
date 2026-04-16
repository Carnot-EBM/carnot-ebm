"""EnvironmentAutoFix — self-injects CARNOT_FORCE_LIVE=1 when GPU hardware is detected.

**Researcher summary (RETRO-022, RESOLVED at root cause 2026-04-16):**
    For SEVEN consecutive milestones (Exp 377 through Exp 411) live GPU experiments
    were blocked because ``CARNOT_FORCE_LIVE=1`` was set in the human shell but did NOT
    propagate into the Claude subprocess spawned by the conductor.  The subprocess env
    was correctly passed via ``env=env`` (line 202 of research_conductor.py) — the real
    gap was that the conductor's own process didn't have the var set when it was spawned
    by a non-interactive harness, so ``os.environ`` at spawn time lacked the var, and
    therefore every child lacked it too.

    Root cause fix (2026-04-16): ``research_conductor.main()`` now calls
    ``apply_env_autofix()`` at startup, so the var propagates to every Popen child via
    the existing ``env={**os.environ, ...}`` line.  This module remains as a belt-and-
    suspenders safeguard — experiments still call it first, in case the conductor hasn't.

    The earlier belief that "we CANNOT modify scripts/research_conductor.py" was a
    self-imposed rule, not a technical constraint, and it deferred the fix for 8+
    milestones.  Documented here so future retros don't re-open it.

**What this module provides:**
    ``apply_env_autofix()`` — detects GPU hardware at process startup and, if GPU is
    present but the env gate is absent, injects ``CARNOT_FORCE_LIVE=1`` into the current
    process's environment.  This makes every GPU experiment self-configuring.

    ``build_env_autofix_artifact()`` — builds a structured JSON-serializable dict
    combining the autofix result with a prior GPU preflight result, including an honest
    verdict that can be used by downstream gating logic.

**Verdict semantics:**
    - ``'gpu_confirmed_live'``        — GPU detected AND var is now '1' (was set or auto-fixed)
    - ``'gpu_detected_env_was_correct'`` — GPU detected AND var was already set (no fix needed)
    - ``'gpu_not_detected'``          — GPU not present or torch not importable
    - ``'auto_fix_applied'``          — auto-fix was applied (var injected because it was absent)

    Note: ``'auto_fix_applied'`` and ``'gpu_detected_env_was_correct'`` both count as
    ``retro_022_resolved=True`` because in either case the var is now '1' and live GPU
    experiments can proceed.

**Why log a warning on auto-fix?**
    Silent self-healing hides the underlying infrastructure problem.  The operator must
    know that env propagation is still broken so they can eventually fix it.  The warning
    is the signal; the auto-fix is just the workaround.

Spec: REQ-INFRA-021, REQ-INFRA-022,
      SCENARIO-INFRA-025, SCENARIO-INFRA-026, SCENARIO-INFRA-027 (Exp 413)
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# EnvironmentAutoFix dataclass
# ---------------------------------------------------------------------------


@dataclass
class EnvironmentAutoFix:
    """Structured result from ``apply_env_autofix()``.

    Fields
    ------
    gpu_detected : bool
        ``True`` iff ``torch.cuda.is_available()`` returned ``True``.
        ``False`` when torch is not importable or CUDA is not available.
    carnot_force_live_was_set : bool
        ``True`` iff ``CARNOT_FORCE_LIVE`` was already in ``os.environ`` before
        the call to ``apply_env_autofix()``.
    auto_fix_applied : bool
        ``True`` iff the fix was needed and applied:
        ``gpu_detected=True`` AND ``carnot_force_live_was_set=False``.
        In this case ``os.environ['CARNOT_FORCE_LIVE']`` is now ``'1'``.
    final_env_value : str | None
        The value of ``os.environ.get('CARNOT_FORCE_LIVE')`` AFTER the fix.
        ``'1'`` when the fix was applied or the var was already set.
        ``None`` when GPU was not detected (no mutation).

    Spec: REQ-INFRA-021, SCENARIO-INFRA-025/026/027
    """

    gpu_detected: bool
    carnot_force_live_was_set: bool
    auto_fix_applied: bool
    final_env_value: str | None


# ---------------------------------------------------------------------------
# apply_env_autofix
# ---------------------------------------------------------------------------


def apply_env_autofix() -> EnvironmentAutoFix:
    """Detect GPU hardware and inject ``CARNOT_FORCE_LIVE=1`` if absent.

    This function is intended to be called at the very top of every GPU experiment
    script, BEFORE ``ExperimentTemplate`` is instantiated.  It is a no-op when the
    env var is already set or when no GPU hardware is present.

    Algorithm
    ---------
    1. Record whether ``CARNOT_FORCE_LIVE`` is already in ``os.environ``.
    2. Try to import ``torch`` and call ``torch.cuda.is_available()``.
       If the import fails, ``gpu_detected=False``.
    3. If ``gpu_detected`` AND NOT ``carnot_force_live_was_set``:
       - Set ``os.environ['CARNOT_FORCE_LIVE'] = '1'``
       - Set ``auto_fix_applied=True``
       - Emit a WARNING log so the operator knows env propagation is still broken.
    4. Return ``EnvironmentAutoFix`` with all four fields populated.

    Returns
    -------
    EnvironmentAutoFix
        Fully populated result describing what was detected and what (if anything)
        was changed.

    Never raises.

    Spec: REQ-INFRA-021, REQ-INFRA-022, SCENARIO-INFRA-025/026/027
    """
    # Step 1: record prior env state
    carnot_force_live_was_set = "CARNOT_FORCE_LIVE" in os.environ

    # Step 2: probe GPU hardware
    gpu_detected = False
    try:
        import torch  # noqa: PLC0415 — intentional late import (CI may lack torch)

        gpu_detected = bool(torch.cuda.is_available())
    except ImportError:
        # torch is not installed — GPU detection impossible, treat as no GPU
        gpu_detected = False

    # Step 3: apply fix if needed
    auto_fix_applied = False
    if gpu_detected and not carnot_force_live_was_set:
        os.environ["CARNOT_FORCE_LIVE"] = "1"
        auto_fix_applied = True
        _log.warning(
            "EnvironmentAutoFix applied CARNOT_FORCE_LIVE=1 "
            "(env propagation broken — conductor subprocess did not inherit the var; "
            "see RETRO-022 for root cause)"
        )

    # Step 4: record final env value
    final_env_value = os.environ.get("CARNOT_FORCE_LIVE")

    return EnvironmentAutoFix(
        gpu_detected=gpu_detected,
        carnot_force_live_was_set=carnot_force_live_was_set,
        auto_fix_applied=auto_fix_applied,
        final_env_value=final_env_value,
    )


# ---------------------------------------------------------------------------
# build_env_autofix_artifact
# ---------------------------------------------------------------------------

# Verdicts that mean RETRO-022 is resolved (var is '1' for this experiment run)
_RESOLVED_VERDICTS = {"gpu_confirmed_live", "gpu_detected_env_was_correct", "auto_fix_applied"}


def build_env_autofix_artifact(
    result: EnvironmentAutoFix,
    preflight_result: dict,
) -> dict:
    """Build a serializable artifact combining autofix result and GPU preflight result.

    Computes an ``honest_verdict`` that faithfully describes the combined state of
    the environment fix and the GPU hardware check, then merges both into a single
    JSON-serializable dict.

    Verdict logic (first match wins):
    1. ``'gpu_not_detected'``             — gpu_detected=False
    2. ``'gpu_detected_env_was_correct'`` — gpu_detected=True AND carnot_force_live_was_set=True
    3. ``'auto_fix_applied'``             — auto_fix_applied=True (was absent, now injected)
    4. ``'gpu_confirmed_live'``           — GPU detected and CARNOT_FORCE_LIVE='1' (fallback)

    Parameters
    ----------
    result : EnvironmentAutoFix
        The result from ``apply_env_autofix()``.
    preflight_result : dict
        Serializable dict from a prior ``run_gpu_preflight()`` call (e.g. the
        contents of ``results/experiment_404_preflight_v2.json``).  Merged into
        the artifact under the ``preflight`` key.

    Returns
    -------
    dict
        JSON-serializable artifact with ``schema='carnot.env_autofix.v1'``,
        ``honest_verdict``, ``retro_022_resolved``, all ``EnvironmentAutoFix``
        fields, and the full ``preflight_result`` nested under ``'preflight'``.

    Spec: REQ-INFRA-021, SCENARIO-INFRA-025/026/027
    """
    if not result.gpu_detected:
        honest_verdict = "gpu_not_detected"
    elif result.carnot_force_live_was_set:
        honest_verdict = "gpu_detected_env_was_correct"
    elif result.auto_fix_applied:
        honest_verdict = "auto_fix_applied"
    else:
        # GPU detected + var is '1' but neither was pre-set nor auto-fixed (shouldn't
        # occur in practice, but cover it as a safe fallback)
        honest_verdict = "gpu_confirmed_live"

    retro_022_resolved = honest_verdict in _RESOLVED_VERDICTS

    return {
        "schema": "carnot.env_autofix.v1",
        "honest_verdict": honest_verdict,
        "retro_022_resolved": retro_022_resolved,
        "gpu_detected": result.gpu_detected,
        "carnot_force_live_was_set": result.carnot_force_live_was_set,
        "auto_fix_applied": result.auto_fix_applied,
        "final_env_value": result.final_env_value,
        "preflight": preflight_result,
    }
