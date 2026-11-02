"""EnvironmentAutoFix — self-injects CARNOT_FORCE_LIVE=1 when GPU hardware is detected.

**Researcher summary (RETRO-022, RESOLVED 2026-04-16; RETRO-053, RESOLVED 2026-04-19):**
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

**RETRO-053 root cause (2026-04-19):**
    For seven consecutive milestones (.33-.39) live benchmarks missed because the
    conductor injected ``CARNOT_FORCE_LIVE='0'`` as a placeholder default.  The old
    ``apply_env_autofix()`` checked only *presence* (``"CARNOT_FORCE_LIVE" in os.environ``)
    — so ``'0'`` satisfied the check and the injection was skipped.  Downstream gates check
    truthiness, so ``'0'`` caused immediate deferral (Exp 514 confirmed: ``final_env_value='0'``).
    Fix: treat any falsy value (``None``, ``''``, ``'0'``, ``'false'``, ``'False'``) as
    equivalent to absent when GPU is confirmed, and override to ``'1'``.

**What this module provides:**
    ``apply_env_autofix()`` — detects GPU hardware at process startup and, if GPU is
    present but the env gate is absent or falsy, injects ``CARNOT_FORCE_LIVE=1`` into the
    current process's environment.  This makes every GPU experiment self-configuring.

    ``build_env_autofix_artifact()`` — builds a structured JSON-serializable dict
    combining the autofix result with a prior GPU preflight result, including an honest
    verdict that can be used by downstream gating logic.

**Verdict semantics:**
    - ``'gpu_confirmed_live'``           — GPU detected AND var is now '1' (fallback)
    - ``'gpu_detected_env_was_correct'`` — GPU detected AND var was already truthy (no fix needed)
    - ``'gpu_not_detected'``             — GPU not present or torch not importable
    - ``'auto_fix_applied'``             — auto-fix applied (var was absent/None, now injected)
    - ``'falsy_override_applied'``       — falsy override applied (var was '0'/'false'/etc., overridden)

    Note: ``'auto_fix_applied'``, ``'falsy_override_applied'``, and
    ``'gpu_detected_env_was_correct'`` all count as ``retro_022_resolved=True`` because in
    each case the var is now '1' and live GPU experiments can proceed.

**Why log a warning on auto-fix?**
    Silent self-healing hides the underlying infrastructure problem.  The operator must
    know that env propagation is still broken so they can eventually fix it.  The warning
    is the signal; the auto-fix is just the workaround.

Spec: REQ-INFRA-021, REQ-INFRA-022, REQ-INFRA-058, REQ-INFRA-059,
      SCENARIO-INFRA-025, SCENARIO-INFRA-026, SCENARIO-INFRA-027 (Exp 413),
      SCENARIO-INFRA-067, SCENARIO-INFRA-068, SCENARIO-INFRA-069 (Exp 526)
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

_log = logging.getLogger(__name__)

# Values of CARNOT_FORCE_LIVE that are treated as "not set" when GPU is detected.
# A value of '0', 'false', 'False', or '' is a conductor placeholder that did NOT
# intend to disable live mode — it just wasn't set to a truthy value.  None means
# the var is genuinely absent.  All of these trigger the falsy override (RETRO-053).
FALSY_OVERRIDE_VALUES: frozenset[str | None] = frozenset({None, "", "0", "false", "False"})


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
        the call to ``apply_env_autofix()``.  Note: this is True even when the
        pre-existing value was falsy (e.g. '0') — it tracks presence, not truthiness.
    auto_fix_applied : bool
        ``True`` iff the var was absent (None) AND gpu_detected=True, causing injection.
        Also ``True`` when the var had a falsy value and was overridden.
    final_env_value : str | None
        The value of ``os.environ.get('CARNOT_FORCE_LIVE')`` AFTER the fix.
        ``'1'`` when the fix was applied or the var was already set to '1'.
        ``None`` when GPU was not detected (no mutation).
    override_applied : bool
        ``True`` iff a non-None falsy value ('0', 'false', 'False', '') was present
        and was overridden to '1'.  Distinguishes the RETRO-053 scenario (explicit
        falsy value) from the classic absent-var scenario.

    Spec: REQ-INFRA-021, REQ-INFRA-059, SCENARIO-INFRA-025/026/027/067/068/069
    """

    gpu_detected: bool
    carnot_force_live_was_set: bool
    auto_fix_applied: bool
    final_env_value: str | None
    override_applied: bool = False


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

    Spec: REQ-INFRA-021, REQ-INFRA-022, REQ-INFRA-058, SCENARIO-INFRA-025/026/027/067/068/069
    """
    # Step 1: record prior env state (presence check, not truthiness)
    carnot_force_live_was_set = "CARNOT_FORCE_LIVE" in os.environ
    current_val: str | None = os.environ.get("CARNOT_FORCE_LIVE")

    # Step 2: probe GPU hardware
    gpu_detected = False
    try:
        import torch  # noqa: PLC0415 — intentional late import (CI may lack torch)

        gpu_detected = bool(torch.cuda.is_available())
    except ImportError:
        # torch is not installed — GPU detection impossible, treat as no GPU
        gpu_detected = False

    # Step 3: apply fix if needed.
    # RETRO-053: the old check used `not carnot_force_live_was_set` which skipped injection
    # when CARNOT_FORCE_LIVE='0' was set as a conductor placeholder.  Now we check the VALUE
    # against FALSY_OVERRIDE_VALUES — any falsy value is treated as "not set" when GPU is live.
    auto_fix_applied = False
    override_applied = False
    if gpu_detected and current_val in FALSY_OVERRIDE_VALUES:
        os.environ["CARNOT_FORCE_LIVE"] = "1"
        auto_fix_applied = True
        # override_applied is True only when a non-None falsy value was explicitly present
        # (distinguishes '0'/'false'/'' from the var being completely absent)
        override_applied = current_val is not None
        _log.warning(
            "EnvironmentAutoFix applied CARNOT_FORCE_LIVE=1 "
            "(env propagation broken — conductor subprocess did not inherit the var; "
            "see RETRO-022 for root cause; RETRO-053 if value was '%s')",
            current_val,
        )

    # Step 4: record final env value
    final_env_value = os.environ.get("CARNOT_FORCE_LIVE")

    return EnvironmentAutoFix(
        gpu_detected=gpu_detected,
        carnot_force_live_was_set=carnot_force_live_was_set,
        auto_fix_applied=auto_fix_applied,
        final_env_value=final_env_value,
        override_applied=override_applied,
    )


# ---------------------------------------------------------------------------
# build_env_autofix_artifact
# ---------------------------------------------------------------------------

# Verdicts that mean RETRO-022 is resolved (var is '1' for this experiment run)
_RESOLVED_VERDICTS = {
    "gpu_confirmed_live",
    "gpu_detected_env_was_correct",
    "auto_fix_applied",
    "falsy_override_applied",
}


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
    2. ``'gpu_detected_env_was_correct'`` — gpu_detected=True AND var was already truthy
    3. ``'falsy_override_applied'``       — override_applied=True (falsy value overridden to '1')
    4. ``'auto_fix_applied'``             — auto_fix_applied=True (var was absent, now injected)
    5. ``'gpu_confirmed_live'``           — GPU detected and CARNOT_FORCE_LIVE='1' (fallback)

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
        fields (including ``override_applied``), and the full ``preflight_result``
        nested under ``'preflight'``.

    Spec: REQ-INFRA-021, REQ-INFRA-058, REQ-INFRA-059,
          SCENARIO-INFRA-025/026/027/067/068/069
    """
    if not result.gpu_detected:
        honest_verdict = "gpu_not_detected"
    elif result.override_applied:
        # RETRO-053: a non-None falsy value was present and overridden to '1'
        honest_verdict = "falsy_override_applied"
    elif result.carnot_force_live_was_set and not result.auto_fix_applied:
        # var was already truthy (e.g. '1') — no fix needed
        honest_verdict = "gpu_detected_env_was_correct"
    elif result.auto_fix_applied:
        # var was completely absent (None) — injected
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
        "override_applied": result.override_applied,
        "final_env_value": result.final_env_value,
        "preflight": preflight_result,
    }
