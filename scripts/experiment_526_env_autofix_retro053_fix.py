#!/usr/bin/env python3
"""Experiment 526 — env_autofix RETRO-053 Fix Verification.

**Researcher summary (RETRO-053, root cause 2026-04-19):**
    For seven consecutive milestones (.33-.39) RETRO-033 (live 100-question benchmark)
    missed because apply_env_autofix() checked PRESENCE of CARNOT_FORCE_LIVE but not
    its VALUE.  The conductor injected CARNOT_FORCE_LIVE='0' as a placeholder.  That
    satisfied the presence check, so injection was skipped and downstream truthiness
    gates deferred immediately (Exp 514: final_env_value='0').

    This experiment verifies the fix: apply_env_autofix() now treats '0', 'false',
    'False', and '' as equivalent to absent when gpu_detected=True, and overrides them
    to '1'.  The EnvironmentAutoFix result gains override_applied=True to distinguish
    the RETRO-053 scenario from the classic absent-var scenario.

**What this script verifies:**
    1. CARNOT_FORCE_LIVE='0' + gpu=True  → final_env_value='1', override_applied=True
    2. CARNOT_FORCE_LIVE='false' + gpu=True → same
    3. CARNOT_FORCE_LIVE='1' + gpu=True  → no change, override_applied=False
    4. CARNOT_FORCE_LIVE=None + gpu=True → auto_fix_applied=True, override_applied=False

Spec: REQ-INFRA-058, REQ-INFRA-059,
      SCENARIO-INFRA-067, SCENARIO-INFRA-068, SCENARIO-INFRA-069
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

# --- repo root on path ---
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# IMPORTANT: apply_env_autofix() is called FIRST, before ExperimentTemplate.
# In this experiment it is the thing under test, so we drive it via mocks.
from carnot.pipeline.env_autofix import apply_env_autofix  # noqa: E402
from carnot.pipeline.deliverable_guard import DeliverableGuard  # noqa: E402
from carnot.pipeline.experiment_watchdog import ExperimentTimeoutWatchdog  # noqa: E402
from scripts.experiment_template import ExperimentTemplate  # noqa: E402

logging.basicConfig(level=logging.INFO)
_log = logging.getLogger(__name__)

_DELIVERABLE = "results/experiment_526_env_autofix_retro053_fix.json"


# ---------------------------------------------------------------------------
# Edge-case verification helpers
# ---------------------------------------------------------------------------


def _run_autofix_with_env(value: str | None, gpu: bool) -> dict:
    """Run apply_env_autofix() in an isolated env with controlled GPU and CARNOT_FORCE_LIVE.

    Returns a summary dict of the EnvironmentAutoFix fields.  Because apply_env_autofix()
    mutates os.environ in-place, we wrap each call in patch.dict so other test cases start
    clean.
    """
    base_env = dict(os.environ)
    if value is None:
        base_env.pop("CARNOT_FORCE_LIVE", None)
    else:
        base_env["CARNOT_FORCE_LIVE"] = value

    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = gpu

    with patch.dict(os.environ, base_env, clear=True):
        with patch.dict(sys.modules, {"torch": mock_torch}):
            result = apply_env_autofix()
        final_env = os.environ.get("CARNOT_FORCE_LIVE")

    return {
        "input_value": value,
        "gpu_detected": result.gpu_detected,
        "auto_fix_applied": result.auto_fix_applied,
        "override_applied": result.override_applied,
        "carnot_force_live_was_set": result.carnot_force_live_was_set,
        "final_env_value": final_env,
    }


def verify_all_cases() -> dict:
    """Run all four RETRO-053 edge cases and assert expected outcomes.

    Returns a structured dict with per-case results and an overall pass/fail.
    """
    cases = []
    all_passed = True

    # --- Case 1: RETRO-053 scenario — '0' must be overridden ---
    c1 = _run_autofix_with_env("0", gpu=True)
    c1["case"] = "retro053_zero_value"
    c1["passed"] = (
        c1["auto_fix_applied"] is True
        and c1["override_applied"] is True
        and c1["final_env_value"] == "1"
    )
    cases.append(c1)
    if not c1["passed"]:
        _log.error("Case 1 FAILED: %s", c1)
        all_passed = False
    else:
        _log.info("Case 1 PASSED: CARNOT_FORCE_LIVE='0' overridden to '1'")

    # --- Case 2: 'false' must be overridden ---
    c2 = _run_autofix_with_env("false", gpu=True)
    c2["case"] = "false_value_override"
    c2["passed"] = (
        c2["auto_fix_applied"] is True
        and c2["override_applied"] is True
        and c2["final_env_value"] == "1"
    )
    cases.append(c2)
    if not c2["passed"]:
        _log.error("Case 2 FAILED: %s", c2)
        all_passed = False
    else:
        _log.info("Case 2 PASSED: CARNOT_FORCE_LIVE='false' overridden to '1'")

    # --- Case 3: '1' must NOT be overridden (SCENARIO-INFRA-069) ---
    c3 = _run_autofix_with_env("1", gpu=True)
    c3["case"] = "truthy_no_override"
    c3["passed"] = (
        c3["auto_fix_applied"] is False
        and c3["override_applied"] is False
        and c3["final_env_value"] == "1"
    )
    cases.append(c3)
    if not c3["passed"]:
        _log.error("Case 3 FAILED: %s", c3)
        all_passed = False
    else:
        _log.info("Case 3 PASSED: CARNOT_FORCE_LIVE='1' left unchanged")

    # --- Case 4: absent (None) → auto_fix_applied=True, override_applied=False ---
    c4 = _run_autofix_with_env(None, gpu=True)
    c4["case"] = "absent_auto_fix"
    c4["passed"] = (
        c4["auto_fix_applied"] is True
        and c4["override_applied"] is False
        and c4["final_env_value"] == "1"
    )
    cases.append(c4)
    if not c4["passed"]:
        _log.error("Case 4 FAILED: %s", c4)
        all_passed = False
    else:
        _log.info("Case 4 PASSED: absent CARNOT_FORCE_LIVE injected, override_applied=False")

    return {"cases": cases, "all_passed": all_passed}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Run Exp 526: verify RETRO-053 fix in apply_env_autofix()."""
    tmpl = ExperimentTemplate(
        526,
        "env_autofix RETRO-053 Fix",
        _DELIVERABLE,
        requires_gpu=False,
    )
    tmpl.setup()

    guard = DeliverableGuard(_DELIVERABLE)

    with ExperimentTimeoutWatchdog(526, timeout_minutes=15):
        verification = verify_all_cases()

        all_passed = verification["all_passed"]
        status = "success" if all_passed else "error"

        artifact = tmpl.build_result(
            {
                "schema": "carnot.env_autofix_fix.v1",
                "retro_053_resolved": all_passed,
                "override_applied_verified": all_passed,
                "honest_verdict": "retro_053_closed" if all_passed else "retro_053_verification_failed",
                "verification": verification,
            },
            status=status,
        )

    output_path = Path(_REPO_ROOT / _DELIVERABLE)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifact, indent=2))
    _log.info("Deliverable written: %s", output_path)

    guard.assert_written()
    tmpl.assert_deliverable_written()

    if not all_passed:
        sys.exit(1)

    _log.info("Exp 526 complete — RETRO-053 closed")


if __name__ == "__main__":
    main()
