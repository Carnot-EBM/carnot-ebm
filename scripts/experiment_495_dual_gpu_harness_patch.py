#!/usr/bin/env python3
"""Experiment 495: DualGPU Harness Patch — apply auto-patch to all 53 flagged scripts.

**What this experiment does:**
    1. Calls apply_env_autofix() first (belt-and-suspenders RETRO-022 fix).
    2. Runs HarnessAudit over scripts/ to get the current set of needs_fix findings.
    3. Applies HarnessPatcher.patch_all() to every finding with needs_fix=True.
    4. Re-runs HarnessAudit to verify n_missing_cuda1 = 0 after patching.
    5. Writes a carnot.harness_patch.v1 artifact with all required fields.

**Why this matters (RETRO-041):**
    Exp 480 documented 53 scripts that load two models without assigning the second
    to cuda:1.  Milestone .36 measured GPU 1 at 11% utilization — documentation without
    execution does not change behavior.  This experiment applies the patch mechanically
    to all 53 scripts so GPU 1 receives correct VRAM assignments from now on.

**CPU-only:** patching is pure text manipulation.  No model loads, no GPU required.

Spec: REQ-INFRA-057, REQ-INFRA-058,
      SCENARIO-INFRA-065, SCENARIO-INFRA-066
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Step 0: apply_env_autofix() BEFORE any other pipeline import.
# ---------------------------------------------------------------------------
from carnot.pipeline.env_autofix import apply_env_autofix

_autofix_result = apply_env_autofix()

# ---------------------------------------------------------------------------
# Pipeline imports (safe after env fix)
# ---------------------------------------------------------------------------
from carnot.pipeline.deliverable_guard import DeliverableGuard
from carnot.pipeline.dual_gpu_harness import HarnessAudit
from carnot.pipeline.experiment_watchdog import ExperimentTimeoutWatchdog
from carnot.pipeline.harness_patcher import HarnessPatcher

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# ExperimentTemplate import
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

from experiment_template import ExperimentTemplate  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EXP_ID = 495
EXP_TITLE = "DualGPU Harness Patch"
DELIVERABLE_REL = "results/experiment_495_dual_gpu_harness_patch.json"
TIMEOUT_MINUTES = 30
SCRIPTS_DIR = str(_REPO_ROOT / "scripts")


def run_experiment(repo_root: Path) -> dict:
    """Run Experiment 495 and return the artifact dict."""
    scripts_dir = str(repo_root / "scripts")
    deliverable_abs = str(repo_root / DELIVERABLE_REL)

    # Phase 1: initial audit
    _log.info("Phase 1: running HarnessAudit on '%s'", scripts_dir)
    audit = HarnessAudit(scripts_dir)
    initial_findings = audit.scan()

    n_scripts_audited = len(initial_findings)
    n_dual_model_scripts = sum(1 for f in initial_findings if f.has_dual_model_load)
    n_needs_fix_before = sum(1 for f in initial_findings if f.needs_fix)

    _log.info(
        "Initial audit: %d total findings, %d dual-model, %d need cuda:1 fix",
        n_scripts_audited,
        n_dual_model_scripts,
        n_needs_fix_before,
    )

    # Phase 2: patch all flagged scripts
    _log.info("Phase 2: patching %d scripts with HarnessPatcher", n_needs_fix_before)
    patcher = HarnessPatcher(scripts_dir)
    patch_results = patcher.patch_all(initial_findings)

    n_patched = sum(1 for r in patch_results if r.was_patched)
    n_patch_errors = sum(1 for r in patch_results if r.error is not None)

    _log.info(
        "Patching complete: %d patched, %d errors",
        n_patched,
        n_patch_errors,
    )

    # Phase 3: re-audit to verify
    _log.info("Phase 3: re-auditing to verify n_missing_cuda1 = 0")
    n_remaining_violations = patcher.verify_clean(scripts_dir)

    _log.info("Post-patch remaining violations: %d (target: 0)", n_remaining_violations)

    # Compute patch_success_rate against dual-model scripts (the denominator that matters)
    patch_success_rate = n_patched / n_needs_fix_before if n_needs_fix_before > 0 else 1.0

    # Honest verdict
    if n_remaining_violations == 0:
        honest_verdict = "all_patched"
    elif n_patched > 0:
        honest_verdict = "partial_patch"
    else:
        honest_verdict = "no_change"

    # Build error details for transparency
    error_details = [
        {"script_path": r.script_path, "error": r.error}
        for r in patch_results
        if r.error is not None
    ]

    return {
        "schema": "carnot.harness_patch.v1",
        "n_scripts_audited": n_scripts_audited,
        "n_dual_model_scripts": n_dual_model_scripts,
        "n_needs_fix_before": n_needs_fix_before,
        "n_patched": n_patched,
        "n_patch_errors": n_patch_errors,
        "n_remaining_violations": n_remaining_violations,
        "patch_success_rate": round(patch_success_rate, 4),
        "honest_verdict": honest_verdict,
        "error_details": error_details,
        "env_autofix": {
            "gpu_detected": _autofix_result.gpu_detected,
            "auto_fix_applied": _autofix_result.auto_fix_applied,
            "final_env_value": _autofix_result.final_env_value,
        },
    }


def main() -> None:
    deliverable_abs = str(_REPO_ROOT / DELIVERABLE_REL)
    guard = DeliverableGuard(deliverable_abs)

    tmpl = ExperimentTemplate(
        exp_id=EXP_ID,
        title=EXP_TITLE,
        deliverable=DELIVERABLE_REL,
        requires_gpu=False,
    )
    tmpl.setup()

    with ExperimentTimeoutWatchdog(EXP_ID, timeout_minutes=TIMEOUT_MINUTES):
        try:
            payload = run_experiment(_REPO_ROOT)
            artifact = tmpl.build_result(payload, status="success")
        except Exception as exc:
            _log.error("Experiment %d failed: %s", EXP_ID, exc, exc_info=True)
            artifact = tmpl.build_result(
                {"error": str(exc)},
                status="error",
                honest_verdict="experiment_error",
            )

    # build_result() overwrites schema with sorted(keys) — restore our schema identifier.
    artifact["schema"] = "carnot.harness_patch.v1"

    # Write deliverable
    Path(deliverable_abs).parent.mkdir(parents=True, exist_ok=True)
    with open(deliverable_abs, "w", encoding="utf-8") as fh:
        json.dump(artifact, fh, indent=2)
    _log.info("Deliverable written: %s", deliverable_abs)

    tmpl.assert_deliverable_written()


if __name__ == "__main__":
    main()
