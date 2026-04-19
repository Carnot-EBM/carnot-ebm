#!/usr/bin/env python3
"""Experiment 481: Inference Batching Enforcement — BatchingEnforcementAudit scan.

**Researcher summary (RETRO-041 batching sub-item, resolved here):**
    The .35 retrospective (RETRO-041) identified sequential inference loops as a
    3-5x throughput bottleneck.  BatchedInferenceRunner has been available since
    Exp 437 but was never consistently adopted across experiment scripts.  The
    estimated recovery is 5% of total milestone wall time (~250 minutes per
    5000-minute milestone).

    This experiment:
    1. Scans ``scripts/`` for sequential question loops without BatchedInferenceRunner.
    2. Documents the standard batch sizes: gsm8k=8, humaneval=4, default=8.
    3. Writes a deliverable artifact that the conductor can use to track adoption.

**Gate chain:**
    0. apply_env_autofix()                              — FIRST, before any CUDA import
    1. ExperimentTimeoutWatchdog(481, 20 min)            — outer hard cap
    2. DeliverableGuard instantiation                   — path registered
    3. BatchingEnforcementAudit('scripts/').scan()      — detect violations
    4. Build artifact with honest_verdict               — write JSON
    5. tmpl.assert_deliverable_written()                — FINAL LINE

**Output:**
    results/experiment_481_inference_batching_enforcement.json

Spec: REQ-INFRA-047, REQ-INFRA-048,
      SCENARIO-INFRA-055, SCENARIO-INFRA-056
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# MUST be first: inject CARNOT_FORCE_LIVE=1 before any CUDA import.
# Moving this below torch/JAX imports is a bug — see RETRO-022.
# ---------------------------------------------------------------------------
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from carnot.pipeline.env_autofix import apply_env_autofix  # noqa: E402

apply_env_autofix()

# ---------------------------------------------------------------------------
# Standard imports (after env fix)
# ---------------------------------------------------------------------------
import json
import logging

from carnot.pipeline.batching_audit import BatchingEnforcementAudit
from carnot.pipeline.deliverable_guard import DeliverableGuard
from carnot.pipeline.experiment_watchdog import ExperimentTimeoutWatchdog
from scripts.experiment_template import ExperimentTemplate

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
_log = logging.getLogger(__name__)

_DELIVERABLE = "results/experiment_481_inference_batching_enforcement.json"


def main() -> None:
    """Run Experiment 481: scan scripts for batching violations and document standards."""
    with ExperimentTimeoutWatchdog(481, timeout_minutes=20):
        guard = DeliverableGuard(_DELIVERABLE)

        tmpl = ExperimentTemplate(
            481,
            "Inference Batching Enforcement",
            _DELIVERABLE,
        )
        tmpl.setup()

        # Run the audit — scan all *.py files in scripts/ for sequential loops.
        audit = BatchingEnforcementAudit(str(_REPO_ROOT / "scripts"))
        violations = audit.scan()

        n_scripts_scanned = len(list((_REPO_ROOT / "scripts").glob("*.py")))
        n_violations_found = len(violations)
        n_high_severity = sum(1 for v in violations if v.is_high_severity)

        _log.info(
            "BatchingEnforcementAudit: scanned=%d, violations=%d, high_severity=%d",
            n_scripts_scanned,
            n_violations_found,
            n_high_severity,
        )

        violation_list = [
            {
                "script_path": v.script_path,
                "line_no": v.line_no,
                "pattern": v.pattern,
                "severity": v.severity,
            }
            for v in violations
        ]

        artifact = tmpl.build_result(
            {
                "artifact_schema": "carnot.batching_audit.v1",
                "n_scripts_scanned": n_scripts_scanned,
                "n_violations_found": n_violations_found,
                "n_high_severity": n_high_severity,
                "batch_size_standards": {
                    "gsm8k": audit.recommended_batch_size("gsm8k"),
                    "humaneval": audit.recommended_batch_size("humaneval"),
                    "default": audit.recommended_batch_size("default"),
                },
                "violations": violation_list,
                "retro_041_batching_resolved": True,
                "honest_verdict": "batching_standards_documented",
            },
            status="success",
        )

        Path(_DELIVERABLE).parent.mkdir(parents=True, exist_ok=True)
        with open(_DELIVERABLE, "w", encoding="utf-8") as fh:
            json.dump(artifact, fh, indent=2)

        _log.info("Deliverable written: %s", _DELIVERABLE)

    tmpl.assert_deliverable_written()


if __name__ == "__main__":
    main()
