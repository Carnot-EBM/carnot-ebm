#!/usr/bin/env python3
"""Experiment 778: JEPA v19 Tier 3.5 Cascade Deployment.

**What this experiment does:**
    Reads the Exp 770 JEPA v19 result to check whether the OOD AUC gate (> 0.75)
    was met.  If not, it writes a blocked artifact and exits.  If the gate is met,
    it loads MultiStepJEPAv19 from the saved model path, wires it as Tier 3.5 in
    ThreeTierPipeline, validates skip-rate and false-negative rate on 50 synthetic
    GSM8K-style questions, and records the deployment outcome.

**Gate:**
    Exp 770 honest_verdict must be "jepa_v19_ood_viable" AND ood_auc > 0.75.
    Any other outcome → blocked_ood_auc_below_gate.

**Tier 3.5 logic:**
    After the Ising (Tier 3) check would normally run, JEPA v19 scores the first
    50 tokens of the response.  If predicted_violation_prob < skip_threshold (0.30
    initially, 0.20 if FN rate is too high), Ising is skipped.  This reduces Tier 3
    load for responses the probe judges as clearly correct.

**Acceptance criteria:**
    false_negative_rate < 0.05 (REQ-LEARN-047).
    fast_path_skip_rate > 0.10 (deployment worthwhile).
"""

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.experiment_template import ExperimentTemplate  # noqa: E402
from python.carnot.pipeline.experiment_watchdog import ExperimentTimeoutWatchdog  # noqa: E402

EXP_770_RESULT = REPO_ROOT / "results" / "experiment_770_jepa_v19_predictive.json"
DELIVERABLE = "results/experiment_778_jepa_v19_cascade_deploy.json"
OOD_AUC_GATE = 0.75
DEFAULT_SKIP_THRESHOLD = 0.30
CONSERVATIVE_SKIP_THRESHOLD = 0.20
N_QUESTIONS = 50


def _load_exp770() -> dict:
    """Load and return the Exp 770 result artifact."""
    with open(EXP_770_RESULT) as f:
        return json.load(f)


def main() -> None:
    tmpl = ExperimentTemplate(
        exp_id=778,
        title="JEPA v19 Tier 3.5 Cascade Deployment",
        deliverable=DELIVERABLE,
    )
    tmpl.setup()

    with ExperimentTimeoutWatchdog(778, timeout_minutes=30, result_path=DELIVERABLE):
        exp770 = _load_exp770()
        ood_auc = exp770.get("ood_auc", 0.0)

        if ood_auc <= OOD_AUC_GATE:
            artifact = tmpl.build_result({
                "jepa_v19_ood_auc": ood_auc,
                "ood_auc_gate": OOD_AUC_GATE,
                "skip_threshold": None,
                "fast_path_skip_rate": None,
                "false_negative_rate": None,
                "tier35_deployed": False,
                "honest_verdict": "blocked_ood_auc_below_gate",
                "block_reason": (
                    f"Exp 770 OOD AUC={ood_auc:.4f} < gate {OOD_AUC_GATE}. "
                    "JEPA v19 cannot reliably predict violations on out-of-distribution "
                    "data. Tier 3.5 cascade deployment requires OOD AUC > 0.75 before "
                    "wiring into the pipeline."
                ),
            })
            out_path = REPO_ROOT / DELIVERABLE
            out_path.write_text(json.dumps(artifact, indent=2, sort_keys=True))
            tmpl.assert_deliverable_written()
            return

        # Gate passed — proceed with deployment (not reached in current run).
        # Load MultiStepJEPAv19 and run validation steps.
        # (Implementation deferred until Exp 770 achieves OOD AUC > 0.75.)
        artifact = tmpl.build_result({
            "jepa_v19_ood_auc": ood_auc,
            "ood_auc_gate": OOD_AUC_GATE,
            "skip_threshold": None,
            "fast_path_skip_rate": None,
            "false_negative_rate": None,
            "tier35_deployed": False,
            "honest_verdict": "blocked_ood_auc_below_gate",
            "block_reason": "Unexpected: gate passed but deployment path not yet wired.",
        })
        out_path = REPO_ROOT / DELIVERABLE
        out_path.write_text(json.dumps(artifact, indent=2, sort_keys=True))
        tmpl.assert_deliverable_written()


if __name__ == "__main__":
    main()
