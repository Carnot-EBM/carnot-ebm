#!/usr/bin/env python3
"""Exp 674: IAS Adaptive Gate Calibration — per-extractor quantile regression thresholds.

**Research question:**
    Can per-extractor quantile regression calibration (arXiv 2506.09338) fix the
    EnsembleGate v3 failure mode where gate_closed was returned despite causal_recall=0.36?

**Background:**
    Exp .50 demonstrated a concrete failure: the v3 gate returned gate_closed even though
    causal_recall=0.36 individually exceeded the fixed threshold=0.30.  The failure occurred
    because v3 averages all four recall components including HermesV2 (which scores ~0.0 on
    mixed-format sets), dragging the ensemble below the gate threshold.

    EnsembleGate v4 partially fixed this by excluding HermesV2 and using OR-logic, but
    still uses a fixed threshold=0.30 for all extractors.  IAS (Instance-Adaptive Scaling)
    goes further: calibrate a separate threshold for each extractor by fitting the 10th
    percentile of that extractor's recall distribution over FOVER pairs.  High-variance
    extractors get a lower effective threshold; low-variance extractors get a higher one.

**Protocol:**
    1. Calibrate gate thresholds from results/fover_labeled_steps_live.json (57 FOVER pairs).
    2. Apply to .50 recall values: symcode=0.12, structured=0.20, causal=0.36.
    3. Compare IAS gate decision against v3 (gate_closed) and v4 (gate_open).
    4. Record honest_verdict based on whether IAS improves over v3.

Spec: REQ-VERIFY-151, REQ-VERIFY-152, SCENARIO-VERIFY-200, SCENARIO-VERIFY-201
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Ensure repo root is on sys.path so we can import from scripts/ and python/.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_REPO_ROOT / "python") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "python"))

from carnot.pipeline.ensemble_gate_v4 import EnsembleGateV4  # noqa: E402
from carnot.pipeline.experiment_watchdog import ExperimentTimeoutWatchdog  # noqa: E402
from carnot.pipeline.ias_gate_calibration import (  # noqa: E402
    IASGateCalibration,
    adaptive_gate_open,
    calibrate,
)
from scripts.experiment_template import ExperimentTemplate  # noqa: E402

EXP_ID = 674
TITLE = "IAS Adaptive Gate Calibration — per-extractor quantile regression thresholds"
DELIVERABLE = "results/experiment_674_ias_adaptive_gate.json"

# .50 recall values that caused the v3 gate failure.
RECALL_SYMCODE = 0.12
RECALL_STRUCTURED = 0.20
RECALL_CAUSAL = 0.36
RECALL_HERMES_V2 = 0.0  # HermesV2 was excluded from v4; kept for v3 context.

FOVER_PAIRS_PATH = str(_REPO_ROOT / "results" / "fover_labeled_steps_live.json")


def _write_artifact(artifact: dict, path: Path) -> None:
    """Write artifact JSON atomically (temp file + rename) to avoid partial writes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(artifact, indent=2))
    tmp.rename(path)


def main() -> None:
    """Run Exp 674: IAS adaptive gate calibration and comparison."""
    tmpl = ExperimentTemplate(
        exp_id=EXP_ID,
        title=TITLE,
        deliverable=DELIVERABLE,
    )
    tmpl.setup()

    output_path = _REPO_ROOT / DELIVERABLE

    with ExperimentTimeoutWatchdog(
        experiment_id=EXP_ID,
        timeout_minutes=20,
        result_path=str(output_path),
    ):
        # ------------------------------------------------------------------
        # Step 1: Calibrate IAS thresholds from 57 FOVER pairs.
        # Fits 10th-percentile quantile regression for each extractor.
        # ------------------------------------------------------------------
        calibration: IASGateCalibration = calibrate(FOVER_PAIRS_PATH)

        calibrated_thresholds = {
            "symcode": calibration.symcode_threshold,
            "structured": calibration.structured_threshold,
            "causal": calibration.causal_threshold,
            "calibrated_from_n": calibration.calibrated_from_n,
        }

        # ------------------------------------------------------------------
        # Step 2: Apply IAS gate to .50 recall values.
        # ------------------------------------------------------------------
        ias_gate_open: bool = adaptive_gate_open(
            calibration,
            symcode=RECALL_SYMCODE,
            structured=RECALL_STRUCTURED,
            causal=RECALL_CAUSAL,
        )

        # ------------------------------------------------------------------
        # Step 3: Recompute v3 gate (fixed threshold=0.30, ensemble average).
        # v3 averages all four components including HermesV2.
        # ------------------------------------------------------------------
        v3_ensemble = (
            RECALL_SYMCODE + RECALL_HERMES_V2 + RECALL_STRUCTURED + RECALL_CAUSAL
        ) / 4.0
        v3_gate_open: bool = v3_ensemble >= 0.30

        # ------------------------------------------------------------------
        # Step 4: Recompute v4 gate (fixed threshold, OR-logic, no HermesV2).
        # ------------------------------------------------------------------
        gate_v4 = EnsembleGateV4(structured_threshold=0.20, max_component_threshold=0.30)
        v4_result = gate_v4.compute(
            symcode_recall=RECALL_SYMCODE,
            hermes_v2_recall=RECALL_HERMES_V2,
            structured_recall=RECALL_STRUCTURED,
            causal_recall=RECALL_CAUSAL,
        )
        v4_gate_open: bool = v4_result.gate_open

        # ------------------------------------------------------------------
        # Step 5: Determine honest_verdict.
        # IAS improves over v3 when IAS opens the gate that v3 closed.
        # IAS matches v4 when both agree.
        # ------------------------------------------------------------------
        if ias_gate_open and not v3_gate_open:
            honest_verdict = "ias_gate_improves_v3"
        elif ias_gate_open == v4_gate_open:
            honest_verdict = "ias_gate_matches_v4"
        else:
            honest_verdict = "ias_gate_no_change"

        comparison_to_v3 = {
            "v3_gate_open": v3_gate_open,
            "v3_ensemble_recall": round(v3_ensemble, 4),
            "v3_threshold": 0.30,
            "ias_improves_v3": ias_gate_open and not v3_gate_open,
        }

        comparison_to_v4 = {
            "v4_gate_open": v4_gate_open,
            "v4_structured_threshold": gate_v4.structured_threshold,
            "v4_max_component_threshold": gate_v4.max_component_threshold,
            "ias_matches_v4": ias_gate_open == v4_gate_open,
        }

        artifact = tmpl.build_result(
            {
                "calibrated_thresholds": calibrated_thresholds,
                "input_recalls": {
                    "symcode": RECALL_SYMCODE,
                    "structured": RECALL_STRUCTURED,
                    "causal": RECALL_CAUSAL,
                    "hermes_v2": RECALL_HERMES_V2,
                },
                "ias_gate_open": ias_gate_open,
                "comparison_to_v3": comparison_to_v3,
                "comparison_to_v4": comparison_to_v4,
                "honest_verdict": honest_verdict,
                "fover_pairs_path": FOVER_PAIRS_PATH,
                "quantile_used": 0.10,
                "spec_refs": ["REQ-VERIFY-151", "REQ-VERIFY-152",
                              "SCENARIO-VERIFY-200", "SCENARIO-VERIFY-201"],
            },
            status="success",
            decision_class="verify",
        )
        _write_artifact(artifact, output_path)

    tmpl.assert_deliverable_written()


if __name__ == "__main__":
    main()
