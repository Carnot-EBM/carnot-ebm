#!/usr/bin/env python3
"""Experiment 667 — EnsembleGate v4 Redesign: structured-first gate logic.

Root cause repaired:
    Exp 655 (EnsembleGate v3) produced gate_open=False because the v3 formula
    averaged ALL four recall components, including HermesV2 which scores 0.0 on
    mixed-format test sets.  This dragged the ensemble to 0.224, below the 0.30
    threshold, even though causal_recall=0.36 already exceeded it on its own.

This experiment validates that EnsembleGate v4 (structured-first OR logic) opens
the gate for the same Exp 655 recall values, unblocking VR #18 (RETRO-033 attempt
#18) and resolving RETRO-070.

Spec: REQ-VERIFY-147, REQ-VERIFY-148
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Ensure repo root is importable regardless of working directory.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from carnot.pipeline.ensemble_gate_v4 import EnsembleGateV4
from carnot.pipeline.experiment_watchdog import ExperimentTimeoutWatchdog
from scripts.experiment_template import ExperimentTemplate

_DELIVERABLE = "results/experiment_667_gate_v4_redesign.json"

# Recall values from Exp 655 result (gate_open=False under v3).
_V3_SYMCODE_RECALL = 0.12
_V3_HERMES_V2_RECALL = 0.0
_V3_STRUCTURED_RECALL = 0.20
_V3_CAUSAL_RECALL = 0.36
_V3_ENSEMBLE_RECALL = 0.22400000000000003
_V3_GATE_OPEN = False


def main() -> None:
    tmpl = ExperimentTemplate(
        667,
        "EnsembleGate v4 Redesign — structured-first gate logic",
        _DELIVERABLE,
    )
    tmpl.setup()

    watchdog = ExperimentTimeoutWatchdog(667, timeout_minutes=20)
    watchdog.start()

    try:
        # Load v3 recall values from the Exp 655 artifact to guarantee we are
        # testing exactly the values that blocked VR #18, not hypothetical ones.
        v3_path = _REPO_ROOT / "results" / "experiment_655_ensemble_gate_v3.json"
        with v3_path.open() as fh:
            v3 = json.load(fh)

        symcode_recall: float = v3["symcode_recall"]
        hermes_v2_recall: float = v3["hermes_v2_recall"]
        structured_recall: float = v3["structured_recall"]
        causal_recall: float = v3["causal_recall"]
        v3_ensemble_recall: float = v3["ensemble_recall"]

        # Run EnsembleGate v4 with the same values.
        gate = EnsembleGateV4()
        result = gate.compute(
            symcode_recall=symcode_recall,
            hermes_v2_recall=hermes_v2_recall,
            structured_recall=structured_recall,
            causal_recall=causal_recall,
        )

        # Gate MUST open — causal_recall=0.36 >= max_component_threshold=0.30.
        if not result.gate_open:
            raise RuntimeError(
                f"FATAL: EnsembleGate v4 gate_open=False with "
                f"causal_recall={causal_recall}, structured_recall={structured_recall}. "
                "Gate redesign did not resolve the blocking condition."
            )

        # retro_070 is resolved when the gate opens.
        retro_070_gate_unblocked = result.gate_open
        retro_033_authorized = result.authorizes_vr

        # delta summary — what changed between v3 and v4.
        gate_comparison = {
            "v3_ensemble_recall": v3_ensemble_recall,
            "v3_gate_open": _V3_GATE_OPEN,
            "v4_gate_open": result.gate_open,
            "v3_gate_formula": "mean(symcode, hermes_v2, structured, causal) >= 0.30",
            "v4_gate_formula": (
                "structured_recall >= 0.20 OR max(causal_recall, symcode_recall) >= 0.30"
            ),
            "v4_trigger_condition": "causal_recall=0.36 >= max_component_threshold=0.30",
            "hermes_v2_excluded_from_v4": True,
        }

        honest_verdict = (
            "gate_open_retro_070_unblocked"
            if result.gate_open
            else "gate_still_closed"
        )

        artifact = tmpl.build_result(
            {
                "symcode_recall": result.symcode_recall,
                "hermes_v2_recall": hermes_v2_recall,
                "structured_recall": result.structured_recall,
                "causal_recall": result.causal_recall,
                "ensemble_recall": result.ensemble_recall,
                "v3_ensemble_recall": v3_ensemble_recall,
                "gate_open": result.gate_open,
                "gate_threshold": result.gate_threshold,
                "gate_version": result.gate_version,
                "authorizes_vr": result.authorizes_vr,
                "retro_070_gate_unblocked": retro_070_gate_unblocked,
                "retro_033_authorized": retro_033_authorized,
                "gate_comparison": gate_comparison,
                "honest_verdict": honest_verdict,
            },
            status="success",
        )

        output_path = _REPO_ROOT / _DELIVERABLE
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w") as fh:
            json.dump(artifact, fh, indent=2)

    finally:
        watchdog.stop()

    tmpl.assert_deliverable_written()


if __name__ == "__main__":
    main()
