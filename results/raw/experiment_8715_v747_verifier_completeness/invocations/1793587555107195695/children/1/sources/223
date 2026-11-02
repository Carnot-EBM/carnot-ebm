"""Exp 1599: KANELÉ hardware LUT-complexity accounting without synthesis.

This module performs a no-synthesis hardware accounting pass for KANs.
It estimates RM, BOP, and NABS and writes the audit artifact to
results/experiment_1599_kanele_audit.json, making no claims of hardware
synthesis or execution.

Spec refs: REQ-KAN-1599, SCENARIO-KAN-1599.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DELIVERABLE_PATH = PROJECT_ROOT / "results" / "experiment_1599_kanele_audit.json"


def compute_kan_metrics(n_inputs: int, k_splines: int, q8_operand_width: int = 8) -> dict[str, int]:
    """Estimate RM, BOP, and NABS for a KAN layer without synthesis."""
    basis_evaluations = n_inputs * k_splines
    rm = int(basis_evaluations)
    bop = int(rm * q8_operand_width)
    index_add_shift = 2 * n_inputs
    interpolation_add_sub = 2 * basis_evaluations
    accumulation_adds = n_inputs * (k_splines - 1)
    nabs = int(index_add_shift + interpolation_add_sub + accumulation_adds)
    return {
        "rm_per_inference": rm,
        "bop_per_inference": bop,
        "nabs_per_inference": nabs,
    }


def run_kanele_audit(deliverable_path: Path = DELIVERABLE_PATH) -> dict[str, Any]:
    """Generate the Exp 1599 hardware accounting artifact."""
    metrics = compute_kan_metrics(n_inputs=32, k_splines=16)

    artifact = {
        "experiment": 1599,
        "schema": "kanele_audit_v1",
        "run_date": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "status": "complete",
        "hardware_execution_confirmed": False,
        "synthesis_performed": False,
        "board_executed": False,
        "rm_per_inference": metrics["rm_per_inference"],
        "bop_per_inference": metrics["bop_per_inference"],
        "nabs_per_inference": metrics["nabs_per_inference"],
        "honest_verdict": "complete_no_synthesis_kan_accounting",
        "notes": [
            "Hardware accounting for KANs without synthesis.",
            "Estimates RM, BOP, and NABS for KANs.",
        ]
    }

    deliverable_path.parent.mkdir(parents=True, exist_ok=True)
    deliverable_path.write_text(json.dumps(artifact, indent=2) + "\n")
    return artifact


if __name__ == "__main__":  # pragma: no cover
    run_kanele_audit()
