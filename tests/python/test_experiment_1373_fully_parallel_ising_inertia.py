"""Tests for Exp 1373 fully parallel Ising inertia CPU validation.

Spec: REQ-SAMPLE-023, REQ-SAMPLE-024
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scripts import experiment_1373_fully_parallel_ising_inertia_cpu_validation as exp


REQUIRED_FIELDS = {
    "status",
    "constraint_problems_tested",
    "inertia_alpha_sweep",
    "best_inertia_alpha",
    "inertia_convergence_speedup",
    "parallel_update_stability",
    "steps_to_convergence_baseline",
    "steps_to_convergence_inertia",
    "fpga_mapping_estimate",
    "hardware_claim_allowed",
    "kv260_claim_allowed",
    "honest_verdict",
}


def test_problem_builder_creates_dense_fover_constraint_graph() -> None:
    """REQ-SAMPLE-023: FoVer rows become dense Ising constraint problems."""
    row = {
        "question_id": "unit",
        "step_text": "If x = 3 and y = 4, then x + y = 7.",
        "label": "correct",
    }

    problem = exp._problem_from_fover_row(row, row_index=0, n_spins=8)

    assert problem.name == "fover_unit_row0"
    assert problem.biases.shape == (8,)
    assert problem.coupling_matrix.shape == (8, 8)
    assert np.allclose(problem.coupling_matrix, problem.coupling_matrix.T)
    assert np.count_nonzero(np.triu(problem.coupling_matrix, k=1)) > 20


def test_run_experiment_writes_required_cpu_only_artifact(tmp_path: Path) -> None:
    """REQ-SAMPLE-024: Exp 1373 artifact has required fields and no HW claim."""
    output_path = tmp_path / "experiment_1373.json"

    artifact = exp.run_experiment(
        output_path=output_path,
        n_problems=3,
        n_spins=8,
        max_sweeps=8,
        seeds=(0,),
        alphas=(0.0, 0.25),
    )

    assert REQUIRED_FIELDS <= artifact.keys()
    assert artifact["status"] == "complete"
    assert artifact["hardware_claim_allowed"] is False
    assert artifact["kv260_claim_allowed"] is False
    assert len(artifact["constraint_problems_tested"]) == 3

    persisted = json.loads(output_path.read_text())
    assert persisted["best_inertia_alpha"] in {0.0, 0.25}
    assert persisted["fpga_mapping_estimate"]["kv260_v4_sparse_lut_estimate"] == 35872
