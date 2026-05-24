"""Tests for Exp 3006 fixed-point energy diagnostic.

Spec refs: REQ-VERIFY-3006, SCENARIO-VERIFY-3006.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import carnot.eval.eqr_fixed_point_energy_diagnostic_v1 as exp
from carnot.eval import solver_to_validator_tree_expansion_v1 as exp3005


REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO_ROOT / "openspec" / "capabilities" / "verification" / "spec.md"
SCRIPT_PATH = REPO_ROOT / "scripts" / "experiment_3006_eqr_fixed_point_energy_diagnostic_v1.py"


def _seed_exp3005_cache(tmp_path: Path) -> Path:
    config = exp3005.ExperimentConfig(
        repo_root=tmp_path,
        output_path=tmp_path / "results" / exp3005.OUTPUT_FILENAME,
        manifest_path=tmp_path / exp3005.VALIDATOR_MANIFEST_REL_PATH,
        z3_transcript_dir=tmp_path / exp3005.Z3_TRANSCRIPT_REL_DIR,
        runtime_transcript_dir=tmp_path / exp3005.RUNTIME_TRANSCRIPT_REL_DIR,
        started_at=20.0,
        clock=lambda: 20.25,
    )
    exp3005.run_experiment(config)
    return config.resolved_manifest_path()


def _config(tmp_path: Path, manifest_path: Path) -> exp.ExperimentConfig:
    return exp.ExperimentConfig(
        repo_root=tmp_path,
        output_path=tmp_path / "results" / exp.OUTPUT_FILENAME,
        table_path=tmp_path / exp.DIAGNOSTIC_TABLE_REL_PATH,
        manifest_path=manifest_path,
        started_at=30.0,
        clock=lambda: 30.5,
    )


def test_req_verify_3006_spec_and_script_anchor_exists() -> None:
    """REQ-VERIFY-3006: the diagnostic is OpenSpec anchored and scriptable."""
    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-VERIFY-3006" in spec
    assert "SCENARIO-VERIFY-3006" in spec
    assert exp.OUTPUT_FILENAME in spec
    assert "native_eqr_claim_made=false" in spec
    assert "negative_control_rejection_rate" in spec
    assert SCRIPT_PATH.exists()


def test_scenario_verify_3006_scores_cached_trajectory_energy_descent(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-3006: invalid feedback, valid partial, full candidate descend."""
    manifest_path = _seed_exp3005_cache(tmp_path)
    rows = exp.load_manifest(manifest_path)
    trajectories = exp.load_cached_trajectories(manifest_path)

    assert len(trajectories) >= exp.MIN_TRAJECTORIES
    assert len(trajectories) == len(rows)
    for trajectory in trajectories:
        energies = trajectory["energy_sequence"]
        state_names = [state["state"] for state in trajectory["states"]]

        assert state_names == [
            "invalid_partial_feedback",
            "valid_extendable_partial",
            "full_exact_candidate",
        ]
        assert energies[0] > energies[1] > energies[2]
        assert energies[-1] == 0.0
        assert trajectory["converged_to_fixed_point"] is True
        assert trajectory["energy_monotonic"] is True
        assert trajectory["native_eqr_claim_made"] is False


def test_req_verify_3006_negative_controls_and_basin_sensitivity(tmp_path: Path) -> None:
    """REQ-VERIFY-3006: perturbations are measured and invalid controls reject."""
    manifest_path = _seed_exp3005_cache(tmp_path)
    rows = exp.load_manifest(manifest_path)
    trajectory = exp.build_trajectory(rows[0])
    controls = exp.build_negative_controls(rows[0], rows)
    basin = exp.measure_basin_sensitivity(rows[:4])

    assert trajectory["states"][0]["rejection_reasons"]
    assert {control["control"] for control in controls} == {
        "permuted_partial_constraints",
        "swapped_incompatible_validator",
        "contradiction_node_injection",
    }
    assert all(control["diagnostic_rejected"] for control in controls)
    assert all(control["energy"] > trajectory["states"][-1]["energy"] for control in controls)
    assert basin["trajectory_count"] == 4
    assert basin["perturbation_count"] == 12
    assert 0.0 <= basin["accepted_perturbation_rate"] <= 1.0
    assert basin["sensitive_trajectory_rate"] == 1.0
    assert basin["mean_energy_delta"] > 0.0


def test_scenario_verify_3006_runner_writes_artifact_and_table(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-3006: run writes authority JSON and inspectable table."""
    manifest_path = _seed_exp3005_cache(tmp_path)
    artifact = exp.run_diagnostic(_config(tmp_path, manifest_path))
    saved = json.loads((tmp_path / "results" / exp.OUTPUT_FILENAME).read_text(encoding="utf-8"))
    table_path = tmp_path / artifact["diagnostic_table_path"]
    table_rows = exp.load_diagnostic_table(table_path)

    assert saved == artifact
    assert set(exp.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert artifact["fixed_point_diagnostic_ready"] is True
    assert artifact["n_trajectories"] >= exp.MIN_TRAJECTORIES
    assert artifact["convergence_rate"] == 1.0
    assert artifact["energy_monotonicity_rate"] == 1.0
    assert artifact["negative_control_rejection_rate"] == 1.0
    assert artifact["diagnostic_table_path"] == str(exp.DIAGNOSTIC_TABLE_REL_PATH)
    assert artifact["native_eqr_claim_made"] is False
    assert artifact["honest_verdict"].startswith("ready:")
    assert len(table_rows) == artifact["n_trajectories"]
    assert table_rows[0]["energy_sequence"][-1] == 0.0

    exp.validate_artifact(artifact)


def test_req_verify_3006_validation_rejects_claim_or_schema_drift(tmp_path: Path) -> None:
    """REQ-VERIFY-3006: validation blocks native EqR claims and missing evidence."""
    manifest_path = _seed_exp3005_cache(tmp_path)
    artifact = exp.run_diagnostic(_config(tmp_path, manifest_path))
    exp.validate_artifact(artifact)

    with pytest.raises(ValueError, match="missing required fields"):
        exp.validate_artifact({"honest_verdict": "ready: incomplete"})
    with pytest.raises(ValueError, match="native_eqr_claim_made"):
        exp.validate_artifact(artifact | {"native_eqr_claim_made": True})
    with pytest.raises(ValueError, match="n_trajectories"):
        exp.validate_artifact(artifact | {"n_trajectories": 0})
    with pytest.raises(ValueError, match="energy_definition"):
        exp.validate_artifact(artifact | {"energy_definition": ""})
    with pytest.raises(ValueError, match="convergence_rate"):
        exp.validate_artifact(artifact | {"convergence_rate": 1.25})
    with pytest.raises(ValueError, match="basin_sensitivity_summary"):
        exp.validate_artifact(artifact | {"basin_sensitivity_summary": []})
    with pytest.raises(ValueError, match="diagnostic_table_path"):
        exp.validate_artifact(artifact | {"diagnostic_table_path": ""})
    with pytest.raises(ValueError, match="honest_verdict"):
        exp.validate_artifact(artifact | {"honest_verdict": "complete: wrong prefix"})
