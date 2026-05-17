"""Tests for Exp 2246 CASAL vs AdamFLIP constraint-violation benchmark.

Spec coverage: REQ-SAMPLE-2246, REQ-SAMPLE-2246-1, REQ-SAMPLE-2246-2,
REQ-SAMPLE-2246-5, REQ-SAMPLE-2246-6, SCENARIO-SAMPLE-2246.
"""

from __future__ import annotations

import json
from pathlib import Path

from carnot.reporting import casal_vs_adamflip as mod


def test_req_sample_2246_import_preconditions_pass() -> None:
    """REQ-SAMPLE-2246-1: CASAL and AdamFLIP import preconditions are explicit."""

    preconditions = mod.check_preconditions()

    assert [item["status"] for item in preconditions] == ["passed", "passed"]
    assert preconditions[0]["module"] == "carnot.samplers.casal"
    assert preconditions[1]["module"] == "carnot.training.adamflip"


def test_req_sample_2246_benchmark_has_3d_ebm_and_two_constraints() -> None:
    """REQ-SAMPLE-2246-2: benchmark shape is 3D with two equality constraints."""

    benchmark = mod.build_benchmark()

    assert benchmark.model.variables == 3
    assert benchmark.constraint_matrix.shape == (2, 3)
    assert benchmark.constraint_target.shape == (2,)


def test_scenario_sample_2246_writes_terminal_artifact(tmp_path: Path) -> None:
    """SCENARIO-SAMPLE-2246: 100-sample comparison writes the required artifact."""

    output = tmp_path / mod.OUTPUT_FILE

    artifact = mod.run_experiment(output_path=output)

    assert json.loads(output.read_text(encoding="utf-8")) == artifact
    mod.validate_artifact(artifact)
    assert artifact["n_samples"] >= 100
    assert artifact["random_seed"] == mod.RANDOM_SEED
    assert artifact["casal_violation_mean"] <= artifact["adamflip_violation_mean"] / 2.0
    assert artifact["casal_validated"] is True
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["regimes"]["A_adamflip_soft_penalty_mcmc"]["n_samples"] == 100
    assert artifact["regimes"]["B_casal_primal_dual"]["n_samples"] == 100


def test_req_sample_2246_validation_boolean_matches_formula(tmp_path: Path) -> None:
    """REQ-SAMPLE-2246-6: casal_validated is exactly the half-violation gate."""

    artifact = mod.run_experiment(output_path=tmp_path / mod.OUTPUT_FILE)

    expected = artifact["casal_violation_mean"] <= artifact["adamflip_violation_mean"] / 2.0
    assert artifact["casal_validated"] is expected
    assert set(mod.REQUIRED_ARTIFACT_FIELDS).issubset(artifact)


def test_req_sample_2246_blocked_artifact_when_casal_import_missing(
    tmp_path: Path,
) -> None:
    """REQ-SAMPLE-2246-1: missing CASAL import writes the requested blocker."""

    output = tmp_path / mod.OUTPUT_FILE

    artifact = mod.run_experiment(
        output_path=output,
        casal_module="carnot.samplers.missing_casal_for_test",
    )

    assert json.loads(output.read_text(encoding="utf-8")) == artifact
    mod.validate_artifact(artifact)
    assert artifact["status"] == "blocked"
    assert artifact["honest_verdict"] == "blocked_casal_missing"
    assert artifact["casal_validated"] is False
    assert artifact["n_samples"] == 0
