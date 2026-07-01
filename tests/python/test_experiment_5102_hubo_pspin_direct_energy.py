"""Tests for Exp 5102 direct HUBO/p-spin energy versus QUBO gadgets.

Spec refs: REQ-VERIFY-5102, SCENARIO-VERIFY-5102.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot import experiment_5102_hubo_pspin_direct_energy as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/verification/spec.md"
ARTIFACT_PATH = REPO / mod.RESULT_RELATIVE_PATH


def test_req_verify_5102_spec_declares_hubo_qubo_contract() -> None:
    """REQ-VERIFY-5102: OpenSpec anchors the experiment and required fields."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    for marker in (
        "REQ-VERIFY-5102",
        "SCENARIO-VERIFY-5102",
        "python/carnot/experiment_5102_hubo_pspin_direct_energy.py",
        "results/experiment_5102_hubo_pspin_direct_energy_v468.json",
        mod.INFERENCE_SUBSTRATE,
        mod.SUCCESS_VERDICT,
        mod.NO_ADVANTAGE_VERDICT,
    ):
        assert marker in spec
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert field in spec


def test_scenario_verify_5102_family_contains_tiny_high_order_csp_instances() -> None:
    """SCENARIO-VERIFY-5102: the deterministic family has high-order parity clauses."""

    family = mod.build_instance_family()
    optima = [mod.enumerate_hubo(mod.build_hubo_encoding(instance)).optimum_energy for instance in family]

    assert len(family) >= 3
    assert {instance.family for instance in family} == {mod.INSTANCE_FAMILY}
    assert any(optimum > 0 for optimum in optima)
    for instance in family:
        assert instance.n_vars <= 5
        assert all(clause.parity in (0, 1) for clause in instance.clauses)
        assert max(len(clause.variables) for clause in instance.clauses) >= 3


def test_req_verify_5102_qubo_gadgets_match_hubo_optima_and_projections() -> None:
    """REQ-VERIFY-5102: exact enumeration proves optimum and assignment equivalence."""

    rows = [mod.compare_instance(instance) for instance in mod.build_instance_family()]

    assert all(row["exact_optima_verified"] for row in rows)
    assert all(row["direct_optimum_energy"] == row["qubo_optimum_energy"] for row in rows)
    assert all(row["direct_optimal_assignments"] == row["projected_qubo_optimal_assignments"] for row in rows)
    assert all(row["qubo_variable_count"] > row["hubo_variable_count"] for row in rows)
    assert all(row["auxiliary_variable_count"] > 0 for row in rows)
    assert all(row["energy_scale_distortion"] > 1.0 for row in rows)


def test_scenario_verify_5102_projection_energy_matches_for_every_assignment() -> None:
    """SCENARIO-VERIFY-5102: best QUBO extension equals direct energy for each projection."""

    for instance in mod.build_instance_family():
        hubo = mod.build_hubo_encoding(instance)
        qubo = mod.build_qubo_gadget_encoding(instance)
        hubo_by_projection = mod.energy_by_projection(hubo)
        qubo_by_projection = mod.best_qubo_energy_by_projection(qubo)

        assert qubo.auxiliary_definitions
        assert qubo_by_projection == hubo_by_projection


def test_req_verify_5102_artifact_fields_principles_and_metrics(tmp_path: Path) -> None:
    """REQ-VERIFY-5102: artifact emits required metrics and principle annotations."""

    artifact = mod.write_artifact(root=tmp_path)
    payload = json.loads((tmp_path / mod.RESULT_RELATIVE_PATH).read_text(encoding="utf-8"))

    assert payload == artifact
    mod.validate_artifact(artifact)
    assert set(mod.REQUIRED_ARTIFACT_FIELDS).issubset(artifact)
    assert set(mod.REQUIRED_ARTIFACT_FIELDS).issubset(artifact["field_principles"])
    assert artifact["honest_verdict"].startswith(mod.SUCCESS_VERDICT)
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["exact_optima_verified"] is True
    assert artifact["direct_hubo_advantage"] is True
    assert artifact["flagged_adversarial"] is False
    assert artifact["auxiliary_variable_blowup"]["mean_auxiliary_variables"] > 0
    assert artifact["energy_scale_distortion"]["max_qubo_to_hubo_coefficient_ratio"] > 1.0
    assert "native high-order" in artifact["hardware_mapping_notes"]
    assert "QUBO" in artifact["hardware_mapping_notes"]
    assert len(artifact["reproducibility_checksum"]) == 64


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("honest_verdict", "optimistic", "honest_verdict"),
        ("duration_s", -1.0, "duration_s"),
        ("inference_substrate", "live_llm_inference", "inference_substrate"),
        ("instance_family", "other", "instance_family"),
        ("exact_optima_verified", False, "exact_optima_verified"),
        ("direct_hubo_advantage", False, "direct_hubo_advantage"),
        ("flagged_adversarial", True, "flagged_adversarial"),
    ],
)
def test_req_verify_5102_validate_artifact_rejects_schema_violations(
    field: str,
    value: object,
    message: str,
) -> None:
    """REQ-VERIFY-5102: malformed or non-clean terminal artifacts fail closed."""

    artifact = mod.run()
    artifact[field] = value

    with pytest.raises(ValueError, match=message):
        mod.validate_artifact(artifact)


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (lambda artifact: artifact.pop("hubo_variable_counts"), "missing required fields"),
        (lambda artifact: artifact.update({"field_principles": {}}), "field_principles"),
        (
            lambda artifact: artifact.update({"energy_scale_distortion": {"max_qubo_to_hubo_coefficient_ratio": 0.5}}),
            "energy_scale_distortion",
        ),
        (
            lambda artifact: artifact.update({"coupling_density_qubo": {"mean": -0.1}}),
            "coupling_density_qubo",
        ),
    ],
)
def test_req_verify_5102_validate_artifact_rejects_consistency_violations(
    mutator: object,
    message: str,
) -> None:
    """REQ-VERIFY-5102: coherent-looking but inconsistent artifacts fail closed."""

    artifact = mod.run()
    mutator(artifact)

    with pytest.raises(ValueError, match=message):
        mod.validate_artifact(artifact)


def test_req_verify_5102_main_writes_default_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-VERIFY-5102: CLI entrypoint writes the configured result path."""

    monkeypatch.setenv("CARNOT_EXP5102_ROOT", str(tmp_path))

    assert mod.main() == 0
    payload = json.loads((tmp_path / mod.RESULT_RELATIVE_PATH).read_text(encoding="utf-8"))
    mod.validate_artifact(payload)
    assert payload["result_path"] == mod.RESULT_RELATIVE_PATH


def test_deliverable_file_validates_for_req_verify_5102() -> None:
    """SCENARIO-VERIFY-5102: checked-in deliverable satisfies the terminal schema."""

    artifact = json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))

    mod.validate_artifact(artifact)
    assert artifact["honest_verdict"].startswith(mod.SUCCESS_VERDICT)
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
