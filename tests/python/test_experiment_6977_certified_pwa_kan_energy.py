"""Tests for calibration-only PWA-KAN residual certification.

Spec refs: REQ-KAN-6977 and SCENARIO-KAN-6977-*.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from carnot import experiment_6977_certified_pwa_kan_energy as exp


REPO = Path(__file__).resolve().parents[2]
SPEC = REPO / "openspec/capabilities/kan-verifier/spec.md"


def _raw_candidate(
    *,
    variables: int = 1,
    direction: str = "same",
    scale: str = "1",
) -> str:
    return json.dumps(
        {
            "schema_version": "carnot.constraint_ir.mapping.v1",
            "variable_map": [
                {"source": f"x{i}", "target": f"y{i}", "scale": scale, "offset": "0"}
                for i in range(variables)
            ],
            "objective_map": {"direction": direction, "scale": scale, "offset": "0"},
        },
        sort_keys=True,
    )


def _certified_row(
    ordinal: int,
    *,
    split: str = "calibration",
    success: bool = False,
    parse_success: bool = True,
    schedule: str = "direct",
) -> dict[str, object]:
    return {
        "attempt_key": f"model|pair-{ordinal}|{schedule}",
        "ordinal": ordinal,
        "hf_id": exp.MODEL_FAMILIES[ordinal % len(exp.MODEL_FAMILIES)],
        "pair_id": f"pair-{ordinal}",
        "split": split,
        "formulation_family": exp.FORMULATION_FAMILIES[ordinal % len(exp.FORMULATION_FAMILIES)],
        "schedule_id": schedule,
        "raw_sha256": f"sha256:{ordinal:064x}",
        "parse_outcome": "parsed" if parse_success else "rejected",
        "parse_success": parse_success,
        "parse_reason": None if parse_success else "malformed_json",
        "schema_outcome": "valid" if parse_success else "not_evaluated",
        "domain_correspondence_outcome": "passed" if success else "failed",
        "objective_direction_outcome": "passed" if success else "failed",
        "objective_order_outcome": "passed" if success else "failed",
        "satisfiability_outcome": "passed" if success else "failed",
        "certified_relation": "equivalent" if success else "non_equivalent",
        "exact_semantic_success": success,
        "terminal": True,
    }


def _bank_row(row: dict[str, object], raw: str) -> dict[str, object]:
    return {
        "attempt_key": row["attempt_key"],
        "candidate_raw_text": raw,
        "candidate_raw_sha256": row["raw_sha256"],
    }


def test_req_kan_6977_spec_anchors_contract_and_required_fields() -> None:
    """REQ-KAN-6977: OpenSpec owns every field and required scenario."""

    text = SPEC.read_text(encoding="utf-8")
    for scenario in (
        "PRECONDITIONS",
        "FEATURES",
        "TRAINING",
        "PWA",
        "BUDGET",
        "MILP",
        "INFEASIBILITY",
        "BARE",
    ):
        assert f"SCENARIO-KAN-6977-{scenario}" in text
    for field in exp.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in text


def test_scenario_kan_6977_features_are_frozen_and_oracle_isolated() -> None:
    """SCENARIO-KAN-6977-FEATURES: exact labels and identities stay outside inputs."""

    rows = [
        _certified_row(0, success=True),
        _certified_row(1, success=False, parse_success=False, schedule="trigger_switched"),
    ]
    bank_rows = [
        _bank_row(rows[0], _raw_candidate(variables=2, direction="same")),
        _bank_row(rows[1], "{bad"),
    ]
    schema, feature_rows = exp.freeze_features(rows, bank_rows)

    assert schema["feature_names"] == list(exp.FEATURE_NAMES)
    assert schema["dimension"] == len(exp.FEATURE_NAMES)
    assert len(feature_rows) == 2
    assert feature_rows[0]["candidate_id"].startswith("candidate-")
    assert feature_rows[0]["feature_vector"] != feature_rows[1]["feature_vector"]
    assert exp.audit_feature_isolation(schema, feature_rows)["passed"] is True
    encoded = exp.canonical_json(feature_rows).decode("ascii")
    for forbidden in exp.FORBIDDEN_FEATURE_TOKENS:
        assert forbidden not in encoded
    assert "pair-0" not in encoded
    assert "exact_semantic_success" not in encoded


def test_req_kan_6977_feature_audit_rejects_schema_or_width_drift() -> None:
    """REQ-KAN-6977: the feature audit fails closed on forbidden or malformed data."""

    rows = [_certified_row(0, success=True)]
    schema, features = exp.freeze_features(rows, [_bank_row(rows[0], _raw_candidate())])
    bad_schema = deepcopy(schema)
    bad_schema["feature_names"].append("correctness_hint")
    assert exp.audit_feature_isolation(bad_schema, features)["passed"] is False
    bad_rows = deepcopy(features)
    bad_rows[0]["feature_vector"] = []
    assert exp.audit_feature_isolation(schema, bad_rows)["passed"] is False
    bad_rows = deepcopy(features)
    bad_rows[0]["feature_vector"][0] = True
    assert exp.audit_feature_isolation(schema, bad_rows)["passed"] is False


def test_scenario_kan_6977_seeded_training_is_deterministic() -> None:
    """SCENARIO-KAN-6977-TRAINING: equal calibration inputs produce equal fits."""

    features = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]], dtype=np.float64)
    labels = np.array([0.0, 1.0, 1.0, 0.0], dtype=np.float64)
    first = exp.fit_calibration_models(features, labels, seed=77, epochs=8)
    second = exp.fit_calibration_models(features, labels, seed=77, epochs=8)

    assert first.training_rows == second.training_rows
    assert first.checkpoint_hash == second.checkpoint_hash
    assert first.kan_predictions == second.kan_predictions
    assert first.mlp_predictions == second.mlp_predictions
    assert len(first.training_rows) == 16
    assert {row["arm"] for row in first.training_rows} == {"kan", "mlp"}
    assert all(row["terminal"] for row in first.training_rows)
    assert first.model_parameter_count["kan"] > 0
    assert first.model_parameter_count["mlp"] > 0
    with pytest.raises(ValueError, match="shape"):
        exp.fit_calibration_models(features[:, 0], labels, seed=77, epochs=1)
    with pytest.raises(ValueError, match="nonconstant"):
        exp.fit_calibration_models(features, np.zeros(4), seed=77, epochs=1)


def test_scenario_kan_6977_pwa_envelopes_are_sound() -> None:
    """SCENARIO-KAN-6977-PWA: quadratic unit values stay inside local bounds."""

    unit = exp.QuadraticUnit(curvature=1.25, slope=-0.2, intercept=0.4)
    abstraction = exp.build_unit_pwa(unit, unit_index=0, lower=0.0, upper=1.0, pieces=4)
    probes = np.linspace(0.0, 1.0, 401)
    gaps = []
    for probe in probes:
        lower, upper = abstraction.bounds(float(probe))
        actual = unit(float(probe))
        assert lower - 1e-12 <= actual <= upper + 1e-12
        gaps.append(upper - lower)
    assert max(gaps) <= abstraction.local_error_bound + 1e-12
    assert abstraction.local_error_bound == pytest.approx(1.25 / (4 * 4**2))
    assert abstraction.segments[0].as_serializable()["index"] == 0
    with pytest.raises(ValueError, match="outside"):
        abstraction.bounds(1.1)
    with pytest.raises(ValueError, match="pieces"):
        exp.build_unit_pwa(unit, unit_index=0, lower=0.0, upper=1.0, pieces=0)
    with pytest.raises(ValueError, match="domain"):
        exp.build_unit_pwa(unit, unit_index=0, lower=1.0, upper=0.0, pieces=1)
    with pytest.raises(ValueError, match="curvature"):
        exp.build_unit_pwa(
            exp.QuadraticUnit(curvature=-1.0, slope=0.0, intercept=0.0),
            unit_index=0,
            lower=0.0,
            upper=1.0,
            pieces=1,
        )


def test_scenario_kan_6977_budget_uses_dp_and_network_knapsack() -> None:
    """SCENARIO-KAN-6977-BUDGET: allocation minimizes propagated error."""

    units = [
        exp.QuadraticUnit(curvature=4.0, slope=0.0, intercept=0.0),
        exp.QuadraticUnit(curvature=1.0, slope=0.0, intercept=0.0),
    ]
    plan = exp.allocate_piece_budget(units, [(0.0, 1.0)] * 2, budget=7, min_pieces=2)

    assert plan.piece_counts == (4, 3)
    assert sum(plan.piece_counts) == 7
    assert plan.unit_dp_rows
    assert plan.knapsack_rows
    assert plan.propagated_error_bound == pytest.approx(sum(plan.local_error_bounds))
    with pytest.raises(ValueError, match="budget"):
        exp.allocate_piece_budget(units, [(0.0, 1.0)] * 2, budget=3, min_pieces=2)
    with pytest.raises(ValueError, match="count"):
        exp.allocate_piece_budget([], [], budget=1, min_pieces=1)


def test_scenario_kan_6977_real_milp_is_non_tautological() -> None:
    """SCENARIO-KAN-6977-MILP: SciPy HiGHS executes a mixed-integer bound solve."""

    smoke = exp.milp_solver_smoke_test()
    assert smoke["executed"] is True
    assert smoke["status"] == "optimal"
    assert smoke["integer_variable_count"] == 1
    assert smoke["objective_value"] == pytest.approx(1.0)

    units = [exp.QuadraticUnit(curvature=1.0, slope=0.0, intercept=0.0)]
    plan = exp.allocate_piece_budget(units, [(0.0, 1.0)], budget=2, min_pieces=2)
    abstractions = [exp.build_unit_pwa(units[0], unit_index=0, lower=0.0, upper=1.0, pieces=2)]
    receipt = exp.solve_pwa_output_bounds(abstractions, output_bias=0.0)

    assert receipt["solver"] == "scipy.optimize.milp/HiGHS"
    assert receipt["status"] == "optimal"
    assert receipt["executed"] is True
    assert receipt["integer_variable_count"] == 4
    assert receipt["constraint_count"] > 0
    assert receipt["certified_upper_bound"] == pytest.approx(1.0)
    assert receipt["certified_lower_bound"] <= 0.0
    assert receipt["upper_witness"] == pytest.approx([1.0])
    assert plan.propagated_error_bound == pytest.approx(0.0625)
    with pytest.raises(ValueError, match="required"):
        exp.solve_pwa_output_bounds([], output_bias=0.0)


def test_req_kan_6977_solver_failures_do_not_certify(monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-KAN-6977: solver exceptions and nonoptimal results fail closed."""

    def raise_solver(**_kwargs: object) -> object:
        raise RuntimeError("injected")

    monkeypatch.setattr(exp, "milp", raise_solver)
    receipt = exp.milp_solver_smoke_test()
    assert receipt["executed"] is False
    assert receipt["status"] == "solver_error"

    monkeypatch.setattr(
        exp,
        "milp",
        lambda **_kwargs: SimpleNamespace(
            status=1, success=False, x=None, fun=None, message="time limit"
        ),
    )
    unit = exp.QuadraticUnit(curvature=1.0, slope=0.0, intercept=0.0)
    abstraction = exp.build_unit_pwa(unit, unit_index=0, lower=0.0, upper=1.0, pieces=2)
    with pytest.raises(RuntimeError, match="nonoptimal"):
        exp.solve_pwa_output_bounds([abstraction], output_bias=0.0)


def test_scenario_kan_6977_infeasibility_is_outside_residual() -> None:
    """SCENARIO-KAN-6977-INFEASIBILITY: no residual can reverse hard rejection."""

    infeasible = _certified_row(0, success=False)
    feasible = _certified_row(1, success=True)
    for residual in (-1e9, -1.0, 0.0, 1.0, 1e9):
        assert exp.combine_hard_feasibility(infeasible, residual, threshold=0.5) is False
    assert exp.combine_hard_feasibility(feasible, 0.1, threshold=0.5) is True
    assert exp.combine_hard_feasibility(feasible, 0.9, threshold=0.5) is False
    certificate = exp.certify_infeasibility_preservation([infeasible, feasible])
    assert certificate["proved"] is True
    assert certificate["hard_infeasible_count"] == 1


def test_scenario_kan_6977_actual_constant_labels_emit_blocked_contract(tmp_path: Path) -> None:
    """SCENARIO-KAN-6977-PRECONDITIONS: Exp6976's constant labels block fitting."""

    output = tmp_path / "experiment_6977.json"
    artifact = exp.run(date="20260904", repo_root=REPO, output_path=output)

    assert json.loads(output.read_text(encoding="utf-8")) == artifact
    exp.validate_artifact(artifact)
    assert artifact["verdict_class"] == "blocked"
    assert artifact["honest_verdict"] == "blocked_certified_pwa_kan_energy"
    assert artifact["certified_pwa_energy_ready_score"] == 0
    assert artifact["pwa_energy_heldout_positive_score"] == 0
    assert artifact["training_rows"] == []
    assert artifact["checkpoint_hash"] is None
    assert artifact["feature_rows"]
    failed = artifact["gate_check_summary"]["failed_checks"]
    label_gate = next(
        row for row in failed if row["failed_check"] == "calibration_labels_nonconstant"
    )
    assert label_gate["expected_value"] is True
    assert label_gate["observed_value"] is False
    assert all("exact_semantic_success" not in json.dumps(row) for row in artifact["feature_rows"])
    assert all(field in artifact["field_principles"] for field in exp.REQUIRED_ARTIFACT_FIELDS)


def test_scenario_kan_6977_bare_scores_and_checksum_fail_closed(tmp_path: Path) -> None:
    """SCENARIO-KAN-6977-BARE: wrapped scores and changed payloads are rejected."""

    artifact = exp.build_from_paths(REPO, date="20260904")
    exp.validate_artifact(artifact)

    wrapped = deepcopy(artifact)
    wrapped["certified_pwa_energy_ready_score"] = {"value": 0}
    with pytest.raises(ValueError, match="bare_binary_score"):
        exp.validate_artifact(wrapped)

    changed = deepcopy(artifact)
    changed["random_seed"] += 1
    with pytest.raises(ValueError, match="checksum"):
        exp.validate_artifact(changed)

    wrong_class = deepcopy(artifact)
    wrong_class["verdict_class"] = "positive"
    wrong_class["reproducibility_checksum"] = exp.payload_checksum(wrong_class)
    with pytest.raises(ValueError, match="blocked_verdict_class"):
        exp.validate_artifact(wrong_class)


def test_req_kan_6977_input_and_validation_failures_are_explicit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-KAN-6977: malformed inputs and every blocked-schema drift fail closed."""

    assert exp._read_object(tmp_path / "missing.json") is None
    malformed = tmp_path / "bad.json"
    malformed.write_text("{bad", encoding="utf-8")
    assert exp._read_object(malformed) is None
    scalar = tmp_path / "scalar.json"
    scalar.write_text("[]", encoding="utf-8")
    assert exp._read_object(scalar) is None

    artifact = exp.build_from_paths(REPO, date="20260904")
    cases = []
    missing = deepcopy(artifact)
    del missing["rows"]
    cases.append((missing, "missing_required_fields"))
    principle = deepcopy(artifact)
    del principle["field_principles"]["rows"]
    principle["reproducibility_checksum"] = exp.payload_checksum(principle)
    cases.append((principle, "missing_field_principles"))
    substrate = deepcopy(artifact)
    substrate["inference_substrate"] = "wrong"
    substrate["reproducibility_checksum"] = exp.payload_checksum(substrate)
    cases.append((substrate, "inference_substrate"))
    oracle = deepcopy(artifact)
    oracle["verifier_is_oracle"] = True
    oracle["reproducibility_checksum"] = exp.payload_checksum(oracle)
    cases.append((oracle, "verifier_is_oracle"))
    verdict = deepcopy(artifact)
    verdict["verdict_class"] = "unknown"
    verdict["reproducibility_checksum"] = exp.payload_checksum(verdict)
    cases.append((verdict, "invalid_verdict_class"))
    prefix = deepcopy(artifact)
    prefix["honest_verdict"] = "complete_wrong"
    prefix["reproducibility_checksum"] = exp.payload_checksum(prefix)
    cases.append((prefix, "blocked_verdict_prefix"))
    readiness = deepcopy(artifact)
    readiness["certified_pwa_energy_ready_score"] = 1
    readiness["reproducibility_checksum"] = exp.payload_checksum(readiness)
    cases.append((readiness, "blocked_readiness"))
    trained = deepcopy(artifact)
    trained["training_rows"] = [{"epoch": 1}]
    trained["reproducibility_checksum"] = exp.payload_checksum(trained)
    cases.append((trained, "must_not_train"))
    gate_shape = deepcopy(artifact)
    gate_shape["gate_check_summary"]["failed_checks"][0]["extra"] = True
    gate_shape["reproducibility_checksum"] = exp.payload_checksum(gate_shape)
    cases.append((gate_shape, "failed_gate_shape"))
    for changed, message in cases:
        with pytest.raises(ValueError, match=message):
            exp.validate_artifact(changed)

    monkeypatch.setattr(
        exp,
        "check_preconditions",
        lambda *_args: ([exp.gate_check("injected", True, True)], [], {"status": "optimal"}),
    )
    with pytest.raises(RuntimeError, match="unexpected_all_preconditions"):
        exp.build_from_paths(tmp_path)


def test_req_kan_6977_cli_writes_to_requested_temporary_path(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """REQ-KAN-6977: the dated command surface writes and reports one blocked artifact."""

    output = tmp_path / "cli.json"
    assert exp.main(["--date", "20260904", "--repo-root", str(REPO), "--output", str(output)]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["output"] == str(output)
    assert report["honest_verdict"] == "blocked_certified_pwa_kan_energy"
