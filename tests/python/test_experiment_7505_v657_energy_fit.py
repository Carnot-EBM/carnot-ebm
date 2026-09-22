"""Tests for REQ-VERIFY-7505 and SCENARIO-VERIFY-7505-*.

The fixture exercises training and calibration only. It does not make a
held-out efficacy claim.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest

from carnot import experiment_7505_v657_energy_fit as exp


def _rows(role: str, count: int, offset: int = 0) -> list[exp.JsonDict]:
    """Build deterministic fit-visible rows with both binary classes."""

    rows = []
    for index in range(count):
        value = index + offset
        unsupported = value % 2
        center = (value - count / 2) / max(count, 1)
        features = [
            center + unsupported * 0.4,
            0.2 + 0.5 * unsupported,
            0.3 + 0.4 * unsupported,
            0.4 + 0.4 * unsupported,
            0.01 + 0.02 * (value % 3),
            0.05 * (value % 4),
            0.03 * (value % 5),
            1.0 + 0.1 * (value % 3),
            5.0 + 0.05 * value,
            6.0 + 0.02 * value,
        ]
        rows.append(
            {
                "group_id": f"{role}-{index}",
                "role": role,
                "source_hash": f"sha256:{role}-{index}",
                "features": features,
                "raw_whole_expectation": 0.2 + 0.6 * unsupported,
                "raw_max_window_probability": 0.3 + 0.5 * unsupported,
                "label": unsupported,
            }
        )
    return rows


def test_binary_energy_is_finite_and_exactly_normalized() -> None:
    """SCENARIO-VERIFY-7505-ENERGY keeps the two-state equation exact."""

    logits = np.asarray([-1000.0, -1.0, 0.0, 2.0, 1000.0])
    energies, probabilities = exp.binary_energy(logits)
    assert np.array_equal(energies[:, 0], np.zeros(5))
    assert np.array_equal(energies[:, 1], -logits)
    assert np.all(np.isfinite(probabilities))
    assert np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-15)
    assert np.allclose(probabilities[:, 1], exp.stable_sigmoid(logits))
    with pytest.raises(ValueError, match="logits_nonfinite"):
        exp.binary_energy([float("nan")])


def test_local_spline_design_reuses_bounded_basis() -> None:
    """REQ-VERIFY-7505 caps the reusable ten-feature spline head."""

    training = np.asarray([row["features"] for row in _rows("training", 12)])
    knots = exp.fit_spline_knots(training)
    design = exp.spline_design_matrix(training, knots)
    assert knots.shape == (10, exp.KNOT_VECTOR_SIZE)
    assert design.shape == (12, 10 * exp.COEFFICIENTS_PER_INPUT)
    assert design.shape[1] + 1 <= 256
    assert np.allclose(design.reshape(12, 10, -1).sum(axis=2), 1.0)
    with pytest.raises(ValueError, match="feature_matrix_shape_invalid"):
        exp.fit_spline_knots(np.ones((4, 9)))
    with pytest.raises(ValueError, match="knots_shape_invalid"):
        exp.spline_design_matrix(training, np.ones((10, 2)))


def test_fit_bundle_records_equal_candidates_and_controls() -> None:
    """SCENARIO-VERIFY-7505-BUDGET and CONTROLS cover bounded fitting."""

    training = _rows("training", 40)
    calibration = _rows("calibration_tuning", 20, 40)
    bundle = exp.fit_energy_bundle(training, calibration, steps=20)
    assert bundle["roles_consumed"] == ["training", "calibration_tuning"]
    assert bundle["heldout_labels_consumed"] is False
    assert bundle["seeds"] == list(exp.TRAINING_SEEDS)
    assert len(bundle["candidate_rows"]) == 45
    assert not [row for row in bundle["candidate_rows"] if row["status"] != "complete"]
    for arm in exp.TRAINABLE_ARMS:
        candidates = [row for row in bundle["candidate_rows"] if row["arm"] == arm]
        assert {row["regularization"] for row in candidates} == set(exp.REGULARIZATION_GRID)
        assert {row["seed"] for row in candidates} == set(exp.TRAINING_SEEDS)
        assert all(row["optimizer_steps"] == 20 for row in candidates)
        assert all(row["parameter_count"] <= 256 for row in candidates)
        assert bundle["selected_heads"][arm]["selection_metric"] == "calibration_brier"
    assert bundle["controls"]["constant_feature"]["passed"] is True
    assert bundle["controls"]["label_shuffle"]["passed"] is True
    assert bundle["normalization_checks"]["passed"] is True
    assert bundle["temperature_baseline"]["selection_metric"] == "calibration_brier"
    assert bundle["simple_baseline"]["arm"] in {
        "raw_whole_expectation",
        "raw_max_window_probability",
    }
    assert bundle["bundle_sha256"] == exp.bundle_hash(bundle)


def test_fit_rejects_roles_shapes_and_excess_steps() -> None:
    """REQ-VERIFY-7505 fails closed before an invalid optimizer starts."""

    training = _rows("training", 8)
    calibration = _rows("calibration_tuning", 6, 8)
    bad_role = deepcopy(training)
    bad_role[0]["role"] = "test"
    with pytest.raises(ValueError, match="training_role_invalid"):
        exp.fit_energy_bundle(bad_role, calibration, steps=2)
    bad_label = deepcopy(calibration)
    bad_label[0]["label"] = None
    with pytest.raises(ValueError, match="binary_label_invalid"):
        exp.fit_energy_bundle(training, bad_label, steps=2)
    with pytest.raises(ValueError, match="optimizer_steps_invalid"):
        exp.fit_energy_bundle(training, calibration, steps=201)


def test_expected_action_uses_registered_tie_order() -> None:
    """SCENARIO-VERIFY-7505-POLICY freezes direct expected-cost actions."""

    assert exp.expected_action(0.0, false_accept_cost=5.0, escalation_cost=0.5) == "accept"
    assert exp.expected_action(1.0, false_accept_cost=5.0, escalation_cost=0.5) == "reject"
    # At p=0.5 with unit false-decision costs, accept and escalate tie.
    assert exp.expected_action(0.5, false_accept_cost=1.0, escalation_cost=0.5) == "accept"
    policies = exp.freeze_policies([0.1, 0.5, 0.9], [0.2, 0.6, 0.8])
    assert len(policies["rows"]) == 18
    assert policies["tie_breaking"] == ["accept", "escalate", "reject"]
    assert policies["policy_sha256"] == exp.policy_hash(policies)
    assert {row["probability_source"] for row in policies["rows"]} == {
        "window_gibbs",
        "temperature_whole",
    }


def test_fixture_artifact_cold_reduces_and_detects_mutation() -> None:
    """SCENARIO-VERIFY-7505-READINESS keeps readiness separate from efficacy."""

    artifact = exp.fixture_artifact()
    assert exp.validate_artifact(artifact, verify_sources=False) == []
    reduced = exp.independent_reduce(artifact)
    assert reduced == {"energy_fit_ready_score": 1, "baseline_ready_score": 1}
    assert artifact["predictive_benefit_measured"] is False
    assert artifact["verdict_class"] == "circular_positive"
    assert artifact["execution_venue"] == "host"
    assert artifact["honest_verdict"].startswith("complete_")
    assert set(exp.REQUIRED_FIELDS) <= set(artifact)
    assert set(artifact) <= set(artifact["field_principles"])
    mutated = deepcopy(artifact)
    mutated["energy_fit_ready_score"] = 0
    assert "independent_reduction_mismatch" in exp.validate_artifact(mutated, verify_sources=False)
    mutated = deepcopy(artifact)
    mutated["reproducibility_checksum"] = "sha256:changed"
    assert "reproducibility_checksum_mismatch" in exp.validate_artifact(
        mutated, verify_sources=False
    )


def test_real_preconditions_and_fit_reader_keep_heldout_closed() -> None:
    """SCENARIO-VERIFY-7505-ACCESS authenticates Exp7504 before label access."""

    preconditions = exp.collect_preconditions(exp.REPO_ROOT)
    assert preconditions["passed"] is True
    assert all(row["passed"] for row in preconditions["rows"])
    fit = exp.load_fit_rows(exp.REPO_ROOT)
    assert len(fit["training"]) == 176
    assert len(fit["calibration"]) == 60
    assert fit["access_receipt"]["label_roles_opened"] == [
        "calibration_tuning",
        "training",
    ]
    assert fit["access_receipt"]["held_out_labels_opened"] is False
    assert {row["role"] for row in fit["training"]} == {"training"}
    assert {row["role"] for row in fit["calibration"]} == {"calibration_tuning"}
    assert {row["label"] for row in fit["training"]} == {0, 1}


def test_source_verification_and_serialized_cold_replay(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-7505-E2E rejects source and serialized-byte drift."""

    artifact = exp.fixture_artifact()
    source = tmp_path / "source.txt"
    source.write_text("bound", encoding="utf-8")
    artifact["source_artifact_hashes"] = [exp.source_hash_row(source, tmp_path)]
    artifact["reproducibility_checksum"] = exp.artifact_checksum(artifact)
    assert exp.validate_artifact(artifact, root=tmp_path) == []
    source.write_text("changed", encoding="utf-8")
    assert any(
        error.startswith("source_hash_mismatch")
        for error in exp.validate_artifact(artifact, root=tmp_path)
    )
    candidate = tmp_path / "candidate.json"
    candidate.write_text(json.dumps(exp.fixture_artifact()), encoding="utf-8")
    assert exp.cold_replay(candidate, verify_sources=False) == []


def test_blocked_artifact_names_exact_missing_precondition(tmp_path: Path) -> None:
    """REQ-VERIFY-7505 records external absence as blocked, not a null."""

    failed = {
        "check": "resource_readable",
        "upstream": "Exp7504",
        "artifact_field": "path",
        "expected": "readable_nonempty_bytes",
        "observed": "missing",
        "path": "results/experiment_7504_v657_evidence_interface.json",
        "passed": False,
    }
    artifact = exp.blocked_artifact(failed, root=tmp_path)
    assert artifact["verdict_class"] == "blocked"
    assert artifact["honest_verdict"].startswith("complete_blocked_")
    assert artifact["gate_check_summary"]["observed"] == "missing"
    assert artifact["energy_fit_ready_score"] == 0
    assert artifact["baseline_ready_score"] == 0


def test_numeric_and_policy_guards_cover_malformed_inputs() -> None:
    """REQ-VERIFY-7505 rejects nonfinite, unsupported, and incomplete fit inputs."""

    matrix = np.asarray([row["features"] for row in _rows("training", 8)])
    bad = matrix.copy()
    bad[0, 0] = np.nan
    with pytest.raises(ValueError, match="feature_matrix_nonfinite"):
        exp.fit_spline_knots(bad)
    knots = exp.fit_spline_knots(matrix)
    with pytest.raises(ValueError, match="feature_matrix_shape_invalid"):
        exp.spline_design_matrix(np.ones((2, 9)), knots)
    with pytest.raises(ValueError, match="spline_input_nonfinite"):
        exp.spline_design_matrix(bad, knots)
    with pytest.raises(ValueError, match="optimizer_steps_invalid"):
        exp._fit_linear_head(matrix, np.ones(8), seed=1, regularization=0.0, steps=0)
    with pytest.raises(ValueError, match="optimizer_input_shape_invalid"):
        exp._fit_linear_head(matrix, np.ones(7), seed=1, regularization=0.0, steps=1)
    malformed = _rows("training", 8)
    malformed[0]["features"] = [0.0]
    with pytest.raises(ValueError, match="feature_vector_invalid"):
        exp.fit_energy_bundle(malformed, _rows("calibration_tuning", 6, 8), steps=1)
    malformed = _rows("training", 8)
    malformed[0]["raw_whole_expectation"] = 2.0
    with pytest.raises(ValueError, match="raw_probability_invalid"):
        exp.fit_energy_bundle(malformed, _rows("calibration_tuning", 6, 8), steps=1)
    with pytest.raises(ValueError, match="binary_support_invalid"):
        exp.fit_energy_bundle(
            [{**row, "label": 1} for row in _rows("training", 8)],
            _rows("calibration_tuning", 6, 8),
            steps=1,
        )
    with pytest.raises(ValueError, match="probability_invalid"):
        exp.expected_action(float("nan"), false_accept_cost=1.0, escalation_cost=0.5)
    with pytest.raises(ValueError, match="policy_probabilities_invalid"):
        exp.freeze_policies([], [0.5])
    with pytest.raises(ValueError, match="candidate_grid_incomplete"):
        exp._selected_head([], "window_gibbs")


def test_sidecar_writes_and_failure_reductions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-VERIFY-7505-E2E binds real sidecar bytes and failed reductions."""

    training = _rows("training", 12)
    calibration = _rows("calibration_tuning", 8, 12)
    bundle = exp.fit_energy_bundle(training, calibration, steps=1)
    calibration_rows = exp.build_calibration_rows(bundle, calibration)
    sidecars = exp.write_fit_sidecars(tmp_path, bundle, calibration_rows)
    assert sidecars["calibration_rows"]["rows"] == 48
    assert all((tmp_path / row["path"]).is_file() for row in sidecars.values())
    external = tmp_path / "external.txt"
    external.write_text("x", encoding="utf-8")
    receipt = exp.source_hash_row(external, tmp_path / "different-root")
    assert receipt["path"] == str(external)
    empty_root = tmp_path / "empty"
    empty_root.mkdir()
    assert exp.collect_preconditions(empty_root)["passed"] is False
    monkeypatch.setattr(
        exp,
        "read_mode",
        lambda *_args, **_kwargs: {"rows": [], "access_receipt": {}},
    )
    with pytest.raises(ValueError, match="fit_role_counts_invalid"):
        exp.load_fit_rows(tmp_path)
    assert exp.independent_reduce({}) == {
        "energy_fit_ready_score": 0,
        "baseline_ready_score": 0,
    }
    summary = exp._gate_summary([exp._gate("failed", "validity", True, False, "eq", False)])
    assert summary["failed_checks"] == ["failed"]


def test_validator_rejects_each_provenance_declaration() -> None:
    """REQ-VERIFY-7505 keeps current inference and schema declarations fail-closed."""

    base = exp.fixture_artifact()
    mutations = (
        ("schema", "wrong", "identity_mismatch"),
        ("MODEL_SPECS", ["wrong"], "model_specs_must_be_empty"),
        ("model_specs", ["wrong"], "model_specs_must_be_empty"),
        ("model_invoked", True, "current_model_calls_nonzero"),
        ("inference_substrate", "wrong", "inference_substrate_mismatch"),
        ("inference_substrate_class", "wrong", "inference_substrate_class_mismatch"),
        ("predictive_benefit_measured", True, "heldout_benefit_claimed"),
    )
    for field, replacement, expected in mutations:
        changed = deepcopy(base)
        changed[field] = replacement
        changed["reproducibility_checksum"] = exp.artifact_checksum(changed)
        assert expected in exp.validate_artifact(changed, verify_sources=False)
    missing = deepcopy(base)
    del missing["milestone"]
    missing["reproducibility_checksum"] = exp.artifact_checksum(missing)
    assert any(
        error.startswith("required_fields_missing")
        for error in exp.validate_artifact(missing, verify_sources=False)
    )
    principles = deepcopy(base)
    del principles["field_principles"]["milestone"]
    principles["reproducibility_checksum"] = exp.artifact_checksum(principles)
    assert "field_principles_incomplete" in exp.validate_artifact(principles, verify_sources=False)
