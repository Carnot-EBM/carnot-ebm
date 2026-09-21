"""Tests for REQ-KAN-7468 delayed-feedback local residual learning."""

from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path

import numpy as np
import pytest

from carnot import experiment_7468_v654_residual_learner as exp


ROOT = Path(__file__).resolve().parents[2]


def _training() -> np.ndarray:
    """Return training-only values that give disjoint endpoint support."""

    return np.asarray(
        [
            [-2.0, -1.0, 0.0, 1.0],
            [-1.5, -0.5, 0.5, 1.5],
            [-1.0, 0.0, 1.0, 2.0],
            [0.0, 0.5, 1.5, 2.5],
            [1.0, 1.0, 2.0, 3.0],
            [2.0, 1.5, 2.5, 3.5],
        ],
        dtype=np.float64,
    )


def _passing_receipts() -> list[dict[str, object]]:
    """Return the frozen validation receipt names with passing outcomes."""

    return [
        {
            "name": name,
            "required": True,
            "passed": True,
            "exit_code": 0,
            "timed_out": False,
        }
        for name in (*exp.AFFECTED_CHECK_NAMES, *exp.TERMINAL_CHECK_NAMES)
    ]


def test_req_kan_7468_spec_precedes_implementation() -> None:
    """REQ-KAN-7468: the new behavior has a requirement and all scenarios."""

    text = (ROOT / exp.SPEC_PATH).read_text(encoding="utf-8")
    assert "## REQ-KAN-7468:" in text
    for number in range(1, 10):
        assert f"SCENARIO-KAN-7468-{number:02d}" in text


def test_scenario_kan_7468_01_energy_difference_has_probability_sign() -> None:
    """SCENARIO-KAN-7468-01: positive good-vs-bad energy gap raises p(good)."""

    head = exp.LocalResidualHead.from_training(_training(), seed=7468)
    head.coefficients.fill(0.0)
    head.bias = 0.0
    features = [0.0, 0.5, 1.5, 2.5]
    for frozen_probability in (0.2, 0.5, 0.8):
        prediction = head.predict(features, frozen_probability=frozen_probability)
        expected_gap = math.log(frozen_probability / (1.0 - frozen_probability))
        assert prediction["energy_bad_minus_good"] == pytest.approx(expected_gap, abs=1e-14)
        assert prediction["residual_prediction"] == pytest.approx(frozen_probability, abs=1e-14)
    head.bias = 0.4
    shifted = head.predict(features, frozen_probability=0.5)
    assert shifted["energy_bad_minus_good"] > 0.0
    assert shifted["residual_prediction"] > 0.5


def test_scenario_kan_7468_02_03_prediction_is_immutable_and_feedback_fails_closed() -> None:
    """SCENARIO-KAN-7468-02/03: reveal order and identity gate every update."""

    head = exp.LocalResidualHead.from_training(_training(), seed=7)
    first = head.seal_prediction(
        event_id="event-a",
        source_version="analytic-v1",
        prediction_time=0,
        reveal_time=8,
        features=[-1.0, 0.0, 1.0, 2.0],
        frozen_probability=0.6,
    )
    second = head.seal_prediction(
        event_id="event-b",
        source_version="analytic-v1",
        prediction_time=1,
        reveal_time=9,
        features=[1.0, 1.0, 2.0, 3.0],
        frozen_probability=0.4,
    )
    assert "label" not in first
    assert first["active_support"]
    assert first["prediction_event_hash"] == exp.prediction_event_hash(first)
    before = head.state_vector.copy()
    assert head.apply_feedback("missing", label=1, visible_at=9)["status"] == "missing_prediction"
    assert head.apply_feedback("event-a", label=1, visible_at=7)["status"] == "not_revealed"
    assert head.apply_feedback("event-b", label=0, visible_at=9)["status"] == "reordered"
    assert np.array_equal(before, head.state_vector)
    committed = head.apply_feedback("event-a", label=1, visible_at=8)
    assert committed["status"] == "committed"
    assert committed["updates_applied"] == 1
    assert head.apply_feedback("event-a", label=1, visible_at=10)["status"] == "duplicate"
    assert head.apply_feedback("event-a", label=0, visible_at=10)["status"] == "identity_conflict"
    assert head.apply_feedback("event-b", label=0, visible_at=9)["status"] == "committed"
    assert first["predictor_state_hash"] != head.state_hash
    assert second["prediction_event_hash"] == exp.prediction_event_hash(second)


def test_scenario_kan_7468_04_sparse_and_independent_dense_gradients_match() -> None:
    """SCENARIO-KAN-7468-04: local and dense gradients agree within 1e-10."""

    head = exp.LocalResidualHead.from_training(_training(), seed=11)
    features = np.asarray([-0.75, 0.2, 1.2, 2.2], dtype=np.float64)
    sparse, support = head.sparse_gradient(features, 1, frozen_probability=0.35)
    dense = exp.independent_dense_gradient(head, features, 1, frozen_probability=0.35)
    assert np.max(np.abs(sparse - dense)) <= exp.PARITY_TOLERANCE
    assert len(support) <= exp.FEATURE_COUNT * (exp.DEGREE + 1)
    assert np.count_nonzero(sparse[:-1]) <= len(support)


def test_scenario_kan_7468_05_guard_rejects_and_rolls_back() -> None:
    """SCENARIO-KAN-7468-05: a harmful replay update leaves exact prior state."""

    guard_features = [-1.0, 0.0, 1.0, 2.0]
    head = exp.LocalResidualHead.from_training(
        _training(),
        seed=17,
        guard_rows=[{"features": guard_features, "label": 0, "frozen_probability": 0.5}],
        guard_tolerance=0.0,
    )
    head.coefficients.fill(0.0)
    head.bias = 0.0
    head.seal_prediction(
        event_id="guard-conflict",
        source_version="analytic-v1",
        prediction_time=0,
        reveal_time=0,
        features=guard_features,
        frozen_probability=0.5,
    )
    before = head.state_vector.copy()
    receipt = head.apply_feedback("guard-conflict", label=1, visible_at=0)
    assert receipt["status"] == "rejected_guard"
    assert receipt["rolled_back"] is True
    assert receipt["guard_loss_after"] > receipt["guard_loss_before"]
    assert receipt["guard_rows_evaluated"] == 2
    assert receipt["guard_basis_evaluations"] == 8
    assert receipt["guard_elapsed_ns"] >= 0
    assert np.array_equal(before, head.state_vector)


def test_scenario_kan_7468_06_disjoint_support_retains_only_local_weights() -> None:
    """SCENARIO-KAN-7468-06: distant coefficients stay fixed; bias is isolated."""

    head = exp.LocalResidualHead.from_training(_training(), seed=23)
    head.coefficients.fill(0.0)
    head.bias = 0.0
    low = _training().min(axis=0)
    high = _training().max(axis=0)
    low_prediction = head.predict(low, frozen_probability=0.5)
    high_before = head.predict(high, frozen_probability=0.5)
    assert set(low_prediction["active_support"]).isdisjoint(high_before["active_support"])
    high_coefficients = head.coefficients.reshape(-1)[high_before["active_support"]].copy()
    head.seal_prediction(
        event_id="local-low",
        source_version="analytic-v1",
        prediction_time=0,
        reveal_time=0,
        features=low,
        frozen_probability=0.5,
    )
    receipt = head.apply_feedback("local-low", label=1, visible_at=0)
    high_after = head.predict(high, frozen_probability=0.5)
    assert receipt["status"] == "committed"
    assert np.array_equal(
        high_coefficients, head.coefficients.reshape(-1)[high_before["active_support"]]
    )
    assert (
        high_after["local_coefficient_contribution"]
        == high_before["local_coefficient_contribution"]
    )
    assert high_after["shared_bias_contribution"] != high_before["shared_bias_contribution"]


def test_scenario_kan_7468_07_checkpoint_restart_and_controls(tmp_path: Path) -> None:
    """SCENARIO-KAN-7468-07: cold restart equals uninterrupted delayed replay."""

    head = exp.LocalResidualHead.from_training(_training(), seed=29)
    for index, delay in enumerate((0, 8)):
        head.seal_prediction(
            event_id=f"restart-{index}",
            source_version="analytic-v1",
            prediction_time=index,
            reveal_time=index + delay,
            features=_training()[index + 1],
            frozen_probability=0.45 + 0.1 * index,
        )
    assert head.apply_feedback("restart-0", label=1, visible_at=0)["status"] == "committed"
    checkpoint = tmp_path / "residual.json"
    head.save_checkpoint(checkpoint)
    uninterrupted = deepcopy(head)
    assert uninterrupted.apply_feedback("restart-1", label=0, visible_at=9)["status"] == "committed"
    restored = exp.LocalResidualHead.load_checkpoint(checkpoint)
    assert restored.apply_feedback("restart-1", label=0, visible_at=9)["status"] == "committed"
    assert np.array_equal(restored.state_vector, uninterrupted.state_vector)
    assert restored.state_hash == uninterrupted.state_hash

    controls = exp.run_control_comparison(_training())
    assert set(controls) == {
        "local_residual",
        "matched_linear_residual",
        "frozen_residual",
        "no_feedback",
    }
    assert controls["frozen_residual"]["update_count"] == 0
    assert controls["no_feedback"]["state_unchanged"] is True


def test_scenario_kan_7468_08_exp7469_protocol_is_frozen() -> None:
    """SCENARIO-KAN-7468-08: all prospective protocol choices are explicit."""

    protocol = exp.exp7469_protocol()
    assert protocol["seeds"] == [746900, 746901, 746902, 746903, 746904]
    assert protocol["label_blind_group_orders"] == [
        "stable_hash_ascending",
        "stable_hash_descending",
    ]
    assert protocol["delays"] == [0, 8]
    assert protocol["controls"] == ["frozen_residual", "matched_linear_residual"]
    assert protocol["uniform_audit_probability"] == 0.5
    assert protocol["full_feedback_diagnostic"]["confirmatory"] is False
    assert protocol["ipw_clip"] == [0.1, 10.0]
    assert protocol["moving_block_lengths"] == [16, 32]
    assert protocol["bootstrap_resamples"] == 10000
    assert protocol["state_carry_across_replicates"] is False
    assert protocol["outcome_tuning_allowed"] is False


def test_scenario_kan_7468_09_artifact_reduction_and_mutations_fail_closed(
    tmp_path: Path,
) -> None:
    """SCENARIO-KAN-7468-09: raw evidence determines the circular result."""

    artifact = exp.build_fixture_artifact(validation_receipts=_passing_receipts())
    assert exp.validate_artifact(artifact, verify_sources=False) == []
    assert exp.independent_reduce(artifact)["residual_learner_ready_score"] == 1
    assert artifact["verdict_class"] == "circular_positive"
    assert artifact["verifier_is_oracle"] is True
    assert artifact["MODEL_SPECS"] == artifact["model_specs"] == []
    assert artifact["model_invoked"] is False
    assert artifact["invocation_counts"] == exp.ZERO_INVOCATION_COUNTS
    assert artifact["inference_substrate_class"] == "no_model_load"
    assert artifact["small_ebm_training"]["generator_weights_fitted"] is False
    assert artifact["online_protocol"] == exp.exp7469_protocol()

    for field, replacement, expected in (
        ("residual_learner_ready_score", 0, "residual_learner_ready_score_mismatch"),
        ("model_invoked", True, "model_declaration_mismatch"),
        ("reproducibility_checksum", "sha256:changed", "reproducibility_checksum_mismatch"),
    ):
        changed = deepcopy(artifact)
        changed[field] = replacement
        assert expected in exp.validate_artifact(changed, verify_sources=False)
    changed = deepcopy(artifact)
    changed["analytic_checks"]["gradient_parity"]["max_abs_gap"] = 1e-5
    changed["reproducibility_checksum"] = exp.reproducibility_checksum(changed)
    assert "residual_learner_ready_score_mismatch" in exp.validate_artifact(
        changed, verify_sources=False
    )

    path = tmp_path / "candidate.json"
    path.write_text(json.dumps(artifact), encoding="utf-8")
    assert exp.cold_replay(path, verify_sources=False) == []
    path.write_text("[]", encoding="utf-8")
    assert exp.cold_replay(path, verify_sources=False) == ["artifact_unreadable_or_not_object"]


def test_req_kan_7468_rejects_malformed_numeric_ledger_and_checkpoint_state(
    tmp_path: Path,
) -> None:
    """REQ-KAN-7468: malformed inputs cannot become update or restart state."""

    training = _training()
    with pytest.raises(ValueError, match="shape"):
        exp.fit_local_knots([[0.0, 1.0]])
    changed = training.copy()
    changed[0, 0] = math.nan
    with pytest.raises(ValueError, match="nonfinite"):
        exp.fit_local_knots(changed)
    head = exp.LocalResidualHead.from_training(training, seed=31)
    with pytest.raises(ValueError, match="shape"):
        head.predict([0.0], frozen_probability=0.5)
    with pytest.raises(ValueError, match="nonfinite"):
        head.predict([0.0, 0.5, math.inf, 2.5], frozen_probability=0.5)
    with pytest.raises(ValueError, match="strictly"):
        head.predict(training[0], frozen_probability=1.0)
    with pytest.raises(ValueError, match="nonfinite"):
        exp.LocalResidualHead(head.knots, head.coefficients, math.nan)
    with pytest.raises(ValueError, match="range"):
        exp.LocalResidualHead(head.knots, head.coefficients, 0.0, learning_rate=0.0)
    with pytest.raises(ValueError, match="guard_label"):
        exp.LocalResidualHead.from_training(
            training,
            seed=31,
            guard_rows=[{"features": training[0], "label": 2, "frozen_probability": 0.5}],
        )

    with pytest.raises(ValueError, match="identity"):
        head.seal_prediction(
            event_id="",
            source_version="analytic-v1",
            prediction_time=0,
            reveal_time=0,
            features=training[0],
            frozen_probability=0.5,
        )
    head.seal_prediction(
        event_id="valid",
        source_version="analytic-v1",
        prediction_time=0,
        reveal_time=0,
        features=training[0],
        frozen_probability=0.5,
    )
    with pytest.raises(ValueError, match="duplicate"):
        head.seal_prediction(
            event_id="valid",
            source_version="analytic-v1",
            prediction_time=0,
            reveal_time=0,
            features=training[0],
            frozen_probability=0.5,
        )
    with pytest.raises(ValueError, match="time"):
        head.seal_prediction(
            event_id="bad-time",
            source_version="analytic-v1",
            prediction_time=2,
            reveal_time=1,
            features=training[0],
            frozen_probability=0.5,
        )
    with pytest.raises(ValueError, match="label"):
        head.sparse_gradient(training[0], 2, frozen_probability=0.5)
    with pytest.raises(ValueError, match="label"):
        head.apply_feedback("valid", label=2, visible_at=0)
    with pytest.raises(ValueError, match="linear_training"):
        exp.MatchedLinearResidual([[0.0, 1.0]])

    checkpoint = tmp_path / "checkpoint.json"
    head.save_checkpoint(checkpoint)
    invalid_schema = json.loads(checkpoint.read_text(encoding="utf-8"))
    invalid_schema["schema"] = "wrong"
    checkpoint.write_text(json.dumps(invalid_schema), encoding="utf-8")
    with pytest.raises(ValueError, match="schema"):
        exp.LocalResidualHead.load_checkpoint(checkpoint)
    head.save_checkpoint(checkpoint)
    invalid_hash = json.loads(checkpoint.read_text(encoding="utf-8"))
    invalid_hash["bias"] = 99.0
    checkpoint.write_text(json.dumps(invalid_hash), encoding="utf-8")
    with pytest.raises(ValueError, match="hash"):
        exp.LocalResidualHead.load_checkpoint(checkpoint)


def test_scenario_kan_7468_09_cold_reader_names_schema_and_source_defects(
    tmp_path: Path,
) -> None:
    """SCENARIO-KAN-7468-09: each changed declaration or source fails closed."""

    artifact = exp.build_fixture_artifact(validation_receipts=_passing_receipts())
    mutations = (
        ("schema", "wrong", "identity_mismatch"),
        ("run_date", "19000101", "run_identity_mismatch"),
        ("invocation_counts", {}, "invocation_counts_mismatch"),
        ("inference_substrate_class", "wrong", "inference_substrate_class_mismatch"),
        ("execution_venue", "wrong", "execution_venue_mismatch"),
        ("verifier_is_oracle", False, "verifier_oracle_mismatch"),
        ("verdict_class", "positive", "positive_forbidden_for_oracle"),
        ("online_protocol", {}, "online_protocol_mismatch"),
    )
    for field, replacement, expected in mutations:
        changed = deepcopy(artifact)
        changed[field] = replacement
        assert expected in exp.validate_artifact(changed, verify_sources=False)

    changed = deepcopy(artifact)
    changed["source_artifact_hashes"] = {"bad": "not-a-row"}
    assert "source_hash_row_invalid:bad" in exp.validate_artifact(changed)
    changed["source_artifact_hashes"] = {
        "bad": {"path": str(tmp_path / "missing"), "sha256": "sha256:missing"}
    }
    assert "source_hash_invalid:bad" in exp.validate_artifact(changed)

    malformed = tmp_path / "malformed.json"
    malformed.write_text("{", encoding="utf-8")
    assert exp.cold_replay(malformed, verify_sources=False) == ["artifact_unreadable_or_not_object"]
