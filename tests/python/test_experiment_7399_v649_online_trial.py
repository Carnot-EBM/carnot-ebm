"""Tests for the V649 sealed delayed-adapter online trial.

Spec refs: REQ-CL-7399 and SCENARIO-CL-7399-GATE through
SCENARIO-CL-7399-ARTIFACT.
"""

from __future__ import annotations

from copy import deepcopy
import json
import math

import pytest

from carnot import experiment_7399_v649_online_trial as trial


def _weights() -> dict[str, object]:
    return {
        "w1": [[0.5, -0.25], [0.2, 0.3], [-0.4, 0.1], [0.6, -0.2]],
        "b1": [0.1, -0.1, 0.0, 0.2],
        "w_out": [0.3, -0.2, 0.4, 0.1],
        "b_out": -0.05,
    }


def _training_record(seed: int = trial.TRAINING_SEEDS[0]) -> dict[str, object]:
    return {
        "seed": seed,
        "gibbs_objective": "natural_prevalence_bernoulli_gibbs",
        "gibbs_steps": 4,
        "future_stream_groups_used": 0,
        "initial_group_count": 4,
        "initial_group_ids_sha256": trial.canonical_hash(["i0", "i1", "i2", "i3"]),
        "weights": _weights(),
        "affine": {"a": 1.25, "b": -0.2},
        "logistic_weights": {"coef": [0.1, -0.3], "bias": 0.05},
    }


def _row(group: str, index: int, label: int) -> dict[str, object]:
    return {
        "group_id": group,
        "source_row_index": index,
        "entity_uptake": float(index % 3) / 2.0,
        "falsifiability_score": float((index + 1) % 4) / 3.0,
        "label": label,
        "partition": "training",
    }


def _states() -> dict[str, dict[str, object]]:
    labels = [0, 1, 0, 1]
    return trial.states_from_numeric_checkpoint(
        trial.make_numeric_checkpoint(_training_record()), labels
    )


def _policy() -> dict[str, object]:
    return {
        "accept_threshold": 0.05,
        "reject_threshold": 0.99,
        "accept_enabled": True,
        "reject_enabled": False,
    }


def test_numeric_checkpoint_is_code_free_and_numpy_scoreable() -> None:
    """SCENARIO-CL-7399-CHECKPOINT: plain numeric state reproduces scoring."""

    checkpoint = trial.make_numeric_checkpoint(_training_record())
    assert trial.numeric_checkpoint_errors(checkpoint) == []
    assert checkpoint["training_authority"] == "frozen_initialization_groups_only"
    assert checkpoint["future_stream_groups_used"] == 0
    score = trial.score_numeric_checkpoint(checkpoint, [0.25, 0.75])
    hidden_linear = [
        sum(weight * value for weight, value in zip(row, [0.25, 0.75], strict=True)) + bias
        for row, bias in zip(_weights()["w1"], _weights()["b1"], strict=True)
    ]
    hidden = [value / (1.0 + math.exp(-value)) for value in hidden_linear]
    expected_energy = (
        sum(weight * value for weight, value in zip(_weights()["w_out"], hidden, strict=True))
        + _weights()["b_out"]
    )
    expected_probability = 1.0 / (1.0 + math.exp(-(1.25 * expected_energy - 0.2)))
    assert score["energy"] == pytest.approx(expected_energy)
    assert score["probability"] == pytest.approx(expected_probability)
    assert score["checkpoint_hash"] == trial.checkpoint_hash(checkpoint)

    changed = deepcopy(checkpoint)
    changed["affine"]["b"] = 0.1
    assert trial.checkpoint_hash(changed) != trial.checkpoint_hash(checkpoint)
    malformed = deepcopy(checkpoint)
    malformed["weights"]["w1"] = [[float("nan"), 0.0]] * 4
    assert "numeric_checkpoint_invalid" in trial.numeric_checkpoint_errors(malformed)
    assert trial.numeric_checkpoint_errors([]) == ["numeric_checkpoint_not_object"]
    with pytest.raises(ValueError, match="non-numeric"):
        trial._numeric_tree("opaque")
    invalid_record = _training_record()
    invalid_record["gibbs_steps"] = trial.MAX_STEPS + 1
    with pytest.raises(ValueError, match="numeric_checkpoint_invalid"):
        trial.make_numeric_checkpoint(invalid_record)
    with pytest.raises(ValueError, match="numeric_checkpoint_invalid"):
        trial.score_numeric_checkpoint(malformed, [0.0, 1.0])
    with pytest.raises(ValueError, match="two finite"):
        trial.score_numeric_checkpoint(checkpoint, [0.0])
    with pytest.raises(ValueError, match="numeric checkpoint invalid"):
        trial.states_from_numeric_checkpoint(malformed, [0, 1])


def test_replay_rows_keep_preupdate_actions_feedback_hashes_and_costs(tmp_path) -> None:
    """SCENARIO-CL-7399-REPLAY: paired rows expose authority and full cost."""

    stream = [_row(f"g{index}", index, index % 2) for index in range(6)]
    result = trial.measure_replay_unit(
        _states(),
        stream,
        ordering="fixed_hash_order",
        condition={"name": "primary_delay_1", "delay": 1, "missing_fraction": 0.0},
        seed=trial.TRAINING_SEEDS[0],
        thresholds=_policy(),
        missing_group_ids=set(),
        checkpoint=trial.make_numeric_checkpoint(_training_record()),
        durable_path=tmp_path / "unit.json",
    )
    rows = result["rows"]
    assert len(rows) == len(stream) * len(trial.ARMS)
    required = {
        "pre_update_probability",
        "typed_action",
        "label_availability_time",
        "feedback_disposition",
        "state_hash_before_update",
        "state_hash_after_update",
        "brier_loss",
        "log_loss",
        "feature_extraction_latency_s",
        "energy_scoring_latency_s",
        "affine_prediction_latency_s",
        "update_latency_s",
        "durable_state_write_latency_s",
        "orchestration_latency_s",
        "full_cost_s",
    }
    assert all(required <= set(row) for row in rows)
    assert all(row["pre_update_probability"] == row["probability"] for row in rows)
    assert all(row["typed_action"] == row["decision"] for row in rows)
    assert all(row["full_cost_s"] >= row["update_latency_s"] for row in rows)
    assert any(row["feedback_disposition"] == "committed" for row in rows)
    assert result["durable_state_bytes"] > 0
    assert result["failed_units"] == 0
    assert result["unscoreable_units"] == 0


def test_replay_counts_independent_score_mismatch(monkeypatch, tmp_path) -> None:
    """SCENARIO-CL-7399-CHECKPOINT: changed NumPy scoring is unscoreable."""

    original = trial.score_numeric_checkpoint

    def changed_score(checkpoint, features):
        value = original(checkpoint, features)
        value["energy"] += 10.0
        return value

    monkeypatch.setattr(trial, "score_numeric_checkpoint", changed_score)
    stream = [_row(f"g{index}", index, index % 2) for index in range(4)]
    result = trial.measure_replay_unit(
        _states(),
        stream,
        ordering="fixed_hash_order",
        condition={"name": "primary_delay_1", "delay": 1, "missing_fraction": 0.0},
        seed=trial.TRAINING_SEEDS[0],
        thresholds=_policy(),
        missing_group_ids=set(),
        checkpoint=trial.make_numeric_checkpoint(_training_record()),
        durable_path=tmp_path / "changed.json",
    )
    assert result["unscoreable_units"] > 0


def test_causal_erasure_cold_restart_matches_no_feedback() -> None:
    """SCENARIO-CL-7399-ERASURE: erased prior labels restore the control state."""

    stream = [_row(f"g{index}", index, index % 2) for index in range(8)]
    row = trial.causal_erasure_probe(
        _states(),
        stream,
        seed=trial.TRAINING_SEEDS[0],
        prefix_count=4,
    )
    assert row["cold_restart_performed"] is True
    assert row["erased_prior_update_count"] == 4
    assert row["erased_probability"] == pytest.approx(row["no_feedback_probability"])
    assert row["erased_state_hash"] == row["no_feedback_state_hash"]
    assert row["current_query_label_used"] is False
    assert row["future_label_used"] is False
    assert row["sample_order_preserved"] is True
    assert row["passed"] is True
    with pytest.raises(ValueError, match="prefix"):
        trial.causal_erasure_probe(_states(), stream, seed=1, prefix_count=len(stream))


def test_latency_report_separates_all_registered_stages() -> None:
    """SCENARIO-CL-7399-COST: stage distributions remain explicit."""

    rows = [
        {
            "feature_extraction_latency_s": 0.001 * scale,
            "energy_scoring_latency_s": 0.002 * scale,
            "affine_prediction_latency_s": 0.003 * scale,
            "update_latency_s": 0.004 * scale,
            "durable_state_write_latency_s": 0.005 * scale,
            "orchestration_latency_s": 0.006 * scale,
            "full_cost_s": 0.021 * scale,
        }
        for scale in (1.0, 2.0, 3.0)
    ]
    report = trial.latency_report(rows, serialized_bytes=123, update_operations=7)
    assert set(report["stage_distributions"]) == set(trial.LATENCY_FIELDS)
    assert report["stage_distributions"]["feature_extraction_latency_s"]["count"] == 3
    assert report["serialized_bytes"] == 123
    assert report["update_operation_count"] == 7
    assert report["measured_hardware_path"] == "bounded_cpu_scalar_updates"
    assert report["measured_100x_claim"] is False
    assert (
        trial.latency_report([], serialized_bytes=0, update_operations=0)["stage_distributions"][
            "full_cost_s"
        ]["count"]
        == 0
    )


def test_reducer_keeps_capture_separate_from_registered_value() -> None:
    """SCENARIO-CL-7399-GATE-REDUCTION: a complete efficacy miss stays null."""

    artifact = trial.build_fixture_artifact()
    assert trial.validate_artifact(artifact) == []
    assert artifact["online_capture_complete_score"] == 1
    assert artifact["online_value_score"] == 0
    assert artifact["verdict_class"] == "null"
    assert artifact["MODEL_SPECS"] == []
    assert artifact["model_invoked"] is False
    assert artifact["inference_substrate_class"] == "no_model_load"
    assert artifact["execution_venue"] == "host"
    assert artifact["promotion_score"] == 0

    changed = deepcopy(artifact)
    changed["rows"][0]["brier_loss"] = 0.9
    assert "reproducibility_checksum_mismatch" in trial.validate_artifact(changed)
    changed = deepcopy(artifact)
    changed["online_capture_complete_score"] = 0
    assert "independent_reduction_mismatch" in trial.validate_artifact(changed)
    changed = deepcopy(artifact)
    changed["numeric_checkpoint_manifest"][0]["checkpoint_hash"] = "sha256:bad"
    assert "numeric_checkpoint_manifest_invalid" in trial.validate_artifact(changed)
    assert trial._manifest_valid([]) is False
    assert trial._gate("ge", "completion", 1, 2, ">=")["passed"] is True
    assert trial._gate("le", "completion", 2, 1, "<=")["passed"] is True

    for field, value, error in (
        ("MODEL_SPECS", ["bad"], "substrate_declaration_mismatch"),
        ("verifier_is_oracle", True, "oracle_declaration_mismatch"),
        ("promotion_score", 1, "promotion_nonzero"),
        ("verdict_class", "bad", "verdict_class_invalid"),
        ("field_principles", {}, "field_principles_incomplete"),
    ):
        changed = deepcopy(artifact)
        changed[field] = value
        assert error in trial.validate_artifact(changed)
    changed = deepcopy(artifact)
    changed["online_value_score"] = 1
    assert "independent_reduction_mismatch" in trial.validate_artifact(changed)
    changed = deepcopy(artifact)
    changed["flagged_adversarial"] = True
    assert "adversarial_scores_nonzero" in trial.validate_artifact(changed)


def test_blocked_artifact_names_exact_failed_gate() -> None:
    """SCENARIO-CL-7399-GATE: unchanged prerequisite absence blocks row-free."""

    checks = [
        trial.precondition_row(
            "exp7397_ready",
            "results/experiment_7397_v649_delayed_adapter.json",
            "delayed_adapter_ready_score",
            1,
            None,
        )
    ]
    blocked = trial.build_blocked_artifact(checks, {})
    assert blocked["verdict_class"] == "blocked"
    assert blocked["rows"] == []
    assert blocked["causal_erasure_rows"] == []
    assert blocked["online_capture_complete_score"] == 0
    assert blocked["online_value_score"] == 0
    failure = blocked["gate_check_summary"]["first_required_failure"]
    assert failure["upstream"] == "results/experiment_7397_v649_delayed_adapter.json"
    assert failure["observed"] is None
    assert trial.validate_artifact(blocked) == []

    malformed = deepcopy(blocked)
    malformed["rows"] = [{}]
    malformed["online_capture_complete_score"] = 1
    malformed["gate_check_summary"]["first_required_failure"] = None
    errors = trial.validate_artifact(malformed)
    assert "blocked_artifact_has_dependent_work" in errors
    assert "blocked_scores_nonzero" in errors
    assert "blocked_gate_summary_missing" in errors


def test_validation_manifest_is_exact_and_has_no_full_suite(tmp_path) -> None:
    """SCENARIO-CL-7399-ARTIFACT: Exp7358 freezes the eight affected checks."""

    commands = trial.build_validation_commands(trial.REPO_ROOT, tmp_path)
    assert [command.name for command in commands] == list(
        trial.validation_scope.REQUIRED_CHECK_NAMES
    )
    assert all(command.name != "full_python_suite" for command in commands)
    assert trial.validate_command_plan(trial.REPO_ROOT, trial.V649_MANIFEST, commands) == []
    report = next(
        command for command in commands if command.name == "changed_module_coverage_report"
    )
    assert "COVERAGE_FILE" in dict(report.command_environment)


def test_main_helpers_reject_invalid_identity_and_write_atomic_json(tmp_path) -> None:
    """REQ-CL-7399: identity, hashes, and atomic writes fail closed."""

    path = tmp_path / "nested" / "value.json"
    trial.atomic_json(path, {"value": 1})
    assert json.loads(path.read_text()) == {"value": 1}
    assert trial.sha256_file(path).startswith("sha256:")
    assert trial.validate_artifact([]) == ["artifact_not_object"]
    artifact = trial.build_fixture_artifact()
    artifact["schema"] = "bad"
    assert "identity_mismatch" in trial.validate_artifact(artifact)
    assert trial._percentile([1.0, 2.0, 3.0], 0.5) == 2.0
    assert trial._percentile([], 0.5) == 0.0
