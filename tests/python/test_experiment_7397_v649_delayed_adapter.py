"""Tests for the V649 delayed energy-calibration experiment.

Spec refs: REQ-REPORT-7397 and SCENARIO-REPORT-7397-REPLAY through
SCENARIO-REPORT-7397-ARTIFACT.
"""

from __future__ import annotations

from copy import deepcopy
import json

import pytest

from carnot import experiment_7397_v649_delayed_adapter as exp


def _row(group: str, index: int, label: int, partition: str = "training") -> dict[str, object]:
    return {
        "group_id": group,
        "source_row_index": index,
        "entity_uptake": float(index % 3) / 2.0,
        "falsifiability_score": float((index + 1) % 4) / 3.0,
        "label": label,
        "partition": partition,
    }


def _protocol() -> dict[str, object]:
    rows = [_row(f"init-{index}", index, index % 2) for index in range(8)]
    rows += [_row(f"later-{index}", index + 8, index % 2) for index in range(8)]
    rows += [
        _row(f"policy-{index}", index + 16, index % 2, "policy_calibration") for index in range(8)
    ]
    rows += [_row("secret-final", 99, 1, "final_test")]
    return {
        "feature_rows": rows,
        "online_replay": {
            "initialization_group_ids": [f"init-{index}" for index in range(8)],
            "later_group_ids": [f"later-{index}" for index in range(8)],
        },
    }


def test_streams_preserve_membership_without_final_test_labels() -> None:
    """SCENARIO-REPORT-7397-REPLAY: both orders contain only frozen later groups."""

    streams = exp.build_streams(_protocol(), block_length=3)
    fixed_ids = [row["group_id"] for row in streams["fixed_hash_order"]]
    reversed_ids = [row["group_id"] for row in streams["reversed_block_order"]]
    assert fixed_ids == [f"later-{index}" for index in range(8)]
    assert sorted(reversed_ids) == sorted(fixed_ids)
    assert reversed_ids == [
        "later-6",
        "later-7",
        "later-3",
        "later-4",
        "later-5",
        "later-0",
        "later-1",
        "later-2",
    ]
    assert "secret-final" not in fixed_ids + reversed_ids

    broken = _protocol()
    broken["online_replay"]["later_group_ids"].append("missing")
    with pytest.raises(ValueError, match="missing"):
        exp.build_streams(broken)
    duplicate = _protocol()
    duplicate["online_replay"]["later_group_ids"][-1] = "later-0"
    with pytest.raises(ValueError, match="unique"):
        exp.build_streams(duplicate)
    with pytest.raises(ValueError, match="positive"):
        exp.build_streams(_protocol(), block_length=0)


def test_feedback_conditions_and_masks_are_frozen() -> None:
    """REQ-REPORT-7397: delay and missing-label sensitivities stay separate."""

    conditions = exp.feedback_conditions()
    assert conditions == (
        {"name": "primary_delay_1", "delay": 1, "missing_fraction": 0.0},
        {"name": "sensitivity_delay_32", "delay": 32, "missing_fraction": 0.0},
        {"name": "sensitivity_missing_25", "delay": 1, "missing_fraction": 0.25},
    )
    ids = [f"group-{index}" for index in range(12)]
    mask = exp.deterministic_missing_mask(ids)
    assert len(mask) == 3
    assert mask == exp.deterministic_missing_mask(ids)
    disabled = exp._policy_action(
        0.0,
        {
            "accept_threshold": 0.1,
            "reject_threshold": 0.9,
            "accept_enabled": False,
            "reject_enabled": True,
        },
        "fixture",
    )
    assert disabled["decision"] == "escalate"


def test_tiny_replay_freezes_gibbs_and_pairs_every_arm() -> None:
    """SCENARIO-REPORT-7397-FROZEN: replay updates scalars, not Gibbs weights."""

    protocol = _protocol()
    initialization = exp.initialization_rows(protocol)
    states = exp.initialize_seed_states(initialization, seed=exp.TRAINING_SEEDS[0], steps=4)
    before_hash = states["adaptive_affine_gibbs"]["adapter"]["weights_hash"]
    stream = exp.build_streams(protocol)["fixed_hash_order"]
    policy_rows = exp.policy_calibration_rows(protocol)
    thresholds = exp.select_fixed_thresholds(states, policy_rows)
    result = exp.replay_condition(
        states,
        stream,
        ordering="fixed_hash_order",
        condition={"name": "primary_delay_1", "delay": 1, "missing_fraction": 0.0},
        seed=exp.TRAINING_SEEDS[0],
        thresholds=thresholds,
        missing_group_ids=set(),
    )
    assert len(result["rows"]) == len(stream) * len(exp.ARMS)
    assert len(result["event_ledger"]) == len(stream)
    assert all(row["prediction_before_feedback"] is True for row in result["rows"])
    assert {row["arm"] for row in result["rows"]} == set(exp.ARMS)
    assert result["final_states"]["adaptive_affine_gibbs"]["adapter"]["weights_hash"] == before_hash
    assert result["final_states"]["no_feedback_adaptive_control"]["update_count"] == 0

    checkpoint = exp.Path("/tmp/exp7397-test-midpoint.json")
    missing = exp.replay_condition(
        states,
        stream,
        ordering="fixed_hash_order",
        condition={"name": "sensitivity_missing_25", "delay": 1, "missing_fraction": 0.25},
        seed=exp.TRAINING_SEEDS[0],
        thresholds={**thresholds, "accept_enabled": False},
        missing_group_ids={str(stream[0]["group_id"])},
        checkpoint_path=checkpoint,
    )
    assert missing["event_ledger"][0]["missing"] is True
    assert (
        missing["event_ledger"][0]["feedback"]["arm_results"]["adaptive_affine_gibbs"]["status"]
        == "missing"
    )
    with pytest.raises(ValueError, match="unregistered"):
        exp.replay_condition(
            states,
            stream,
            ordering="bad-order",
            condition={"name": "primary_delay_1", "delay": 1, "missing_fraction": 0.0},
            seed=exp.TRAINING_SEEDS[0],
            thresholds=thresholds,
            missing_group_ids=set(),
        )


def test_analytic_controls_cover_authority_restart_and_no_feedback() -> None:
    """SCENARIO-REPORT-7397-CONTROLS: all pre-measurement fixtures pass."""

    controls = exp.run_analytic_controls()
    expected = {
        "informative_energy",
        "constant_energy",
        "future_label_denied",
        "duplicate_feedback_denied",
        "restart_equality",
        "erased_update",
        "nonfinite_denied",
        "constant_prediction",
        "no_feedback_unchanged",
    }
    assert set(controls) == expected
    assert all(row["passed"] is True for row in controls.values())


def test_bootstrap_averages_seeds_inside_event_units() -> None:
    """SCENARIO-REPORT-7397-GATE: paired intervals use event units, not seeds."""

    rows: list[dict[str, object]] = []
    for index in range(12):
        for seed in exp.TRAINING_SEEDS[:2]:
            for arm, loss in (
                ("adaptive_affine_gibbs", 0.1),
                ("static_affine_gibbs", 0.2),
                ("online_logistic_raw_features", 0.3),
                ("recent_frequency_beta_binomial", 0.4),
            ):
                rows.append(
                    {
                        "ordering": "fixed_hash_order",
                        "feedback_condition": "primary_delay_1",
                        "stream_index": index,
                        "group_id": f"g-{index}",
                        "seed": seed,
                        "arm": arm,
                        "brier_loss": loss,
                    }
                )
    intervals = exp.paired_moving_block_intervals(rows, draws=200, block_lengths=(4, 6))
    assert len(intervals) == 6
    assert all(row["effective_event_groups"] == 12 for row in intervals)
    assert all(
        row["seed_reduction"] == "average_within_event_before_resampling" for row in intervals
    )
    assert all(row["ci95_upper"] < 0.0 for row in intervals)
    with pytest.raises(ValueError, match="require primary"):
        exp.paired_moving_block_intervals([])
    with pytest.raises(ValueError, match="positive"):
        exp.paired_moving_block_intervals(rows, draws=2, block_lengths=(0,))
    unmatched = [
        row
        for row in rows
        if not (row["arm"] == exp.PRACTICAL_CONTROLS[0] and row["group_id"] == "g-0")
    ]
    with pytest.raises(ValueError, match="identical"):
        exp.paired_moving_block_intervals(unmatched, draws=2, block_lengths=(4,))


def test_condition_reports_cover_actions_costs_and_empty_percentile() -> None:
    """REQ-REPORT-7397: every condition reports risk, churn, and latency."""

    rows: list[dict[str, object]] = []
    for ordering in exp.ORDERINGS:
        for condition in exp.feedback_conditions():
            for arm in exp.ARMS:
                for index, decision in enumerate(("accept", "reject", "escalate")):
                    rows.append(
                        {
                            "ordering": ordering,
                            "feedback_condition": condition["name"],
                            "arm": arm,
                            "group_id": f"{ordering}-{condition['name']}-{arm}-{index}",
                            "label": index % 2,
                            "brier_loss": 0.1 + index * 0.01,
                            "log_loss": 0.2 + index * 0.01,
                            "decision": decision,
                            "churn_contribution": int(index > 0),
                            "prediction_latency_s": 0.001,
                            "update_latency_s": 0.002,
                            "total_latency_s": 0.003,
                        }
                    )
    reports = exp.condition_reports(rows)
    assert len(reports) == len(exp.ORDERINGS) * len(exp.feedback_conditions())
    assert all(row["arm_metrics"][exp.ARMS[0]]["accept_count"] == 1 for row in reports)
    assert exp._percentile([], 0.5) == 0.0


def test_artifact_validator_detects_mutations_and_blocked_shape() -> None:
    """SCENARIO-REPORT-7397-ARTIFACT: cold reduction rejects changed evidence."""

    artifact = exp.build_fixture_artifact()
    assert exp.validate_artifact(artifact) == []
    assert artifact["MODEL_SPECS"] == []
    assert artifact["inference_substrate_class"] == "no_model_load"
    assert isinstance(artifact["inference_substrate"], str)
    assert artifact["execution_venue"] == "host"
    assert artifact["promotion_score"] == 0
    assert artifact["verifier_is_oracle"] is False

    changed = deepcopy(artifact)
    changed["rows"][0]["brier_loss"] = 0.9
    assert "reproducibility_checksum_mismatch" in exp.validate_artifact(changed)

    changed = deepcopy(artifact)
    changed["delayed_adapter_ready_score"] = 0
    assert "independent_reduction_mismatch" in exp.validate_artifact(changed)

    blocked = exp.build_blocked_artifact(
        [
            {
                "check": "fixture",
                "upstream": "missing.json",
                "artifact_field": "status",
                "expected": "complete",
                "observed": None,
                "passed": False,
            }
        ],
        {},
    )
    assert blocked["verdict_class"] == "blocked"
    assert blocked["rows"] == []
    assert blocked["gate_check_summary"]["first_required_failure"]["observed"] is None
    assert exp.validate_artifact(blocked) == []

    assert exp.validate_artifact([]) == ["artifact_not_object"]
    for field, value, expected in (
        ("schema", "bad", "identity_mismatch"),
        ("MODEL_SPECS", ["bad"], "substrate_declaration_mismatch"),
        ("verifier_is_oracle", True, "oracle_declaration_mismatch"),
        ("promotion_score", 1, "promotion_nonzero"),
        ("verdict_class", "bad", "verdict_class_invalid"),
        ("field_principles", {}, "field_principles_incomplete"),
    ):
        changed = deepcopy(artifact)
        changed[field] = value
        assert expected in exp.validate_artifact(changed)

    changed = deepcopy(artifact)
    changed["independent_reduction"]["row_completeness_passed"] = False
    assert "independent_reduction_mismatch" in exp.validate_artifact(changed)
    changed = deepcopy(artifact)
    changed["delayed_adapter_value_score"] = 1
    assert "independent_reduction_mismatch" in exp.validate_artifact(changed)
    changed = deepcopy(artifact)
    changed["flagged_adversarial"] = True
    changed["delayed_adapter_ready_score"] = 1
    assert "adversarial_scores_nonzero" in exp.validate_artifact(changed)

    malformed_blocked = deepcopy(blocked)
    malformed_blocked["rows"] = [{}]
    malformed_blocked["gate_check_summary"]["first_required_failure"] = None
    malformed_blocked["delayed_adapter_ready_score"] = 1
    errors = exp.validate_artifact(malformed_blocked)
    assert "blocked_artifact_has_dependent_work" in errors
    assert "blocked_gate_summary_missing" in errors
    assert "blocked_scores_nonzero" in errors


def test_primary_reducer_keeps_readiness_separate_from_efficacy() -> None:
    """REQ-REPORT-7397: a failed benefit conjunction stays a valid null."""

    intervals = [
        {
            "ordering": "fixed_hash_order",
            "feedback_condition": "primary_delay_1",
            "block_length": 32,
            "control_arm": control,
            "ci95_upper": -0.01,
        }
        for control in exp.PRACTICAL_CONTROLS
    ]
    report = {
        "ordering": "fixed_hash_order",
        "feedback_condition": "primary_delay_1",
        "arm_metrics": {
            arm: {
                "mean_log_loss": 0.2 if arm == "adaptive_affine_gibbs" else 0.3,
                "coverage": 0.5,
                "accept_risk": 0.0,
                "reject_risk": 0.0,
                "accept_count": 2,
                "reject_count": 2,
            }
            for arm in exp.ARMS
        },
    }
    passed = exp.reduce_primary_value(intervals, [report])
    assert passed["passed"] is True
    failed_intervals = deepcopy(intervals)
    failed_intervals[0]["ci95_upper"] = 0.0
    failed = exp.reduce_primary_value(failed_intervals, [report])
    assert failed["passed"] is False
    absent = exp.reduce_primary_value(intervals, [])
    assert absent["passed"] is False
    assert exp._gate("ge", "completion", 1, 2, ">=")["passed"] is True
    assert exp._gate("le", "completion", 2, 1, "<=")["passed"] is True


def test_atomic_helpers_membership_and_checkpoint_reconstruction(tmp_path) -> None:
    """SCENARIO-REPORT-7397-ARTIFACT: file and checkpoint identities are stable."""

    path = tmp_path / "nested" / "value.json"
    exp.atomic_json(path, {"value": 1})
    assert json.loads(path.read_text()) == {"value": 1}
    assert exp.sha256_file(path).startswith("sha256:")
    protocol = _protocol()
    assert exp._online_membership_hash(protocol).startswith("sha256:")

    states = exp.initialize_seed_states(
        exp.initialization_rows(protocol), exp.TRAINING_SEEDS[0], steps=1
    )
    record = states["training_record"]
    rebuilt = exp._states_from_record(record, [0, 1, 0, 1])
    assert set(rebuilt) == set(exp.ARMS)
