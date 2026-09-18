"""Tests for the V648 delayed-feedback decision replay.

Spec refs: REQ-AUTO-7386 and SCENARIO-AUTO-7386-01 through
SCENARIO-AUTO-7386-05.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from carnot import experiment_7386_v648_online_decisions as exp


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _rows(count: int = 16) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index in range(count):
        rows.append(
            {
                "source_row_index": index,
                "group_id": f"g{index:02d}",
                "partition": "training",
                "entity_uptake": (index % 5) / 4.0,
                "falsifiability_score": ((index * 3) % 7) / 6.0,
                "label": int(index in {2, 7, 11, 14}),
            }
        )
    return rows


def _protocol(rows: list[dict[str, object]]) -> dict[str, object]:
    midpoint = len(rows) // 2
    return {
        "status": "complete_decision_protocol_ready",
        "verdict_class": "null",
        "flagged_adversarial": False,
        "decision_protocol_ready_score": 1,
        "gate_check_summary": {"all_required_passed": True},
        "protocol_manifest": {
            "partition_membership_sha256": exp.EXPECTED_PARTITION_SHA256,
            "source_hashes": {exp.CORPUS_PATH.as_posix(): exp.EXPECTED_CORPUS_SHA256},
        },
        "online_replay": {
            "initialization_group_ids": [row["group_id"] for row in rows[:midpoint]],
            "later_group_ids": [row["group_id"] for row in rows[midpoint:]],
            "ordering_salt": "v648-calibration-7382-online-order",
            "block_length": 32,
            "sensitivity_block_length": 64,
            "moving_block_draws": 10_000,
            "random_seed": 7_386_307,
        },
        "feature_rows": rows,
    }


def test_protocol_gate_checks_terminal_class_adversarial_flag_and_stream() -> None:
    """SCENARIO-AUTO-7386-01 authenticates every upstream readiness boundary."""

    root = Path(__file__).resolve().parents[2]
    path = root / exp.PROTOCOL_PATH
    protocol = json.loads(path.read_text(encoding="utf-8"))
    observed = {
        exp.PROTOCOL_PATH.as_posix(): _sha256(path),
        exp.CORPUS_PATH.as_posix(): _sha256(root / exp.CORPUS_PATH),
    }
    assert exp.authenticate_protocol(protocol, observed) == []
    json.dumps(exp._protocol_gate_rows(protocol, observed))

    for field, value in (
        ("status", "partial"),
        ("verdict_class", "blocked"),
        ("flagged_adversarial", True),
        ("decision_protocol_ready_score", 0),
    ):
        changed = deepcopy(protocol)
        changed[field] = value
        assert any(field in error for error in exp.authenticate_protocol(changed, observed))

    changed = deepcopy(protocol)
    changed["online_replay"]["later_group_ids"][0] = "changed"
    assert any(
        "online_membership" in error for error in exp.authenticate_protocol(changed, observed)
    )
    changed_hashes = dict(observed)
    changed_hashes[exp.PROTOCOL_PATH.as_posix()] = "sha256:" + "0" * 64
    assert any(
        "artifact_sha256" in error for error in exp.authenticate_protocol(protocol, changed_hashes)
    )


def test_streams_use_only_sealed_membership_and_construct_quantile_blocks() -> None:
    """REQ-AUTO-7386 keeps membership fixed while constructing a stress order."""

    rows = _rows()
    streams = exp.build_streams(_protocol(rows))
    assert [row["group_id"] for row in streams["fixed_hash_order"]] == [
        row["group_id"] for row in rows[8:]
    ]
    stress = streams["feature_quantile_four_block"]
    assert {row["group_id"] for row in stress} == {row["group_id"] for row in rows[8:]}
    assert {row["group_id"]: row["label"] for row in stress} == {
        row["group_id"]: row["label"] for row in rows[8:]
    }
    assert [row["quantile_block"] for row in stress] == sorted(
        row["quantile_block"] for row in stress
    )

    broken = _protocol(rows)
    broken["feature_rows"] = rows[:-1]
    with pytest.raises(ValueError, match="sealed later group"):
        exp.build_streams(broken)


def test_replay_predicts_before_feedback_and_bounds_updates() -> None:
    """SCENARIO-AUTO-7386-02 enforces causal prediction and bounded state."""

    rows = _rows(24)
    initial = rows[:12]
    stream = rows[12:]
    states = exp.initialize_heads(initial, exp.TRAINING_SEEDS[0], steps=3)
    replay = exp.replay_condition(
        states,
        stream,
        ordering="fixed_hash_order",
        delay=8,
        seed=exp.TRAINING_SEEDS[0],
    )
    assert len(replay["rows"]) == len(stream) * len(exp.ONLINE_ARMS)
    assert {row["arm"] for row in replay["rows"]} == set(exp.ONLINE_ARMS)
    assert all(row["prediction_before_feedback"] is True for row in replay["rows"])
    assert all(row["replay_buffer_size"] <= 128 for row in replay["rows"])
    online = [row for row in replay["rows"] if row["arm"] == "bounded_online_gibbs"]
    assert all(row["update_count"] == 0 for row in online[:8])
    assert online[-1]["update_count"] > 0
    assert all(
        row["update_count"] == 0
        for row in replay["rows"]
        if row["arm"] in {"frozen_initialized_gibbs", "no_feedback_gibbs_control"}
    )
    omitted = [row for row in replay["feedback_ledger"] if row["omitted"]]
    assert {row["stream_index"] for row in omitted} == {3, 7, 11}
    assert all(row["admitted_update"] is False for row in omitted)
    assert replay["restart_receipt"]["prediction_parity"] is True

    immediate = exp.replay_condition(
        states,
        stream[:4],
        ordering="fixed_hash_order",
        delay=0,
        seed=exp.TRAINING_SEEDS[0],
        revoked_group_ids={stream[0]["group_id"]},
    )
    revoked = next(row for row in immediate["feedback_ledger"] if row["revoked"])
    assert revoked["admitted_update"] is False
    assert immediate["rows"][0]["update_count"] == 0


def test_durable_restart_corruption_rollback_and_erasure_controls(tmp_path: Path) -> None:
    """SCENARIO-AUTO-7386-03 and -04 preserve and remove exact updates."""

    rows = _rows(24)
    states = exp.initialize_heads(rows[:12], exp.TRAINING_SEEDS[1], steps=2)
    controls = exp.run_development_controls(states, rows[12:])
    assert controls["cold_restart"]["passed"] is True
    assert controls["corrupt_state_rollback"]["passed"] is True
    assert controls["corrupt_state_rollback"]["update_admitted"] is False
    assert controls["revoked_label_update"]["passed"] is True
    assert controls["revoked_label_update"]["state_unchanged"] is True

    replay = exp.replay_condition(
        states,
        rows[12:],
        ordering="feature_quantile_four_block",
        delay=0,
        seed=exp.TRAINING_SEEDS[1],
        checkpoint_path=tmp_path / "midpoint.json",
    )
    assert len(replay["erasure_rows"]) == len(rows[12:])
    assert len(replay["permutation_rows"]) == len(rows[12:])
    assert replay["permutation_update_count"] == replay["true_feedback_update_count"]
    assert all(row["same_group"] is True for row in replay["erasure_rows"])
    assert any(
        row["actual_probability"] != row["erased_probability"] for row in replay["erasure_rows"][1:]
    )
    assert replay["restart_receipt"]["checkpoint_sha256"] == exp.sha256_file(
        tmp_path / "midpoint.json"
    )

    with pytest.raises(ValueError, match="unregistered replay condition"):
        exp.replay_condition(states, rows[12:], ordering="unknown", delay=0, seed=1)
    bad = deepcopy(states["bounded_online_gibbs"])
    bad["arm"] = "unknown"
    with pytest.raises(ValueError, match="unknown durable arm"):
        exp._validate_state(bad)
    bad = deepcopy(states["bounded_online_gibbs"])
    bad["update_count"] = -1
    with pytest.raises(ValueError, match="invalid update count"):
        exp._validate_state(bad)
    bad = deepcopy(states["bounded_online_gibbs"])
    bad["buffer"] = [{}] * 129
    with pytest.raises(ValueError, match="exceeds bound"):
        exp._validate_state(bad)


def test_moving_block_intervals_are_paired_and_value_is_condition_local() -> None:
    """SCENARIO-AUTO-7386-04 and -05 keep paired conditions separate."""

    rows: list[dict[str, object]] = []
    for ordering in exp.ORDERINGS:
        for delay in exp.FEEDBACK_DELAYS:
            for seed in exp.TRAINING_SEEDS[:2]:
                for index in range(12):
                    label = int(index % 5 == 0)
                    for arm, probability in (
                        ("bounded_online_gibbs", 0.08 if label else 0.02),
                        ("frozen_initialized_gibbs", 0.2),
                        ("online_logistic_calibration", 0.18),
                    ):
                        rows.append(
                            {
                                "ordering": ordering,
                                "delay": delay,
                                "seed": seed,
                                "stream_index": index,
                                "arm": arm,
                                "label": label,
                                "probability": probability,
                                "brier_loss": (probability - label) ** 2,
                                "decision": "accept" if probability <= 0.05 else "escalate",
                            }
                        )
    intervals = exp.paired_moving_block_intervals(rows, draws=200, block_lengths=(3, 4))
    assert len(intervals) == len(exp.ORDERINGS) * len(exp.FEEDBACK_DELAYS) * 2 * 2
    assert all(row["paired_identical_block_indices"] is True for row in intervals)
    assert all(row["descriptive_fixed_archive_only"] is True for row in intervals)

    erasures = [
        {
            "ordering": row["ordering"],
            "delay": row["delay"],
            "seed": row["seed"],
            "stream_index": row["stream_index"],
            "erased_brier_loss": row["brier_loss"] + 0.01,
        }
        for row in rows
        if row["arm"] == "bounded_online_gibbs"
    ]
    permutations = [
        {
            "ordering": row["ordering"],
            "delay": row["delay"],
            "seed": row["seed"],
            "stream_index": row["stream_index"],
            "brier_loss": row["brier_loss"] + 0.02,
        }
        for row in rows
        if row["arm"] == "bounded_online_gibbs"
    ]
    block32 = exp.paired_moving_block_intervals(rows, draws=20, block_lengths=(32,))
    reports = exp._condition_reports(rows, erasures, permutations, block32)
    assert len(reports) == len(exp.ORDERINGS) * len(exp.FEEDBACK_DELAYS)
    assert all(report["iid_policy_certificate"] is False for report in reports)

    with pytest.raises(ValueError, match="requires condition rows"):
        exp.paired_moving_block_intervals([], draws=2, block_lengths=(2,))
    mismatched = [
        row
        for row in deepcopy(rows)
        if not (
            row["ordering"] == exp.ORDERINGS[-1]
            and row["delay"] == exp.FEEDBACK_DELAYS[-1]
            and row["arm"] == "online_logistic_calibration"
            and row["stream_index"] == 11
        )
    ]
    with pytest.raises(ValueError, match="identical group indices"):
        exp.paired_moving_block_intervals(mismatched, draws=2, block_lengths=(2,))

    condition_rows = [
        {
            "ordering": ordering,
            "delay": delay,
            "later_brier_ci95_upper_delta_vs_frozen": -0.01,
            "later_brier_ci95_upper_delta_vs_online_logistic": -0.001,
            "incorrect_accept_delta_at_matched_coverage": 0,
            "true_feedback_benefit": 0.02,
            "erased_feedback_benefit": 0.0,
            "permutation_benefit": 0.0,
            "decision_change_count": 1,
        }
        for ordering in exp.ORDERINGS
        for delay in exp.FEEDBACK_DELAYS
    ]
    assert exp.reduce_online_value(condition_rows)["passed"] is True
    condition_rows[0]["decision_change_count"] = 0
    reduced = exp.reduce_online_value(condition_rows)
    assert reduced["passed"] is False
    assert reduced["online_learning_value_score"] == 0


def test_blocked_and_complete_null_artifacts_preserve_terminal_semantics(tmp_path: Path) -> None:
    """SCENARIO-AUTO-7386-01 and -05 keep readiness separate from efficacy."""

    blocked = exp.build_blocked_artifact(
        [
            {
                "check": "exp7382:verdict_class",
                "upstream": exp.PROTOCOL_PATH.as_posix(),
                "artifact_field": "verdict_class",
                "expected": ["positive", "circular_positive", "null"],
                "observed": "partial",
                "passed": False,
            }
        ],
        {exp.PROTOCOL_PATH.as_posix(): "sha256:fixture"},
    )
    assert blocked["verdict_class"] == "blocked"
    assert blocked["online_capture_complete_score"] == 0
    assert blocked["gate_check_summary"]["first_required_failure"]["observed"] == "partial"
    assert exp.validate_artifact(blocked) == []

    payload_path = tmp_path / "atomic.json"
    exp.atomic_json(payload_path, {"value": 1})
    assert json.loads(payload_path.read_text(encoding="utf-8")) == {"value": 1}
    assert exp.sha256_file(payload_path).startswith("sha256:")

    artifact = exp.build_fixture_artifact()
    assert artifact["status"] == "complete_online_decisions_null"
    assert artifact["online_capture_complete_score"] == 1
    assert artifact["online_learning_value_score"] == 0
    assert artifact["promotion_score"] == 0
    assert artifact["inference_substrate_class"] == "no_model_load"
    assert artifact["inference_substrate"]["backend"] == "cpu_jax"
    assert artifact["execution_venue"] == "host"
    assert artifact["MODEL_SPECS"] == []
    assert artifact["model_invoked"] is False
    assert artifact["continuous_self_learning_task"] is True
    assert set(artifact["field_principles"]) == set(artifact)
    assert exp.validate_artifact(artifact) == []

    changed = deepcopy(artifact)
    changed["rows"] = []
    assert "row_completeness_mismatch" in exp.validate_artifact(changed)
    changed = deepcopy(artifact)
    changed["execution_venue"] = "host_cpu"
    assert "substrate_declaration_mismatch" in exp.validate_artifact(changed)
    changed = deepcopy(artifact)
    changed["online_learning_value_score"] = 1
    assert "stored_reduction_mismatch" in exp.validate_artifact(changed)
    assert exp.validate_artifact([]) == ["artifact_not_object"]
    changed = deepcopy(artifact)
    changed["schema"] = "wrong"
    assert "identity_mismatch" in exp.validate_artifact(changed)
    changed = deepcopy(artifact)
    changed["field_principles"] = {}
    assert "field_principles_incomplete" in exp.validate_artifact(changed)
    changed = deepcopy(blocked)
    changed["rows"] = [{}]
    changed["gate_check_summary"]["first_required_failure"] = None
    changed["online_capture_complete_score"] = 1
    errors = exp.validate_artifact(changed)
    assert "blocked_artifact_has_dependent_work" in errors
    assert "blocked_gate_summary_missing" in errors
    assert "blocked_scores_nonzero" in errors
    changed = deepcopy(artifact)
    changed["promotion_score"] = 1
    assert "promotion_nonzero" in exp.validate_artifact(changed)

    gates = [
        exp._gate("minimum", "completion", 2, 3, ">="),
        exp._gate("value", "scientific_efficacy", True, False),
    ]
    summary = exp._gate_summary(gates)
    assert gates[0]["passed"] is True
    assert summary["failed_required_count"] == 0
    assert summary["failed_scientific_gate_count"] == 1
