"""Tests for REQ-AUTO-7454 and SCENARIO-AUTO-7454-*.

The fixtures use small numeric streams. They test the new evidence reducer and
leave the unchanged V652 scientific formulas to their existing shared tests.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_7454_v653_continuous_learning as exp


ROOT = Path(__file__).resolve().parents[2]


def _passing_receipts() -> list[dict[str, Any]]:
    """Build one passing receipt for each frozen affected and terminal check."""

    return [
        {"name": name, "required": True, "passed": True, "exit_code": 0}
        for name in (*exp.AFFECTED_CHECK_NAMES, *exp.TERMINAL_CHECK_NAMES)
    ]


def _fixture(tmp_path: Path, *, groups: int = 40) -> dict[str, Any]:
    """Capture one complete cell with durable label-free prediction events."""

    return exp.build_fixture_evidence(
        tmp_path,
        relative_dir=Path("raw"),
        group_count=groups,
        ordering="hash_order",
        delay=8,
        seed=exp.TRAINING_SEEDS[0],
    )


def test_req_auto_7454_spec_preconditions_and_protocol_are_exact() -> None:
    """REQ-AUTO-7454 authenticates prerequisites and keeps V652 frozen."""

    text = (ROOT / exp.SPEC_PATH).read_text(encoding="utf-8")
    assert "### REQ-AUTO-7454:" in text
    for number in range(1, 7):
        assert f"SCENARIO-AUTO-7454-{number:02d}" in text

    checks, hashes, loaded = exp.collect_preconditions(ROOT)
    assert checks and all(row["passed"] for row in checks)
    assert loaded["ledger"]["prediction_ledger_ready_score"] == 1
    assert loaded["ledger"]["verdict_class"] == "circular_positive"
    assert loaded["ledger"]["flagged_adversarial"] is False
    assert loaded["decisions"]["decision_capture_complete_score"] == 1
    assert hashes[exp.LEDGER_PATH.as_posix()]["original_verdict_class"] == "circular_positive"

    protocol = exp.frozen_protocol(loaded)
    assert protocol["source_group_count"] == 753
    assert protocol["training_seeds"] == [65201, 65202, 65203, 65204, 65205]
    assert protocol["orders"] == ["hash_order", "domain_blocked_shift_order"]
    assert protocol["delays"] == [0, 8]
    assert len(protocol["arms"]) == 7
    assert protocol["moving_block_lengths"] == [32, 64]
    assert protocol["bootstrap_draws"] == 10_000
    assert protocol["reused_corpus"] is True
    assert protocol["fresh_deployment_evidence"] is False


def test_scenario_auto_7454_01_predictions_precede_outcomes(tmp_path: Path) -> None:
    """SCENARIO-AUTO-7454-01 keeps labels outside durable prediction bytes."""

    evidence = _fixture(tmp_path)
    predictions = exp.load_event_shards(tmp_path, evidence["prediction_event_shards"])
    outcomes = exp.load_event_shards(tmp_path, evidence["outcome_event_shards"])
    first = predictions[0]
    assert set(first["expert_probabilities"]) == set(exp.EXPERT_NAMES)
    assert set(first["mixture_weights"]) == set(exp.EXPERT_NAMES)
    assert not ({"label", "loss", "brier", "feedback_label"} & set(first))
    assert exp.prediction_event_hash(first) == first["event_hash"]
    assert outcomes[0]["ledger_sequence"] > max(
        row["ledger_sequence"]
        for row in predictions
        if row["group_id"] == outcomes[0]["group_id"]
    )
    assert set(outcomes[0]["prediction_event_hashes"].values()) <= {
        row["event_hash"] for row in predictions
    }
    assert all(row["durable_acknowledged"] for row in predictions)


def test_scenario_auto_7454_02_saved_losses_replay_without_producer_math(
    tmp_path: Path,
) -> None:
    """SCENARIO-AUTO-7454-02 independently rebuilds each stored weight edge."""

    evidence = _fixture(tmp_path)
    reduced = exp.independent_reduce_evidence(evidence, root=tmp_path, bootstrap_draws=40)
    assert reduced["causal_capture_valid"] is True
    assert reduced["expert_loss_max_abs_error"] <= 1e-12
    assert reduced["numeric_update_max_abs_error"] <= 1e-12
    assert reduced["prediction_event_count"] == 40 * len(exp.ARMS)
    assert reduced["feedback_event_count"] == 10 * len(exp.UPDATING_ARMS)
    assert reduced["checkpoint_lineage_valid"] is True

    rows = exp.load_event_shards(tmp_path, evidence["feedback_event_shards"])
    changed = deepcopy(rows[0])
    changed["per_expert_loss"][exp.EXPERT_NAMES[0]] += 0.25
    changed["event_hash"] = exp.feedback_event_hash(changed)
    with pytest.raises(ValueError, match="expert_loss_mismatch"):
        exp.replay_feedback_rows(
            [changed],
            initial_state=exp.initial_audit_state(changed["arm"]),
            prediction_by_hash={
                row["event_hash"]: row
                for row in exp.load_event_shards(
                    tmp_path, evidence["prediction_event_shards"]
                )
            },
        )


def test_scenario_auto_7454_03_feedback_restart_revocation_and_no_feedback(
    tmp_path: Path,
) -> None:
    """SCENARIO-AUTO-7454-03 covers duplicate, revoke, restart, and erasure."""

    evidence = _fixture(tmp_path)
    reduced = exp.independent_reduce_evidence(evidence, root=tmp_path, bootstrap_draws=40)
    controls = reduced["feedback_handling_controls"]
    assert controls == {
        "duplicate_feedback_rejected": True,
        "revoked_feedback_replayed": True,
        "restart_replay_equal": True,
        "no_feedback_state_unchanged": True,
    }

    predictions = exp.load_event_shards(tmp_path, evidence["prediction_event_shards"])
    feedback = exp.load_event_shards(tmp_path, evidence["feedback_event_shards"])
    learned = [row for row in feedback if row["arm"] == exp.LEARNED_MIXTURE]
    prediction_by_hash = {row["event_hash"]: row for row in predictions}
    with pytest.raises(ValueError, match="duplicate_feedback"):
        exp.replay_feedback_rows(
            [learned[0], deepcopy(learned[0])],
            initial_state=exp.initial_audit_state(exp.LEARNED_MIXTURE),
            prediction_by_hash=prediction_by_hash,
        )


def test_scenario_auto_7454_04_later_query_causality_and_checkpoints(
    tmp_path: Path,
) -> None:
    """SCENARIO-AUTO-7454-04 binds earlier updates to later prediction state."""

    evidence = _fixture(tmp_path, groups=70)
    reduced = exp.independent_reduce_evidence(evidence, root=tmp_path, bootstrap_draws=40)
    assert reduced["later_query_causality"] is True
    assert reduced["checkpoint_lineage_valid"] is True
    checkpoints = exp.load_event_shards(tmp_path, evidence["checkpoint_event_shards"])
    learned_positions = [
        row["completed_groups"]
        for row in checkpoints
        if row["arm"] == exp.LEARNED_MIXTURE
    ]
    assert learned_positions == [32, 64, 70]
    assert all(exp.checkpoint_event_hash(row) == row["event_hash"] for row in checkpoints)


def test_scenario_auto_7454_05_registered_reduction_reuses_v652_bars() -> None:
    """SCENARIO-AUTO-7454-05 retains the 12-contrast conjunctive family."""

    rows = exp.v652.synthetic_metric_rows(groups=96, seeds=5)
    intervals = exp.v652.paired_moving_block_intervals(rows, draws=100, seed=7454)
    reports, passing_intervals, controls = exp.v652.synthetic_reduction_inputs(passing=True)
    positive = exp.v652.reduce_online_value(reports, passing_intervals, controls)
    assert len(intervals) == 24
    assert {row["block_length"] for row in intervals} == {32, 64}
    assert all(row["fit_seeds_averaged_before_resampling"] == 5 for row in intervals)
    assert positive["online_value_score"] == 1
    changed = deepcopy(passing_intervals)
    changed[0]["upper"] = 0.0
    null = exp.v652.reduce_online_value(reports, changed, controls)
    assert null["online_capture_complete_score"] == 1
    assert null["online_value_score"] == 0


def test_scenario_auto_7454_06_terminal_fixture_is_complete_null(
    tmp_path: Path,
) -> None:
    """SCENARIO-AUTO-7454-06 keeps valid capture distinct from online value."""

    artifact = exp.build_fixture_artifact(
        tmp_path,
        validation_receipts=_passing_receipts(),
        relative_dir=Path("fixture-artifact"),
    )
    assert artifact["MODEL_SPECS"] == []
    assert artifact["model_invoked"] is False
    assert artifact["invocation_counts"] == exp.ZERO_INVOCATION_COUNTS
    assert artifact["inference_substrate_class"] == "no_model_load"
    assert artifact["execution_venue"] == "host"
    assert artifact["online_capture_complete_score"] == 1
    assert artifact["online_value_score"] == 0
    assert artifact["verdict_class"] == "null"
    assert artifact["honest_verdict"] == "complete_null_insufficient_online_benefit"
    assert artifact["mixture_construction_retired"] is True
    assert artifact["continuous_self_learning_task"] is True
    assert artifact["no_model_weight_mutation"] is True
    assert artifact["promotion_score"] == 0
    assert artifact["verifier_is_oracle"] is False
    assert set(artifact["stage_timing_rows"]["stages"]) == set(exp.SERVICE_STAGES)
    assert exp.validate_artifact(artifact, root=tmp_path) == []

    changed = deepcopy(artifact)
    changed["no_model_weight_mutation"] = False
    assert "model_weight_mutation_invalid" in exp.validate_artifact(changed, root=tmp_path)


def test_req_auto_7454_shard_hash_and_event_hash_mutations_fail_closed(
    tmp_path: Path,
) -> None:
    """REQ-AUTO-7454 rejects changed durable bytes and changed prediction values."""

    evidence = _fixture(tmp_path)
    shard = tmp_path / evidence["prediction_event_shards"][0]["path"]
    with shard.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"changed": True}) + "\n")
    with pytest.raises(ValueError, match="event_shard_invalid"):
        exp.independent_reduce_evidence(evidence, root=tmp_path, bootstrap_draws=20)

    event = {
        "schema": exp.PREDICTION_SCHEMA,
        "event_id": "event",
        "group_id": "group",
        "row_key": "row",
        "task_type": "QA",
        "arm": exp.LEARNED_MIXTURE,
        "ordering": "hash_order",
        "delay": 0,
        "seed": exp.TRAINING_SEEDS[0],
        "prediction_index": 0,
        "request_order": 0,
        "ledger_sequence": 0,
        "expert_probabilities": {name: 0.5 for name in exp.EXPERT_NAMES},
        "mixture_weights": {name: 0.25 for name in exp.EXPERT_NAMES},
        "mixture_probability": 0.5,
        "expert_checkpoint_hashes": {name: "sha256:x" for name in exp.EXPERT_NAMES},
        "label_propensity": 0.25,
        "revealed": True,
        "pre_feedback_state_hash": "sha256:state",
        "proposed_action": "escalate",
        "deployed_action": "escalate",
        "shadow_only": True,
        "certified_safe": False,
        "domain_changed": False,
        "durable_acknowledged": True,
        "predict_duration_ns": 1,
        "hash_duration_ns": 1,
        "event_hash": "wrong",
    }
    with pytest.raises(ValueError, match="prediction_event_hash_mismatch"):
        exp.validate_prediction_event(event)


def test_req_auto_7454_blocked_result_and_cli_modes_are_strict(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """REQ-AUTO-7454 preserves exact blockers and fresh-process reader modes."""

    failed = {
        "check": "prediction_ledger_ready",
        "upstream": "exp7450-prediction-ledger",
        "path": exp.LEDGER_PATH.as_posix(),
        "field": "prediction_ledger_ready_score",
        "operator": "==",
        "expected": 1,
        "observed": 0,
        "passed": False,
    }
    blocked = exp.build_blocked_artifact(failed)
    assert blocked["verdict_class"] == "blocked"
    assert blocked["honest_verdict"] == "blocked_prediction_ledger_ready"
    assert blocked["gate_check_summary"]["observed"] == 0
    assert blocked["gate_check_summary"]["path"] == exp.LEDGER_PATH.as_posix()

    candidate = tmp_path / "candidate.json"
    candidate.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(exp, "validate_artifact", lambda value, root: [])
    assert exp.main(["--date", exp.RUN_DATE, "--root", str(tmp_path), "--cold-replay", str(candidate)]) == 0
    monkeypatch.setattr(exp, "independent_reduce", lambda value, root: {"capture": 1})
    assert exp.main(["--date", exp.RUN_DATE, "--root", str(tmp_path), "--independent-reduce", str(candidate)]) == 0
    monkeypatch.setattr(exp, "run_experiment", lambda root, run_date, output_path: {"ok": True})
    assert exp.main(["--date", exp.RUN_DATE, "--root", str(tmp_path)]) == 0
    with pytest.raises(SystemExit, match="--date"):
        exp.main(["--date", "wrong"])
    assert "capture" in capsys.readouterr().out
