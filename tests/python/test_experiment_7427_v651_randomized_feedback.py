"""Tests for REQ-AUTO-7427 randomized delayed-feedback adaptation."""

from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_7427_v651_randomized_feedback as exp


ROOT = Path(__file__).resolve().parents[2]


def _features(index: int) -> dict[str, float]:
    return {
        name: ((index + offset) % 11) / 10 for offset, name in enumerate(exp.SOURCE_FEATURE_NAMES)
    }


def _stream_row(index: int, label: int | None = None) -> dict[str, Any]:
    return {
        "row_key": f"row-{index:04d}",
        "group_id": f"group-{index:04d}",
        "partition": "prospective_stream",
        "task_type": ("QA", "Summary", "Data2txt")[index % 3],
        "features": _features(index),
        "label": index % 2 if label is None else label,
        "authority": exp.LABEL_AUTHORITY,
    }


def _passing_receipts() -> list[dict[str, Any]]:
    return [
        {
            "name": name,
            "passed": True,
            "required": True,
            "exit_code": 0,
            "timed_out": False,
        }
        for name in (*exp.AFFECTED_CHECK_NAMES, *exp.TERMINAL_CHECK_NAMES)
    ]


def test_req_auto_7427_spec_and_static_prerequisite() -> None:
    """REQ-AUTO-7427: the spec precedes code and the exact upstream is eligible."""

    text = (ROOT / exp.SPEC_PATH).read_text(encoding="utf-8")
    assert "### REQ-AUTO-7427:" in text
    for number in range(1, 9):
        assert f"SCENARIO-AUTO-7427-{number:02d}" in text
    checks, hashes, upstream = exp.collect_preconditions(ROOT)
    assert checks and all(row["passed"] for row in checks)
    assert hashes[exp.UPSTREAM_PATH.as_posix()]["original_flagged_adversarial"] is False
    assert upstream["decision_capture_complete_score"] == 1
    states = exp.load_initial_states(ROOT, upstream)
    assert len(states) == len(exp.TRAINING_SEEDS) * 3
    assert {row["arm"] for row in states} == {
        exp.ONLINE_SPLINE_ARM,
        exp.ONLINE_GIBBS_ARM,
        exp.ONLINE_LOGISTIC_ARM,
    }
    assert all(row["online_labels_seen"] == 0 for row in states)


def test_scenario_7427_01_stream_orders_are_grouped_and_label_blind() -> None:
    """SCENARIO-AUTO-7427-01: both orders seal the same independent groups."""

    rows = [_stream_row(index) for index in range(70)]
    streams = exp.build_streams(rows)
    assert set(streams) == set(exp.ORDERS)
    assert len(streams["hash_order"]) == 70
    assert streams["hash_order"] != streams["domain_blocked_shift_order"]
    expected = {row["group_id"] for row in rows}
    assert all({row["group_id"] for row in stream} == expected for stream in streams.values())
    changed = [{**row, "label": 1 - int(row["label"])} for row in rows]
    assert {
        name: [row["observation_id"] for row in stream]
        for name, stream in exp.build_streams(changed).items()
    } == {name: [row["observation_id"] for row in stream] for name, stream in streams.items()}
    with pytest.raises(ValueError, match="source group"):
        exp.build_streams([{**rows[0], "group_id": ""}])


def test_scenario_7427_02_shared_schedules_and_exact_propensities() -> None:
    """SCENARIO-AUTO-7427-02: selection uses frozen risks, never labels."""

    rows = exp.build_streams([_stream_row(index) for index in range(37)])["hash_order"]
    risks = {row["observation_id"]: index / 100 for index, row in enumerate(rows)}
    schedules = exp.build_reveal_schedules(rows, risks, seed=65101)
    full = rows[:32]
    tail = rows[32:]
    by_schedule = {
        name: {row["observation_id"]: row for row in selected}
        for name, selected in schedules.items()
    }
    assert (
        sum(by_schedule["top_risk_eight"][row["observation_id"]]["revealed"] for row in full) == 8
    )
    assert sum(by_schedule["uniform_eight"][row["observation_id"]]["revealed"] for row in full) == 8
    assert (
        sum(by_schedule["hybrid_four_plus_four"][row["observation_id"]]["revealed"] for row in full)
        == 8
    )
    assert {
        row["propensity"]
        for row in by_schedule["uniform_eight"].values()
        if row["block_size"] == 32
    } == {0.25}
    hybrid_full = [by_schedule["hybrid_four_plus_four"][row["observation_id"]] for row in full]
    assert {row["propensity"] for row in hybrid_full} == {1.0, 4 / 28}
    assert sum(by_schedule["uniform_eight"][row["observation_id"]]["revealed"] for row in tail) == 1
    assert all(
        row["propensity"] == 0.2
        for row in by_schedule["uniform_eight"].values()
        if row["block_size"] == 5
    )
    changed = [{**row, "label": 1 - int(row["label"])} for row in rows]
    assert exp.build_reveal_schedules(changed, risks, seed=65101) == schedules
    with pytest.raises(ValueError, match="risk"):
        exp.build_reveal_schedules(rows, {}, seed=65101)


def test_scenario_7427_03_prediction_precedes_clipped_single_update() -> None:
    """SCENARIO-AUTO-7427-03: one delayed label updates after its prediction."""

    checkpoint = json.loads(
        (
            ROOT / "results/raw/experiment_7426_v651_static_decisions/checkpoints/"
            "full_source--raw_l2_logistic--65101.json"
        ).read_text(encoding="utf-8")
    )
    learner = exp.OnlineLearner.from_checkpoint(checkpoint, exp.ONLINE_LOGISTIC_ARM)
    probability, decision = learner.predict(_features(1))
    assert 0 <= probability <= 1
    assert decision["action"] in {"accept", "reject", "escalate"}
    prediction = learner.record_prediction("event-1", _features(1), index=0, available_at=8)
    before = learner.state_hash
    assert prediction["prediction_before_feedback"] is True
    with pytest.raises(exp.FutureFeedbackError):
        learner.commit_feedback("event-1", 1, visible_at=7)
    committed = learner.commit_feedback("event-1", 1, visible_at=8)
    assert committed["parent_state_hash"] == before
    assert committed["state_hash"] != before
    assert committed["gradient_norm_after_clip"] <= exp.GRADIENT_CLIP
    assert committed["touched_coefficients"] <= 7
    duplicate = learner.commit_feedback("event-1", 1, visible_at=9)
    assert duplicate["status"] == "duplicate"
    assert learner.commit_feedback("unknown", 1, visible_at=9)["status"] == "unknown_event"
    with pytest.raises(ValueError, match="label"):
        learner.commit_feedback("event-1", 2, visible_at=9)
    with pytest.raises(ValueError, match="finite"):
        learner.record_prediction(
            "nan", {**_features(1), exp.SOURCE_FEATURE_NAMES[0]: math.nan}, index=1, available_at=1
        )


def test_scenario_7427_04_revocation_restart_and_corruption_rollback() -> None:
    """SCENARIO-AUTO-7427-04: trusted events reconstruct and bad descendants roll back."""

    controls, revocations = exp.run_analytic_controls(ROOT)
    assert controls and all(row["passed"] for row in controls.values())
    assert {row["operation"] for row in revocations} == {
        "replace_label",
        "erase_feedback",
        "rollback_corrupted_state",
    }
    assert all(row["trusted_journal_replayed"] for row in revocations)
    assert all(row["reconstruction_duration_s"] >= 0 for row in revocations)


def test_scenario_7427_05_ipw_support_and_biased_top_risk() -> None:
    """SCENARIO-AUTO-7427-05: IPW uses positive propensities and support fails closed."""

    rows = exp.synthetic_metric_rows(groups=96, seeds=5)
    reports = exp.condition_reports(rows)
    top = next(row for row in reports if row["schedule"] == "top_risk_eight")
    hybrid = next(row for row in reports if row["schedule"] == exp.PRIMARY_SCHEDULE)
    assert top["biased_zero_probability_omissions"] is True
    assert hybrid["ipw_log_loss"] >= 0
    support = {"independent_online_groups": 96, "label_counts": {"0": 39, "1": 57}}
    intervals = exp.paired_moving_block_intervals(rows, draws=200)
    reduced = exp.reduce_online_value(reports, intervals, support)
    assert reduced["support_passed"] is False
    assert reduced["online_value_score"] == 0
    assert reduced["terminal_verdict"] == "complete_null_insufficient_online_support"
    assert {row["block_length"] for row in intervals} == {32, 64}
    assert all(row["simultaneous_comparison_count"] == 8 for row in intervals)
    with pytest.raises(ValueError, match="positive"):
        exp.paired_moving_block_intervals(rows, draws=0)


def test_scenarios_7427_06_07_condition_preservation_and_controls(tmp_path: Path) -> None:
    """SCENARIO-AUTO-7427-06/07: conditions persist and no-feedback stays frozen."""

    states = exp.load_initial_states(ROOT, json.loads((ROOT / exp.UPSTREAM_PATH).read_text()))
    rows = exp.build_streams([_stream_row(index) for index in range(12)])["hash_order"]
    risks = exp.frozen_gibbs_risks(states, rows)
    schedules = exp.build_reveal_schedules(rows, risks, seed=65101)
    replay = exp.replay_condition(
        states,
        rows,
        schedules[exp.PRIMARY_SCHEDULE],
        ordering="hash_order",
        schedule=exp.PRIMARY_SCHEDULE,
        delay=0,
        journal_path=tmp_path / "journal.jsonl",
    )
    event_rows = replay["feedback_event_rows"]
    assert len(event_rows) == 12 * len(exp.TRAINING_SEEDS) * len(exp.ARMS)
    frozen = {
        (row["observation_id"], row["seed"]): row["probability"]
        for row in event_rows
        if row["arm"] == exp.FROZEN_SPLINE_ARM
    }
    no_feedback = {
        (row["observation_id"], row["seed"]): row["probability"]
        for row in event_rows
        if row["arm"] == exp.NO_FEEDBACK_ARM
    }
    assert frozen == no_feedback
    spline_state = next(
        row
        for row in states
        if row["arm"] == exp.ONLINE_SPLINE_ARM and row["seed"] == exp.TRAINING_SEEDS[0]
    )
    for source_row in rows:
        expected = exp._calibrated_probability(spline_state, exp._feature_vector(source_row))
        assert frozen[(source_row["observation_id"], exp.TRAINING_SEEDS[0])] == expected
    assert all(row["prediction_persisted"] for row in event_rows)
    assert all(row["prediction_before_feedback"] for row in event_rows)
    assert replay["pending_feedback_at_end"] == 0
    assert (tmp_path / "journal.jsonl").stat().st_size > 0
    with pytest.raises(ValueError, match="registered replay"):
        exp.replay_condition(
            states,
            rows,
            schedules[exp.PRIMARY_SCHEDULE],
            ordering="bad",
            schedule=exp.PRIMARY_SCHEDULE,
            delay=0,
        )


def test_scenario_7427_08_fixture_cold_reduction_and_mutations(tmp_path: Path) -> None:
    """SCENARIO-AUTO-7427-08: cold readers reject material replay drift."""

    artifact = exp.build_fixture_artifact(validation_receipts=_passing_receipts())
    assert exp.validate_artifact(artifact, root=ROOT) == []
    reduced = exp.independent_reduce(artifact, root=ROOT)
    assert reduced["online_capture_complete_score"] == 1
    assert reduced["online_value_score"] == 0
    assert artifact["promotion_score"] == 0
    assert artifact["continuous_self_learning_task"] is True
    assert artifact["hardware_path"]["kernel_only_used_as_end_to_end"] is False

    changed = deepcopy(artifact)
    changed["feedback_event_rows"][0]["probability"] = 2.0
    assert "event_probability_invalid:0" in exp.validate_artifact(changed, root=ROOT)
    changed = deepcopy(artifact)
    changed["reproducibility_checksum"] = "sha256:changed"
    assert "reproducibility_checksum_mismatch" in exp.validate_artifact(changed, root=ROOT)
    path = tmp_path / "candidate.json"
    path.write_text(json.dumps(artifact), encoding="utf-8")
    assert exp.cold_replay(path, root=ROOT) == []
    path.write_text("[]", encoding="utf-8")
    assert exp.cold_replay(path, root=ROOT) == ["artifact_unreadable_or_not_object"]


def test_blocked_artifact_and_cli_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-AUTO-7427: blocked fields and command modes retain exact semantics."""

    failed = {
        "check": "upstream_capture_complete",
        "upstream": "exp7426-static-decisions",
        "path": exp.UPSTREAM_PATH.as_posix(),
        "field": "decision_capture_complete_score",
        "operator": "==",
        "expected": 1,
        "observed": 0,
        "passed": False,
    }
    blocked = exp.build_blocked_artifact(failed)
    assert blocked["verdict_class"] == "blocked"
    assert blocked["honest_verdict"].startswith("blocked_")
    assert blocked["gate_check_summary"]["blocked_field"] == "decision_capture_complete_score"
    with pytest.raises(SystemExit, match="--date"):
        exp.parse_args(["--date", "20260918"])
    args = exp.parse_args(["--date", exp.RUN_DATE, "--cold-replay", "candidate.json"])
    assert args.cold_replay == Path("candidate.json")

    monkeypatch.setattr(exp, "cold_replay", lambda *_args, **_kwargs: [])
    assert exp.main(["--date", exp.RUN_DATE, "--cold-replay", "candidate.json"]) == 0
    monkeypatch.setattr(exp, "_load_object", lambda _path: {})
    assert exp.main(["--date", exp.RUN_DATE, "--independent-reduce", "candidate.json"]) == 1
    called: list[tuple[Path, str, Path]] = []
    monkeypatch.setattr(
        exp,
        "run_experiment",
        lambda root, date, *, output_path: called.append((root, date, output_path)),
    )
    assert exp.main(["--date", exp.RUN_DATE]) == 0
    assert called == [(exp.REPO_ROOT, exp.RUN_DATE, exp.RESULT_PATH)]


def test_defensive_checkpoint_learner_and_replay_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-AUTO-7427: malformed state, ordering, labels, and lineage fail closed."""

    upstream = json.loads((ROOT / exp.UPSTREAM_PATH).read_text(encoding="utf-8"))
    with pytest.raises(ValueError, match="manifest"):
        exp.load_initial_states(ROOT, {})
    changed = deepcopy(upstream)
    relevant = next(
        row
        for row in changed["checkpoint_manifest"]
        if row["condition"] == "full_source" and row["arm"] == "raw_l2_logistic"
    )
    relevant["sha256"] = "sha256:changed"
    with pytest.raises(ValueError, match="identity"):
        exp.load_initial_states(ROOT, changed)
    monkeypatch.setattr(exp, "_load_object", lambda _path: {"condition": "bad"})
    with pytest.raises(ValueError, match="content"):
        exp.load_initial_states(ROOT, upstream)
    monkeypatch.undo()
    missing = deepcopy(upstream)
    missing["checkpoint_manifest"] = [
        row
        for row in missing["checkpoint_manifest"]
        if not (
            row["condition"] == "full_source"
            and row["arm"] == "raw_l2_logistic"
            and row["seed"] == exp.TRAINING_SEEDS[0]
        )
    ]
    with pytest.raises(ValueError, match="fifteen"):
        exp.load_initial_states(ROOT, missing)

    states = exp.load_initial_states(ROOT, upstream)
    rows = exp.build_streams([_stream_row(index) for index in range(4)])["hash_order"]
    with pytest.raises(ValueError, match="features"):
        exp._feature_vector({"features": {}})
    with pytest.raises(ValueError, match="five frozen"):
        exp.frozen_gibbs_risks(
            [
                row
                for row in states
                if not (
                    row["arm"] == exp.ONLINE_GIBBS_ARM and row["seed"] == exp.TRAINING_SEEDS[-1]
                )
            ],
            rows,
        )
    risks = exp.frozen_gibbs_risks(states, rows)
    with pytest.raises(ValueError, match="block size"):
        exp.build_reveal_schedules(rows, risks, seed=1, block_size=0)

    payload = json.loads(
        (
            ROOT / "results/raw/experiment_7426_v651_static_decisions/checkpoints/"
            "full_source--raw_l2_logistic--65101.json"
        ).read_text(encoding="utf-8")
    )
    with pytest.raises(ValueError, match="online learner"):
        exp.OnlineLearner(
            arm="bad",
            seed=1,
            checkpoint={},
            calibration={},
            policy={},
        )
    with pytest.raises(ValueError, match="checkpoint arm"):
        exp.OnlineLearner.from_checkpoint(payload, exp.ONLINE_GIBBS_ARM)
    learner = exp.OnlineLearner.from_checkpoint(payload, exp.ONLINE_LOGISTIC_ARM)
    learner.record_prediction("a", _features(0), index=0, available_at=0)
    with pytest.raises(ValueError, match="new"):
        learner.record_prediction("a", _features(0), index=0, available_at=0)
    with pytest.raises(ValueError, match="availability"):
        learner.record_prediction("b", _features(0), index=1, available_at=0)
    with pytest.raises(ValueError, match="replacement"):
        learner.revoke_feedback("a", 2)
    assert learner.revoke_feedback("missing", None)["status"] == "unknown_event"

    schedules = exp.build_reveal_schedules(rows, risks, seed=1)
    with pytest.raises(ValueError, match="cover"):
        exp.replay_condition(
            states,
            rows,
            schedules[exp.PRIMARY_SCHEDULE][:-1],
            ordering="hash_order",
            schedule=exp.PRIMARY_SCHEDULE,
            delay=0,
        )
    bad_rows = deepcopy(rows)
    bad_rows[0]["label"] = None
    with pytest.raises(ValueError, match="binary evaluator"):
        exp.replay_condition(
            states,
            bad_rows,
            schedules[exp.PRIMARY_SCHEDULE],
            ordering="hash_order",
            schedule=exp.PRIMARY_SCHEDULE,
            delay=0,
        )
    delayed = exp.replay_condition(
        states,
        rows,
        schedules[exp.PRIMARY_SCHEDULE],
        ordering="hash_order",
        schedule=exp.PRIMARY_SCHEDULE,
        delay=8,
    )
    assert delayed["pending_feedback_at_end"] == 0
    with pytest.raises(ValueError, match="paired group"):
        exp._paired_group_deltas([], ordering="hash_order", delay=0, control=exp.FROZEN_SPLINE_ARM)


def test_sharded_rows_and_artifact_validation_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-AUTO-7427-08: sidecar counts, hashes, and schema fields are bound."""

    rows = exp.synthetic_metric_rows(groups=2, seeds=1)[:8]
    monkeypatch.setattr(exp, "MAX_SHARD_BYTES", 1200)
    directory = tmp_path / "rows"
    shards = exp.write_row_shards(directory, rows)
    assert len(shards) > 1
    reference = {
        "feedback_event_rows": {
            "directory": "rows",
            "row_count": len(rows),
            "shards": shards,
        }
    }
    assert exp._load_event_rows(reference, tmp_path) == rows
    with pytest.raises(ValueError, match="required"):
        exp._load_event_rows({}, tmp_path)
    bad = deepcopy(reference)
    bad["feedback_event_rows"]["shards"][0]["sha256"] = "sha256:changed"
    with pytest.raises(ValueError, match="hash"):
        exp._load_event_rows(bad, tmp_path)
    bad = deepcopy(reference)
    bad["feedback_event_rows"]["shards"][0]["rows"] += 1
    with pytest.raises(ValueError, match="row mismatch"):
        exp._load_event_rows(bad, tmp_path)
    bad = deepcopy(reference)
    bad["feedback_event_rows"]["row_count"] += 1
    with pytest.raises(ValueError, match="row count"):
        exp._load_event_rows(bad, tmp_path)

    checkpoint = {"coef": [0.1], "bias": 0.0}
    lineage_row = {
        "observation_id": "observation-1",
        "seed": 65101,
        "arm": exp.ONLINE_LOGISTIC_ARM,
        "ordering": "hash_order",
        "schedule": exp.PRIMARY_SCHEDULE,
        "delay": 0,
        "label": 1,
        "label_authority": exp.LABEL_AUTHORITY,
        "parent_state_hash": "sha256:parent",
        "arrival_index": 0,
        "update_count": 1,
        "checkpoint": checkpoint,
    }
    lineage_row["event_hash"] = exp.canonical_hash(
        {
            "observation_id": lineage_row["observation_id"],
            "label": 1,
            "authority": exp.LABEL_AUTHORITY,
            "arrival_index": 0,
        }
    )
    lineage_row["state_hash"] = exp.canonical_hash(
        {
            "arm": exp.ONLINE_LOGISTIC_ARM,
            "seed": 65101,
            "checkpoint": checkpoint,
            "update_count": 1,
        }
    )
    lineage_shards = exp.write_row_shards(tmp_path / "lineage", [lineage_row])
    lineage_value = {
        "checkpoint_lineage": {
            "directory": "lineage",
            "row_count": 1,
            "shards": lineage_shards,
        }
    }
    assert exp._load_lineage_rows(lineage_value, tmp_path) == [lineage_row]
    assert exp._lineage_errors([lineage_row]) == []
    changed_lineage = deepcopy(lineage_row)
    changed_lineage["state_hash"] = "sha256:changed"
    assert exp._lineage_errors([changed_lineage]) == ["lineage_state_hash_mismatch:0"]
    changed_lineage = deepcopy(lineage_row)
    changed_lineage["event_hash"] = "sha256:changed"
    assert exp._lineage_errors([changed_lineage]) == ["lineage_event_hash_mismatch:0"]
    second = deepcopy(lineage_row)
    second["update_count"] = 2
    second["state_hash"] = exp.canonical_hash(
        {
            "arm": second["arm"],
            "seed": second["seed"],
            "checkpoint": second["checkpoint"],
            "update_count": 2,
        }
    )
    assert "lineage_parent_hash_mismatch:1" in exp._lineage_errors([lineage_row, second])
    assert exp._lineage_errors([{}]) == ["lineage_row_invalid:0"]
    with pytest.raises(ValueError, match="lineage is required"):
        exp._load_lineage_rows({}, tmp_path)
    bad_lineage = deepcopy(lineage_value)
    bad_lineage["checkpoint_lineage"]["shards"][0]["sha256"] = "sha256:changed"
    with pytest.raises(ValueError, match="lineage shard hash"):
        exp._load_lineage_rows(bad_lineage, tmp_path)
    bad_lineage = deepcopy(lineage_value)
    bad_lineage["checkpoint_lineage"]["shards"][0]["rows"] = 2
    with pytest.raises(ValueError, match="lineage shard row"):
        exp._load_lineage_rows(bad_lineage, tmp_path)
    bad_lineage = deepcopy(lineage_value)
    bad_lineage["checkpoint_lineage"]["row_count"] = 2
    with pytest.raises(ValueError, match="lineage row count"):
        exp._load_lineage_rows(bad_lineage, tmp_path)

    artifact = exp.build_fixture_artifact(validation_receipts=_passing_receipts())
    assert (
        exp.independent_reduce({**artifact, "feedback_event_rows": None})[
            "online_capture_complete_score"
        ]
        == 0
    )
    mutations = {
        "schema": "bad",
        "experiment_id": "bad",
        "milestone": "bad",
        "MODEL_SPECS": ["bad"],
        "invocation_counts": {},
        "online_capture_complete_score": 0,
        "online_value_score": 1,
        "promotion_score": 1,
        "continuous_self_learning_task": False,
        "verdict_class": "bad",
    }
    expected = {
        "schema": "schema_mismatch",
        "experiment_id": "experiment_id_mismatch",
        "milestone": "run_identity_mismatch",
        "MODEL_SPECS": "model_declaration_mismatch",
        "invocation_counts": "invocation_counts_mismatch",
        "online_capture_complete_score": "online_capture_complete_score_mismatch",
        "online_value_score": "online_value_score_mismatch",
        "promotion_score": "promotion_score_mismatch",
        "continuous_self_learning_task": "continuous_self_learning_task_mismatch",
        "verdict_class": "verdict_class_invalid",
    }
    for field, value in mutations.items():
        changed = deepcopy(artifact)
        changed[field] = value
        assert expected[field] in exp.validate_artifact(changed, root=ROOT)
    changed = deepcopy(artifact)
    changed["verdict_class"] = "blocked"
    changed["honest_verdict"] = "complete_wrong"
    assert "blocked_verdict_prefix_invalid" in exp.validate_artifact(changed, root=ROOT)
    changed = deepcopy(artifact)
    changed["honest_verdict"] = "wrong"
    assert "complete_verdict_prefix_invalid" in exp.validate_artifact(changed, root=ROOT)
    changed = deepcopy(artifact)
    changed["feedback_event_rows"] = None
    assert any(
        row.startswith("feedback_rows_invalid:")
        for row in exp.validate_artifact(changed, root=ROOT)
    )
    changed = deepcopy(artifact)
    changed["checkpoint_lineage"] = None
    assert any(
        row.startswith("checkpoint_lineage_invalid:")
        for row in exp.validate_artifact(changed, root=ROOT)
    )
    changed = deepcopy(artifact)
    changed["source_artifact_hashes"] = {"bad": "not-a-reference"}
    assert "source_hash_reference_invalid:bad" in exp.validate_artifact(changed, root=ROOT)
    changed = deepcopy(artifact)
    changed["source_artifact_hashes"] = {"bad": {"path": "missing", "sha256": "sha256:missing"}}
    assert "source_hash_mismatch:bad" in exp.validate_artifact(changed, root=ROOT)
