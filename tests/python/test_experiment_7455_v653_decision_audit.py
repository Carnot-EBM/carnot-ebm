"""Tests for REQ-REPORT-7455 and SCENARIO-REPORT-7455-*.

The small fixtures expose each numeric operand. The real-evidence test then
applies the same reducer to the hash-bound Exp7454 event shards.
"""

from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path

import pytest

from carnot import experiment_7455_v653_decision_audit as audit


def _online_fixture() -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    state = audit.initial_state("learned_mixture")
    prediction = {
        "schema": audit.PREDICTION_SCHEMA,
        "row_type": "prediction_event",
        "arm": "learned_mixture",
        "ordering": "hash_order",
        "delay": 8,
        "seed": 65201,
        "group_id": "g1",
        "prediction_index": 0,
        "request_order": 0,
        "ledger_sequence": 0,
        "expert_probabilities": {
            "adaptive_gibbs": 0.7,
            "adaptive_spline": 0.8,
            "frozen_gibbs": 0.6,
            "frozen_spline": 0.9,
        },
        "mixture_weights": {name: 0.25 for name in audit.EXPERT_NAMES},
        "mixture_probability": 0.75,
        "pre_feedback_state_hash": state["state_hash"],
        "durable_acknowledged": True,
        "revealed": True,
        "shadow_only": True,
    }
    prediction["event_hash"] = audit.event_hash(prediction)
    numeric = audit.scalar_fixed_share_update(
        state["log_weights"],
        prediction["expert_probabilities"],
        1,
        eta=1.0,
        fixed_share=0.01,
    )
    next_state = audit.advance_state(state, prediction, 1)
    feedback = {
        "schema": audit.FEEDBACK_SCHEMA,
        "row_type": "feedback_event",
        "arm": "learned_mixture",
        "ordering": "hash_order",
        "delay": 8,
        "seed": 65201,
        "group_id": "g1",
        "prediction_index": 0,
        "request_order": 0,
        "reveal_order": 8,
        "ledger_sequence": 2,
        "prediction_event_hash": prediction["event_hash"],
        "feedback_label": 1,
        "true_label": 1,
        "per_expert_loss": numeric["losses"],
        "old_log_weights": state["log_weights"],
        "numeric_update": {
            "penalized_log_weights": numeric["penalized_log_weights"],
            "posterior_before_share": numeric["posterior_before_share"],
        },
        "normalizer": {
            "maximum": numeric["maximum"],
            "sum_exp": numeric["sum_exp"],
        },
        "new_log_weights": next_state["log_weights"],
        "parent_state_hash": state["state_hash"],
        "child_state_hash": next_state["state_hash"],
        "feedback_count_before": 0,
        "feedback_count_after": 1,
        "probability_source": "immutable_prediction_event",
        "update_applied": True,
        "revoked": False,
    }
    feedback["event_hash"] = audit.event_hash(feedback)
    outcome = {
        "schema": audit.OUTCOME_SCHEMA,
        "row_type": "outcome_event",
        "group_id": "g1",
        "ordering": "hash_order",
        "delay": 8,
        "seed": 65201,
        "request_order": 0,
        "ledger_sequence": 1,
        "true_label": 1,
        "revealed": True,
        "prediction_event_hashes": {"learned_mixture": prediction["event_hash"]},
    }
    outcome["event_hash"] = audit.event_hash(outcome)
    checkpoint = {
        "schema": audit.CHECKPOINT_SCHEMA,
        "row_type": "checkpoint_event",
        "arm": "learned_mixture",
        "ordering": "hash_order",
        "delay": 8,
        "seed": 65201,
        "completed_groups": 1,
        "ledger_sequence": 3,
        "state_hash": next_state["state_hash"],
        "log_weights": next_state["log_weights"],
        "feedback_count": 1,
        "previous_checkpoint_hash": None,
    }
    checkpoint["event_hash"] = audit.event_hash(checkpoint)
    return [prediction], [outcome], [feedback], [checkpoint]


def test_scalar_reducer_recomputes_losses_normalizer_and_weights() -> None:
    """SCENARIO-REPORT-7455-ONLINE-REPLAY uses saved prediction probabilities."""

    probabilities = dict(zip(audit.EXPERT_NAMES, (0.7, 0.8, 0.6, 0.9), strict=True))
    old = {name: math.log(0.25) for name in audit.EXPERT_NAMES}
    reduced = audit.scalar_fixed_share_update(old, probabilities, 1, eta=1.0, fixed_share=0.01)
    assert reduced["losses"]["frozen_spline"] == pytest.approx(-math.log(0.9))
    assert math.fsum(math.exp(value) for value in reduced["new_log_weights"].values()) == (
        pytest.approx(1.0)
    )
    assert reduced["sum_exp"] > 0.0
    with pytest.raises(ValueError, match="binary_label_invalid"):
        audit.scalar_fixed_share_update(old, probabilities, 2, eta=1.0, fixed_share=0.01)
    with pytest.raises(ValueError, match="expert_vector_invalid"):
        audit.scalar_fixed_share_update(old, {}, 1, eta=1.0, fixed_share=0.01)


def test_online_replay_checks_causality_hashes_and_no_feedback() -> None:
    """SCENARIO-REPORT-7455-ONLINE-REPLAY reconstructs causal state transitions."""

    predictions, outcomes, feedback, checkpoints = _online_fixture()
    result = audit.reduce_online_rows(predictions, outcomes, feedback, checkpoints)
    assert result["errors"] == []
    assert result["independent_update_replay"]["updates_replayed"] == 1
    assert result["independent_update_replay"]["max_expert_loss_error"] < 1e-12
    assert result["independent_update_replay"]["max_numeric_error"] < 1e-12
    assert result["prediction_before_reveal"] is True

    no_feedback_state = audit.initial_state("no_feedback_frozen_prior_mixture")
    no_feedback = deepcopy(predictions[0])
    no_feedback.update(
        {
            "arm": "no_feedback_frozen_prior_mixture",
            "pre_feedback_state_hash": no_feedback_state["state_hash"],
        }
    )
    no_feedback["event_hash"] = audit.event_hash(no_feedback)
    result = audit.reduce_online_rows([*predictions, no_feedback], outcomes, feedback, checkpoints)
    assert result["no_feedback_state_unchanged"] is True


def test_registered_mutations_each_break_the_named_claim() -> None:
    """SCENARIO-REPORT-7455-MUTATIONS rejects all six required corruptions."""

    rows = audit.run_mutation_controls()
    assert {row["mutation"] for row in rows} == set(audit.REQUIRED_MUTATIONS)
    assert all(row["passed"] is True for row in rows)
    assert all(row["rejected_claims"] for row in rows)

    predictions, outcomes, feedback, checkpoints = _online_fixture()
    changed = deepcopy(predictions)
    changed[0]["expert_probabilities"]["adaptive_gibbs"] = 0.1
    assert (
        "prediction_event_hash_mismatch"
        in audit.reduce_online_rows(changed, outcomes, feedback, checkpoints)["errors"]
    )


def test_branch_local_absence_does_not_hide_available_peer(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7455-INDEPENDENT keeps a missing static branch local."""

    (tmp_path / "results").mkdir()
    (tmp_path / "research-roadmap.yaml").write_text(
        """milestone: 2026.09.653
tasks:
- id: exp7453-energy-calibration
  deliverable: results/static.json
- id: exp7454-continuous-learning
  deliverable: results/online.json
""",
        encoding="utf-8",
    )
    (tmp_path / "results/online.json").write_text(
        json.dumps(
            {
                "experiment_id": "exp7454-v653-continuous-learning",
                "milestone": audit.MILESTONE,
                "verdict_class": "null",
                "flagged_adversarial": False,
                "online_capture_complete_score": 1,
                "honest_verdict": "complete_null_insufficient_online_benefit",
            }
        ),
        encoding="utf-8",
    )
    checks, hashes, producers = audit.collect_preconditions(tmp_path)
    assert producers["static"] == {}
    assert producers["online"]["verdict_class"] == "null"
    assert "results/online.json" in hashes
    static = audit.blocked_branch("static", [row for row in checks if row["branch"] == "static"])
    assert static["verdict_class"] == "blocked"
    assert static["gate_check_summary"]["blocked_observed"] is False
    assert static["gate_check_summary"]["blocked_path"] == "results/static.json"


def test_classification_and_retirement_separate_validity_from_value() -> None:
    """SCENARIO-REPORT-7455-RETIREMENT keeps a valid null and retires repetition."""

    static = {"branch": "static", "verdict_class": "blocked", "valid": False}
    online = {
        "branch": "online",
        "available": True,
        "verdict_class": "null",
        "valid": True,
        "complete": True,
        "value": False,
        "honest_verdict": "complete_null_insufficient_online_benefit",
    }
    classified = audit.classify_terminal([static, online], validation_passed=True)
    assert classified["verdict_class"] == "blocked"
    assert classified["decision_audit_complete_score"] == 1
    continuations = audit.continuation_rows(
        static_verdict="blocked",
        online_verdict=online["honest_verdict"],
        online_retired=True,
    )
    online_continuation = next(row for row in continuations if row["branch"] == "online")
    assert online_continuation["status"] == "retired"
    assert "repeated valid null" in online_continuation["reason"]

    defective = {**online, "valid": False, "verdict_class": "disqualified"}
    assert (
        audit.classify_terminal([static, defective], validation_passed=True)["verdict_class"]
        == "disqualified"
    )
    assert audit.classify_terminal([online], validation_passed=False)["verdict_class"] == (
        "disqualified"
    )


def test_fixture_artifact_and_fresh_process_modes(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """SCENARIO-REPORT-7455-ARTIFACT binds terminal reductions and checksum."""

    artifact = audit.build_artifact_for_test()
    assert audit.validate_artifact(artifact, verify_source_bytes=False) == []
    path = tmp_path / "candidate.json"
    path.write_text(json.dumps(artifact), encoding="utf-8")
    assert audit.cold_replay(path, verify_source_bytes=False) == []
    assert audit.main(["--cold-replay", str(path), "--skip-source-bytes"]) == 0
    assert json.loads(capsys.readouterr().out)["errors"] == []
    assert audit.main(["--independent-reduce", str(path), "--skip-source-bytes"]) == 0
    assert json.loads(capsys.readouterr().out)["errors"] == []

    changed = deepcopy(artifact)
    changed["promotion_score"] = 1
    changed["reproducibility_checksum"] = audit.reproducibility_checksum(changed)
    assert "promotion_score_invalid" in audit.validate_artifact(changed, verify_source_bytes=False)
    changed = deepcopy(artifact)
    changed["branch_rows"][0]["original_flagged_adversarial"] = True
    changed["reproducibility_checksum"] = audit.reproducibility_checksum(changed)
    assert "branch_rows_mismatch" in audit.validate_artifact(changed, verify_source_bytes=False)
    with pytest.raises(SystemExit, match="--date is required"):
        audit.main([])


def test_validation_plan_is_exact_and_private(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7455-ARTIFACT freezes the Exp7358 affected command plan."""

    private = tmp_path / "private"
    private.mkdir()
    commands = audit.build_validation_commands(audit.REPO_ROOT, private)
    assert [command.name for command in commands] == list(audit.AFFECTED_CHECK_NAMES)
    focused = next(command for command in commands if command.name == "focused_pytest")
    assert "-n" in focused.argv and "0" in focused.argv
    assert "--no-cov" in focused.argv
    assert any(arg.startswith("--basetemp=") for arg in focused.argv)
    assert "tests/python" not in focused.argv


@pytest.mark.memory_watchdog_skip
def test_actual_exp7454_rows_replay_while_exp7453_remains_blocked() -> None:
    """REQ-REPORT-7455 audits both real branches even when one producer is absent."""

    checks, hashes, producers, static, online = audit.audit_sources(audit.REPO_ROOT)
    assert (
        hashes["results/experiment_7454_v653_continuous_learning.json"]["original_verdict_class"]
        == "null"
    )
    assert producers["static"]["schema"] == "blocked_gate_check_v1"
    assert static["verdict_class"] == "blocked"
    assert online["valid"] is True
    assert online["verdict_class"] == "null"
    assert online["prediction_row_count"] == 5271
    assert online["feedback_row_count"] == 940
    assert online["independent_update_replay"]["updates_replayed"] == 940
    assert online["independent_update_replay"]["max_numeric_error"] < 1e-12
    assert any(row["branch"] == "static" and not row["passed"] for row in checks)


def test_event_validators_and_scalar_boundaries() -> None:
    """SCENARIO-REPORT-7455-MUTATIONS rejects malformed numeric operands."""

    predictions, _outcomes, feedback, _checkpoints = _online_fixture()
    prediction = predictions[0]
    row = deepcopy(prediction)
    row["label"] = 1
    with pytest.raises(ValueError, match="contains_label"):
        audit.validate_prediction_event(row)
    row = deepcopy(prediction)
    row["expert_probabilities"] = {}
    with pytest.raises(ValueError, match="experts_invalid"):
        audit.validate_prediction_event(row)
    row = deepcopy(prediction)
    row["mixture_weights"] = "bad"
    with pytest.raises(ValueError, match="weights_invalid"):
        audit.validate_prediction_event(row)
    row = deepcopy(prediction)
    row["mixture_weights"]["adaptive_gibbs"] = 0.9
    with pytest.raises(ValueError, match="weights_invalid"):
        audit.validate_prediction_event(row)

    fb = deepcopy(feedback[0])
    fb["true_label"] = 2
    with pytest.raises(ValueError, match="true_label_invalid"):
        audit.validate_feedback_event(fb)
    fb = deepcopy(feedback[0])
    fb["prediction_event_hash"] = ""
    with pytest.raises(ValueError, match="missing_prediction_hash"):
        audit.validate_feedback_event(fb)
    fb = deepcopy(feedback[0])
    fb["event_hash"] = "sha256:wrong"
    with pytest.raises(ValueError, match="hash_mismatch"):
        audit.validate_feedback_event(fb)

    old = {name: math.log(0.25) for name in audit.EXPERT_NAMES}
    probabilities = {name: 0.5 for name in audit.EXPERT_NAMES}
    with pytest.raises(ValueError, match="update_parameter_invalid"):
        audit.scalar_fixed_share_update(old, probabilities, 1, eta=-1.0, fixed_share=0.0)
    with pytest.raises(ValueError, match="binary label"):
        audit.bernoulli_log_loss(True, 0.5)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="finite probability"):
        audit.bernoulli_log_loss(1, math.inf)
    with pytest.raises(ValueError, match="arm_does_not_accept_feedback"):
        audit.advance_state(audit.initial_state("unknown"), prediction, 1)


def test_online_reducer_defensive_corruptions() -> None:
    """SCENARIO-REPORT-7455-MUTATIONS names every raw-event corruption."""

    predictions, outcomes, feedback, checkpoints = _online_fixture()
    all_errors: set[str] = set()

    duplicate_predictions = [*predictions, deepcopy(predictions[0])]
    all_errors.update(
        audit.reduce_online_rows(
            duplicate_predictions, outcomes, feedback, checkpoints, bootstrap_draws=4
        )["errors"]
    )

    bad_outcomes = [deepcopy(outcomes[0]) for _ in range(4)]
    bad_outcomes[0]["schema"] = "wrong"
    bad_outcomes[0]["prediction_event_hashes"] = {"learned_mixture": "sha256:missing"}
    bad_outcomes[1]["true_label"] = 2
    all_errors.update(
        audit.reduce_online_rows(
            predictions, bad_outcomes, feedback, checkpoints, bootstrap_draws=4
        )["errors"]
    )

    bad_feedback = [deepcopy(feedback[0])]
    bad_feedback[0].update(
        {
            "ledger_sequence": 0,
            "probability_source": "future_label",
            "revoked": True,
            "parent_state_hash": "sha256:wrong",
            "feedback_count_before": 9,
            "feedback_count_after": 9,
            "child_state_hash": "sha256:wrong",
        }
    )
    bad_feedback[0]["per_expert_loss"]["adaptive_gibbs"] += 1.0
    bad_feedback[0]["normalizer"]["maximum"] = None
    bad_feedback[0]["event_hash"] = audit.event_hash(bad_feedback[0])
    all_errors.update(
        audit.reduce_online_rows(
            predictions, outcomes, bad_feedback, checkpoints, bootstrap_draws=4
        )["errors"]
    )

    missing_prediction = [deepcopy(feedback[0])]
    missing_prediction[0]["prediction_event_hash"] = "sha256:missing"
    missing_prediction[0]["event_hash"] = audit.event_hash(missing_prediction[0])
    all_errors.update(
        audit.reduce_online_rows(
            predictions, outcomes, missing_prediction, checkpoints, bootstrap_draws=4
        )["errors"]
    )

    invalid_label = [deepcopy(feedback[0])]
    invalid_label[0]["feedback_label"] = 2
    invalid_label[0]["event_hash"] = audit.event_hash(invalid_label[0])
    all_errors.update(
        audit.reduce_online_rows(
            predictions, outcomes, invalid_label, checkpoints, bootstrap_draws=4
        )["errors"]
    )

    duplicate_feedback = [deepcopy(feedback[0]), deepcopy(feedback[0])]
    all_errors.update(
        audit.reduce_online_rows(
            predictions, outcomes, duplicate_feedback, checkpoints, bootstrap_draws=4
        )["errors"]
    )

    bad_checkpoint = [deepcopy(checkpoints[0])]
    bad_checkpoint[0]["previous_checkpoint_hash"] = "sha256:wrong"
    bad_checkpoint[0]["state_hash"] = "sha256:wrong"
    all_errors.update(
        audit.reduce_online_rows(
            predictions, outcomes, feedback, bad_checkpoint, bootstrap_draws=4
        )["errors"]
    )

    no_feedback = deepcopy(predictions[0])
    no_feedback.update(
        {
            "arm": "no_feedback_frozen_prior_mixture",
            "group_id": "g2",
            "pre_feedback_state_hash": "sha256:wrong",
        }
    )
    no_feedback["event_hash"] = audit.event_hash(no_feedback)
    all_errors.update(
        audit.reduce_online_rows(
            [*predictions, no_feedback], outcomes, feedback, checkpoints, bootstrap_draws=4
        )["errors"]
    )

    expected = {
        "duplicate_source_group",
        "outcome_event_hash_mismatch",
        "outcome_label_invalid",
        "duplicate_outcome_group",
        "outcome_prediction_reference_missing",
        "prediction_not_before_reveal",
        "feedback_probability_source_invalid",
        "feedback_admission_invalid",
        "parent_state_hash_mismatch",
        "feedback_count_before_mismatch",
        "feedback_count_after_mismatch",
        "expert_loss_mismatch",
        "numeric_update_mismatch",
        "child_state_hash_mismatch",
        "feedback_prediction_missing",
        "feedback_label_invalid",
        "duplicate_feedback_event",
        "feedback_event_order_invalid",
        "checkpoint_event_hash_mismatch",
        "checkpoint_lineage_invalid",
        "checkpoint_state_mismatch",
        "no_feedback_state_mutated",
        "prediction_outcome_missing",
    }
    assert expected <= all_errors


def test_evidence_boundaries_and_reader_failures(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """SCENARIO-REPORT-7455-ARTIFACT fails closed on unreadable evidence."""

    missing = audit.load_evidence_slot(tmp_path, "missing", Path("results/missing.json"))
    assert missing["authenticated"] is False
    missing_fallback = audit.load_evidence_slot(
        tmp_path,
        "missing",
        Path("results/missing.json"),
        fallback_path=Path("results/also-missing.json"),
    )
    assert missing_fallback["source_kind"] == "missing"
    assert audit._roadmap_deliverables(tmp_path / "missing.yaml") == {}
    malformed = tmp_path / "malformed.yaml"
    malformed.write_text("tasks: [", encoding="utf-8")
    assert audit._roadmap_deliverables(malformed) == {}

    shard = tmp_path / "rows.jsonl"
    shard.write_text('{"x": 1}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="shard_hash_mismatch"):
        audit._load_manifest_rows(
            tmp_path, [{"path": "rows.jsonl", "sha256": "sha256:wrong", "rows": 1}]
        )
    with pytest.raises(ValueError, match="shard_row_count_mismatch"):
        audit._load_manifest_rows(
            tmp_path,
            [{"path": "rows.jsonl", "sha256": audit.sha256_file(shard), "rows": 2}],
        )

    blocked = audit.audit_online_branch(tmp_path, missing, [])
    assert blocked["verdict_class"] == "blocked"
    static_available = {**missing, "available": True, "valid": True, "verdict_class": "null"}
    assert audit.audit_static_branch(tmp_path, static_available, [])["available"] is True

    artifact = audit.build_artifact_for_test()
    path = tmp_path / "fixture.json"
    path.write_text(json.dumps(artifact), encoding="utf-8")
    assert audit.independent_replay(path, root=audit.REPO_ROOT)
    assert audit.cold_replay(tmp_path / "absent.json") == ["candidate_artifact_unreadable"]
    assert audit.main(["--independent-reduce", str(path)]) == 1
    assert json.loads(capsys.readouterr().out)["errors"]


def test_artifact_defensive_fields_and_runtime_plans(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7455-ARTIFACT validates each terminal control field."""

    base = audit.build_artifact_for_test()
    mutations = (
        ("schema", "wrong", "artifact_identity_invalid"),
        ("milestone", "wrong", "artifact_schedule_invalid"),
        ("MODEL_SPECS", ["model"], "current_model_declaration_invalid"),
        ("invocation_counts", {}, "current_invocation_counts_invalid"),
        ("inference_substrate_class", "gpu", "inference_substrate_class_invalid"),
        ("execution_venue", "remote", "execution_venue_invalid"),
        ("decision_audit_complete_score", 2, "decision_audit_complete_score_invalid"),
        ("mutation_rows", [], "mutation_controls_invalid"),
    )
    for field, changed_value, expected in mutations:
        changed = deepcopy(base)
        changed[field] = changed_value
        assert expected in audit.validate_artifact(changed, verify_source_bytes=False)

    missing_field = deepcopy(base)
    del missing_field["rows"]
    assert "required_field_missing:rows" in audit.validate_artifact(
        missing_field, verify_source_bytes=False
    )
    bad_source = deepcopy(base)
    bad_source["source_artifact_hashes"] = {"bad": "not-a-mapping"}
    assert "source_reference_invalid:bad" in audit.validate_artifact(
        bad_source, verify_source_bytes=True
    )
    bad_source["source_artifact_hashes"] = {
        "missing": {"path": "missing", "sha256": "sha256:missing"}
    }
    assert "source_hash_mismatch:missing" in audit.validate_artifact(
        bad_source, root=tmp_path, verify_source_bytes=True
    )

    assert len(audit._terminal_commands(tmp_path / "candidate.json")) == 4
    span = audit._span("fixture", 1.0, 0.0, 2)
    assert span["completed_units"] == 2
    audit._progress(0.0, "fixture", "event", units=1)
    assert (
        audit.classify_terminal(
            [
                {"branch": "static", "available": True, "valid": True, "value": True},
                {"branch": "online", "available": True, "valid": True, "value": False},
            ],
            validation_passed=True,
        )["verdict_class"]
        == "positive"
    )
    assert (
        audit.classify_terminal(
            [
                {"branch": "static", "available": True, "valid": True, "value": False},
                {"branch": "online", "available": True, "valid": True, "value": False},
            ],
            validation_passed=True,
        )["verdict_class"]
        == "null"
    )

    with pytest.raises(SystemExit, match="--date must be"):
        audit.run_experiment(audit.REPO_ROOT, "wrong")


def test_remaining_reader_boundaries(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-REPORT-7455-ARTIFACT covers remaining fail-closed readers."""

    predictions, outcomes, feedback, checkpoints = _online_fixture()
    assert math.isinf(audit._max_gap({}, {}))
    bad_hash = [deepcopy(feedback[0])]
    bad_hash[0]["event_hash"] = "sha256:wrong"
    assert (
        "feedback_event_hash_mismatch"
        in audit.reduce_online_rows(
            predictions, outcomes, bad_hash, checkpoints, bootstrap_draws=4
        )["errors"]
    )
    assert (
        audit.online_integrity_errors(
            predictions, feedback, event_order_override=[predictions[0], feedback[0]]
        )
        == []
    )
    assert len(audit.run_mutation_controls(predictions, [feedback[0], feedback[0]])) == 6

    nonlist = tmp_path / "nonlist.yaml"
    nonlist.write_text("tasks: mapping\n", encoding="utf-8")
    assert audit._roadmap_deliverables(nonlist) == {}
    assert audit._load_jsonl_rows(tmp_path / "absent.jsonl") == []

    slot = audit.load_evidence_slot(
        audit.REPO_ROOT,
        "exp7454-continuous-learning",
        Path("results/experiment_7454_v653_continuous_learning.json"),
    )
    skipped = audit.audit_online_branch(audit.REPO_ROOT, slot, [], skip_rows=True)
    assert skipped["valid"] is True
    broken = deepcopy(slot)
    broken["payload"]["prediction_event_shards"][0]["sha256"] = "sha256:wrong"
    assert audit.audit_online_branch(audit.REPO_ROOT, broken, [])["verdict_class"] == (
        "disqualified"
    )

    missing_root = tmp_path / "missing-root"
    missing_root.mkdir()
    _checks, _hashes, _producers, static, online = audit.audit_sources(missing_root)
    assert static["verdict_class"] == online["verdict_class"] == "blocked"

    malformed_source_root = tmp_path / "malformed-source-root"
    (malformed_source_root / "results").mkdir(parents=True)
    (malformed_source_root / "research-roadmap.yaml").write_text(
        """milestone: 2026.09.653
tasks:
- id: exp7454-continuous-learning
  deliverable: results/online.json
""",
        encoding="utf-8",
    )
    (malformed_source_root / "results/online.json").write_text(
        json.dumps(
            {
                "experiment_id": "exp7454-v653-continuous-learning",
                "milestone": audit.MILESTONE,
                "verdict_class": "null",
                "flagged_adversarial": False,
                "source_artifact_hashes": {"malformed": "not-a-mapping"},
            }
        ),
        encoding="utf-8",
    )
    malformed_checks, _malformed_hashes, _malformed_loaded = audit.collect_preconditions(
        malformed_source_root
    )
    assert any(row["check"] == "online:artifact_exists" for row in malformed_checks)

    nonfixture = audit.build_artifact_for_test()
    nonfixture["fixture_artifact"] = False
    nonfixture["reproducibility_checksum"] = audit.reproducibility_checksum(nonfixture)
    assert audit.validate_artifact(nonfixture, verify_source_bytes=False) == []

    monkeypatch.setattr(audit, "validate_command_plan", lambda *_args: ["forced"])
    with pytest.raises(ValueError, match="validation_plan_invalid"):
        audit.build_validation_commands(audit.REPO_ROOT, tmp_path)
