"""Tests for REQ-REPORT-7441 and SCENARIO-REPORT-7441-*.

The fixtures are small enough to make every expected reduction visible. They
exercise the same evidence boundaries as the full artifact without importing a
producer reducer or rewriting a producer result.
"""

from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path

import pytest

from carnot import experiment_7441_v652_decision_audit as audit


def _certificate(head: str, policy_kind: str, accept: float, reject: float) -> dict:
    return {
        "head": head,
        "policy_kind": policy_kind,
        "frozen_policy": {
            "accept_enabled": True,
            "accept_threshold": accept,
            "reject_enabled": True,
            "reject_threshold": reject,
            "selection_partition": (
                "policy_tuning" if policy_kind == "tuned" else "v651_fixed_threshold_protocol"
            ),
        },
        "certificate": {
            "accept_check": {"alpha_allocated": audit.ALPHA_PER_CHECK},
            "reject_check": {"alpha_allocated": audit.ALPHA_PER_CHECK},
            "coverage_check": {"alpha_allocated": audit.ALPHA_PER_CHECK},
        },
    }


def _static_fixture() -> tuple[list[dict], list[dict]]:
    certificates = [
        _certificate("sparse_spline_49", "tuned", 0.75, 0.25),
        _certificate("sparse_spline_49", "old_fixed", 0.9, 0.1),
        _certificate("raw_l2_logistic", "tuned", 0.75, 0.25),
    ]
    rows: list[dict] = []
    values = [("g1", 1, 0.9), ("g2", 0, 0.8), ("g3", 0, 0.2), ("g4", 1, 0.4)]
    for stage in ("certification", "final_test"):
        for head, kind in (
            ("sparse_spline_49", "tuned"),
            ("sparse_spline_49", "old_fixed"),
            ("raw_l2_logistic", "tuned"),
        ):
            for group, label, probability in values:
                row = {
                    "stage": stage,
                    "head": head,
                    "policy_kind": kind,
                    "group_id": f"{stage}-{group}",
                    "row_key": f"{stage}-{group}",
                    "label": label,
                    "probability": probability,
                }
                if stage == "final_test":
                    row["action"] = audit.policy_action(
                        probability,
                        certificates[
                            {
                                ("sparse_spline_49", "tuned"): 0,
                                ("sparse_spline_49", "old_fixed"): 1,
                                ("raw_l2_logistic", "tuned"): 2,
                            }[(head, kind)]
                        ]["frozen_policy"],
                    )
                rows.append(row)
    return rows, certificates


def _prediction(
    group: str,
    *,
    arm: str = "learned_mixture",
    ordering: str = "hash_order",
    delay: int = 0,
    seed: int = 1,
    index: int = 0,
    probability: float = 0.8,
    label: int = 1,
) -> dict:
    return {
        "row_type": "prediction",
        "observation_id": group,
        "group_id": group,
        "row_key": group,
        "arm": arm,
        "ordering": ordering,
        "delay": delay,
        "seed": seed,
        "prediction_index": index,
        "request_order": index,
        "available_at": index + delay,
        "probability": probability,
        "proposed_action": "accept",
        "deployed_action": "escalate",
        "shadow_only": True,
        "certified_safe": False,
        "revealed": True,
        "propensity": 0.25,
        "label": label,
        "label_read_at_prediction": False,
        "prediction_before_feedback": True,
        "loss": audit.binary_log_loss(label, probability),
        "brier": (probability - label) ** 2,
    }


def _online_fixture() -> list[dict]:
    prediction = _prediction("g1")
    losses = {
        "adaptive_gibbs": audit.binary_log_loss(1, 0.7),
        "adaptive_spline": audit.binary_log_loss(1, 0.8),
        "frozen_gibbs": audit.binary_log_loss(1, 0.6),
        "frozen_spline": audit.binary_log_loss(1, 0.9),
    }
    before = {name: 0.25 for name in audit.EXPERT_NAMES}
    after = audit.replay_weight_update(before, losses)
    feedback = {
        "row_type": "feedback_event",
        "observation_id": "g1",
        "group_id": "g1",
        "arm": "learned_mixture",
        "ordering": "hash_order",
        "delay": 0,
        "seed": 1,
        "prediction_index": 0,
        "request_order": 0,
        "arrival_index": 0,
        "reveal_probability": 0.25,
        "true_label": 1,
        "feedback_label": 1,
        "prediction_commit_hash": "sha256:prediction",
        "parent_state_hash": "sha256:parent",
        "event_hash": "sha256:event",
        "child_state_hash": "sha256:child",
        "probability_source": "stored_at_prediction",
        "prediction_time_loss": prediction["loss"],
        "expert_prediction_time_losses": losses,
        "update_count": 1,
        "revoked": False,
    }
    weights = {
        "row_type": "weight_trajectory",
        "observation_id": "g1",
        "arm": "learned_mixture",
        "ordering": "hash_order",
        "delay": 0,
        "seed": 1,
        "arrival_index": 0,
        "weights_before": before,
        "weights_after": after,
    }
    lineage = {
        "row_type": "checkpoint_lineage",
        "observation_id": "g1",
        "arm": "learned_mixture",
        "ordering": "hash_order",
        "delay": 0,
        "seed": 1,
        "parent_state_hash": "sha256:parent",
        "event_hash": "sha256:event",
        "child_state_hash": "sha256:child",
        "exactly_once": True,
    }
    return [prediction, feedback, weights, lineage]


def test_binary_losses_bounds_and_weight_replay() -> None:
    """REQ-REPORT-7441 recomputes scalar evidence without producer helpers."""

    assert audit.binary_log_loss(1, 0.5) == pytest.approx(math.log(2))
    with pytest.raises(ValueError, match="binary label"):
        audit.binary_log_loss(2, 0.5)
    upper = audit.clopper_pearson_upper(0, 100, audit.ALPHA_PER_CHECK)
    lower = audit.clopper_pearson_lower(80, 100, audit.ALPHA_PER_CHECK)
    assert 0.0 < upper < 0.1
    assert 0.6 < lower < 0.8
    with pytest.raises(ValueError, match="binomial"):
        audit.clopper_pearson_upper(2, 1, 0.1)

    losses = {name: float(index) for index, name in enumerate(audit.EXPERT_NAMES)}
    updated = audit.replay_weight_update({name: 0.25 for name in audit.EXPERT_NAMES}, losses)
    assert sum(updated.values()) == pytest.approx(1.0)
    assert updated[audit.EXPERT_NAMES[0]] > updated[audit.EXPERT_NAMES[-1]]
    with pytest.raises(ValueError, match="complete expert"):
        audit.replay_weight_update({name: 0.25 for name in audit.EXPERT_NAMES}, {})


def test_static_reduction_checks_groups_bounds_scores_and_equal_probabilities() -> None:
    """SCENARIO-REPORT-7441-STATIC reduces policies from raw group rows."""

    rows, certificates = _static_fixture()
    assert audit.static_integrity_errors(rows, certificates) == []
    reduced = audit.reduce_static_evidence(rows, certificates, draws=40, seed=4)
    tuned = reduced["certificates"]["sparse_spline_49:tuned"]
    metrics = reduced["metrics"]["sparse_spline_49:tuned"]
    assert tuned["accept_check"]["selected_groups"] == 2
    assert tuned["accept_check"]["harmful_outcomes"] == 1
    assert tuned["reject_check"]["selected_groups"] == 1
    assert tuned["alpha_allocation"] == pytest.approx(audit.ALPHA_PER_CHECK)
    assert metrics["row_count"] == 4
    assert metrics["brier"] == pytest.approx((0.01 + 0.64 + 0.04 + 0.36) / 4)
    assert metrics["registered_utility"] == pytest.approx(-18 / 4)
    assert reduced["paired_coverage_intervals"]["tuned_spline_minus_old"]["draws"] == 40

    duplicate = deepcopy(rows)
    duplicate.append(deepcopy(duplicate[0]))
    assert "duplicate_source_group" in audit.static_integrity_errors(duplicate, certificates)
    mismatched = deepcopy(rows)
    target = next(
        row
        for row in mismatched
        if row["stage"] == "final_test"
        and row["head"] == "sparse_spline_49"
        and row["policy_kind"] == "old_fixed"
    )
    target["probability"] = 0.123
    assert "selector_probability_mismatch" in audit.static_integrity_errors(
        mismatched, certificates
    )


def test_static_integrity_rejects_role_leakage_wrong_units_and_false_zero() -> None:
    """SCENARIO-REPORT-7441-MUTATIONS rejects static evidence defects."""

    rows, certificates = _static_fixture()
    leaked = deepcopy(certificates)
    leaked[0]["frozen_policy"]["selection_partition"] = "final_test"
    assert "final_test_threshold_influence" in audit.static_integrity_errors(rows, leaked)

    wrong_unit = deepcopy(rows)
    wrong_unit[0]["group_id"] = wrong_unit[1]["group_id"]
    assert "duplicate_source_group" in audit.static_integrity_errors(wrong_unit, certificates)

    false_zero = deepcopy(certificates)
    false_zero[1]["certificate"]["accept_check"].update(
        {"selected_groups": 0, "empirical_risk": 0.0}
    )
    assert "zero_selected_risk_coercion" in audit.static_integrity_errors(rows, false_zero)

    missing = deepcopy(rows)
    del missing[0]["label"]
    assert "static_required_field_missing" in audit.static_integrity_errors(missing, certificates)


def test_online_integrity_replays_complete_events_and_rejects_mutations() -> None:
    """SCENARIO-REPORT-7441-ONLINE requires replayable causal evidence."""

    rows = _online_fixture()
    assert audit.online_integrity_errors(rows) == []
    replay = audit.rebuild_weight_updates(rows)
    assert replay["updates_replayed"] == 1
    assert replay["max_weight_gap"] < 1e-12

    mutations = audit.run_mutation_controls()
    assert {row["attack"] for row in mutations} == set(audit.REQUIRED_MUTATIONS)
    assert all(row["passed"] for row in mutations)

    missing = deepcopy(rows)
    del missing[1]["expert_prediction_time_losses"]
    assert "missing_expert_predictions" in audit.online_integrity_errors(missing)
    future = deepcopy(rows)
    future[1]["arrival_index"] = -1
    assert "future_label_access" in audit.online_integrity_errors(future)
    duplicate = deepcopy(rows)
    duplicate.append(deepcopy(duplicate[1]))
    assert "duplicate_feedback_event" in audit.online_integrity_errors(duplicate)
    revoked = deepcopy(rows)
    revoked[1]["revoked"] = True
    assert "revoked_label_applied" in audit.online_integrity_errors(revoked)


def test_online_metrics_controls_and_moving_blocks() -> None:
    """REQ-REPORT-7441 recomputes online scores and moving-block intervals."""

    rows: list[dict] = []
    arms = ("learned_mixture", *audit.PRIMARY_COMPARATORS)
    for ordering in audit.ORDERS:
        for delay in audit.DELAYS:
            for index in range(4):
                for seed in (1, 2):
                    for arm_index, arm in enumerate(arms):
                        probability = 0.7 - arm_index * 0.02
                        rows.append(
                            _prediction(
                                f"{ordering}-{delay}-g{index}",
                                arm=arm,
                                ordering=ordering,
                                delay=delay,
                                seed=seed,
                                index=index,
                                probability=probability,
                            )
                        )
    reports = audit.reduce_online_metrics(rows)
    assert len(reports) == 16
    intervals = audit.moving_block_intervals(rows, draws=30, seed=8, block_lengths=(2, 3))
    assert len(intervals) == 24
    assert {row["fit_seeds_averaged_before_resampling"] for row in intervals} == {2}

    controls = audit.online_controls(rows)
    assert controls["uniform_reveal_propensity"] is True
    assert controls["shadow_only_actions"] is True
    assert controls["no_feedback_equality"] is True
    assert controls["calibration_theorem_applied"] is False
    assert controls["no_share_regret_theorem_applied"] is False
    assert controls["energy_representation"] == "logistic_reexpression_of_probability"

    bad = deepcopy(rows)
    bad[0]["propensity"] = 0.5
    assert "nonuniform_reveal_propensity" in audit.online_integrity_errors(bad)
    bad = deepcopy(rows)
    bad[0]["shadow_only"] = False
    assert "adaptive_action_not_shadow_only" in audit.online_integrity_errors(bad)


def test_branch_classification_keeps_null_blocked_and_disqualified_distinct() -> None:
    """SCENARIO-REPORT-7441-INDEPENDENT preserves every branch disposition."""

    null = {"branch": "static", "available": True, "valid": True, "complete": True, "value": False}
    blocked = {"branch": "online", "available": False, "valid": False, "complete": False}
    result = audit.classify_terminal([null, blocked], validation_passed=True)
    assert result["verdict_class"] == "blocked"
    assert result["honest_verdict"].startswith("blocked_")

    invalid = {"branch": "online", "available": True, "valid": False, "complete": True}
    result = audit.classify_terminal([null, invalid], validation_passed=True)
    assert result == {
        "verdict_class": "disqualified",
        "honest_verdict": "complete_disqualified_invalid_branch_evidence",
    }
    result = audit.classify_terminal([null, {**null, "branch": "online"}], validation_passed=True)
    assert result["verdict_class"] == "null"
    assert "null" in result["honest_verdict"]
    assert (
        audit.classify_terminal([null], validation_passed=False)["verdict_class"] == "disqualified"
    )


def test_fixture_artifact_validates_cold_and_mutations_fail(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7441-ARTIFACT binds terminal fields and checksum."""

    artifact = audit.build_artifact_for_test()
    assert audit.validate_artifact(artifact, verify_source_bytes=False) == []
    path = tmp_path / "artifact.json"
    path.write_text(json.dumps(artifact), encoding="utf-8")
    assert audit.cold_replay(path, verify_source_bytes=False) == []

    for field, value, expected in (
        ("promotion_score", 1, "promotion_score_invalid"),
        ("MODEL_SPECS", ["model"], "current_model_declaration_invalid"),
        ("verdict_class", "partial", "terminal_classification_mismatch"),
    ):
        changed = deepcopy(artifact)
        changed[field] = value
        changed["reproducibility_checksum"] = audit.reproducibility_checksum(changed)
        assert expected in audit.validate_artifact(changed, verify_source_bytes=False)

    changed = deepcopy(artifact)
    changed["reproducibility_checksum"] = "sha256:wrong"
    assert "reproducibility_checksum_mismatch" in audit.validate_artifact(
        changed, verify_source_bytes=False
    )


def test_preconditions_preserve_missing_none_zero_and_flags(tmp_path: Path) -> None:
    """REQ-REPORT-7441 records exact observed prerequisite values."""

    roadmap = tmp_path / "research-roadmap.yaml"
    roadmap.write_text(
        """milestone: 2026.09.652
tasks:
- id: exp7436-selection-protocol
  deliverable: results/a.json
- id: exp7438-mixture-prototype
  deliverable: results/b.json
- id: exp7439-certified-decisions
  deliverable: results/c.json
- id: exp7440-mixture-learning
  deliverable: results/d.json
""",
        encoding="utf-8",
    )
    results = tmp_path / "results"
    results.mkdir()
    (results / "a.json").write_text(
        json.dumps(
            {
                "experiment_id": "exp7436-v652-selection-protocol",
                "milestone": "2026.09.652",
                "verdict_class": "null",
                "flagged_adversarial": False,
                "selection_protocol_ready_score": 1,
            }
        ),
        encoding="utf-8",
    )
    (results / "b.json").write_text("{}", encoding="utf-8")
    (results / "c.json").write_text("null", encoding="utf-8")
    checks, _hashes, producers = audit.collect_preconditions(tmp_path)
    assert producers["selection"]["verdict_class"] == "null"
    observed = {row["check"]: row["observed"] for row in checks if not row["passed"]}
    assert observed["prototype:experiment_id"] is None
    assert observed["static:artifact_object"] is None
    assert observed["online:artifact_exists"] is False
    assert any(row["observed_type"] == "missing" for row in checks if not row["passed"])


def test_argument_modes_and_main_readers(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """SCENARIO-REPORT-7441-ARTIFACT exposes bounded fresh-process modes."""

    artifact = audit.build_artifact_for_test()
    path = tmp_path / "artifact.json"
    path.write_text(json.dumps(artifact), encoding="utf-8")
    args = audit.parse_args(["--cold-replay", str(path), "--skip-source-bytes"])
    assert args.cold_replay == path
    assert audit.main(["--cold-replay", str(path), "--skip-source-bytes"]) == 0
    assert json.loads(capsys.readouterr().out)["errors"] == []
    assert audit.main(["--independent-reduce", str(path), "--skip-source-bytes"]) == 0
    assert json.loads(capsys.readouterr().out)["errors"] == []
    with pytest.raises(SystemExit, match="--date is required"):
        audit.main([])


@pytest.mark.memory_watchdog_skip
def test_actual_producer_rows_run_both_independent_branches() -> None:
    """SCENARIO-REPORT-7441-INDEPENDENT audits complete null producers."""

    checks, hashes, producers, static, online = audit._audit_sources(audit.REPO_ROOT)
    assert checks and all(row["passed"] for row in checks)
    assert set(producers) == set(audit.PRODUCERS)
    assert len(hashes) == 4
    assert static["valid"] is True
    assert static["verdict_class"] == "null"
    assert static["raw_row_count"] == 5016
    assert online["valid"] is False
    assert online["verdict_class"] == "disqualified"
    assert online["prediction_row_count"] == 105420
    assert online["errors"] == [
        "missing_expert_predictions",
        "weight_update_replay_incomplete",
    ]
    assert online["controls"]["uniform_reveal_propensity"] is True
    assert online["controls"]["shadow_only_actions"] is True


def test_defensive_reducer_boundaries_and_validation_plan(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-REPORT-7441-MUTATIONS covers malformed private evidence."""

    assert audit._binomial_cdf(-1, 2, 0.5) == 0.0
    assert audit._binomial_cdf(2, 2, 0.5) == 1.0
    assert audit.clopper_pearson_upper(1, 1, 0.1) == 1.0
    assert audit.clopper_pearson_lower(0, 1, 0.1) == 0.0
    with pytest.raises(ValueError, match="finite probability"):
        audit.binary_log_loss(1, math.inf)
    with pytest.raises(ValueError, match="metric rows"):
        audit.reduce_policy_metrics([])
    with pytest.raises(ValueError, match="positive draws"):
        audit.paired_coverage_intervals([], draws=0, seed=1)
    with pytest.raises(ValueError, match="unique source groups"):
        audit.paired_coverage_intervals(
            [
                {"group_id": "g", "tuned_spline": 1, "old_spline": 0, "tuned_logistic": 0},
                {"group_id": "g", "tuned_spline": 1, "old_spline": 0, "tuned_logistic": 0},
            ],
            draws=1,
            seed=1,
        )
    with pytest.raises(ValueError, match="positive and finite"):
        audit.replay_weight_update(
            {name: (0.0 if index == 0 else 0.25) for index, name in enumerate(audit.EXPERT_NAMES)},
            {name: 0.5 for name in audit.EXPERT_NAMES},
        )

    base = _online_fixture()
    invalid_prediction = deepcopy(base[0])
    del invalid_prediction["label"]
    duplicate_prediction = deepcopy(base[0])
    bad_metrics = deepcopy(base[0])
    bad_metrics["loss"] = 123.0
    bad_metrics["brier"] = 123.0
    invalid_metrics = deepcopy(base[0])
    invalid_metrics["loss"] = None
    future_prediction = deepcopy(base[0])
    future_prediction["label_read_at_prediction"] = True
    future_prediction["prediction_before_feedback"] = False
    deployed = deepcopy(base[0])
    deployed["deployed_action"] = "accept"
    orphan_feedback = deepcopy(base[1])
    orphan_feedback["observation_id"] = "missing"
    bad_feedback = deepcopy(base[1])
    bad_feedback["update_count"] = 0
    bad_feedback["prediction_time_loss"] = 99.0
    invalid_feedback = deepcopy(base[1])
    invalid_feedback["prediction_time_loss"] = None
    duplicate_weight = deepcopy(base[2])
    orphan_lineage = deepcopy(base[3])
    orphan_lineage["observation_id"] = "other"
    bad_lineage = deepcopy(base[3])
    bad_lineage["event_hash"] = "wrong"
    bad_once = deepcopy(base[3])
    bad_once["exactly_once"] = False
    revocation = {"row_type": "revocation", "replayed": False}
    errors = audit.online_integrity_errors(
        [
            invalid_prediction,
            base[0],
            duplicate_prediction,
            bad_metrics,
            invalid_metrics,
            future_prediction,
            deployed,
            base[1],
            orphan_feedback,
            bad_feedback,
            invalid_feedback,
            base[2],
            duplicate_weight,
            base[3],
            orphan_lineage,
            bad_lineage,
            bad_once,
            revocation,
        ]
    )
    assert {
        "online_required_field_missing",
        "duplicate_prediction_event",
        "wrong_group_unit",
        "future_label_access",
        "adaptive_action_not_shadow_only",
        "prediction_loss_mismatch",
        "prediction_brier_mismatch",
        "prediction_metric_invalid",
        "feedback_without_prediction",
        "feedback_update_semantics_invalid",
        "stored_prediction_loss_mismatch",
        "stored_prediction_loss_invalid",
        "duplicate_weight_update",
        "duplicate_lineage_event",
        "lineage_without_feedback",
        "lineage_hash_mismatch",
        "lineage_not_exactly_once",
        "revoked_label_not_replayed",
    }.issubset(errors)

    with pytest.raises(ValueError, match="complete expert prediction losses"):
        audit.rebuild_weight_updates([base[0], base[2]])
    with pytest.raises(ValueError, match="bootstrap draws"):
        audit.moving_block_intervals([], draws=0)
    with pytest.raises(ValueError, match="complete paired rows"):
        audit.moving_block_intervals([base[0]], draws=1)
    with pytest.raises(ValueError, match="complete seed rows"):
        audit.moving_block_intervals([{"row_type": "not-a-prediction"}], draws=1)
    incomplete_cells = [
        _prediction(
            f"{delay}-g",
            arm=arm,
            ordering="hash_order",
            delay=delay,
            seed=1,
        )
        for delay in audit.DELAYS
        for arm in ("learned_mixture", *audit.PRIMARY_COMPARATORS)
    ]
    with pytest.raises(ValueError, match="all registered cells"):
        audit.moving_block_intervals(incomplete_cells, draws=1)

    malformed_static, malformed_certificates = _static_fixture()
    malformed_static[0]["label"] = "not-binary"
    malformed_static[1]["group_id"] = malformed_static[1]["group_id"].replace(
        "certification", "final_test"
    )
    malformed_certificates[0]["certificate"]["accept_check"]["alpha_allocated"] = 0.5
    static_errors = audit.static_integrity_errors(malformed_static, malformed_certificates)
    assert {
        "static_operand_invalid",
        "role_group_overlap",
        "alpha_allocation_mismatch",
    }.issubset(static_errors)

    assert audit._roadmap_deliverables(tmp_path / "absent.yaml") == {}
    assert audit._observed_type(None, present=True) == "null"
    assert audit._observed_type(0, present=True) == "integer_zero"

    shard = tmp_path / "rows.jsonl"
    shard.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="row_shard_hash_mismatch"):
        audit._load_bound_rows(tmp_path, [{"path": shard.name, "sha256": "wrong", "rows": 1}])
    shard.write_text("[]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="row_shard_object_invalid"):
        audit._load_bound_rows(
            tmp_path,
            [{"path": shard.name, "sha256": audit.sha256_file(shard), "rows": 1}],
        )
    shard.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="row_shard_count_mismatch"):
        audit._load_bound_rows(
            tmp_path,
            [{"path": shard.name, "sha256": audit.sha256_file(shard), "rows": 2}],
        )

    failed = [
        audit._precondition(
            "static",
            "missing",
            "producer",
            "missing.json",
            "field",
            True,
            None,
            present=False,
        )
    ]
    assert audit.audit_static_branch(tmp_path, {}, failed)["verdict_class"] == "blocked"
    assert audit.audit_online_branch(tmp_path, {}, failed)["verdict_class"] == "blocked"
    monkeypatch.setattr(
        audit, "_load_bound_rows", lambda *_args: (_ for _ in ()).throw(ValueError("bad-shard"))
    )
    broken_static = audit.audit_static_branch(tmp_path, {"experiment_id": "static"}, [])
    broken_online = audit.audit_online_branch(tmp_path, {"experiment_id": "online"}, [])
    assert broken_static["verdict_class"] == "disqualified"
    assert "bad-shard" in broken_static["errors"]
    assert broken_online["verdict_class"] == "disqualified"
    assert "bad-shard" in broken_online["errors"]

    positive = {"branch": "static", "available": True, "valid": True, "value": True}
    assert (
        audit.classify_terminal([positive], validation_passed=True)["verdict_class"] == "positive"
    )

    private = tmp_path / "private"
    commands = audit.build_validation_commands(audit.REPO_ROOT, private)
    assert [row.name for row in commands] == list(audit.validation_scope.REQUIRED_CHECK_NAMES)
    assert {row.spec.name for row in audit._terminal_commands(tmp_path / "candidate.json")} == {
        "cold_artifact_replay",
        "independent_branch_recompute",
        "adversarial_verify",
        "verdict_row_consistency_strict",
    }
    started = audit.time.monotonic()
    audit._progress(started, "test", "boundary", units=1)
    assert "phase=test" in capsys.readouterr().out
    span = audit._span("test", started, started, 1)
    assert span["completed_units"] == 1
    assert audit._utc_now().endswith("+00:00")

    monkeypatch.setattr(audit, "validate_command_plan", lambda *_args: ["drift"])
    with pytest.raises(ValueError, match="validation_plan_invalid:drift"):
        audit.build_validation_commands(audit.REPO_ROOT, tmp_path / "invalid-plan")


def test_cold_reader_and_artifact_validation_error_boundaries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-REPORT-7441-ARTIFACT rejects identity and source drift."""

    unreadable = tmp_path / "missing.json"
    assert audit.cold_replay(unreadable) == ["candidate_artifact_unreadable"]
    artifact = audit.build_artifact_for_test()
    mutations = (
        ({"schema": "wrong"}, "artifact_identity_invalid"),
        ({"run_date": "wrong"}, "artifact_schedule_invalid"),
        ({"invocation_counts": {}}, "current_invocation_counts_invalid"),
        ({"inference_substrate_class": "wrong"}, "inference_substrate_class_invalid"),
        ({"execution_venue": "wrong"}, "execution_venue_invalid"),
        ({"audit_mutation_rows": []}, "mutation_controls_invalid"),
    )
    for replacement, expected in mutations:
        changed = {**deepcopy(artifact), **replacement}
        changed["reproducibility_checksum"] = audit.reproducibility_checksum(changed)
        assert expected in audit.validate_artifact(changed, verify_source_bytes=False)

    changed = deepcopy(artifact)
    del changed["schema"]
    changed["reproducibility_checksum"] = audit.reproducibility_checksum(changed)
    assert "required_field_missing:schema" in audit.validate_artifact(
        changed, verify_source_bytes=False
    )
    source = tmp_path / "source.json"
    source.write_text("{}", encoding="utf-8")
    changed = deepcopy(artifact)
    changed["source_artifact_hashes"] = {
        "source": {"path": str(source), "sha256": "sha256:wrong"},
        "invalid": "not-a-reference",
    }
    changed["reproducibility_checksum"] = audit.reproducibility_checksum(changed)
    errors = audit.validate_artifact(changed, root=tmp_path)
    assert "source_hash_mismatch:source" in errors
    assert "source_reference_invalid:invalid" in errors

    nonfixture = deepcopy(artifact)
    nonfixture["fixture_artifact"] = False
    nonfixture["reproducibility_checksum"] = audit.reproducibility_checksum(nonfixture)
    assert audit.validate_artifact(nonfixture, verify_source_bytes=False) == []

    different_static = {**artifact["static_audit"], "raw_row_count": 1}
    different_online = {**artifact["online_audit"], "raw_row_count": 1}
    monkeypatch.setattr(
        audit,
        "_audit_sources",
        lambda _root: ([], {}, {}, different_static, different_online),
    )
    path = tmp_path / "replay.json"
    path.write_text(json.dumps(artifact), encoding="utf-8")
    replay_errors = audit.independent_replay(path, root=tmp_path)
    assert "static_independent_reduction_mismatch" in replay_errors
    assert "online_independent_reduction_mismatch" in replay_errors

    calls: list[tuple[Path, str, Path]] = []
    monkeypatch.setattr(
        audit,
        "run_experiment",
        lambda root, date, *, output_path: calls.append((root, date, output_path)),
    )
    output = tmp_path / "output.json"
    assert audit.main(["--date", audit.RUN_DATE, "--output", str(output)]) == 0
    assert calls == [(audit.REPO_ROOT, audit.RUN_DATE, output)]
