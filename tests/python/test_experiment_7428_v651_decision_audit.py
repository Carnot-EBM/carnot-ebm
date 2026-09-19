"""Tests for REQ-REPORT-7428 and SCENARIO-REPORT-7428-*.

The fixtures are small enough to make every safety mutation explicit. The
entrypoint later applies the same reducers to the shipped producer shards.
"""

from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path

import pytest

from carnot import experiment_7428_v651_decision_audit as audit


def _predictors() -> list[dict[str, object]]:
    return [
        {
            "row_key": "r-fit",
            "group_id": "g-fit",
            "partition": "fit",
            "task_type": "QA",
            "source_text": "alpha source",
            "response_text": "unsupported claim",
            "features": {name: 0.25 for name in audit.SOURCE_FEATURE_NAMES},
        },
        {
            "row_key": "r-test",
            "group_id": "g-test",
            "partition": "final_test",
            "task_type": "QA",
            "source_text": "beta source",
            "response_text": "supported answer",
            "features": {name: 0.75 for name in audit.SOURCE_FEATURE_NAMES},
        },
    ]


def _evaluators() -> list[dict[str, object]]:
    return [
        {
            "row_key": "r-fit",
            "group_id": "g-fit",
            "partition": "fit",
            "task_type": "QA",
            "response_id": "response-a",
            "source_id": "source-a",
            "annotations": [
                {
                    "start": 0,
                    "end": 11,
                    "text": "unsupported",
                    "implicit_true": True,
                }
            ],
            "primary_label": 0,
            "implicit_true_excluded_label": 1,
        },
        {
            "row_key": "r-test",
            "group_id": "g-test",
            "partition": "final_test",
            "task_type": "QA",
            "response_id": "response-b",
            "source_id": "source-b",
            "annotations": [],
            "primary_label": 1,
            "implicit_true_excluded_label": 1,
        },
    ]


def _metric_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for arm, probabilities in (
        ("sparse_spline_49", (0.2, 0.8)),
        ("dense_spline_logistic", (0.2, 0.8)),
    ):
        for index, (label, probability, action) in enumerate(
            ((0, probabilities[0], "reject"), (1, probabilities[1], "accept"))
        ):
            rows.append(
                {
                    "condition": "full_source",
                    "arm": arm,
                    "row_key": f"r-{index}",
                    "group_id": f"g-{index}",
                    "task_type": "QA",
                    "label": label,
                    "probability": probability,
                    "action": action,
                    "brier_contribution": (probability - label) ** 2,
                    "log_loss_contribution": audit.binary_log_loss(label, probability),
                }
            )
    return rows


def _online_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for arm in ("frozen_spline", "online_sparse_spline"):
        for seed in (1, 2):
            for index, (identity, label, probability, revealed) in enumerate(
                (("o-1", 0, 0.2, True), ("o-2", 1, 0.7, False))
            ):
                admitted = arm == "online_sparse_spline" and revealed
                rows.append(
                    {
                        "observation_id": identity,
                        "row_key": identity,
                        "group_id": identity,
                        "task_type": "QA",
                        "arm": arm,
                        "seed": seed,
                        "ordering": "hash_order",
                        "schedule": "uniform_eight",
                        "delay": 1,
                        "prediction_index": index,
                        "available_at": index + 1,
                        "delayed_arrival_index": index + 1 if revealed else None,
                        "probability": probability,
                        "label": label,
                        "label_authority": audit.LABEL_AUTHORITY,
                        "action": "escalate",
                        "revealed": revealed,
                        "propensity": 0.5,
                        "selection_component": "uniform_without_replacement",
                        "block_index": 0,
                        "block_size": 2,
                        "prediction_before_feedback": True,
                        "prediction_persisted": True,
                        "journal_duration_s": 0.001,
                        "update_admitted": admitted,
                        "feedback_status": "committed" if admitted else "not_revealed",
                        "parent_state_hash": f"parent-{arm}-{seed}-{index}",
                        "state_hash_at_prediction": f"parent-{arm}-{seed}-{index}",
                        "new_state_hash": (
                            f"new-{arm}-{seed}-{index}"
                            if admitted
                            else f"parent-{arm}-{seed}-{index}"
                        ),
                        "commit_after_prediction": True if revealed else None,
                        "brier_contribution": (probability - label) ** 2,
                        "log_loss_contribution": audit.binary_log_loss(label, probability),
                    }
                )
    return rows


def _branches() -> tuple[dict[str, object], dict[str, object]]:
    attacks = [
        {"attack": name, "passed": True, "observation": "mutation_rejected"}
        for name in audit.REQUIRED_ATTACKS
    ]
    static = {
        "branch": "static",
        "upstream": "exp7426-v651-static-decisions",
        "available": True,
        "eligible": True,
        "valid": True,
        "complete": True,
        "value": False,
        "producer_verdict_class": "null",
        "label_authority": audit.LABEL_AUTHORITY,
        "reduced_metrics": audit.reduce_static_metrics(_metric_rows()),
        "errors": [],
        "leakage_attack_rows": attacks[:5] + [attacks[-1]],
    }
    online = {
        "branch": "online",
        "upstream": "exp7427-v651-randomized-feedback",
        "available": True,
        "eligible": True,
        "valid": True,
        "complete": True,
        "value": False,
        "producer_verdict_class": "null",
        "label_authority": audit.LABEL_AUTHORITY,
        "reduced_metrics": audit.reduce_online_metrics(_online_rows()),
        "errors": [],
        "leakage_attack_rows": attacks[5:-1],
    }
    return static, online


def test_static_labels_splits_metrics_bounds_and_capacity_fail_closed() -> None:
    """SCENARIO-REPORT-7428-STATIC and SCENARIO-REPORT-7428-CAPACITY are independent."""

    predictors = _predictors()
    evaluators = _evaluators()
    assert audit.static_integrity_errors(predictors, evaluators) == []
    reduced = audit.reduce_static_metrics(_metric_rows())
    sparse = reduced["by_arm"]["sparse_spline_49"]
    assert sparse["rows"] == 2
    assert sparse["brier"] == pytest.approx(0.04)
    assert sparse["log_loss"] == pytest.approx(-math.log(0.8))
    assert sparse["accept_count"] == sparse["reject_count"] == 1
    bound = audit.clopper_pearson_upper(0, 10, 0.01)
    assert 0.0 < bound < 1.0
    parity = audit.spline_capacity_errors(
        [
            {
                "parameter_count": 49,
                "sparse_dense_parameter_gap": 0.0,
                "max_probability_gap": 0.0,
                "max_gradient_gap": 0.0,
            }
        ]
    )
    assert parity == []

    mutations = {
        "mislabeled_implicit_true": lambda p, e: e[0].update({"implicit_true_excluded_label": 0}),
        "cross_split_source_alias": lambda p, e: p[1].update({"source_text": p[0]["source_text"]}),
        "response_sibling_cross_split": lambda p, e: e[1].update(
            {"response_id": e[0]["response_id"]}
        ),
        "changed_annotation_offset": lambda p, e: e[0]["annotations"][0].update({"start": 1}),
        "annotation_field_in_predictor": lambda p, e: p[0].update({"primary_label": 0}),
    }
    for expected, mutate in mutations.items():
        changed_predictors, changed_evaluators = deepcopy(predictors), deepcopy(evaluators)
        mutate(changed_predictors, changed_evaluators)
        assert expected in audit.static_integrity_errors(changed_predictors, changed_evaluators)

    changed = _metric_rows()
    changed[0]["brier_contribution"] = 99.0
    assert "static_brier_contribution_mismatch:0" in audit.static_metric_errors(changed)
    assert audit.spline_capacity_errors(
        [{"parameter_count": 50, "sparse_dense_parameter_gap": 0.0}]
    ) == ["spline_parameter_count_mismatch:0", "spline_parity_fields_missing:0"]


def test_online_order_propensity_updates_and_metrics_fail_closed() -> None:
    """SCENARIO-REPORT-7428-ONLINE: ordering and reveal authority reject mutations."""

    rows = _online_rows()
    assert audit.online_integrity_errors(rows) == []
    reports = audit.reduce_online_metrics(rows)
    primary = next(row for row in reports if row["arm"] == "online_sparse_spline")
    assert primary["scored_rows"] == 4
    assert primary["revealed_groups"] == 1
    assert primary["ipw_brier"] == pytest.approx(0.04)

    mutations = {
        "future_label_access": lambda value: value[0].update(
            {"available_at": -1, "delayed_arrival_index": 0}
        ),
        "propensity_substitution": lambda value: value[0].update({"propensity": 0.75}),
        "duplicate_update": lambda value: value.append(deepcopy(value[5])),
        "hidden_full_feedback_access": lambda value: value[3].update(
            {"update_admitted": True, "feedback_status": "committed"}
        ),
        "omitted_persistence_time": lambda value: value[0].update({"journal_duration_s": None}),
    }
    for expected, mutate in mutations.items():
        changed = deepcopy(rows)
        mutate(changed)
        assert any(error.startswith(expected) for error in audit.online_integrity_errors(changed))


def test_numeric_update_lineage_and_revocation_replay_are_independent() -> None:
    """SCENARIO-REPORT-7428-ONLINE: numeric parents and cold replay reproduce."""

    initial = {"coef": [0.0] * 6, "bias": 0.0}
    calibration = {"affine": {"slope": 1.0, "intercept": 0.0}}
    features = {name: 0.5 for name in audit.SOURCE_FEATURE_NAMES}
    updated, receipt = audit.reapply_numeric_update(
        "online_raw_logistic", initial, calibration, features, 1
    )
    assert updated["bias"] > 0.0
    assert receipt["gradient_norm_after_clip"] <= 1.0
    restored = audit.replay_trusted_journal(
        "online_raw_logistic",
        initial,
        calibration,
        [
            {"event_id": "a", "features": features, "label": 1, "active": True},
            {"event_id": "b", "features": features, "label": 0, "active": False},
        ],
    )
    assert restored == updated

    lineage = [
        {
            "observation_id": "a",
            "label": 1,
            "arrival_index": 1,
            "update_count": 1,
            "parent_state_hash": audit.numeric_state_hash("online_raw_logistic", 1, initial, 0),
            "state_hash": audit.numeric_state_hash("online_raw_logistic", 1, updated, 1),
            "checkpoint": updated,
            "arm": "online_raw_logistic",
            "seed": 1,
        }
    ]
    assert (
        audit.lineage_errors(
            lineage,
            initial_states={(1, "online_raw_logistic"): initial},
            calibrations={(1, "online_raw_logistic"): calibration},
            features_by_observation={"a": features},
        )
        == []
    )
    changed = deepcopy(lineage)
    changed[0]["parent_state_hash"] = "wrong"
    assert "lineage_parent_hash_mismatch:0" in audit.lineage_errors(
        changed,
        initial_states={(1, "online_raw_logistic"): initial},
        calibrations={(1, "online_raw_logistic"): calibration},
        features_by_observation={"a": features},
    )


def test_branch_preconditions_and_terminal_classification_are_independent(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-7428-INDEPENDENT: an absent producer blocks only its branch."""

    definitions = audit.PRODUCERS
    for key in ("annotated", "spline", "static"):
        definition = definitions[key]
        path = tmp_path / definition["path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "schema": definition["schema"],
                    "experiment_id": definition["experiment_id"],
                    "milestone": audit.MILESTONE,
                    "flagged_adversarial": False,
                    "verdict_class": "null",
                    definition["completion_field"]: 1,
                    "gate_check_summary": {"required_checks_passed": True},
                }
            ),
            encoding="utf-8",
        )
    checks, hashes, producers = audit.collect_preconditions(tmp_path, verify_sources=False)
    assert producers["online"] == {}
    assert definitions["static"]["path"].as_posix() in hashes
    online_checks = [row for row in checks if row["branch"] == "online"]
    assert any(row["observed"] is None and not row["passed"] for row in online_checks)

    static, online = _branches()
    online = audit.blocked_branch("online", online_checks)
    classified = audit.classify_terminal(
        [static, online], required_validation_passed=True, own_work_complete=True
    )
    assert classified == {
        "honest_verdict": "blocked_online_external_producer_unavailable_static_audit_preserved",
        "verdict_class": "blocked",
        "status": "blocked_online_external_producer_unavailable_static_audit_preserved",
        "static_audit_complete_score": 1,
        "online_audit_complete_score": 0,
    }
    invalid = deepcopy(static)
    invalid["valid"] = False
    assert (
        audit.classify_terminal(
            [invalid, online], required_validation_passed=True, own_work_complete=True
        )["verdict_class"]
        == "disqualified"
    )
    assert (
        audit.classify_terminal(
            list(_branches()), required_validation_passed=True, own_work_complete=False
        )["verdict_class"]
        == "partial"
    )


def test_artifact_fields_checksum_and_cold_reader_fail_closed(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7428-ARTIFACT: terminal fields and checksum are strict."""

    static, online = _branches()
    artifact = audit.build_artifact_for_test(static, online)
    assert audit.validate_artifact(artifact, verify_source_bytes=False) == []
    assert artifact["MODEL_SPECS"] == []
    assert artifact["model_invoked"] is False
    assert artifact["invocation_counts"] == audit.ZERO_INVOCATION_COUNTS
    assert artifact["inference_substrate_class"] == "aggregation"
    assert artifact["execution_venue"] == "host"
    assert artifact["verdict_class"] == "null"
    assert artifact["promotion_score"] == 0
    assert artifact["static_audit_complete_score"] == 1
    assert artifact["online_audit_complete_score"] == 1
    assert set(artifact["field_principles"]) >= set(audit.REQUIRED_FIELDS)
    assert {row["attack"] for row in artifact["leakage_attack_rows"]} == set(audit.REQUIRED_ATTACKS)

    target = tmp_path / "artifact.json"
    audit.atomic_json(target, artifact)
    assert audit.cold_replay(target, verify_source_bytes=False) == []
    for field, value, expected in (
        ("schema", "wrong", "identity_mismatch"),
        ("MODEL_SPECS", ["archive"], "model_declaration_mismatch"),
        ("promotion_score", 1, "promotion_score_mismatch"),
        ("static_audit_complete_score", 0, "branch_score_mismatch"),
        ("verdict_class", "positive", "terminal_classification_mismatch"),
        ("field_principles", {}, "field_principles_incomplete"),
    ):
        changed = deepcopy(artifact)
        changed[field] = value
        assert expected in audit.validate_artifact(changed, verify_source_bytes=False)
    assert audit.validate_artifact([], verify_source_bytes=False) == ["artifact_not_object"]
    assert audit.cold_replay(tmp_path / "missing.json") == ["artifact_unreadable_or_not_object"]


def test_scoped_plan_and_thin_entrypoint_are_fixed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-REPORT-7428-ARTIFACT: only the frozen affected plan is accepted."""

    private = tmp_path / "private"
    commands = audit.build_validation_commands(Path.cwd(), private)
    assert [command.name for command in commands] == list(
        audit.validation_scope.REQUIRED_CHECK_NAMES
    )
    assert audit.validate_command_plan(Path.cwd(), audit.V651_MANIFEST, commands) == []
    argv = [argument for command in commands for argument in command.argv]
    assert "tests/python" not in argv
    assert any(argument.startswith("--basetemp=") for argument in argv)
    assert "full_python_suite" not in [command.name for command in commands]

    static, online = _branches()
    artifact = audit.build_artifact_for_test(static, online)
    target = tmp_path / "candidate.json"
    audit.atomic_json(target, artifact)
    assert audit.main(["--cold-replay", str(target), "--skip-source-bytes"]) == 0
    assert audit.main(["--independent-reduce", str(target), "--skip-source-bytes"]) == 0
    monkeypatch.setattr(audit, "independent_replay", lambda _path: [])
    assert audit.main(["--independent-reduce", str(target)]) == 0
    monkeypatch.setattr(audit, "run_experiment", lambda *args, **kwargs: artifact)
    assert audit.main(["--date", audit.RUN_DATE, "--output", str(target)]) == 0
    with pytest.raises(SystemExit, match="--date is required"):
        audit.main([])


def test_defensive_reducers_reject_malformed_operands(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-REPORT-7428: every independent reducer fails closed at its boundary."""

    with pytest.raises(ValueError, match="binary label"):
        audit.binary_log_loss(2, 0.5)
    with pytest.raises(ValueError, match="binomial counts"):
        audit.clopper_pearson_upper(2, 1, 0.05)
    assert audit.clopper_pearson_upper(4, 4, 0.05) == 1.0
    assert 0.0 < audit.clopper_pearson_upper(2, 10, 0.05) < 1.0

    predictors, evaluators = _predictors(), _evaluators()
    missing = deepcopy(evaluators[1:])
    assert "evaluator_row_missing:r-fit" in audit.static_integrity_errors(predictors, missing)
    changed = deepcopy(evaluators)
    changed[0]["group_id"] = "wrong"
    changed[1]["annotations"] = "bad"
    errors = audit.static_integrity_errors(predictors, changed)
    assert "predictor_evaluator_identity_mismatch:r-fit" in errors
    assert "annotation_shape_invalid:r-test" in errors
    changed = deepcopy(evaluators)
    changed[0]["annotations"] = [None]
    assert "changed_annotation_offset" in audit.static_integrity_errors(predictors, changed)
    changed_predictors = deepcopy(predictors)
    changed_predictors[1]["group_id"] = changed_predictors[0]["group_id"]
    assert "cross_split_source_alias" in audit.static_integrity_errors(
        changed_predictors, evaluators
    )

    malformed_metrics = deepcopy(_metric_rows())
    malformed_metrics[0]["log_loss_contribution"] = 99.0
    malformed_metrics[0]["action"] = "invented"
    malformed_metrics.append({"label": "bad"})
    metric_errors = audit.static_metric_errors(malformed_metrics)
    assert "static_log_loss_contribution_mismatch:0" in metric_errors
    assert "static_action_invalid:0" in metric_errors
    assert "static_metric_row_invalid:4" in metric_errors
    assert audit.spline_capacity_errors(
        [
            {
                "parameter_count": 49,
                "sparse_dense_parameter_gap": audit.PARITY_TOLERANCE * 2,
                "max_probability_gap": 0.0,
                "max_gradient_gap": 0.0,
            }
        ]
    ) == ["spline_parity_tolerance_exceeded:0"]

    top = [
        {
            "observation_id": str(index),
            "block_size": 2,
            "schedule": "top_risk_eight",
            "revealed": index == 0,
        }
        for index in range(2)
    ]
    assert audit._expected_propensity(top) == {"0": 1.0, "1": 0.0}
    hybrid = [
        {
            "observation_id": str(index),
            "block_size": 4,
            "schedule": "hybrid_four_plus_four",
            "revealed": index < 2,
            "selection_component": "deterministic_top" if index == 0 else "uniform_remainder",
        }
        for index in range(4)
    ]
    assert audit._expected_propensity(hybrid) == {
        "0": 1.0,
        "1": pytest.approx(1 / 3),
        "2": pytest.approx(1 / 3),
        "3": pytest.approx(1 / 3),
    }
    assert audit._expected_propensity([]) == {}
    unknown = deepcopy(top)
    for row in unknown:
        row["schedule"] = "unknown"
    assert audit._expected_propensity(unknown) == {}

    online = deepcopy(_online_rows())
    online[1]["revealed"] = True
    online[1]["propensity"] = 0.5
    online[1]["selection_component"] = "different"
    online[1]["prediction_before_feedback"] = False
    online[1]["new_state_hash"] = "changed-without-update"
    online[1]["delayed_arrival_index"] = "bad"
    online[2]["prediction_index"] = "bad"
    online[0]["schedule"] = "unknown"
    online[0]["block_size"] = 99
    online_errors = audit.online_integrity_errors(online)
    assert any(error.startswith("propensity_substitution") for error in online_errors)
    assert any(error.startswith("future_label_access") for error in online_errors)
    assert any(error.startswith("state_changed_without_update") for error in online_errors)
    assert any(error.startswith("propensity_block_invalid") for error in online_errors)
    early_arrival = deepcopy(_online_rows())
    early_arrival[0]["delayed_arrival_index"] = 0
    assert "future_label_access:0" in audit.online_integrity_errors(early_arrival)
    bad_propensity = deepcopy(_online_rows())
    bad_propensity[0]["propensity"] = "bad"
    assert "propensity_substitution:o-1" in audit.online_integrity_errors(bad_propensity)

    assert audit._sigmoid(-2.0) < 0.5
    with pytest.raises(ValueError, match="six finite features"):
        audit._feature_vector({name: math.nan for name in audit.SOURCE_FEATURE_NAMES})
    with pytest.raises(ValueError, match="registered learner"):
        audit.reapply_numeric_update("unknown", {}, {}, {}, 1)

    features = {name: 0.5 for name in audit.SOURCE_FEATURE_NAMES}
    calibration = {"affine": {"slope": 1.0, "intercept": -100.0}}
    knots = [[0.0] * 4 + [0.2, 0.4, 0.6, 0.8] + [1.0] * 4 for _ in range(6)]
    spline = {"coef": [[0.0] * 8 for _ in range(6)], "bias": 0.0, "knots": knots}
    gibbs = {
        "w1": [[0.0] * 6 for _ in range(4)],
        "b1": [0.0] * 4,
        "w_out": [0.0] * 4,
        "b_out": 0.0,
    }
    for arm, checkpoint in (
        ("online_sparse_spline", spline),
        ("online_gibbs", gibbs),
    ):
        updated, receipt = audit.reapply_numeric_update(arm, checkpoint, calibration, features, 0)
        assert updated != checkpoint or receipt["coefficient_change_l2"] == 0.0

    malformed = tmp_path / "malformed.json"
    malformed.write_text("{", encoding="utf-8")
    assert audit._load_object(malformed) == {}
    non_object = tmp_path / "list.json"
    non_object.write_text("[]", encoding="utf-8")
    assert audit._load_object(non_object) == {}

    receipt_names = audit.validation_scope.REQUIRED_CHECK_NAMES
    assert audit._producer_validation_passed(
        {
            "validation_receipts": [
                {"name": name, "required": True, "passed": True, "exit_code": 0}
                for name in receipt_names
            ]
        }
    )

    real_checks, _hashes, _producers = audit.collect_preconditions(Path.cwd())
    assert real_checks


def test_lineage_classification_and_artifact_defenses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-REPORT-7428: lineage and terminal mutations cannot become valid evidence."""

    initial = {"coef": [0.0] * 6, "bias": 0.0}
    calibration = {"affine": {"slope": 1.0, "intercept": 0.0}}
    features = {name: 0.5 for name in audit.SOURCE_FEATURE_NAMES}
    updated, _receipt = audit.reapply_numeric_update(
        "online_raw_logistic", initial, calibration, features, 1
    )
    event = {
        "observation_id": "a",
        "label": 1,
        "label_authority": audit.LABEL_AUTHORITY,
        "arrival_index": 1,
        "update_count": 2,
        "parent_state_hash": "wrong",
        "state_hash": "wrong",
        "checkpoint": initial,
        "event_hash": "wrong",
        "arm": "online_raw_logistic",
        "seed": 1,
    }
    times = iter((0.0, 61.0, 62.0))
    monkeypatch.setattr(audit.time, "monotonic", lambda: next(times))
    heartbeats: list[int] = []
    errors = audit.lineage_errors(
        [event],
        initial_states={(1, "online_raw_logistic"): initial},
        calibrations={(1, "online_raw_logistic"): calibration},
        features_by_observation={"a": features},
        heartbeat=heartbeats.append,
    )
    assert heartbeats == [1]
    assert {
        "lineage_parent_hash_mismatch:0",
        "lineage_state_hash_mismatch:0",
        "lineage_update_count_mismatch:0",
        "lineage_checkpoint_mismatch:0",
        "lineage_event_hash_mismatch:0",
    }.issubset(errors)
    monkeypatch.undo()
    assert "lineage_row_invalid:0" in audit.lineage_errors(
        [{"arm": "online_raw_logistic", "seed": 99}],
        initial_states={},
        calibrations={},
        features_by_observation={},
    )
    assert updated != initial

    static, online = _branches()
    no_validation = audit.classify_terminal(
        [static, online], required_validation_passed=False, own_work_complete=True
    )
    assert no_validation["verdict_class"] == "disqualified"
    positive = deepcopy(static)
    positive["value"] = True
    assert (
        audit.classify_terminal(
            [positive, online], required_validation_passed=True, own_work_complete=True
        )["verdict_class"]
        == "positive"
    )
    both_blocked = [
        audit.blocked_branch("static", []),
        audit.blocked_branch("online", []),
    ]
    assert (
        audit.classify_terminal(
            both_blocked, required_validation_passed=True, own_work_complete=True
        )["honest_verdict"]
        == "blocked_external_producer_unavailable"
    )
    online_only = deepcopy(online)
    assert (
        audit.classify_terminal(
            [both_blocked[0], online_only],
            required_validation_passed=True,
            own_work_complete=True,
        )["honest_verdict"]
        == "blocked_static_external_producer_unavailable_online_audit_preserved"
    )

    artifact = audit.build_artifact_for_test(static, online)
    changed = deepcopy(artifact)
    changed["leakage_attack_rows"] = []
    assert "leakage_attack_rows_incomplete" in audit.validate_artifact(
        changed, verify_source_bytes=False
    )
    changed = deepcopy(artifact)
    changed["verifier_is_oracle"] = True
    assert "oracle_declaration_mismatch" in audit.validate_artifact(
        changed, verify_source_bytes=False
    )
    changed = deepcopy(artifact)
    changed["source_artifact_hashes"] = {"bad": []}
    assert "source_hash_row_invalid:bad" in audit.validate_artifact(changed)
    changed["source_artifact_hashes"] = {"bad": {"path": "missing.json", "sha256": "sha256:no"}}
    assert "source_hash_mismatch:bad" in audit.validate_artifact(changed)
