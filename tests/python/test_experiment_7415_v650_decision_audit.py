"""Tests for the V650 independent decision-evidence audit.

Spec refs: REQ-REPORT-7415 and SCENARIO-REPORT-7415-INDEPENDENT through
SCENARIO-REPORT-7415-ARTIFACT.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from carnot import experiment_7415_v650_decision_audit as audit


def _static_producer() -> dict[str, object]:
    metric_rows: list[dict[str, object]] = []
    for arm, offset in (("source", 0.0), ("baseline", 0.1)):
        for seed in (1, 2):
            for group, label, probability in (("g1", 0, 0.1), ("g2", 1, 0.8)):
                value = min(0.99, probability + offset)
                metric_rows.append(
                    {
                        "arm": arm,
                        "condition": "full_source",
                        "seed": seed,
                        "group_id": group,
                        "row_key": f"{group}-{seed}",
                        "partition": "final_test",
                        "label": label,
                        "label_authority": "machine_annotation",
                        "probability": value,
                        "brier_contribution": (value - label) ** 2,
                        "log_loss_contribution": audit.binary_log_loss(label, value),
                        "decision": "accept" if value <= 0.2 else "reject",
                        "scored": True,
                    }
                )
            metric_rows.append(
                {
                    "arm": arm,
                    "condition": "full_source",
                    "seed": seed,
                    "group_id": "g3",
                    "row_key": f"g3-{seed}",
                    "partition": "final_test",
                    "label": None,
                    "label_authority": "machine_annotation",
                    "probability": None,
                    "brier_contribution": None,
                    "log_loss_contribution": None,
                    "decision": "unscored",
                    "scored": False,
                }
            )
    ablations = []
    for condition, probability in (("source_removed", 0.5), ("source_permuted", 0.6)):
        for seed in (1, 2):
            for group, label in (("g1", 0), ("g2", 1)):
                ablations.append(
                    {
                        "arm": "source",
                        "condition": condition,
                        "seed": seed,
                        "group_id": group,
                        "row_key": f"{group}-{seed}",
                        "partition": "final_test",
                        "label": label,
                        "label_authority": "machine_annotation",
                        "probability": probability,
                        "brier_contribution": (probability - label) ** 2,
                        "log_loss_contribution": audit.binary_log_loss(label, probability),
                        "decision": "escalate",
                        "scored": True,
                    }
                )
    return {
        "schema": audit.STATIC_SCHEMA,
        "experiment_id": "exp7413-source-calibration",
        "milestone": audit.MILESTONE,
        "status": "complete",
        "calibration_capture_complete_score": 1,
        "calibration_value_score": 0,
        "verdict_class": "null",
        "flagged_adversarial": False,
        "gate_check_summary": {"required_checks_passed": True},
        "label_authority": "machine_annotation",
        "rows": [
            {
                "comparative_unit": f"full_source:{arm}:{seed}",
                "condition": "full_source",
                "arm": arm,
                "seed": seed,
                "status": "completed",
            }
            for arm in ("source", "baseline")
            for seed in (1, 2)
        ],
        "paired_metric_rows": metric_rows,
        "source_ablation_rows": ablations,
        "checkpoint_manifest": [
            {
                "unit_id": f"full_source:{arm}:{seed}",
                "condition": "full_source",
                "arm": arm,
                "seed": seed,
                "selected_policy": {
                    "accept_enabled": True,
                    "accept_threshold": 0.2,
                    "reject_enabled": True,
                    "reject_threshold": 0.7,
                    "selection_partition": "policy_calibration",
                },
            }
            for arm in ("source", "baseline")
            for seed in (1, 2)
        ],
        "paired_group_bootstrap": {"contrasts": {}},
        "source_artifact_hashes": {},
        "feature_names": ["response_score", "source_score"],
        "partition_articles": {
            "train": ["a1"],
            "probability_calibration": ["a2"],
            "policy_calibration": ["a3"],
            "final_test": ["a4"],
        },
        "fit_label_partitions": ["train", "probability_calibration", "policy_calibration"],
        "official_row_count_per_unit": 3,
    }


def _online_producer() -> dict[str, object]:
    events: list[dict[str, object]] = []
    arms = ("adaptive", "frozen", "no_feedback")
    for arm in arms:
        prior = f"{arm}-s0"
        for index, observation_id in enumerate(("o1", "o2")):
            admitted = arm == "adaptive" and index == 1
            new = f"{arm}-s{index + 1}" if admitted else prior
            events.append(
                {
                    "arm": arm,
                    "seed": 1,
                    "ordering": "hash_order",
                    "feedback_regime": "selected",
                    "delay": 1,
                    "observation_id": observation_id,
                    "prediction_index": index,
                    "available_at": index + 1,
                    "reveal_visible_at": index + 1,
                    "registered_feedback_mask": True,
                    "diagnostic_learner_selected_mask": observation_id == "o2",
                    "prediction_before_feedback": True,
                    "commit_after_prediction": True,
                    "prediction_persisted": True,
                    "update_admitted": admitted,
                    "feedback_status": "committed" if admitted else "not_selected",
                    "prior_state_hash": prior,
                    "state_hash_at_prediction": prior,
                    "new_state_hash": new,
                    "probability": 0.2 + 0.1 * index,
                    "label": index,
                    "label_authority": "machine_annotation",
                    "brier_contribution": (0.2 + 0.1 * index - index) ** 2,
                    "log_loss_contribution": audit.binary_log_loss(index, 0.2 + 0.1 * index),
                }
            )
            prior = new
    return {
        "schema": audit.ONLINE_SCHEMA,
        "experiment_id": "exp7414-selected-feedback",
        "milestone": audit.MILESTONE,
        "status": "complete",
        "online_capture_complete_score": 1,
        "online_value_score": 0,
        "verdict_class": "null",
        "flagged_adversarial": False,
        "gate_check_summary": {"required_checks_passed": True},
        "label_authority": "machine_annotation",
        "rows": [
            {
                "comparative_unit": f"hash_order:selected:delay1:{arm}:1",
                "arm": arm,
                "seed": 1,
                "ordering": "hash_order",
                "feedback_regime": "selected",
                "delay": 1,
                "status": "completed",
            }
            for arm in arms
        ],
        "feedback_event_rows": events,
        "feedback_journal_manifest": [],
        "initial_state_manifest": [{"seed": 1, "sha256": "adaptive-s0"}],
        "final_state_hashes": {
            "adaptive:1:hash_order:selected:1": "adaptive-s2",
            "frozen:1:hash_order:selected:1": "frozen-s0",
            "no_feedback:1:hash_order:selected:1": "no_feedback-s0",
        },
        "paired_moving_block_intervals": [],
        "analytic_controls": {
            "erased_update_replay": {"passed": True},
            "no_feedback_equality": {"passed": True},
            "prediction_before_update": {"passed": True},
        },
        "revocation_rows": [
            {
                "operation": "replace_label",
                "trusted_journal_replayed": True,
                "state_hash_after": "replacement-state",
            }
        ],
        "source_artifact_hashes": {},
    }


def test_static_reduction_and_leakage_attacks_are_independent() -> None:
    """SCENARIO-REPORT-7415-STATIC: raw scores and every leakage attack reproduce."""

    producer = _static_producer()
    result = audit.audit_static_branch(producer)
    assert result["available"] is True
    assert result["valid"] is True
    assert result["complete"] is True
    assert result["value"] is False
    assert result["label_authority"] == "machine_annotation"
    assert result["proper_scores"]["source"]["brier"] == pytest.approx(0.025)
    assert result["proper_scores"]["source"]["log_loss"] > 0.0
    assert result["unscored_examples"]["expected"] == 4
    assert result["unscored_examples"]["observed"] == 4
    assert all(row["passed"] for row in result["leakage_attack_rows"])
    assert {row["attack"] for row in result["leakage_attack_rows"]} == set(audit.LEAKAGE_ATTACKS)
    assert result["threshold_recomputation"]["mismatches"] == []

    mutations = {
        "oracle_label_feature": lambda value: value.update(
            {"feature_names": ["response_score", "label"]}
        ),
        "article_overlap": lambda value: value["partition_articles"]["final_test"].append("a1"),
        "test_label_exposure": lambda value: value["fit_label_partitions"].append("final_test"),
        "unscored_omission": lambda value: value.update({"official_row_count_per_unit": 4}),
    }
    for attack, mutate in mutations.items():
        changed = deepcopy(producer)
        mutate(changed)
        reduced = audit.audit_static_branch(changed)
        row = next(item for item in reduced["leakage_attack_rows"] if item["attack"] == attack)
        assert row["passed"] is False
        assert reduced["valid"] is False


def test_online_reduction_rejects_mask_order_update_and_state_mutations() -> None:
    """SCENARIO-REPORT-7415-ONLINE: masks, delayed updates, and hashes fail closed."""

    producer = _online_producer()
    result = audit.audit_online_branch(producer)
    assert result["available"] is True
    assert result["valid"] is True
    assert result["complete"] is True
    assert result["value"] is False
    assert result["event_count"] == 6
    assert result["admitted_update_count"] == 1
    assert result["shared_mask_verified"] is True
    assert result["no_feedback_control_verified"] is True
    assert result["erased_update_control_verified"] is True
    assert result["revocation_reconstruction_verified"] is True

    mutations = {
        "mask": lambda value: value["feedback_event_rows"][2].update(
            {"registered_feedback_mask": False}
        ),
        "early": lambda value: value["feedback_event_rows"][1].update({"available_at": 0}),
        "chain": lambda value: value["feedback_event_rows"][1].update(
            {"prior_state_hash": "wrong"}
        ),
        "double": lambda value: value["feedback_event_rows"][0].update(
            {"update_admitted": True, "new_state_hash": "adaptive-s1"}
        ),
        "final": lambda value: value.update({"final_state_hashes": {}}),
        "revocation": lambda value: value.update({"revocation_rows": []}),
    }
    for name, mutate in mutations.items():
        changed = deepcopy(producer)
        mutate(changed)
        reduced = audit.audit_online_branch(changed)
        assert reduced["valid"] is False, name
        assert reduced["errors"], name


def test_branch_preconditions_and_terminal_classification_stay_separate(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7415-INDEPENDENT: one absent branch does not hide the other."""

    static_path = tmp_path / audit.STATIC_PATH
    static_path.parent.mkdir(parents=True)
    static_path.write_text(json.dumps(_static_producer()), encoding="utf-8")
    checks, hashes, producers = audit.collect_preconditions(
        tmp_path,
        expected_hashes={audit.STATIC_PATH: audit.sha256_file(static_path)},
        verify_sources=False,
    )
    static_checks = [row for row in checks if row["branch"] == "static"]
    online_checks = [row for row in checks if row["branch"] == "online"]
    assert all(row["passed"] for row in static_checks)
    assert any(not row["passed"] for row in online_checks)
    assert audit.STATIC_PATH.as_posix() in hashes
    assert producers["static"]["experiment_id"] == "exp7413-source-calibration"
    assert producers["online"] == {}

    static = audit.audit_static_branch(producers["static"])
    online = audit.blocked_branch("online", online_checks)
    classified = audit.classify_terminal(
        [static, online], required_validation_passed=True, own_work_complete=True
    )
    assert classified["verdict_class"] == "blocked"
    assert classified["honest_verdict"].startswith("blocked_")
    assert classified["static_audit_complete_score"] == 1
    assert classified["online_audit_complete_score"] == 0

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
            [static, audit.audit_online_branch(_online_producer())],
            required_validation_passed=True,
            own_work_complete=False,
        )["verdict_class"]
        == "partial"
    )
    assert (
        audit.classify_terminal(
            [static, audit.audit_online_branch(_online_producer())],
            required_validation_passed=False,
            own_work_complete=True,
        )["verdict_class"]
        == "disqualified"
    )
    assert (
        audit.classify_terminal(
            [static, audit.audit_online_branch(_online_producer())],
            required_validation_passed=True,
            own_work_complete=True,
        )["verdict_class"]
        == "null"
    )


def test_artifact_validation_checksum_and_closed_fields() -> None:
    """SCENARIO-REPORT-7415-CLASSIFY: ordinary fields preserve honest nulls."""

    artifact = audit.build_artifact_for_test(
        audit.audit_static_branch(_static_producer()),
        audit.audit_online_branch(_online_producer()),
    )
    assert audit.validate_artifact(artifact, verify_source_bytes=False) == []
    assert artifact["MODEL_SPECS"] == []
    assert artifact["model_invoked"] is False
    assert artifact["invocation_counts"] == audit.ZERO_INVOCATION_COUNTS
    assert artifact["inference_substrate_class"] == "aggregation"
    assert artifact["execution_venue"] == "host"
    assert artifact["static_audit_complete_score"] == 1
    assert artifact["online_audit_complete_score"] == 1
    assert artifact["promotion_score"] == 0
    assert artifact["verdict_class"] == "null"
    assert artifact["honest_verdict"].startswith("complete_")
    assert len(artifact["branch_rows"]) == 2
    assert {row["attack"] for row in artifact["leakage_attack_rows"]} == set(audit.LEAKAGE_ATTACKS)

    for field, value, expected in (
        ("schema", "wrong", "identity_mismatch"),
        ("MODEL_SPECS", ["historical"], "model_declaration_mismatch"),
        ("promotion_score", 1, "promotion_score_mismatch"),
        ("static_audit_complete_score", 0, "branch_score_mismatch"),
        ("verdict_class", "positive", "terminal_classification_mismatch"),
        ("field_principles", {}, "field_principles_incomplete"),
    ):
        changed = deepcopy(artifact)
        changed[field] = value
        assert expected in audit.validate_artifact(changed, verify_source_bytes=False)
    assert audit.validate_artifact([], verify_source_bytes=False) == ["artifact_not_object"]


def test_scoped_plan_atomic_io_and_cold_entrypoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-REPORT-7415-ARTIFACT: exact scoped checks and cold replay stay fixed."""

    private = tmp_path / "private"
    commands = audit.build_validation_commands(Path.cwd(), private)
    assert [command.name for command in commands] == list(
        audit.validation_scope.REQUIRED_CHECK_NAMES
    )
    assert audit.validate_command_plan(Path.cwd(), audit.V650_MANIFEST, commands) == []
    argv = [argument for command in commands for argument in command.argv]
    assert "tests/python" not in argv
    assert any(argument.startswith("--basetemp=") for argument in argv)
    assert "full_python_suite" not in [command.name for command in commands]

    target = tmp_path / "audit.json"
    artifact = audit.build_artifact_for_test(
        audit.audit_static_branch(_static_producer()),
        audit.audit_online_branch(_online_producer()),
    )
    audit.atomic_json(target, artifact)
    assert audit.cold_replay(target, verify_source_bytes=False) == []
    assert audit.main(["--cold-replay", str(target), "--skip-source-bytes"]) == 0
    assert audit.main(["--cold-replay", str(target), "--skip-source-bytes", "--preterminal"]) == 0
    assert (
        audit.parse_args(["--date", audit.RUN_DATE, "--output", str(target)]).date == audit.RUN_DATE
    )
    monkeypatch.setattr(audit, "run_experiment", lambda *args, **kwargs: artifact)
    assert audit.main(["--date", audit.RUN_DATE, "--output", str(target)]) == 0
    with pytest.raises(SystemExit, match="--date is required"):
        audit.main([])


def test_real_producer_bytes_recompute_all_registered_evidence() -> None:
    """SCENARIO-REPORT-7415-STATIC/ONLINE: shipped raw bytes reproduce independently."""

    checks, hashes, producers = audit.collect_preconditions(Path.cwd())
    assert (
        hashes[audit.STATIC_PATH.as_posix()]["sha256"]
        == audit.EXPECTED_PRODUCER_HASHES[audit.STATIC_PATH]
    )
    assert (
        hashes[audit.ONLINE_PATH.as_posix()]["sha256"]
        == audit.EXPECTED_PRODUCER_HASHES[audit.ONLINE_PATH]
    )
    assert all(row["passed"] for row in checks)
    assert any(row.get("authenticated_from") == "tracked_historical_bytes" for row in checks)
    by_branch = {
        branch: [row for row in checks if row["branch"] == branch]
        for branch in ("static", "online")
    }
    static = audit.audit_static_branch(
        audit._augment_static_context(Path.cwd(), producers["static"]),
        precondition_rows=by_branch["static"],
    )
    online = audit.audit_online_branch(producers["online"], precondition_rows=by_branch["online"])
    assert static["valid"] is True
    assert static["bootstrap_recomputation"]["mismatches"] == []
    assert static["threshold_recomputation"]["checked_rows"] == 39900
    assert online["valid"] is True
    assert online["event_count"] == 48800
    assert online["interval_recomputation"]["mismatches"] == []


def test_defensive_boundaries_fail_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-REPORT-7415-ARTIFACT: malformed private evidence fails its boundary."""

    with pytest.raises(ValueError, match="binary label"):
        audit.binary_log_loss(2, 0.5)
    missing = tmp_path / "missing.json"
    assert audit._load_object(missing) == {}
    missing.write_text("[]", encoding="utf-8")
    assert audit._load_object(missing) == {}
    missing.write_text("{", encoding="utf-8")
    assert audit._load_object(missing) == {}
    assert audit._close(None, None)
    assert not audit._close(None, 0)
    assert audit._close(True, True)
    assert not audit._close(float("nan"), float("nan"))
    assert audit._close("same", "same")

    monkeypatch.setattr(audit, "_historical_git_hash_exists", lambda *args: False)
    source_checks = audit._source_hash_checks(
        tmp_path,
        "static",
        {
            "source_artifact_hashes": {
                "bad": "not-a-row",
                "missing": {"path": "missing.txt", "sha256": "sha256:missing"},
            }
        },
    )
    assert [row["passed"] for row in source_checks] == [False, False]

    policies, errors = audit._checkpoint_policies(
        {"checkpoint_manifest": ["bad", {"path": "missing"}]}, lambda row: {}
    )
    assert policies == {}
    assert errors == ["checkpoint_manifest_row_invalid", "checkpoint_policy_missing:missing"]
    assert audit._static_checkpoint_loader({"path": "missing"}) == {}

    static = _static_producer()
    static["paired_metric_rows"][0]["decision"] = "reject"
    changed_static = audit.audit_static_branch(static)
    assert "static_threshold_decision_mismatch" in changed_static["errors"]
    static = _static_producer()
    static["rows"][0]["status"] = "failed"
    assert "static_unit_accounting_incomplete" in audit.audit_static_branch(static)["errors"]
    static = _static_producer()
    static["schema"] = "wrong"
    failed_precondition = [{"passed": False, "upstream": "x", "path": "x"}]
    reduced = audit.audit_static_branch(static, precondition_rows=failed_precondition)
    assert "static_producer_ineligible" in reduced["errors"]
    assert "static_precondition_failed" in reduced["errors"]
    assert audit.audit_static_branch({})["available"] is False

    online = _online_producer()
    online["feedback_event_rows"][0]["prediction_index"] = 4
    online["feedback_event_rows"][0]["prediction_persisted"] = False
    online["feedback_event_rows"][0]["new_state_hash"] = "changed-without-update"
    online["feedback_event_rows"][0].pop("state_hash_at_prediction")
    errors = audit.audit_online_branch(online)["errors"]
    assert any(value.startswith("event_order_mismatch") for value in errors)
    assert any(value.startswith("prediction_commit_order") for value in errors)
    assert any(value.startswith("state_hash_missing") for value in errors)
    assert any(value.startswith("state_changed_without_update") for value in errors)

    online = _online_producer()
    duplicate = deepcopy(online["feedback_event_rows"][1])
    duplicate["prediction_index"] = 2
    duplicate["available_at"] = 3
    duplicate["reveal_visible_at"] = 3
    duplicate["prior_state_hash"] = duplicate["state_hash_at_prediction"] = "adaptive-s2"
    duplicate["new_state_hash"] = "adaptive-s3"
    online["feedback_event_rows"].append(duplicate)
    assert any(
        value.startswith("duplicate_update")
        for value in audit.audit_online_branch(online)["errors"]
    )

    online = _online_producer()
    online["schema"] = "wrong"
    online["analytic_controls"]["no_feedback_equality"]["passed"] = False
    online["rows"][0]["status"] = "failed"
    online["condition_reports"] = [
        {
            **audit._online_reports(online["feedback_event_rows"])[0],
            "brier": 99.0,
        }
    ]
    online["paired_moving_block_intervals"] = [
        {"feedback_regime": "missing", "control_arm": "missing", "estimate": 1.0}
    ]
    reduced = audit.audit_online_branch(online, precondition_rows=failed_precondition)
    assert "online_producer_ineligible" in reduced["errors"]
    assert "online_precondition_failed" in reduced["errors"]
    assert "online_control_failed:no_feedback_equality" in reduced["errors"]
    assert "online_unit_accounting_incomplete" in reduced["errors"]
    assert "online_interval_estimate_mismatch" in reduced["errors"]
    assert any(value.startswith("online_metric_mismatch") for value in reduced["errors"])
    assert audit.audit_online_branch({})["available"] is False

    positive_static = audit.audit_static_branch(_static_producer())
    positive_online = audit.audit_online_branch(_online_producer())
    positive_static["value"] = positive_online["value"] = True
    assert (
        audit.classify_terminal(
            [positive_static, positive_online],
            required_validation_passed=True,
            own_work_complete=True,
        )["verdict_class"]
        == "positive"
    )

    artifact = audit.build_artifact_for_test(
        audit.audit_static_branch(_static_producer()),
        audit.audit_online_branch(_online_producer()),
    )
    artifact["verifier_is_oracle"] = True
    artifact["leakage_attack_rows"] = []
    artifact["source_artifact_hashes"] = {
        "bad": "bad",
        "missing": {"path": "missing", "sha256": "sha256:missing"},
    }
    validation_errors = audit.validate_artifact(artifact, root=tmp_path)
    assert "oracle_declaration_mismatch" in validation_errors
    assert "leakage_attack_rows_incomplete" in validation_errors
    assert "source_hash_row_invalid" in validation_errors
    assert "source_hash_mismatch:missing" in validation_errors
    assert audit.cold_replay(tmp_path / "absent.json") == ["artifact_unreadable_or_not_object"]

    monkeypatch.setattr(audit, "independent_replay", lambda path: [])
    assert audit.main(["--independent-reduce", str(tmp_path / "candidate.json")]) == 0


def test_numeric_and_history_mismatch_paths_are_observable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-REPORT-7415-STATIC: numeric and historical mismatches stay explicit."""

    monkeypatch.setattr(
        audit.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=1, stdout=""),
    )
    assert audit._historical_git_hash_exists(Path.cwd(), Path("missing"), "sha256:x") is False
    calls = iter(
        (
            SimpleNamespace(returncode=0, stdout="commit\n"),
            SimpleNamespace(returncode=1, stdout=b""),
        )
    )
    monkeypatch.setattr(audit.subprocess, "run", lambda *args, **kwargs: next(calls))
    assert audit._historical_git_hash_exists(Path.cwd(), Path("missing"), "sha256:x") is False

    assert (
        audit._recompute_threshold_decisions(
            [{"arm": "x", "seed": 1, "condition": "x", "probability": 0.5}], {}
        )["checked_rows"]
        == 0
    )

    row = {
        "condition": "full_source",
        "seed": 1,
        "group_id": "g",
        "row_key": "r",
        "scored": True,
        "label": 0,
        "probability": 0.2,
        "brier_contribution": 0.04,
        "log_loss_contribution": audit.binary_log_loss(0, 0.2),
        "decision": "accept",
    }
    bootstrap = {
        "paired_metric_rows": [
            {**row, "arm": "source_aware_6_4_1_gibbs"},
            {**row, "arm": "baseline"},
        ],
        "paired_group_bootstrap": {
            "draws": 2,
            "seed": 1,
            "upper_critical_value": 99.0,
            "contrasts": {
                "baseline": {
                    "brier_delta": {
                        "mean": 99.0,
                        "simultaneous_ci95": [99.0, 99.0],
                        "marginal_ci95": [99.0, 99.0],
                    },
                    "log_loss_delta": {"mean": 99.0, "ci95": [99.0, 99.0]},
                    "coverage_delta": {"mean": 99.0, "ci95": [99.0, 99.0]},
                },
                "missing": {},
            },
        },
    }
    proper = {
        "source_aware_6_4_1_gibbs": {},
        "baseline": {},
        "ghost_without_group_rows": {},
    }
    reduced = audit._bootstrap_summary(bootstrap, proper)
    assert "contrast:baseline" in reduced["mismatches"]
    assert "missing_vector:missing" in reduced["mismatches"]
    assert "upper_critical_value" in reduced["mismatches"]
    no_draws = deepcopy(bootstrap)
    no_draws["paired_group_bootstrap"]["draws"] = 0
    assert audit._bootstrap_summary(no_draws, proper)["mismatches"] == [
        "bootstrap_protocol_missing"
    ]

    static = _static_producer()
    static["paired_metric_rows"] = []
    assert "static_scored_rows_missing" in audit.audit_static_branch(static)["errors"]
    static = _static_producer()
    static["calibration_metrics"] = {
        "by_arm": {"source": {"brier": 99.0, "log_loss": 99.0, "coverage": 99.0}}
    }
    assert any(
        value.startswith("static_metric_mismatch")
        for value in audit.audit_static_branch(static)["errors"]
    )
    monkeypatch.setattr(
        audit,
        "_bootstrap_summary",
        lambda *args, **kwargs: {"mismatches": ["forced"]},
    )
    assert (
        "static_bootstrap_contrast_mismatch"
        in audit.audit_static_branch(_static_producer())["errors"]
    )

    online = _online_producer()
    online["feedback_event_rows"] = []
    online.pop("final_state_hashes")
    assert "online_final_state_missing" in audit.audit_online_branch(online)["errors"]
