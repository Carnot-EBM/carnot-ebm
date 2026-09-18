"""Tests for the V649 static calibrated-decision diagnosis.

Spec refs: REQ-REPORT-7396 and SCENARIO-REPORT-7396-*.
"""

from __future__ import annotations

from copy import deepcopy
import math
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_7396_v649_decision_diagnosis as mod


ROOT = Path(__file__).resolve().parents[2]


def _score_row(
    arm: str,
    seed: int,
    group: str,
    label: int,
    raw_energy: float,
    probability: float,
    decision: str,
) -> dict[str, Any]:
    """Build one complete scored row for independent reducer tests."""

    clipped = min(max(probability, 1e-15), 1.0 - 1e-15)
    return {
        "unit_id": f"{arm}:{seed}",
        "arm": arm,
        "seed": seed,
        "source_row_index": int(group.removeprefix("g")),
        "group_id": group,
        "label": label,
        "raw_energy": raw_energy,
        "probability": probability,
        "decision": decision,
        "brier_contribution": (probability - label) ** 2,
        "log_loss_contribution": -(label * math.log(clipped) + (1 - label) * math.log1p(-clipped)),
        "measured_cost": {"cpu_scoring_duration_s": 0.001, "current_llm_calls": 0},
    }


def _command_receipt(name: str, passed: bool = True) -> dict[str, Any]:
    """Build one exact receipt consumed by the cold artifact validator."""

    row: dict[str, Any] = {
        "name": name,
        "command": f"check {name}",
        "command_argv": ["check", name],
        "command_environment": {"COVERAGE_FILE": "/tmp/exp7396/.coverage"},
        "scope": "unit_fixture",
        "exit_code": 0 if passed else 1,
        "duration_s": 0.01,
        "log_path": f"/tmp/{name}.log",
        "log_sha256": "sha256:" + "1" * 64,
        "passed": passed,
        "timed_out": False,
        "output_tail": "ok" if passed else "failed",
        "required": True,
    }
    if name == "worktree_imports":
        row["resolved_imports"] = {
            "carnot.experiment_7396_v649_decision_diagnosis": str(
                (ROOT / mod.MODULE_PATH).resolve()
            )
        }
    return row


def _passing_receipts(terminal: bool = True) -> list[dict[str, Any]]:
    """Return every frozen affected and terminal receipt."""

    names = list(mod.AFFECTED_CHECK_NAMES)
    if terminal:
        names.extend(mod.TERMINAL_CHECK_NAMES)
    return [_command_receipt(name) for name in names]


def test_scenario_report_7396_metrics_recomputes_proper_scores_and_actions() -> None:
    """SCENARIO-REPORT-7396-METRICS recomputes rows without the producer reducer."""

    rows = [
        _score_row("learned", 1, "g0", 0, -2.0, 0.1, "accept"),
        _score_row("learned", 1, "g1", 1, 2.0, 0.8, "reject"),
        _score_row("learned", 1, "g2", 0, -1.0, 0.4, "escalate"),
        _score_row("learned", 1, "g3", 1, 1.0, 0.7, "accept"),
    ]
    reduced = mod.recompute_static_metrics(rows, draws=200, seed=7396307)
    unit = reduced["by_arm_seed"]["learned:1"]
    assert unit["brier"] == pytest.approx((0.01 + 0.04 + 0.16 + 0.09) / 4)
    assert unit["pr_auc"] == pytest.approx(1.0)
    assert unit["action_confusion"] == {
        "accept_correct": 1,
        "accept_incorrect": 1,
        "reject_correct": 1,
        "reject_incorrect": 0,
        "escalate_label_correct": 1,
        "escalate_label_incorrect": 0,
    }
    assert unit["calibration_bounds"]
    assert reduced["row_integrity_errors"] == []


def test_scenario_report_7396_metrics_controls_detect_signal_and_permutation() -> None:
    """SCENARIO-REPORT-7396-METRICS proves the reducer detects informative scores."""

    controls = mod.run_reducer_controls()
    assert controls["analytic_informative"]["pr_auc"] == 1.0
    assert controls["analytic_informative"]["brier"] < controls["constant"]["brier"]
    assert controls["permuted_labels"]["pr_auc"] < controls["analytic_informative"]["pr_auc"]
    assert controls["all_passed"] is True


def test_scenario_report_7396_features_reports_sample_collision_floor() -> None:
    """SCENARIO-REPORT-7396-FEATURES keeps collision floors descriptive."""

    features = [
        {
            "source_row_index": 0,
            "group_id": "a",
            "partition": "training",
            "entity_uptake": 0.5,
            "falsifiability_score": 0.0,
            "label": 0,
            "normalized_text_sha256": "sha256:a",
        },
        {
            "source_row_index": 1,
            "group_id": "b",
            "partition": "training",
            "entity_uptake": 0.5,
            "falsifiability_score": 0.0,
            "label": 1,
            "normalized_text_sha256": "sha256:b",
        },
        {
            "source_row_index": 2,
            "group_id": "c",
            "partition": "probability_calibration",
            "entity_uptake": 1.0,
            "falsifiability_score": 1.0,
            "label": 1,
            "normalized_text_sha256": "sha256:c",
        },
        {
            "source_row_index": 3,
            "group_id": "held-out",
            "partition": "final_test",
            "entity_uptake": 0.5,
            "falsifiability_score": 0.0,
            "normalized_text_sha256": "sha256:d",
        },
    ]
    report = mod.diagnose_feature_cells(features, permutation_draws=20, seed=7396307)
    assert report["diagnostic_partitions"] == ["training", "probability_calibration"]
    assert report["excluded_partition_rows"] == 1
    assert report["feature_collision_rows"] == [
        {
            "partition": "training",
            "vector": [0.5, 0.0],
            "vector_hash": mod.canonical_hash([0.5, 0.0]),
            "label_counts": {"correct": 1, "incorrect": 1},
            "sample_count": 2,
            "empirical_in_cell_probability": 0.5,
            "empirical_bayes_brier_floor": 0.25,
            "scope": "descriptive_sample_floor_not_population_impossibility",
        }
    ]
    assert report["group_partition_overlap_count"] == 0
    assert report["future_label_access"] is False


def test_scenario_report_7396_context_records_first_half_fallback() -> None:
    """SCENARIO-REPORT-7396-CONTEXT does not relabel IDs as source questions."""

    features = [
        {
            "source_row_index": 0,
            "partition": "training",
            "entity_uptake": 0.0,
            "falsifiability_score": 0.0,
            "label": 0,
        },
        {
            "source_row_index": 1,
            "partition": "probability_calibration",
            "entity_uptake": 0.0,
            "falsifiability_score": 0.0,
            "label": 1,
        },
    ]
    corpus = [
        {"question_id": "10", "step_text": "Work 2 plus 2. Therefore the answer is 4."},
        {"question_id": "11", "step_text": "Only a computation trail with 8 and 4."},
    ]
    report = mod.trace_context_proxies(features, corpus)
    assert report["pcib_context_argument"] == ""
    assert report["entity_context_proxy"] == "first_half_of_response_text"
    assert report["actual_source_question_available"] is False
    assert report["question_id_establishes_question_provenance"] is False
    assert report["rows_without_conclusion_markers"] == 1
    assert report["entity_uptake_constant"] is True
    assert report["unsupported_arithmetic_false_positive_claims"] == 0


def test_scenario_report_7396_decisions_keeps_answer_identity_fixed() -> None:
    """SCENARIO-REPORT-7396-DECISIONS separates binary and policy changes."""

    rows = [
        _score_row("arm", 1, "g0", 0, 2.0, 0.2, "accept"),
        _score_row("arm", 1, "g1", 1, -2.0, 0.8, "escalate"),
        _score_row("arm", 1, "g2", 0, -2.0, 0.2, "accept"),
    ]
    report = mod.decision_change_counts(rows)
    unit = report[0]
    assert unit["answer_identity_changed_count"] == 0
    assert unit["raw_to_calibrated_binary_changes"] == 2
    assert unit["calibrated_binary_to_typed_action_changes"] == 1
    assert unit["typed_action_counts"] == {"accept": 2, "reject": 0, "escalate": 1}
    assert unit["accepts_nearly_everything"] is False
    assert unit["majority_acceptance_is_learned_value"] is False


def test_scenario_report_7396_preconditions_authenticate_static_inputs_only() -> None:
    """SCENARIO-REPORT-7396-PRECONDITIONS excludes Exp7386 from static gates."""

    checks, hashes, inputs = mod.collect_preconditions(ROOT)
    assert checks
    assert all(row["passed"] for row in checks)
    assert mod.EXP7382_PATH.as_posix() in hashes
    assert mod.EXP7385_PATH.as_posix() in hashes
    assert set(inputs) == {"protocol", "training"}
    assert all("7386" not in str(row["upstream"]) for row in checks)


def test_scenario_report_7396_preconditions_block_exact_failed_field() -> None:
    """SCENARIO-REPORT-7396-PRECONDITIONS names a changed upstream field."""

    protocol = {"experiment_id": "wrong"}
    training = {"experiment_id": "exp7385-decision-training"}
    checks = mod.static_metadata_checks(protocol, training)
    failed = next(row for row in checks if not row["passed"])
    artifact = mod.build_blocked_artifact([failed], {})
    summary = artifact["gate_check_summary"]["first_required_failure"]
    assert summary["upstream"] == mod.EXP7382_PATH.as_posix()
    assert summary["artifact_field"] == "experiment_id"
    assert summary["observed"] == "wrong"
    assert artifact["verdict_class"] == "blocked"
    assert artifact["honest_verdict"].startswith("blocked_")


def test_scenario_report_7396_artifact_rejects_mutations() -> None:
    """SCENARIO-REPORT-7396-ARTIFACT cold reduction catches material drift."""

    artifact = mod.build_artifact_for_test(_passing_receipts())
    assert mod.validate_artifact(artifact) == []
    assert artifact["static_audit_complete_score"] == 1
    assert artifact["static_value_confirmed_score"] == 0
    assert artifact["verdict_class"] == "null"
    for field, value, expected in (
        ("execution_venue", "host_cpu", "execution_venue_invalid"),
        ("static_value_confirmed_score", 1, "static_value_confirmed_score_mismatch"),
        ("reproducibility_checksum", "sha256:bad", "reproducibility_checksum_mismatch"),
    ):
        changed = deepcopy(artifact)
        changed[field] = value
        assert expected in mod.validate_artifact(changed)


def test_scenario_report_7396_artifact_disqualifies_affected_failure() -> None:
    """SCENARIO-REPORT-7396-ARTIFACT keeps a genuine affected failure terminal."""

    receipts = _passing_receipts()
    receipt = next(row for row in receipts if row["name"] == "focused_pytest")
    receipt.update(_command_receipt("focused_pytest", passed=False))
    artifact = mod.build_artifact_for_test(receipts)
    assert artifact["verdict_class"] == "disqualified"
    assert artifact["static_audit_complete_score"] == 0
    assert mod.validate_artifact(artifact) == []


def test_scenario_report_7396_artifact_freezes_scoped_validation(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7396-ARTIFACT uses the Exp7358 and Exp7303 boundary."""

    commands = mod.build_validation_plan(ROOT, tmp_path)
    assert [command.name for command in commands] == list(mod.AFFECTED_CHECK_NAMES)
    assert mod.validate_validation_plan(ROOT, commands) == []
    focused = next(command for command in commands if command.name == "focused_pytest")
    assert ("-n", "0") == (focused.argv[1], focused.argv[2])
    assert "addopts=" in focused.argv
    assert "--no-cov" in focused.argv
    assert any(argument.startswith("--basetemp=") for argument in focused.argv)
    assert all("full_python_suite" not in command.name for command in commands)


def test_req_report_7396_pr_auc_and_bounds_fail_closed() -> None:
    """REQ-REPORT-7396 handles ties, missing classes, and invalid probabilities."""

    assert mod.binary_pr_auc([0, 1], [0.5, 0.5]) == pytest.approx(0.5)
    with pytest.raises(ValueError, match="positive"):
        mod.binary_pr_auc([0, 0], [0.1, 0.2])
    with pytest.raises(ValueError, match="probability"):
        mod.recompute_static_metrics(
            [_score_row("arm", 1, "g0", 0, 0.0, 1.2, "accept")], draws=10, seed=1
        )


def test_req_report_7396_paired_intervals_and_metric_comparison() -> None:
    """REQ-REPORT-7396 independently compares registered group intervals."""

    rows: list[dict[str, Any]] = []
    probabilities = {
        "training_prevalence": (0.2, 0.2),
        "l2_logistic_calibration": (0.1, 0.8),
        "natural_prevalence_bernoulli_gibbs": (0.05, 0.9),
    }
    for arm, values in probabilities.items():
        rows.extend(
            (
                _score_row(arm, 1, "g0", 0, -1.0, values[0], "accept"),
                _score_row(arm, 1, "g1", 1, 1.0, values[1], "reject"),
            )
        )
    reduced = mod.recompute_static_metrics(rows, draws=20, seed=7)
    comparison = reduced["paired_group_intervals"]["natural_prevalence_bernoulli_gibbs"][
        "training_prevalence"
    ]
    assert comparison["draws"] == 20
    assert comparison["brier_delta"]["mean"] < 0

    stored = {
        "calibration_metrics": {
            "by_arm_seed": deepcopy(reduced["by_arm_seed"]),
            "by_arm": deepcopy(reduced["by_arm"]),
        },
        "paired_group_intervals": deepcopy(reduced["paired_group_intervals"]),
        "calibration_value_reduction": {"checks": {"brier_ci_below_both_controls": False}},
    }
    for unit in stored["calibration_metrics"]["by_arm_seed"].values():
        unit.pop("action_confusion")
        unit.pop("calibration_bounds")
        unit.pop("source_cpu_scoring_duration_s")
    assert mod.compare_stored_metrics(stored, reduced)["all_matched"] is True
    stored["calibration_metrics"]["by_arm_seed"]["training_prevalence:1"]["brier"] = 9.0
    stored["calibration_metrics"]["by_arm"]["training_prevalence"]["brier"] = 9.0
    stored["paired_group_intervals"] = {}
    report = mod.compare_stored_metrics(stored, reduced)
    assert report["all_matched"] is False
    assert report["metric_mismatch_count"] == 3
    assert mod._nested_close("same", "same") is True


def test_req_report_7396_integrity_failure_branches(tmp_path: Path) -> None:
    """REQ-REPORT-7396 fails closed on malformed rows and unreadable JSON."""

    missing = tmp_path / "missing.json"
    invalid = tmp_path / "invalid.json"
    invalid.write_text("{", encoding="utf-8")
    assert mod.load_json(missing) is None
    assert mod.load_json(invalid) is None
    assert mod._wilson_bounds(0, 0) == [0.0, 1.0]
    with pytest.raises(ValueError, match="AUROC"):
        mod._binary_auroc([0], [0.2])

    good = _score_row("arm", 1, "g0", 0, 0.0, 0.2, "accept")
    duplicate = deepcopy(good)
    duplicate["label"] = 1
    duplicate["brier_contribution"] = 9.0
    duplicate["log_loss_contribution"] = 9.0
    malformed = {"label": "bad"}
    invalid_label = deepcopy(good)
    invalid_label["label"] = 2
    errors, identities, inconsistent = mod._row_integrity(
        {"rows": [good, duplicate, malformed, invalid_label]}
    )
    assert errors == 5
    assert identities == 1
    assert inconsistent == 1

    bad_brier = deepcopy(good)
    bad_brier["brier_contribution"] = 9.0
    bad_log = deepcopy(good)
    bad_log["group_id"] = "g1"
    bad_log["log_loss_contribution"] = 9.0
    positive = _score_row("arm", 1, "g2", 1, 1.0, 0.8, "reject")
    malformed_score = {"label": 0}
    malformed_label = {"label": "bad", "probability": 0.2}
    reduced = mod.recompute_static_metrics(
        [good, bad_brier, bad_log, positive, malformed_score, malformed_label], draws=0, seed=1
    )
    assert reduced["paired_group_intervals"] == {}
    assert reduced["row_integrity_errors"] == [
        "brier:1",
        "log_loss:2",
        "malformed:4",
        "malformed:5",
    ]
    invalid_probability = deepcopy(good)
    invalid_probability["probability"] = math.nan
    with pytest.raises(ValueError, match="probability"):
        mod._unit_metrics([invalid_probability])


def test_req_report_7396_incomplete_arm_is_not_bootstrapped() -> None:
    """REQ-REPORT-7396 excludes an arm that lacks a paired group."""

    rows = [
        _score_row("complete", 1, "g0", 0, -1.0, 0.2, "accept"),
        _score_row("complete", 1, "g1", 1, 1.0, 0.8, "reject"),
        _score_row("incomplete", 1, "g0", 0, -1.0, 0.2, "accept"),
    ]
    assert mod._paired_intervals(rows, 10, 1) == {}


def test_req_report_7396_restricted_history_and_sidecars(tmp_path: Path) -> None:
    """REQ-REPORT-7396 stores only hash-bound historical and failed-log facts."""

    for relative, value in (
        (
            mod.EXP7382_PATH,
            {"status": "protocol", "verdict_class": "null", "flagged_adversarial": False},
        ),
        (
            mod.EXP7385_PATH,
            {"status": "null", "verdict_class": "null", "flagged_adversarial": False},
        ),
        (
            mod.EXP7386_PATH,
            {
                "status": "failed",
                "verdict_class": "disqualified",
                "flagged_adversarial": True,
                "validation_receipts": [
                    {
                        "name": "full_python_suite",
                        "timed_out": True,
                        "exit_code": -15,
                    },
                    {
                        "name": "independent_reducer",
                        "output_tail": "affected_validation_mismatch",
                    },
                ],
            },
        ),
    ):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(__import__("json").dumps(value), encoding="utf-8")
    diagnosis = mod.exp7386_restricted_diagnosis(tmp_path)
    assert diagnosis["affected_validation_mismatch"] is True
    assert diagnosis["appended_broad_suite_timed_out"] is True
    assert diagnosis["online_metrics_consumed"] is False
    sidecars = mod.write_historical_sidecars(tmp_path, tmp_path / "raw")
    assert len(sidecars) == 2
    assert all((tmp_path / row["path"]).is_file() for row in sidecars)


def test_req_report_7396_validation_plan_rejects_expansion(tmp_path: Path) -> None:
    """REQ-REPORT-7396 rejects missing flags and appended broad commands."""

    commands = mod.build_validation_plan(ROOT, tmp_path)
    changed = list(commands)
    changed[1] = mod.validation_scope.CommandSpec(
        "focused_pytest", ("pytest", mod.TEST_PATH.as_posix()), "explicit_tests"
    )
    changed.append(
        mod.validation_scope.CommandSpec("full_python_suite", ("pytest", "tests/python"), "broad")
    )
    errors = mod.validate_validation_plan(ROOT, changed)
    assert "affected_check_names_mismatch" in errors
    assert "full_python_suite_forbidden" in errors
    assert "focused_pytest_flags_mismatch" in errors


def test_req_report_7396_validator_failure_branches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-REPORT-7396 cold validation identifies each material artifact defect."""

    artifact = mod.build_artifact_for_test(_passing_receipts())
    assert mod.validate_artifact(None) == ["artifact_not_object"]
    assert mod.validate_artifact({})[0].startswith("missing_required_field:")

    mutations = (
        ("schema", "wrong", "identity_invalid"),
        ("MODEL_SPECS", ["model"], "current_substrate_declaration_invalid"),
        ("promotion_score", 1, "promotion_score_nonzero"),
        ("field_principles", {}, "field_principles_incomplete"),
        ("rows", [{}], "audit_row_invalid"),
        ("acceptance_gate_results", [], "acceptance_gate_results_mismatch"),
        ("gate_check_summary", {}, "gate_check_summary_mismatch"),
        ("static_audit_complete_score", 0, "static_audit_complete_score_mismatch"),
        ("verdict_class", "positive", "terminal_state_mismatch"),
    )
    for field, value, expected in mutations:
        changed = deepcopy(artifact)
        changed[field] = value
        assert expected in mod.validate_artifact(changed)

    blocked = mod.build_blocked_artifact(
        [mod._precondition("missing", "source", "field", "yes", None)], {}
    )
    blocked["static_audit_complete_score"] = 1
    blocked["static_value_confirmed_score"] = 1
    blocked_errors = mod.validate_artifact(blocked)
    assert "blocked_audit_score_nonzero" in blocked_errors
    assert "blocked_value_score_nonzero" in blocked_errors

    changed = deepcopy(artifact)
    changed["source_artifact_hashes"] = {"missing.json": "sha256:bad"}
    changed["historical_receipt_sidecars"] = [
        {"path": "missing-sidecar.json", "sha256": "sha256:bad", "scope": "history"}
    ]
    hash_errors = mod.validate_artifact(changed, root=tmp_path)
    assert "source_hash_mismatch:missing.json" in hash_errors
    assert "sidecar_hash_mismatch:missing-sidecar.json" in hash_errors

    monkeypatch.setattr(mod, "load_json", lambda _path: None)
    assert "cold_training_source_missing" in mod.validate_artifact(
        artifact, root=tmp_path, deep=True
    )


def test_req_report_7396_deep_recompute_and_terminal_commands(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-REPORT-7396 fresh replay compares recomputed rows and freezes readers."""

    artifact = mod.build_artifact_for_test(_passing_receipts())
    monkeypatch.setattr(mod, "load_json", lambda _path: {"rows": []})
    monkeypatch.setattr(
        mod,
        "recompute_static_metrics",
        lambda _rows, draws, seed: deepcopy(artifact["static_metric_recomputation"]),
    )
    monkeypatch.setattr(
        mod,
        "compare_stored_metrics",
        lambda _training, _recomputed: deepcopy(artifact["metric_match_report"]),
    )
    assert mod.validate_artifact(artifact, deep=True) == []

    changed = deepcopy(artifact)
    changed["metric_match_report"] = {"all_matched": False}
    deep_errors = mod.validate_artifact(changed, deep=True)
    assert "cold_metric_recomputation_mismatch" in deep_errors
    assert "cold_rows_mismatch" not in deep_errors

    monkeypatch.setattr(
        mod,
        "recompute_static_metrics",
        lambda _rows, draws, seed: {"different": True},
    )
    monkeypatch.setattr(
        mod, "compare_stored_metrics", lambda _training, _recomputed: changed["metric_match_report"]
    )
    assert "cold_rows_mismatch" in mod.validate_artifact(changed, deep=True)

    commands = mod.terminal_command_specs(ROOT, Path("/tmp/candidate.json"))
    assert [command.spec.name for command in commands] == list(mod.TERMINAL_CHECK_NAMES)
