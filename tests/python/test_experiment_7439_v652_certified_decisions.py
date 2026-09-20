"""Tests for REQ-AUTO-7439 and SCENARIO-AUTO-7439-*.

The tests keep policy coverage separate from probability calibration and make
empty typed actions explicit instead of converting them to zero risk.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest

from carnot import experiment_7439_v652_certified_decisions as exp


def _feature_rows(count: int, *, role: str, offset: int = 0) -> list[dict[str, object]]:
    """Build small independent groups with both human support labels."""

    rows = []
    for index in range(count):
        value = (index + 1) / (count + 1)
        rows.append(
            {
                "row_key": f"{role}-{index + offset:03d}",
                "group_id": f"group-{role}-{index + offset:03d}",
                "partition": role,
                "task_type": "QA",
                "features": {
                    name: min(1.0, value + feature_index * 0.01)
                    for feature_index, name in enumerate(exp.SOURCE_FEATURE_NAMES)
                },
                "label": index % 2,
            }
        )
    return rows


def _passing_receipts() -> list[dict[str, object]]:
    """Create the exact receipt names required by a complete fixture."""

    return [
        {"name": name, "required": True, "passed": True, "exit_code": 0}
        for name in (*exp.AFFECTED_CHECK_NAMES, *exp.TERMINAL_CHECK_NAMES)
    ]


def test_req_auto_7439_spec_precedes_implementation() -> None:
    """REQ-AUTO-7439 has a complete contract before implementation."""

    text = exp.SPEC_PATH.read_text(encoding="utf-8")
    assert "### REQ-AUTO-7439:" in text
    for number in range(1, 7):
        assert f"SCENARIO-AUTO-7439-{number:02d}" in text


def test_scenario_auto_7439_01_predictor_projection_uses_only_six_features() -> None:
    """SCENARIO-AUTO-7439-01 excludes identity and evaluator fields from X."""

    rows = _feature_rows(4, role="fit")
    rows[0]["annotation_spans"] = [{"start": 0, "end": 1}]
    rows[0]["model_id"] = "forbidden-model-identity"
    matrix, receipt = exp.predictor_matrix(rows)
    assert matrix.shape == (4, 6)
    assert receipt["input_fields"] == list(exp.SOURCE_FEATURE_NAMES)
    assert set(receipt["excluded_fields"]) >= {
        "label",
        "annotation_spans",
        "model_id",
        "group_id",
    }
    assert receipt["labels_consumed"] is False

    broken = deepcopy(rows)
    del broken[0]["features"][exp.SOURCE_FEATURE_NAMES[0]]  # type: ignore[index]
    with pytest.raises(ValueError, match="exact six"):
        exp.predictor_matrix(broken)
    unbounded = deepcopy(rows)
    unbounded[0]["features"][exp.SOURCE_FEATURE_NAMES[0]] = 2.0  # type: ignore[index]
    with pytest.raises(ValueError, match="finite and bounded"):
        exp.predictor_matrix(unbounded)
    with pytest.raises(ValueError, match="non-empty"):
        exp.predictor_matrix([])


def test_scenario_auto_7439_02_fits_five_seeds_and_averages_before_calibration() -> None:
    """SCENARIO-AUTO-7439-02 uses frozen seeds and a mean-before-affine fit."""

    rows = _feature_rows(12, role="fit")
    fitted = exp.fit_compact_heads(rows, steps=1)
    assert fitted["training_seeds"] == list(exp.TRAINING_SEEDS)
    assert set(fitted["heads"]) == set(exp.ALL_FITTED_HEADS)
    assert all(len(fitted["heads"][head]) == 5 for head in exp.ALL_FITTED_HEADS)
    assert fitted["diagnostics"]["prevalence"]["training_only"] is True
    assert all(row["passed"] for row in fitted["diagnostics"]["dense_spline_parity"])

    seed_probabilities = np.asarray([[0.2, 0.8]] * 5, dtype=np.float64)
    calibration = exp.calibrate_seed_probabilities(seed_probabilities, [0, 1], steps=1)
    assert calibration["mean_raw_probabilities"] == pytest.approx([0.2, 0.8])
    assert calibration["fit_order"] == "five_seed_probability_mean_then_affine_logit"
    with pytest.raises(ValueError, match="five seed"):
        exp.calibrate_seed_probabilities(seed_probabilities[:4], [0, 1], steps=1)
    invalid_probabilities = seed_probabilities.copy()
    invalid_probabilities[0, 0] = 2.0
    with pytest.raises(ValueError, match="finite and bounded"):
        exp.calibrate_seed_probabilities(invalid_probabilities, [0, 1], steps=1)
    with pytest.raises(ValueError, match="both labels"):
        exp.calibrate_seed_probabilities(seed_probabilities, [1, 1], steps=1)
    one_class = deepcopy(rows)
    for row in one_class:
        row["label"] = 1
    with pytest.raises(ValueError, match="both binary labels"):
        exp.fit_compact_heads(one_class, steps=1)


def test_scenario_auto_7439_02_policy_freezes_before_certification() -> None:
    """SCENARIO-AUTO-7439-02 keeps the selected pair fixed during checks."""

    tuning = [
        {**row, "probability": (index + 1) / 25}
        for index, row in enumerate(_feature_rows(24, role="policy_tuning"))
    ]
    policy, candidates = exp.freeze_policy(tuning)
    assert len(candidates) > 1
    assert policy["frozen"] is True
    assert policy["selection_partition"] == "policy_tuning"

    certification = [
        {**row, "probability": (index + 1) / 121}
        for index, row in enumerate(_feature_rows(120, role="certification"))
    ]
    certificate = exp.certify_frozen_policy(certification, policy)
    assert certificate["frozen_candidate_index"] == policy["candidate_index"]
    assert certificate["thresholds_reselected"] is False
    assert len(certificate["checks"]) == 3
    if not certificate["valid"]:
        assert certificate["deployed_policy"] == "all_escalate"


def test_scenario_auto_7439_03_empty_actions_are_null_and_spend_no_alpha() -> None:
    """SCENARIO-AUTO-7439-03 reports disabled and empty actions honestly."""

    disabled = exp.action_risk_check([], action="accept", enabled=False)
    assert disabled["applicable"] is False
    assert disabled["risk"] is None
    assert disabled["upper_bound"] is None
    assert disabled["alpha_spent"] == 0.0

    empty = exp.action_risk_check([], action="accept", enabled=True)
    assert empty["applicable"] is True
    assert empty["risk"] is None
    assert empty["diagnosis"] == "empty_selection"
    assert empty["passed"] is False

    with pytest.raises(ValueError, match="action"):
        exp.action_risk_check([], action="escalate", enabled=True)
    with pytest.raises(ValueError, match="binary"):
        exp.action_risk_check([2], action="reject", enabled=True)


def test_scenario_auto_7439_04_metrics_separate_actions_from_probabilities() -> None:
    """SCENARIO-AUTO-7439-04 separates policy coverage from proper scores."""

    rows = [
        {"label": 1, "probability": 0.8, "action": "accept", "group_id": "a"},
        {"label": 0, "probability": 0.2, "action": "reject", "group_id": "b"},
        {"label": 1, "probability": 0.6, "action": "escalate", "group_id": "c"},
    ]
    metrics = exp.reduce_policy_metrics(rows)
    assert metrics["coverage"] == pytest.approx(2 / 3)
    assert metrics["accept_harm"] == 0.0
    assert metrics["accept_denominator"] == 1
    assert metrics["reject_harm"] == 0.0
    assert metrics["reject_denominator"] == 1
    assert metrics["brier"] == pytest.approx((0.04 + 0.04 + 0.16) / 3)

    empty_actions = exp.reduce_policy_metrics(
        [{"label": 1, "probability": 0.7, "action": "escalate", "group_id": "x"}]
    )
    assert empty_actions["accept_harm"] is None
    assert empty_actions["reject_harm"] is None
    assert empty_actions["coverage"] == 0.0
    with pytest.raises(ValueError, match="non-empty"):
        exp.reduce_policy_metrics([])
    with pytest.raises(ValueError, match="binary"):
        exp.reduce_policy_metrics(
            [{"label": 2, "probability": 0.5, "action": "escalate", "group_id": "x"}]
        )
    with pytest.raises(ValueError, match="typed"):
        exp.reduce_policy_metrics(
            [{"label": 1, "probability": 0.5, "action": "guess", "group_id": "x"}]
        )


def test_scenario_auto_7439_04_paired_bootstrap_has_two_simultaneous_contrasts() -> None:
    """SCENARIO-AUTO-7439-04 resamples paired source groups, not pooled rows."""

    rows = [
        {
            "group_id": f"g-{index}",
            "tuned_spline": int(index % 2 == 0),
            "old_spline": 0,
            "tuned_logistic": int(index % 4 == 0),
        }
        for index in range(20)
    ]
    intervals = exp.paired_coverage_intervals(rows, draws=200, seed=65299)
    assert set(intervals) == {"tuned_spline_minus_old", "tuned_spline_minus_logistic"}
    assert all(row["draws"] == 200 for row in intervals.values())
    assert all(row["paired_source_groups"] == 20 for row in intervals.values())
    assert intervals["tuned_spline_minus_old"]["observed"] == pytest.approx(0.5)
    with pytest.raises(ValueError, match="unique"):
        exp.paired_coverage_intervals([rows[0], rows[0]], draws=10, seed=1)
    with pytest.raises(ValueError, match="positive"):
        exp.paired_coverage_intervals(rows, draws=0, seed=1)


def test_scenario_auto_7439_05_all_benefit_gates_are_conjunctive() -> None:
    """SCENARIO-AUTO-7439-05 preserves complete null measurements."""

    spline = {
        "certificate_valid": True,
        "certificate_coverage_lower_bound": 0.30,
        "utility": 0.4,
        "brier": 0.10,
        "log_loss": 0.30,
    }
    logistic = {"utility": 0.3, "brier": 0.1005, "log_loss": 0.3005}
    gibbs = {"utility": 0.2, "brier": 0.1008, "log_loss": 0.3008}
    intervals = {
        "tuned_spline_minus_old": {"lower": 0.02},
        "tuned_spline_minus_logistic": {"lower": 0.01},
    }
    passed = exp.reduce_decision_value(spline, logistic, gibbs, intervals)
    assert passed["decision_capture_complete_score"] == 1
    assert passed["decision_value_score"] == 1
    assert all(row["passed"] for row in passed["gates"])

    failed_intervals = deepcopy(intervals)
    failed_intervals["tuned_spline_minus_logistic"]["lower"] = 0.0
    limited = exp.reduce_decision_value(spline, logistic, gibbs, failed_intervals)
    assert limited["decision_capture_complete_score"] == 1
    assert limited["decision_value_score"] == 0
    assert limited["energy_advantage"] is None
    assert limited["limited_policy_wrapper_finding"] is True


def test_scenario_auto_7439_06_fixture_replays_and_rejects_mutation(tmp_path: Path) -> None:
    """SCENARIO-AUTO-7439-06 binds rows while denying deployment authority."""

    artifact = exp.build_fixture_artifact(tmp_path, validation_receipts=_passing_receipts())
    assert artifact["MODEL_SPECS"] == []
    assert artifact["model_invoked"] is False
    assert artifact["invocation_counts"] == exp.ZERO_INVOCATION_COUNTS
    assert artifact["inference_substrate_class"] == "no_model_load"
    assert artifact["decision_capture_complete_score"] == 1
    assert artifact["deployment_certificate_valid"] is False
    assert artifact["certificate_scope"] == "exploratory_reused_corpus"
    assert artifact["promotion_score"] == 0
    assert exp.validate_artifact(artifact, root=tmp_path) == []
    assert exp.independent_reduce(artifact, root=tmp_path)["decision_capture_complete_score"] == 1

    changed = deepcopy(artifact)
    shard = tmp_path / changed["probability_row_shards"][0]["path"]
    shard.write_text(json.dumps({"changed": True}) + "\n", encoding="utf-8")
    assert "probability_row_shard_invalid" in exp.validate_artifact(changed, root=tmp_path)


def test_scenario_auto_7439_03_deployed_action_requires_valid_certificate() -> None:
    """SCENARIO-AUTO-7439-03 exercises every deployed action boundary."""

    invalid = {"deployed_policy": "all_escalate"}
    assert exp._policy_action(0.99, invalid) == "escalate"
    valid = {
        "deployed_policy": "frozen_typed_policy",
        "deployed_accept_enabled": True,
        "deployed_reject_enabled": True,
        "accept_threshold": 0.8,
        "reject_threshold": 0.2,
    }
    assert exp._policy_action(0.9, valid) == "accept"
    assert exp._policy_action(0.1, valid) == "reject"
    assert exp._policy_action(0.5, valid) == "escalate"


def test_req_auto_7439_validator_names_all_declaration_mutations(tmp_path: Path) -> None:
    """REQ-AUTO-7439 rejects field, reduction, receipt, and checksum drift."""

    artifact = exp.build_fixture_artifact(tmp_path, validation_receipts=_passing_receipts())
    changed = deepcopy(artifact)
    changed.pop("schema")
    changed.update(
        {
            "experiment_id": "wrong",
            "run_date": "wrong",
            "MODEL_SPECS": [{"name": "forbidden"}],
            "invocation_counts": {},
            "inference_substrate_class": "wrong",
            "execution_venue": "wrong",
            "deployment_certificate_valid": True,
            "certificate_scope": "wrong",
            "promotion_score": 1,
            "fixture_artifact": False,
            "independent_reduction": {},
            "reproducibility_checksum": "wrong",
        }
    )
    errors = exp.validate_artifact(changed, root=tmp_path)
    assert set(errors) >= {
        "required_field_missing:schema",
        "artifact_identity_invalid",
        "artifact_schedule_invalid",
        "current_model_declaration_invalid",
        "current_invocation_counts_invalid",
        "inference_substrate_class_invalid",
        "execution_venue_invalid",
        "deployment_certificate_must_be_false",
        "certificate_scope_invalid",
        "promotion_score_invalid",
        "independent_reduction_mismatch",
        "reproducibility_checksum_mismatch",
    }
    gate = exp._gate("check", "validity", "==", True, True, True, "plain principle")
    assert gate["passed"] is True


def test_req_auto_7439_cli_modes_are_strict(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """REQ-AUTO-7439 covers parsing, replay, reduction, and declared-run dispatch."""

    parsed = exp.parse_args(["--date", exp.RUN_DATE, "--root", str(tmp_path)])
    assert parsed.root == tmp_path
    with pytest.raises(SystemExit, match="--date"):
        exp.main(["--date", "wrong"])

    candidate = tmp_path / "candidate.json"
    candidate.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(exp, "_load_object", lambda _path: {"independent_reduction": {"ok": 1}})
    monkeypatch.setattr(exp, "validate_artifact", lambda _value, root: [])
    assert (
        exp.main(["--date", exp.RUN_DATE, "--root", str(tmp_path), "--cold-replay", str(candidate)])
        == 0
    )
    monkeypatch.setattr(exp, "independent_reduce", lambda _value, root: {"ok": 1})
    assert (
        exp.main(
            [
                "--date",
                exp.RUN_DATE,
                "--root",
                str(tmp_path),
                "--independent-reduce",
                str(candidate),
            ]
        )
        == 0
    )

    called: list[tuple[Path, str, Path]] = []
    monkeypatch.setattr(
        exp,
        "run_experiment",
        lambda root, date, output_path: called.append((root, date, output_path)),
    )
    assert exp.main(["--date", exp.RUN_DATE, "--root", str(tmp_path)]) == 0
    assert called == [(tmp_path.resolve(), exp.RUN_DATE, exp.RESULT_PATH)]
    assert "errors" in capsys.readouterr().out


def test_req_auto_7439_preconditions_preserve_missing_none_and_zero(tmp_path: Path) -> None:
    """REQ-AUTO-7439 records exact upstream observations without fabrication."""

    upstream = {
        "selection_protocol_ready_score": 0,
        "verdict_class": None,
        "flagged_adversarial": False,
    }
    checks = exp.upstream_field_checks(upstream, path=Path("missing.json"))
    by_field = {row["field"]: row for row in checks}
    assert by_field["selection_protocol_ready_score"]["observed"] == 0
    assert by_field["verdict_class"]["observed"] is None
    assert by_field["flagged_adversarial"]["passed"] is True
    assert any(row["passed"] is False for row in checks)
