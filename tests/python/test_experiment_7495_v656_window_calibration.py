"""Tests for REQ-VERIFY-7495 and SCENARIO-VERIFY-7495-*.

The fixtures use small numeric rows. They test the production reducers without
loading a language model or opening the real test labels.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_7495_v656_window_calibration as exp


def _native_rows(group: str, role: str, whole: float, windows: list[float]) -> list[exp.JsonDict]:
    """Build both stable option orders for a whole response and its windows."""

    output: list[exp.JsonDict] = []
    for arm, window_index, value in [("whole_response", None, whole), *[
        ("focused_window", index, score) for index, score in enumerate(windows)
    ]]:
        for order in (
            ["supported", "contains_unsupported"],
            ["contains_unsupported", "supported"],
        ):
            output.append(
                {
                    "group_id": group,
                    "source_group_id": group,
                    "role": role,
                    "arm": arm,
                    "window_index": window_index,
                    "option_order": order,
                    "raw_logits_by_option_id": {
                        "supported": 0.0,
                        "contains_unsupported": value,
                    },
                    "eligible": True,
                    "disposition": "complete",
                }
            )
    return output


def _feature_rows(count: int, role: str) -> list[exp.JsonDict]:
    """Return separable ten-feature rows for bounded optimizer tests."""

    rows: list[exp.JsonDict] = []
    for index in range(count):
        label = index % 2
        sign = 1.0 if label else -1.0
        rows.append(
            {
                "group_id": f"{role}-{index:03d}",
                "role": role,
                "label": label,
                "features": [
                    2.0 * sign,
                    0.9 if label else 0.1,
                    0.8 if label else 0.2,
                    0.7 if label else 0.1,
                    float(index % 3) / 2.0,
                    0.6 if label else 0.2,
                    0.2 if label else 0.8,
                    0.3 if label else 0.9,
                    0.4 if label else 0.8,
                    0.0,
                ],
                "raw_whole_probability": 0.88 if label else 0.12,
                "raw_window_probability": 0.9 if label else 0.1,
                "response_length_bytes": 100 + index * 20,
                "source_family": "family-a" if index % 3 else "family-b",
            }
        )
    return rows


def test_req_verify_7495_spec_and_scenarios_precede_code() -> None:
    """REQ-VERIFY-7495 fixes the calibration boundary before implementation."""

    text = exp.SPEC_PATH.read_text(encoding="utf-8")
    assert "REQ-VERIFY-7495" in text
    for suffix in (
        "PREREQUISITE",
        "SEAL",
        "HEADS",
        "PROBABILITY",
        "UTILITY",
        "NULL",
        "E2E",
    ):
        assert f"SCENARIO-VERIFY-7495-{suffix}" in text


def test_scenario_prerequisite_authenticates_exact_capture_fields() -> None:
    """SCENARIO-VERIFY-7495-PREREQUISITE rejects changed producer flags."""

    fit = json.loads(exp.FIT_ARTIFACT.read_text(encoding="utf-8"))
    evaluation = json.loads(exp.EVAL_ARTIFACT.read_text(encoding="utf-8"))
    reduced = exp.reduce_upstream_gates(fit, evaluation, fit_errors=[], eval_errors=[])
    assert reduced["passed"] is True
    assert all(row["passed"] and "prevent" in row["principle"] for row in reduced["checks"])

    changed = deepcopy(evaluation)
    changed["flagged_adversarial"] = True
    assert exp.reduce_upstream_gates(fit, changed, fit_errors=[], eval_errors=[])["passed"] is False


def test_scenario_heads_builds_lossless_window_features_without_labels() -> None:
    """SCENARIO-VERIFY-7495-HEADS gives energy and logistic the same evidence."""

    native = _native_rows("g-1", "training", -1.0, [-2.0, 0.5, 1.5])
    predictors = [
        {
            "group_id": "g-1",
            "role": "training",
            "source_text": "The source gives 10 units.",
            "response_text": "The response gives 12 units.",
            "corpus": "fixture",
        }
    ]
    rows = exp.build_feature_rows(native, predictors)

    assert len(rows) == 1
    assert rows[0]["label"] is None
    assert len(rows[0]["features"]) == len(exp.FEATURE_NAMES) == 10
    assert rows[0]["features"][0] == pytest.approx(-1.0)
    assert rows[0]["raw_window_probability"] > rows[0]["raw_whole_probability"]
    assert rows[0]["window_count"] == 3


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda rows: rows.pop(), "option_order_pair_invalid"),
        (
            lambda rows: rows[0].__setitem__("raw_logits_by_option_id", None),
            "native_logits_missing",
        ),
        (
            lambda rows: rows[0].__setitem__(
                "raw_logits_by_option_id",
                {"supported": 0.0, "contains_unsupported": float("inf")},
            ),
            "native_logits_nonfinite",
        ),
    ],
)
def test_window_feature_mutations_fail_closed(mutation: object, message: str) -> None:
    """SCENARIO-VERIFY-7495-HEADS rejects incomplete or nonfinite evidence."""

    rows = _native_rows("g-1", "training", -1.0, [])
    mutation(rows)  # type: ignore[operator]
    predictors = [{"group_id": "g-1", "source_text": "a", "response_text": "b"}]
    with pytest.raises(ValueError, match=message):
        exp.build_feature_rows(rows, predictors)


def test_scenario_seal_fits_five_compact_heads_on_train_and_calibration_only() -> None:
    """SCENARIO-VERIFY-7495-SEAL freezes every state before test labels open."""

    training = _feature_rows(16, "training")
    calibration = _feature_rows(12, "calibration_tuning")
    first = exp.fit_numeric_bundle(training, calibration, steps=4)
    second = exp.fit_numeric_bundle(training, calibration, steps=4)

    assert first == second
    assert first["training_seeds"] == list(exp.TRAINING_SEEDS)
    assert first["roles_consumed"] == ["training", "calibration_tuning"]
    assert first["test_labels_consumed"] is False
    assert first["bundle_sha256"] == exp.numeric_bundle_hash(first)
    assert set(first["heads"]) == {
        "conditional_gibbs",
        "whole_only_gibbs",
        "identical_feature_logistic",
        "shuffled_label_conditional_gibbs",
    }
    assert all(len(states) == 5 for states in first["heads"].values())
    assert max(state["spline_coefficient_count"] for state in first["heads"]["conditional_gibbs"]) <= 256
    assert all(state["normalization"] == "exact_binary_partition" for state in first["heads"]["conditional_gibbs"])
    assert len(first["policies"]) == 9
    assert first["frozen_before_test_labels"] is True


def test_fit_rejects_role_leakage_shape_errors_and_one_class_support() -> None:
    """SCENARIO-VERIFY-7495-SEAL prevents test outcomes from selecting states."""

    training = _feature_rows(8, "training")
    calibration = _feature_rows(8, "calibration_tuning")
    bad_role = deepcopy(training)
    bad_role[0]["role"] = "test"
    with pytest.raises(ValueError, match="training_role_invalid"):
        exp.fit_numeric_bundle(bad_role, calibration, steps=1)
    bad_shape = deepcopy(training)
    bad_shape[0]["features"] = [0.0]
    with pytest.raises(ValueError, match="feature_shape_invalid"):
        exp.fit_numeric_bundle(bad_shape, calibration, steps=1)
    one_class = deepcopy(training)
    for row in one_class:
        row["label"] = 0
    with pytest.raises(ValueError, match="class_support_invalid"):
        exp.fit_numeric_bundle(one_class, calibration, steps=1)


def test_frozen_bundle_scores_all_registered_test_arms_without_ablation() -> None:
    """SCENARIO-VERIFY-7495-HEADS retains controls and excludes shuffle from test."""

    bundle = exp.fit_numeric_bundle(
        _feature_rows(12, "training"),
        _feature_rows(10, "calibration_tuning"),
        steps=2,
    )
    rows = _feature_rows(4, "test")
    scored = exp.score_numeric_bundle(bundle, rows)

    assert {row["arm"] for row in scored} == set(exp.ALL_ARMS)
    assert not any("shuffled" in str(row["arm"]) for row in scored)
    assert len([row for row in scored if row["arm"] == "conditional_gibbs"]) == 20
    assert all(0.0 <= row["probability"] <= 1.0 for row in scored)


def _prediction_fixture(groups: int, *, role: str) -> list[exp.JsonDict]:
    output: list[exp.JsonDict] = []
    for index in range(groups):
        label = index % 2
        for arm in exp.ALL_ARMS:
            probability = {
                "conditional_gibbs": 0.98 if label else 0.02,
                "identical_feature_logistic": 0.70 if label else 0.30,
                "whole_only_gibbs": 0.68 if label else 0.32,
                "raw_whole": 0.60 if label else 0.40,
                "raw_window": 0.62 if label else 0.38,
                "temperature_whole": 0.65 if label else 0.35,
            }[arm]
            seeds = exp.TRAINING_SEEDS if arm in exp.LEARNED_ARMS else (None,)
            for seed in seeds:
                output.append(
                    {
                        "group_id": f"{role}-{index:03d}",
                        "role": role,
                        "arm": arm,
                        "seed": seed,
                        "label": label,
                        "probability": probability,
                        "response_length_slice": "short" if index % 2 else "long",
                        "source_family": "family-a" if index % 3 else "family-b",
                        "failed": False,
                    }
                )
    return output


def test_scenario_probability_averages_seeds_and_requires_both_controls() -> None:
    """SCENARIO-VERIFY-7495-PROBABILITY uses groups and both Holm contrasts."""

    rows = _prediction_fixture(100, role="test")
    reduction = exp.reduce_prediction_rows(rows, bootstrap_draws=200)

    assert reduction["confirmatory_support_score"] == 1
    assert reduction["probability_benefit_score"] == 1
    assert reduction["test_group_count"] == 100
    assert set(reduction["probability_comparisons"]["brier"]) == {
        "identical_feature_logistic",
        "whole_only_gibbs",
    }
    assert all(row["group_count"] == 100 for row in reduction["probability_comparisons"]["brier"].values())
    assert reduction["probability_report"]["conditional_gibbs"]["n_groups"] == 100
    assert reduction["length_slices"] and reduction["source_family_slices"]


def test_support_and_all_nine_decision_cells_fail_closed() -> None:
    """SCENARIO-VERIFY-7495-UTILITY and NULL keep support and completion separate."""

    rows = _prediction_fixture(40, role="test")
    reduction = exp.reduce_prediction_rows(rows, bootstrap_draws=100)
    assert reduction["confirmatory_support_score"] == 0
    assert reduction["probability_benefit_score"] == 0
    assert reduction["decision_benefit_score"] == 0
    assert len(reduction["decision_report"]["cells"]) == 9

    changed = deepcopy(reduction["decision_report"])
    for cell in changed["cells"]:
        cell["benefit_passed"] = True
    changed["cells"][0]["benefit_passed"] = False
    assert exp.decision_score(changed, confirmatory_support=True) == 0


def test_fixture_artifact_cold_validates_and_reduces_raw_rows(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-7495-NULL/E2E derives a valid complete null from rows."""

    artifact = exp.build_artifact_for_test(tmp_path)
    assert exp.validate_artifact(artifact, root=tmp_path, require_validation=False) == []
    assert artifact["window_calibration_complete_score"] == 1
    assert artifact["confirmatory_support_score"] == 0
    assert artifact["probability_benefit_score"] == 0
    assert artifact["decision_benefit_score"] == 0
    assert artifact["verdict_class"] == "null"
    assert artifact["honest_verdict"].startswith("complete_null")
    assert artifact["MODEL_SPECS"] == artifact["model_specs"] == []
    assert artifact["model_invoked"] is False
    assert set(artifact["field_principles"]) == set(artifact)
    assert all("prevent" in text for text in artifact["field_principles"].values())
    assert exp.independent_reduce(artifact, root=tmp_path) == artifact["independent_reduction"]


def test_validator_rejects_checksum_rows_scores_principles_and_receipts(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-7495-E2E fails closed when terminal evidence drifts."""

    for expected, mutate in (
        ("identity_mismatch:schema", lambda value: value.__setitem__("schema", "wrong")),
        ("field_principles_incomplete", lambda value: value.__setitem__("field_principles", {})),
        ("independent_reduction_mismatch", lambda value: value.__setitem__("independent_reduction", {})),
        ("score_mismatch:window_calibration_complete_score", lambda value: value.__setitem__("window_calibration_complete_score", 0)),
    ):
        artifact = exp.build_artifact_for_test(tmp_path)
        mutate(artifact)
        assert expected in exp.validate_artifact(artifact, root=tmp_path, require_validation=False)

    artifact = exp.build_artifact_for_test(tmp_path)
    artifact["validation_receipts"] = []
    artifact["reproducibility_checksum"] = exp.artifact_checksum(artifact)
    assert "required_validation_failed" in exp.validate_artifact(
        artifact,
        root=tmp_path,
        require_validation=True,
    )


def test_validation_manifest_is_frozen_to_three_exp7495_paths() -> None:
    """SCENARIO-VERIFY-7495-E2E forbids a broad Python-suite target."""

    manifest = exp.VALIDATION_MANIFEST
    assert manifest.test_paths == (exp.TEST_PATH.as_posix(),)
    assert manifest.changed_modules == (exp.MODULE_PATH.as_posix(),)
    assert manifest.static_paths == (exp.WRAPPER_PATH.as_posix(),)
    assert all(path not in {"tests", "tests/python"} for path in manifest.test_paths)


def test_helpers_reject_invalid_probability_and_policy_inputs() -> None:
    """REQ-VERIFY-7495 keeps malformed numeric rows outside the evidence."""

    with pytest.raises(ValueError, match="probability_metric_input_invalid"):
        exp.probability_metrics([], [])
    with pytest.raises(ValueError, match="policy_input_invalid"):
        exp.select_cost_policy([], [], false_accept_cost=1.0, escalation_cost=0.1)
    with pytest.raises(ValueError, match="bootstrap_input_invalid"):
        exp.paired_bootstrap([], draws=0, seed=1)
