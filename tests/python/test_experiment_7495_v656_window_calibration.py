"""Tests for REQ-VERIFY-7495 and SCENARIO-VERIFY-7495-*.

The fixtures use synthetic logits and labels. They test the numeric contract
without reusing the fresh V656 test outcomes during authoring.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_7495_v656_window_calibration as exp


def _capture_group(
    group_id: str,
    role: str,
    whole: float,
    windows: tuple[float, ...],
) -> list[exp.JsonDict]:
    """Build both stable option orders for one whole response and its windows."""

    rows: list[exp.JsonDict] = []
    for arm, index, odds in (("whole_response", None, whole), *(
        ("focused_window", window_index, value)
        for window_index, value in enumerate(windows)
    )):
        for order in (
            ["supported", "contains_unsupported"],
            ["contains_unsupported", "supported"],
        ):
            rows.append(
                {
                    "group_id": group_id,
                    "source_group_id": group_id,
                    "role": role,
                    "arm": arm,
                    "window_index": index,
                    "option_order": order,
                    "raw_logits_by_option_id": {
                        "supported": 0.0,
                        "contains_unsupported": odds,
                    },
                    "eligible": True,
                    "disposition": "complete",
                    "response_text": "short response" if len(windows) == 2 else "long " * 200,
                    "source_version": {"corpus": "fixture-family"},
                }
            )
    return rows


def _feature_rows(count: int, role: str) -> list[exp.JsonDict]:
    """Make separable group features for fast deterministic fit tests."""

    rows = []
    for index in range(count):
        label = index % 2
        sign = 1.0 if label else -1.0
        rows.append(
            {
                "group_id": f"{role}-{index:03d}",
                "role": role,
                "label": label,
                "features": [sign * (1.0 + offset / 10.0) for offset in range(6)],
                "raw_whole_log_odds": sign,
                "raw_window_log_odds": sign * 1.2,
                "length_bin": "short" if index % 3 else "long",
                "source_family": "fixture-family",
            }
        )
    return rows


def _prediction_fixture(test_count: int = 110) -> list[exp.JsonDict]:
    """Build calibration and test predictions with group-perfect target rows."""

    rows: list[exp.JsonDict] = []
    for role, count in (("calibration_tuning", 50), ("test", test_count)):
        for index in range(count):
            label = index % 2
            common = {
                "group_id": f"{role}-{index:03d}",
                "role": role,
                "label": label,
                "length_bin": "short" if index % 2 else "long",
                "source_family": "fixture-family",
                "failed": False,
            }
            for seed in exp.TRAINING_SEEDS:
                rows.append(
                    {
                        **common,
                        "arm": "conditional_gibbs",
                        "seed": seed,
                        "probability": 0.05 if label == 0 else 0.95,
                    }
                )
                for arm in ("identical_feature_logistic", "whole_only_gibbs"):
                    rows.append(
                        {
                            **common,
                            "arm": arm,
                            "seed": seed,
                            "probability": 0.5,
                        }
                    )
            for arm in ("raw_whole", "raw_window", "temperature_whole"):
                rows.append({**common, "arm": arm, "seed": None, "probability": 0.5})
    return rows


def test_req_verify_7495_spec_and_scenarios_precede_code() -> None:
    """REQ-VERIFY-7495 fixes the measurement boundaries before implementation."""

    text = exp.SPEC_PATH.read_text(encoding="utf-8")
    assert "REQ-VERIFY-7495" in text
    for suffix in ("PREREQUISITE", "SEAL", "HEADS", "PROBABILITY", "UTILITY", "NULL", "E2E"):
        assert f"SCENARIO-VERIFY-7495-{suffix}" in text


def test_capture_reduction_is_order_invariant_and_label_free() -> None:
    """SCENARIO-VERIFY-7495-HEADS reduces whole and window logits without labels."""

    rows = _capture_group("g1", "training", -1.0, (-0.5, 2.0))
    reduced = exp.aggregate_capture_features(rows)

    assert len(reduced) == 1
    assert reduced[0]["group_id"] == "g1"
    assert reduced[0]["features"] == pytest.approx([-1.0, 2.0, 0.75, 2.5, 0.5, 2.0])
    assert reduced[0]["raw_window_log_odds"] == pytest.approx(2.0)
    assert "label" not in reduced[0]
    assert reduced[0]["source_family"] == "fixture-family"
    assert reduced[0]["length_bin"] == "short"


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda rows: rows.pop(), "option_order_pair_invalid"),
        (
            lambda rows: rows[0].__setitem__("raw_logits_by_option_id", None),
            "native_logits_missing",
        ),
        (
            lambda rows: rows[0]["raw_logits_by_option_id"].__setitem__(  # type: ignore[index, union-attr]
                "supported", float("nan")
            ),
            "native_logits_nonfinite",
        ),
    ],
)
def test_capture_feature_mutations_fail_closed(mutate: object, message: str) -> None:
    """SCENARIO-VERIFY-7495-HEADS rejects incomplete or nonfinite capture cells."""

    rows = _capture_group("g1", "training", -1.0, (-0.5, 2.0))
    mutate(rows)  # type: ignore[operator]
    with pytest.raises(ValueError, match=message):
        exp.aggregate_capture_features(rows)


def test_labels_join_by_role_and_convert_supported_gold() -> None:
    """SCENARIO-VERIFY-7495-SEAL opens only the explicitly allowed label role."""

    features = exp.aggregate_capture_features(
        [
            *_capture_group("g1", "training", -1.0, (-0.5, -0.25)),
            *_capture_group("g2", "test", 1.0, (0.5, 0.25)),
        ]
    )
    evaluators = [
        {"group_id": "g1", "role": "training", "label": 1},
        {"group_id": "g2", "role": "test", "label": 0},
    ]

    training = exp.attach_role_labels(features, evaluators, allowed_roles={"training"})
    assert [row["group_id"] for row in training] == ["g1"]
    assert training[0]["label"] == 0
    with pytest.raises(ValueError, match="label_role_not_allowed"):
        exp.attach_role_labels(features, evaluators, allowed_roles={"training", "test"}, expected_role="training")
    with pytest.raises(ValueError, match="evaluator_join_missing"):
        exp.attach_role_labels(features, [], allowed_roles={"training"})


def test_fit_bundle_is_deterministic_bounded_and_calibration_only() -> None:
    """SCENARIO-VERIFY-7495-SEAL/HEADS freezes five bounded exact Gibbs states."""

    training = _feature_rows(20, "training")
    calibration = _feature_rows(12, "calibration_tuning")
    first = exp.fit_numeric_bundle(
        training,
        calibration,
        steps=3,
        knot_counts=(2,),
        regularization=(0.0,),
    )
    second = exp.fit_numeric_bundle(
        training,
        calibration,
        steps=3,
        knot_counts=(2,),
        regularization=(0.0,),
    )

    assert first == second
    assert first["training_seeds"] == list(exp.TRAINING_SEEDS)
    assert first["roles_consumed"] == ["training", "calibration_tuning"]
    assert first["heldout_labels_consumed"] is False
    assert first["bundle_sha256"] == exp.numeric_bundle_hash(first)
    assert set(first["heads"]) >= {
        "conditional_gibbs",
        "whole_only_gibbs",
        "identical_feature_logistic",
        "shuffled_label_gibbs",
    }
    assert all(len(first["heads"][arm]) == 5 for arm in first["heads"])
    for state in first["checkpoint_manifest"]:
        assert state["parameter_count"] <= 256
        assert state["exact_normalization"] == "binary_logsumexp"
        assert "E(y|x)" in state["equation"]
    assert len({row["checkpoint_sha256"] for row in first["checkpoint_manifest"]}) == 20
    assert first["window_feature_transform"]["selection_role"] == "calibration_tuning"
    assert len(first["frozen_cost_policies"]) == 9


def test_fit_bundle_rejects_role_support_and_feature_drift() -> None:
    """SCENARIO-VERIFY-7495-SEAL prevents test leakage and malformed training rows."""

    training = _feature_rows(6, "training")
    calibration = _feature_rows(6, "calibration_tuning")
    wrong_role = deepcopy(training)
    wrong_role[0]["role"] = "test"
    with pytest.raises(ValueError, match="training_role_invalid"):
        exp.fit_numeric_bundle(wrong_role, calibration, steps=1, knot_counts=(2,), regularization=(0.0,))

    one_class = deepcopy(training)
    for row in one_class:
        row["label"] = 0
    with pytest.raises(ValueError, match="training_support_invalid"):
        exp.fit_numeric_bundle(one_class, calibration, steps=1, knot_counts=(2,), regularization=(0.0,))

    malformed = deepcopy(training)
    malformed[0]["features"] = [1.0]
    with pytest.raises(ValueError, match="feature_shape_invalid"):
        exp.fit_numeric_bundle(malformed, calibration, steps=1, knot_counts=(2,), regularization=(0.0,))


def test_frozen_bundle_scores_all_registered_arms_and_seeds() -> None:
    """SCENARIO-VERIFY-7495-HEADS emits per-group arms without multiplying support."""

    training = _feature_rows(10, "training")
    calibration = _feature_rows(8, "calibration_tuning")
    bundle = exp.fit_numeric_bundle(
        training,
        calibration,
        steps=2,
        knot_counts=(2,),
        regularization=(0.0,),
    )
    scored = exp.score_numeric_bundle(bundle, calibration[:2])

    assert {row["arm"] for row in scored} == set(exp.ALL_ARMS)
    assert len([row for row in scored if row["arm"] == "conditional_gibbs"]) == 10
    assert all(0.0 <= row["probability"] <= 1.0 for row in scored)
    changed = deepcopy(bundle)
    changed["bundle_sha256"] = "sha256:changed"
    with pytest.raises(ValueError, match="fit_bundle_hash_invalid"):
        exp.score_numeric_bundle(changed, calibration[:1])


def test_probability_and_decision_gates_use_groups_and_all_nine_cells() -> None:
    """SCENARIO-VERIFY-7495-PROBABILITY/UTILITY enforces both control families."""

    reduced = exp.reduce_prediction_rows(_prediction_fixture(), bootstrap_draws=200)

    assert reduced["confirmatory_support_score"] == 1
    assert reduced["test_group_count"] == 110
    assert reduced["probability_benefit_score"] == 1
    assert set(reduced["probability_comparisons"]["brier"]) == {
        "identical_feature_logistic",
        "whole_only_gibbs",
    }
    assert all(row["group_count"] == 110 for row in reduced["probability_comparisons"]["brier"].values())
    assert len(reduced["decision_report"]["cells"]) == 9
    assert reduced["decision_report"]["decision_benefit_score"] == 1
    assert all(row["benefit_passed"] for row in reduced["decision_report"]["cells"])
    assert set(reduced["slice_report"]) == {"length", "source_family"}


def test_low_support_keeps_descriptive_rows_but_closes_benefit() -> None:
    """SCENARIO-VERIFY-7495-PROBABILITY closes confirmation below support floors."""

    reduced = exp.reduce_prediction_rows(_prediction_fixture(test_count=30), bootstrap_draws=100)
    assert reduced["confirmatory_support_score"] == 0
    assert reduced["probability_benefit_score"] == 0
    assert reduced["decision_benefit_score"] == 0
    assert reduced["probability_report"]["test"]["conditional_gibbs"]["n_groups"] == 30


def test_precondition_rows_preserve_exact_ready_fields_and_flags(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-7495-PREREQUISITE rejects changed producer declarations."""

    fit = {
        "window_fit_ready_score": 1,
        "verdict_class": "null",
        "flagged_adversarial": False,
    }
    evaluation = {
        "window_evaluation_ready_score": 1,
        "verdict_class": "null",
        "flagged_adversarial": False,
    }
    fit_path = tmp_path / "fit.json"
    eval_path = tmp_path / "eval.json"
    fit_path.write_text(json.dumps(fit), encoding="utf-8")
    eval_path.write_text(json.dumps(evaluation), encoding="utf-8")
    checks = exp.upstream_preconditions(fit_path, eval_path)
    assert all(row["passed"] and "prevent" in row["principle"] for row in checks)

    evaluation["flagged_adversarial"] = True
    eval_path.write_text(json.dumps(evaluation), encoding="utf-8")
    failed = [row["check"] for row in exp.upstream_preconditions(fit_path, eval_path) if not row["passed"]]
    assert failed == ["evaluation_unflagged"]


def test_fixture_artifact_cold_validates_and_covers_every_field(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-7495-NULL/E2E validates a complete null independently."""

    artifact = exp.fixture_artifact(tmp_path)
    assert exp.validate_artifact(artifact, root=tmp_path, require_validation=False) == []
    assert artifact["MODEL_SPECS"] == artifact["model_specs"] == []
    assert artifact["model_invoked"] is False
    assert artifact["inference_substrate_class"] == "no_model_load"
    assert artifact["execution_venue"] == "host"
    assert artifact["window_calibration_complete_score"] == 1
    assert artifact["honest_verdict"].startswith("complete_null")
    assert set(artifact["field_principles"]) == set(artifact)
    assert all("prevent" in principle for principle in artifact["field_principles"].values())
    assert all("principle" in gate and "prevent" in gate["principle"] for gate in artifact["acceptance_gate_results"])


@pytest.mark.parametrize(
    ("mutate", "error"),
    [
        (lambda value: value.__setitem__("schema", "wrong"), "identity_invalid"),
        (lambda value: value.__setitem__("MODEL_SPECS", ["model"]), "model_specs_not_empty"),
        (lambda value: value.__setitem__("model_invoked", True), "model_accounting_invalid"),
        (lambda value: value.__setitem__("execution_venue", "board"), "substrate_invalid"),
        (lambda value: value.__setitem__("frozen_policy_manifest", {}), "frozen_policy_hash_mismatch"),
        (lambda value: value.__setitem__("field_principles", {}), "field_principles_incomplete"),
    ],
)
def test_artifact_mutations_fail_closed(tmp_path: Path, mutate: object, error: str) -> None:
    """SCENARIO-VERIFY-7495-E2E names altered terminal evidence."""

    artifact = exp.fixture_artifact(tmp_path)
    mutate(artifact)  # type: ignore[operator]
    assert error in exp.validate_artifact(artifact, root=tmp_path, require_validation=False)


def test_validation_manifest_is_only_the_three_exp7495_paths() -> None:
    """SCENARIO-VERIFY-7495-E2E forbids broad or stale validation scope."""

    manifest = exp.VALIDATION_MANIFEST
    assert manifest.test_paths == (exp.TEST_PATH.as_posix(),)
    assert manifest.changed_modules == (exp.MODULE_PATH.as_posix(),)
    assert manifest.static_paths == (exp.WRAPPER_PATH.as_posix(),)
    assert "tests/python" not in manifest.test_paths
