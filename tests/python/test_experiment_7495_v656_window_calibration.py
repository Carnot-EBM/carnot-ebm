"""Tests for REQ-VERIFY-7495 and SCENARIO-VERIFY-7495-*.

The tests use small numeric fixtures. They test the scientific contract without
loading a model or reading the sealed production test labels.
"""

from __future__ import annotations

from copy import deepcopy
import json

import numpy as np
import pytest

from carnot import experiment_7495_v656_window_calibration as exp


def _readout_rows(role: str = "training") -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for arm, index, first, second in (
        ("whole_response", None, -1.0, -0.8),
        ("focused_window", 0, -2.0, -1.8),
        ("focused_window", 1, 1.0, 1.2),
    ):
        for order, value in (
            (("supported", "contains_unsupported"), first),
            (("contains_unsupported", "supported"), second),
        ):
            rows.append(
                {
                    "eligible": True,
                    "disposition": "complete",
                    "group_id": "g1",
                    "source_group_id": "g1",
                    "role": role,
                    "arm": arm,
                    "window_index": index,
                    "option_order": list(order),
                    "raw_logits_by_option_id": {
                        "supported": 0.0,
                        "contains_unsupported": value,
                    },
                    "source_version": {"corpus": "fixture"},
                }
            )
    return rows


def _numeric_rows(role: str, count: int, offset: int = 0) -> list[dict[str, object]]:
    rows = []
    for index in range(count):
        label = (index + offset) % 2
        signal = 2.0 if label else -2.0
        rows.append(
            {
                "group_id": f"{role}-{index:03d}",
                "role": role,
                "label": label,
                "feature_vector": [
                    signal * 0.6,
                    signal,
                    signal * 0.8,
                    signal * 1.1,
                    1.0,
                    1.0,
                    1.0,
                ],
                "whole_log_odds": signal * 0.6,
                "window_log_odds": [signal, signal * 0.8, signal * 1.1],
                "source_family": "fixture",
                "response_length": 100 + index,
            }
        )
    return rows


def test_aggregate_readouts_preserves_each_window_and_rejects_bad_pairs() -> None:
    """SCENARIO-VERIFY-7495-HEADS: option order and windows stay explicit."""

    groups, controls = exp.aggregate_readouts(_readout_rows())
    assert controls == []
    assert len(groups) == 1
    assert groups[0]["whole_log_odds"] == pytest.approx(-0.9)
    assert groups[0]["window_log_odds"] == pytest.approx([-1.9, 1.1])
    assert groups[0]["feature_vector"] == pytest.approx(
        [-0.9, -1.9, 1.1, 0.0, 1.0, 1.0, 0.0]
    )

    malformed = _readout_rows()[:-1]
    with pytest.raises(ValueError, match="option_order_pair_invalid"):
        exp.aggregate_readouts(malformed)


def test_fit_bundle_uses_frozen_seeds_spline_bound_and_exact_normalization() -> None:
    """SCENARIO-VERIFY-7495-HEADS: fair compact heads normalize exactly."""

    training = _numeric_rows("training", 24)
    calibration = _numeric_rows("calibration_tuning", 12, offset=1)
    bundle = exp.fit_window_bundle(training, calibration, steps=12)

    assert bundle["training_seeds"] == list(exp.TRAINING_SEEDS)
    assert bundle["heldout_labels_consumed"] is False
    assert bundle["feature_transform"]["spline_coefficient_count"] <= 256
    assert bundle["feature_transform"]["input_sha256"] == exp.canonical_hash(
        [row["feature_vector"] for row in training]
    )
    assert set(bundle["heads"]) == {
        "conditional_gibbs",
        "whole_only_gibbs",
        "window_logistic",
        "shuffled_label_gibbs",
    }
    assert all(len(states) == 5 for states in bundle["heads"].values())
    assert all(
        state["hyperparameter_budget"] == bundle["identical_hyperparameter_budget"]
        for states in bundle["heads"].values()
        for state in states
    )
    probabilities = exp.exact_binary_gibbs_probabilities(np.asarray([-3.0, 0.0, 2.0]))
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert np.all(probabilities > 0.0)
    assert bundle["bundle_sha256"] == exp.bundle_hash(bundle)


def test_score_and_reduce_average_seeds_before_fresh_group_inference() -> None:
    """SCENARIO-VERIFY-7495-PROBABILITY: seeds never multiply test support."""

    training = _numeric_rows("training", 24)
    calibration = _numeric_rows("calibration_tuning", 20, offset=1)
    test = _numeric_rows("test", 100)
    bundle = exp.fit_window_bundle(training, calibration, steps=15)
    calibration_predictions = exp.score_bundle(bundle, calibration)
    policies = exp.freeze_cost_policies(calibration_predictions)
    test_predictions = exp.score_bundle(bundle, test)
    reduction = exp.reduce_prediction_rows(
        [*calibration_predictions, *test_predictions],
        frozen_policies=policies,
        bootstrap_draws=200,
    )

    assert reduction["confirmatory_support_score"] == 1
    assert reduction["test_group_count"] == 100
    assert reduction["test_class_support"] == {"0": 50, "1": 50}
    assert reduction["probability_report"]["test"]["conditional_gibbs"]["n_groups"] == 100
    assert set(reduction["probability_comparisons"]["brier"]) == {
        "window_logistic",
        "whole_only_gibbs",
    }
    assert len(reduction["decision_report"]["cells"]) == 9
    assert reduction["decision_benefit_score"] == int(
        all(row["benefit_passed"] for row in reduction["decision_report"]["cells"])
    )
    assert reduction["length_slices"]
    assert reduction["source_family_slices"]["fixture"]


def test_support_and_all_nine_utility_gates_fail_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-VERIFY-7495-UTILITY/NULL: low support or one weak cell is null."""

    rows = _numeric_rows("calibration_tuning", 8) + _numeric_rows("test", 10)
    prediction_rows = []
    for row in rows:
        for arm in exp.ALL_ARMS:
            seeds = exp.TRAINING_SEEDS if arm in exp.SEEDED_ARMS else (None,)
            for seed in seeds:
                prediction_rows.append(
                    {
                        **{key: row[key] for key in exp.PREDICTION_CONTEXT_FIELDS},
                        "arm": arm,
                        "seed": seed,
                        "probability": 0.8 if row["label"] else 0.2,
                        "failed": False,
                    }
                )
    policies = exp.freeze_cost_policies(
        [row for row in prediction_rows if row["role"] == "calibration_tuning"]
    )
    reduction = exp.reduce_prediction_rows(
        prediction_rows, frozen_policies=policies, bootstrap_draws=50
    )
    assert reduction["confirmatory_support_score"] == 0
    assert reduction["probability_benefit_score"] == 0

    fake_cells = [
        {"cell_id": str(index), "benefit_passed": index != 8} for index in range(9)
    ]
    monkeypatch.setattr(exp, "_decision_cells", lambda *args, **kwargs: fake_cells)
    changed = exp.reduce_prediction_rows(
        prediction_rows, frozen_policies=policies, bootstrap_draws=20
    )
    assert changed["decision_benefit_score"] == 0


def test_fixture_artifact_validates_and_mutations_fail(tmp_path) -> None:
    """SCENARIO-VERIFY-7495-E2E: raw rows and frozen hashes cold-replay."""

    artifact = exp.fixture_artifact(tmp_path)
    assert exp.validate_artifact(artifact, root=tmp_path) == []
    assert exp.independent_reduce(artifact, root=tmp_path) == artifact["independent_reduction"]
    assert set(exp.REQUIRED_FIELDS) <= set(artifact["field_principles"])
    assert all("prevent" in row["principle"] for row in artifact["acceptance_gate_results"])

    changed = deepcopy(artifact)
    changed["MODEL_SPECS"] = ["historical-model"]
    assert "current_model_declaration_invalid" in exp.validate_artifact(changed, root=tmp_path)
    changed = deepcopy(artifact)
    changed["frozen_policy_manifest"]["bundle_sha256"] = "sha256:changed"
    assert "frozen_policy_hash_mismatch" in exp.validate_artifact(changed, root=tmp_path)
    changed = deepcopy(artifact)
    changed["window_calibration_complete_score"] = 0
    assert "completion_score_mismatch" in exp.validate_artifact(changed, root=tmp_path)


def test_preconditions_preserve_original_flags_and_missing_input_blocks(tmp_path) -> None:
    """SCENARIO-VERIFY-7495-PREREQUISITE: exact producer fields gate fitting."""

    fit = {
        "experiment_id": exp.FIT_EXPECTED["experiment_id"],
        "schema": exp.FIT_EXPECTED["schema"],
        "milestone": exp.MILESTONE,
        "terminal_status": "complete",
        "honest_verdict": "complete_null_window_fit_capture_ready_predictive_benefit_not_tested",
        "verdict_class": "null",
        "flagged_adversarial": False,
        "window_fit_ready_score": 1,
        "raw_logit_shards": [],
    }
    evaluation = {
        "experiment_id": exp.EVAL_EXPECTED["experiment_id"],
        "schema": exp.EVAL_EXPECTED["schema"],
        "milestone": exp.MILESTONE,
        "terminal_status": "complete",
        "honest_verdict": "complete_null_window_evaluation_capture_ready_predictive_benefit_not_tested",
        "verdict_class": "null",
        "flagged_adversarial": False,
        "window_evaluation_ready_score": 1,
        "raw_logit_shards": [],
    }
    (tmp_path / "fit.json").write_text(json.dumps(fit), encoding="utf-8")
    (tmp_path / "eval.json").write_text(json.dumps(evaluation), encoding="utf-8")
    checks, hashes = exp.capture_preconditions(tmp_path / "fit.json", tmp_path / "eval.json")
    assert all(row["passed"] for row in checks)
    assert hashes["fit.json"]["original_flagged_adversarial"] is False
    assert hashes["eval.json"]["original_verdict_class"] == "null"

    (tmp_path / "eval.json").unlink()
    failed, _ = exp.capture_preconditions(tmp_path / "fit.json", tmp_path / "eval.json")
    assert any(row["passed"] is False for row in failed)

