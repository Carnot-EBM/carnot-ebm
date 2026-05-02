"""Tests for Exp 1131 Lagrangian cascade v2.

Spec: REQ-VERIFY-1131, SCENARIO-VERIFY-1131.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from carnot.eval.lagrangian_cascade_v2 import (
    ALLOWED_HONEST_VERDICTS,
    FEATURE_NAMES,
    MIN_TP_CONSTRAINT,
    REQUIRED_ARTIFACT_FIELDS,
    FeatureNormalizer,
    LagrangianCascadeMLP,
    batch_tp_rate,
    build_exp1131_artifact,
    evaluate_depth_predictions,
    extract_raw_features,
    infer_required_depths,
    load_fover_examples,
    route_depths,
    run_experiment,
    train_mlp,
    update_dual_lambda,
)


class _FakeSemEnergyProbe:
    def score(self, response: str) -> float:
        return 0.42 if "Step 2:" in response else -1.25


class _FallbackSemEnergyProbe:
    def score_response_proxy(self, response: str) -> float:
        return -0.75 if response else 0.0


def test_extract_raw_features_uses_verifier_score_length_and_step_markers() -> None:
    """REQ-VERIFY-1131-1: v2 features include verifier score plus response shape."""
    example = {
        "step_text": "Step 1: Compute 2 + 2.\nStep 2: Therefore the answer is 4.",
        "label": "correct",
    }

    features = extract_raw_features(example, _FakeSemEnergyProbe())

    assert FEATURE_NAMES == ["sem_energy_score", "response_length", "step_count"]
    assert features.tolist() == [pytest.approx(0.42), 13.0, 2.0]


def test_extract_raw_features_falls_back_to_proxy_score_method() -> None:
    """REQ-VERIFY-1131-1: SemEnergyProbe proxy scores remain usable as features."""
    features = extract_raw_features({"response": "plain answer"}, _FallbackSemEnergyProbe())

    assert features.tolist() == [pytest.approx(-0.75), 2.0, 0.0]


def test_feature_normalizer_round_trips_zscore_features() -> None:
    """REQ-VERIFY-1131-1: response length and step count are normalized for the MLP."""
    normalizer = FeatureNormalizer()
    X = np.array([[-1.0, 10.0, 1.0], [0.0, 20.0, 3.0]], dtype=np.float32)

    normalizer.fit(X)
    transformed = normalizer.transform(X)

    assert transformed.mean(axis=0) == pytest.approx([0.0, 0.0, 0.0])
    assert transformed.std(axis=0) == pytest.approx([1.0, 1.0, 1.0])


def test_feature_normalizer_requires_fit_before_transform() -> None:
    """REQ-VERIFY-1131-1: normalizer state must come from the train split."""
    normalizer = FeatureNormalizer()

    with pytest.raises(RuntimeError, match="fit must be called"):
        normalizer.transform(np.zeros((1, 3), dtype=np.float32))


def test_mlp_uses_hidden_size_128_two_relu_layers() -> None:
    """REQ-VERIFY-1131-2: the router architecture is 3 -> 128 -> 128 -> 5."""
    model = LagrangianCascadeMLP(input_dim=3, hidden_dim=128, output_dim=5, seed=7)
    X = np.zeros((2, 3), dtype=np.float32)

    logits, cache = model.forward(X)

    assert model.hidden_dim == 128
    assert model.hidden_layer_count == 2
    assert logits.shape == (2, 5)
    assert cache["h1"].shape == (2, 128)
    assert cache["h2"].shape == (2, 128)


def test_train_mlp_exercises_backprop_and_dual_update() -> None:
    """REQ-VERIFY-1131-2/3: training runs Adam updates and returns lambda."""
    X = np.zeros((4, 3), dtype=np.float32)
    y = np.array([0, 4, 0, 4], dtype=np.int32)
    labels = ["correct", "incorrect", "correct", "incorrect"]
    sem_scores = np.array([-3.0, -3.0, -3.0, -3.0], dtype=np.float32)
    required_depths = np.array([1, 5, 1, 5], dtype=np.int32)
    raw_lengths = np.array([10, 10, 10, 10], dtype=np.float32)

    model, lambda_final = train_mlp(
        X,
        y,
        labels,
        sem_scores,
        required_depths,
        raw_lengths,
        epochs=1,
        batch_size=2,
        lr=1e-3,
    )

    assert isinstance(model, LagrangianCascadeMLP)
    assert lambda_final >= 0.0


def test_lagrangian_dual_increases_only_when_batch_tp_violates_constraint() -> None:
    """REQ-VERIFY-1131-3: lambda increases by 0.01 when TP_batch < 0.90."""
    assert update_dual_lambda(0.2, 0.89) == pytest.approx(0.21)
    assert update_dual_lambda(0.2, MIN_TP_CONSTRAINT) == pytest.approx(0.2)


def test_batch_tp_rate_returns_perfect_when_no_incorrect_examples() -> None:
    """REQ-VERIFY-1131-3: TP constraint is vacuously met for all-correct batches."""
    assert (
        batch_tp_rate(
            ["correct"],
            np.array([1], dtype=np.int32),
            np.array([5], dtype=np.int32),
        )
        == 1.0
    )


def test_route_depths_uses_dual_bias_and_energy_guard() -> None:
    """REQ-VERIFY-1131-1: sem_energy_score is a primary routing signal."""
    model = LagrangianCascadeMLP(input_dim=3, hidden_dim=128, output_dim=5, seed=3)
    X = np.array([[-1.2, 0.0, 0.0], [-0.2, 0.0, 0.0]], dtype=np.float32)
    sem_energy_scores = np.array([-1.2, -0.2], dtype=np.float32)

    depths = route_depths(model, X, sem_energy_scores, lambda_value=0.0)
    conservative_depths = route_depths(model, X, sem_energy_scores, lambda_value=2.0)

    assert depths[1] == 5
    assert conservative_depths.min() >= depths.min()
    assert conservative_depths.max() == 5


def test_route_depths_uses_raw_length_for_short_response_guard() -> None:
    """REQ-VERIFY-1131-1: very short responses are sent to the full cascade."""
    model = LagrangianCascadeMLP(input_dim=3, hidden_dim=128, output_dim=5, seed=3)
    X = np.zeros((2, 3), dtype=np.float32)
    sem_energy_scores = np.array([-3.0, -3.0], dtype=np.float32)
    raw_lengths = np.array([1.0, 10.0], dtype=np.float32)

    depths = route_depths(model, X, sem_energy_scores, 0.0, raw_response_lengths=raw_lengths)

    assert depths.tolist() == [5, 1]


def test_infer_required_depths_covers_all_depth_buckets() -> None:
    """REQ-VERIFY-1131-1: depth labels are derived from verifier-score features."""
    raw_features = np.array(
        [
            [-1.0, 50.0, 2.0],
            [-3.0, 10.0, 0.0],
            [-1.0, 50.0, 0.0],
            [-1.0, 100.0, 2.0],
            [-1.0, 200.0, 2.0],
        ],
        dtype=np.float32,
    )
    labels = ["incorrect", "correct", "correct", "correct", "correct"]

    assert infer_required_depths(raw_features, labels).tolist() == [5, 1, 2, 3, 4]


def test_evaluate_depth_predictions_counts_tp_and_cost_savings() -> None:
    """SCENARIO-VERIFY-1131: evaluation reports TP delta and positive savings."""
    metrics = evaluate_depth_predictions(
        labels=["correct", "incorrect", "correct", "incorrect"],
        predicted_depths=np.array([1, 5, 1, 5], dtype=np.int32),
        required_depths=np.array([1, 5, 1, 5], dtype=np.int32),
    )

    assert metrics["adaptive_tp_rate"] == pytest.approx(1.0)
    assert metrics["fixed_tp_rate"] == pytest.approx(1.0)
    assert metrics["accuracy_delta"] == pytest.approx(0.0)
    assert metrics["cost_savings_pct"] > 0.0


def test_evaluate_depth_predictions_handles_all_correct_holdout() -> None:
    """SCENARIO-VERIFY-1131: no-incorrect holdouts report neutral TP rates."""
    metrics = evaluate_depth_predictions(
        labels=["correct", "correct"],
        predicted_depths=np.array([1, 2], dtype=np.int32),
        required_depths=np.array([1, 2], dtype=np.int32),
    )

    assert metrics["adaptive_tp_rate"] == 1.0
    assert metrics["fixed_tp_rate"] == 1.0


def test_build_artifact_schema_and_honest_verdict() -> None:
    """REQ-VERIFY-1131-4: the Exp1131 deliverable schema is stable."""
    artifact = build_exp1131_artifact(
        n_training_examples=5000,
        n_holdout_examples=500,
        mlp_val_accuracy=0.91,
        lambda_final=0.31,
        metrics={
            "adaptive_tp_rate": 0.96,
            "fixed_tp_rate": 1.0,
            "accuracy_delta": -0.04,
            "cost_savings_pct": 18.5,
            "fixed_cascade_cost_ms": 111.017,
            "adaptive_cascade_cost_ms": 90.0,
        },
        predicted_depth_distribution={1: 250, 2: 100, 3: 75, 4: 25, 5: 50},
        cascade_depth_distribution={1: 260, 2: 90, 3: 80, 4: 20, 5: 50},
        duration_s=1.25,
    )

    for field in REQUIRED_ARTIFACT_FIELDS:
        assert field in artifact, f"missing required field: {field}"
    assert artifact["mlp_hidden_size"] == 128
    assert artifact["min_tp_constraint"] == MIN_TP_CONSTRAINT
    assert artifact["verifier_score_features_used"] == FEATURE_NAMES
    assert artifact["cascade_v2_accuracy_delta_above_neg05"] is True
    assert artifact["cost_savings_pct_positive"] is True
    assert artifact["honest_verdict"] == "savings_positive_accuracy_acceptable"
    assert artifact["honest_verdict"] in ALLOWED_HONEST_VERDICTS


@pytest.mark.parametrize(
    ("cost_savings_pct", "accuracy_delta", "expected"),
    [
        (-1.0, 0.0, "no_improvement_over_exp1123"),
        (1.0, 0.0, "savings_accuracy_both_positive"),
        (1.0, -0.10, "savings_positive_accuracy_still_degraded"),
    ],
)
def test_build_artifact_maps_all_honest_verdicts(
    cost_savings_pct: float,
    accuracy_delta: float,
    expected: str,
) -> None:
    """REQ-VERIFY-1131-4: honest verdict enum is deterministic."""
    artifact = build_exp1131_artifact(
        n_training_examples=4,
        n_holdout_examples=2,
        mlp_val_accuracy=0.5,
        lambda_final=0.0,
        metrics={
            "adaptive_tp_rate": 1.0 + min(accuracy_delta, 0.0),
            "fixed_tp_rate": 1.0,
            "accuracy_delta": accuracy_delta,
            "cost_savings_pct": cost_savings_pct,
            "fixed_cascade_cost_ms": 111.017,
            "adaptive_cascade_cost_ms": 100.0,
        },
        predicted_depth_distribution={1: 1, 2: 0, 3: 0, 4: 0, 5: 1},
        cascade_depth_distribution={1: 1, 2: 0, 3: 0, 4: 0, 5: 1},
        duration_s=0.1,
    )

    assert artifact["honest_verdict"] == expected


def test_load_fover_examples_supports_json_array_and_jsonl(tmp_path: Path) -> None:
    """REQ-VERIFY-1131-4: experiment input loading supports local FoVer formats."""
    array_path = tmp_path / "corpus.json"
    jsonl_path = tmp_path / "corpus.jsonl"
    rows = [{"step_text": "Step 1: ok", "label": "correct"}]
    array_path.write_text(json.dumps(rows), encoding="utf-8")
    jsonl_path.write_text(json.dumps(rows[0]) + "\n", encoding="utf-8")

    assert load_fover_examples(array_path) == rows
    assert load_fover_examples(jsonl_path) == rows


def test_run_experiment_writes_tiny_artifact(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-1131: the runner writes the required artifact schema."""
    corpus_path = tmp_path / "tiny_fover.json"
    result_path = tmp_path / "artifact.json"
    rows = [
        {"step_text": "Step 1: 2 + 2 = 4", "label": "correct"},
        {"step_text": "wrong prose " * 20, "label": "incorrect"},
        {"step_text": "Step 1: 3 + 3 = 6", "label": "correct"},
        {"step_text": "Step 1: 4 + 4 = 8", "label": "correct"},
        {"step_text": "Step 1: 5 + 5 = 10", "label": "correct"},
        {"step_text": "bad answer " * 15, "label": "incorrect"},
    ]
    corpus_path.write_text(json.dumps(rows), encoding="utf-8")

    artifact = run_experiment(corpus_path, result_path, n_train=4, n_holdout=2, epochs=1)

    assert result_path.exists()
    assert artifact["mlp_hidden_size"] == 128
    assert artifact["verifier_score_features_used"] == FEATURE_NAMES
    assert artifact["honest_verdict"] in ALLOWED_HONEST_VERDICTS


def test_generated_artifact_has_required_schema_when_present() -> None:
    """REQ-VERIFY-1131-4: generated artifact keeps required fields and verdict enum."""
    path = Path("results/experiment_1131_lagrangian_cascade_v2.json")
    if not path.exists():
        pytest.skip("Exp1131 artifact has not been generated yet.")

    artifact = json.loads(path.read_text(encoding="utf-8"))

    for field in REQUIRED_ARTIFACT_FIELDS:
        assert field in artifact, f"missing required field: {field}"
    assert artifact["mlp_hidden_size"] == 128
    assert artifact["verifier_score_features_used"] == FEATURE_NAMES
    assert artifact["min_tp_constraint"] == MIN_TP_CONSTRAINT
    assert artifact["adaptive_tp_rate"] >= 0.90 * artifact["fixed_tp_rate"]
    assert artifact["honest_verdict"] in ALLOWED_HONEST_VERDICTS
