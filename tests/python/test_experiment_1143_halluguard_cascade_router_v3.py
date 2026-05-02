"""Tests for Exp 1143 HalluGuard cascade router v3.

Spec: REQ-VERIFY-1143, SCENARIO-VERIFY-1143.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from carnot.eval.halluguard_cascade_router_v3 import (
    ALLOWED_HONEST_VERDICTS,
    FEATURE_NAMES,
    HALLUGUARD_FEATURES_ADDED,
    REQUIRED_ARTIFACT_FIELDS,
    FeatureNormalizer,
    HashingTextEmbedder,
    LagrangianCascadeMLP,
    build_exp1143_artifact,
    build_halluguard_feature_matrix,
    cosine_distance,
    entropy_proxy,
    extract_surface_features,
    evaluate_goodfire_failures,
    load_goodfire_exemplars,
    route_depths_v3,
    run_experiment,
    summarize_goodfire_failure_routing,
    train_halluguard_mlp,
)


class _FakeSemEnergyProbe:
    def score(self, response: str) -> float:
        return 0.75 if "wrong" in response else -3.0


class _FallbackSemEnergyProbe:
    def score_response_proxy(self, response: str) -> float:
        return -0.5 if response else 0.0


class _FakeEmbedder:
    def encode(self, texts: list[str], **_: object) -> np.ndarray:
        vectors: list[list[float]] = []
        for text in texts:
            if "far" in text or "novel" in text:
                vectors.append([-1.0, 0.0, 0.0])
            elif "beta" in text:
                vectors.append([0.8, 0.6, 0.0])
            else:
                vectors.append([1.0, 0.0, 0.0])
        return np.array(vectors, dtype=np.float32)


def test_entropy_proxy_counts_unique_token_ratio() -> None:
    """REQ-VERIFY-1143: entropy_proxy is unique tokens / total tokens."""
    assert entropy_proxy("red red blue") == pytest.approx(2.0 / 3.0)
    assert entropy_proxy("Answer: 4, answer 4.") == pytest.approx(2.0 / 4.0)
    assert entropy_proxy("") == 0.0


def test_halluguard_feature_matrix_adds_entropy_and_embedding_distance() -> None:
    """REQ-VERIFY-1143: router features expand from three inputs to five."""
    examples = [
        {"prompt": "alpha train", "response": "Step 1: ok ok", "label": "correct"},
        {"prompt": "beta train", "response": "Step 1: ok", "label": "correct"},
        {"prompt": "far holdout", "response": "wrong novel token stream", "label": "incorrect"},
    ]

    raw, centroid = build_halluguard_feature_matrix(
        examples,
        _FakeSemEnergyProbe(),
        _FakeEmbedder(),
        train_indices=np.array([0, 1], dtype=np.int32),
    )

    assert FEATURE_NAMES == [
        "sem_energy_score",
        "response_length",
        "step_count",
        "entropy_proxy",
        "embedding_distance",
    ]
    assert HALLUGUARD_FEATURES_ADDED == ["entropy_proxy", "embedding_distance"]
    assert raw.shape == (3, 5)
    assert centroid.shape == (3,)
    assert raw[2, 3] > raw[0, 3]
    assert raw[2, 4] > raw[0, 4]


def test_surface_features_cover_fallback_text_and_semenergy_paths() -> None:
    """REQ-VERIFY-1143: feature extraction tolerates sparse local corpus rows."""
    features = extract_surface_features(
        {"completion": "fallback text text"}, _FallbackSemEnergyProbe()
    )
    empty_features = extract_surface_features({}, _FallbackSemEnergyProbe())
    raw, _centroid = build_halluguard_feature_matrix(
        [{"buggy_response": "only response text"}],
        _FallbackSemEnergyProbe(),
        _FakeEmbedder(),
        train_indices=np.array([0], dtype=np.int32),
    )

    assert features.tolist() == [pytest.approx(-0.5), 3.0, 0.0, pytest.approx(2.0 / 3.0)]
    assert empty_features.tolist() == [0.0, 0.0, 0.0, 0.0]
    assert raw.shape == (1, 5)


def test_cosine_distance_handles_identical_opposite_and_zero_vectors() -> None:
    """REQ-VERIFY-1143: embedding_distance is cosine distance to the centroid."""
    assert cosine_distance(np.array([1.0, 0.0]), np.array([1.0, 0.0])) == pytest.approx(0.0)
    assert cosine_distance(np.array([-1.0, 0.0]), np.array([1.0, 0.0])) == pytest.approx(2.0)
    assert cosine_distance(np.array([0.0, 0.0]), np.array([1.0, 0.0])) == pytest.approx(1.0)


def test_hashing_text_embedder_is_deterministic_and_normalized() -> None:
    """REQ-VERIFY-1143: local embedding fallback still yields measured distances."""
    embedder = HashingTextEmbedder(dim=16)

    vectors_a = embedder.encode(["same text", "different text"])
    vectors_b = embedder.encode(["same text", "different text"])

    assert vectors_a.shape == (2, 16)
    assert np.allclose(vectors_a, vectors_b)
    assert np.linalg.norm(vectors_a, axis=1).tolist() == pytest.approx([1.0, 1.0])


def test_load_goodfire_exemplars_supports_empty_json_array_and_jsonl(tmp_path: Path) -> None:
    """REQ-VERIFY-1143: Goodfire exemplar loader handles local artifact formats."""
    empty_path = tmp_path / "empty.jsonl"
    array_path = tmp_path / "rows.json"
    jsonl_path = tmp_path / "rows.jsonl"
    rows = [{"id": "ex1", "buggy_response": "wrong"}]
    empty_path.write_text("", encoding="utf-8")
    array_path.write_text(json.dumps(rows), encoding="utf-8")
    jsonl_path.write_text(json.dumps(rows[0]) + "\n", encoding="utf-8")

    assert load_goodfire_exemplars(empty_path) == []
    assert load_goodfire_exemplars(array_path) == rows
    assert load_goodfire_exemplars(jsonl_path) == rows


def test_train_halluguard_mlp_uses_five_feature_input_layer() -> None:
    """REQ-VERIFY-1143: MLP architecture is 5 -> 128 -> 128 -> 5."""
    X = np.zeros((4, 5), dtype=np.float32)
    y = np.array([0, 4, 0, 4], dtype=np.int32)
    labels = ["correct", "incorrect", "correct", "incorrect"]
    sem_scores = np.array([-3.0, 0.5, -3.0, 0.5], dtype=np.float32)
    required_depths = np.array([1, 5, 1, 5], dtype=np.int32)
    raw_lengths = np.array([10.0, 10.0, 10.0, 10.0], dtype=np.float32)
    entropy_scores = np.array([0.2, 0.9, 0.2, 0.9], dtype=np.float32)
    embedding_distances = np.array([0.1, 0.8, 0.1, 0.8], dtype=np.float32)

    model, lambda_final = train_halluguard_mlp(
        X,
        y,
        labels,
        sem_scores,
        required_depths,
        raw_lengths,
        entropy_scores,
        embedding_distances,
        epochs=1,
        batch_size=2,
        lr=1e-3,
    )

    assert isinstance(model, LagrangianCascadeMLP)
    assert model.W1.shape == (5, 128)
    assert model.hidden_layer_count == 2
    assert lambda_final >= 0.0


def test_route_depths_v3_escalates_high_entropy_or_embedding_distance() -> None:
    """SCENARIO-VERIFY-1143: HalluGuard risk features can force k=5 routing."""
    model = LagrangianCascadeMLP(input_dim=5, hidden_dim=128, output_dim=5, seed=1143)
    X = np.zeros((3, 5), dtype=np.float32)
    sem_scores = np.array([-3.0, -3.0, -3.0], dtype=np.float32)
    raw_lengths = np.array([10.0, 10.0, 10.0], dtype=np.float32)
    entropy_scores = np.array([0.95, 0.2, 0.2], dtype=np.float32)
    embedding_distances = np.array([0.1, 0.95, 0.1], dtype=np.float32)

    depths = route_depths_v3(
        model,
        X,
        sem_scores,
        lambda_value=0.0,
        raw_response_lengths=raw_lengths,
        entropy_scores=entropy_scores,
        embedding_distances=embedding_distances,
        entropy_threshold=0.9,
        distance_threshold=0.9,
    )

    assert depths.tolist() == [5, 5, 1]


def test_summarize_goodfire_failure_routing_detects_thinkprm_miss_explanation() -> None:
    """SCENARIO-VERIFY-1143: Goodfire ThinkPRM misses are checked for k=5 routing."""
    per_exemplar_results = [
        {"id": "ex1", "tier_results": {"tier_0a_thinkprm": False}},
        {"id": "ex2", "tier_results": {"tier_0a_thinkprm": True}},
    ]
    exemplar_rows = [
        {"id": "ex1", "source": "goodfire_published"},
        {"id": "ex2", "source": "goodfire_published"},
    ]

    summary = summarize_goodfire_failure_routing(
        per_exemplar_results=per_exemplar_results,
        exemplar_rows=exemplar_rows,
        predicted_depths=np.array([5, 1], dtype=np.int32),
        entropy_scores=np.array([0.95, 0.1], dtype=np.float32),
        embedding_distances=np.array([0.2, 0.1], dtype=np.float32),
        entropy_threshold=0.9,
        distance_threshold=0.9,
    )

    assert summary["thinkprm_miss_count"] == 1
    assert summary["thinkprm_miss_k5_route_rate"] == pytest.approx(1.0)
    assert summary["hallu_feature_flag_rate"] == pytest.approx(1.0)
    assert summary["halluguard_features_explain_goodfire_failures"] is True


def test_summarize_goodfire_failure_routing_handles_missing_ids_and_no_misses() -> None:
    """SCENARIO-VERIFY-1143: Goodfire analysis is honest when ThinkPRM has no misses."""
    summary = summarize_goodfire_failure_routing(
        per_exemplar_results=[
            {"id": "missing", "tier_results": {"tier_0a_thinkprm": False}},
            {"id": "ex1", "tier_results": {"tier_0a_thinkprm": True}},
        ],
        exemplar_rows=[{"id": "ex1"}],
        predicted_depths=np.array([1], dtype=np.int32),
        entropy_scores=np.array([0.1], dtype=np.float32),
        embedding_distances=np.array([0.1], dtype=np.float32),
    )

    assert summary["thinkprm_miss_count"] == 0
    assert summary["halluguard_features_explain_goodfire_failures"] is False


def test_evaluate_goodfire_failures_scores_real_exemplar_files(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-1143: Goodfire files are joined by id before analysis."""
    goodfire_artifact = tmp_path / "exp1132.json"
    exemplar_corpus = tmp_path / "llm_failure_exemplars.jsonl"
    goodfire_artifact.write_text(
        json.dumps(
            {
                "per_exemplar_results": [
                    {"id": "missing", "tier_results": {"tier_0a_thinkprm": False}},
                    {"id": "ex1", "tier_results": {"tier_0a_thinkprm": False}},
                    {"id": "ex2", "tier_results": {"tier_0a_thinkprm": True}},
                ]
            }
        ),
        encoding="utf-8",
    )
    exemplar_corpus.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "id": "ex1",
                        "source": "goodfire_published",
                        "prompt": "novel far prompt",
                        "buggy_response": "wrong novel token stream",
                    }
                ),
                json.dumps(
                    {
                        "id": "ex2",
                        "source": "goodfire_published",
                        "prompt": "alpha prompt",
                        "buggy_response": "ok ok ok ok",
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    normalizer = FeatureNormalizer()
    normalizer.fit(np.zeros((2, 5), dtype=np.float32))
    model = LagrangianCascadeMLP(input_dim=5, hidden_dim=128, output_dim=5, seed=1143)

    summary = evaluate_goodfire_failures(
        model=model,
        normalizer=normalizer,
        semenergy_probe=_FakeSemEnergyProbe(),
        embedding_model=_FakeEmbedder(),
        centroid=np.array([1.0, 0.0, 0.0], dtype=np.float32),
        goodfire_artifact_path=goodfire_artifact,
        exemplar_corpus_path=exemplar_corpus,
        lambda_value=0.0,
        entropy_threshold=0.7,
        distance_threshold=0.7,
    )

    assert summary["goodfire_exemplars_scored"] == 2
    assert summary["thinkprm_miss_count"] == 1
    assert summary["halluguard_features_explain_goodfire_failures"] is True


def test_build_exp1143_artifact_schema_and_verdict() -> None:
    """REQ-VERIFY-1143: Exp1143 artifact exposes all required schema fields."""
    artifact = build_exp1143_artifact(
        training_set_size=5000,
        holdout_set_size=500,
        mlp_val_accuracy=0.91,
        lambda_final=0.12,
        metrics={
            "adaptive_tp_rate": 1.0,
            "fixed_tp_rate": 1.0,
            "accuracy_delta": 0.0,
            "cost_savings_pct": 12.5,
            "fixed_cascade_cost_ms": 111.017,
            "adaptive_cascade_cost_ms": 97.0,
        },
        predicted_depth_distribution={1: 100, 2: 50, 3: 25, 4: 25, 5: 300},
        cascade_depth_distribution={1: 100, 2: 50, 3: 25, 4: 25, 5: 300},
        goodfire_summary={
            "halluguard_features_explain_goodfire_failures": True,
            "thinkprm_miss_count": 10,
            "thinkprm_miss_k5_route_rate": 0.9,
            "hallu_feature_flag_rate": 0.9,
        },
        duration_s=1.25,
        embedding_backend="fake",
    )

    for field in REQUIRED_ARTIFACT_FIELDS:
        assert field in artifact, f"missing required field: {field}"
    assert artifact["halluguard_features_added"] == HALLUGUARD_FEATURES_ADDED
    assert artifact["n_router_features_before"] == 3
    assert artifact["n_router_features_after"] == 5
    assert artifact["halluguard_routing_feature_measured"] is True
    assert artifact["honest_verdict"] == "features_explain_goodfire_failures"
    assert artifact["honest_verdict"] in ALLOWED_HONEST_VERDICTS


@pytest.mark.parametrize(
    ("cost_savings_pct", "accuracy_delta", "expected"),
    [
        (12.5, 0.0, "routing_improved_with_halluguard_features"),
        (0.0, 0.0, "routing_degraded"),
        (12.5, -0.10, "routing_degraded"),
        (12.5, -0.02, "routing_unchanged"),
    ],
)
def test_build_exp1143_artifact_maps_non_goodfire_verdicts(
    cost_savings_pct: float,
    accuracy_delta: float,
    expected: str,
) -> None:
    """REQ-VERIFY-1143: honest verdict enum is deterministic."""
    artifact = build_exp1143_artifact(
        training_set_size=4,
        holdout_set_size=2,
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
        goodfire_summary={"halluguard_features_explain_goodfire_failures": False},
        duration_s=0.1,
        embedding_backend="fake",
    )

    assert artifact["honest_verdict"] == expected


def test_run_experiment_writes_tiny_halluguard_artifact(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-1143: runner writes the HalluGuard router deliverable."""
    corpus_path = tmp_path / "tiny_fover.json"
    result_path = tmp_path / "experiment_1143.json"
    goodfire_artifact = tmp_path / "exp1132.json"
    exemplar_corpus = tmp_path / "llm_failure_exemplars.jsonl"

    rows = [
        {"prompt": "alpha train", "step_text": "Step 1: 2 + 2 = 4", "label": "correct"},
        {"prompt": "beta train", "step_text": "Step 1: 3 + 3 = 6", "label": "correct"},
        {"prompt": "alpha train", "step_text": "Step 1: 4 + 4 = 8", "label": "correct"},
        {"prompt": "beta train", "step_text": "Step 1: 5 + 5 = 10", "label": "correct"},
        {"prompt": "far holdout", "step_text": "wrong novel token stream", "label": "incorrect"},
        {"prompt": "far holdout", "step_text": "wrong novel token stream", "label": "incorrect"},
    ]
    corpus_path.write_text(json.dumps(rows), encoding="utf-8")
    goodfire_artifact.write_text(
        json.dumps(
            {"per_exemplar_results": [{"id": "ex1", "tier_results": {"tier_0a_thinkprm": False}}]}
        ),
        encoding="utf-8",
    )
    exemplar_corpus.write_text(
        json.dumps(
            {
                "id": "ex1",
                "source": "goodfire_published",
                "prompt": "far holdout",
                "buggy_response": "wrong novel token stream",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    artifact = run_experiment(
        corpus_path=corpus_path,
        results_path=result_path,
        goodfire_artifact_path=goodfire_artifact,
        exemplar_corpus_path=exemplar_corpus,
        n_train=4,
        n_holdout=2,
        epochs=1,
        semenergy_probe=_FakeSemEnergyProbe(),
        embedding_model=_FakeEmbedder(),
        embedding_backend="fake",
    )

    assert result_path.exists()
    assert artifact["training_set_size"] == 4
    assert artifact["holdout_set_size"] == 2
    assert artifact["n_router_features_after"] == 5
    assert artifact["halluguard_routing_feature_measured"] is True
    assert artifact["honest_verdict"] in ALLOWED_HONEST_VERDICTS


def test_run_experiment_default_probe_path_validates_minimum_corpus(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-VERIFY-1143: runner default dependencies are wired before split checks."""
    import carnot.eval.halluguard_cascade_router_v3 as exp1143

    corpus_path = tmp_path / "one_row.json"
    result_path = tmp_path / "experiment_1143.json"
    goodfire_artifact = tmp_path / "exp1132.json"
    exemplar_corpus = tmp_path / "llm_failure_exemplars.jsonl"
    corpus_path.write_text(
        json.dumps([{"prompt": "alpha", "step_text": "Step 1: ok", "label": "correct"}]),
        encoding="utf-8",
    )
    goodfire_artifact.write_text(json.dumps({"per_exemplar_results": []}), encoding="utf-8")
    exemplar_corpus.write_text("", encoding="utf-8")
    monkeypatch.setattr(exp1143, "load_embedding_model", lambda: (_FakeEmbedder(), "fake-default"))

    with pytest.raises(ValueError, match="at least two examples"):
        run_experiment(
            corpus_path=corpus_path,
            results_path=result_path,
            goodfire_artifact_path=goodfire_artifact,
            exemplar_corpus_path=exemplar_corpus,
            n_train=1,
            n_holdout=1,
        )
