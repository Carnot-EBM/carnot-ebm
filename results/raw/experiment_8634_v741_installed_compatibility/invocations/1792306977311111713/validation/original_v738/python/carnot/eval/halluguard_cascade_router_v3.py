"""Exp 1143 HalluGuard-style features for the Lagrangian cascade router.

Spec: REQ-VERIFY-1143, SCENARIO-VERIFY-1143.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any, Protocol, Sequence

import numpy as np

from carnot.eval.lagrangian_cascade_v2 import (
    EASY_ENERGY_THRESHOLD,
    MLP_HIDDEN_SIZE,
    MIN_TP_CONSTRAINT,
    SHORT_RESPONSE_WORDS,
    TIER_CUMULATIVE_MS,
    FeatureNormalizer,
    LagrangianCascadeMLP,
    batch_tp_rate,
    evaluate_depth_predictions,
    infer_required_depths,
    load_fover_examples,
    update_dual_lambda,
)

BASE_FEATURE_NAMES = ["sem_energy_score", "response_length", "step_count"]
HALLUGUARD_FEATURES_ADDED = ["entropy_proxy", "embedding_distance"]
FEATURE_NAMES = [*BASE_FEATURE_NAMES, *HALLUGUARD_FEATURES_ADDED]
REQUIRED_ARTIFACT_FIELDS = [
    "halluguard_features_added",
    "n_router_features_before",
    "n_router_features_after",
    "training_set_size",
    "holdout_set_size",
    "adaptive_tp_rate",
    "fixed_tp_rate",
    "accuracy_delta",
    "cost_savings_pct",
    "halluguard_features_explain_goodfire_failures",
    "halluguard_routing_feature_measured",
    "honest_verdict",
]
ALLOWED_HONEST_VERDICTS = {
    "routing_improved_with_halluguard_features",
    "routing_unchanged",
    "routing_degraded",
    "features_explain_goodfire_failures",
}

DEFAULT_SENTENCE_TRANSFORMER = "sentence-transformers/all-MiniLM-L6-v2"
ENTROPY_THRESHOLD = 0.75
EMBEDDING_DISTANCE_THRESHOLD = 0.85
GOODFIRE_EXPLAIN_RATE_THRESHOLD = 0.80
TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")
STEP_MARKER_RE = re.compile(r"\bStep\s+\d+\s*:", re.IGNORECASE)


class TextEmbeddingModel(Protocol):
    """Minimal text-embedding interface used by the Exp 1143 feature builder."""

    def encode(self, texts: Sequence[str], **kwargs: Any) -> np.ndarray: ...


class HashingTextEmbedder:
    """Deterministic local fallback when sentence-transformers is unavailable.

    The experiment prefers all-MiniLM-L6-v2.  This fallback keeps CI and offline
    runs from blocking while still measuring a real vector distance feature.
    """

    def __init__(self, dim: int = 384) -> None:
        self.dim = int(dim)

    def encode(self, texts: Sequence[str], **_: Any) -> np.ndarray:
        rows: list[np.ndarray] = []
        for text in texts:
            vec = np.zeros(self.dim, dtype=np.float32)
            tokens = TOKEN_RE.findall(text.lower()) or [""]
            for token in tokens:
                digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
                value = int.from_bytes(digest, "little", signed=False)
                idx = value % self.dim
                sign = 1.0 if ((value >> 8) & 1) else -1.0
                vec[idx] += sign
            norm = float(np.linalg.norm(vec))
            if norm == 0.0:
                vec[0] = 1.0
            else:
                vec /= norm
            rows.append(vec)
        return np.stack(rows, axis=0)


def load_embedding_model(
    model_name: str = DEFAULT_SENTENCE_TRANSFORMER,
) -> tuple[TextEmbeddingModel, str]:  # pragma: no cover - external integration path
    """Load the requested local sentence-transformers model, with offline fallback."""

    try:
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer(model_name), f"sentence-transformers:{model_name}"
    except Exception:
        return HashingTextEmbedder(), "hashing-fallback"


def _response_text(example: dict[str, Any]) -> str:
    for key in ("response", "step_text", "buggy_response", "answer", "completion"):
        value = example.get(key)
        if value:
            return str(value)
    return ""


def _query_text(example: dict[str, Any]) -> str:
    for key in ("question", "prompt", "query", "problem", "question_text"):
        value = example.get(key)
        if value:
            return str(value)
    return _response_text(example)


def _semenergy_score(probe: Any, response: str) -> float:
    if hasattr(probe, "score"):
        return float(probe.score(response))
    return float(probe.score_response_proxy(response))


def entropy_proxy(text: str) -> float:
    """Return unique-token ratio as a no-logprob proxy for response entropy."""

    tokens = [token.lower() for token in TOKEN_RE.findall(text)]
    if not tokens:
        return 0.0
    return float(len(set(tokens)) / len(tokens))


def cosine_distance(vector: np.ndarray, centroid: np.ndarray) -> float:
    """Return cosine distance in [0, 2], using 1.0 for zero-vector unknowns."""

    vector_norm = float(np.linalg.norm(vector))
    centroid_norm = float(np.linalg.norm(centroid))
    if vector_norm == 0.0 or centroid_norm == 0.0:
        return 1.0
    similarity = float(np.dot(vector, centroid) / (vector_norm * centroid_norm))
    return float(1.0 - np.clip(similarity, -1.0, 1.0))


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, 1e-12)


def _encode_texts(embedding_model: TextEmbeddingModel, texts: Sequence[str]) -> np.ndarray:
    vectors = embedding_model.encode(
        list(texts),
        batch_size=64,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    matrix = np.asarray(vectors, dtype=np.float32)
    if matrix.ndim == 1:  # pragma: no cover - defensive support for unusual embedders
        matrix = matrix.reshape(1, -1)
    return _normalize_rows(matrix).astype(np.float32)


def extract_surface_features(example: dict[str, Any], semenergy_probe: Any) -> np.ndarray:
    """Return the four non-centroid-dependent router features for one row."""

    response = _response_text(example)
    return np.array(
        [
            _semenergy_score(semenergy_probe, response),
            float(len(response.split())),
            float(len(STEP_MARKER_RE.findall(response))),
            entropy_proxy(response),
        ],
        dtype=np.float32,
    )


def transform_examples_with_centroid(
    examples: Sequence[dict[str, Any]],
    semenergy_probe: Any,
    embedding_model: TextEmbeddingModel,
    centroid: np.ndarray,
) -> np.ndarray:
    """Build five HalluGuard router features using an existing FoVer centroid."""

    surface = np.stack(
        [extract_surface_features(example, semenergy_probe) for example in examples], axis=0
    )
    embeddings = _encode_texts(embedding_model, [_query_text(example) for example in examples])
    distances = np.array(
        [cosine_distance(vector, centroid) for vector in embeddings], dtype=np.float32
    )
    return np.concatenate([surface, distances[:, None]], axis=1).astype(np.float32)


def build_halluguard_feature_matrix(
    examples: Sequence[dict[str, Any]],
    semenergy_probe: Any,
    embedding_model: TextEmbeddingModel,
    train_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Build five router features and the FoVer training embedding centroid."""

    embeddings = _encode_texts(embedding_model, [_query_text(example) for example in examples])
    centroid = embeddings[train_indices].mean(axis=0)
    centroid_norm = float(np.linalg.norm(centroid))
    if centroid_norm > 0.0:
        centroid = centroid / centroid_norm
    raw = transform_examples_with_centroid(examples, semenergy_probe, embedding_model, centroid)
    return raw.astype(np.float32), centroid.astype(np.float32)


def route_depths_v3(
    model: LagrangianCascadeMLP,
    X: np.ndarray,
    sem_energy_scores: np.ndarray,
    lambda_value: float,
    *,
    raw_response_lengths: np.ndarray | None = None,
    entropy_scores: np.ndarray | None = None,
    embedding_distances: np.ndarray | None = None,
    entropy_threshold: float = ENTROPY_THRESHOLD,
    distance_threshold: float = EMBEDDING_DISTANCE_THRESHOLD,
) -> np.ndarray:
    """Route with Exp 1131 MLP plus HalluGuard risk escalation guards."""

    depths = model.predict(X, lambda_value=lambda_value)
    hallu_guard_risk = np.zeros(len(depths), dtype=bool)
    if entropy_scores is not None:
        hallu_guard_risk = hallu_guard_risk | (entropy_scores >= entropy_threshold)
    if embedding_distances is not None:
        hallu_guard_risk = hallu_guard_risk | (embedding_distances >= distance_threshold)

    deep_guard = (sem_energy_scores > EASY_ENERGY_THRESHOLD) | hallu_guard_risk
    if raw_response_lengths is not None:
        deep_guard = deep_guard | (raw_response_lengths < SHORT_RESPONSE_WORDS)

    easy_guard = sem_energy_scores <= EASY_ENERGY_THRESHOLD
    if raw_response_lengths is not None:
        easy_guard = easy_guard & (raw_response_lengths >= SHORT_RESPONSE_WORDS)
    easy_guard = easy_guard & ~hallu_guard_risk

    depths = np.where(deep_guard, 5, depths)
    depths = np.where(easy_guard, 1, depths)
    return depths.astype(np.int32)


def train_halluguard_mlp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    labels_train: list[str],
    sem_scores_train: np.ndarray,
    required_depths_train: np.ndarray,
    raw_lengths_train: np.ndarray,
    entropy_scores_train: np.ndarray,
    embedding_distances_train: np.ndarray,
    epochs: int = 20,
    batch_size: int = 64,
    lr: float = 1e-3,
) -> tuple[LagrangianCascadeMLP, float]:
    """Train the Exp 1131 MLP architecture with a five-feature input layer."""

    model = LagrangianCascadeMLP(input_dim=X_train.shape[1], hidden_dim=MLP_HIDDEN_SIZE, seed=1143)
    rng = np.random.default_rng(1143)
    lambda_value = 0.0
    n = len(X_train)
    for _ in range(epochs):
        for start in range(0, n, batch_size):
            idx = rng.permutation(n)[start : start + batch_size]
            xb = X_train[idx]
            yb = y_train[idx]
            logits, cache = model.forward(xb)
            model.step(model.backward(cache, logits, yb), lr=lr)
            pred = route_depths_v3(
                model,
                xb,
                sem_scores_train[idx],
                lambda_value,
                raw_response_lengths=raw_lengths_train[idx],
                entropy_scores=entropy_scores_train[idx],
                embedding_distances=embedding_distances_train[idx],
            )
            tp = batch_tp_rate(
                [labels_train[int(i)] for i in idx], pred, required_depths_train[idx]
            )
            lambda_value = update_dual_lambda(lambda_value, tp, min_tp_constraint=MIN_TP_CONSTRAINT)
    return model, lambda_value


def load_goodfire_exemplars(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text.startswith("["):
        return list(json.loads(text))
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def summarize_goodfire_failure_routing(
    *,
    per_exemplar_results: Sequence[dict[str, Any]],
    exemplar_rows: Sequence[dict[str, Any]],
    predicted_depths: np.ndarray,
    entropy_scores: np.ndarray,
    embedding_distances: np.ndarray,
    entropy_threshold: float = ENTROPY_THRESHOLD,
    distance_threshold: float = EMBEDDING_DISTANCE_THRESHOLD,
) -> dict[str, Any]:
    """Summarize whether HalluGuard features explain ThinkPRM exemplar misses."""

    index_by_id = {str(row.get("id")): idx for idx, row in enumerate(exemplar_rows)}
    miss_indices: list[int] = []
    for result in per_exemplar_results:
        row_idx = index_by_id.get(str(result.get("id")))
        if row_idx is None:
            continue
        tier_results = result.get("tier_results") or {}
        if tier_results.get("tier_0a_thinkprm") is False:
            miss_indices.append(row_idx)

    if not miss_indices:
        return {
            "goodfire_exemplars_scored": int(len(exemplar_rows)),
            "thinkprm_miss_count": 0,
            "thinkprm_miss_k5_route_rate": 0.0,
            "hallu_feature_flag_rate": 0.0,
            "halluguard_features_explain_goodfire_failures": False,
        }

    miss_idx = np.array(miss_indices, dtype=np.int32)
    routed_k5 = predicted_depths[miss_idx] == 5
    feature_flag = (entropy_scores[miss_idx] >= entropy_threshold) | (
        embedding_distances[miss_idx] >= distance_threshold
    )
    k5_rate = float(routed_k5.mean())
    flag_rate = float(feature_flag.mean())
    explains = bool(
        k5_rate >= GOODFIRE_EXPLAIN_RATE_THRESHOLD and flag_rate >= GOODFIRE_EXPLAIN_RATE_THRESHOLD
    )
    return {
        "goodfire_exemplars_scored": int(len(exemplar_rows)),
        "thinkprm_miss_count": int(len(miss_indices)),
        "thinkprm_miss_k5_route_rate": k5_rate,
        "hallu_feature_flag_rate": flag_rate,
        "halluguard_features_explain_goodfire_failures": explains,
    }


def evaluate_goodfire_failures(
    *,
    model: LagrangianCascadeMLP,
    normalizer: FeatureNormalizer,
    semenergy_probe: Any,
    embedding_model: TextEmbeddingModel,
    centroid: np.ndarray,
    goodfire_artifact_path: Path,
    exemplar_corpus_path: Path,
    lambda_value: float,
    entropy_threshold: float = ENTROPY_THRESHOLD,
    distance_threshold: float = EMBEDDING_DISTANCE_THRESHOLD,
) -> dict[str, Any]:
    """Score Goodfire exemplar ThinkPRM misses with the trained v3 router."""

    goodfire_artifact = json.loads(goodfire_artifact_path.read_text(encoding="utf-8"))
    per_exemplar_results = list(goodfire_artifact.get("per_exemplar_results", []))
    rows_by_id = {str(row.get("id")): row for row in load_goodfire_exemplars(exemplar_corpus_path)}

    ordered_results: list[dict[str, Any]] = []
    ordered_rows: list[dict[str, Any]] = []
    for result in per_exemplar_results:
        row = rows_by_id.get(str(result.get("id")))
        if row is None:
            continue
        ordered_results.append(result)
        ordered_rows.append(
            {
                "id": row.get("id"),
                "prompt": row.get("prompt", ""),
                "response": row.get("buggy_response", row.get("response", "")),
                "label": "incorrect",
            }
        )

    raw = transform_examples_with_centroid(ordered_rows, semenergy_probe, embedding_model, centroid)
    X = normalizer.transform(raw)
    depths = route_depths_v3(
        model,
        X,
        raw[:, 0],
        lambda_value,
        raw_response_lengths=raw[:, 1],
        entropy_scores=raw[:, 3],
        embedding_distances=raw[:, 4],
        entropy_threshold=entropy_threshold,
        distance_threshold=distance_threshold,
    )
    return summarize_goodfire_failure_routing(
        per_exemplar_results=ordered_results,
        exemplar_rows=ordered_rows,
        predicted_depths=depths,
        entropy_scores=raw[:, 3],
        embedding_distances=raw[:, 4],
        entropy_threshold=entropy_threshold,
        distance_threshold=distance_threshold,
    )


def _verdict(cost_savings_pct: float, accuracy_delta: float, goodfire_explained: bool) -> str:
    if goodfire_explained:
        return "features_explain_goodfire_failures"
    if cost_savings_pct > 0.0 and accuracy_delta >= 0.0:
        return "routing_improved_with_halluguard_features"
    if cost_savings_pct <= 0.0 or accuracy_delta < -0.05:
        return "routing_degraded"
    return "routing_unchanged"


def build_exp1143_artifact(
    *,
    training_set_size: int,
    holdout_set_size: int,
    mlp_val_accuracy: float,
    lambda_final: float,
    metrics: dict[str, float],
    predicted_depth_distribution: dict[int, int],
    cascade_depth_distribution: dict[int, int],
    goodfire_summary: dict[str, Any],
    duration_s: float,
    embedding_backend: str,
) -> dict[str, Any]:
    """Build the stable Exp 1143 result artifact."""

    accuracy_delta = float(metrics["accuracy_delta"])
    cost_savings_pct = float(metrics["cost_savings_pct"])
    goodfire_explained = bool(goodfire_summary["halluguard_features_explain_goodfire_failures"])
    return {
        "experiment": "exp1143",
        "schema": "v1",
        "title": "HalluGuard cascade router v3 with entropy and embedding distance",
        "run_date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "duration_s": round(float(duration_s), 2),
        "status": "success",
        "mlp_hidden_size": MLP_HIDDEN_SIZE,
        "mlp_hidden_layer_count": 2,
        "mlp_val_accuracy": round(float(mlp_val_accuracy), 4),
        "verifier_score_features_used": list(FEATURE_NAMES),
        "halluguard_features_added": list(HALLUGUARD_FEATURES_ADDED),
        "n_router_features_before": len(BASE_FEATURE_NAMES),
        "n_router_features_after": len(FEATURE_NAMES),
        "training_set_size": int(training_set_size),
        "holdout_set_size": int(holdout_set_size),
        "min_tp_constraint": MIN_TP_CONSTRAINT,
        "lambda_final": round(float(lambda_final), 4),
        "adaptive_tp_rate": round(float(metrics["adaptive_tp_rate"]), 4),
        "fixed_tp_rate": round(float(metrics["fixed_tp_rate"]), 4),
        "accuracy_delta": round(accuracy_delta, 4),
        "fixed_cascade_cost_ms": round(float(metrics["fixed_cascade_cost_ms"]), 4),
        "adaptive_cascade_cost_ms": round(float(metrics["adaptive_cascade_cost_ms"]), 4),
        "cost_savings_pct": round(cost_savings_pct, 2),
        "halluguard_features_explain_goodfire_failures": goodfire_explained,
        "halluguard_routing_feature_measured": True,
        "honest_verdict": _verdict(cost_savings_pct, accuracy_delta, goodfire_explained),
        "embedding_backend": embedding_backend,
        "entropy_threshold": ENTROPY_THRESHOLD,
        "embedding_distance_threshold": EMBEDDING_DISTANCE_THRESHOLD,
        "goodfire_thinkprm_miss_count": int(goodfire_summary.get("thinkprm_miss_count", 0)),
        "goodfire_thinkprm_miss_k5_route_rate": round(
            float(goodfire_summary.get("thinkprm_miss_k5_route_rate", 0.0)), 4
        ),
        "goodfire_hallu_feature_flag_rate": round(
            float(goodfire_summary.get("hallu_feature_flag_rate", 0.0)), 4
        ),
        "predicted_depth_distribution": {
            str(depth): int(count) for depth, count in sorted(predicted_depth_distribution.items())
        },
        "cascade_depth_distribution": {
            str(depth): int(count) for depth, count in sorted(cascade_depth_distribution.items())
        },
        "tier_cumulative_latencies_ms": list(TIER_CUMULATIVE_MS),
        "source": "arXiv:2601.18753 HalluGuard data/reasoning decomposition",
        "note": (
            "Exp1143 extends the Exp1131 Lagrangian router with entropy_proxy "
            "and embedding_distance features to estimate reasoning-driven "
            "instability and data-driven mismatch."
        ),
    }


def _split_indices(total: int, n_train: int, n_holdout: int) -> tuple[np.ndarray, np.ndarray]:
    holdout_count = min(n_holdout, max(1, total - 1))
    train_count = min(n_train, total - holdout_count)
    if train_count <= 0:
        raise ValueError("at least two examples are required for Exp1143")
    holdout_idx = np.arange(holdout_count, dtype=np.int32)
    train_idx = np.arange(holdout_count, holdout_count + train_count, dtype=np.int32)
    return train_idx, holdout_idx


def run_experiment(
    corpus_path: Path,
    results_path: Path,
    goodfire_artifact_path: Path,
    exemplar_corpus_path: Path,
    *,
    n_train: int = 5000,
    n_holdout: int = 500,
    epochs: int = 20,
    seed: int = 1143,
    semenergy_probe: Any | None = None,
    embedding_model: TextEmbeddingModel | None = None,
    embedding_backend: str | None = None,
) -> dict[str, Any]:
    """Run the full Exp 1143 train/holdout/Goodfire analysis."""

    t0 = time.time()
    if semenergy_probe is None:
        from carnot.verify.semenergy_probe import SemEnergyProbe

        semenergy_probe = SemEnergyProbe()
    if embedding_model is None:
        embedding_model, detected_backend = load_embedding_model()
        embedding_backend = embedding_backend or detected_backend
    else:
        embedding_backend = embedding_backend or embedding_model.__class__.__name__

    examples = load_fover_examples(corpus_path)
    rng = np.random.default_rng(seed)
    total_needed = min(len(examples), n_train + n_holdout)
    selected = rng.permutation(len(examples))[:total_needed]
    examples = [examples[int(i)] for i in selected]
    train_idx, holdout_idx = _split_indices(len(examples), n_train, n_holdout)

    raw_features, centroid = build_halluguard_feature_matrix(
        examples,
        semenergy_probe,
        embedding_model,
        train_idx,
    )
    labels = [str(example.get("label", "")) for example in examples]
    required_depths = infer_required_depths(raw_features[:, : len(BASE_FEATURE_NAMES)], labels)

    normalizer = FeatureNormalizer()
    normalizer.fit(raw_features[train_idx])
    X_train = normalizer.transform(raw_features[train_idx])
    X_holdout = normalizer.transform(raw_features[holdout_idx])
    y_train = required_depths[train_idx] - 1
    y_holdout = required_depths[holdout_idx] - 1

    model, lambda_final = train_halluguard_mlp(
        X_train,
        y_train,
        [labels[int(i)] for i in train_idx],
        raw_features[train_idx, 0],
        required_depths[train_idx],
        raw_features[train_idx, 1],
        raw_features[train_idx, 3],
        raw_features[train_idx, 4],
        epochs=epochs,
    )
    predicted_depths = route_depths_v3(
        model,
        X_holdout,
        raw_features[holdout_idx, 0],
        lambda_final,
        raw_response_lengths=raw_features[holdout_idx, 1],
        entropy_scores=raw_features[holdout_idx, 3],
        embedding_distances=raw_features[holdout_idx, 4],
    )
    mlp_val_accuracy = float((model.predict(X_holdout, lambda_final) == (y_holdout + 1)).mean())
    holdout_labels = [labels[int(i)] for i in holdout_idx]
    metrics = evaluate_depth_predictions(
        holdout_labels,
        predicted_depths,
        required_depths[holdout_idx],
    )
    goodfire_summary = evaluate_goodfire_failures(
        model=model,
        normalizer=normalizer,
        semenergy_probe=semenergy_probe,
        embedding_model=embedding_model,
        centroid=centroid,
        goodfire_artifact_path=goodfire_artifact_path,
        exemplar_corpus_path=exemplar_corpus_path,
        lambda_value=lambda_final,
    )
    artifact = build_exp1143_artifact(
        training_set_size=len(train_idx),
        holdout_set_size=len(holdout_idx),
        mlp_val_accuracy=mlp_val_accuracy,
        lambda_final=lambda_final,
        metrics=metrics,
        predicted_depth_distribution={
            depth: int((predicted_depths == depth).sum()) for depth in range(1, 6)
        },
        cascade_depth_distribution={
            depth: int((required_depths[holdout_idx] == depth).sum()) for depth in range(1, 6)
        },
        goodfire_summary=goodfire_summary,
        duration_s=time.time() - t0,
        embedding_backend=embedding_backend,
    )
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    return artifact
