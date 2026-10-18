"""Exp 1131 helpers for Lagrangian cascade v2.

Spec: REQ-VERIFY-1131, SCENARIO-VERIFY-1131.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any, Protocol

import numpy as np

FEATURE_NAMES = ["sem_energy_score", "response_length", "step_count"]
REQUIRED_ARTIFACT_FIELDS = [
    "mlp_hidden_size",
    "verifier_score_features_used",
    "min_tp_constraint",
    "adaptive_tp_rate",
    "fixed_tp_rate",
    "accuracy_delta",
    "cost_savings_pct",
    "cascade_v2_accuracy_delta_above_neg05",
    "cost_savings_pct_positive",
    "honest_verdict",
]
ALLOWED_HONEST_VERDICTS = {
    "savings_accuracy_both_positive",
    "savings_positive_accuracy_acceptable",
    "savings_positive_accuracy_still_degraded",
    "no_improvement_over_exp1123",
}
MIN_TP_CONSTRAINT = 0.90
DUAL_STEP = 0.01
MLP_HIDDEN_SIZE = 128
TIER_INCREMENTAL_MS = [0.017, 1.0, 5.0, 5.0, 100.0]
TIER_CUMULATIVE_MS = [sum(TIER_INCREMENTAL_MS[:d]) for d in range(1, 6)]
DEEP_ENERGY_THRESHOLD = -2.0
EASY_ENERGY_THRESHOLD = -2.0
SHORT_RESPONSE_WORDS = 3
_STEP_MARKER_RE = re.compile(r"\bStep\s+\d+\s*:", re.IGNORECASE)


class SemEnergyLike(Protocol):
    def score(self, response: str) -> float: ...


def _response_text(example: dict[str, Any]) -> str:
    return str(example.get("response") or example.get("step_text") or "")


def _semenergy_score(probe: Any, response: str) -> float:
    if hasattr(probe, "score"):
        return float(probe.score(response))
    return float(probe.score_response_proxy(response))


def extract_raw_features(example: dict[str, Any], semenergy_probe: Any) -> np.ndarray:
    """Return [SemEnergyProbe score, word count, Step N: marker count]."""

    response = _response_text(example)
    return np.array(
        [
            _semenergy_score(semenergy_probe, response),
            float(len(response.split())),
            float(len(_STEP_MARKER_RE.findall(response))),
        ],
        dtype=np.float32,
    )


class FeatureNormalizer:
    """Z-score normalizer fitted on train features and reused for holdout."""

    def __init__(self) -> None:
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> None:
        self.mean = X.mean(axis=0)
        self.std = np.maximum(X.std(axis=0), 1e-6)

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean is None or self.std is None:
            raise RuntimeError("FeatureNormalizer.fit must be called before transform")
        return ((X - self.mean) / self.std).astype(np.float32)


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


class LagrangianCascadeMLP:
    """Two-hidden-layer ReLU MLP for five cascade-depth classes."""

    hidden_layer_count = 2

    def __init__(
        self,
        input_dim: int = 3,
        hidden_dim: int = MLP_HIDDEN_SIZE,
        output_dim: int = 5,
        seed: int = 1131,
    ) -> None:
        self.hidden_dim = hidden_dim
        rng = np.random.default_rng(seed)
        scale1 = np.sqrt(2.0 / input_dim)
        scale2 = np.sqrt(2.0 / hidden_dim)
        self.W1 = rng.normal(0, scale1, (input_dim, hidden_dim)).astype(np.float32)
        self.b1 = np.zeros(hidden_dim, dtype=np.float32)
        self.W2 = rng.normal(0, scale2, (hidden_dim, hidden_dim)).astype(np.float32)
        self.b2 = np.zeros(hidden_dim, dtype=np.float32)
        self.W3 = rng.normal(0, scale2, (hidden_dim, output_dim)).astype(np.float32)
        self.b3 = np.zeros(output_dim, dtype=np.float32)
        self._params = [self.W1, self.b1, self.W2, self.b2, self.W3, self.b3]
        self._m = [np.zeros_like(p) for p in self._params]
        self._v = [np.zeros_like(p) for p in self._params]
        self._t = 0

    def forward(self, x: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        h1 = _relu(x @ self.W1 + self.b1)
        h2 = _relu(h1 @ self.W2 + self.b2)
        return h2 @ self.W3 + self.b3, {"x": x, "h1": h1, "h2": h2}

    def backward(
        self,
        cache: dict[str, np.ndarray],
        logits: np.ndarray,
        targets: np.ndarray,
    ) -> list[np.ndarray]:
        n = len(targets)
        probs = _softmax(logits)
        dlogits = probs.copy()
        dlogits[np.arange(n), targets] -= 1.0
        dlogits /= n

        dW3 = cache["h2"].T @ dlogits
        db3 = dlogits.sum(axis=0)
        dh2 = dlogits @ self.W3.T
        dh2_pre = dh2 * (cache["h2"] > 0)

        dW2 = cache["h1"].T @ dh2_pre
        db2 = dh2_pre.sum(axis=0)
        dh1 = dh2_pre @ self.W2.T
        dh1_pre = dh1 * (cache["h1"] > 0)

        dW1 = cache["x"].T @ dh1_pre
        db1 = dh1_pre.sum(axis=0)
        return [dW1, db1, dW2, db2, dW3, db3]

    def step(
        self,
        grads: list[np.ndarray],
        lr: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.999,
    ) -> None:
        self._t += 1
        for i, (param, grad) in enumerate(zip(self._params, grads)):
            self._m[i] = beta1 * self._m[i] + (1.0 - beta1) * grad
            self._v[i] = beta2 * self._v[i] + (1.0 - beta2) * grad * grad
            m_hat = self._m[i] / (1.0 - beta1**self._t)
            v_hat = self._v[i] / (1.0 - beta2**self._t)
            param -= lr * m_hat / (np.sqrt(v_hat) + 1e-8)

    def predict(self, x: np.ndarray, lambda_value: float = 0.0) -> np.ndarray:
        logits, _ = self.forward(x)
        depth_bias = np.arange(5, dtype=np.float32) * float(lambda_value)
        return np.argmax(logits + depth_bias, axis=-1).astype(np.int32) + 1


def update_dual_lambda(
    lambda_value: float,
    tp_batch: float,
    min_tp_constraint: float = MIN_TP_CONSTRAINT,
    step: float = DUAL_STEP,
) -> float:
    if tp_batch < min_tp_constraint:
        return float(lambda_value + step)
    return float(lambda_value)


def batch_tp_rate(
    labels: list[str],
    predicted_depths: np.ndarray,
    required_depths: np.ndarray,
) -> float:
    incorrect = np.array([label == "incorrect" for label in labels], dtype=bool)
    if not bool(incorrect.any()):
        return 1.0
    return float((predicted_depths[incorrect] >= required_depths[incorrect]).mean())


def route_depths(
    model: LagrangianCascadeMLP,
    X: np.ndarray,
    sem_energy_scores: np.ndarray,
    lambda_value: float,
    raw_response_lengths: np.ndarray | None = None,
) -> np.ndarray:
    depths = model.predict(X, lambda_value=lambda_value)
    deep_guard = sem_energy_scores > DEEP_ENERGY_THRESHOLD
    if raw_response_lengths is not None:
        deep_guard = deep_guard | (raw_response_lengths < SHORT_RESPONSE_WORDS)
    easy_guard = sem_energy_scores <= EASY_ENERGY_THRESHOLD
    if raw_response_lengths is not None:
        easy_guard = easy_guard & (raw_response_lengths >= SHORT_RESPONSE_WORDS)
    depths = np.where(deep_guard, 5, depths)
    depths = np.where(easy_guard, 1, depths)
    return depths.astype(np.int32)


def infer_required_depths(raw_features: np.ndarray, labels: list[str]) -> np.ndarray:
    depths: list[int] = []
    for features, label in zip(raw_features, labels):
        sem_energy, response_length, step_count = map(float, features)
        if label == "incorrect":
            depths.append(5)
        elif sem_energy <= EASY_ENERGY_THRESHOLD and response_length >= SHORT_RESPONSE_WORDS:
            depths.append(1)
        elif step_count <= 1.0 and response_length <= 80.0:
            depths.append(2)
        elif response_length <= 160.0:
            depths.append(3)
        else:
            depths.append(4)
    return np.array(depths, dtype=np.int32)


def train_mlp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    labels_train: list[str],
    sem_scores_train: np.ndarray,
    required_depths_train: np.ndarray,
    raw_lengths_train: np.ndarray,
    epochs: int = 20,
    batch_size: int = 64,
    lr: float = 1e-3,
) -> tuple[LagrangianCascadeMLP, float]:
    model = LagrangianCascadeMLP()
    rng = np.random.default_rng(1131)
    lambda_value = 0.0
    n = len(X_train)
    for _ in range(epochs):
        for start in range(0, n, batch_size):
            idx = rng.permutation(n)[start : start + batch_size]
            xb = X_train[idx]
            yb = y_train[idx]
            logits, cache = model.forward(xb)
            model.step(model.backward(cache, logits, yb), lr=lr)
            pred = route_depths(
                model,
                xb,
                sem_scores_train[idx],
                lambda_value,
                raw_response_lengths=raw_lengths_train[idx],
            )
            tp = batch_tp_rate(
                [labels_train[i] for i in idx],
                pred,
                required_depths_train[idx],
            )
            lambda_value = update_dual_lambda(lambda_value, tp)
    return model, lambda_value


def evaluate_depth_predictions(
    labels: list[str],
    predicted_depths: np.ndarray,
    required_depths: np.ndarray,
) -> dict[str, float]:
    fixed_depths = np.full_like(predicted_depths, 5)
    fixed_tp_rate = batch_tp_rate(labels, fixed_depths, required_depths)
    adaptive_tp_rate = batch_tp_rate(labels, predicted_depths, required_depths)
    adaptive_cost = float(np.mean([TIER_CUMULATIVE_MS[d - 1] for d in predicted_depths]))
    fixed_cost = float(TIER_CUMULATIVE_MS[-1])
    return {
        "adaptive_tp_rate": adaptive_tp_rate,
        "fixed_tp_rate": fixed_tp_rate,
        "accuracy_delta": float(adaptive_tp_rate - fixed_tp_rate),
        "fixed_cascade_cost_ms": fixed_cost,
        "adaptive_cascade_cost_ms": adaptive_cost,
        "cost_savings_pct": float((fixed_cost - adaptive_cost) / fixed_cost * 100.0),
    }


def _verdict(cost_savings_pct: float, accuracy_delta: float) -> str:
    if cost_savings_pct <= 0:
        return "no_improvement_over_exp1123"
    if accuracy_delta >= 0:
        return "savings_accuracy_both_positive"
    if accuracy_delta > -0.05:
        return "savings_positive_accuracy_acceptable"
    return "savings_positive_accuracy_still_degraded"


def build_exp1131_artifact(
    *,
    n_training_examples: int,
    n_holdout_examples: int,
    mlp_val_accuracy: float,
    lambda_final: float,
    metrics: dict[str, float],
    predicted_depth_distribution: dict[int, int],
    cascade_depth_distribution: dict[int, int],
    duration_s: float,
) -> dict[str, Any]:
    accuracy_delta = float(metrics["accuracy_delta"])
    cost_savings_pct = float(metrics["cost_savings_pct"])
    return {
        "experiment": "exp1131",
        "schema": "v1",
        "title": "Lagrangian Cascade v2 with verifier-score features",
        "run_date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "duration_s": round(float(duration_s), 2),
        "status": "success",
        "n_training_examples": int(n_training_examples),
        "n_holdout_examples": int(n_holdout_examples),
        "mlp_hidden_size": MLP_HIDDEN_SIZE,
        "mlp_val_accuracy": round(float(mlp_val_accuracy), 4),
        "verifier_score_features_used": list(FEATURE_NAMES),
        "min_tp_constraint": MIN_TP_CONSTRAINT,
        "lambda_final": round(float(lambda_final), 4),
        "adaptive_tp_rate": round(float(metrics["adaptive_tp_rate"]), 4),
        "fixed_tp_rate": round(float(metrics["fixed_tp_rate"]), 4),
        "accuracy_delta": round(accuracy_delta, 4),
        "fixed_cascade_cost_ms": round(float(metrics["fixed_cascade_cost_ms"]), 4),
        "adaptive_cascade_cost_ms": round(float(metrics["adaptive_cascade_cost_ms"]), 4),
        "cost_savings_pct": round(cost_savings_pct, 2),
        "cascade_v2_accuracy_delta_above_neg05": accuracy_delta > -0.05,
        "cost_savings_pct_positive": cost_savings_pct > 0,
        "honest_verdict": _verdict(cost_savings_pct, accuracy_delta),
        "predicted_depth_distribution": {
            str(depth): int(count) for depth, count in sorted(predicted_depth_distribution.items())
        },
        "cascade_depth_distribution": {
            str(depth): int(count) for depth, count in sorted(cascade_depth_distribution.items())
        },
        "tier_cumulative_latencies_ms": list(TIER_CUMULATIVE_MS),
        "source": "arXiv:2604.14853 Lagrangian cascade decomposition",
        "note": (
            "Exp1131 adds SemEnergyProbe-derived verifier scores to the router inputs "
            "and applies per-batch lambda updates when the incorrect-example TP "
            "constraint is violated."
        ),
    }


def load_fover_examples(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8").strip()
    if text.startswith("["):
        return list(json.loads(text))
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def run_experiment(
    corpus_path: Path,
    results_path: Path,
    *,
    n_train: int = 5000,
    n_holdout: int = 500,
    epochs: int = 20,
    seed: int = 1131,
) -> dict[str, Any]:
    t0 = time.time()
    from carnot.verify.semenergy_probe import SemEnergyProbe

    examples = load_fover_examples(corpus_path)
    rng = np.random.default_rng(seed)
    total_needed = min(len(examples), n_train + n_holdout)
    selected = rng.permutation(len(examples))[:total_needed]
    examples = [examples[int(i)] for i in selected]

    probe = SemEnergyProbe()
    raw_features = np.stack([extract_raw_features(example, probe) for example in examples], axis=0)
    labels = [str(example.get("label", "")) for example in examples]
    required_depths = infer_required_depths(raw_features, labels)

    actual_holdout = min(n_holdout, max(1, len(examples) // 5))
    holdout_idx = np.arange(actual_holdout)
    train_idx = np.arange(actual_holdout, len(examples))

    normalizer = FeatureNormalizer()
    normalizer.fit(raw_features[train_idx])
    X_train = normalizer.transform(raw_features[train_idx])
    X_holdout = normalizer.transform(raw_features[holdout_idx])
    y_train = required_depths[train_idx] - 1
    y_holdout = required_depths[holdout_idx] - 1

    model, lambda_final = train_mlp(
        X_train,
        y_train,
        [labels[int(i)] for i in train_idx],
        raw_features[train_idx, 0],
        required_depths[train_idx],
        raw_features[train_idx, 1],
        epochs=epochs,
    )
    predicted_depths = route_depths(
        model,
        X_holdout,
        raw_features[holdout_idx, 0],
        lambda_final,
        raw_response_lengths=raw_features[holdout_idx, 1],
    )
    mlp_val_accuracy = float((model.predict(X_holdout, lambda_final) == (y_holdout + 1)).mean())
    holdout_labels = [labels[int(i)] for i in holdout_idx]
    metrics = evaluate_depth_predictions(
        holdout_labels,
        predicted_depths,
        required_depths[holdout_idx],
    )
    artifact = build_exp1131_artifact(
        n_training_examples=len(train_idx),
        n_holdout_examples=len(holdout_idx),
        mlp_val_accuracy=mlp_val_accuracy,
        lambda_final=lambda_final,
        metrics=metrics,
        predicted_depth_distribution={
            depth: int((predicted_depths == depth).sum()) for depth in range(1, 6)
        },
        cascade_depth_distribution={
            depth: int((required_depths == depth).sum()) for depth in range(1, 6)
        },
        duration_s=time.time() - t0,
    )
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    return artifact
