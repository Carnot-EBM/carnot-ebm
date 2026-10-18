"""KAN-CL learner for n=256 split-task constraint learning.

Spec refs: REQ-KAN-1826, SCENARIO-KAN-1826, SCENARIO-KAN-1826-N256.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np

from carnot.models.kan import KAN


@dataclass(frozen=True)
class SplitTask:
    """Synthetic split-task constraint corpus used by the KAN-CL smoke benchmark."""

    task_id: str
    domain: str
    X: np.ndarray
    y: np.ndarray


class KanClLearner:
    """Importance-regularized KAN classifier for sequential constraint tasks.

    The learner uses one spline coefficient per constraint dimension.  After each
    task, it stores the activation frequency of every knot.  On later tasks, the
    weight update adds an importance-weighted L2 gradient that pulls protected
    coefficients back toward their post-task anchor values.
    """

    def __init__(
        self,
        n_params: int = 256,
        learning_rate: float = 0.08,
        regularization_strength: float = 8.0,
        epochs: int = 180,
        seed: int = 42,
    ) -> None:
        if n_params <= 0:
            raise ValueError("n_params must be positive")
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if regularization_strength < 0:
            raise ValueError("regularization_strength must be non-negative")
        if epochs <= 0:
            raise ValueError("epochs must be positive")

        self.model = KAN(n_params=n_params, seed=seed)
        self.n_params = n_params
        self.learning_rate = float(learning_rate)
        self.regularization_strength = float(regularization_strength)
        self.epochs = int(epochs)
        self.seed = int(seed)
        self.task_importances: dict[str, np.ndarray] = {}
        self.cumulative_importance = np.zeros(n_params, dtype=np.float64)
        self.anchor_coefficients = self.model.coefficients.copy()
        self.fit_history: list[dict[str, Any]] = []

    def compute_importance(self, X: np.ndarray) -> np.ndarray:
        """Compute per-knot importance as activation frequency over a task batch."""
        return self.model.activation_frequency(X)

    def fit(self, X: np.ndarray, y: np.ndarray, task_id: str) -> "KanClLearner":
        """Fit one task and store its per-knot importance."""
        basis, labels = self._validate_xy(X, y)
        n_examples = labels.shape[0]

        for _ in range(self.epochs):
            logits = basis @ self.model.coefficients
            probs = _sigmoid(logits)
            data_grad = basis.T @ (probs - labels) / n_examples
            reg_grad = (
                2.0
                * self.regularization_strength
                * self.cumulative_importance
                * (self.model.coefficients - self.anchor_coefficients)
            )
            self.model.coefficients -= self.learning_rate * (data_grad + reg_grad)

        importance = self.compute_importance(basis)
        self.task_importances[str(task_id)] = importance
        self.cumulative_importance = np.maximum(self.cumulative_importance, importance)
        self.anchor_coefficients = self.model.coefficients.copy()
        self.fit_history.append(
            {
                "task_id": str(task_id),
                "active_knots": int(np.count_nonzero(importance)),
                "mean_importance": float(np.mean(importance)),
            }
        )
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return positive-class probabilities for a batch."""
        return _sigmoid(self.model.logits(X))

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return binary constraint-satisfaction predictions."""
        return (self.predict_proba(X) >= 0.5).astype(np.int64)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Return classification accuracy."""
        labels = np.asarray(y, dtype=np.int64)
        return float(np.mean(self.predict(X) == labels))

    def _validate_xy(self, X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        basis = self.model.basis(X)
        labels = np.asarray(y, dtype=np.float64)
        if labels.ndim != 1:
            raise ValueError("y must be a 1D array")
        if basis.shape[0] != labels.shape[0]:
            raise ValueError("X and y must contain the same number of examples")
        unique = set(np.unique(labels).astype(int).tolist())
        if not unique.issubset({0, 1}):
            raise ValueError("y must contain binary labels 0/1")
        return basis, labels


def make_split_task_constraint_tasks(
    n_params: int = 256,
    examples_per_task: int = 50,
    seed: int = 42,
) -> list[SplitTask]:
    """Create deterministic arithmetic/code/logic tasks over a 256-knot space."""
    if n_params != 256:
        raise ValueError("the split-task benchmark is defined for n_params=256")
    if examples_per_task <= 0:
        raise ValueError("examples_per_task must be positive")

    rng = np.random.default_rng(seed)
    base_a = rng.choice(np.array([-1.0, 1.0]), size=64)
    base_b = rng.choice(np.array([-1.0, 1.0]), size=64)
    domains = [
        ("task_1", "arithmetic", slice(0, 128), [(slice(64, 128), base_a, 1.0)]),
        (
            "task_2",
            "code",
            slice(64, 192),
            [(slice(64, 128), -base_a, 1.0), (slice(128, 192), base_b, 0.45)],
        ),
        (
            "task_3",
            "logic",
            slice(128, 256),
            [(slice(128, 192), -base_b, 1.0), (slice(192, 256), base_a, 0.45)],
        ),
    ]

    tasks: list[SplitTask] = []
    for task_id, domain, active_slice, signal_parts in domains:
        X = np.zeros((examples_per_task, n_params), dtype=np.float64)
        width = active_slice.stop - active_slice.start
        X[:, active_slice] = rng.choice(np.array([-1.0, 1.0]), size=(examples_per_task, width))

        margin = np.zeros(examples_per_task, dtype=np.float64)
        for part_slice, signs, scale in signal_parts:
            margin += scale * (X[:, part_slice] @ signs)
        y = (margin >= 0.0).astype(np.int64)
        tasks.append(SplitTask(task_id=task_id, domain=domain, X=X, y=y))

    return tasks


def build_split_task_benchmark_payload(seed: int = 42) -> dict[str, Any]:
    """Run the three-task KAN-CL benchmark and return the artifact payload."""
    tasks = make_split_task_constraint_tasks(seed=seed)
    baseline = KanClLearner(
        n_params=256,
        learning_rate=0.08,
        regularization_strength=0.0,
        epochs=180,
        seed=seed,
    )
    kancl = KanClLearner(
        n_params=256,
        learning_rate=0.08,
        regularization_strength=8.0,
        epochs=180,
        seed=seed,
    )

    baseline_trace = _run_sequential_trace(baseline, tasks)
    kancl_trace = _run_sequential_trace(kancl, tasks)
    forgetting_without = _mean_forgetting(baseline_trace)
    forgetting_with_kancl = _mean_forgetting(kancl_trace)
    if forgetting_without <= 1e-12:
        forgetting_reduction_pct = 0.0
    else:
        forgetting_reduction_pct = 100.0 * (1.0 - forgetting_with_kancl / forgetting_without)
    forgetting_reduction_pct = float(max(0.0, forgetting_reduction_pct))

    return {
        "honest_verdict": "kancl_n256_validated"
        if forgetting_reduction_pct >= 50.0
        else "kancl_n256_not_validated",
        "kancl_n256_validated": forgetting_reduction_pct >= 50.0,
        "forgetting_reduction_pct": forgetting_reduction_pct,
        "forgetting_with_kancl": forgetting_with_kancl,
        "forgetting_without": forgetting_without,
        "n_tasks": 3,
        "n_params": 256,
        "random_seed": seed,
        "task_domains": [task.domain for task in tasks],
        "examples_per_task": 50,
        "baseline_trace": baseline_trace,
        "kancl_trace": kancl_trace,
        "spec_traces": ["REQ-KAN-1826", "SCENARIO-KAN-1826-N256"],
    }


def write_split_task_benchmark_artifact(
    path: str | Path = "results/experiment_2356_kancl_n256.json",
    seed: int = 42,
) -> dict[str, Any]:
    """Write the KAN-CL n=256 benchmark artifact and return its payload."""
    payload = build_split_task_benchmark_payload(seed=seed)
    artifact_path = Path(path)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def _run_sequential_trace(learner: KanClLearner, tasks: list[SplitTask]) -> list[dict[str, Any]]:
    trace: list[dict[str, Any]] = []
    for train_index, task in enumerate(tasks):
        learner.fit(task.X, task.y, task_id=task.task_id)
        accuracies = {
            seen.task_id: learner.score(seen.X, seen.y) for seen in tasks[: train_index + 1]
        }
        trace.append(
            {
                "trained_task": task.task_id,
                "domain": task.domain,
                "seen_task_accuracy": accuracies,
            }
        )
    return trace


def _mean_forgetting(trace: list[dict[str, Any]]) -> float:
    task_ids = list(trace[-1]["seen_task_accuracy"].keys())
    final_acc = trace[-1]["seen_task_accuracy"]
    forgetting = []
    for task_id in task_ids[:-1]:
        best_seen = max(step["seen_task_accuracy"].get(task_id, 0.0) for step in trace)
        forgetting.append(max(0.0, best_seen - final_acc[task_id]))
    return float(np.mean(forgetting)) if forgetting else 0.0


def _sigmoid(logits: np.ndarray) -> np.ndarray:
    clipped = np.clip(logits, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-clipped))
