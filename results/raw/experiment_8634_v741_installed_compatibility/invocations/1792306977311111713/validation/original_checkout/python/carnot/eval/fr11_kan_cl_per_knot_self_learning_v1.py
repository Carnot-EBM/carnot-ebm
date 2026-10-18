"""Exp 2933 KAN/KAC-inspired per-knot structural-memory self-learning probe.

This is a bounded local-training simulation, not a claim about closed-model
retraining. The probe uses exact synthetic constraint verifiers to update local
RBF-center importance, then compares that structural memory against a no-update
baseline and a replay-scheduler-only baseline.

Spec: REQ-LEARN-2933,
      SCENARIO-LEARN-2933,
      SCENARIO-LEARN-2933-GUARD.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_FILENAME = "experiment_2933_kan_cl_per_knot_self_learning_v1.json"
RUN_DATE = "20260523"
RANDOM_SEED = 2933
INFERENCE_SUBSTRATE = "local_training_simulation"
FORGETTING_THRESHOLD = 0.05

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "kan_cl_self_learning_ready",
    "continuous_self_learning_targeted",
    "random_seed",
    "dataset_manifest",
    "baselines",
    "kan_update_config",
    "utility_delta_vs_replay_only",
    "energy_proxy_delta",
    "forgetting_rate",
    "forgetting_threshold",
    "non_forgetting_passed",
    "updated_knot_or_rbf_count",
    "tests_run",
    "inference_substrate",
    "duration_s",
    "run_date",
)


@dataclass(frozen=True)
class ConstraintRule:
    """Exact verifier rule for one synthetic constraint family."""

    constraint_id: str
    positive_centers: tuple[int, ...]
    negative_centers: tuple[int, ...]

    @property
    def center_indices(self) -> tuple[int, ...]:
        return self.positive_centers + self.negative_centers


@dataclass(frozen=True)
class ConstraintExample:
    """One stream row with a deterministic exact-verifier label."""

    example_id: str
    constraint_id: str
    split: str
    features: np.ndarray
    label: int


@dataclass(frozen=True)
class ConstraintStream:
    """Train/holdout splits plus the fixed RBF centers used by every policy."""

    random_seed: int
    centers: np.ndarray
    rules: tuple[ConstraintRule, ...]
    train_by_constraint: dict[str, list[ConstraintExample]]
    holdout_by_constraint: dict[str, list[ConstraintExample]]

    @property
    def rule_by_id(self) -> dict[str, ConstraintRule]:
        return {rule.constraint_id: rule for rule in self.rules}

    def all_rows(self) -> list[ConstraintExample]:
        rows: list[ConstraintExample] = []
        for rule in self.rules:
            rows.extend(self.train_by_constraint[rule.constraint_id])
            rows.extend(self.holdout_by_constraint[rule.constraint_id])
        return rows

    def all_holdout(self) -> list[ConstraintExample]:
        rows: list[ConstraintExample] = []
        for rule in self.rules:
            rows.extend(self.holdout_by_constraint[rule.constraint_id])
        return rows

    def manifest(self) -> dict[str, Any]:
        return {
            "random_seed": self.random_seed,
            "constraint_count": len(self.rules),
            "constraint_ids": [rule.constraint_id for rule in self.rules],
            "train_example_count": sum(len(rows) for rows in self.train_by_constraint.values()),
            "holdout_example_count": sum(len(rows) for rows in self.holdout_by_constraint.values()),
            "rbf_center_count": int(self.centers.shape[0]),
            "feature_dim": int(self.centers.shape[1]),
            "train_per_constraint": {
                key: len(value) for key, value in self.train_by_constraint.items()
            },
            "holdout_per_constraint": {
                key: len(value) for key, value in self.holdout_by_constraint.items()
            },
            "dataset_checksum": _dataset_checksum(self.all_rows()),
        }


@dataclass(frozen=True)
class ExperimentConfig:
    """Runtime paths, clocks, and fixed thresholds for Exp 2933."""

    repo_root: Path = REPO_ROOT
    results_dir: Path | None = None
    random_seed: int = RANDOM_SEED
    run_date: str = RUN_DATE
    forgetting_threshold: float = FORGETTING_THRESHOLD
    started_at: float | None = None
    clock: Callable[[], float] = time.time

    def output_dir(self) -> Path:
        return self.results_dir if self.results_dir is not None else self.repo_root / "results"

    def output_path(self) -> Path:
        return self.output_dir() / OUTPUT_FILENAME

    def start_time(self) -> float:
        return self.clock() if self.started_at is None else self.started_at


class ReplaySchedulerOnly:
    """External replay-priority table with no feature-local model structure."""

    def __init__(self) -> None:
        self.priority_by_constraint: dict[str, float] = {}

    def predict_proba(self, row: ConstraintExample) -> float:
        return self.priority_by_constraint.get(row.constraint_id, 0.5)

    def update(
        self, rows: Sequence[ConstraintExample], rule_by_id: Mapping[str, ConstraintRule]
    ) -> None:
        grouped: dict[str, list[int]] = {}
        for row in rows:
            grouped.setdefault(row.constraint_id, []).append(
                exact_verifier(row, rule_by_id[row.constraint_id])
            )
        for constraint_id, labels in grouped.items():
            self.priority_by_constraint[constraint_id] = _round_float(sum(labels) / len(labels))


class RBFImportanceMemory:
    """KAN/KAC-style local structural memory over fixed RBF centers.

    Each RBF center behaves like a local spline knot: examples only update nearby
    centers, so later constraints can learn new regions without rewriting earlier
    regions. `importance` records activation mass; `signed_evidence` records
    verifier-approved positive or negative evidence.
    """

    def __init__(
        self,
        *,
        centers: np.ndarray,
        sigma: float = 0.09,
        learning_rate: float = 1.0,
        active_threshold: float = 0.35,
    ) -> None:
        self.centers = np.asarray(centers, dtype=np.float64)
        self.sigma = float(sigma)
        self.learning_rate = float(learning_rate)
        self.active_threshold = float(active_threshold)
        self.importance = np.zeros(self.centers.shape[0], dtype=np.float64)
        self.signed_evidence = np.zeros(self.centers.shape[0], dtype=np.float64)

    def _activations(self, features: np.ndarray) -> np.ndarray:
        deltas = self.centers - np.asarray(features, dtype=np.float64).reshape(1, -1)
        squared_distances = np.sum(deltas * deltas, axis=1)
        return np.exp(-squared_distances / (2.0 * self.sigma * self.sigma))

    def predict_proba(self, features: np.ndarray) -> float:
        activations = self._activations(features)
        logit = float(np.dot(self.signed_evidence, activations))
        return _round_float(_sigmoid(logit))

    def update(
        self,
        rows: Sequence[ConstraintExample],
        rule_by_id: Mapping[str, ConstraintRule],
    ) -> None:
        for row in rows:
            activations = self._activations(row.features)
            active = activations >= self.active_threshold
            target = 1.0 if exact_verifier(row, rule_by_id[row.constraint_id]) == 1 else -1.0
            local_step = self.learning_rate * activations * active
            self.importance += np.abs(local_step)
            self.signed_evidence += target * local_step

    def updated_count(self) -> int:
        return int(np.count_nonzero(self.importance > 0.0))


def build_constraint_stream(random_seed: int = RANDOM_SEED) -> ConstraintStream:
    """Build the deterministic seed-2933 constraint stream used by the probe."""

    centers = _rbf_centers()
    rules = (
        ConstraintRule("arithmetic_bounds", (0, 1), (2, 3)),
        ConstraintRule("code_shape", (4, 5), (6, 7)),
        ConstraintRule("logic_coherence", (8, 9), (10, 11)),
    )
    rng = np.random.default_rng(random_seed)
    train_by_constraint = {
        rule.constraint_id: _make_examples(rule, centers, "train", 6, rng) for rule in rules
    }
    holdout_by_constraint = {
        rule.constraint_id: _make_examples(rule, centers, "holdout", 4, rng) for rule in rules
    }
    return ConstraintStream(
        random_seed=random_seed,
        centers=centers,
        rules=rules,
        train_by_constraint=train_by_constraint,
        holdout_by_constraint=holdout_by_constraint,
    )


def exact_verifier(row: ConstraintExample, rule: ConstraintRule) -> int:
    """Return the exact verifier label by nearest active RBF center for the rule."""

    centers = _rbf_centers()[list(rule.center_indices)]
    nearest_local = int(np.argmin(np.sum((centers - row.features.reshape(1, -1)) ** 2, axis=1)))
    nearest_center = rule.center_indices[nearest_local]
    return 1 if nearest_center in rule.positive_centers else 0


def run_experiment(
    config: ExperimentConfig | None = None,
    *,
    write: bool = True,
    tests_run: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Run the bounded Exp 2933 comparison and optionally write its artifact."""

    active_config = config or ExperimentConfig()
    started_at = active_config.start_time()
    stream = build_constraint_stream(active_config.random_seed)
    no_update_metrics = _run_no_update_baseline(stream)
    replay_metrics = _run_replay_scheduler_baseline(stream)
    kan_metrics, forgetting_rate, updated_count = _run_kan_importance_update(stream)

    utility_delta = _round_float(
        kan_metrics["final_holdout_utility"] - replay_metrics["final_holdout_utility"]
    )
    energy_delta = _round_float(
        replay_metrics["final_energy_proxy"] - kan_metrics["final_energy_proxy"]
    )
    artifact = {
        "honest_verdict": "complete: provisional",
        "kan_cl_self_learning_ready": False,
        "continuous_self_learning_targeted": True,
        "random_seed": active_config.random_seed,
        "dataset_manifest": stream.manifest(),
        "baselines": {
            "no_update": no_update_metrics,
            "replay_scheduler_only": replay_metrics,
            "kan_rbf_importance_update": kan_metrics,
        },
        "kan_update_config": {
            "memory_type": "rbf_per_center_importance",
            "rbf_center_count": int(stream.centers.shape[0]),
            "sigma": 0.09,
            "learning_rate": 1.0,
            "active_threshold": 0.35,
            "exact_verifier_rewards": True,
        },
        "utility_delta_vs_replay_only": utility_delta,
        "energy_proxy_delta": energy_delta,
        "forgetting_rate": _round_float(forgetting_rate),
        "forgetting_threshold": active_config.forgetting_threshold,
        "non_forgetting_passed": False,
        "updated_knot_or_rbf_count": updated_count,
        "tests_run": list(tests_run or []),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": _round_float(active_config.clock() - started_at),
        "run_date": active_config.run_date,
    }
    artifact = apply_headline_gate(artifact)
    validate_artifact(artifact)
    if write:
        _write_json(active_config.output_path(), artifact)
    return artifact


def apply_headline_gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Apply the utility and non-forgetting gates that control the headline flag."""

    gated = dict(payload)
    non_forgetting = float(gated["forgetting_rate"]) <= float(gated["forgetting_threshold"])
    utility_improved = float(gated["utility_delta_vs_replay_only"]) > 0.0
    updated = int(gated.get("updated_knot_or_rbf_count", 0)) > 0
    ready = non_forgetting and utility_improved and updated
    gated["non_forgetting_passed"] = non_forgetting
    gated["kan_cl_self_learning_ready"] = ready
    if ready:
        gated["honest_verdict"] = "complete: kan_rbf_importance_self_learning_passed"
    elif not non_forgetting:
        gated["honest_verdict"] = "complete: kan_rbf_importance_probe_forgetting_guard_failed"
    else:
        gated["honest_verdict"] = "complete: kan_rbf_importance_probe_not_ready"
    return gated


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Fail closed if required fields or headline gates drift before delivery."""

    missing = sorted(set(REQUIRED_ARTIFACT_FIELDS) - set(artifact))
    if missing:
        raise AssertionError(f"missing required fields: {missing}")
    gated = apply_headline_gate(artifact)
    readiness_fields = ("honest_verdict", "kan_cl_self_learning_ready", "non_forgetting_passed")
    if {field: artifact[field] for field in readiness_fields} != {
        field: gated[field] for field in readiness_fields
    }:
        raise AssertionError("headline readiness disagrees with forgetting/utility gates")


def _run_no_update_baseline(stream: ConstraintStream) -> dict[str, Any]:
    pre = []
    post = []
    for rule in stream.rules:
        current = stream.holdout_by_constraint[rule.constraint_id]
        pre.append(_evaluate(current, lambda row: 0.5))
        post.append(_evaluate(current, lambda row: 0.5))
    final_metrics = _evaluate(stream.all_holdout(), lambda row: 0.5)
    return _summarize_policy(pre, post, final_metrics)


def _run_replay_scheduler_baseline(stream: ConstraintStream) -> dict[str, Any]:
    scheduler = ReplaySchedulerOnly()
    pre = []
    post = []
    for rule in stream.rules:
        current = stream.holdout_by_constraint[rule.constraint_id]
        pre.append(_evaluate(current, scheduler.predict_proba))
        scheduler.update(stream.train_by_constraint[rule.constraint_id], stream.rule_by_id)
        post.append(_evaluate(current, scheduler.predict_proba))
    final_metrics = _evaluate(stream.all_holdout(), scheduler.predict_proba)
    summary = _summarize_policy(pre, post, final_metrics)
    summary["replay_scheduler_updated"] = bool(scheduler.priority_by_constraint)
    summary["updated_priority_count"] = len(scheduler.priority_by_constraint)
    return summary


def _run_kan_importance_update(stream: ConstraintStream) -> tuple[dict[str, Any], float, int]:
    memory = RBFImportanceMemory(centers=stream.centers)
    pre = []
    post = []
    post_utility_by_constraint: dict[str, float] = {}
    for rule in stream.rules:
        current = stream.holdout_by_constraint[rule.constraint_id]
        pre.append(_evaluate(current, lambda row: memory.predict_proba(row.features)))
        memory.update(stream.train_by_constraint[rule.constraint_id], stream.rule_by_id)
        post_metrics = _evaluate(current, lambda row: memory.predict_proba(row.features))
        post.append(post_metrics)
        post_utility_by_constraint[rule.constraint_id] = post_metrics["utility"]
    final_by_constraint = {
        rule.constraint_id: _evaluate(
            stream.holdout_by_constraint[rule.constraint_id],
            lambda row: memory.predict_proba(row.features),
        )["utility"]
        for rule in stream.rules
    }
    previous_constraints = [rule.constraint_id for rule in stream.rules[:-1]]
    forgetting_rate = _mean(
        max(0.0, post_utility_by_constraint[constraint_id] - final_by_constraint[constraint_id])
        for constraint_id in previous_constraints
    )
    final_metrics = _evaluate(stream.all_holdout(), lambda row: memory.predict_proba(row.features))
    summary = _summarize_policy(pre, post, final_metrics)
    summary["post_update_utility_by_constraint"] = {
        key: _round_float(value) for key, value in post_utility_by_constraint.items()
    }
    summary["final_utility_by_constraint"] = {
        key: _round_float(value) for key, value in final_by_constraint.items()
    }
    summary["updated_knot_or_rbf_count"] = memory.updated_count()
    return summary, forgetting_rate, memory.updated_count()


def _summarize_policy(
    pre: Sequence[dict[str, float]],
    post: Sequence[dict[str, float]],
    final_metrics: dict[str, float],
) -> dict[str, Any]:
    return {
        "mean_pre_update_utility": _round_float(_mean(item["utility"] for item in pre)),
        "mean_post_update_utility": _round_float(_mean(item["utility"] for item in post)),
        "final_holdout_utility": final_metrics["utility"],
        "final_energy_proxy": final_metrics["energy_proxy"],
    }


def _evaluate(
    rows: Sequence[ConstraintExample],
    predict_proba: Callable[[ConstraintExample], float],
) -> dict[str, float]:
    probabilities = [float(predict_proba(row)) for row in rows]
    correct = [
        (probability >= 0.5) == bool(row.label)
        for probability, row in zip(probabilities, rows, strict=True)
    ]
    energy = [
        abs(probability - float(row.label))
        for probability, row in zip(probabilities, rows, strict=True)
    ]
    return {"utility": _round_float(_mean(correct)), "energy_proxy": _round_float(_mean(energy))}


def _make_examples(
    rule: ConstraintRule,
    centers: np.ndarray,
    split: str,
    per_center: int,
    rng: np.random.Generator,
) -> list[ConstraintExample]:
    rows = []
    for replica in range(per_center):
        for center_index in rule.center_indices:
            features = centers[center_index] + rng.normal(0.0, 0.018, size=2)
            label = 1 if center_index in rule.positive_centers else 0
            rows.append(
                ConstraintExample(
                    example_id=f"{rule.constraint_id}:{split}:{center_index}:{replica}",
                    constraint_id=rule.constraint_id,
                    split=split,
                    features=features.astype(np.float64),
                    label=label,
                )
            )
    return rows


def _rbf_centers() -> np.ndarray:
    return np.array(
        [
            [-0.75, -0.55],
            [-0.25, -0.55],
            [0.25, -0.55],
            [0.75, -0.55],
            [-0.75, 0.0],
            [-0.25, 0.0],
            [0.25, 0.0],
            [0.75, 0.0],
            [-0.75, 0.55],
            [-0.25, 0.55],
            [0.25, 0.55],
            [0.75, 0.55],
        ],
        dtype=np.float64,
    )


def _dataset_checksum(rows: Sequence[ConstraintExample]) -> str:
    stable_rows = [
        {
            "example_id": row.example_id,
            "label": row.label,
            "features": [_round_float(value) for value in row.features],
        }
        for row in rows
    ]
    payload = json.dumps(stable_rows, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-40.0, min(40.0, value))))


def _mean(values: Sequence[float] | Sequence[bool] | Any) -> float:
    collected = list(values)
    return float(sum(float(value) for value in collected) / len(collected))


def _round_float(value: float) -> float:
    return round(float(value), 12)
