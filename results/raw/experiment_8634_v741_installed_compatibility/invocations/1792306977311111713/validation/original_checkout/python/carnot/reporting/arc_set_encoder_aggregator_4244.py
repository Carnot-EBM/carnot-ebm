"""Exp 4244 grown-pool ARC set-encoder aggregator.

Spec refs: REQ-VERIFY-4244, SCENARIO-VERIFY-4244,
SCENARIO-VERIFY-4244-NO-GAIN, SCENARIO-VERIFY-4244-DEFERRED.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import random
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from carnot.reporting import oracle_distinct_arc_aggregator_4231 as agg4231


RANDOM_SEED = 4244
BOOTSTRAP_N = 1000
DEFAULT_N_FOLDS = 5
DEFAULT_TRAINING_EPOCHS = 32
DEFAULT_HIDDEN_DIM = 32
DEFAULT_LR = 0.01
BASELINE_392_HIGH_AUROC = 0.84
BASELINE_IMPROVEMENT_EPSILON = 0.001
OUTPUT_REL = Path("results/experiment_4244_arc_set_encoder_aggregator_build.json")
MODEL_REL = Path("results/experiment_4244_arc_set_encoder_aggregator_model.json")
GROWN_POOL_BUILD_REL = Path("results/experiment_4243_arc_candidate_pool_grow.json")
FEATURE_NAMES = agg4231.FEATURE_NAMES
SPEC_REFS = [
    "REQ-VERIFY-4244",
    "SCENARIO-VERIFY-4244",
    "SCENARIO-VERIFY-4244-NO-GAIN",
    "SCENARIO-VERIFY-4244-DEFERRED",
]

FIELD_PRINCIPLES = {
    "honest_verdict": (
        "Terminal-prefixed. A trained out-of-fold set-encoder OR an honest 'no gain "
        "over the .392 logistic baseline' is COMPLETE -- both feed A3, and the "
        "ablation makes a no-gain a real negative not a sparsity artifact."
    ),
    "aggregator_trained": (
        "BARE bool: A3's gate compares this raw value (gated-fields-must-be-bare); "
        "true iff a learned permutation-invariant Set-Encoder artifact was persisted out-of-fold."
    ),
    "oracle_distinct_auroc": (
        "BARE float: off-fold detection AUROC of the SET-ENCODER vs is_correct -- "
        "the oracle-distinct discrimination; >0.5 CI95-excl is the precondition "
        "for a beats-vote win, and it should improve on the .392 augmented-logistic."
    ),
    "set_encoder_vs_logistic_auroc_delta": (
        "BARE float: set_encoder off-fold AUROC minus the .392 augmented-logistic "
        "AUROC on the SAME folds -- the decision-grade ablation answering whether "
        "a true cross-candidate encoder helps over hand-built summary features."
    ),
    "wrong_majority_n": (
        "BARE int: held-out wrong-majority tasks the A3 gate scores on -- carried "
        "from the grown pool; the ARBITER/AggLM headroom the aggregator targets."
    ),
    "held_out_task_n": (
        "BARE int: tasks the A3 gate will score on -- target >=40 so the A3 result is not under-powered."
    ),
    "learned_verifier_path": (
        "The persisted Set-Encoder artifact A3 loads to rerank held-out ARC candidates; the build deliverable."
    ),
    "verifier_is_oracle": (
        "BARE bool=false -- the set-encoder scores WITHOUT executing demos "
        "(Circularity Discipline); this is what makes an A3 win headline/gate-eligible, "
        "unlike a circular execution verifier."
    ),
    "model_specs": (
        "The set-encoder architecture (DeepSets/attention) + cross-candidate "
        "feature set + calibrated imbalance-aware loss; required methodology."
    ),
    "random_seed": (
        "Determinism precondition; fold split + model init seeded so the AUROC is reproducible."
    ),
    "reproducibility_checksum": (
        "Hash of the grown pool + fold split + features; catches silent drift before A3 measures."
    ),
}

REQUIRED_FIELDS = (
    "honest_verdict",
    "aggregator_trained",
    "oracle_distinct_auroc",
    "oracle_distinct_auroc_ci95",
    "set_encoder_vs_logistic_auroc_delta",
    "wrong_majority_n",
    "held_out_task_n",
    "learned_verifier_path",
    "verifier_is_oracle",
    "model_specs",
    "random_seed",
    "reproducibility_checksum",
    "field_principles",
    "spec_refs",
    "acceptance_gate",
)


class DeferredRun(RuntimeError):
    """Expected precondition failure that writes a terminal deferred artifact."""


@dataclass(frozen=True)
class GrownPoolRow:
    task_id: str
    candidate_id: str
    candidate_index: int
    correct: bool
    features: dict[str, float]
    vote_weight: float


@dataclass(frozen=True)
class GrownPoolCorpus:
    rows: list[GrownPoolRow]
    pool_artifact_path: Path
    pool_artifact_sha256: str
    upstream_checksum: str
    held_out_task_n: int
    wrong_majority_n: int
    positive_candidate_n: int


@dataclass(frozen=True)
class OOFRow:
    task_id: str
    candidate_id: str
    correct: bool
    score: float
    fold: int
    train_task_ids: tuple[str, ...]


@dataclass(frozen=True)
class OOFModelReport:
    auroc: float
    ci95: tuple[float, float]
    fold_task_ids: list[list[str]]
    rows: list[OOFRow]
    final_model: dict[str, Any]


class DeepSetsContextScorer(torch.nn.Module):
    """Permutation-equivariant scorer: candidate embedding plus pooled set context."""

    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.candidate_encoder = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
        )
        self.context_head = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim * 3, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.candidate_encoder(x)
        mean_pool = encoded.mean(dim=0, keepdim=True)
        max_pool = encoded.max(dim=0, keepdim=True).values
        context = torch.cat([mean_pool, max_pool], dim=1).expand(encoded.shape[0], -1)
        return self.context_head(torch.cat([encoded, context], dim=1)).squeeze(-1)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _as_float(value: Any) -> float:
    if isinstance(value, bool):
        return 0.0
    if isinstance(value, (int, float)):
        value = float(value)
        return value if math.isfinite(value) else 0.0
    return 0.0


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise DeferredRun("complete_arc_set_encoder_deferred_no_grown_pool") from exc
    if not isinstance(payload, dict):
        raise DeferredRun("complete_arc_set_encoder_deferred_no_grown_pool")
    return payload


def _resolve_pool_path(root: Path, build: dict[str, Any]) -> Path:
    if build.get("arc_pool_grown") is not True:
        raise DeferredRun("complete_arc_set_encoder_deferred_no_grown_pool")
    rel = build.get("pool_artifact_path")
    if not isinstance(rel, str) or not rel:
        raise DeferredRun("complete_arc_set_encoder_deferred_no_grown_pool")
    pool_path = Path(rel)
    if not pool_path.is_absolute():
        pool_path = root / pool_path
    if not pool_path.exists():
        raise DeferredRun("complete_arc_set_encoder_deferred_no_grown_pool")
    return pool_path


def _load_pool_payload(pool_path: Path) -> dict[str, Any]:
    try:
        with gzip.open(pool_path, "rt", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception as exc:
        raise DeferredRun("complete_arc_set_encoder_deferred_no_grown_pool") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("tasks"), list):
        raise DeferredRun("complete_arc_set_encoder_deferred_no_grown_pool")
    return payload


def _candidate_features(candidate: dict[str, Any]) -> dict[str, float]:
    raw = candidate.get("features")
    if not isinstance(raw, dict):
        raw = {}
    return {name: _as_float(raw.get(name)) for name in FEATURE_NAMES}


def load_grown_pool(repo_root: Path | str = Path(".")) -> GrownPoolCorpus:
    """SCENARIO-VERIFY-4244: load Exp 4243 task-grouped candidate sets."""

    root = Path(repo_root)
    build_path = root / GROWN_POOL_BUILD_REL
    if not build_path.exists():
        raise DeferredRun("complete_arc_set_encoder_deferred_no_grown_pool")
    build = _read_json(build_path)
    pool_path = _resolve_pool_path(root, build)
    payload = _load_pool_payload(pool_path)

    rows: list[GrownPoolRow] = []
    for task in payload["tasks"]:
        if not isinstance(task, dict):
            continue
        task_id = str(task.get("task_id") or "")
        candidates = task.get("candidates")
        if not task_id or not isinstance(candidates, list):
            continue
        for fallback_index, candidate in enumerate(candidates):
            if not isinstance(candidate, dict):
                continue
            candidate_id = str(candidate.get("candidate_id") or f"{task_id}::candidate{fallback_index}")
            features = _candidate_features(candidate)
            rows.append(
                GrownPoolRow(
                    task_id=task_id,
                    candidate_id=candidate_id,
                    candidate_index=int(candidate.get("candidate_index", fallback_index)),
                    correct=candidate.get("is_correct") is True,
                    features=features,
                    vote_weight=features["vote_weight"],
                )
            )

    if not rows:
        raise DeferredRun("complete_arc_set_encoder_deferred_no_grown_pool")
    return GrownPoolCorpus(
        rows=rows,
        pool_artifact_path=pool_path.resolve(),
        pool_artifact_sha256=_sha256_file(pool_path),
        upstream_checksum=str(
            payload.get("reproducibility_checksum") or build.get("reproducibility_checksum") or ""
        ),
        held_out_task_n=int(payload.get("task_n") or build.get("held_out_task_n") or 0),
        wrong_majority_n=int(payload.get("wrong_majority_n") or build.get("wrong_majority_n") or 0),
        positive_candidate_n=int(
            payload.get("positive_candidate_n") or build.get("positive_candidate_n") or 0
        ),
    )


def accepted_rejected_counts(rows: list[GrownPoolRow]) -> dict[str, int]:
    accepted = sum(1 for row in rows if row.correct)
    return {"accepted": accepted, "rejected": len(rows) - accepted, "total": len(rows)}


def _rows_by_task(rows: list[GrownPoolRow]) -> dict[str, list[GrownPoolRow]]:
    grouped: dict[str, list[GrownPoolRow]] = defaultdict(list)
    for row in rows:
        grouped[row.task_id].append(row)
    return {task_id: sorted(items, key=lambda row: row.candidate_index) for task_id, items in grouped.items()}


def split_task_folds(
    rows: list[GrownPoolRow], random_seed: int = RANDOM_SEED, n_folds: int = DEFAULT_N_FOLDS
) -> list[set[str]]:
    grouped = _rows_by_task(rows)
    positive_tasks = [task_id for task_id, items in grouped.items() if any(row.correct for row in items)]
    negative_tasks = [task_id for task_id in grouped if task_id not in positive_tasks]
    rng = random.Random(random_seed)
    rng.shuffle(positive_tasks)
    rng.shuffle(negative_tasks)
    fold_count = max(2, min(int(n_folds), len(grouped)))
    folds = [set() for _ in range(fold_count)]
    for index, task_id in enumerate(positive_tasks):
        folds[index % fold_count].add(task_id)
    for index, task_id in enumerate(negative_tasks):
        folds[index % fold_count].add(task_id)
    return folds


def _feature_vector(row: GrownPoolRow) -> list[float]:
    return [float(row.features.get(name, 0.0)) for name in FEATURE_NAMES]


def _standardizer(rows: list[GrownPoolRow]) -> tuple[list[float], list[float]]:
    vectors = [_feature_vector(row) for row in rows]
    if not vectors:
        return [0.0 for _ in FEATURE_NAMES], [1.0 for _ in FEATURE_NAMES]
    means = [sum(vector[index] for vector in vectors) / len(vectors) for index in range(len(FEATURE_NAMES))]
    scales: list[float] = []
    for index, mean in enumerate(means):
        variance = sum((vector[index] - mean) ** 2 for vector in vectors) / len(vectors)
        scales.append(math.sqrt(variance) or 1.0)
    return means, scales


def _standardized_vector(features: dict[str, float], means: list[float], scales: list[float]) -> list[float]:
    return [
        (float(features.get(name, 0.0)) - means[index]) / scales[index]
        for index, name in enumerate(FEATURE_NAMES)
    ]


def _tensor_for_rows(rows: list[GrownPoolRow], means: list[float], scales: list[float]) -> torch.Tensor:
    return torch.tensor(
        [_standardized_vector(row.features, means, scales) for row in rows],
        dtype=torch.float32,
    )


def _labels_tensor(rows: list[GrownPoolRow]) -> torch.Tensor:
    return torch.tensor([1.0 if row.correct else 0.0 for row in rows], dtype=torch.float32)


def _set_deterministic_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:  # pragma: no cover - older torch compatibility.
        pass


def _train_task_order(grouped: dict[str, list[GrownPoolRow]], rng: random.Random) -> list[str]:
    positive = [task_id for task_id, rows in grouped.items() if any(row.correct for row in rows)]
    rng.shuffle(positive)
    return positive


def _state_to_json(model: DeepSetsContextScorer) -> dict[str, Any]:
    return {name: tensor.detach().cpu().tolist() for name, tensor in model.state_dict().items()}


def _state_from_json(model: DeepSetsContextScorer, state: dict[str, Any]) -> None:
    tensors = {name: torch.tensor(value, dtype=torch.float32) for name, value in state.items()}
    model.load_state_dict(tensors)


def _train_deepsets_model(
    rows: list[GrownPoolRow],
    *,
    random_seed: int,
    hidden_dim: int,
    training_epochs: int,
    lr: float,
) -> dict[str, Any]:
    means, scales = _standardizer(rows)
    counts = accepted_rejected_counts(rows)
    if counts["accepted"] < 1 or counts["rejected"] < 1:
        base_rate = counts["accepted"] / counts["total"] if counts["total"] else 0.0
        return {
            "model_type": "constant_set_score",
            "constant_score": float(base_rate),
            "feature_names": list(FEATURE_NAMES),
            "feature_means": means,
            "feature_scales": scales,
            "hidden_dim": int(hidden_dim),
            "temperature": 1.0,
        }

    _set_deterministic_seed(random_seed)
    model = DeepSetsContextScorer(len(FEATURE_NAMES), hidden_dim)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    grouped = _rows_by_task(rows)
    rng = random.Random(random_seed)
    for _epoch in range(int(training_epochs)):
        for task_id in _train_task_order(grouped, rng):
            task_rows = grouped[task_id]
            y = _labels_tensor(task_rows)
            pos = float(y.sum().item())
            neg = float(len(task_rows) - pos)
            pos_weight = torch.tensor([neg / max(pos, 1.0)], dtype=torch.float32)
            x = _tensor_for_rows(task_rows, means, scales)
            logits = model(x)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                logits,
                y,
                pos_weight=pos_weight,
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    train_logits, train_labels = _score_deepsets_logits(model, rows, means, scales)
    return {
        "model_type": "standardized_deepsets_context_temperature_calibrated",
        "feature_names": list(FEATURE_NAMES),
        "feature_means": [float(value) for value in means],
        "feature_scales": [float(value) for value in scales],
        "hidden_dim": int(hidden_dim),
        "state_dict": _state_to_json(model),
        "temperature": _fit_temperature(train_logits, train_labels),
    }


def _build_deepsets_model(model_payload: dict[str, Any]) -> DeepSetsContextScorer:
    model = DeepSetsContextScorer(len(FEATURE_NAMES), int(model_payload["hidden_dim"]))
    _state_from_json(model, model_payload["state_dict"])
    model.eval()
    return model


def _score_deepsets_logits(
    model: DeepSetsContextScorer,
    rows: list[GrownPoolRow],
    means: list[float],
    scales: list[float],
) -> tuple[list[float], list[bool]]:
    grouped = _rows_by_task(rows)
    logits_by_id: dict[str, float] = {}
    with torch.no_grad():
        for task_rows in grouped.values():
            logits = model(_tensor_for_rows(task_rows, means, scales)).detach().cpu().tolist()
            for row, logit in zip(task_rows, logits, strict=True):
                logits_by_id[row.candidate_id] = float(logit)
    return [logits_by_id[row.candidate_id] for row in rows], [row.correct for row in rows]


def _fit_temperature(logits: list[float], labels: list[bool]) -> float:
    if len(set(labels)) < 2:
        return 1.0
    best_temp = 1.0
    best_loss = float("inf")
    y = torch.tensor([1.0 if label else 0.0 for label in labels], dtype=torch.float32)
    raw = torch.tensor(logits, dtype=torch.float32)
    for temp in (0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0):
        loss = float(torch.nn.functional.binary_cross_entropy_with_logits(raw / temp, y).item())
        if loss < best_loss:
            best_loss = loss
            best_temp = temp
    return float(best_temp)


def _sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def _score_with_payload(model_payload: dict[str, Any], task_rows: list[GrownPoolRow]) -> dict[str, float]:
    if model_payload.get("model_type") == "constant_set_score":
        return {row.candidate_id: float(model_payload.get("constant_score", 0.0)) for row in task_rows}
    model = _build_deepsets_model(model_payload)
    means = [float(value) for value in model_payload["feature_means"]]
    scales = [float(value) for value in model_payload["feature_scales"]]
    temperature = max(float(model_payload.get("temperature", 1.0)), 1e-6)
    with torch.no_grad():
        logits = model(_tensor_for_rows(task_rows, means, scales)).detach().cpu().tolist()
    return {
        row.candidate_id: _sigmoid(float(logit) / temperature)
        for row, logit in zip(task_rows, logits, strict=True)
    }


def score_with_set_encoder(
    model_payload: dict[str, Any], candidate: GrownPoolRow, task_rows: list[GrownPoolRow]
) -> float:
    model = model_payload.get("model", model_payload)
    scores = _score_with_payload(model, task_rows)
    return float(scores[candidate.candidate_id])


def _fit_isotonic(raw_scores: list[float], labels: list[bool]) -> dict[str, list[float]]:
    if len(set(raw_scores)) < 2 or len(set(labels)) < 2:
        base = sum(labels) / float(len(labels)) if labels else 0.0
        return {"x": [0.0, 1.0], "y": [base, base]}
    calibrator = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    calibrator.fit(raw_scores, [int(label) for label in labels])
    return {
        "x": [float(value) for value in calibrator.X_thresholds_],
        "y": [float(value) for value in calibrator.y_thresholds_],
    }


def _apply_isotonic(value: float, calibration: dict[str, list[float]]) -> float:
    xs = [float(item) for item in calibration.get("x", [])]
    ys = [float(item) for item in calibration.get("y", [])]
    if not xs or not ys or len(xs) != len(ys):
        return value
    if value <= xs[0]:
        return ys[0]
    if value >= xs[-1]:
        return ys[-1]
    for index in range(1, len(xs)):
        if value <= xs[index]:
            left_x, right_x = xs[index - 1], xs[index]
            left_y, right_y = ys[index - 1], ys[index]
            frac = 0.0 if right_x == left_x else (value - left_x) / (right_x - left_x)
            return left_y + frac * (right_y - left_y)
    return ys[-1]  # pragma: no cover - bounded thresholds return before this fallback.


def _train_logistic_model(rows: list[GrownPoolRow], random_seed: int) -> dict[str, Any]:
    means, scales = _standardizer(rows)
    counts = accepted_rejected_counts(rows)
    if counts["accepted"] < 1 or counts["rejected"] < 1:
        base_rate = counts["accepted"] / counts["total"] if counts["total"] else 0.0
        return {
            "model_type": "constant_logistic_score",
            "constant_score": float(base_rate),
            "feature_names": list(FEATURE_NAMES),
            "feature_means": means,
            "feature_scales": scales,
        }
    x_train = [_standardized_vector(row.features, means, scales) for row in rows]
    labels = [int(row.correct) for row in rows]
    model = LogisticRegression(
        random_state=random_seed,
        solver="liblinear",
        max_iter=1000,
        class_weight="balanced",
    )
    model.fit(x_train, labels)
    raw_scores = [float(value) for value in model.predict_proba(x_train)[:, 1]]
    return {
        "model_type": "standardized_logistic_regression_isotonic_calibrated",
        "feature_names": list(FEATURE_NAMES),
        "feature_means": [float(value) for value in means],
        "feature_scales": [float(value) for value in scales],
        "intercept": float(model.intercept_[0]),
        "coefficients": [float(value) for value in model.coef_[0]],
        "isotonic_calibration": _fit_isotonic(raw_scores, [row.correct for row in rows]),
    }


def _score_logistic(model_payload: dict[str, Any], row: GrownPoolRow) -> float:
    if model_payload.get("model_type") == "constant_logistic_score":
        return float(model_payload.get("constant_score", 0.0))
    means = [float(value) for value in model_payload["feature_means"]]
    scales = [float(value) for value in model_payload["feature_scales"]]
    values = _standardized_vector(row.features, means, scales)
    logit = float(model_payload["intercept"]) + sum(
        float(weight) * value for weight, value in zip(model_payload["coefficients"], values, strict=True)
    )
    raw = _sigmoid(logit)
    return 0.99 * _apply_isotonic(raw, model_payload.get("isotonic_calibration", {})) + 0.01 * raw


def _auroc(labels: list[bool], scores: list[float]) -> float:
    positives = [score for label, score in zip(labels, scores, strict=True) if label]
    negatives = [score for label, score in zip(labels, scores, strict=True) if not label]
    if not positives or not negatives:
        return 0.0
    wins = 0.0
    for positive in positives:
        for negative in negatives:
            wins += 1.0 if positive > negative else 0.5 if positive == negative else 0.0
    return wins / float(len(positives) * len(negatives))


def _bootstrap_auroc_ci95(
    labels: list[bool],
    scores: list[float],
    random_seed: int,
    bootstrap_n: int = BOOTSTRAP_N,
) -> tuple[float, float]:
    if len(set(labels)) < 2 or not scores:
        return (0.0, 0.0)
    rng = random.Random(random_seed)
    samples: list[float] = []
    n = len(labels)
    for _ in range(bootstrap_n):
        indices = [rng.randrange(n) for _ in range(n)]
        sample_labels = [labels[index] for index in indices]
        if len(set(sample_labels)) < 2:
            continue
        samples.append(_auroc(sample_labels, [scores[index] for index in indices]))
    if not samples:
        point = _auroc(labels, scores)
        return point, point
    samples.sort()
    return samples[int(0.025 * (len(samples) - 1))], samples[int(0.975 * (len(samples) - 1))]


def _round_metric(value: float) -> float:
    return round(float(value), 10)


def _oof_rows_to_payload(rows: list[OOFRow]) -> list[dict[str, Any]]:
    return [
        {
            "candidate_id": row.candidate_id,
            "correct": row.correct,
            "fold": row.fold,
            "score": _round_metric(row.score),
            "task_id": row.task_id,
            "train_task_ids": list(row.train_task_ids),
        }
        for row in rows
    ]


def train_oof_set_encoder(
    rows: list[GrownPoolRow],
    *,
    folds: list[set[str]] | None = None,
    random_seed: int = RANDOM_SEED,
    bootstrap_n: int = BOOTSTRAP_N,
    hidden_dim: int = DEFAULT_HIDDEN_DIM,
    training_epochs: int = DEFAULT_TRAINING_EPOCHS,
    lr: float = DEFAULT_LR,
) -> OOFModelReport:
    """SCENARIO-VERIFY-4244: score candidates with task-held-out set encoders."""

    task_folds = folds if folds is not None else split_task_folds(rows, random_seed)
    grouped = _rows_by_task(rows)
    oof_scores: dict[str, float] = {}
    oof_rows: list[OOFRow] = []
    for fold, heldout in enumerate(task_folds):
        train_rows = [row for row in rows if row.task_id not in heldout]
        model_payload = _train_deepsets_model(
            train_rows,
            random_seed=random_seed + fold,
            hidden_dim=hidden_dim,
            training_epochs=training_epochs,
            lr=lr,
        )
        train_task_ids = tuple(sorted({row.task_id for row in train_rows}))
        for task_id in sorted(heldout):
            task_scores = _score_with_payload(model_payload, grouped[task_id])
            for row in grouped[task_id]:
                score = task_scores[row.candidate_id]
                oof_scores[row.candidate_id] = score
                oof_rows.append(
                    OOFRow(row.task_id, row.candidate_id, row.correct, score, fold, train_task_ids)
                )
    labels = [row.correct for row in rows]
    scores = [oof_scores[row.candidate_id] for row in rows]
    final_model = _train_deepsets_model(
        rows,
        random_seed=random_seed,
        hidden_dim=hidden_dim,
        training_epochs=training_epochs,
        lr=lr,
    )
    return OOFModelReport(
        auroc=_auroc(labels, scores),
        ci95=_bootstrap_auroc_ci95(labels, scores, random_seed, bootstrap_n),
        fold_task_ids=[sorted(fold) for fold in task_folds],
        rows=oof_rows,
        final_model=final_model,
    )


def train_oof_logistic(
    rows: list[GrownPoolRow],
    *,
    folds: list[set[str]],
    random_seed: int = RANDOM_SEED,
    bootstrap_n: int = BOOTSTRAP_N,
) -> OOFModelReport:
    grouped = _rows_by_task(rows)
    oof_scores: dict[str, float] = {}
    oof_rows: list[OOFRow] = []
    for fold, heldout in enumerate(folds):
        train_rows = [row for row in rows if row.task_id not in heldout]
        model_payload = _train_logistic_model(train_rows, random_seed + fold)
        train_task_ids = tuple(sorted({row.task_id for row in train_rows}))
        for task_id in sorted(heldout):
            for row in grouped[task_id]:
                score = _score_logistic(model_payload, row)
                oof_scores[row.candidate_id] = score
                oof_rows.append(
                    OOFRow(row.task_id, row.candidate_id, row.correct, score, fold, train_task_ids)
                )
    labels = [row.correct for row in rows]
    scores = [oof_scores[row.candidate_id] for row in rows]
    return OOFModelReport(
        auroc=_auroc(labels, scores),
        ci95=_bootstrap_auroc_ci95(labels, scores, random_seed + 17, bootstrap_n),
        fold_task_ids=[sorted(fold) for fold in folds],
        rows=oof_rows,
        final_model=_train_logistic_model(rows, random_seed),
    )


def _model_specs(
    *,
    status: str,
    hidden_dim: int = DEFAULT_HIDDEN_DIM,
    training_epochs: int = DEFAULT_TRAINING_EPOCHS,
) -> dict[str, Any]:
    return {
        "architecture": "deepsets_pooled_context_set_encoder",
        "architecture_note": (
            "Permutation-equivariant DeepSets scorer: per-candidate MLP embedding, "
            "mean/max pooled full-set context, and per-candidate logits conditioned on that context."
        ),
        "set_attention_reference": (
            "CPU-fast fallback for the Set-LLM set-attention-mask / Set-Encoder cross-candidate "
            "attention lever; still consumes the full candidate set without positional features."
        ),
        "feature_set": list(FEATURE_NAMES),
        "cross_candidate_conditioning": (
            "vote priors, self-consistency margins, modal-grid agreement, duplicate counts, "
            "shape/palette family indicators, and DeepSets mean/max latent pooling over all candidates"
        ),
        "imbalance_loss": (
            "class_weighted_bce_with_positive_task_minibatches; negative-only tasks are scored "
            "held-out but do not form training minibatches"
        ),
        "calibration": "train_fold_temperature_scaling_for_set_encoder_and_isotonic_for_logistic",
        "logistic_ablation": "same_fold_augmented_logistic_392",
        "training_recipe": "task_held_out_oof_candidate_detection",
        "hidden_dim": int(hidden_dim),
        "training_epochs": int(training_epochs),
        "status": status,
    }


def reproducibility_checksum(
    corpus: GrownPoolCorpus,
    set_report: OOFModelReport,
    logistic_report: OOFModelReport,
    *,
    random_seed: int = RANDOM_SEED,
) -> str:
    payload = {
        "feature_names": list(FEATURE_NAMES),
        "fold_task_ids": set_report.fold_task_ids,
        "logistic_fold_task_ids": logistic_report.fold_task_ids,
        "pool_artifact_sha256": corpus.pool_artifact_sha256,
        "random_seed": random_seed,
        "rows": [
            {
                "candidate_id": row.candidate_id,
                "correct": row.correct,
                "features": row.features,
                "task_id": row.task_id,
            }
            for row in corpus.rows
        ],
        "upstream_checksum": corpus.upstream_checksum,
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return "sha256:" + hashlib.sha256(raw.encode("utf-8")).hexdigest()


def persist_set_encoder(
    path: Path,
    *,
    corpus: GrownPoolCorpus,
    set_report: OOFModelReport,
    logistic_report: OOFModelReport,
    checksum: str,
    counts: dict[str, int],
    random_seed: int,
    no_gain_reason: str | None,
    hidden_dim: int,
    training_epochs: int,
) -> None:
    payload = {
        "model": set_report.final_model,
        "model_type": set_report.final_model.get("model_type"),
        "feature_names": list(FEATURE_NAMES),
        "accepted_rejected_n": counts,
        "held_out_task_n": corpus.held_out_task_n,
        "wrong_majority_n": corpus.wrong_majority_n,
        "positive_candidate_n": corpus.positive_candidate_n,
        "model_specs": _model_specs(
            status="trained",
            hidden_dim=hidden_dim,
            training_epochs=training_epochs,
        ),
        "no_gain_reason": no_gain_reason,
        "set_encoder_oof": {
            "auroc": _round_metric(set_report.auroc),
            "ci95": [_round_metric(value) for value in set_report.ci95],
            "fold_task_ids": set_report.fold_task_ids,
            "rows": _oof_rows_to_payload(set_report.rows),
        },
        "logistic_ablation": {
            "auroc": _round_metric(logistic_report.auroc),
            "ci95": [_round_metric(value) for value in logistic_report.ci95],
            "fold_task_ids": logistic_report.fold_task_ids,
            "rows": _oof_rows_to_payload(logistic_report.rows),
            "final_model": logistic_report.final_model,
        },
        "random_seed": int(random_seed),
        "reproducibility_checksum": checksum,
        "pool_artifact_path": str(corpus.pool_artifact_path),
        "pool_artifact_sha256": corpus.pool_artifact_sha256,
        "spec_refs": SPEC_REFS,
        "verifier_is_oracle": False,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_set_encoder(path: Path | str) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("set-encoder artifact must be a JSON object")
    return payload


def _no_gain_reason(
    set_auroc: float,
    logistic_auroc: float,
    *,
    baseline_392_high: float = BASELINE_392_HIGH_AUROC,
) -> str | None:
    delta = set_auroc - logistic_auroc
    if delta <= BASELINE_IMPROVEMENT_EPSILON:
        return "no_gain_over_same_fold_logistic"
    if set_auroc <= baseline_392_high + BASELINE_IMPROVEMENT_EPSILON:
        return "no_gain_over_392_augmented_logistic_range"
    return None


def _blocked_checksum(repo_root: Path | str) -> str:
    root = Path(repo_root)
    build_path = root / GROWN_POOL_BUILD_REL
    payload = {
        "build_exists": build_path.exists(),
        "build_sha256": _sha256_file(build_path) if build_path.exists() else "",
        "feature_names": list(FEATURE_NAMES),
        "random_seed": RANDOM_SEED,
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return "sha256:" + hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _blocked_artifact(reason: str, *, random_seed: int, checksum: str, duration_s: float) -> dict[str, Any]:
    return {
        "experiment": "experiment_4244_arc_set_encoder_aggregator_build",
        "schema": "carnot.arc_set_encoder_aggregator_4244.v1",
        "honest_verdict": reason,
        "aggregator_trained": False,
        "oracle_distinct_auroc": 0.0,
        "oracle_distinct_auroc_ci95": [0.0, 0.0],
        "set_encoder_vs_logistic_auroc_delta": 0.0,
        "logistic_auroc": 0.0,
        "logistic_auroc_ci95": [0.0, 0.0],
        "wrong_majority_n": 0,
        "held_out_task_n": 0,
        "learned_verifier_path": "",
        "verifier_is_oracle": False,
        "model_specs": _model_specs(status="deferred"),
        "random_seed": int(random_seed),
        "reproducibility_checksum": checksum,
        "field_principles": FIELD_PRINCIPLES,
        "spec_refs": SPEC_REFS,
        "acceptance_gate": True,
        "inference_substrate": "cached_grown_arc_pool_oof_deepsets_set_encoder",
        "duration_s": round(duration_s, 6),
    }


def _complete_artifact(
    corpus: GrownPoolCorpus,
    set_report: OOFModelReport,
    logistic_report: OOFModelReport,
    *,
    checksum: str,
    counts: dict[str, int],
    model_path: Path,
    random_seed: int,
    duration_s: float,
    no_gain_reason: str | None,
    hidden_dim: int,
    training_epochs: int,
) -> dict[str, Any]:
    set_auroc = _round_metric(set_report.auroc)
    logistic_auroc = _round_metric(logistic_report.auroc)
    delta = _round_metric(set_report.auroc - logistic_report.auroc)
    if no_gain_reason:
        verdict = f"complete_arc_set_encoder_no_gain_over_logistic_auroc{set_auroc:.4f}"
    else:
        verdict = f"complete: arc_set_encoder_trained_auroc_{set_auroc:.4f}"
    return {
        "experiment": "experiment_4244_arc_set_encoder_aggregator_build",
        "schema": "carnot.arc_set_encoder_aggregator_4244.v1",
        "honest_verdict": verdict,
        "aggregator_trained": True,
        "oracle_distinct_auroc": set_auroc,
        "oracle_distinct_auroc_ci95": [_round_metric(value) for value in set_report.ci95],
        "set_encoder_vs_logistic_auroc_delta": delta,
        "logistic_auroc": logistic_auroc,
        "logistic_auroc_ci95": [_round_metric(value) for value in logistic_report.ci95],
        "wrong_majority_n": int(corpus.wrong_majority_n),
        "held_out_task_n": int(corpus.held_out_task_n),
        "learned_verifier_path": str(model_path),
        "verifier_is_oracle": False,
        "model_specs": _model_specs(
            status="trained",
            hidden_dim=hidden_dim,
            training_epochs=training_epochs,
        ),
        "random_seed": int(random_seed),
        "reproducibility_checksum": checksum,
        "field_principles": FIELD_PRINCIPLES,
        "spec_refs": SPEC_REFS,
        "acceptance_gate": True,
        "accepted_rejected_n": counts,
        "positive_candidate_n": int(corpus.positive_candidate_n),
        "no_gain_reason": no_gain_reason,
        "oof_folds": len(set_report.fold_task_ids),
        "pool_artifact_path": str(corpus.pool_artifact_path),
        "pool_artifact_sha256": corpus.pool_artifact_sha256,
        "inference_substrate": "cached_grown_arc_pool_oof_deepsets_set_encoder",
        "duration_s": round(duration_s, 6),
    }


def validate_artifact(artifact: dict[str, Any]) -> None:
    missing = [field for field in REQUIRED_FIELDS if field not in artifact]
    if missing:
        raise ValueError(f"missing required fields: {missing}")
    verdict = artifact["honest_verdict"]
    prefixes = ("complete:", "success:", "passed:", "shipped:", "complete_", "blocked_")
    if not isinstance(verdict, str) or not verdict.startswith(prefixes):
        raise ValueError("honest_verdict must be terminal-prefixed")
    if not isinstance(artifact["aggregator_trained"], bool):
        raise ValueError("aggregator_trained must be a bare bool")
    if not isinstance(artifact["oracle_distinct_auroc"], float):
        raise ValueError("oracle_distinct_auroc must be a bare float")
    if not isinstance(artifact["set_encoder_vs_logistic_auroc_delta"], float):
        raise ValueError("set_encoder_vs_logistic_auroc_delta must be a bare float")
    if not isinstance(artifact["wrong_majority_n"], int):
        raise ValueError("wrong_majority_n must be a bare int")
    if not isinstance(artifact["held_out_task_n"], int):
        raise ValueError("held_out_task_n must be a bare int")
    if artifact["verifier_is_oracle"] is not False:
        raise ValueError("verifier_is_oracle must be the bare bool false")
    if artifact["field_principles"] != FIELD_PRINCIPLES:
        raise ValueError("field_principles do not match REQ-VERIFY-4244")
    if artifact["aggregator_trained"] and not Path(artifact["learned_verifier_path"]).exists():
        raise ValueError("trained artifacts require a persisted learned_verifier_path")


def run(
    repo_root: Path | str = Path("."),
    *,
    random_seed: int = RANDOM_SEED,
    n_folds: int = DEFAULT_N_FOLDS,
    bootstrap_n: int = BOOTSTRAP_N,
    training_epochs: int = DEFAULT_TRAINING_EPOCHS,
    hidden_dim: int = DEFAULT_HIDDEN_DIM,
    lr: float = DEFAULT_LR,
    baseline_392_high: float = BASELINE_392_HIGH_AUROC,
) -> dict[str, Any]:
    start = time.perf_counter()
    root = Path(repo_root)
    output_path = root / OUTPUT_REL
    try:
        corpus = load_grown_pool(root)
        folds = split_task_folds(corpus.rows, random_seed=random_seed, n_folds=n_folds)
        set_report = train_oof_set_encoder(
            corpus.rows,
            folds=folds,
            random_seed=random_seed,
            bootstrap_n=bootstrap_n,
            hidden_dim=hidden_dim,
            training_epochs=training_epochs,
            lr=lr,
        )
        logistic_report = train_oof_logistic(
            corpus.rows,
            folds=folds,
            random_seed=random_seed,
            bootstrap_n=bootstrap_n,
        )
        counts = accepted_rejected_counts(corpus.rows)
        checksum = reproducibility_checksum(
            corpus,
            set_report,
            logistic_report,
            random_seed=random_seed,
        )
        no_gain_reason = _no_gain_reason(
            set_report.auroc,
            logistic_report.auroc,
            baseline_392_high=baseline_392_high,
        )
        model_path = (root / MODEL_REL).resolve()
        persist_set_encoder(
            model_path,
            corpus=corpus,
            set_report=set_report,
            logistic_report=logistic_report,
            checksum=checksum,
            counts=counts,
            random_seed=random_seed,
            no_gain_reason=no_gain_reason,
            hidden_dim=hidden_dim,
            training_epochs=training_epochs,
        )
        artifact = _complete_artifact(
            corpus,
            set_report,
            logistic_report,
            checksum=checksum,
            counts=counts,
            model_path=model_path,
            random_seed=random_seed,
            duration_s=time.perf_counter() - start,
            no_gain_reason=no_gain_reason,
            hidden_dim=hidden_dim,
            training_epochs=training_epochs,
        )
    except DeferredRun as exc:
        artifact = _blocked_artifact(
            str(exc),
            random_seed=random_seed,
            checksum=_blocked_checksum(root),
            duration_s=time.perf_counter() - start,
        )
    validate_artifact(artifact)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def main() -> None:  # pragma: no cover - exercised by the result entrypoint.
    repo_root = Path(__file__).resolve().parents[3]
    print(json.dumps(run(repo_root), indent=2, sort_keys=True))
