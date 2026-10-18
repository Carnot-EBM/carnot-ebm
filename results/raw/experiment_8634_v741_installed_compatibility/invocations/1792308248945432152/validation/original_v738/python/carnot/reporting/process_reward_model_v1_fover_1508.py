"""Exp 1423 lightweight FoVer process reward model v1.

Process reward modeling only helps Carnot if it can make a cheap local
step-correctness prediction without replaying certificate extraction, parsing,
and validation for every candidate step.  This module reconstructs the local
labels already available from Exp 1395, FoVer, step-level PRM rows, and Exp
1397 certificate outputs, then trains a deterministic hashed-feature logistic
classifier on CPU.

Spec: REQ-VERIFY-1423, SCENARIO-VERIFY-1423.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RESULTS_DIR = REPO_ROOT / "results"
DEFAULT_DATA_DIR = REPO_ROOT / "data"
DEFAULT_MODELS_DIR = REPO_ROOT / "python" / "carnot" / "models"

EXP1395_FILE = "experiment_1395_fr11_self_learning_v5.json"
EXP1397_FILE = "experiment_1397_fullscale_pipeline_v2_200cases.json"
OUTPUT_FILE = "experiment_1423_process_reward_model_v1_fover_1508.json"

DEFAULT_EXP1395_PATH = DEFAULT_RESULTS_DIR / EXP1395_FILE
DEFAULT_EXP1397_PATH = DEFAULT_RESULTS_DIR / EXP1397_FILE
DEFAULT_FOVER_PATH = DEFAULT_DATA_DIR / "fover_corpus.jsonl"
DEFAULT_STEP_PRM_PATH = DEFAULT_DATA_DIR / "step_level_prm_training.jsonl"
DEFAULT_OUTPUT_PATH = DEFAULT_RESULTS_DIR / OUTPUT_FILE
DEFAULT_CHECKPOINT_PATH = DEFAULT_MODELS_DIR / "prmv1_fover_1508_checkpoint.pt"

EXPERIMENT = "1423_process_reward_model_v1_fover_1508"
SCHEMA = "process_reward_model_v1_fover_1508_v1"
RUN_DATE = "20260506"
PROMOTED_PREFIX = "dvi_v2:fover:"
TRAINING_METHOD = "deterministic_hashed_feature_logistic_prm_v1"
FRESH_VERIFIED_CASE_COUNT = 1508

HASH_FEATURES = 64
NUMERIC_FEATURES = 8
FEATURE_DIM = HASH_FEATURES + NUMERIC_FEATURES
FOVER_SPLIT_SEED = 1423
N_EPOCHS = 80
LEARNING_RATE = 0.4
L2_WEIGHT_DECAY = 1e-4

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "training_traces_used",
    "step_labels_available",
    "prmv1_trained",
    "prmv1_auroc",
    "prmv1_step_precision",
    "prmv1_step_recall",
    "checkpoint_path",
    "honest_verdict",
)

_TOKEN_RE = re.compile(r"[A-Za-z_]+|\d+(?:\.\d+)?|[=+\-*/<>]+")
_CORRECT_VALUES = {"correct", "true", "supported", "entailed", "sat", "pass", "1"}
_WRONG_VALUES = {
    "incorrect",
    "wrong",
    "false",
    "violated",
    "violation",
    "repair_hint",
    "unsat",
    "fail",
    "0",
}


@dataclass(frozen=True)
class StepLabel:
    """One local process-supervision row for a candidate reasoning step.

    ``correct`` is the process-reward target: true means the step should receive
    positive reward, false means the step should be rejected or repaired.  The
    source fields keep the label auditable because Exp 1423 intentionally mixes
    direct FoVer labels, prior step-level PRM labels, and certificate-derived
    labels instead of pretending they came from a single annotation pass.
    """

    case_id: str
    text: str
    correct: bool
    label_source: str = "unit"
    trace_source: str = "unknown"
    prefix_fraction: float = 1.0


@dataclass(frozen=True)
class LabelCoverage:
    """Coverage summary for the 1508 promoted Exp 1395 trace IDs."""

    promoted_traces: int
    training_traces_used: int
    missing_trace_labels: int
    positive_step_labels: int
    negative_step_labels: int


@dataclass(frozen=True)
class TrainingResult:
    """Classifier metrics and checkpoint state for artifact serialization."""

    trained: bool
    auroc: float | None
    precision: float | None
    recall: float | None
    checkpoint_path: str | None
    threshold: float
    loss_history: list[float]
    train_labels_used: int
    heldout_labels_used: int


def _write_json(path: Path | str, artifact: Mapping[str, Any]) -> dict[str, Any]:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(artifact)
    destination.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    return payload


def _metadata(project_root: str | Path, run_date: str) -> dict[str, str]:
    return {"project_root": str(project_root), "run_date": run_date}


def write_in_progress_artifact(
    out_path: Path | str = DEFAULT_OUTPUT_PATH,
    *,
    project_root: str | Path = REPO_ROOT,
    run_date: str = RUN_DATE,
) -> dict[str, Any]:
    """REQ-VERIFY-1423: write the bootstrap artifact before loading labels."""

    return _write_json(
        out_path,
        {
            "experiment": EXPERIMENT,
            "schema": SCHEMA,
            "artifact_metadata": _metadata(project_root, run_date),
            "run_date": run_date,
            "status": "in_progress",
            "training_traces_used": 0,
            "step_labels_available": 0,
            "prmv1_trained": False,
            "prmv1_auroc": None,
            "prmv1_step_precision": None,
            "prmv1_step_recall": None,
            "checkpoint_path": None,
            "honest_verdict": "in_progress",
            "fresh_llm_inference_used": False,
            "cpu_only": True,
        },
    )


def load_json(path: Path | str) -> dict[str, Any]:
    """Load a JSON object artifact."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"artifact must be a JSON object: {path}")
    return payload


def load_jsonl_rows(path: Path | str) -> list[dict[str, Any]]:
    """Load JSONL rows, ignoring blanks and malformed lines."""

    rows: list[dict[str, Any]] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def promoted_case_ids(
    exp1395_artifact: Mapping[str, Any],
    *,
    expected_count: int | None = FRESH_VERIFIED_CASE_COUNT,
) -> list[str]:
    """Return Exp 1395's promoted FoVer case IDs without the memory prefix."""

    promoted = exp1395_artifact.get("memory_updates", {}).get("promoted", [])
    if not isinstance(promoted, Sequence) or isinstance(promoted, (str, bytes)):
        raise ValueError("Exp 1395 memory_updates.promoted must be a list")

    case_ids = [
        str(item)[len(PROMOTED_PREFIX) :]
        for item in promoted
        if str(item).startswith(PROMOTED_PREFIX)
    ]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("Exp 1395 promoted FoVer IDs contain duplicates")
    if expected_count is not None and len(case_ids) != int(expected_count):
        raise ValueError(
            "Exp 1395 fresh verified count mismatch: "
            f"ids={len(case_ids)} expected={int(expected_count)}"
        )
    return case_ids


def collect_promoted_step_labels(
    exp1395_artifact: Mapping[str, Any],
    *,
    fover_rows: Sequence[Mapping[str, Any]],
    step_prm_rows: Sequence[Mapping[str, Any]],
    exp1397_artifact: Mapping[str, Any] | None = None,
    expected_promoted_count: int | None = FRESH_VERIFIED_CASE_COUNT,
) -> tuple[list[StepLabel], LabelCoverage]:
    """REQ-VERIFY-1423: reconstruct local labels for promoted Exp 1395 traces."""

    case_ids = promoted_case_ids(exp1395_artifact, expected_count=expected_promoted_count)
    promoted = set(case_ids)
    labels: list[StepLabel] = []
    seen: set[tuple[str, str]] = set()
    labeled_traces: set[str] = set()

    def add_label(
        *,
        case_id: str,
        text: str,
        correct: bool | None,
        label_source: str,
        trace_source: str = "unknown",
        prefix_fraction: float = 1.0,
    ) -> None:
        clean_text = " ".join(str(text or "").split())
        if case_id not in promoted or correct is None or not clean_text:
            return
        key = (case_id, clean_text)
        if key in seen:
            return
        seen.add(key)
        labeled_traces.add(case_id)
        labels.append(
            StepLabel(
                case_id=case_id,
                text=clean_text,
                correct=bool(correct),
                label_source=label_source,
                trace_source=trace_source,
                prefix_fraction=float(prefix_fraction),
            )
        )

    for row in fover_rows:
        case_id = _case_id(row)
        add_label(
            case_id=case_id,
            text=_row_text(row),
            correct=_row_correct(row),
            label_source="fover_corpus_label",
            trace_source=str(row.get("source") or "fover"),
            prefix_fraction=1.0,
        )

    for row in step_prm_rows:
        case_id = _case_id(row)
        add_label(
            case_id=case_id,
            text=_row_text(row),
            correct=_label_value_is_correct(row.get("step_label")),
            label_source="step_level_prm_training",
            trace_source="step_level_prm_training",
            prefix_fraction=_float(row.get("prefix_fraction"), 1.0),
        )

    if exp1397_artifact is not None:
        generation_text = {
            str(row.get("case_id")): _row_text(row)
            for row in exp1397_artifact.get("generation_rows", [])
            if isinstance(row, Mapping)
        }
        for row in exp1397_artifact.get("certificate_rows", []):
            if not isinstance(row, Mapping) or row.get("parseable") is False:
                continue
            case_id = str(row.get("case_id") or "")
            state = str(row.get("expected_state") or row.get("tag_state") or "").lower()
            text = generation_text.get(case_id) or f"certificate_state={state}"
            add_label(
                case_id=case_id,
                text=text,
                correct=_label_value_is_correct(state),
                label_source="exp1397_certificate_label",
                trace_source=str(row.get("generation_source") or "exp1397"),
                prefix_fraction=1.0,
            )

    positives = sum(1 for label in labels if label.correct)
    negatives = len(labels) - positives
    coverage = LabelCoverage(
        promoted_traces=len(case_ids),
        training_traces_used=len(labeled_traces),
        missing_trace_labels=max(0, len(case_ids) - len(labeled_traces)),
        positive_step_labels=positives,
        negative_step_labels=negatives,
    )
    return labels, coverage


def tie_aware_auroc(labels: Sequence[int], scores: Sequence[float]) -> float:
    """Compute AUROC where label 1 and larger score mean step-correct."""

    pos = [float(score) for label, score in zip(labels, scores) if int(label) == 1]
    neg = [float(score) for label, score in zip(labels, scores) if int(label) == 0]
    if not pos or not neg:
        return 0.5
    wins = 0.0
    for pos_score in pos:
        for neg_score in neg:
            if pos_score > neg_score:
                wins += 1.0
            elif pos_score == neg_score:
                wins += 0.5
    return wins / (len(pos) * len(neg))


def classification_metrics(
    labels: Sequence[int],
    scores: Sequence[float],
    *,
    threshold: float,
) -> dict[str, float]:
    """Return precision and recall for the predicted-correct class."""

    tp = fp = fn = 0
    for label, score in zip(labels, scores):
        predicted = float(score) >= float(threshold)
        actual = int(label) == 1
        if predicted and actual:
            tp += 1
        elif predicted and not actual:
            fp += 1
        elif not predicted and actual:
            fn += 1
    precision = 0.0 if tp + fp == 0 else tp / (tp + fp)
    recall = 0.0 if tp + fn == 0 else tp / (tp + fn)
    return {"precision": precision, "recall": recall}


def train_and_evaluate(
    labels: Sequence[StepLabel],
    *,
    checkpoint_path: Path | str = DEFAULT_CHECKPOINT_PATH,
    n_epochs: int = N_EPOCHS,
    learning_rate: float = LEARNING_RATE,
    l2_weight_decay: float = L2_WEIGHT_DECAY,
) -> TrainingResult:
    """Train a CPU hashed-feature logistic PRM and save its checkpoint."""

    materialized = list(labels)
    positives = [label for label in materialized if label.correct]
    negatives = [label for label in materialized if not label.correct]
    if not positives or not negatives:
        return TrainingResult(False, None, None, None, None, 0.5, [], 0, 0)

    train_labels, heldout_labels = _stratified_split(materialized)
    train_x = np.stack([extract_features(label) for label in train_labels]).astype(np.float32)
    train_y = np.asarray([1.0 if label.correct else 0.0 for label in train_labels], dtype=np.float32)
    weights = np.zeros(FEATURE_DIM, dtype=np.float32)
    bias = _prior_logit(float(np.mean(train_y)))
    losses: list[float] = []

    for _ in range(max(1, int(n_epochs))):
        probs = _sigmoid_array(train_x @ weights + bias)
        losses.append(_binary_cross_entropy(train_y, probs))
        error = probs - train_y
        grad_w = train_x.T @ error / len(train_y) + float(l2_weight_decay) * weights
        grad_b = float(np.mean(error))
        weights = (weights - float(learning_rate) * grad_w).astype(np.float32)
        bias -= float(learning_rate) * grad_b

    train_scores = _predict_scores(train_labels, weights, bias)
    threshold = _best_threshold([1 if label.correct else 0 for label in train_labels], train_scores)
    heldout_scores = _predict_scores(heldout_labels, weights, bias)
    heldout_y = [1 if label.correct else 0 for label in heldout_labels]
    auroc = tie_aware_auroc(heldout_y, heldout_scores)
    metrics = classification_metrics(heldout_y, heldout_scores, threshold=threshold)
    _save_checkpoint(
        checkpoint_path,
        weights=weights,
        bias=bias,
        threshold=threshold,
        loss_history=losses,
        train_labels_used=len(train_labels),
        heldout_labels_used=len(heldout_labels),
    )
    return TrainingResult(
        trained=Path(checkpoint_path).exists(),
        auroc=auroc,
        precision=metrics["precision"],
        recall=metrics["recall"],
        checkpoint_path=str(checkpoint_path),
        threshold=threshold,
        loss_history=losses,
        train_labels_used=len(train_labels),
        heldout_labels_used=len(heldout_labels),
    )


def extract_features(label: StepLabel) -> np.ndarray:
    """Convert a step into a stable hashed text + numeric feature vector."""

    text = label.text
    tokens = _TOKEN_RE.findall(text.lower())
    features = np.zeros(FEATURE_DIM, dtype=np.float32)
    scale = 1.0 / math.sqrt(max(1, len(tokens)))
    for token in tokens:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        value = int.from_bytes(digest, "little")
        bucket = value % HASH_FEATURES
        sign = 1.0 if value & (1 << 63) else -1.0
        features[bucket] += sign * scale

    digit_count = sum(char.isdigit() for char in text)
    op_count = sum(text.count(op) for op in ("+", "-", "*", "/", "="))
    lower = text.lower()
    numeric = np.asarray(
        [
            math.log1p(len(text)) / 10.0,
            math.log1p(len(tokens)) / 5.0,
            min(1.0, digit_count / max(1, len(text))),
            min(1.0, op_count / max(1, len(tokens))),
            float(any(word in lower for word in ("sat", "valid", "correct", "therefore"))),
            float(any(word in lower for word in ("repair", "invalid", "wrong", "incorrect"))),
            max(0.0, min(1.0, float(label.prefix_fraction))),
            float(label.label_source == "exp1397_certificate_label"),
        ],
        dtype=np.float32,
    )
    features[HASH_FEATURES:] = numeric
    return features


def build_artifact(
    *,
    labels: Sequence[StepLabel],
    coverage: LabelCoverage,
    training_result: TrainingResult,
    started_at: str,
    duration_s: float,
    tests_run: Sequence[str],
    project_root: str | Path = REPO_ROOT,
    run_date: str = RUN_DATE,
) -> dict[str, Any]:
    """Build a complete or blocked Exp 1423 artifact."""

    trained = bool(training_result.trained)
    artifact = {
        "experiment": EXPERIMENT,
        "schema": SCHEMA,
        "artifact_metadata": _metadata(project_root, run_date),
        "run_date": run_date,
        "started_at": started_at,
        "finished_at": datetime.now(tz=UTC).isoformat(),
        "duration_s": round(float(duration_s), 3),
        "status": "complete" if trained else "blocked",
        "spec": ["REQ-VERIFY-1423", "SCENARIO-VERIFY-1423"],
        "source_artifacts": [
            f"results/{EXP1395_FILE}",
            f"results/{EXP1397_FILE}",
            "data/fover_corpus.jsonl",
            "data/step_level_prm_training.jsonl",
        ],
        "training_method": TRAINING_METHOD,
        "promoted_traces_available": int(coverage.promoted_traces),
        "training_traces_used": int(coverage.training_traces_used),
        "missing_trace_labels": int(coverage.missing_trace_labels),
        "step_labels_available": len(labels),
        "positive_step_labels": int(coverage.positive_step_labels),
        "negative_step_labels": int(coverage.negative_step_labels),
        "missing_positive_step_labels": max(0, 1 - int(coverage.positive_step_labels)),
        "missing_negative_step_labels": max(0, 1 - int(coverage.negative_step_labels)),
        "prmv1_trained": trained,
        "prmv1_auroc": _rounded_or_none(training_result.auroc),
        "prmv1_step_precision": _rounded_or_none(training_result.precision),
        "prmv1_step_recall": _rounded_or_none(training_result.recall),
        "checkpoint_path": training_result.checkpoint_path if trained else None,
        "feature_dim": FEATURE_DIM,
        "fover_split_seed": FOVER_SPLIT_SEED,
        "epochs_run": len(training_result.loss_history),
        "learning_rate": LEARNING_RATE,
        "l2_weight_decay": L2_WEIGHT_DECAY,
        "classification_threshold": round(float(training_result.threshold), 6),
        "training_loss_history": [round(float(loss), 6) for loss in training_result.loss_history],
        "train_step_labels_used": int(training_result.train_labels_used),
        "heldout_step_labels_used": int(training_result.heldout_labels_used),
        "tests_run": list(tests_run),
        "fresh_llm_inference_used": False,
        "cpu_only": True,
        "honest_verdict": _honest_verdict(trained, coverage),
    }
    validate_artifact(artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """SCENARIO-VERIFY-1423: enforce required fields and checkpoint invariants."""

    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        raise AssertionError(f"missing required fields: {missing}")
    if artifact["status"] not in {"in_progress", "complete", "blocked"}:
        raise AssertionError(f"unsupported status: {artifact['status']}")
    if artifact["status"] == "complete":
        if not artifact["prmv1_trained"]:
            raise AssertionError("complete PRM artifact requires prmv1_trained=true")
        for field in ("prmv1_auroc", "prmv1_step_precision", "prmv1_step_recall"):
            if artifact[field] is None:
                raise AssertionError(f"complete PRM artifact requires {field}")
        if not Path(str(artifact["checkpoint_path"])).exists():
            raise AssertionError("trained PRM artifact requires an existing checkpoint path")
    if artifact["status"] == "blocked" and artifact.get("checkpoint_path") is not None:
        raise AssertionError("blocked PRM artifacts must not expose a checkpoint path")


def run(
    *,
    exp1395_path: Path | str = DEFAULT_EXP1395_PATH,
    exp1397_path: Path | str = DEFAULT_EXP1397_PATH,
    fover_path: Path | str = DEFAULT_FOVER_PATH,
    step_prm_path: Path | str = DEFAULT_STEP_PRM_PATH,
    out_path: Path | str = DEFAULT_OUTPUT_PATH,
    checkpoint_path: Path | str = DEFAULT_CHECKPOINT_PATH,
    project_root: str | Path = REPO_ROOT,
    run_date: str = RUN_DATE,
    expected_promoted_count: int | None = FRESH_VERIFIED_CASE_COUNT,
    n_epochs: int = N_EPOCHS,
    tests_run: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Run Exp 1423 end-to-end on local labels and write the final artifact."""

    started_at = datetime.now(tz=UTC).isoformat()
    t0 = time.perf_counter()
    write_in_progress_artifact(out_path, project_root=project_root, run_date=run_date)
    exp1395 = load_json(exp1395_path)
    exp1397 = load_json(exp1397_path) if Path(exp1397_path).exists() else None
    labels, coverage = collect_promoted_step_labels(
        exp1395,
        fover_rows=load_jsonl_rows(fover_path),
        step_prm_rows=load_jsonl_rows(step_prm_path),
        exp1397_artifact=exp1397,
        expected_promoted_count=expected_promoted_count,
    )
    trainable = coverage.positive_step_labels > 0 and coverage.negative_step_labels > 0
    training_result = (
        train_and_evaluate(labels, checkpoint_path=checkpoint_path, n_epochs=n_epochs)
        if trainable
        else TrainingResult(False, None, None, None, None, 0.5, [], 0, 0)
    )
    artifact = build_artifact(
        labels=labels,
        coverage=coverage,
        training_result=training_result,
        started_at=started_at,
        duration_s=time.perf_counter() - t0,
        tests_run=list(tests_run or []),
        project_root=project_root,
        run_date=run_date,
    )
    return _write_json(out_path, artifact)


def _case_id(row: Mapping[str, Any]) -> str:
    return str(row.get("question_id") or row.get("case_id") or row.get("id") or "")


def _row_text(row: Mapping[str, Any]) -> str:
    return str(
        row.get("step_text")
        or row.get("partial_cot")
        or row.get("reasoning_text")
        or row.get("response")
        or row.get("step")
        or ""
    ).strip()


def _row_correct(row: Mapping[str, Any]) -> bool | None:
    if "is_correct" in row:
        return bool(row["is_correct"])
    if "step_correct" in row:
        return bool(row["step_correct"])
    return _label_value_is_correct(row.get("label"))


def _label_value_is_correct(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(int(value))
    normalized = str(value or "").strip().lower()
    if normalized in _CORRECT_VALUES:
        return True
    if normalized in _WRONG_VALUES:
        return False
    return None


def _stratified_split(labels: Sequence[StepLabel]) -> tuple[list[StepLabel], list[StepLabel]]:
    train: list[StepLabel] = []
    heldout: list[StepLabel] = []
    for correct in (True, False):
        group = [label for label in labels if label.correct is correct]
        ordered = sorted(group, key=lambda label: _stable_sort_key(label.case_id, label.text))
        if len(ordered) <= 1:
            train.extend(ordered)
            continue
        holdout_count = min(len(ordered) - 1, max(1, int(round(len(ordered) * 0.25))))
        heldout.extend(ordered[:holdout_count])
        train.extend(ordered[holdout_count:])
    return train, heldout or train


def _stable_sort_key(case_id: str, text: str) -> str:
    seed = f"{FOVER_SPLIT_SEED}:{case_id}:{text}".encode("utf-8")
    return hashlib.sha256(seed).hexdigest()


def _predict_scores(labels: Sequence[StepLabel], weights: np.ndarray, bias: float) -> list[float]:
    return [
        float(_sigmoid_scalar(float(np.dot(weights, extract_features(label)) + bias)))
        for label in labels
    ]


def _best_threshold(labels: Sequence[int], scores: Sequence[float]) -> float:
    candidates = sorted(set(float(score) for score in scores))
    if not candidates:
        return 0.5
    best_threshold = candidates[0]
    best_f1 = -1.0
    for threshold in candidates:
        metrics = classification_metrics(labels, scores, threshold=threshold)
        precision = metrics["precision"]
        recall = metrics["recall"]
        f1 = 0.0 if precision + recall == 0 else 2.0 * precision * recall / (precision + recall)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    return float(best_threshold)


def _save_checkpoint(
    path: Path | str,
    *,
    weights: np.ndarray,
    bias: float,
    threshold: float,
    loss_history: Sequence[float],
    train_labels_used: int,
    heldout_labels_used: int,
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as handle:
        np.savez(
            handle,
            weights=np.asarray(weights, dtype=np.float32),
            bias=np.asarray([bias], dtype=np.float32),
            threshold=np.asarray([threshold], dtype=np.float32),
            loss_history=np.asarray(loss_history, dtype=np.float32),
            feature_dim=np.asarray([FEATURE_DIM], dtype=np.int32),
            train_labels_used=np.asarray([train_labels_used], dtype=np.int32),
            heldout_labels_used=np.asarray([heldout_labels_used], dtype=np.int32),
            training_method=np.asarray([TRAINING_METHOD]),
        )


def _prior_logit(prior: float) -> float:
    clipped = min(0.999, max(0.001, float(prior)))
    return float(math.log(clipped / (1.0 - clipped)))


def _sigmoid_array(values: np.ndarray) -> np.ndarray:
    return (1.0 / (1.0 + np.exp(-np.clip(values, -40.0, 40.0)))).astype(np.float32)


def _sigmoid_scalar(value: float) -> float:
    return float(1.0 / (1.0 + math.exp(-max(-40.0, min(40.0, value)))))


def _binary_cross_entropy(labels: np.ndarray, probs: np.ndarray) -> float:
    eps = 1e-9
    return float(
        -np.mean(labels * np.log(probs + eps) + (1.0 - labels) * np.log(1.0 - probs + eps))
    )


def _honest_verdict(trained: bool, coverage: LabelCoverage) -> str:
    if trained:
        if coverage.missing_trace_labels:
            return (
                "prmv1_trained_on_available_step_labels_with_"
                f"{coverage.missing_trace_labels}_promoted_traces_missing_local_labels"
            )
        return "prmv1_trained_on_all_promoted_step_labels"
    missing = []
    if coverage.positive_step_labels <= 0:
        missing.append("positive_step_labels")
    if coverage.negative_step_labels <= 0:
        missing.append("negative_step_labels")
    if coverage.training_traces_used <= 0:
        missing.append("labeled_traces")
    return "prmv1_blocked_missing_" + "_and_".join(missing or ["unknown_label_requirement"])


def _rounded_or_none(value: float | None) -> float | None:
    if value is None:
        return None
    return round(float(value), 6)


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


if __name__ == "__main__":  # pragma: no cover
    print(json.dumps(run(), indent=2, sort_keys=True))
