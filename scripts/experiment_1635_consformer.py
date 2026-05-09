#!/usr/bin/env python3
"""Exp 1635 ConsFormer-style label-free refiner for FoVer CSP rows.

This is a compact CPU-only prototype.  It turns each FoVer reasoning step into
numeric constraint-satisfaction features, fits a transformer-style attention
refinement pass from those features without labels, and then evaluates held-out
accuracy against FoVer labels.

Spec: REQ-LEARN-1635, SCENARIO-LEARN-1635.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
import operator
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CORPUS_PATH = REPO_ROOT / "data" / "fover_corpus.jsonl"
DEFAULT_OUTPUT_PATH = REPO_ROOT / "results" / "experiment_1635_consformer.json"

EXPERIMENT_ID = 1635
SCHEMA = "carnot.self_learning.consformer_refiner.v1"
SPEC_REFS = ["REQ-LEARN-1635", "SCENARIO-LEARN-1635"]
RUN_DATE = "20260509"
DEFAULT_EVAL_FRACTION = 0.25
DEFAULT_THRESHOLD_QUANTILE = 0.20
EPSILON = 1e-8

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "schema",
    "experiment_id",
    "spec_refs",
    "dataset_rows",
    "train_rows",
    "eval_rows",
    "refiner_accuracy",
    "baseline_accuracy",
    "label_free_training",
    "tests_run",
    "honest_verdict",
)

_NUMBER_RE = re.compile(r"[-+]?\d+(?:\.\d+)?")
_EQUATION_RE = re.compile(
    r"(?P<left>[-+*/().\d\s]{3,80})=\s*(?P<right>[-+]?\d+(?:\.\d+)?)"
)
_WORD_RE = re.compile(r"[a-zA-Z]+")
_ALLOWED_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}
_ALLOWED_UNARYOPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}


@dataclass(frozen=True)
class FoVerRow:
    """One normalized FoVer row used by Exp 1635."""

    row_id: str
    step_text: str
    label: bool
    source: str = "unknown"


@dataclass(frozen=True)
class CSPFeatures:
    """Label-free constraint features extracted from a reasoning step."""

    equality_consistency: float
    equation_presence: float
    arithmetic_density: float
    numeric_coverage: float
    lexical_completion: float
    contradiction_absence: float
    repetition_absence: float
    truncation_absence: float

    def to_vector(self) -> np.ndarray:
        return np.asarray(
            (
                self.equality_consistency,
                self.equation_presence,
                self.arithmetic_density,
                self.numeric_coverage,
                self.lexical_completion,
                self.contradiction_absence,
                self.repetition_absence,
                self.truncation_absence,
            ),
            dtype=np.float64,
        )


@dataclass(frozen=True)
class EvaluationSummary:
    """Held-out evaluation result for the label-free refiner."""

    accuracy: float
    baseline_accuracy: float
    predictions: tuple[bool, ...]


@dataclass(frozen=True)
class ConsFormerRefiner:
    """A lightweight self-attention refiner over CSP feature tokens.

    The model stores only unlabeled distribution statistics plus an attention
    vector.  Labels are intentionally absent from the fitted state.
    """

    center: tuple[float, ...]
    scale: tuple[float, ...]
    attention: tuple[float, ...]
    threshold: float
    threshold_quantile: float = DEFAULT_THRESHOLD_QUANTILE
    label_free_training: bool = True

    def refine(self, features: CSPFeatures) -> np.ndarray:
        values = features.to_vector()
        centered = np.clip((values - np.asarray(self.center)) / np.asarray(self.scale), -4.0, 4.0)
        logits = np.outer(centered, centered) / math.sqrt(values.size)
        row_attention = _softmax_rows(logits)
        return row_attention @ values

    def score(self, features: CSPFeatures) -> float:
        refined = self.refine(features)
        attention = np.asarray(self.attention)
        return float(np.dot(attention, refined) / (np.sum(attention) + EPSILON))

    def predict(self, features: CSPFeatures) -> bool:
        return self.score(features) >= self.threshold

    def to_json(self) -> JsonDict:
        return {
            "center": [_round_float(value) for value in self.center],
            "scale": [_round_float(value) for value in self.scale],
            "attention": [_round_float(value) for value in self.attention],
            "threshold": _round_float(self.threshold),
            "threshold_quantile": self.threshold_quantile,
            "label_free_training": self.label_free_training,
        }


def _round_float(value: float) -> float:
    return round(float(value), 10)


def _write_json(path: Path | str, artifact: Mapping[str, Any]) -> JsonDict:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(artifact)
    destination.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    return payload


def _label_to_bool(label: object) -> bool | None:
    if label == "correct" or label is True:
        return True
    if label == "incorrect" or label is False:
        return False
    return None


def _row_id(raw: Mapping[str, Any], line_number: int) -> str:
    explicit = raw.get("row_id")
    if explicit:
        return str(explicit)
    question_id = raw.get("question_id", "row")
    return f"{question_id}:{line_number}"


def load_fover_rows(path: Path | str, *, limit: int | None = None) -> list[FoVerRow]:
    """REQ-LEARN-1635-2: load valid FoVer JSONL rows in deterministic order."""

    rows: list[FoVerRow] = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if limit is not None and len(rows) >= limit:
            break
        try:
            raw = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(raw, dict):
            continue
        label = _label_to_bool(raw.get("label"))
        step_text = raw.get("step_text")
        if label is None or not isinstance(step_text, str) or not step_text.strip():
            continue
        rows.append(
            FoVerRow(
                row_id=_row_id(raw, line_number),
                step_text=step_text.strip(),
                label=label,
                source=str(raw.get("source", "unknown")),
            )
        )
    return rows


def _eval_ast(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, int | float):
        return float(node.value)
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_BINOPS:
        left = _eval_ast(node.left)
        right = _eval_ast(node.right)
        return float(_ALLOWED_BINOPS[type(node.op)](left, right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_UNARYOPS:
        return float(_ALLOWED_UNARYOPS[type(node.op)](_eval_ast(node.operand)))
    raise ValueError("unsupported arithmetic expression")


def _safe_arithmetic_value(expression: str) -> float | None:
    try:
        parsed = ast.parse(expression, mode="eval")
        value = _eval_ast(parsed.body)
    except (SyntaxError, ValueError, ZeroDivisionError):
        return None
    return value if math.isfinite(value) else None


def _equality_consistency(text: str) -> tuple[float, int]:
    residuals: list[float] = []
    for match in _EQUATION_RE.finditer(text):
        predicted = _safe_arithmetic_value(match.group("left"))
        stated = _safe_arithmetic_value(match.group("right"))
        if predicted is None or stated is None:
            continue
        residuals.append(abs(predicted - stated) / (abs(stated) + 1.0))
    if not residuals:
        return 0.56, 0
    return max(0.0, 1.0 - min(1.0, min(residuals))), len(residuals)


def _repetition_absence(words: Sequence[str]) -> float:
    if not words:
        return 0.5
    repeated = sum(1 for index, word in enumerate(words[1:], 1) if word == words[index - 1])
    return 1.0 - min(1.0, repeated / max(1, len(words) - 1))


def extract_csp_features(step_text: str) -> CSPFeatures:
    """Extract label-free CSP features from one FoVer reasoning step."""

    text = step_text.strip()
    lower = text.lower()
    words = _WORD_RE.findall(lower)
    numbers = _NUMBER_RE.findall(text)
    equality_score, equation_count = _equality_consistency(text)
    contradiction_terms = ("contradiction", "impossible", "cannot", "not enough information")
    contradiction_absence = 0.0 if any(term in lower for term in contradiction_terms) else 1.0
    truncation_absence = 0.0 if text.endswith(("+", "-", "*", "/", "=", ",")) else 1.0
    sentence_completion = 1.0 if text.endswith((".", "!", "?", "}", ")")) else 0.82
    return CSPFeatures(
        equality_consistency=equality_score,
        equation_presence=min(1.0, equation_count / 2.0),
        arithmetic_density=min(1.0, len(numbers) / max(1, len(words)) * 5.0),
        numeric_coverage=min(1.0, len(numbers) / 8.0),
        lexical_completion=sentence_completion,
        contradiction_absence=contradiction_absence,
        repetition_absence=_repetition_absence(words),
        truncation_absence=truncation_absence,
    )


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values)
    exp_values = np.exp(shifted)
    return exp_values / (np.sum(exp_values) + EPSILON)


def _softmax_rows(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values, axis=1, keepdims=True)
    exp_values = np.exp(shifted)
    return exp_values / (np.sum(exp_values, axis=1, keepdims=True) + EPSILON)


def _feature_matrix(rows: Sequence[FoVerRow]) -> np.ndarray:
    return np.asarray([extract_csp_features(row.step_text).to_vector() for row in rows])


def train_label_free_refiner(
    rows: Sequence[FoVerRow],
    *,
    threshold_quantile: float = DEFAULT_THRESHOLD_QUANTILE,
) -> ConsFormerRefiner:
    """REQ-LEARN-1635-3: fit unlabeled scoring statistics from CSP features."""

    if not rows:
        raise ValueError("at least one train row is required")
    matrix = _feature_matrix(rows)
    center = matrix.mean(axis=0)
    scale = matrix.std(axis=0) + 0.05
    normalized = (matrix - center) / scale
    salience = np.mean(np.abs(normalized), axis=0) + np.mean(matrix, axis=0)
    attention = _softmax(salience)
    provisional = ConsFormerRefiner(
        center=tuple(float(value) for value in center),
        scale=tuple(float(value) for value in scale),
        attention=tuple(float(value) for value in attention),
        threshold=0.0,
        threshold_quantile=threshold_quantile,
    )
    train_scores = np.asarray(
        [provisional.score(extract_csp_features(row.step_text)) for row in rows],
        dtype=np.float64,
    )
    threshold = float(np.quantile(train_scores, threshold_quantile))
    return ConsFormerRefiner(
        center=provisional.center,
        scale=provisional.scale,
        attention=provisional.attention,
        threshold=threshold,
        threshold_quantile=threshold_quantile,
    )


def split_rows(
    rows: Sequence[FoVerRow],
    *,
    eval_fraction: float = DEFAULT_EVAL_FRACTION,
) -> tuple[list[FoVerRow], list[FoVerRow]]:
    """Create a deterministic held-out split without consulting labels."""

    if len(rows) < 2:
        raise ValueError("at least two rows are required for train/eval split")
    eval_count = max(1, min(len(rows) - 1, round(len(rows) * eval_fraction)))
    keyed = sorted(
        ((hashlib.sha256(row.row_id.encode("utf-8")).hexdigest(), row.row_id) for row in rows)
    )
    eval_ids = {row_id for _, row_id in keyed[:eval_count]}
    train_rows = [row for row in rows if row.row_id not in eval_ids]
    eval_rows = [row for row in rows if row.row_id in eval_ids]
    return train_rows, eval_rows


def evaluate_refiner(refiner: ConsFormerRefiner, rows: Sequence[FoVerRow]) -> EvaluationSummary:
    """Evaluate held-out FoVer labels after label-free training has finished."""

    predictions = tuple(refiner.predict(extract_csp_features(row.step_text)) for row in rows)
    correct = sum(prediction is row.label for prediction, row in zip(predictions, rows, strict=True))
    positives = sum(row.label for row in rows)
    baseline = max(positives, len(rows) - positives) / len(rows)
    return EvaluationSummary(
        accuracy=correct / len(rows),
        baseline_accuracy=baseline,
        predictions=predictions,
    )


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """REQ-LEARN-1635-4/5: validate the terminal artifact contract."""

    missing = set(REQUIRED_ARTIFACT_FIELDS) - set(artifact)
    _require(not missing, f"missing required fields: {sorted(missing)}")
    _require(artifact["schema"] == SCHEMA, "schema mismatch")
    _require(artifact["experiment_id"] == EXPERIMENT_ID, "experiment_id mismatch")
    _require(artifact["spec_refs"] == SPEC_REFS, "spec_refs mismatch")
    _require(artifact["dataset_rows"] >= 2, "dataset_rows must allow a split")
    _require(artifact["train_rows"] >= 1, "train_rows must be positive")
    _require(artifact["eval_rows"] >= 1, "eval_rows must be positive")
    _require(0.0 <= artifact["refiner_accuracy"] <= 1.0, "refiner_accuracy out of range")
    _require(0.0 <= artifact["baseline_accuracy"] <= 1.0, "baseline_accuracy out of range")
    _require(artifact["label_free_training"] is True, "label_free_training must be true")
    if artifact["status"] == "complete":
        _require(
            artifact["honest_verdict"] == "consformer_refiner_evaluated_label_free",
            "complete artifact must use the label-free verdict",
        )


def build_artifact(
    *,
    rows: Sequence[FoVerRow],
    tests_run: Sequence[str],
    eval_fraction: float = DEFAULT_EVAL_FRACTION,
) -> JsonDict:
    """Build the terminal Exp 1635 artifact from already-loaded FoVer rows."""

    train_rows, eval_rows = split_rows(rows, eval_fraction=eval_fraction)
    refiner = train_label_free_refiner(train_rows)
    evaluation = evaluate_refiner(refiner, eval_rows)
    artifact: JsonDict = {
        "status": "complete",
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "spec_refs": list(SPEC_REFS),
        "run_date": RUN_DATE,
        "dataset_rows": len(rows),
        "train_rows": len(train_rows),
        "eval_rows": len(eval_rows),
        "refiner_accuracy": _round_float(evaluation.accuracy),
        "baseline_accuracy": _round_float(evaluation.baseline_accuracy),
        "label_free_training": refiner.label_free_training,
        "threshold": _round_float(refiner.threshold),
        "threshold_quantile": refiner.threshold_quantile,
        "refiner": refiner.to_json(),
        "prediction_counts": {
            "correct": int(sum(evaluation.predictions)),
            "incorrect": int(len(evaluation.predictions) - sum(evaluation.predictions)),
        },
        "tests_run": list(tests_run),
        "honest_verdict": "consformer_refiner_evaluated_label_free",
    }
    validate_artifact(artifact)
    return artifact


def run_experiment(
    *,
    corpus_path: Path | str = DEFAULT_CORPUS_PATH,
    output_path: Path | str = DEFAULT_OUTPUT_PATH,
    tests_run: Sequence[str] = (),
    eval_fraction: float = DEFAULT_EVAL_FRACTION,
    limit: int | None = None,
) -> JsonDict:
    """Run Exp 1635 and write `results/experiment_1635_consformer.json`."""

    rows = load_fover_rows(corpus_path, limit=limit)
    artifact = build_artifact(rows=rows, tests_run=tests_run, eval_fraction=eval_fraction)
    return _write_json(output_path, artifact)


def main() -> None:  # pragma: no cover
    run_experiment(output_path=DEFAULT_OUTPUT_PATH)


if __name__ == "__main__":  # pragma: no cover
    main()
