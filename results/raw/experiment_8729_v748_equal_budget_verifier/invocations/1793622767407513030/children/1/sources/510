"""Exp 1130 helpers for measuring Zenil alpha_t after verifier retraining.

The prior SOTA measurement in Exp 1077 used alpha_t as the fraction of examples
where Carnot's verifier changed the selection relative to a verifier-free
temperature baseline.  Exp 1130 keeps that operational definition so the new
post-retrain number can be compared directly with the 0.38 baseline.

Spec: REQ-FR11-1130.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from statistics import median
from typing import Any, Sequence

ALPHA_T_PRIOR = 0.38
EXP1120_AUROC = 0.977419
ALPHA_T_METHOD = "exp1077_temperature_disagreement"

REQUIRED_ARTIFACT_FIELDS = [
    "alpha_t_prior",
    "alpha_t_post_retrain",
    "alpha_t_improved",
    "verifier_auroc_used",
    "n_evaluation_examples",
    "inference_mode",
    "fr11_self_learning_data_point_logged",
    "zenil_alpha_t_post_retrain_measured",
    "honest_verdict",
]


@dataclass(frozen=True)
class EvaluationExample:
    """One SOTA response with a known final-answer label.

    The label uses the usual binary convention: 1 means the response is correct,
    0 means it is incorrect.  Keeping the label integer-valued makes correlation
    and AUROC diagnostics unambiguous downstream.

    Spec: REQ-FR11-1130.
    """

    example_id: str
    question: str
    response: str
    correct_answer: int | float | str
    label: int


@dataclass(frozen=True)
class AlphaTMeasurement:
    """Exp1077-compatible alpha_t result for one scored batch."""

    alpha_t: float
    n_total: int
    n_disagreements: int
    disagreement_ids: list[str]
    verifier_verdicts: list[str]
    temperature_verdicts: list[str]


def pearson_corr(xs: Sequence[float], ys: Sequence[float | int]) -> float:
    """Return Pearson correlation, or 0.0 for degenerate vectors.

    The post-retrain verifier is an energy model where lower energy means more
    likely correct.  Callers that want a positive "grounding" direction should
    pass ``-energy`` as ``xs`` and the binary correctness label as ``ys``.
    """

    if len(xs) != len(ys):
        raise ValueError(f"xs and ys must have same length, got {len(xs)} vs {len(ys)}")
    n = len(xs)
    if n == 0:
        return 0.0
    x_vals = [float(x) for x in xs]
    y_vals = [float(y) for y in ys]
    x_mean = sum(x_vals) / n
    y_mean = sum(y_vals) / n
    x_centered = [x - x_mean for x in x_vals]
    y_centered = [y - y_mean for y in y_vals]
    x_norm = math.sqrt(sum(x * x for x in x_centered))
    y_norm = math.sqrt(sum(y * y for y in y_centered))
    if x_norm == 0.0 or y_norm == 0.0:
        return 0.0
    return sum(x * y for x, y in zip(x_centered, y_centered)) / (x_norm * y_norm)


def calibrate_low_energy_threshold(energies: Sequence[float], labels: Sequence[int]) -> float:
    """Choose a threshold for ``energy <= threshold`` meaning "correct".

    Exp 1120 fixed the ordering inversion, so correct examples should sit at
    lower energy than incorrect examples.  The threshold is calibrated by
    maximizing training accuracy over all midpoints between observed energies.
    Degenerate single-class batches fall back to the median energy.

    Spec: REQ-FR11-1130.
    """

    if len(energies) != len(labels):
        raise ValueError(
            f"energies and labels must have same length, got {len(energies)} vs {len(labels)}"
        )
    if not energies:
        return 0.0

    e_vals = [float(e) for e in energies]
    y_vals = [1 if int(y) else 0 for y in labels]
    if len(set(y_vals)) < 2:
        return float(median(e_vals))

    uniq = sorted(set(e_vals))
    if len(uniq) == 1:
        return uniq[0]

    candidates = [uniq[0] - 1e-9]
    candidates.extend((a + b) / 2.0 for a, b in zip(uniq, uniq[1:]))
    candidates.append(uniq[-1] + 1e-9)

    best_acc = -1.0
    best_thresholds: list[float] = []
    for threshold in candidates:
        preds = [1 if energy <= threshold else 0 for energy in e_vals]
        acc = sum(1 for pred, label in zip(preds, y_vals) if pred == label) / len(y_vals)
        if acc > best_acc:
            best_acc = acc
            best_thresholds = [threshold]
        elif acc == best_acc:
            best_thresholds.append(threshold)

    return float(median(best_thresholds))


def measure_alpha_t_against_temperature(
    examples: Sequence[EvaluationExample],
    energy_scores: Sequence[float],
    energy_threshold: float,
) -> AlphaTMeasurement:
    """Measure alpha_t as Carnot-vs-temperature verdict disagreement.

    This is the Exp1077 method: Carnot contributes exogenous signal on examples
    where its verifier changes the selection relative to the verifier-free
    length-percentile temperature baseline.

    Spec: REQ-FR11-1130.
    """

    if len(examples) != len(energy_scores):
        raise ValueError(
            f"examples and energy_scores must have same length, got "
            f"{len(examples)} vs {len(energy_scores)}"
        )
    if not examples:
        return AlphaTMeasurement(0.0, 0, 0, [], [], [])

    verifier_verdicts = [
        "correct" if float(energy) <= energy_threshold else "incorrect" for energy in energy_scores
    ]
    lengths = sorted(len(ex.response) for ex in examples)
    median_len = lengths[len(lengths) // 2]
    temperature_verdicts = [
        "correct" if len(ex.response) >= median_len else "incorrect" for ex in examples
    ]

    disagreement_ids = [
        ex.example_id
        for ex, verifier, temp in zip(examples, verifier_verdicts, temperature_verdicts)
        if verifier != temp
    ]
    n_total = len(examples)
    n_disagreements = len(disagreement_ids)
    return AlphaTMeasurement(
        alpha_t=n_disagreements / n_total,
        n_total=n_total,
        n_disagreements=n_disagreements,
        disagreement_ids=disagreement_ids,
        verifier_verdicts=verifier_verdicts,
        temperature_verdicts=temperature_verdicts,
    )


def load_cached_sota_examples(path: Path, n_examples: int) -> list[EvaluationExample]:
    """Load cached Qwen3.6-35B SOTA rows from an FR-11 JSONL artifact.

    Exp 1074 and Exp 1077 used slightly different row schemas, so this loader
    accepts both ``question``/``response``/``correct`` and
    ``prompt``/``completion``/``is_correct``.  Only Qwen3.6/35B rows are used
    because the Exp 1130 comparison is against the SOTA-tier baseline.

    Spec: SCENARIO-FR11-1131.
    """

    if n_examples <= 0:
        return []
    if not path.exists():
        return []

    examples: list[EvaluationExample] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue

        model = str(row.get("model") or row.get("model_name") or "")
        if "Qwen3.6" not in model and "35B-A3B" not in model:
            continue

        question = str(row.get("question") or row.get("prompt") or "")
        response = str(row.get("response") or row.get("completion") or "")
        if not question or not response:
            continue

        if "correct" in row:
            label_raw = row.get("correct")
        elif "is_correct" in row:
            label_raw = row.get("is_correct")
        else:
            continue
        label = 1 if bool(label_raw) else 0

        examples.append(
            EvaluationExample(
                example_id=str(row.get("question_id") or f"cached_{len(examples):03d}"),
                question=question,
                response=response,
                correct_answer=row.get("correct_answer", row.get("answer", "")),
                label=label,
            )
        )

    return examples[-n_examples:]


def build_exp1130_artifact(
    *,
    alpha_t_post_retrain: float,
    verifier_auroc_used: float,
    n_evaluation_examples: int,
    inference_mode: str,
    measurement_complete: bool,
    fr11_logged: bool,
    verifier_ground_truth_corr: float,
    thinkprm_ground_truth_corr: float,
    alpha_t_method: str,
    score_summary: dict[str, Any],
    examples_path: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the Exp 1130 result artifact with the required schema fields."""

    alpha_t_post_retrain = float(alpha_t_post_retrain)
    improved = bool(measurement_complete and alpha_t_post_retrain > ALPHA_T_PRIOR)
    if not measurement_complete or n_evaluation_examples <= 0:
        honest_verdict = "measurement_incomplete"
    elif alpha_t_post_retrain > ALPHA_T_PRIOR:
        honest_verdict = "alpha_t_improved"
    elif math.isclose(alpha_t_post_retrain, ALPHA_T_PRIOR, rel_tol=0.0, abs_tol=1e-12):
        honest_verdict = "alpha_t_unchanged"
    else:
        honest_verdict = "alpha_t_degraded"

    artifact: dict[str, Any] = {
        "experiment": 1130,
        "title": "Zenil alpha_t post-retrain energy verifier measurement",
        "schema_version": "1.0",
        "run_date": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "spec_trace": "REQ-FR11-1130",
        "alpha_t_prior": ALPHA_T_PRIOR,
        "alpha_t_post_retrain": round(alpha_t_post_retrain, 6),
        "alpha_t_improved": improved,
        "verifier_auroc_used": round(float(verifier_auroc_used), 6),
        "n_evaluation_examples": int(n_evaluation_examples),
        "inference_mode": inference_mode,
        "fr11_self_learning_data_point_logged": bool(fr11_logged),
        "zenil_alpha_t_post_retrain_measured": True,
        "honest_verdict": honest_verdict,
        "alpha_t_method": alpha_t_method,
        "verifier_ground_truth_corr": round(float(verifier_ground_truth_corr), 6),
        "thinkprm_ground_truth_corr": round(float(thinkprm_ground_truth_corr), 6),
        "score_summary": score_summary,
        "examples_path": examples_path,
        "ops_doc_reconciliation_deferred": True,
    }
    if extra:
        artifact.update(extra)
    return artifact


def summarize_scores(
    energy_scores: Sequence[float],
    labels: Sequence[int],
    thinkprm_scores: Sequence[float],
) -> dict[str, Any]:
    """Return compact score diagnostics for the Exp 1130 artifact."""

    if not energy_scores:
        return {}
    correct = [float(e) for e, label in zip(energy_scores, labels) if int(label) == 1]
    incorrect = [float(e) for e, label in zip(energy_scores, labels) if int(label) == 0]
    return {
        "mean_energy": round(sum(float(e) for e in energy_scores) / len(energy_scores), 6),
        "mean_correct_energy": round(sum(correct) / len(correct), 6) if correct else None,
        "mean_incorrect_energy": round(sum(incorrect) / len(incorrect), 6) if incorrect else None,
        "mean_thinkprm_score": (
            round(sum(float(s) for s in thinkprm_scores) / len(thinkprm_scores), 6)
            if thinkprm_scores
            else None
        ),
        "n_correct_ground_truth": int(sum(1 for label in labels if int(label) == 1)),
        "n_incorrect_ground_truth": int(sum(1 for label in labels if int(label) == 0)),
    }
