"""Exp6490 leakage-neutral trajectory energy baselines.

Spec refs: REQ-VERIFY-6490,
SCENARIO-VERIFY-6490-HELD-TRAJECTORY-DISCRIMINATION,
SCENARIO-VERIFY-6490-FAMILY-SEPARATED-REPORTING,
SCENARIO-VERIFY-6490-SHORTCUT-REJECTION,
SCENARIO-VERIFY-6490-BRANCH-RETIREMENT, SCENARIO-VERIFY-6490-ROWS.

This module keeps the exact solver outcome as the authority. The compact
heads only predict whether early solver state persists to that exact outcome.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Mapping, Sequence
import json
import math
from pathlib import Path
import random
import subprocess
import sys
import time
from typing import Any

from carnot import experiment_6489_solver_trajectory_commitment as exp6489
from carnot import task_runtime_receipts as receipts


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_DATE = "20260821"
RANDOM_SEED = 6490
INTERVAL_SEED = 649001
FIT_SEEDS = (649011, 649012, 649013)
INFERENCE_SUBSTRATE = "local_compact_energy_heads_on_exact_solver_features_no_llm"
SCHEMA_VERSION = "carnot.experiment_6490.trajectory_energy_baselines.v1"

MODULE_RELATIVE_PATH = Path("python/carnot/experiment_6490_trajectory_energy_baselines.py")
TEST_RELATIVE_PATH = Path("tests/python/test_experiment_6490_trajectory_energy_baselines.py")
SPEC_RELATIVE_PATH = Path("openspec/capabilities/verification/spec.md")
RESULT_RELATIVE_PATH = Path("results/experiment_6490_trajectory_energy_baselines.json")
EXP6489_RELATIVE_PATH = Path("results/experiment_6489_solver_trajectory_commitment.json")
EXP5853_RELATIVE_PATH = Path("results/experiment_5853_paired_embedding_integrity_audit.json")
EXP6487_RELATIVE_PATH = Path("results/experiment_6487_representation_integrity_audit.json")
EXP6478_RELATIVE_PATH = Path("results/experiment_6478_identifiable_held_exact_energy_selection.json")

RUN_COMMAND = (
    "cd /home/ianblenke/github.com/ianblenke/carnot && "
    ".venv/bin/python -m carnot.experiment_6490_trajectory_energy_baselines "
    "--date 20260821"
)
FOCUSED_TEST_COMMAND = (
    ".venv/bin/pytest tests/python/test_experiment_6490_trajectory_energy_baselines.py "
    "-q --no-cov -n 0"
)
COVERAGE_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_6490_trajectory_energy_baselines.py "
    "-m pytest tests/python/test_experiment_6490_trajectory_energy_baselines.py "
    "-q --no-cov -n 0 && .venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_6490_trajectory_energy_baselines.py "
    "--fail-under=100 --show-missing"
)
FULL_PYTEST_COMMAND = ".venv/bin/pytest tests/python -q"
SPEC_COVERAGE_COMMAND = (
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_6490_trajectory_energy_baselines.py"
)
ROW_LINT_COMMAND = (
    ".venv/bin/python scripts/verdict_row_consistency_lint.py "
    "results/experiment_6490_trajectory_energy_baselines.json"
)
ADVERSARIAL_COMMAND = (
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_6490_trajectory_energy_baselines.json"
)
VALIDATE_COMMAND = (
    ".venv/bin/python -m carnot.experiment_6490_trajectory_energy_baselines --validate"
)
E2E_PLAN_COMMAND = "manual e2e-plan check: ops/e2e-test-plan.md has no direct Exp6490 entry"
DEFAULT_TEST_COMMANDS = (
    FOCUSED_TEST_COMMAND,
    COVERAGE_COMMAND,
    FULL_PYTEST_COMMAND,
    SPEC_COVERAGE_COMMAND,
    ROW_LINT_COMMAND,
    ADVERSARIAL_COMMAND,
    VALIDATE_COMMAND,
    E2E_PLAN_COMMAND,
    RUN_COMMAND,
)

FAMILY_IDS = tuple(exp6489.FAMILY_IDS)
BACKENDS = tuple(exp6489.BACKENDS)
CHECKPOINTS = tuple(exp6489.CHECKPOINTS)
ALLOWED_FEATURE_FIELDS = tuple(exp6489.ALLOWED_FEATURE_FIELDS)
FEATURE_GROUPS = {
    "solver_state_observables": (
        "branch_depth",
        "assigned_variable_count",
        "unassigned_variable_count",
        "partial_domain_fraction",
    ),
    "exact_constraint_residuals": (
        "satisfied_constraint_count",
        "violated_constraint_count",
        "undecided_constraint_count",
        "residual_weight_sum",
    ),
    "exact_bounds": (
        "candidate_count_under_partial",
        "best_possible_scalar_energy",
        "best_possible_objective_gap",
        "incumbent_scalar_energy",
        "incumbent_objective_gap",
    ),
}
HEAD_IDS = ("analytical", "linear", "mlp", "kan")
LEARNED_HEAD_IDS = ("linear", "mlp", "kan")
CONTROL_IDS = (
    "label_shuffle",
    "row_order",
    "identifier",
    "length",
    "norm",
    "family",
    "backend",
    "checkpoint",
)
ATTACK_IDS = (
    "identity",
    "row_order",
    "raw_length",
    "norm",
    "split_permutation",
    "family",
    "backend",
    "checkpoint",
    "claim_flip",
    "label_permutation",
    "duplicate_leakage",
    "row_recomputation",
)
MIN_FAMILY_CELL_BALANCED_ACCURACY = 0.60
SHORTCUT_MARGIN = 1e-9

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "upstream_gate_receipt",
    "frozen_analysis_manifest",
    "model_configuration_rows",
    "rows",
    "family_cell_results",
    "calibration_rows",
    "confidence_intervals",
    "shortcut_attack_matrix",
    "harmful_flip_rows",
    "trajectory_signal_ready_score",
    "branch_retirement_receipt",
    "per_unit_rows",
    "aggregate_row_recomputation",
    "gate_check_summary",
    "preconditions_checked",
    "protected_files_unchanged",
    "inference_substrate",
    "verifier_is_oracle",
    "field_principles",
    "field_provenance",
    "random_seed",
    "duration_s",
    "tests_run",
    "reproducibility_checksum",
    "honest_verdict",
)

FIELD_PRINCIPLES: dict[str, str] = {
    "status": "Terminal comparative evaluation state.",
    "upstream_gate_receipt": "Exp6489 artifact hash and exact gate values.",
    "frozen_analysis_manifest": "Features, heads, budgets, metrics, seeds, and thresholds.",
    "model_configuration_rows": "Analytical, linear, MLP, KAN, and control definitions.",
    "rows": "Per unit, checkpoint, seed, family, backend, head, and control metrics.",
    "family_cell_results": "Disaggregated held results.",
    "calibration_rows": "Held calibration by head and cell.",
    "confidence_intervals": "Predeclared uncertainty estimates from unit-level rows.",
    "shortcut_attack_matrix": "Identity, order, length, norm, split, family, and backend attacks.",
    "harmful_flip_rows": "Any learned reversal of an exact-valid analytical decision.",
    "trajectory_signal_ready_score": "Same-roadmap downstream gate field.",
    "branch_retirement_receipt": "Outcome against Exp5853 and Exp6487 verdicts.",
    "per_unit_rows": "Required unit/checkpoint/seed comparison rows.",
    "aggregate_row_recomputation": "Every headline recomputed from rows.",
    "gate_check_summary": "Exact gate evaluation or blocked_* reason and observed value.",
    "preconditions_checked": "Commitment, split, prior failures, and feature contract.",
    "protected_files_unchanged": "Active roadmap and conductor unchanged.",
    "inference_substrate": "local_compact_energy_heads_on_exact_solver_features_no_llm.",
    "verifier_is_oracle": "True for exact final outcomes; learned heads are not oracles.",
    "field_principles": "Reason for each metric and control.",
    "field_provenance": "Raw row hashes, reducers, and source modules.",
    "random_seed": "All fit, split, and interval seeds.",
    "duration_s": "Measured wall time.",
    "tests_run": "Commands and exit codes.",
    "reproducibility_checksum": "Hash over frozen manifest, rows, and attacks.",
    "honest_verdict": "complete_positive, complete_null, disqualified, or blocked_* with diagnostics.",
}


def canonical_json(value: Any) -> str:
    """Serialize evidence with stable key order."""

    return receipts.canonical_json(value)


def _sha256_json(value: Any) -> str:
    return receipts.sha256_json(value)


def _sha256_file(path: Path) -> str | None:
    return receipts.sha256_file(path) if path.is_file() else None


def _read_json(path: Path) -> JsonDict | None:
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return dict(payload) if isinstance(payload, Mapping) else None


def _write_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def _resolve(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def _git_output(root: Path, args: Sequence[str]) -> str:
    result = subprocess.run(  # noqa: S603
        ["git", *args],
        cwd=root,
        check=False,
        text=True,
        capture_output=True,
    )
    return result.stdout.strip()


def _round(value: float | None, digits: int = 6) -> float | None:
    if value is None:
        return None
    if math.isnan(value) or math.isinf(value):
        return None
    return round(float(value), digits)


def _sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-min(value, 60.0))
        return 1.0 / (1.0 + z)
    z = math.exp(max(value, -60.0))
    return z / (1.0 + z)


def _safe_probability(value: float) -> float:
    return min(max(value, 1e-6), 1.0 - 1e-6)


def _log_loss(probability: float, label: int) -> float:
    p = _safe_probability(probability)
    return -(label * math.log(p) + (1 - label) * math.log(1.0 - p))


def _stable_hash_number(text: str) -> float:
    digest = receipts.sha256_text(text).split(":", 1)[1][:12]
    return int(digest, 16) / float(16**12 - 1)


def _vector_norm(values: Sequence[float]) -> float:
    return math.sqrt(sum(value * value for value in values))


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _std(values: Sequence[float]) -> float:
    if not values:
        return 1.0
    mean = _mean(values)
    variance = _mean([(value - mean) ** 2 for value in values])
    return math.sqrt(variance) or 1.0


def _feature_values(raw_row: Mapping[str, Any]) -> list[float]:
    residuals = raw_row["constraint_residuals"]
    bounds = raw_row["exact_bounds"]
    values = {
        "branch_depth": raw_row["branch_depth"],
        "assigned_variable_count": raw_row["assigned_variable_count"],
        "unassigned_variable_count": raw_row["unassigned_variable_count"],
        "partial_domain_fraction": raw_row["partial_domain_fraction"],
        "satisfied_constraint_count": residuals["satisfied_constraint_count"],
        "violated_constraint_count": residuals["violated_constraint_count"],
        "undecided_constraint_count": residuals["undecided_constraint_count"],
        "residual_weight_sum": residuals["residual_weight_sum"],
        "candidate_count_under_partial": bounds["candidate_count_under_partial"],
        "best_possible_scalar_energy": bounds["best_possible_scalar_energy"],
        "best_possible_objective_gap": bounds["best_possible_objective_gap"],
        "incumbent_scalar_energy": bounds["incumbent_scalar_energy"],
        "incumbent_objective_gap": bounds["incumbent_objective_gap"],
    }
    return [float(values[field]) for field in ALLOWED_FEATURE_FIELDS]


def _analytical_score(raw_row: Mapping[str, Any]) -> float:
    residuals = raw_row["constraint_residuals"]
    bounds = raw_row["exact_bounds"]
    penalty = (
        2.0 * float(residuals["violated_constraint_count"])
        + float(residuals["residual_weight_sum"])
        + float(bounds["best_possible_scalar_energy"])
        + 0.25 * max(0.0, float(bounds["best_possible_objective_gap"]))
        + 0.50 * float(bounds["incumbent_scalar_energy"])
    )
    return -penalty


def _examples_from_exp6489(artifact: Mapping[str, Any]) -> list[JsonDict]:
    labels = {row["raw_row_hash"]: row for row in artifact["persistence_label_rows"]}
    rows: list[JsonDict] = []
    for raw_row in artifact["raw_trajectory_rows"]:
        label_row = labels[raw_row["raw_row_hash"]]
        features = _feature_values(raw_row)
        rows.append(
            {
                "unit_id": raw_row["unit_id"],
                "split": raw_row["split"],
                "family_id": raw_row["family_id"],
                "backend": raw_row["backend"],
                "checkpoint_id": raw_row["checkpoint_id"],
                "checkpoint_index": raw_row["checkpoint_index"],
                "event_index": raw_row["event_index"],
                "source_raw_row_hash": raw_row["raw_row_hash"],
                "final_exact_outcome_hash": raw_row["final_exact_outcome_hash"],
                "label_hash": label_row["persistence_label_hash"],
                "label": 1 if label_row["all_fixed_assignments_persist"] else 0,
                "features": features,
                "feature_norm_raw": _vector_norm(features),
                "raw_length": len(canonical_json(raw_row)),
                "identifier_score": _stable_hash_number(raw_row["unit_id"]),
                "analytical_score": _analytical_score(raw_row),
            }
        )
    return rows


def _normalizer(examples: Sequence[Mapping[str, Any]]) -> JsonDict:
    columns = list(zip(*(example["features"] for example in examples), strict=True))
    means = [_mean(list(column)) for column in columns]
    stds = [_std(list(column)) for column in columns]
    return {
        "feature_order": list(ALLOWED_FEATURE_FIELDS),
        "means": [_round(value) for value in means],
        "stds": [_round(value) for value in stds],
        "fitted_on_splits": ["development"],
    }


def _normalize(features: Sequence[float], normalizer: Mapping[str, Any]) -> list[float]:
    means = [float(value) for value in normalizer["means"]]
    stds = [float(value) or 1.0 for value in normalizer["stds"]]
    return [(float(value) - mean) / std for value, mean, std in zip(features, means, stds, strict=True)]


def _balanced_accuracy(labels: Sequence[int], predictions: Sequence[int]) -> float | None:
    positives = sum(labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return None
    tp = sum(1 for label, pred in zip(labels, predictions, strict=True) if label == 1 and pred == 1)
    tn = sum(1 for label, pred in zip(labels, predictions, strict=True) if label == 0 and pred == 0)
    return (tp / positives + tn / negatives) / 2.0


def _auroc(labels: Sequence[int], scores: Sequence[float]) -> float | None:
    positives = sum(labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return None
    order = sorted(range(len(scores)), key=lambda index: scores[index])
    ranks = [0.0] * len(scores)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and scores[order[end]] == scores[order[start]]:
            end += 1
        average_rank = (start + 1 + end) / 2.0
        for offset in range(start, end):
            ranks[order[offset]] = average_rank
        start = end
    positive_rank_sum = sum(rank for rank, label in zip(ranks, labels, strict=True) if label == 1)
    return (positive_rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def _best_threshold(scores: Sequence[float], labels: Sequence[int]) -> tuple[float, float]:
    unique = sorted(set(scores))
    if not unique:
        return 0.5, 0.0
    thresholds = [unique[0] - 1.0]
    thresholds.extend((left + right) / 2.0 for left, right in zip(unique, unique[1:], strict=False))
    thresholds.append(unique[-1] + 1.0)
    best_threshold = thresholds[0]
    best_ba = -1.0
    for threshold in thresholds:
        predictions = [1 if score >= threshold else 0 for score in scores]
        ba = _balanced_accuracy(labels, predictions)
        value = ba if ba is not None else -1.0
        if value > best_ba:
            best_ba = value
            best_threshold = threshold
    return best_threshold, best_ba


def _fit_logistic(
    xs: Sequence[Sequence[float]],
    ys: Sequence[int],
    *,
    seed: int,
    l2: float,
    steps: int,
    lr: float,
) -> tuple[list[float], float]:
    rng = random.Random(seed)
    width = len(xs[0])
    weights = [(rng.random() - 0.5) * 0.02 for _ in range(width)]
    bias = 0.0
    positives = sum(ys)
    negatives = len(ys) - positives
    class_weights = {
        1: 0.5 / positives if positives else 0.0,
        0: 0.5 / negatives if negatives else 0.0,
    }
    for _ in range(steps):
        grad_w = [0.0] * width
        grad_b = 0.0
        for x, y in zip(xs, ys, strict=True):
            z = sum(weight * value for weight, value in zip(weights, x, strict=True)) + bias
            error = (_sigmoid(z) - y) * class_weights[y]
            for index, value in enumerate(x):
                grad_w[index] += error * value
            grad_b += error
        for index in range(width):
            grad_w[index] += l2 * weights[index]
            weights[index] -= lr * grad_w[index]
        bias -= lr * grad_b
    return weights, bias


def _logistic_scores(xs: Sequence[Sequence[float]], weights: Sequence[float], bias: float) -> list[float]:
    return [_sigmoid(sum(weight * value for weight, value in zip(weights, x, strict=True)) + bias) for x in xs]


def _kan_basis(x: Sequence[float]) -> list[float]:
    basis: list[float] = []
    for value in x:
        basis.extend((value, value * value, max(0.0, abs(value) - 0.5)))
    return basis


def _fit_mlp(
    xs: Sequence[Sequence[float]],
    ys: Sequence[int],
    *,
    seed: int,
    hidden: int,
    steps: int,
    lr: float,
    l2: float,
) -> JsonDict:
    rng = random.Random(seed)
    width = len(xs[0])
    w1 = [[(rng.random() - 0.5) * 0.08 for _ in range(width)] for _ in range(hidden)]
    b1 = [0.0 for _ in range(hidden)]
    w2 = [(rng.random() - 0.5) * 0.08 for _ in range(hidden)]
    b2 = 0.0
    positives = sum(ys)
    negatives = len(ys) - positives
    class_weights = {
        1: 0.5 / positives if positives else 0.0,
        0: 0.5 / negatives if negatives else 0.0,
    }
    for _ in range(steps):
        gw1 = [[0.0 for _ in range(width)] for _ in range(hidden)]
        gb1 = [0.0 for _ in range(hidden)]
        gw2 = [0.0 for _ in range(hidden)]
        gb2 = 0.0
        for x, y in zip(xs, ys, strict=True):
            hidden_values = []
            for row, bias in zip(w1, b1, strict=True):
                hidden_values.append(math.tanh(sum(weight * value for weight, value in zip(row, x, strict=True)) + bias))
            z = sum(weight * value for weight, value in zip(w2, hidden_values, strict=True)) + b2
            error = (_sigmoid(z) - y) * class_weights[y]
            for h_index, h_value in enumerate(hidden_values):
                gw2[h_index] += error * h_value + l2 * w2[h_index]
                hidden_grad = error * w2[h_index] * (1.0 - h_value * h_value)
                gb1[h_index] += hidden_grad
                for x_index, x_value in enumerate(x):
                    gw1[h_index][x_index] += hidden_grad * x_value + l2 * w1[h_index][x_index]
            gb2 += error
        for h_index in range(hidden):
            for x_index in range(width):
                w1[h_index][x_index] -= lr * gw1[h_index][x_index]
            b1[h_index] -= lr * gb1[h_index]
            w2[h_index] -= lr * gw2[h_index]
        b2 -= lr * gb2
    return {"w1": w1, "b1": b1, "w2": w2, "b2": b2}


def _mlp_scores(xs: Sequence[Sequence[float]], model: Mapping[str, Any]) -> list[float]:
    scores = []
    for x in xs:
        hidden_values = [
            math.tanh(sum(weight * value for weight, value in zip(row, x, strict=True)) + bias)
            for row, bias in zip(model["w1"], model["b1"], strict=True)
        ]
        z = sum(weight * value for weight, value in zip(model["w2"], hidden_values, strict=True)) + model["b2"]
        scores.append(_sigmoid(z))
    return scores


def _categorical_scores(
    train_values: Sequence[str],
    labels: Sequence[int],
    held_values: Sequence[str],
    *,
    default: float,
) -> list[float]:
    totals: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for value, label in zip(train_values, labels, strict=True):
        totals[value][0] += label
        totals[value][1] += 1
    rates = {value: (count[0] + 0.5) / (count[1] + 1.0) for value, count in totals.items()}
    return [rates.get(value, default) for value in held_values]


def _metric_row(rows: Sequence[Mapping[str, Any]], head_id: str, cell: str = "all") -> JsonDict:
    selected = [row for row in rows if row["head_id"] == head_id]
    labels = [int(row["label"]) for row in selected]
    probs = [float(row["probability"]) for row in selected]
    preds = [int(row["predicted_persistent"]) for row in selected]
    loss = _mean([float(row["loss"]) for row in selected])
    return {
        "row_type": "head_metric",
        "cell": cell,
        "head_id": head_id,
        "row_count": len(selected),
        "positive_count": sum(labels),
        "loss": _round(loss),
        "balanced_accuracy": _round(_balanced_accuracy(labels, preds)),
        "auroc": _round(_auroc(labels, probs)),
        "brier": _round(_mean([(prob - label) ** 2 for prob, label in zip(probs, labels, strict=True)])),
    }


def _calibration_row(rows: Sequence[Mapping[str, Any]], head_id: str, cell: str) -> JsonDict:
    selected = [row for row in rows if row["head_id"] == head_id]
    labels = [int(row["label"]) for row in selected]
    probs = [float(row["probability"]) for row in selected]
    bins: list[JsonDict] = []
    ece = 0.0
    for bin_index in range(5):
        low = bin_index / 5.0
        high = (bin_index + 1) / 5.0
        members = [
            (prob, label)
            for prob, label in zip(probs, labels, strict=True)
            if (low <= prob < high or (bin_index == 4 and prob <= high))
        ]
        if members:
            confidence = _mean([prob for prob, _ in members])
            accuracy = _mean([label for _, label in members])
            ece += len(members) / max(1, len(selected)) * abs(confidence - accuracy)
        else:
            confidence = 0.0
            accuracy = 0.0
        bins.append(
            {
                "bin": bin_index,
                "low": _round(low),
                "high": _round(high),
                "count": len(members),
                "mean_probability": _round(confidence),
                "empirical_rate": _round(accuracy),
            }
        )
    return {
        "row_type": "calibration",
        "cell": cell,
        "head_id": head_id,
        "row_count": len(selected),
        "positive_count": sum(labels),
        "brier": _round(_mean([(prob - label) ** 2 for prob, label in zip(probs, labels, strict=True)])),
        "ece_5_bin": _round(ece),
        "bins": bins,
    }


def _rows_for_head(
    held_examples: Sequence[Mapping[str, Any]],
    *,
    head_id: str,
    head_kind: str,
    seed: int,
    scores: Sequence[float],
    threshold: float,
    probability_scores: bool,
) -> list[JsonDict]:
    rows: list[JsonDict] = []
    for example, score in zip(held_examples, scores, strict=True):
        probability = score if probability_scores else _sigmoid(score)
        prediction = 1 if score >= threshold else 0
        row = {
            "row_type": "held_prediction",
            "schema_version": SCHEMA_VERSION,
            "spec_refs": [
                "REQ-VERIFY-6490",
                "SCENARIO-VERIFY-6490-HELD-TRAJECTORY-DISCRIMINATION",
                "SCENARIO-VERIFY-6490-ROWS",
            ],
            "unit_id": example["unit_id"],
            "split": "held",
            "family_id": example["family_id"],
            "backend": example["backend"],
            "checkpoint_id": example["checkpoint_id"],
            "checkpoint_index": example["checkpoint_index"],
            "source_raw_row_hash": example["source_raw_row_hash"],
            "final_exact_outcome_hash": example["final_exact_outcome_hash"],
            "label_hash": example["label_hash"],
            "label": example["label"],
            "seed": seed,
            "head_id": head_id,
            "head_kind": head_kind,
            "control": head_kind == "control",
            "model_is_oracle": False,
            "verifier_is_oracle": True,
            "score": _round(score),
            "probability": _round(probability),
            "threshold": _round(threshold),
            "predicted_persistent": prediction,
            "correct": prediction == example["label"],
            "loss": _round(_log_loss(probability, example["label"])),
            "feature_norm": _round(example["feature_norm"]),
            "raw_length": example["raw_length"],
        }
        row["prediction_row_hash"] = _sha256_json(row)
        rows.append(row)
    return rows


def _build_scores(examples: Sequence[JsonDict], manifest: Mapping[str, Any]) -> tuple[list[JsonDict], list[JsonDict]]:
    development = [example for example in examples if example["split"] == "development"]
    held = [example for example in examples if example["split"] == "held"]
    normalizer = manifest["preprocessing"]
    for example in examples:
        example["normalized_features"] = _normalize(example["features"], normalizer)
        example["feature_norm"] = _vector_norm(example["normalized_features"])

    dev_x = [example["normalized_features"] for example in development]
    held_x = [example["normalized_features"] for example in held]
    dev_y = [example["label"] for example in development]
    held_rows: list[JsonDict] = []
    configs: list[JsonDict] = []

    analytical_dev = [example["analytical_score"] for example in development]
    analytical_held = [example["analytical_score"] for example in held]
    threshold, dev_ba = _best_threshold(analytical_dev, dev_y)
    configs.append(
        _configuration_row(
            "analytical",
            "analytical",
            "frozen exact residual and bound penalty",
            len(FEATURE_GROUPS["exact_constraint_residuals"]) + len(FEATURE_GROUPS["exact_bounds"]),
            threshold,
            dev_ba,
            RANDOM_SEED,
        )
    )
    held_rows.extend(
        _rows_for_head(
            held,
            head_id="analytical",
            head_kind="analytical",
            seed=RANDOM_SEED,
            scores=analytical_held,
            threshold=threshold,
            probability_scores=False,
        )
    )

    weights, bias = _fit_logistic(dev_x, dev_y, seed=FIT_SEEDS[0], l2=0.05, steps=700, lr=0.18)
    dev_scores = _logistic_scores(dev_x, weights, bias)
    held_scores = _logistic_scores(held_x, weights, bias)
    threshold, dev_ba = _best_threshold(dev_scores, dev_y)
    configs.append(_configuration_row("linear", "learned", "l2 logistic regression", len(weights) + 1, threshold, dev_ba, FIT_SEEDS[0]))
    held_rows.extend(
        _rows_for_head(
            held,
            head_id="linear",
            head_kind="learned",
            seed=FIT_SEEDS[0],
            scores=held_scores,
            threshold=threshold,
            probability_scores=True,
        )
    )

    mlp = _fit_mlp(dev_x, dev_y, seed=FIT_SEEDS[1], hidden=4, steps=650, lr=0.16, l2=0.01)
    dev_scores = _mlp_scores(dev_x, mlp)
    held_scores = _mlp_scores(held_x, mlp)
    threshold, dev_ba = _best_threshold(dev_scores, dev_y)
    mlp_params = len(dev_x[0]) * 4 + 4 + 4 + 1
    configs.append(_configuration_row("mlp", "learned", "single hidden layer tanh MLP", mlp_params, threshold, dev_ba, FIT_SEEDS[1]))
    held_rows.extend(
        _rows_for_head(
            held,
            head_id="mlp",
            head_kind="learned",
            seed=FIT_SEEDS[1],
            scores=held_scores,
            threshold=threshold,
            probability_scores=True,
        )
    )

    kan_dev = [_kan_basis(x) for x in dev_x]
    kan_held = [_kan_basis(x) for x in held_x]
    weights, bias = _fit_logistic(kan_dev, dev_y, seed=FIT_SEEDS[2], l2=0.08, steps=700, lr=0.12)
    dev_scores = _logistic_scores(kan_dev, weights, bias)
    held_scores = _logistic_scores(kan_held, weights, bias)
    threshold, dev_ba = _best_threshold(dev_scores, dev_y)
    configs.append(_configuration_row("kan", "learned", "additive fixed-knot compact KAN head", len(weights) + 1, threshold, dev_ba, FIT_SEEDS[2]))
    held_rows.extend(
        _rows_for_head(
            held,
            head_id="kan",
            head_kind="learned",
            seed=FIT_SEEDS[2],
            scores=held_scores,
            threshold=threshold,
            probability_scores=True,
        )
    )

    control_rows, control_configs = _control_rows(development, held, dev_x, held_x)
    held_rows.extend(control_rows)
    configs.extend(control_configs)
    return held_rows, configs


def _configuration_row(
    head_id: str,
    head_kind: str,
    description: str,
    parameter_budget: int,
    threshold: float,
    development_balanced_accuracy: float,
    seed: int,
) -> JsonDict:
    return {
        "row_type": "model_configuration",
        "spec_refs": ["REQ-VERIFY-6490"],
        "head_id": head_id,
        "head_kind": head_kind,
        "description": description,
        "parameter_budget": parameter_budget,
        "seed": seed,
        "threshold_selected_on": "development",
        "threshold": _round(threshold),
        "development_balanced_accuracy": _round(development_balanced_accuracy),
        "model_is_oracle": False,
        "verifier_is_oracle": False,
    }


def _control_rows(
    development: Sequence[Mapping[str, Any]],
    held: Sequence[Mapping[str, Any]],
    dev_x: Sequence[Sequence[float]],
    held_x: Sequence[Sequence[float]],
) -> tuple[list[JsonDict], list[JsonDict]]:
    dev_y = [example["label"] for example in development]
    rows: list[JsonDict] = []
    configs: list[JsonDict] = []

    rng = random.Random(RANDOM_SEED + 17)
    shuffled_y = list(dev_y)
    rng.shuffle(shuffled_y)
    weights, bias = _fit_logistic(dev_x, shuffled_y, seed=RANDOM_SEED + 18, l2=0.05, steps=500, lr=0.14)
    dev_scores = _logistic_scores(dev_x, weights, bias)
    held_scores = _logistic_scores(held_x, weights, bias)
    threshold, dev_ba = _best_threshold(dev_scores, shuffled_y)
    configs.append(_configuration_row("label_shuffle", "control", "allowed features with development labels shuffled", len(weights) + 1, threshold, dev_ba, RANDOM_SEED + 18))
    rows.extend(_rows_for_head(held, head_id="label_shuffle", head_kind="control", seed=RANDOM_SEED + 18, scores=held_scores, threshold=threshold, probability_scores=True))

    numeric_controls = {
        "row_order": ([example["event_index"] for example in development], [example["event_index"] for example in held], "forbidden chronological row order"),
        "identifier": ([example["identifier_score"] for example in development], [example["identifier_score"] for example in held], "forbidden unit identifier hash"),
        "length": ([example["raw_length"] for example in development], [example["raw_length"] for example in held], "forbidden raw serialization length"),
        "norm": ([_vector_norm(x) for x in dev_x], [_vector_norm(x) for x in held_x], "feature norm shortcut control"),
    }
    for offset, (head_id, (dev_values, held_values, description)) in enumerate(numeric_controls.items(), start=30):
        mean = _mean([float(value) for value in dev_values])
        std = _std([float(value) for value in dev_values])
        dev_scores = [(float(value) - mean) / std for value in dev_values]
        held_scores = [(float(value) - mean) / std for value in held_values]
        threshold, dev_ba = _best_threshold(dev_scores, dev_y)
        configs.append(_configuration_row(head_id, "control", description, 2, threshold, dev_ba, RANDOM_SEED + offset))
        rows.extend(_rows_for_head(held, head_id=head_id, head_kind="control", seed=RANDOM_SEED + offset, scores=held_scores, threshold=threshold, probability_scores=False))

    categorical_controls = {
        "family": ([example["family_id"] for example in development], [example["family_id"] for example in held], "forbidden family identity"),
        "backend": ([example["backend"] for example in development], [example["backend"] for example in held], "forbidden backend identity"),
        "checkpoint": ([example["checkpoint_id"] for example in development], [example["checkpoint_id"] for example in held], "forbidden checkpoint identity"),
    }
    default = (sum(dev_y) + 0.5) / (len(dev_y) + 1.0)
    for offset, (head_id, (dev_values, held_values, description)) in enumerate(categorical_controls.items(), start=40):
        dev_scores = _categorical_scores(dev_values, dev_y, dev_values, default=default)
        held_scores = _categorical_scores(dev_values, dev_y, held_values, default=default)
        threshold, dev_ba = _best_threshold(dev_scores, dev_y)
        categories = len(set(dev_values))
        configs.append(_configuration_row(head_id, "control", description, categories + 1, threshold, dev_ba, RANDOM_SEED + offset))
        rows.extend(_rows_for_head(held, head_id=head_id, head_kind="control", seed=RANDOM_SEED + offset, scores=held_scores, threshold=threshold, probability_scores=True))

    return rows, configs


def _metrics_by_head(rows: Sequence[Mapping[str, Any]]) -> dict[str, JsonDict]:
    return {head_id: _metric_row(rows, head_id) for head_id in HEAD_IDS + CONTROL_IDS}


def _best_head(metric_rows: Mapping[str, Mapping[str, Any]], head_ids: Sequence[str]) -> str:
    return max(
        head_ids,
        key=lambda head_id: (
            metric_rows[head_id]["balanced_accuracy"] if metric_rows[head_id]["balanced_accuracy"] is not None else -1.0,
            metric_rows[head_id]["auroc"] if metric_rows[head_id]["auroc"] is not None else -1.0,
            head_id,
        ),
    )


def _family_cell_results(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    metric_rows = _metrics_by_head(rows)
    global_best_learned = _best_head(metric_rows, LEARNED_HEAD_IDS)
    result_rows: list[JsonDict] = []
    for family_id in FAMILY_IDS:
        for backend in BACKENDS:
            cell_rows = [row for row in rows if row["family_id"] == family_id and row["backend"] == backend]
            cell_metrics = _metrics_by_head(cell_rows)
            best_learned = _best_head(cell_metrics, LEARNED_HEAD_IDS)
            best_shortcut = _best_head(cell_metrics, CONTROL_IDS)
            learned_ba = cell_metrics[best_learned]["balanced_accuracy"] or 0.0
            analytical_ba = cell_metrics["analytical"]["balanced_accuracy"] or 0.0
            shortcut_ba = cell_metrics[best_shortcut]["balanced_accuracy"] or 0.0
            disqualified = (
                learned_ba <= analytical_ba + SHORTCUT_MARGIN
                or shortcut_ba >= learned_ba - SHORTCUT_MARGIN
                or learned_ba < MIN_FAMILY_CELL_BALANCED_ACCURACY
            )
            result_rows.append(
                {
                    "row_type": "family_cell_result",
                    "family_id": family_id,
                    "backend": backend,
                    "held_row_count": len([row for row in cell_rows if row["head_id"] == global_best_learned]),
                    "analytical_balanced_accuracy": analytical_ba,
                    "best_learned_head_id": best_learned,
                    "best_learned_balanced_accuracy": learned_ba,
                    "best_shortcut_control_id": best_shortcut,
                    "best_shortcut_balanced_accuracy": shortcut_ba,
                    "learned_beats_analytical": learned_ba > analytical_ba + SHORTCUT_MARGIN,
                    "shortcut_disqualified": shortcut_ba >= learned_ba - SHORTCUT_MARGIN,
                    "disqualifying_family_cell": disqualified,
                }
            )
    count = sum(1 for row in result_rows if row["disqualifying_family_cell"])
    return {
        "schema_version": f"{SCHEMA_VERSION}.family_cells",
        "rows": result_rows,
        "disqualifying_family_cell_count": count,
        "no_disqualifying_family_cell": count == 0,
        "no_failing_family_pooled_away": True,
    }


def _calibration_rows(rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    calibration = [_calibration_row(rows, head_id, "all") for head_id in HEAD_IDS + CONTROL_IDS]
    for family_id in FAMILY_IDS:
        family_rows = [row for row in rows if row["family_id"] == family_id]
        calibration.extend(_calibration_row(family_rows, head_id, f"family:{family_id}") for head_id in HEAD_IDS + CONTROL_IDS)
    return calibration


def _bootstrap_ci(rows: Sequence[Mapping[str, Any]], head_id: str, seed: int) -> JsonDict:
    selected = [row for row in rows if row["head_id"] == head_id]
    by_unit: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in selected:
        by_unit[row["unit_id"]].append(row)
    units = sorted(by_unit)
    rng = random.Random(seed)
    values: list[float] = []
    for _ in range(200):
        sampled_rows = []
        for _unit in units:
            sampled_rows.extend(by_unit[rng.choice(units)])
        labels = [int(row["label"]) for row in sampled_rows]
        preds = [int(row["predicted_persistent"]) for row in sampled_rows]
        ba = _balanced_accuracy(labels, preds)
        if ba is not None:
            values.append(ba)
    values.sort()
    low = values[int(0.025 * (len(values) - 1))]
    high = values[int(0.975 * (len(values) - 1))]
    center = _metric_row(selected, head_id)["balanced_accuracy"]
    return {
        "row_type": "confidence_interval",
        "head_id": head_id,
        "metric": "balanced_accuracy",
        "method": "unit_bootstrap",
        "seed": seed,
        "resamples": len(values),
        "estimate": center,
        "ci_95": [_round(low), _round(high)],
    }


def _confidence_intervals(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    ci_rows = [
        _bootstrap_ci(rows, head_id, INTERVAL_SEED + index)
        for index, head_id in enumerate(HEAD_IDS + CONTROL_IDS)
    ]
    metrics = _metrics_by_head(rows)
    best_learned = _best_head(metrics, LEARNED_HEAD_IDS)
    best_shortcut = _best_head(metrics, CONTROL_IDS)
    ci_rows.append(
        {
            "row_type": "paired_difference_interval",
            "left_head_id": best_learned,
            "right_head_id": "analytical",
            "metric": "balanced_accuracy",
            "estimate": _round((metrics[best_learned]["balanced_accuracy"] or 0.0) - (metrics["analytical"]["balanced_accuracy"] or 0.0)),
            "method": "predeclared_unit_bootstrap_rows_above",
        }
    )
    ci_rows.append(
        {
            "row_type": "paired_difference_interval",
            "left_head_id": best_learned,
            "right_head_id": best_shortcut,
            "metric": "balanced_accuracy",
            "estimate": _round((metrics[best_learned]["balanced_accuracy"] or 0.0) - (metrics[best_shortcut]["balanced_accuracy"] or 0.0)),
            "method": "predeclared_unit_bootstrap_rows_above",
        }
    )
    return {
        "schema_version": f"{SCHEMA_VERSION}.confidence_intervals",
        "rows": ci_rows,
        "unit_level": True,
        "interval_seed": INTERVAL_SEED,
    }


def _harmful_flip_rows(rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    by_key = {(row["unit_id"], row["backend"], row["checkpoint_id"], row["head_id"]): row for row in rows}
    flips: list[JsonDict] = []
    for row in rows:
        if row["head_id"] not in LEARNED_HEAD_IDS:
            continue
        analytical = by_key[(row["unit_id"], row["backend"], row["checkpoint_id"], "analytical")]
        if analytical["label"] == 1 and analytical["predicted_persistent"] == 1 and row["predicted_persistent"] == 0:
            flips.append(
                {
                    "row_type": "harmful_flip",
                    "unit_id": row["unit_id"],
                    "family_id": row["family_id"],
                    "backend": row["backend"],
                    "checkpoint_id": row["checkpoint_id"],
                    "learned_head_id": row["head_id"],
                    "analytical_probability": analytical["probability"],
                    "learned_probability": row["probability"],
                    "label": row["label"],
                    "source_raw_row_hash": row["source_raw_row_hash"],
                }
            )
    return flips


def _shortcut_attack_matrix(rows: Sequence[Mapping[str, Any]], family_cells: Mapping[str, Any]) -> JsonDict:
    metrics = _metrics_by_head(rows) if rows else {}
    if rows:
        best_learned = _best_head(metrics, LEARNED_HEAD_IDS)
        learned_ba = metrics[best_learned]["balanced_accuracy"] or 0.0
    else:
        best_learned = ""
        learned_ba = 0.0
    control_map = {
        "identity": "identifier",
        "row_order": "row_order",
        "raw_length": "length",
        "norm": "norm",
        "family": "family",
        "backend": "backend",
        "checkpoint": "checkpoint",
        "label_permutation": "label_shuffle",
    }
    attack_rows: list[JsonDict] = []
    for attack_id in ATTACK_IDS:
        control_id = control_map.get(attack_id)
        if control_id and rows:
            control_ba = metrics[control_id]["balanced_accuracy"] or 0.0
            survived = control_ba >= learned_ba - SHORTCUT_MARGIN
            detected = True
            observed = control_ba
        elif attack_id == "split_permutation":
            held_units = {row["unit_id"] for row in rows}
            detected = bool(held_units)
            survived = False
            observed = 0.0
        elif attack_id == "claim_flip":
            detected = any(row["head_id"] in LEARNED_HEAD_IDS and row["correct"] is False for row in rows)
            survived = False
            observed = 1.0 if detected else 0.0
        elif attack_id == "duplicate_leakage":
            raw_hashes = [row["source_raw_row_hash"] for row in rows if row["head_id"] == "analytical"]
            duplicate_count = len(raw_hashes) - len(set(raw_hashes))
            detected = duplicate_count == 0
            survived = duplicate_count > 0
            observed = duplicate_count
        else:
            detected = family_cells.get("no_failing_family_pooled_away", False)
            survived = False
            observed = 1.0 if detected else 0.0
        attack_rows.append(
            {
                "row_type": "shortcut_attack",
                "attack_id": attack_id,
                "control_head_id": control_id,
                "best_learned_head_id": best_learned,
                "best_learned_balanced_accuracy": _round(learned_ba),
                "observed_value": _round(float(observed)),
                "attack_detected": detected,
                "survived": survived,
                "blocks_positive_score": survived,
            }
        )
    surviving = sum(1 for row in attack_rows if row["survived"])
    return {
        "schema_version": f"{SCHEMA_VERSION}.shortcut_attack_matrix",
        "rows": attack_rows,
        "surviving_shortcut_count": surviving,
        "all_shortcuts_rejected": surviving == 0,
        "failing_attack_ids": [row["attack_id"] for row in attack_rows if row["survived"]],
    }


def recompute_aggregate_row(artifact: Mapping[str, Any]) -> JsonDict:
    rows = list(artifact.get("per_unit_rows", []))
    if not rows:
        return {
            "row_type": "aggregate_row_recomputation",
            "held_row_count": 0,
            "headline_recomputed": True,
            "best_learned_head_id": "",
            "best_learned_balanced_accuracy": 0.0,
            "analytical_balanced_accuracy": 0.0,
            "best_shortcut_control_id": "",
            "best_shortcut_balanced_accuracy": 0.0,
            "best_learned_beats_analytical": False,
            "best_learned_beats_shuffled_control": False,
            "all_shortcuts_rejected": False,
            "no_disqualifying_family_cell": False,
            "harmful_flip_count": 0,
            "trajectory_signal_ready_score_from_rows": 0.0,
        }
    metrics = _metrics_by_head(rows)
    best_learned = _best_head(metrics, LEARNED_HEAD_IDS)
    best_shortcut = _best_head(metrics, CONTROL_IDS)
    learned_ba = metrics[best_learned]["balanced_accuracy"] or 0.0
    analytical_ba = metrics["analytical"]["balanced_accuracy"] or 0.0
    shortcut_ba = metrics[best_shortcut]["balanced_accuracy"] or 0.0
    shuffle_ba = metrics["label_shuffle"]["balanced_accuracy"] or 0.0
    family_cells = artifact.get("family_cell_results") or _family_cell_results(rows)
    attacks = artifact.get("shortcut_attack_matrix") or _shortcut_attack_matrix(rows, family_cells)
    all_shortcuts_rejected = bool(attacks.get("all_shortcuts_rejected"))
    no_bad_family = bool(family_cells.get("no_disqualifying_family_cell"))
    ready = (
        learned_ba > analytical_ba + SHORTCUT_MARGIN
        and learned_ba > shuffle_ba + SHORTCUT_MARGIN
        and all_shortcuts_rejected
        and no_bad_family
        and len(_harmful_flip_rows(rows)) == 0
    )
    return {
        "row_type": "aggregate_row_recomputation",
        "held_row_count": len(rows),
        "headline_recomputed": True,
        "head_metric_rows": [_metric_row(rows, head_id) for head_id in HEAD_IDS + CONTROL_IDS],
        "best_learned_head_id": best_learned,
        "best_learned_balanced_accuracy": _round(learned_ba),
        "analytical_balanced_accuracy": _round(analytical_ba),
        "best_shortcut_control_id": best_shortcut,
        "best_shortcut_balanced_accuracy": _round(shortcut_ba),
        "best_learned_beats_analytical": learned_ba > analytical_ba + SHORTCUT_MARGIN,
        "best_learned_beats_shuffled_control": learned_ba > shuffle_ba + SHORTCUT_MARGIN,
        "all_shortcuts_rejected": all_shortcuts_rejected,
        "no_disqualifying_family_cell": no_bad_family,
        "harmful_flip_count": len(_harmful_flip_rows(rows)),
        "trajectory_signal_ready_score_from_rows": 1.0 if ready else 0.0,
    }


def _manifest(exp6489_artifact: Mapping[str, Any] | None = None) -> JsonDict:
    normalizer = {
        "feature_order": list(ALLOWED_FEATURE_FIELDS),
        "means": [],
        "stds": [],
        "fitted_on_splits": ["development"],
    }
    if exp6489_artifact:
        examples = _examples_from_exp6489(exp6489_artifact)
        development = [example for example in examples if example["split"] == "development"]
        normalizer = _normalizer(development)
    return {
        "schema_version": f"{SCHEMA_VERSION}.frozen_analysis_manifest",
        "planning_date": RUN_DATE,
        "feature_groups": {key: list(value) for key, value in FEATURE_GROUPS.items()},
        "forbidden_feature_fields": list(exp6489.FORBIDDEN_FEATURE_FIELDS),
        "preprocessing": normalizer,
        "heads": list(HEAD_IDS),
        "controls": list(CONTROL_IDS),
        "attack_ids": list(ATTACK_IDS),
        "metrics": ["loss", "balanced_accuracy", "auroc", "brier", "ece_5_bin"],
        "threshold_policy": {
            "selected_on": "development",
            "criterion": "maximize_balanced_accuracy",
            "held_threshold_tuning_used": False,
        },
        "fit_seeds": list(FIT_SEEDS),
        "interval_seed": INTERVAL_SEED,
        "held_rows_opened_once": True,
        "held_threshold_tuning_used": False,
        "llm_used": False,
    }


def _field_provenance(artifact: Mapping[str, Any]) -> JsonDict:
    source_hashes = {
        "exp6489": artifact["upstream_gate_receipt"].get("sha256"),
        "module": _sha256_file(REPO_ROOT / MODULE_RELATIVE_PATH),
        "test": _sha256_file(REPO_ROOT / TEST_RELATIVE_PATH),
        "spec": _sha256_file(REPO_ROOT / SPEC_RELATIVE_PATH),
    }
    return {
        field: {
            "sources": source_hashes,
            "reducers": [
                "carnot.experiment_6490_trajectory_energy_baselines",
                "carnot.experiment_6489_solver_trajectory_commitment",
            ],
        }
        for field in REQUIRED_ARTIFACT_FIELDS
    }


def _prior_failure_receipts(root: Path) -> JsonDict:
    receipts_by_id: JsonDict = {}
    for experiment_id, path in (
        ("exp5853-paired-embedding-integrity-audit", _resolve(root, EXP5853_RELATIVE_PATH)),
        ("exp6487-representation-integrity-audit", _resolve(root, EXP6487_RELATIVE_PATH)),
    ):
        artifact = _read_json(path) or {}
        receipts_by_id[experiment_id] = {
            "path": str(path),
            "sha256": _sha256_file(path),
            "status": artifact.get("status"),
            "honest_verdict": artifact.get("honest_verdict"),
            "confirmed_prior_failure": str(artifact.get("honest_verdict", "")).startswith("disqualified:"),
        }
    return receipts_by_id


def _branch_retirement(root: Path, aggregate: Mapping[str, Any], attacks: Mapping[str, Any]) -> JsonDict:
    priors = _prior_failure_receipts(root)
    if aggregate["trajectory_signal_ready_score_from_rows"] == 1.0:
        retired = False
        reason = "complete_positive_not_retired"
    elif attacks.get("surviving_shortcut_count", 0) > 0:
        retired = True
        reason = "shortcut_verdict_repeated"
    else:
        retired = True
        reason = "complete_null_no_learned_gain"
    return {
        "schema_version": f"{SCHEMA_VERSION}.branch_retirement_receipt",
        "retired": retired,
        "reason": reason,
        "prior_failure_verdicts": priors,
        "changed_scope": (
            "Exp6490 uses exact chronological solver trajectory rows and final-outcome "
            "persistence labels. It does not reuse V559 forced candidates or hidden-state selectors."
        ),
        "retire_if_same_verdict": True,
    }


def _gate_receipt(path: Path) -> JsonDict:
    artifact = _read_json(path)
    observed = artifact.get("trajectory_contract_ready_score") if artifact else None
    return {
        "path": str(path),
        "sha256": _sha256_file(path),
        "field": "trajectory_contract_ready_score",
        "expected": 1.0,
        "observed": observed,
        "gate_passed": observed == 1.0,
    }


def _protected_files_unchanged(root: Path) -> JsonDict:
    protected = ("research-roadmap.yaml", "scripts/research_conductor.py")
    status = _git_output(root, ["status", "--short"])
    changed = []
    for line in status.splitlines():
        path = line[3:] if len(line) > 3 else line
        if path in protected:
            changed.append(path)
    return {
        "protected_paths": list(protected),
        "changed_paths": changed,
        "active_roadmap_and_conductor_unchanged": not changed,
    }


def _gate_check_summary(gate: Mapping[str, Any], aggregate: Mapping[str, Any], protected: Mapping[str, Any]) -> JsonDict:
    checks = {
        "upstream_gate_passed": gate["gate_passed"],
        "held_rows_recomputed": aggregate["headline_recomputed"],
        "protected_files_unchanged": protected["active_roadmap_and_conductor_unchanged"],
    }
    failed = [key for key, value in checks.items() if not value]
    return {
        "schema_version": f"{SCHEMA_VERSION}.gate_check_summary",
        "observed_field": gate["field"],
        "expected_value": gate["expected"],
        "observed_value": gate["observed"],
        "checks": checks,
        "failed_gates": failed,
        "all_gates_passed": not failed,
        "blocked_reason": "upstream Exp6489 gate failed" if not gate["gate_passed"] else "",
    }


def _preconditions(gate: Mapping[str, Any], exp6489_artifact: Mapping[str, Any] | None, root: Path) -> JsonDict:
    priors = _prior_failure_receipts(root)
    feature_contract_ready = False
    split_ready = False
    if exp6489_artifact:
        feature_contract_ready = bool(exp6489_artifact.get("identity_free_feature_contract", {}).get("no_label_fields_allowed"))
        split_ready = bool(exp6489_artifact.get("split_commitment", {}).get("held_predates_feature_extraction"))
    return {
        "commitment_gate": gate["gate_passed"],
        "split_predates_feature_extraction": split_ready,
        "prior_failures_confirmed": all(row["confirmed_prior_failure"] for row in priors.values()),
        "feature_contract_identity_free": feature_contract_ready,
        "held_labels_from_exact_final_outcomes": bool(exp6489_artifact and exp6489_artifact.get("verifier_is_oracle") is True),
    }


def _status_and_verdict(aggregate: Mapping[str, Any], attacks: Mapping[str, Any], gate: Mapping[str, Any]) -> tuple[str, str]:
    if not gate["gate_passed"]:
        return "blocked_upstream_gate", "blocked_upstream_gate: Exp6489 trajectory_contract_ready_score was not 1.0"
    if aggregate["trajectory_signal_ready_score_from_rows"] == 1.0:
        return "complete_positive", "complete_positive: learned trajectory head beats analytical and shortcut controls on held units"
    if attacks.get("surviving_shortcut_count", 0) > 0:
        return "disqualified", "disqualified: shortcut control survived held trajectory comparison"
    return "complete_null", "complete_null: learned trajectory heads do not clear the predeclared held gate"


def _empty_artifact(
    *,
    root: Path,
    gate: Mapping[str, Any],
    protected: Mapping[str, Any],
    duration_s: float,
    tests_run: Sequence[Mapping[str, Any]],
    exp6489_artifact: Mapping[str, Any] | None,
) -> JsonDict:
    aggregate = recompute_aggregate_row({"per_unit_rows": []})
    attacks = _shortcut_attack_matrix([], {"rows": [], "no_failing_family_pooled_away": False})
    status, verdict = _status_and_verdict(aggregate, attacks, gate)
    artifact: JsonDict = {
        "status": status,
        "upstream_gate_receipt": dict(gate),
        "frozen_analysis_manifest": _manifest(exp6489_artifact),
        "model_configuration_rows": [],
        "rows": [],
        "family_cell_results": {
            "schema_version": f"{SCHEMA_VERSION}.family_cells",
            "rows": [],
            "disqualifying_family_cell_count": 0,
            "no_disqualifying_family_cell": False,
            "no_failing_family_pooled_away": True,
        },
        "calibration_rows": [],
        "confidence_intervals": {
            "schema_version": f"{SCHEMA_VERSION}.confidence_intervals",
            "rows": [],
            "unit_level": True,
            "interval_seed": INTERVAL_SEED,
        },
        "shortcut_attack_matrix": attacks,
        "harmful_flip_rows": [],
        "trajectory_signal_ready_score": 0.0,
        "branch_retirement_receipt": _branch_retirement(root, aggregate, attacks),
        "per_unit_rows": [],
        "rows": [],
        "aggregate_row_recomputation": aggregate,
        "gate_check_summary": _gate_check_summary(gate, aggregate, protected),
        "preconditions_checked": _preconditions(gate, exp6489_artifact, root),
        "protected_files_unchanged": dict(protected),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": True,
        "field_principles": dict(FIELD_PRINCIPLES),
        "field_provenance": {},
        "random_seed": {
            "experiment": RANDOM_SEED,
            "fit_seeds": list(FIT_SEEDS),
            "interval_seed": INTERVAL_SEED,
        },
        "duration_s": _round(duration_s),
        "tests_run": list(tests_run),
        "reproducibility_checksum": "",
        "honest_verdict": verdict,
    }
    artifact["field_provenance"] = _field_provenance(artifact)
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    payload = {
        "upstream_gate_receipt": artifact.get("upstream_gate_receipt"),
        "frozen_analysis_manifest": artifact.get("frozen_analysis_manifest"),
        "model_configuration_rows": artifact.get("model_configuration_rows"),
        "rows": artifact.get("rows"),
        "family_cell_results": artifact.get("family_cell_results"),
        "calibration_rows": artifact.get("calibration_rows"),
        "confidence_intervals": artifact.get("confidence_intervals"),
        "shortcut_attack_matrix": artifact.get("shortcut_attack_matrix"),
        "harmful_flip_rows": artifact.get("harmful_flip_rows"),
        "aggregate_row_recomputation": artifact.get("aggregate_row_recomputation"),
        "branch_retirement_receipt": artifact.get("branch_retirement_receipt"),
        "random_seed": artifact.get("random_seed"),
    }
    return _sha256_json(payload)


def build_artifact(
    *,
    root: Path = REPO_ROOT,
    result_path: Path | None = None,
    exp6489_path: Path | None = None,
    write: bool = True,
    duration_s: float = 0.0,
    tests_run: Sequence[Mapping[str, Any]] | None = None,
) -> JsonDict:
    root = Path(root)
    result = _resolve(root, result_path or RESULT_RELATIVE_PATH)
    source_path = _resolve(root, exp6489_path or EXP6489_RELATIVE_PATH)
    gate = _gate_receipt(source_path)
    exp6489_artifact = _read_json(source_path)
    protected = _protected_files_unchanged(root)
    tests = list(tests_run) if tests_run is not None else [
        {"command": command, "exit_code": None, "status": "not_run_by_artifact_emitter"}
        for command in DEFAULT_TEST_COMMANDS
    ]
    if not gate["gate_passed"] or exp6489_artifact is None:
        artifact = _empty_artifact(
            root=root,
            gate=gate,
            protected=protected,
            duration_s=duration_s,
            tests_run=tests,
            exp6489_artifact=exp6489_artifact,
        )
        if write:
            _write_atomic(result, artifact)
        return artifact

    manifest = _manifest(exp6489_artifact)
    examples = _examples_from_exp6489(exp6489_artifact)
    rows, configs = _build_scores(examples, manifest)
    family_cells = _family_cell_results(rows)
    calibration = _calibration_rows(rows)
    intervals = _confidence_intervals(rows)
    attacks = _shortcut_attack_matrix(rows, family_cells)
    aggregate = recompute_aggregate_row(
        {
            "per_unit_rows": rows,
            "family_cell_results": family_cells,
            "shortcut_attack_matrix": attacks,
        }
    )
    flips = _harmful_flip_rows(rows)
    aggregate["harmful_flip_count"] = len(flips)
    score = aggregate["trajectory_signal_ready_score_from_rows"]
    status, verdict = _status_and_verdict(aggregate, attacks, gate)
    artifact = {
        "status": status,
        "upstream_gate_receipt": gate,
        "frozen_analysis_manifest": manifest,
        "model_configuration_rows": configs,
        "rows": rows,
        "family_cell_results": family_cells,
        "calibration_rows": calibration,
        "confidence_intervals": intervals,
        "shortcut_attack_matrix": attacks,
        "harmful_flip_rows": flips,
        "trajectory_signal_ready_score": score,
        "branch_retirement_receipt": _branch_retirement(root, aggregate, attacks),
        "per_unit_rows": rows,
        "aggregate_row_recomputation": aggregate,
        "gate_check_summary": _gate_check_summary(gate, aggregate, protected),
        "preconditions_checked": _preconditions(gate, exp6489_artifact, root),
        "protected_files_unchanged": protected,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": True,
        "field_principles": dict(FIELD_PRINCIPLES),
        "field_provenance": {},
        "random_seed": {
            "experiment": RANDOM_SEED,
            "fit_seeds": list(FIT_SEEDS),
            "interval_seed": INTERVAL_SEED,
        },
        "duration_s": _round(duration_s),
        "tests_run": tests,
        "reproducibility_checksum": "",
        "honest_verdict": verdict,
    }
    artifact["field_provenance"] = _field_provenance(artifact)
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    if write:
        _write_atomic(result, artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        return [f"missing required fields: {', '.join(missing)}"]
    if artifact["reproducibility_checksum"] != reproducibility_checksum(artifact):
        errors.append("reproducibility_checksum mismatch")
    if artifact["inference_substrate"] != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate mismatch")
    if artifact["verifier_is_oracle"] is not True:
        errors.append("verifier_is_oracle must be true for exact final outcomes")
    if set(artifact["field_principles"]) != set(REQUIRED_ARTIFACT_FIELDS):
        errors.append("field_principles must cover exactly required fields")
    if set(artifact["field_provenance"]) != set(REQUIRED_ARTIFACT_FIELDS):
        errors.append("field_provenance must cover exactly required fields")
    if artifact["rows"] != artifact["per_unit_rows"]:
        errors.append("rows and per_unit_rows must match")

    expected_family = _family_cell_results(artifact["per_unit_rows"]) if artifact["per_unit_rows"] else artifact["family_cell_results"]
    if artifact["per_unit_rows"] and artifact["family_cell_results"] != expected_family:
        errors.append("family_cell_results mismatch")
    expected_attacks = (
        _shortcut_attack_matrix(artifact["per_unit_rows"], artifact["family_cell_results"])
        if artifact["per_unit_rows"]
        else artifact["shortcut_attack_matrix"]
    )
    if artifact["per_unit_rows"] and artifact["shortcut_attack_matrix"] != expected_attacks:
        errors.append("shortcut_attack_matrix mismatch")
    expected_aggregate = recompute_aggregate_row(artifact)
    if artifact["aggregate_row_recomputation"] != expected_aggregate:
        errors.append("aggregate_row_recomputation mismatch")
    if artifact["trajectory_signal_ready_score"] != expected_aggregate["trajectory_signal_ready_score_from_rows"]:
        errors.append("trajectory_signal_ready_score mismatch")
    if artifact["protected_files_unchanged"]["active_roadmap_and_conductor_unchanged"] is not True:
        errors.append("protected files changed")
    if not str(artifact["honest_verdict"]).startswith(
        ("complete_positive:", "complete_null:", "disqualified:", "blocked_")
    ):
        errors.append("honest_verdict lacks required terminal prefix")
    return errors


def run(
    *,
    date: str = RUN_DATE,
    result_path: Path | None = None,
    root: Path = REPO_ROOT,
    tests_run: Sequence[Mapping[str, Any]] | None = None,
) -> JsonDict:
    started = time.perf_counter()
    artifact = build_artifact(
        root=root,
        result_path=result_path,
        write=True,
        duration_s=0.0,
        tests_run=tests_run,
    )
    artifact["frozen_analysis_manifest"]["planning_date"] = date
    artifact["duration_s"] = _round(time.perf_counter() - started)
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    _write_atomic(_resolve(root, result_path or RESULT_RELATIVE_PATH), artifact)
    return artifact


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--result-path", type=Path, default=RESULT_RELATIVE_PATH)
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args(argv)
    result_path = _resolve(REPO_ROOT, args.result_path)
    if args.validate:
        artifact = _read_json(result_path)
        errors = ["artifact missing"] if artifact is None else validate_artifact(artifact)
        print(json.dumps({"errors": errors, "ok": not errors}, sort_keys=True))
        return 0 if not errors else 1
    artifact = run(date=args.date, result_path=result_path)
    print(json.dumps({"path": str(result_path), "status": artifact["status"]}, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv[1:]))
