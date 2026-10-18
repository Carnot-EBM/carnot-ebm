#!/usr/bin/env python3
"""Exp 5047: KAN/PURM-style calibration over powered D1 energy margins.

Spec refs: REQ-VERIFY-5047, SCENARIO-VERIFY-5047.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping, Sequence
import hashlib
import json
import math
from pathlib import Path
import random
import sys
import time
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "python") not in sys.path:  # pragma: no cover - direct script execution
    sys.path.insert(0, str(REPO_ROOT / "python"))

from carnot import experiment_5031_lora_ebm_scorer_musr_v3 as d1  # noqa: E402
from carnot import moat_benchmark_harness as harness  # noqa: E402


JsonDict = dict[str, Any]
JsonMap = Mapping[str, Any]
UpstreamLoader = Callable[[Path], JsonDict]
PanelLoader = Callable[[Path, JsonMap], list[JsonDict]]
Clock = Callable[[], float]

EXPERIMENT_ID = 5047
EXPERIMENT_NAME = "experiment_5047_kan_purm_energy_calibration"
SCHEMA = "carnot.experiment_5047_kan_purm_energy_calibration.v1"
RESULT_RELATIVE_PATH = "results/experiment_5047_kan_purm_energy_calibration.json"
EXP5045_RELATIVE_PATH = "results/experiment_5045_powered_lora_ebm_eorm_musr.json"
MUSR_CHECKPOINT_RELATIVE_DIR = "results/distributional_energy_verifier_musr_checkpoints"
SPEC_REFS = ["REQ-VERIFY-5047", "SCENARIO-VERIFY-5047"]
RANDOM_SEED = harness.DEFAULT_RANDOM_SEED
ABSTENTION_LIMIT = harness.ABSTENTION_DEGENERACY_THRESHOLD
COLLAPSE_LIMIT = 0.999999

RAW_FEATURE_NAMES = [
    "neg_energy_delta",
    "selected_by_powered_d1",
    "answer_support",
    "neg_cache_index",
    "powered_margin",
    "energy_std",
    "answer_entropy",
    "d1_sc_agree",
]
KAN_FEATURE_NAMES = [
    "close_energy_membership",
    "far_energy_membership",
    "low_margin_membership",
    "high_margin_membership",
    "support_membership",
    "entropy_membership",
    "powered_selected_high_margin_rule",
    "non_powered_low_margin_rule",
    "agreement_rule",
]
THRESHOLD_GRID = [round(value / 20.0, 2) for value in range(0, 20)]

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "model_specs",
    "calibration_available",
    "calibrated_accuracy",
    "powered_d1_accuracy",
    "delta_vs_powered_d1",
    "delta_vs_tuned_sc",
    "paired_ci95",
    "mcnemar_p",
    "abstention_rate",
    "degeneracy_guard_fired",
    "verifier_is_oracle",
    "headroom_present",
    "schema",
    "experiment",
    "experiment_id",
    "spec_refs",
    "result_path",
    "source_artifacts",
    "split_diagnostics",
    "baselines",
    "readout",
    "n_questions",
    "n_candidate_rows",
    "duration_s",
    "reproducibility_checksum",
)


def _json_dumps(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _read_json(path: Path) -> Any:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def write_json(path: Path, payload: JsonMap) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _number(value: Any, default: float = 0.0) -> float:
    if isinstance(value, bool):
        return default
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _answer_counts(candidates: Sequence[JsonMap]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for candidate in candidates:
        answer = str(candidate.get("answer") or "")
        if answer:
            counts[answer] += 1
    return counts


def _normalized_entropy(counts: Counter[str]) -> float:
    total = sum(counts.values())
    if total <= 1 or len(counts) <= 1:
        return 0.0
    entropy = 0.0
    for count in counts.values():
        probability = count / total
        entropy -= probability * math.log(probability)
    return entropy / math.log(len(counts))


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _std(values: Sequence[float]) -> float:
    if len(values) <= 1:
        return 0.0
    mu = _mean(values)
    return math.sqrt(sum((value - mu) ** 2 for value in values) / len(values))


def build_readout_rows(
    rows: Sequence[JsonMap],
    energy_by_id: Mapping[str, float],
    *,
    tuned_sc_predictions: Sequence[Any],
) -> list[JsonDict]:
    """REQ-VERIFY-5047: build held-out-safe feature rows from candidate energies."""

    readout_rows: list[JsonDict] = []
    for row_index, row in enumerate(rows):
        candidates = [dict(candidate) for candidate in row.get("candidates", [])]
        scored: list[tuple[float, str, JsonDict]] = []
        for candidate in candidates:
            candidate_id = str(candidate.get("candidate_id") or "")
            energy = _number(energy_by_id.get(candidate_id), math.inf)
            if math.isfinite(energy):
                scored.append((energy, candidate_id, candidate))
        if not scored:
            continue
        scored.sort(key=lambda item: (item[0], item[1]))
        best_energy, _best_id, best_candidate = scored[0]
        second_energy = scored[1][0] if len(scored) > 1 else best_energy
        margin = max(0.0, second_energy - best_energy)
        energies = [item[0] for item in scored]
        energy_std = _std(energies)
        counts = _answer_counts(candidates)
        entropy = _normalized_entropy(counts)
        total_answers = max(1, sum(counts.values()))
        powered_answer = str(best_candidate.get("answer")) if best_candidate.get("answer") else None
        tuned_answer = (
            str(tuned_sc_predictions[row_index])
            if row_index < len(tuned_sc_predictions) and tuned_sc_predictions[row_index] is not None
            else None
        )
        row_candidates: list[JsonDict] = []
        for energy, candidate_id, candidate in scored:
            answer = str(candidate.get("answer") or "")
            label = int(answer == str(row.get("gold")))
            selected_by_powered = int(answer == str(powered_answer))
            support = counts[answer] / total_answers if answer else 0.0
            cache_index = _number(candidate.get("cache_index"))
            row_candidates.append(
                {
                    "candidate_id": candidate_id,
                    "answer": answer,
                    "energy": round(energy, 12),
                    "energy_delta": round(max(0.0, energy - best_energy), 12),
                    "cache_index": int(cache_index),
                    "answer_support": round(support, 12),
                    "selected_by_powered_d1": selected_by_powered,
                    "label": label,
                }
            )
        readout_rows.append(
            {
                "row_id": str(row.get("row_id", row_index)),
                "question_index": row_index,
                "gold": str(row.get("gold") or ""),
                "powered_d1_answer": powered_answer,
                "tuned_sc_answer": tuned_answer,
                "powered_d1_correct": harness._is_correct(powered_answer, row.get("gold")),  # noqa: SLF001
                "tuned_sc_correct": harness._is_correct(tuned_answer, row.get("gold")),  # noqa: SLF001
                "powered_margin": round(margin, 12),
                "answer_entropy": round(entropy, 12),
                "energy_std": round(energy_std, 12),
                "d1_sc_agree": int(powered_answer == tuned_answer),
                "candidates": row_candidates,
            }
        )
    return readout_rows


def make_cv_splits(n_rows: int, *, n_folds: int = 5, seed: int = RANDOM_SEED) -> list[JsonDict]:
    """REQ-VERIFY-5047: deterministic question-level folds."""

    if n_rows <= 1:
        return [{"train_indices": list(range(n_rows)), "test_indices": list(range(n_rows))}]
    folds = max(2, min(int(n_folds), n_rows))
    indices = list(range(n_rows))
    rng = random.Random(seed)
    rng.shuffle(indices)
    buckets = [indices[offset::folds] for offset in range(folds)]
    splits: list[JsonDict] = []
    universe = set(range(n_rows))
    for bucket in buckets:
        test = sorted(bucket)
        train = sorted(universe.difference(test))
        splits.append({"train_indices": train, "test_indices": test})
    return splits


def split_integrity_errors(splits: Sequence[JsonMap], *, n_rows: int) -> list[str]:
    """REQ-VERIFY-5047: validate no train/test overlap and full held-out coverage."""

    errors: list[str] = []
    seen_test: list[int] = []
    valid = set(range(n_rows))
    for split in splits:
        train = {int(index) for index in split.get("train_indices", [])}
        test = {int(index) for index in split.get("test_indices", [])}
        if train.intersection(test):
            errors.append("train_test_overlap")
        if not train:
            errors.append("empty_train_split")
        if not test:
            errors.append("empty_test_split")
        if not train.issubset(valid) or not test.issubset(valid):
            errors.append("split_index_out_of_range")
        seen_test.extend(sorted(test))
    if sorted(seen_test) != list(range(n_rows)):
        errors.append("test_coverage_not_exactly_once")
    return sorted(set(errors))


def degeneracy_guard(
    predictions: Sequence[Any],
    powered_d1_predictions: Sequence[Any],
    *,
    abstention_rate: float,
) -> JsonDict:
    """REQ-VERIFY-5047: fire on >50% abstention or collapse to D1 choices."""

    pairs = list(zip(predictions, powered_d1_predictions))
    collapse_rate = (
        sum(1 for prediction, powered in pairs if prediction == powered) / len(pairs)
        if pairs
        else 0.0
    )
    reasons: list[str] = []
    if float(abstention_rate) > ABSTENTION_LIMIT:
        reasons.append("abstention_gt_0p50")
    if pairs and collapse_rate >= COLLAPSE_LIMIT:
        reasons.append("collapsed_to_powered_d1")
    return {
        "degeneracy_guard_fired": bool(reasons),
        "abstention_rate": round(float(abstention_rate), 6),
        "abstention_threshold": ABSTENTION_LIMIT,
        "collapse_rate": round(collapse_rate, 6),
        "collapse_threshold": COLLAPSE_LIMIT,
        "reasons": reasons,
    }


def _raw_features(row: JsonMap, candidate: JsonMap) -> list[float]:
    margin = _number(row.get("powered_margin"))
    entropy = _number(row.get("answer_entropy"))
    energy_std = _number(row.get("energy_std"))
    return [
        -_number(candidate.get("energy_delta")),
        _number(candidate.get("selected_by_powered_d1")),
        _number(candidate.get("answer_support")),
        -_number(candidate.get("cache_index")) / 10.0,
        margin,
        energy_std,
        entropy,
        _number(row.get("d1_sc_agree")),
    ]


def _kan_features(row: JsonMap, candidate: JsonMap) -> list[float]:
    delta = _number(candidate.get("energy_delta"))
    margin = _number(row.get("powered_margin"))
    support = _number(candidate.get("answer_support"))
    entropy = _number(row.get("answer_entropy"))
    powered = _number(candidate.get("selected_by_powered_d1"))
    agree = _number(row.get("d1_sc_agree"))
    close = 1.0 / (1.0 + max(0.0, delta))
    far = 1.0 - close
    low_margin = 1.0 / (1.0 + max(0.0, margin))
    high_margin = 1.0 - low_margin
    return [
        close,
        far,
        low_margin,
        high_margin,
        support,
        entropy,
        powered * high_margin,
        (1.0 - powered) * low_margin,
        agree * powered,
    ]


def _candidate_records(rows: Sequence[JsonMap], indices: Sequence[int]) -> tuple[list[list[float]], list[int]]:
    features: list[list[float]] = []
    labels: list[int] = []
    for index in indices:
        row = rows[index]
        for candidate in row.get("candidates", []):
            features.append(_kan_features(row, candidate))
            labels.append(int(candidate.get("label") or 0))
    return features, labels


def _raw_candidate_records(
    rows: Sequence[JsonMap], indices: Sequence[int]
) -> tuple[list[list[float]], list[int]]:
    features: list[list[float]] = []
    labels: list[int] = []
    for index in indices:
        row = rows[index]
        for candidate in row.get("candidates", []):
            features.append(_raw_features(row, candidate))
            labels.append(int(candidate.get("label") or 0))
    return features, labels


def _standardize(features: Sequence[Sequence[float]]) -> tuple[list[float], list[float]]:
    if not features:
        return [], []
    width = len(features[0])
    means = [_mean([row[col] for row in features]) for col in range(width)]
    scales = []
    for col in range(width):
        col_std = _std([row[col] for row in features])
        scales.append(col_std if col_std > 1e-9 else 1.0)
    return means, scales


def _sigmoid(value: float) -> float:
    if value >= 0.0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def _fit_logistic(
    features: Sequence[Sequence[float]],
    labels: Sequence[int],
    *,
    feature_names: Sequence[str],
    epochs: int,
    learning_rate: float = 0.15,
    l2: float = 0.001,
) -> JsonDict:
    means, scales = _standardize(features)
    width = len(feature_names)
    weights = [0.0 for _ in range(width)]
    bias = 0.0
    n = max(1, len(labels))
    for _epoch in range(max(1, int(epochs))):
        grad_w = [0.0 for _ in range(width)]
        grad_b = 0.0
        for raw, label in zip(features, labels):
            x = [(raw[col] - means[col]) / scales[col] for col in range(width)]
            pred = _sigmoid(bias + sum(weight * value for weight, value in zip(weights, x)))
            error = pred - int(label)
            for col, value in enumerate(x):
                grad_w[col] += error * value
            grad_b += error
        for col in range(width):
            weights[col] -= learning_rate * ((grad_w[col] / n) + l2 * weights[col])
        bias -= learning_rate * grad_b / n
    return {
        "weights": [round(value, 12) for value in weights],
        "bias": round(bias, 12),
        "means": [round(value, 12) for value in means],
        "scales": [round(value, 12) for value in scales],
        "feature_names": list(feature_names),
    }


def _predict_probability(model: JsonMap, features: Sequence[float]) -> float:
    means = [_number(value) for value in model.get("means", [])]
    scales = [_number(value, 1.0) or 1.0 for value in model.get("scales", [])]
    weights = [_number(value) for value in model.get("weights", [])]
    x = [(features[col] - means[col]) / scales[col] for col in range(len(weights))]
    return _sigmoid(_number(model.get("bias")) + sum(weight * value for weight, value in zip(weights, x)))


def _candidate_probabilities(row: JsonMap, model: JsonMap, *, kind: str) -> list[tuple[float, JsonMap]]:
    out: list[tuple[float, JsonMap]] = []
    for candidate in row.get("candidates", []):
        features = _raw_features(row, candidate) if kind == "logistic" else _kan_features(row, candidate)
        out.append((_predict_probability(model, features), candidate))
    return out


def _select_answer(
    row: JsonMap,
    model: JsonMap,
    *,
    threshold: float,
    kind: str,
) -> tuple[str | None, bool]:
    probabilities = _candidate_probabilities(row, model, kind=kind)
    if not probabilities:
        return None, True
    probability, candidate = max(
        probabilities,
        key=lambda item: (
            item[0],
            -_number(item[1].get("energy_delta")),
            -_number(item[1].get("cache_index")),
            str(item[1].get("candidate_id") or ""),
        ),
    )
    if probability < threshold:
        answer = row.get("tuned_sc_answer")
        return (str(answer) if answer is not None else None), True
    answer = candidate.get("answer")
    return (str(answer) if answer is not None else None), False


def _correct(predictions: Sequence[Any], rows: Sequence[JsonMap]) -> list[int]:
    return [
        harness._is_correct(str(prediction) if prediction is not None else None, row.get("gold"))  # noqa: SLF001
        for prediction, row in zip(predictions, rows)
    ]


def _accuracy(correct: Sequence[int]) -> float:
    return round(sum(correct) / len(correct), 6) if correct else 0.0


def _choose_threshold(
    rows: Sequence[JsonMap],
    indices: Sequence[int],
    model: JsonMap,
    *,
    kind: str,
    thresholds: Sequence[float] = THRESHOLD_GRID,
) -> JsonDict:
    candidates: list[JsonDict] = []
    for threshold in thresholds:
        predictions: list[str | None] = []
        abstentions = 0
        selected_rows = [rows[index] for index in indices]
        for row in selected_rows:
            prediction, abstained = _select_answer(row, model, threshold=float(threshold), kind=kind)
            predictions.append(prediction)
            abstentions += int(abstained)
        correct = _correct(predictions, selected_rows)
        abstention_rate = abstentions / len(selected_rows) if selected_rows else 0.0
        candidates.append(
            {
                "threshold": float(threshold),
                "accuracy": _accuracy(correct),
                "abstention_rate": round(abstention_rate, 6),
            }
        )
    capped = [row for row in candidates if float(row["abstention_rate"]) <= ABSTENTION_LIMIT]
    options = capped or candidates
    best = max(
        options,
        key=lambda item: (
            float(item["accuracy"]),
            -float(item["abstention_rate"]),
            -float(item["threshold"]),
        ),
    )
    return dict(best)


def _fit_fold_models(
    rows: Sequence[JsonMap],
    train_indices: Sequence[int],
    *,
    logistic_epochs: int,
) -> tuple[JsonDict, JsonDict]:
    kan_features, kan_labels = _candidate_records(rows, train_indices)
    raw_features, raw_labels = _raw_candidate_records(rows, train_indices)
    kan_model = _fit_logistic(
        kan_features,
        kan_labels,
        feature_names=KAN_FEATURE_NAMES,
        epochs=logistic_epochs,
    )
    raw_model = _fit_logistic(
        raw_features,
        raw_labels,
        feature_names=RAW_FEATURE_NAMES,
        epochs=logistic_epochs,
    )
    return kan_model, raw_model


def evaluate_cross_validated_readout(
    rows: Sequence[JsonMap],
    *,
    n_folds: int = 5,
    seed: int = RANDOM_SEED,
    bootstrap_samples: int = 2000,
    logistic_epochs: int = 160,
) -> JsonDict:
    """SCENARIO-VERIFY-5047: fit and evaluate held-out readouts."""

    rows_list = [dict(row) for row in rows if row.get("candidates")]
    splits = make_cv_splits(len(rows_list), n_folds=n_folds, seed=seed)
    errors = split_integrity_errors(splits, n_rows=len(rows_list))
    if errors:
        raise ValueError(f"split integrity failed: {errors}")

    calibrated_predictions: list[str | None] = [None for _row in rows_list]
    logistic_predictions: list[str | None] = [None for _row in rows_list]
    abstentions = 0
    thresholds: list[float] = []
    fold_summaries: list[JsonDict] = []
    final_readout: JsonDict = {}
    final_logistic: JsonDict = {}

    for split_id, split in enumerate(splits):
        train_indices = [int(index) for index in split["train_indices"]]
        test_indices = [int(index) for index in split["test_indices"]]
        kan_model, raw_model = _fit_fold_models(
            rows_list,
            train_indices,
            logistic_epochs=logistic_epochs,
        )
        kan_threshold = _choose_threshold(rows_list, train_indices, kan_model, kind="kan")
        raw_threshold = _choose_threshold(rows_list, train_indices, raw_model, kind="logistic")
        thresholds.append(float(kan_threshold["threshold"]))
        final_readout = kan_model
        final_logistic = raw_model
        fold_abstentions = 0
        for index in test_indices:
            calibrated, abstained = _select_answer(
                rows_list[index],
                kan_model,
                threshold=float(kan_threshold["threshold"]),
                kind="kan",
            )
            logistic, _raw_abstained = _select_answer(
                rows_list[index],
                raw_model,
                threshold=float(raw_threshold["threshold"]),
                kind="logistic",
            )
            calibrated_predictions[index] = calibrated
            logistic_predictions[index] = logistic
            abstentions += int(abstained)
            fold_abstentions += int(abstained)
        fold_summaries.append(
            {
                "fold_id": split_id,
                "train_n": len(train_indices),
                "test_n": len(test_indices),
                "kan_threshold": float(kan_threshold["threshold"]),
                "logistic_threshold": float(raw_threshold["threshold"]),
                "fold_abstention_rate": round(fold_abstentions / len(test_indices), 6),
            }
        )

    powered_predictions = [row.get("powered_d1_answer") for row in rows_list]
    tuned_predictions = [row.get("tuned_sc_answer") for row in rows_list]
    calibrated_correct = _correct(calibrated_predictions, rows_list)
    powered_correct = _correct(powered_predictions, rows_list)
    tuned_correct = _correct(tuned_predictions, rows_list)
    logistic_correct = _correct(logistic_predictions, rows_list)
    abstention_rate = abstentions / len(rows_list) if rows_list else 0.0
    guard = degeneracy_guard(
        calibrated_predictions,
        powered_predictions,
        abstention_rate=abstention_rate,
    )
    calibrated_accuracy = _accuracy(calibrated_correct)
    powered_accuracy = _accuracy(powered_correct)
    tuned_accuracy = _accuracy(tuned_correct)
    return {
        "n_rows": len(rows_list),
        "n_candidate_rows": sum(len(row.get("candidates", [])) for row in rows_list),
        "calibrated_predictions": calibrated_predictions,
        "powered_d1_predictions": powered_predictions,
        "tuned_sc_predictions": tuned_predictions,
        "calibrated_correct": calibrated_correct,
        "powered_d1_correct": powered_correct,
        "tuned_sc_correct": tuned_correct,
        "logistic_correct": logistic_correct,
        "calibrated_accuracy": calibrated_accuracy,
        "powered_d1_accuracy": powered_accuracy,
        "tuned_sc_accuracy": tuned_accuracy,
        "logistic_baseline_accuracy": _accuracy(logistic_correct),
        "delta_vs_powered_d1": round(calibrated_accuracy - powered_accuracy, 6),
        "delta_vs_tuned_sc": round(calibrated_accuracy - tuned_accuracy, 6),
        "paired_ci95": harness.paired_bootstrap_ci(
            calibrated_correct,
            powered_correct,
            seed=seed,
            samples=bootstrap_samples,
        ),
        "mcnemar_p": harness.mcnemar_exact_p(calibrated_correct, powered_correct),
        "abstention_rate": round(abstention_rate, 6),
        "degeneracy_guard": guard,
        "splits": splits,
        "fold_summaries": fold_summaries,
        "readout_model": final_readout,
        "logistic_model": final_logistic,
        "selected_threshold_summary": {
            "mean": round(_mean(thresholds), 6),
            "min": round(min(thresholds), 6) if thresholds else 0.0,
            "max": round(max(thresholds), 6) if thresholds else 0.0,
        },
    }


def _format_delta(delta: float) -> str:
    return f"{delta:+.3f}".replace("+", "plus_").replace("-", "minus_").replace(".", "p")


def _ci_includes_zero(ci95: Sequence[float]) -> bool:
    return len(ci95) == 2 and float(ci95[0]) <= 0.0 <= float(ci95[1])


def reproducibility_checksum(payload: JsonMap) -> str:
    return "sha256:" + hashlib.sha256(_json_dumps(payload).encode("utf-8")).hexdigest()


def default_upstream_loader(root: Path) -> JsonDict:  # pragma: no cover - filesystem glue
    payload = _read_json(Path(root) / EXP5045_RELATIVE_PATH)
    return dict(payload) if isinstance(payload, Mapping) else {}


def _tuned_predictions_from_upstream(upstream: JsonMap) -> list[Any]:
    evaluation = upstream.get("evaluation") if isinstance(upstream.get("evaluation"), Mapping) else {}
    tuned = (
        evaluation.get("tuned_self_consistency")
        if isinstance(evaluation.get("tuned_self_consistency"), Mapping)
        else {}
    )
    return list(tuned.get("predictions") or [])


def default_panel_loader(root: Path, upstream: JsonMap) -> list[JsonDict]:  # pragma: no cover - live scoring
    checkpoint = str(upstream.get("checkpoint_path") or "")
    if not checkpoint:
        return []
    n_rows = int(upstream.get("n_questions") or 0) or None
    narratives = d1._default_narratives_loader(n_rows or 200)
    raw_rows = d1.load_musr_eval_rows(
        Path(root) / MUSR_CHECKPOINT_RELATIVE_DIR,
        narratives=narratives,
        limit=n_rows,
    )
    config = d1.TrainingConfig(seed=RANDOM_SEED)
    energy_by_id = d1.precompute_candidate_energies(
        checkpoint,
        raw_rows,
        score_fn=d1.default_score_fn(config),
    )
    tuned_predictions = _tuned_predictions_from_upstream(upstream)
    if not tuned_predictions:
        tuned_predictions = list(harness.tuned_self_consistency(raw_rows).get("predictions", []))
    return build_readout_rows(raw_rows, energy_by_id, tuned_sc_predictions=tuned_predictions)


def _base_artifact(
    *,
    honest_verdict: str,
    upstream: JsonMap,
    duration_s: float,
    calibration_available: bool,
) -> JsonDict:
    return {
        "schema": SCHEMA,
        "experiment": EXPERIMENT_NAME,
        "experiment_id": EXPERIMENT_ID,
        "spec_refs": list(SPEC_REFS),
        "result_path": RESULT_RELATIVE_PATH,
        "honest_verdict": honest_verdict,
        "model_specs": {
            "inherited_headline_models": dict(upstream.get("model_specs") or {}),
            "calibration_readout": {
                "kind": "additive_fuzzy_kan_purm",
                "logistic_baseline_included": True,
                "fresh_llm_generation": False,
            },
        },
        "calibration_available": calibration_available,
        "calibrated_accuracy": None,
        "powered_d1_accuracy": None,
        "delta_vs_powered_d1": None,
        "delta_vs_tuned_sc": None,
        "paired_ci95": None,
        "mcnemar_p": None,
        "abstention_rate": None,
        "degeneracy_guard_fired": False,
        "verifier_is_oracle": False,
        "headroom_present": bool(upstream.get("headroom_present")),
        "source_artifacts": {
            "powered_d1": EXP5045_RELATIVE_PATH,
            "musr_candidate_cache": MUSR_CHECKPOINT_RELATIVE_DIR,
        },
        "split_diagnostics": {},
        "baselines": {},
        "readout": {"kind": "additive_fuzzy_kan_purm"},
        "n_questions": 0,
        "n_candidate_rows": 0,
        "duration_s": round(float(duration_s), 6),
        "reproducibility_checksum": "",
    }


def _complete_artifact(
    *,
    upstream: JsonMap,
    rows: Sequence[JsonMap],
    evaluation: JsonMap,
    duration_s: float,
) -> JsonDict:
    delta = float(evaluation["delta_vs_powered_d1"])
    ci95 = [float(value) for value in evaluation["paired_ci95"]]
    mcnemar_p = float(evaluation["mcnemar_p"])
    guard = dict(evaluation["degeneracy_guard"])
    headroom_present = bool(upstream.get("headroom_present"))
    win = delta > 0.0 and ci95[0] > 0.0 and mcnemar_p < 0.05 and not guard[
        "degeneracy_guard_fired"
    ] and headroom_present
    delta_label = _format_delta(delta)
    if win:
        verdict = f"success_kan_purm_beats_powered_d1_musr_{delta_label}"
    elif guard["degeneracy_guard_fired"]:
        verdict = f"complete_kan_purm_no_incremental_lift_over_powered_d1_{delta_label}_degenerate"
    elif _ci_includes_zero(ci95):
        verdict = f"complete_kan_purm_no_incremental_lift_over_powered_d1_{delta_label}_ci_incl_0"
    else:
        verdict = f"complete_kan_purm_no_incremental_lift_over_powered_d1_{delta_label}_mcnemar_or_headroom_gate"
    artifact = _base_artifact(
        honest_verdict=verdict,
        upstream=upstream,
        duration_s=duration_s,
        calibration_available=True,
    )
    artifact.update(
        {
            "calibrated_accuracy": float(evaluation["calibrated_accuracy"]),
            "powered_d1_accuracy": float(evaluation["powered_d1_accuracy"]),
            "delta_vs_powered_d1": delta,
            "delta_vs_tuned_sc": float(evaluation["delta_vs_tuned_sc"]),
            "paired_ci95": ci95,
            "mcnemar_p": mcnemar_p,
            "abstention_rate": float(evaluation["abstention_rate"]),
            "degeneracy_guard_fired": bool(guard["degeneracy_guard_fired"]),
            "headroom_present": headroom_present,
            "split_diagnostics": {
                "n_folds": len(evaluation["splits"]),
                "split_integrity_errors": split_integrity_errors(
                    evaluation["splits"], n_rows=int(evaluation["n_rows"])
                ),
                "fold_summaries": list(evaluation["fold_summaries"]),
            },
            "baselines": {
                "logistic_baseline_accuracy": float(evaluation["logistic_baseline_accuracy"]),
                "genuine_tuned_sc_accuracy": float(evaluation["tuned_sc_accuracy"]),
            },
            "readout": {
                "kind": "additive_fuzzy_kan_purm",
                "feature_names": list(KAN_FEATURE_NAMES),
                "weights": list(evaluation["readout_model"].get("weights", [])),
                "bias": evaluation["readout_model"].get("bias"),
                "selected_threshold_summary": dict(evaluation["selected_threshold_summary"]),
                "degeneracy_guard": guard,
            },
            "n_questions": int(evaluation["n_rows"]),
            "n_candidate_rows": int(evaluation["n_candidate_rows"]),
            "evaluation": {
                "calibrated_predictions": list(evaluation["calibrated_predictions"]),
                "powered_d1_predictions": list(evaluation["powered_d1_predictions"]),
                "tuned_sc_predictions": list(evaluation["tuned_sc_predictions"]),
                "paired_correct": {
                    "calibrated": list(evaluation["calibrated_correct"]),
                    "powered_d1": list(evaluation["powered_d1_correct"]),
                    "tuned_sc": list(evaluation["tuned_sc_correct"]),
                    "logistic_baseline": list(evaluation["logistic_correct"]),
                },
            },
            "candidate_feature_summary": {
                "mean_powered_margin": round(_mean([_number(row.get("powered_margin")) for row in rows]), 6),
                "mean_answer_entropy": round(_mean([_number(row.get("answer_entropy")) for row in rows]), 6),
                "low_margin_rate_lt_0p1": round(
                    sum(1 for row in rows if _number(row.get("powered_margin")) < 0.1)
                    / len(rows),
                    6,
                )
                if rows
                else 0.0,
            },
        }
    )
    artifact["reproducibility_checksum"] = reproducibility_checksum(
        {
            "upstream_model_specs": upstream.get("model_specs"),
            "evaluation": artifact["evaluation"],
            "readout": artifact["readout"],
            "spec_refs": SPEC_REFS,
        }
    )
    return artifact


def run(
    *,
    root: Path = REPO_ROOT,
    artifact_path: Path | None = None,
    upstream_loader: UpstreamLoader = default_upstream_loader,
    panel_loader: PanelLoader = default_panel_loader,
    n_folds: int = 5,
    bootstrap_samples: int = 2000,
    logistic_epochs: int = 160,
    now: Clock = time.time,
    write: bool = True,
) -> JsonDict:
    root = Path(root)
    artifact_path = Path(artifact_path) if artifact_path else root / RESULT_RELATIVE_PATH
    start = float(now())
    upstream = upstream_loader(root)
    if not upstream:
        artifact = _base_artifact(
            honest_verdict="blocked_exp5045_artifact_unavailable",
            upstream={},
            duration_s=float(now()) - start,
            calibration_available=False,
        )
        artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
        if write:
            write_json(artifact_path, artifact)
        return artifact
    try:
        rows = panel_loader(root, upstream)
        if len(rows) < 2:
            raise RuntimeError(f"only {len(rows)} calibrated rows available")
        evaluation = evaluate_cross_validated_readout(
            rows,
            n_folds=n_folds,
            seed=RANDOM_SEED,
            bootstrap_samples=bootstrap_samples,
            logistic_epochs=logistic_epochs,
        )
        artifact = _complete_artifact(
            upstream=upstream,
            rows=rows,
            evaluation=evaluation,
            duration_s=float(now()) - start,
        )
    except Exception as exc:
        artifact = _base_artifact(
            honest_verdict="blocked_candidate_energy_panel_unavailable",
            upstream=upstream,
            duration_s=float(now()) - start,
            calibration_available=False,
        )
        artifact["blocked_error"] = f"{type(exc).__name__}: {exc}"[:1000]
        artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    if write:
        write_json(artifact_path, artifact)
    return artifact


def artifact_schema_errors(artifact: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in artifact:
            errors.append(field)
    if artifact.get("spec_refs") != SPEC_REFS:
        errors.append("spec_refs")
    if artifact.get("verifier_is_oracle") is not False:
        errors.append("verifier_is_oracle")
    for field in ("calibration_available", "degeneracy_guard_fired", "headroom_present"):
        if not isinstance(artifact.get(field), bool):
            errors.append(field)
    for field in (
        "calibrated_accuracy",
        "powered_d1_accuracy",
        "mcnemar_p",
        "abstention_rate",
    ):
        value = artifact.get(field)
        if value is not None and not (
            isinstance(value, (int, float)) and 0.0 <= float(value) <= 1.0
        ):
            errors.append(field)
    for field in ("delta_vs_powered_d1", "delta_vs_tuned_sc"):
        if artifact.get(field) is not None and not isinstance(artifact.get(field), (int, float)):
            errors.append(field)
    ci95 = artifact.get("paired_ci95")
    if ci95 is not None and (
        not isinstance(ci95, list)
        or len(ci95) != 2
        or not all(isinstance(value, (int, float)) for value in ci95)
    ):
        errors.append("paired_ci95")
    if not isinstance(artifact.get("model_specs"), Mapping):
        errors.append("model_specs")
    if not isinstance(artifact.get("split_diagnostics"), Mapping):
        errors.append("split_diagnostics")
    if not isinstance(artifact.get("baselines"), Mapping):
        errors.append("baselines")
    if not isinstance(artifact.get("readout"), Mapping):
        errors.append("readout")
    verdict = str(artifact.get("honest_verdict", ""))
    if not verdict.startswith(("blocked_", "complete_", "success_")):
        errors.append("honest_verdict")
    return sorted(set(errors))


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI entrypoint
    _ = argv
    artifact = run()
    errors = artifact_schema_errors(artifact)
    print(
        json.dumps(
            {
                "result_path": str(REPO_ROOT / RESULT_RELATIVE_PATH),
                "honest_verdict": artifact.get("honest_verdict"),
                "calibration_available": artifact.get("calibration_available"),
                "calibrated_accuracy": artifact.get("calibrated_accuracy"),
                "powered_d1_accuracy": artifact.get("powered_d1_accuracy"),
                "delta_vs_powered_d1": artifact.get("delta_vs_powered_d1"),
            },
            sort_keys=True,
        )
    )
    if errors:
        print(f"schema_errors={errors}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv[1:]))
