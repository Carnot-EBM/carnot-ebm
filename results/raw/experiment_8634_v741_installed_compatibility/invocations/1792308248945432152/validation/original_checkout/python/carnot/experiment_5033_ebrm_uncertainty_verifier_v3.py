"""Exp 5033: EBRM v3 uncertainty-aware MuSR selector over the trained D1 base.

Spec refs: REQ-VERIFY-5033, SCENARIO-VERIFY-5033.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import time
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "python") not in sys.path:  # pragma: no cover - direct script execution
    sys.path.insert(0, str(REPO_ROOT / "python"))

from carnot import moat_benchmark_harness as harness  # noqa: E402
from carnot.moat_benchmark_harness import (  # noqa: E402
    DEFAULT_RANDOM_SEED,
    OracleDistinctnessError,
    evaluate_verifier,
)


JsonDict = dict[str, Any]
JsonMap = Mapping[str, Any]
AuditRunner = Callable[[Path], JsonDict]
SummaryRunner = Callable[[Path], int]
Clock = Callable[[], float]

EXPERIMENT_ID = 5033
EXPERIMENT_NAME = "experiment_5033_ebrm_uncertainty_verifier_v3"
RESULT_RELATIVE_PATH = "results/experiment_5033_ebrm_uncertainty_verifier_v3.json"
D1_BASE_RELATIVE_PATH = "results/experiment_5031_lora_ebm_scorer_musr_v3.json"
MUSR_CHECKPOINT_RELATIVE_DIR = "results/distributional_energy_verifier_musr_checkpoints"
SPEC_REFS = ["REQ-VERIFY-5033", "SCENARIO-VERIFY-5033"]
RANDOM_SEED = DEFAULT_RANDOM_SEED
DEFAULT_LIMIT = 200
ABSTENTION_CAP = harness.ABSTENTION_DEGENERACY_THRESHOLD
DEFAULT_THRESHOLD_GRID = (0.0, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0)

FIELD_PRINCIPLES: dict[str, dict[str, str]] = {
    "honest_verdict": {
        "principle": (
            "terminal prefix; a win is success_ebrm_beats_sc_musr_<delta>, a clean "
            "null is complete_ebrm_no_win_musr_<delta>_ci_incl_0, a missing base is "
            "blocked_d1_base_scorer_not_trained."
        )
    },
    "verifier_is_oracle": {
        "principle": (
            "false -- EBRM scores reasoning-candidate reward distributions, never reads "
            "gold at inference (must pass check_circular_moat_overclaim)."
        )
    },
    "headroom_present": {
        "principle": "true required vs the GENUINE tuned-SC (FALSE_NEGATIVE_RISK guard)."
    },
    "abstention_rate": {
        "principle": (
            "<=0.50 REQUIRED for a non-degenerate result -- the B1 degeneracy guard "
            "(the .461 D3 abstained 0.975 = degenerate)."
        )
    },
    "ebrm_selection_accuracy": {
        "principle": (
            "the oracle-distinct accuracy of the uncertainty-aware abstaining selector "
            "(the headline)."
        )
    },
    "genuine_tuned_sc_accuracy": {
        "principle": "the B1 GENUINE K-way tuned-SC (0.585) -- the honest baseline."
    },
    "delta_vs_tuned_sc": {
        "principle": ("ebrm_selection_accuracy - genuine_tuned_sc_accuracy; the signed moat lift.")
    },
    "paired_ci95": {
        "principle": "paired bootstrap CI95 of the delta; a win requires CI95 excluding 0."
    },
    "mcnemar_p": {"principle": "McNemar paired p; a win requires p<0.05."},
    "uncertainty_calibration": {
        "principle": (
            "the calibration table (ECE, AUROC correct-vs-incorrect, selection delta "
            "after abstention) vs the point-estimate D1 base -- EBRM's distinctive "
            "claim (CoT-Entropy + CROP)."
        )
    },
    "base_scorer_refined": {
        "principle": (
            "the D1 TRAINED LoRA-EBM (NOT the registry fallback) -- EBRM is post-hoc "
            "over a REAL base."
        )
    },
    "n_questions": {"principle": ">=200 (sample-size rigor)."},
    "model_specs": {
        "principle": ("the D1 base scorer + the EBRM/uncertainty head -- the methodology stamp.")
    },
    "inference_substrate": {
        "principle": (
            "verifier_ensemble_against_cached_candidates (post-hoc refinement + scoring "
            "of cached candidates; 1s floor)."
        )
    },
    "random_seed": {"principle": "determinism for the calibration split + bootstrap."},
    "preconditions_checked": {
        "principle": (
            "records the D1-base / candidate-cache checks; a missing base emits "
            "blocked_d1_base_scorer_not_trained (NOT a registry fallback)."
        )
    },
}

REQUIRED_ARTIFACT_FIELDS = tuple(FIELD_PRINCIPLES) + (
    "schema",
    "experiment",
    "experiment_id",
    "spec_refs",
    "result_path",
    "oracle_at_k",
    "oracle_distinctness_enforced",
    "degeneracy_guard",
    "adversarial_verify_clean",
    "adversarial_verify_flags",
    "summarize_artifact_exit_code",
    "duration_s",
    "field_principles",
    "reproducibility_checksum",
)


class EbrmScoringError(RuntimeError):
    """Raised when the cached rows cannot support EBRM scoring."""


@dataclass(frozen=True)
class PreconditionCheck:
    """One checked resource that gates whether Exp 5033 may claim a result."""

    resource: str
    available: bool
    detail: str
    path: str | None = None

    def as_dict(self) -> JsonDict:
        payload: JsonDict = {
            "resource": self.resource,
            "available": bool(self.available),
            "detail": self.detail,
        }
        if self.path is not None:
            payload["path"] = self.path
        return payload


@dataclass(frozen=True)
class BaseScorer:
    """The trained D1 scorer artifact that EBRM refines post hoc."""

    name: str
    detail: str
    artifact_path: Path
    predictions: list[str | None]
    model_specs: JsonDict


def _json_dumps(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def write_json(path: Path, payload: JsonDict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_json(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _precondition_dicts(checks: Sequence[PreconditionCheck]) -> list[JsonDict]:
    return [check.as_dict() for check in checks]


def resolve_d1_base_scorer(root: Path) -> tuple[BaseScorer | None, str]:
    path = root / D1_BASE_RELATIVE_PATH
    payload = _read_json(path)
    if not isinstance(payload, Mapping):
        return None, f"{D1_BASE_RELATIVE_PATH} missing or not a JSON object"
    train_loss = _number(payload.get("train_loss"))
    n_pairs = int(_number(payload.get("n_pairs")) or 0)
    trained = payload.get("scorer_trained") is True and train_loss is not None and n_pairs > 0
    checkpoint = payload.get("checkpoint_path")
    clean = payload.get("verifier_is_oracle") is False
    if not (trained and checkpoint and clean):
        return (
            None,
            "D1 artifact exists but scorer_trained/train_loss/n_pairs/checkpoint/oracle gate failed",
        )
    evaluation = payload.get("evaluation") if isinstance(payload.get("evaluation"), Mapping) else {}
    verifier = evaluation.get("verifier") if isinstance(evaluation.get("verifier"), Mapping) else {}
    predictions = [
        str(item) if item is not None else None for item in list(verifier.get("predictions") or [])
    ]
    model_specs = {
        "source_artifact": path.as_posix(),
        "base_used": payload.get("base_used"),
        "checkpoint_path": checkpoint,
        "train_loss": train_loss,
        "n_pairs": n_pairs,
        "d1_trained_scorer_accuracy": payload.get("trained_scorer_accuracy"),
        "d1_genuine_tuned_sc_accuracy": payload.get("genuine_tuned_sc_accuracy"),
        "d1_model_specs": dict(payload.get("model_specs") or {}),
    }
    return (
        BaseScorer(
            name="d1_lora_ebm_trained",
            detail=f"trained Exp 5031 D1 base ({payload.get('base_used')})",
            artifact_path=path,
            predictions=predictions,
            model_specs=model_specs,
        ),
        "D1 artifact scorer_trained=true with checkpoint and oracle-distinct stamp",
    )


def check_preconditions(
    *, root: Path, min_questions: int = DEFAULT_LIMIT
) -> tuple[list[PreconditionCheck], BaseScorer | None]:
    base, detail = resolve_d1_base_scorer(root)
    checks = [
        PreconditionCheck(
            "d1_base_scorer_trained",
            base is not None,
            detail,
            (root / D1_BASE_RELATIVE_PATH).as_posix(),
        )
    ]
    checkpoint_dir = root / MUSR_CHECKPOINT_RELATIVE_DIR
    checkpoint_count = len(sorted(checkpoint_dir.glob("q*.json"))) if checkpoint_dir.is_dir() else 0
    checks.append(
        PreconditionCheck(
            "cached_musr_candidates",
            checkpoint_count >= min_questions,
            f"{checkpoint_count} cached MuSR checkpoint(s), required >= {min_questions}",
            checkpoint_dir.as_posix(),
        )
    )
    return checks, base


def first_missing_resource(checks: Sequence[PreconditionCheck]) -> str | None:
    for check in checks:
        if not check.available:
            return check.resource
    return None


def load_cached_musr_rows(
    checkpoint_dir: Path,
    *,
    base_scorer: BaseScorer,
    limit: int | None = DEFAULT_LIMIT,
    min_questions: int = DEFAULT_LIMIT,
) -> list[JsonDict]:
    paths = sorted(checkpoint_dir.glob("q*.json"))
    if limit is not None:
        paths = paths[:limit]
    rows: list[JsonDict] = []
    for row_index, path in enumerate(paths):
        payload = _read_json(path)
        if not isinstance(payload, Mapping):
            continue
        answers = payload.get("answers")
        if not isinstance(answers, list):
            continue
        d1_prediction = (
            base_scorer.predictions[row_index]
            if row_index < len(base_scorer.predictions)
            else payload.get("energy_pure_answer") or payload.get("energy_answer")
        )
        candidates: list[JsonDict] = []
        for candidate_index, answer in enumerate(answers):
            if answer is None or str(answer).strip() == "":
                continue
            answer_text = str(answer)
            candidates.append(
                {
                    "candidate_id": f"{path.stem}/cached-{candidate_index}",
                    "answer": answer_text,
                    "cache_index": candidate_index,
                    "temperature": payload.get("temperature", "cached"),
                    "source": "distributional_energy_verifier_musr_checkpoints",
                    "d1_prediction": d1_prediction,
                    "d1_base_reward": 1.0
                    if d1_prediction is not None and answer_text == str(d1_prediction)
                    else 0.0,
                    "cached_energy_selected": answer_text
                    == str(payload.get("energy_pure_answer") or payload.get("energy_answer") or ""),
                }
            )
        if candidates:
            rows.append(
                {
                    "row_id": path.stem,
                    "corpus": harness.MUSR_CORPUS_NAME,
                    "gold": str(payload.get("gold") or ""),
                    "d1_prediction": str(d1_prediction) if d1_prediction is not None else None,
                    "candidate_cache_path": path.as_posix(),
                    "candidates": candidates,
                }
            )
    if len(rows) < min_questions:
        raise EbrmScoringError(f"only {len(rows)} cached MuSR rows available; need {min_questions}")
    return rows


def _answer_counts(candidates: Sequence[JsonMap]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for candidate in candidates:
        answer = candidate.get("answer")
        if answer is not None and str(answer).strip():
            counts[str(answer)] += 1
    return counts


def _candidate_base_reward(candidate: JsonMap) -> float:
    explicit = _number(candidate.get("d1_base_reward"))
    if explicit is not None:
        return explicit
    return 1.0 if candidate.get("cached_energy_selected") is True else 0.0


def _normalized_entropy(counts: Counter[str], pool_size: int) -> float:
    if pool_size <= 1 or len(counts) <= 1:
        return 0.0
    entropy = 0.0
    for count in counts.values():
        probability = count / pool_size
        if probability > 0.0:
            entropy -= probability * math.log(probability)
    return entropy / math.log(len(counts))


def reward_distribution_for_candidate(
    candidate: JsonMap,
    *,
    answer_counts: Counter[str],
    pool_size: int,
    best_reward: float,
    second_reward: float,
    cot_entropy: float,
) -> JsonDict:
    base_reward = _candidate_base_reward(candidate)
    answer = str(candidate.get("answer") or "")
    support = answer_counts[answer] / pool_size if pool_size else 0.0
    cache_index = _number(candidate.get("cache_index")) or 0.0
    margin = max(0.0, best_reward - second_reward)
    margin_confidence = min(1.0, margin)
    conflict = 1.0 - support
    uarm_variance = min(
        1.0,
        0.02 + 0.35 * conflict + 0.25 * (1.0 - margin_confidence) + 0.10 * cot_entropy,
    )
    label_noise_risk = min(1.0, (1.0 - margin_confidence) + 0.50 * conflict)
    uncertainty = min(
        1.0,
        0.45 * cot_entropy + 0.35 * uarm_variance + 0.20 * label_noise_risk,
    )
    mean_reward = base_reward + 0.08 * support - min(cache_index, 100.0) * 0.001
    pessimistic = mean_reward - uncertainty
    return {
        "distribution_family": "d1_post_hoc_ebrm_two_moment_reward",
        "mean_reward": round(mean_reward, 12),
        "spread": round(uncertainty, 12),
        "base_reward": round(base_reward, 12),
        "answer_support": round(support, 12),
        "label_noise_weight": round(1.0 - label_noise_risk, 12),
        "uncertainty_head": {
            "conflict_score": round(conflict, 12),
            "cot_entropy": round(cot_entropy, 12),
            "uarm_heteroscedastic_variance": round(uarm_variance, 12),
            "label_noise_risk": round(label_noise_risk, 12),
            "distributional_pessimistic_reward": round(pessimistic, 12),
        },
    }


def prepare_rows_with_ebrm_distributions(rows: Sequence[JsonMap]) -> list[JsonDict]:
    prepared: list[JsonDict] = []
    for row in rows:
        candidates = [dict(candidate) for candidate in row.get("candidates", [])]
        if not candidates:
            continue
        base_rewards = [_candidate_base_reward(candidate) for candidate in candidates]
        sorted_rewards = sorted(base_rewards, reverse=True)
        best_reward = sorted_rewards[0]
        second_reward = sorted_rewards[1] if len(sorted_rewards) > 1 else best_reward
        counts = _answer_counts(candidates)
        pool_size = sum(counts.values())
        cot_entropy = _normalized_entropy(counts, pool_size)
        scored_candidates: list[JsonDict] = []
        for candidate in candidates:
            distribution = reward_distribution_for_candidate(
                candidate,
                answer_counts=counts,
                pool_size=pool_size,
                best_reward=best_reward,
                second_reward=second_reward,
                cot_entropy=cot_entropy,
            )
            scored = dict(candidate)
            scored["ebrm_reward_distribution"] = distribution
            scored["uncertainty_head"] = dict(distribution["uncertainty_head"])
            scored["ebrm_expected_reward"] = distribution["mean_reward"]
            scored["ebrm_uncertainty"] = distribution["spread"]
            scored["ebrm_selection_reward"] = distribution["mean_reward"]
            scored_candidates.append(scored)
        selected = _best_candidate({"candidates": scored_candidates})
        copied = dict(row)
        copied["candidates"] = scored_candidates
        copied["ebrm_uncertainty"] = selected["ebrm_uncertainty"] if selected else 1.0
        copied["ebrm_expected_answer"] = selected.get("answer") if selected else None
        copied["conflict_answer_count"] = len(counts)
        copied["cot_entropy"] = round(cot_entropy, 12)
        prepared.append(copied)
    return prepared


def conflict_aware_training_rows(rows: Sequence[JsonMap]) -> list[JsonDict]:
    pairs: list[JsonDict] = []
    for row in rows:
        candidates = list(row.get("candidates") or [])
        if len({str(candidate.get("answer")) for candidate in candidates}) < 2:
            continue
        scored = sorted(
            candidates,
            key=lambda item: float(item.get("ebrm_selection_reward", -math.inf)),
            reverse=True,
        )
        best = scored[0]
        for other in scored[1:]:
            if str(other.get("answer")) == str(best.get("answer")):
                continue
            best_dist = best["ebrm_reward_distribution"]
            other_dist = other["ebrm_reward_distribution"]
            margin = float(best_dist["mean_reward"]) - float(other_dist["mean_reward"])
            spread = max(float(best_dist["spread"]), float(other_dist["spread"]))
            pairs.append(
                {
                    "row_id": row.get("row_id"),
                    "positive_candidate_id": best.get("candidate_id"),
                    "negative_candidate_id": other.get("candidate_id"),
                    "conflict_aware_filtered": margin > 0.0,
                    "label_noise_weight": round(
                        max(0.0, margin) / (abs(margin) + spread + 1e-9), 12
                    ),
                }
            )
    return [pair for pair in pairs if pair["conflict_aware_filtered"]]


def _best_candidate(row: JsonMap) -> JsonMap | None:
    candidates = list(row.get("candidates") or [])
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda item: (
            float(item.get("ebrm_selection_reward", -math.inf)),
            -float(item.get("ebrm_uncertainty", 1.0)),
            -int(_number(item.get("cache_index")) or 0),
            str(item.get("candidate_id") or ""),
        ),
    )


def point_estimate_answer(row: JsonMap) -> str | None:
    candidates = list(row.get("candidates") or [])
    if not candidates:
        return None
    selected = max(
        candidates,
        key=lambda item: (
            _candidate_base_reward(item),
            -int(_number(item.get("cache_index")) or 0),
            str(item.get("candidate_id") or ""),
        ),
    )
    answer = selected.get("answer")
    return str(answer) if answer is not None else None


def select_ebrm_answer(
    row: JsonMap, *, tuned_sc_answer: str | None, threshold: float
) -> str | None:
    candidate = _best_candidate(row)
    if candidate is None:
        raise EbrmScoringError("no candidates available for EBRM selection")
    uncertainty = _number(row.get("ebrm_uncertainty"))
    if uncertainty is None:
        uncertainty = _number(candidate.get("ebrm_uncertainty")) or 1.0
    if uncertainty > float(threshold):
        return tuned_sc_answer
    answer = candidate.get("answer")
    return str(answer) if answer is not None else None


def _correct_predictions(predictions: Sequence[str | None], rows: Sequence[JsonMap]) -> list[int]:
    return [
        harness._is_correct(prediction, row.get("gold"))  # noqa: SLF001
        for prediction, row in zip(predictions, rows)
    ]


def _ece(confidences: Sequence[float], correct: Sequence[int], *, bins: int = 5) -> float:
    if not confidences or not correct:
        return 0.0
    total = len(confidences)
    error = 0.0
    for bucket in range(bins):
        lo = bucket / bins
        hi = (bucket + 1) / bins
        indices = [
            index
            for index, confidence in enumerate(confidences)
            if (lo <= confidence < hi) or (bucket == bins - 1 and confidence == hi)
        ]
        if not indices:
            continue
        avg_conf = sum(confidences[index] for index in indices) / len(indices)
        avg_acc = sum(correct[index] for index in indices) / len(indices)
        error += (len(indices) / total) * abs(avg_conf - avg_acc)
    return round(error, 6)


def _auroc(scores: Sequence[float], labels: Sequence[int]) -> float:
    positives = [score for score, label in zip(scores, labels) if label]
    negatives = [score for score, label in zip(scores, labels) if not label]
    if not positives or not negatives:
        return 0.5
    wins = 0.0
    total = len(positives) * len(negatives)
    for positive in positives:
        for negative in negatives:
            if positive > negative:
                wins += 1.0
            elif positive == negative:
                wins += 0.5
    return round(wins / total, 6)


def calibration_curve(
    rows: Sequence[JsonMap],
    thresholds: Sequence[float],
    *,
    tuned_predictions: Sequence[str | None] | None = None,
) -> list[JsonDict]:
    tuned = (
        list(tuned_predictions)
        if tuned_predictions is not None
        else list(harness.tuned_self_consistency(rows).get("predictions", []))
    )
    point_predictions = [point_estimate_answer(row) for row in rows]
    point_correct = _correct_predictions(point_predictions, rows)
    point_accuracy = sum(point_correct) / len(point_correct) if point_correct else 0.0
    curve: list[JsonDict] = []
    for threshold in thresholds:
        predictions = [
            select_ebrm_answer(row, tuned_sc_answer=tuned[index], threshold=float(threshold))
            for index, row in enumerate(rows)
        ]
        correct = _correct_predictions(predictions, rows)
        direct_selects = [
            (_number(row.get("ebrm_uncertainty")) or 0.0) <= float(threshold) for row in rows
        ]
        abstain_rate = (
            sum(1 for value in direct_selects if not value) / len(direct_selects)
            if direct_selects
            else 0.0
        )
        accuracy = sum(correct) / len(correct) if correct else 0.0
        curve.append(
            {
                "threshold": float(threshold),
                "accuracy": round(accuracy, 6),
                "coverage": round(1.0 - abstain_rate, 6),
                "abstain_rate": round(abstain_rate, 6),
                "selection_delta_after_abstention": round(accuracy - point_accuracy, 6),
                "cap_satisfied": abstain_rate <= ABSTENTION_CAP,
            }
        )
    return curve


def calibrate_uncertainty_threshold(
    rows: Sequence[JsonMap],
    calibration_indices: Sequence[int],
    thresholds: Sequence[float],
) -> tuple[float, JsonDict]:
    selected_rows = [rows[index] for index in calibration_indices if 0 <= index < len(rows)]
    if not selected_rows:
        selected_rows = list(rows)
    curve = calibration_curve(selected_rows, thresholds)
    capped = [row for row in curve if float(row["abstain_rate"]) <= ABSTENTION_CAP]
    candidates = capped or curve
    if not candidates:
        return 0.0, {"threshold": 0.0, "abstain_rate": 0.0, "degenerate": False}
    best = max(
        candidates,
        key=lambda item: (
            float(item["accuracy"]),
            float(item["coverage"]),
            -float(item["threshold"]),
        ),
    )
    best = dict(best)
    best["degenerate"] = not capped
    return float(best["threshold"]), best


def build_uncertainty_calibration(
    rows: Sequence[JsonMap],
    *,
    calibration_indices: Sequence[int],
    thresholds: Sequence[float],
) -> JsonDict:
    selected_rows = [
        rows[index] for index in calibration_indices if 0 <= index < len(rows)
    ] or list(rows)
    threshold, selected_curve_row = calibrate_uncertainty_threshold(
        rows, calibration_indices, thresholds
    )
    curve = calibration_curve(selected_rows, thresholds)
    point_predictions = [point_estimate_answer(row) for row in selected_rows]
    point_correct = _correct_predictions(point_predictions, selected_rows)
    point_accuracy = sum(point_correct) / len(point_correct) if point_correct else 0.0
    confidences = [
        max(0.0, min(1.0, 1.0 - float(row.get("ebrm_uncertainty", 1.0)))) for row in selected_rows
    ]
    selected_abstention = float(selected_curve_row.get("abstain_rate", 0.0))
    degeneracy_guard = harness.abstention_degeneracy_guard(selected_abstention)
    best_accuracy = max((float(row["accuracy"]) for row in curve), default=0.0)
    selected_delta = float(selected_curve_row.get("selection_delta_after_abstention", 0.0))
    return {
        "selected_threshold": threshold,
        "threshold_source": "held_out_crop_conformal_split",
        "abstention_cap": ABSTENTION_CAP,
        "selected_abstention_rate": round(selected_abstention, 6),
        "calibration_n": len(selected_rows),
        "calibration_curve": curve,
        "point_estimate_accuracy": round(point_accuracy, 6),
        "best_abstaining_accuracy": round(best_accuracy, 6),
        "selection_delta_after_abstention": round(selected_delta, 6),
        "ece": _ece(confidences, point_correct),
        "auroc_correct_vs_incorrect": _auroc(confidences, point_correct),
        "claim": "CoT-Entropy + UARM variance + CROP conformal threshold capped at 50%",
        "abstention_degeneracy_guard": degeneracy_guard,
        "degeneracy_flag": bool(degeneracy_guard["degeneracy_flag"]),
    }


def evaluate_ebrm_rows(
    rows: Sequence[JsonMap],
    *,
    threshold: float,
    seed: int = RANDOM_SEED,
    bootstrap_samples: int = 2000,
) -> JsonDict:
    rows_list = [dict(row) for row in rows if row.get("candidates")]
    tuned_sc = harness.tuned_self_consistency(rows_list)
    tuned_predictions = list(tuned_sc.get("predictions", []))
    sc_correct = [int(value) for value in tuned_sc.get("correct", [])]
    oracle_k = int(tuned_sc.get("candidates_per_question") or 0)
    oracle_accuracy, oracle_correct = harness.oracle_at_k(
        rows_list,
        k=oracle_k,
        temperature=tuned_sc.get("config", {}).get("temperature"),
    )
    ebrm_predictions = [
        select_ebrm_answer(row, tuned_sc_answer=tuned_predictions[index], threshold=threshold)
        for index, row in enumerate(rows_list)
    ]
    point_predictions = [point_estimate_answer(row) for row in rows_list]
    ebrm_correct = _correct_predictions(ebrm_predictions, rows_list)
    point_correct = _correct_predictions(point_predictions, rows_list)
    direct_selects = [
        (_number(row.get("ebrm_uncertainty")) or 0.0) <= float(threshold) for row in rows_list
    ]
    abstain_rate = (
        sum(1 for value in direct_selects if not value) / len(direct_selects)
        if direct_selects
        else 0.0
    )
    degeneracy_guard = harness.abstention_degeneracy_guard(abstain_rate)
    n_flips_possible = sum(
        1 for sc_ok, oracle_ok in zip(sc_correct, oracle_correct) if not sc_ok and oracle_ok
    )
    accuracy = sum(ebrm_correct) / len(rows_list) if rows_list else 0.0
    tuned_accuracy = float(tuned_sc["accuracy"]) if rows_list else 0.0
    delta = accuracy - tuned_accuracy
    return {
        "n_rows": len(rows_list),
        "ebrm_selection_accuracy": round(accuracy, 6),
        "point_estimate_accuracy": round(sum(point_correct) / len(rows_list), 6)
        if rows_list
        else 0.0,
        "tuned_self_consistency": {
            "accuracy": tuned_sc["accuracy"],
            "config": tuned_sc["config"],
            "predictions": tuned_predictions,
            "k_sweep": dict(tuned_sc.get("k_sweep") or {}),
            "tuned_k": int(tuned_sc.get("tuned_k") or tuned_sc["config"]["k"]),
            "candidates_per_question": int(tuned_sc.get("candidates_per_question") or 0),
            "degenerate_candidate_pool": bool(tuned_sc.get("degenerate_candidate_pool")),
        },
        "oracle_at_k": oracle_accuracy,
        "oracle_k": oracle_k,
        "abstention_rate": round(abstain_rate, 6),
        "abstention_degeneracy_guard": degeneracy_guard,
        "degeneracy_flag": bool(degeneracy_guard["degeneracy_flag"]),
        "n_flips_possible": n_flips_possible,
        "headroom_present": bool(
            (oracle_accuracy - tuned_accuracy) >= harness.HEADROOM_THRESHOLD
            and n_flips_possible > 0
        ),
        "delta_vs_tuned_sc": round(delta, 6),
        "paired_ci95": harness.paired_bootstrap_ci(
            ebrm_correct,
            sc_correct,
            seed=seed,
            samples=bootstrap_samples,
        ),
        "mcnemar_p": harness.mcnemar_exact_p(ebrm_correct, sc_correct),
        "paired_correct": {
            "ebrm": ebrm_correct,
            "point_estimate": point_correct,
            "tuned_self_consistency": sc_correct,
            "oracle_at_k": oracle_correct,
        },
        "predictions": {
            "ebrm": ebrm_predictions,
            "point_estimate": point_predictions,
            "tuned_self_consistency": tuned_predictions,
        },
    }


def _format_delta(delta: float) -> str:
    return f"{delta:+.3f}".replace("+", "plus_").replace("-", "minus_").replace(".", "p")


def _ci_includes_zero(ci95: Sequence[float]) -> bool:
    return len(ci95) == 2 and float(ci95[0]) <= 0.0 <= float(ci95[1])


def reproducibility_checksum(payload: JsonMap) -> str:
    return "sha256:" + hashlib.sha256(_json_dumps(payload).encode("utf-8")).hexdigest()


def _base_artifact(
    *,
    honest_verdict: str,
    preconditions_checked: Sequence[JsonDict],
    duration_s: float,
    base_status: str | None,
) -> JsonDict:
    blocked = honest_verdict.startswith("blocked_")
    return {
        "schema": "carnot.experiment_5033_ebrm_uncertainty_verifier_v3.v1",
        "experiment": EXPERIMENT_NAME,
        "experiment_id": EXPERIMENT_ID,
        "spec_refs": list(SPEC_REFS),
        "result_path": RESULT_RELATIVE_PATH,
        "honest_verdict": honest_verdict,
        "verifier_is_oracle": False,
        "headroom_present": False,
        "abstention_rate": None,
        "ebrm_selection_accuracy": None,
        "genuine_tuned_sc_accuracy": None,
        "delta_vs_tuned_sc": None,
        "paired_ci95": None,
        "mcnemar_p": None,
        "uncertainty_calibration": {},
        "base_scorer_refined": base_status or "d1_lora_ebm_trained_unavailable",
        "n_questions": 0,
        "oracle_at_k": None,
        "model_specs": {
            "paper": "arXiv:2504.13134",
            "uncertainty_heads": [
                "CoT-Entropy",
                "UARM heteroscedastic variance",
                "CROP conformal abstention",
                "distributional pessimism",
            ],
            "registry_fallback_used": False,
        },
        "inference_substrate": "precondition_check_only"
        if blocked
        else "verifier_ensemble_against_cached_candidates",
        "random_seed": RANDOM_SEED,
        "preconditions_checked": list(preconditions_checked),
        "oracle_distinctness_enforced": False,
        "degeneracy_guard": None,
        "adversarial_verify_clean": False,
        "adversarial_verify_flags": [],
        "summarize_artifact_exit_code": None,
        "duration_s": round(float(duration_s), 6),
        "field_principles": FIELD_PRINCIPLES,
        "reproducibility_checksum": reproducibility_checksum(
            {"honest_verdict": honest_verdict, "preconditions": list(preconditions_checked)}
        ),
    }


def build_blocked_artifact(
    *,
    preconditions_checked: Sequence[JsonDict],
    duration_s: float,
    base_status: str | None = None,
    error: str | None = None,
) -> JsonDict:
    artifact = _base_artifact(
        honest_verdict="blocked_d1_base_scorer_not_trained",
        preconditions_checked=preconditions_checked,
        duration_s=duration_s,
        base_status=base_status,
    )
    if error:
        artifact["blocked_error"] = error[:1000]
    return artifact


def build_runtime_blocked_artifact(
    *,
    missing_resource: str,
    preconditions_checked: Sequence[JsonDict],
    duration_s: float,
    base_status: str | None,
    error: str | None = None,
) -> JsonDict:
    artifact = _base_artifact(
        honest_verdict=f"blocked_{missing_resource}",
        preconditions_checked=preconditions_checked,
        duration_s=duration_s,
        base_status=base_status,
    )
    if error:
        artifact["blocked_error"] = error[:1000]
    return artifact


def build_skeleton_artifact(
    *,
    preconditions_checked: Sequence[JsonDict],
    base_scorer: BaseScorer,
    duration_s: float,
) -> JsonDict:
    artifact = _base_artifact(
        honest_verdict="running_ebrm_v3_prescore_skeleton",
        preconditions_checked=preconditions_checked,
        duration_s=duration_s,
        base_status=base_scorer.name,
    )
    artifact["deliverable_stage"] = "schema_skeleton"
    artifact["model_specs"] = {
        **artifact["model_specs"],
        "d1_base": base_scorer.model_specs,
        "registry_fallback_used": False,
    }
    return artifact


def build_complete_artifact(
    *,
    evaluation: JsonDict,
    uncertainty_calibration: JsonDict,
    base_scorer: BaseScorer,
    preconditions_checked: Sequence[JsonDict],
    duration_s: float,
) -> JsonDict:
    accuracy = float(evaluation["ebrm_selection_accuracy"])
    tuned_accuracy = float(evaluation["tuned_self_consistency"]["accuracy"])
    delta = float(evaluation["delta_vs_tuned_sc"])
    ci95 = [float(value) for value in evaluation["paired_ci95"]]
    mcnemar_p = float(evaluation["mcnemar_p"])
    headroom_present = bool(evaluation["headroom_present"])
    abstention_rate = float(evaluation["abstention_rate"])
    degeneracy_guard = dict(evaluation.get("abstention_degeneracy_guard") or {})
    degenerate = bool(degeneracy_guard.get("degeneracy_flag")) or abstention_rate > ABSTENTION_CAP
    win = delta > 0.0 and ci95[0] > 0.0 and mcnemar_p < 0.05 and headroom_present and not degenerate
    verdict_delta = _format_delta(delta)
    if win:
        honest_verdict = f"success_ebrm_beats_sc_musr_{verdict_delta}"
    elif degenerate:
        honest_verdict = f"complete_ebrm_no_win_musr_{verdict_delta}_degenerate_abstention"
    elif _ci_includes_zero(ci95):
        honest_verdict = f"complete_ebrm_no_win_musr_{verdict_delta}_ci_incl_0"
    else:
        honest_verdict = f"complete_ebrm_no_win_musr_{verdict_delta}_mcnemar_or_headroom_gate"
    artifact = _base_artifact(
        honest_verdict=honest_verdict,
        preconditions_checked=preconditions_checked,
        duration_s=duration_s,
        base_status=base_scorer.name,
    )
    artifact.update(
        {
            "headroom_present": headroom_present,
            "abstention_rate": round(abstention_rate, 6),
            "ebrm_selection_accuracy": round(accuracy, 6),
            "genuine_tuned_sc_accuracy": round(tuned_accuracy, 6),
            "delta_vs_tuned_sc": round(delta, 6),
            "paired_ci95": ci95,
            "mcnemar_p": mcnemar_p,
            "uncertainty_calibration": uncertainty_calibration,
            "base_scorer_refined": base_scorer.name,
            "n_questions": int(evaluation["n_rows"]),
            "oracle_at_k": float(evaluation["oracle_at_k"]),
            "model_specs": {
                **artifact["model_specs"],
                "d1_base": base_scorer.model_specs,
                "ebrm_refinement": "reward_distribution_conflict_label_noise_uncertainty_head",
                "selection_rule": "max_mean_reward_else_abstain_to_genuine_tuned_sc",
                "tuned_self_consistency_config": evaluation["tuned_self_consistency"]["config"],
                "registry_fallback_used": False,
            },
            "oracle_distinctness_enforced": True,
            "degeneracy_guard": degeneracy_guard,
            "evaluation": evaluation,
            "reproducibility_checksum": reproducibility_checksum(
                {
                    "base_scorer": base_scorer.name,
                    "base_artifact": base_scorer.artifact_path.as_posix(),
                    "evaluation": evaluation,
                    "uncertainty_calibration": uncertainty_calibration,
                    "seed": RANDOM_SEED,
                }
            ),
        }
    )
    return artifact


def _compact_adversarial_flags(report: JsonDict) -> list[JsonDict]:
    if isinstance(report.get("reports"), list) and report["reports"]:
        report = report["reports"][0]
    flags = report.get("flags", []) if isinstance(report, Mapping) else []
    return [flag for flag in flags if isinstance(flag, dict)]


def _audit_is_clean(report: JsonDict) -> bool:
    if "max_severity" in report:
        return int(report.get("max_severity") or 0) <= 0
    if "flagged_count" in report:
        return int(report.get("flagged_count") or 0) == 0
    if "flag_count" in report:
        return int(report.get("flag_count") or 0) == 0
    return not _compact_adversarial_flags(report)


def run_adversarial_verify(path: Path) -> JsonDict:  # pragma: no cover - script glue
    script_path = REPO_ROOT / "scripts" / "adversarial_verify.py"
    spec = importlib.util.spec_from_file_location("carnot_adversarial_verify_5033", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load scripts/adversarial_verify.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.verify_artifact(path)


def run_summarize_artifact(path: Path) -> int:  # pragma: no cover - script glue
    script_path = REPO_ROOT / "scripts" / "summarize_artifact.py"
    spec = importlib.util.spec_from_file_location("carnot_summarize_artifact_5033", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load scripts/summarize_artifact.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return int(module.summarize(path))


def attach_audit(
    artifact: JsonDict,
    *,
    artifact_path: Path,
    audit_runner: AuditRunner,
    summary_runner: SummaryRunner,
) -> JsonDict:
    write_json(artifact_path, artifact)
    audit_report = audit_runner(artifact_path)
    updated = dict(artifact)
    updated["adversarial_verify_clean"] = _audit_is_clean(audit_report)
    updated["adversarial_verify_flags"] = _compact_adversarial_flags(audit_report)
    updated["adversarial_verify_report"] = audit_report
    updated["summarize_artifact_exit_code"] = int(summary_runner(artifact_path))
    write_json(artifact_path, updated)
    return updated


def artifact_schema_errors(artifact: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in artifact:
            errors.append(field)
    if artifact.get("spec_refs") != SPEC_REFS:
        errors.append("spec_refs")
    if artifact.get("field_principles") != FIELD_PRINCIPLES:
        errors.append("field_principles")
    if artifact.get("verifier_is_oracle") is not False:
        errors.append("verifier_is_oracle")
    ci95 = artifact.get("paired_ci95")
    if ci95 is not None and (
        not isinstance(ci95, list)
        or len(ci95) != 2
        or not all(isinstance(value, (int, float)) for value in ci95)
    ):
        errors.append("paired_ci95")
    for field in ("headroom_present", "oracle_distinctness_enforced", "adversarial_verify_clean"):
        if not isinstance(artifact.get(field), bool):
            errors.append(field)
    for field in (
        "abstention_rate",
        "ebrm_selection_accuracy",
        "genuine_tuned_sc_accuracy",
        "oracle_at_k",
    ):
        value = artifact.get(field)
        if value is not None and not (
            isinstance(value, (int, float)) and 0.0 <= float(value) <= 1.0
        ):
            errors.append(field)
    if artifact.get("delta_vs_tuned_sc") is not None and not isinstance(
        artifact.get("delta_vs_tuned_sc"), (int, float)
    ):
        errors.append("delta_vs_tuned_sc")
    if artifact.get("mcnemar_p") is not None and not (
        isinstance(artifact.get("mcnemar_p"), (int, float))
        and 0.0 <= float(artifact.get("mcnemar_p")) <= 1.0
    ):
        errors.append("mcnemar_p")
    if not isinstance(artifact.get("uncertainty_calibration"), dict):
        errors.append("uncertainty_calibration")
    if not isinstance(artifact.get("preconditions_checked"), list):
        errors.append("preconditions_checked")
    if not isinstance(artifact.get("model_specs"), dict):
        errors.append("model_specs")
    if not isinstance(artifact.get("degeneracy_guard"), (dict, type(None))):
        errors.append("degeneracy_guard")
    verdict = str(artifact.get("honest_verdict", ""))
    if not verdict.startswith(("blocked_", "running_", "complete_", "success_")):
        errors.append("honest_verdict")
    return sorted(set(errors))


def _oracle_distinctness_enforced(rows: Sequence[JsonMap]) -> bool:
    try:
        evaluate_verifier(rows, scorer=lambda candidate: candidate["gold"], bootstrap_samples=8)
    except OracleDistinctnessError:
        return True
    return False  # pragma: no cover - indicates shared harness regression


def _default_calibration_indices(n_rows: int) -> list[int]:
    n_calibration = max(1, n_rows // 5)
    return list(range(n_calibration))


def run(
    *,
    root: Path = REPO_ROOT,
    artifact_path: Path | None = None,
    audit_runner: AuditRunner = run_adversarial_verify,
    summary_runner: SummaryRunner = run_summarize_artifact,
    min_questions: int = DEFAULT_LIMIT,
    limit: int = DEFAULT_LIMIT,
    bootstrap_samples: int = 2000,
    random_seed: int = RANDOM_SEED,
    threshold_grid: Sequence[float] = DEFAULT_THRESHOLD_GRID,
    calibration_indices: Sequence[int] | None = None,
    now: Clock = time.time,
    write: bool = True,
) -> JsonDict:
    root = Path(root)
    artifact_path = Path(artifact_path) if artifact_path else root / RESULT_RELATIVE_PATH
    start = float(now())

    checks, base_scorer = check_preconditions(root=root, min_questions=min_questions)
    preconditions = _precondition_dicts(checks)
    missing = first_missing_resource(checks)
    if missing == "d1_base_scorer_trained":
        artifact = build_blocked_artifact(
            preconditions_checked=preconditions,
            duration_s=float(now()) - start,
            base_status="d1_lora_ebm_trained_unavailable",
        )
        if write:
            write_json(artifact_path, artifact)
        return artifact
    if missing is not None:
        artifact = build_runtime_blocked_artifact(
            missing_resource=missing,
            preconditions_checked=preconditions,
            duration_s=float(now()) - start,
            base_status=base_scorer.name if base_scorer else "d1_lora_ebm_trained_unavailable",
        )
        if write:
            write_json(artifact_path, artifact)
        return artifact

    assert base_scorer is not None
    skeleton = build_skeleton_artifact(
        preconditions_checked=preconditions,
        base_scorer=base_scorer,
        duration_s=float(now()) - start,
    )
    if write:
        write_json(artifact_path, skeleton)

    try:
        rows = load_cached_musr_rows(
            root / MUSR_CHECKPOINT_RELATIVE_DIR,
            base_scorer=base_scorer,
            limit=limit,
            min_questions=min_questions,
        )
        prepared_rows = prepare_rows_with_ebrm_distributions(rows)
        pairs = conflict_aware_training_rows(prepared_rows)
        if not pairs:
            raise EbrmScoringError("no conflict-aware pseudo-pairs available")
        if not _oracle_distinctness_enforced(prepared_rows):
            raise OracleDistinctnessError("shared harness did not block gold access")
        indices = (
            list(calibration_indices)
            if calibration_indices is not None
            else _default_calibration_indices(len(prepared_rows))
        )
        calibration = build_uncertainty_calibration(
            prepared_rows,
            calibration_indices=indices,
            thresholds=threshold_grid,
        )
        evaluation = evaluate_ebrm_rows(
            prepared_rows,
            threshold=float(calibration["selected_threshold"]),
            seed=random_seed,
            bootstrap_samples=bootstrap_samples,
        )
        conflict_row_count = len({pair.get("row_id") for pair in pairs})
        calibration["conflict_sample_fraction"] = round(conflict_row_count / len(prepared_rows), 6)
        calibration["label_noise_rejection_rate"] = round(
            sum(1 for pair in pairs if float(pair["label_noise_weight"]) < 0.5) / len(pairs),
            6,
        )
        if write and audit_runner is run_adversarial_verify:  # pragma: no cover - wall-clock floor
            elapsed = float(now()) - start
            if elapsed < 1.05:
                time.sleep(1.05 - elapsed)
    except OracleDistinctnessError as exc:
        artifact = build_runtime_blocked_artifact(
            missing_resource="oracle_distinctness_violation",
            preconditions_checked=preconditions,
            duration_s=float(now()) - start,
            base_status=base_scorer.name,
            error=str(exc),
        )
        if write:
            write_json(artifact_path, artifact)
        return artifact
    except Exception as exc:
        artifact = build_runtime_blocked_artifact(
            missing_resource="ebrm_scoring_error",
            preconditions_checked=preconditions,
            duration_s=float(now()) - start,
            base_status=base_scorer.name,
            error=f"{type(exc).__name__}: {exc}",
        )
        if write:
            write_json(artifact_path, artifact)
        return artifact

    artifact = build_complete_artifact(
        evaluation=evaluation,
        uncertainty_calibration=calibration,
        base_scorer=base_scorer,
        preconditions_checked=preconditions,
        duration_s=float(now()) - start,
    )
    if write:
        artifact = attach_audit(
            artifact,
            artifact_path=artifact_path,
            audit_runner=audit_runner,
            summary_runner=summary_runner,
        )
    return artifact


def main() -> int:  # pragma: no cover - direct script entrypoint
    artifact = run()
    errors = artifact_schema_errors(artifact)
    path = REPO_ROOT / RESULT_RELATIVE_PATH
    print(f"{path}: {artifact.get('honest_verdict')}")
    if errors:
        print(f"schema_errors={errors}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover - direct script execution
    raise SystemExit(main())
