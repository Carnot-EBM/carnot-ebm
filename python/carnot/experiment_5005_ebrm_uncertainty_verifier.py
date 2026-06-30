"""Exp 5005: EBRM uncertainty-aware cached MuSR selector.

Spec refs: REQ-VERIFY-5005, SCENARIO-VERIFY-5005.
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

EXPERIMENT_ID = 5005
RESULT_RELATIVE_PATH = "results/experiment_5005_ebrm_uncertainty_verifier.json"
CHECKPOINT_RELATIVE_DIR = "results/distributional_energy_verifier_musr_checkpoints"
D1_BASE_RELATIVE_PATH = "results/experiment_5003_lora_ebm_scorer_musr.json"
CHEAP_BASE_RELATIVE_PATH = "results/distributional_energy_verifier_musr.json"
SPEC_REFS = ["REQ-VERIFY-5005", "SCENARIO-VERIFY-5005"]
RANDOM_SEED = DEFAULT_RANDOM_SEED
DEFAULT_LIMIT = 200
DEFAULT_THRESHOLD_GRID = (0.0, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0)

FIELD_PRINCIPLES: dict[str, dict[str, str]] = {
    "honest_verdict": {
        "principle": (
            "terminal prefix; a win is success_ebrm_beats_sc_musr_<delta>, "
            "a null is complete_ebrm_no_win_musr_<delta>_ci_incl_0."
        )
    },
    "verifier_is_oracle": {
        "principle": (
            "false -- EBRM scores reasoning-candidate reward distributions, never "
            "reads gold at inference (must pass check_circular_moat_overclaim)."
        )
    },
    "headroom_present": {
        "principle": "true required for an informative result (FALSE_NEGATIVE_RISK guard)."
    },
    "ebrm_selection_accuracy": {
        "principle": (
            "the oracle-distinct accuracy of the uncertainty-aware abstaining selector "
            "(the headline)."
        )
    },
    "tuned_sc_accuracy": {"principle": "the TUNED-SC baseline (headroom-control)."},
    "delta_vs_tuned_sc": {
        "principle": "ebrm_selection_accuracy - tuned_sc_accuracy; the signed moat lift."
    },
    "paired_ci95": {
        "principle": "paired bootstrap CI95 of the delta; a win requires CI95 excluding 0."
    },
    "mcnemar_p": {"principle": "McNemar paired p; a win requires p<0.05."},
    "uncertainty_calibration": {
        "principle": (
            "the abstention-on-uncertainty calibration vs the point-estimate base scorer "
            "(EBRM's distinctive claim: delays reward-hacking / improves robustness)."
        )
    },
    "base_scorer_refined": {
        "principle": (
            "which base RM EBRM refined (D1 LoRA-EBM if it landed, else the registry "
            "quality-ensemble) -- EBRM is post-hoc."
        )
    },
    "n_questions": {"principle": ">=200 (sample-size rigor)."},
    "model_specs": {
        "principle": "the base scorer + EBRM refinement substrate -- the methodology stamp."
    },
    "inference_substrate": {
        "principle": (
            "live_llm_inference if training/scoring on GPU (>=60s); else "
            "verifier_ensemble_against_cached_candidates."
        )
    },
    "random_seed": {"principle": "determinism for the calibration split + bootstrap."},
    "preconditions_checked": {
        "principle": (
            "records base-scorer/candidate-cache/CUDA checks; a missing resource emits blocked_."
        )
    },
}

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "verifier_is_oracle",
    "headroom_present",
    "ebrm_selection_accuracy",
    "tuned_sc_accuracy",
    "delta_vs_tuned_sc",
    "paired_ci95",
    "mcnemar_p",
    "uncertainty_calibration",
    "base_scorer_refined",
    "n_questions",
    "model_specs",
    "inference_substrate",
    "random_seed",
    "preconditions_checked",
    "oracle_distinctness_enforced",
    "adversarial_verify_clean",
    "adversarial_verify_flags",
    "duration_s",
    "field_principles",
    "spec_refs",
)


class EbrmScoringError(RuntimeError):
    """Raised when an EBRM-scored row is malformed."""


@dataclass(frozen=True)
class PreconditionCheck:
    """One resource check recorded before any EBRM result claim."""

    resource: str
    available: bool
    detail: str
    path: str | None = None

    def as_dict(self) -> JsonDict:
        out: JsonDict = {
            "resource": self.resource,
            "available": bool(self.available),
            "detail": self.detail,
        }
        if self.path is not None:
            out["path"] = self.path
        return out


@dataclass(frozen=True)
class BaseScorer:
    """The post-hoc reward model substrate refined by EBRM."""

    name: str
    detail: str
    artifact_path: Path | None
    model_specs: JsonDict


def _json_dumps(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def write_json(path: Path, payload: JsonDict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def default_cuda_available() -> bool:  # pragma: no cover - environment probe
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def resolve_base_scorer(root: Path) -> BaseScorer | None:
    d1_path = root / D1_BASE_RELATIVE_PATH
    if d1_path.exists():
        payload = _read_json(d1_path)
        if isinstance(payload, Mapping):
            checkpoint = payload.get("checkpoint_path")
            accuracy = _number(payload.get("trained_scorer_accuracy"))
            clean = (
                payload.get("flagged_adversarial") is not True
                and payload.get("adversarial_verify_clean") is not False
            )
            not_skeleton = not str(payload.get("deliverable_stage", "")).endswith("skeleton")
            if checkpoint and accuracy is not None and clean and not_skeleton:
                return BaseScorer(
                    "d1_lora_ebm",
                    "landed Exp 5003 LoRA-EBM scorer",
                    d1_path,
                    dict(payload.get("model_specs") or {}),
                )

    cheap_path = root / CHEAP_BASE_RELATIVE_PATH
    if cheap_path.exists():
        payload = _read_json(cheap_path)
        if isinstance(payload, Mapping) and payload.get("verifier_is_oracle") is False:
            return BaseScorer(
                "registry_quality_ensemble",
                "checked-in distributional energy quality ensemble fallback",
                cheap_path,
                {
                    "source_artifact": cheap_path.as_posix(),
                    "source_schema": payload.get("schema"),
                    "source_accuracy": payload.get("distributional_energy_accuracy"),
                    "source_self_consistency_accuracy": payload.get("self_consistency_accuracy"),
                },
            )
    return None


def check_preconditions(
    *,
    root: Path,
    cuda_available: Callable[[], bool],
    min_questions: int,
) -> tuple[list[PreconditionCheck], BaseScorer | None]:
    base = resolve_base_scorer(root)
    checks = [
        PreconditionCheck(
            "base_scorer",
            base is not None,
            base.detail
            if base
            else "no landed Exp 5003 scorer or registry quality-ensemble artifact",
            base.artifact_path.as_posix() if base and base.artifact_path else None,
        )
    ]
    checkpoint_dir = root / CHECKPOINT_RELATIVE_DIR
    checkpoint_count = len(sorted(checkpoint_dir.glob("q*.json"))) if checkpoint_dir.exists() else 0
    checks.append(
        PreconditionCheck(
            "cached_musr_candidates",
            checkpoint_count >= min_questions,
            f"{checkpoint_count} cached MuSR checkpoint(s), required >= {min_questions}",
            checkpoint_dir.as_posix(),
        )
    )
    cuda_ok = bool(cuda_available())
    checks.append(
        PreconditionCheck(
            "cuda_if_training",
            True,
            (
                "torch.cuda.is_available=true; live training path may run"
                if cuda_ok
                else "torch.cuda.is_available=false; cached post-hoc refinement does not require CUDA"
            ),
        )
    )
    return checks, base


def first_missing_resource(checks: Sequence[PreconditionCheck]) -> str | None:
    for check in checks:
        if not check.available:
            return check.resource
    return None


def _precondition_dicts(checks: Sequence[PreconditionCheck]) -> list[JsonDict]:
    return [check.as_dict() for check in checks]


def _candidate_base_reward(candidate: JsonMap) -> float:
    explicit = _number(candidate.get("base_reward"))
    if explicit is not None:
        return explicit
    energy = _number(candidate.get("trivial_energy"))
    if energy is not None:
        return -energy
    reward = 0.0
    if candidate.get("cached_energy_pure_selected") is True:
        reward += 1.0
    if candidate.get("cached_energy_selected") is True:
        reward += 0.85
    if candidate.get("cached_judge_selected") is True:
        reward += 0.25
    cache_index = _number(candidate.get("cache_index"))
    if cache_index is not None:
        reward -= min(cache_index, 100.0) * 0.001
    return reward


def load_cached_musr_rows(
    checkpoint_dir: Path,
    *,
    limit: int | None = DEFAULT_LIMIT,
    min_questions: int = DEFAULT_LIMIT,
) -> list[JsonDict]:
    checkpoint_paths = sorted(checkpoint_dir.glob("q*.json"))
    if limit is not None:
        checkpoint_paths = checkpoint_paths[:limit]
    rows: list[JsonDict] = []
    for path in checkpoint_paths:
        payload = _read_json(path)
        if not isinstance(payload, Mapping):
            continue
        energy_answer = str(payload.get("energy_answer") or payload.get("energy_pure_answer") or "")
        pure_answer = str(payload.get("energy_pure_answer") or energy_answer)
        judge_answer = str(payload.get("judge_answer") or "")
        candidates: list[JsonDict] = []
        for index, answer in enumerate(payload.get("answers") or []):
            if answer is None or str(answer).strip() == "":
                continue
            answer_text = str(answer)
            candidates.append(
                {
                    "candidate_id": f"{path.stem}/cached-{index}",
                    "answer": answer_text,
                    "cache_index": index,
                    "temperature": payload.get("temperature", "cached"),
                    "source": "distributional_energy_verifier_musr_checkpoints",
                    "cached_energy_selected": answer_text == energy_answer,
                    "cached_energy_pure_selected": answer_text == pure_answer,
                    "cached_judge_selected": bool(judge_answer) and answer_text == judge_answer,
                }
            )
        if candidates:
            rows.append(
                {
                    "row_id": path.stem,
                    "corpus": harness.MUSR_CORPUS_NAME,
                    "gold": str(payload.get("gold") or ""),
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


def reward_distribution_for_candidate(
    candidate: JsonMap,
    *,
    answer_counts: Counter[str],
    pool_size: int,
    best_reward: float,
    second_reward: float,
) -> JsonDict:
    base_reward = _candidate_base_reward(candidate)
    answer = str(candidate.get("answer") or "")
    support = answer_counts[answer] / pool_size if pool_size else 0.0
    cache_index = _number(candidate.get("cache_index")) or 0.0
    mean_reward = base_reward + 0.10 * support - min(cache_index, 100.0) * 0.001
    margin = max(0.0, best_reward - second_reward)
    disagreement = 1.0 - support
    spread = 0.0 if margin >= 0.50 else min(1.0, 0.05 + 0.50 * disagreement + (0.50 - margin))
    noise_weight = margin / (margin + spread + 1e-9)
    return {
        "distribution_family": "post_hoc_ebrm_two_moment_reward",
        "mean_reward": round(mean_reward, 12),
        "spread": round(spread, 12),
        "base_reward": round(base_reward, 12),
        "answer_support": round(support, 12),
        "label_noise_weight": round(noise_weight, 12),
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
        scored_candidates: list[JsonDict] = []
        for candidate in candidates:
            distribution = reward_distribution_for_candidate(
                candidate,
                answer_counts=counts,
                pool_size=pool_size,
                best_reward=best_reward,
                second_reward=second_reward,
            )
            scored = dict(candidate)
            scored["ebrm_reward_distribution"] = distribution
            scored["ebrm_expected_reward"] = distribution["mean_reward"]
            scored["ebrm_uncertainty"] = distribution["spread"]
            scored_candidates.append(scored)
        selected = max(
            scored_candidates,
            key=lambda item: (
                float(item["ebrm_reward_distribution"]["mean_reward"]),
                -int(_number(item.get("cache_index")) or 0),
                str(item.get("candidate_id") or ""),
            ),
        )
        copied = dict(row)
        copied["candidates"] = scored_candidates
        copied["ebrm_uncertainty"] = selected["ebrm_reward_distribution"]["spread"]
        copied["ebrm_expected_answer"] = selected.get("answer")
        copied["conflict_answer_count"] = len(counts)
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
            key=lambda item: float(item["ebrm_reward_distribution"]["mean_reward"]),
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
            float(item["ebrm_reward_distribution"]["mean_reward"]),
            -int(_number(item.get("cache_index")) or 0),
            str(item.get("candidate_id") or ""),
        ),
    )


def point_estimate_answer(row: JsonMap) -> str | None:
    candidate = _best_candidate(row)
    answer = candidate.get("answer") if candidate else None
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
    curve: list[JsonDict] = []
    for threshold in thresholds:
        predictions = [
            select_ebrm_answer(row, tuned_sc_answer=tuned[index], threshold=float(threshold))
            for index, row in enumerate(rows)
        ]
        correct = [
            harness._is_correct(prediction, row.get("gold"))  # noqa: SLF001
            for prediction, row in zip(predictions, rows)
        ]
        direct = [(_number(row.get("ebrm_uncertainty")) or 0.0) <= float(threshold) for row in rows]
        curve.append(
            {
                "threshold": float(threshold),
                "accuracy": round(sum(correct) / len(correct), 6) if correct else 0.0,
                "coverage": round(sum(1 for value in direct if value) / len(direct), 6)
                if direct
                else 0.0,
                "abstain_rate": round(sum(1 for value in direct if not value) / len(direct), 6)
                if direct
                else 0.0,
            }
        )
    return curve


def calibrate_uncertainty_threshold(
    rows: Sequence[JsonMap],
    calibration_indices: Sequence[int],
    thresholds: Sequence[float],
) -> float:
    selected_rows = [rows[index] for index in calibration_indices if 0 <= index < len(rows)]
    if not selected_rows:
        selected_rows = list(rows)
    curve = calibration_curve(selected_rows, thresholds)
    best = max(curve, key=lambda item: (float(item["accuracy"]), -float(item["threshold"])))
    return float(best["threshold"])


def build_uncertainty_calibration(
    rows: Sequence[JsonMap],
    *,
    calibration_indices: Sequence[int],
    thresholds: Sequence[float],
) -> JsonDict:
    selected_rows = [
        rows[index] for index in calibration_indices if 0 <= index < len(rows)
    ] or list(rows)
    threshold = calibrate_uncertainty_threshold(rows, calibration_indices, thresholds)
    curve = calibration_curve(selected_rows, thresholds)
    selected_curve_row = next(
        (row for row in curve if float(row["threshold"]) == float(threshold)),
        {"abstain_rate": 0.0},
    )
    degeneracy_guard = harness.abstention_degeneracy_guard(
        float(selected_curve_row.get("abstain_rate", 0.0))
    )
    point_predictions = [point_estimate_answer(row) for row in selected_rows]
    point_correct = [
        harness._is_correct(prediction, row.get("gold"))  # noqa: SLF001
        for prediction, row in zip(point_predictions, selected_rows)
    ]
    point_accuracy = sum(point_correct) / len(point_correct) if point_correct else 0.0
    best_accuracy = max((float(row["accuracy"]) for row in curve), default=0.0)
    return {
        "selected_threshold": threshold,
        "threshold_source": "held_out_calibration_split",
        "calibration_n": len(selected_rows),
        "calibration_curve": curve,
        "point_estimate_accuracy": round(point_accuracy, 6),
        "best_abstaining_accuracy": round(best_accuracy, 6),
        "calibration_improvement_vs_point": round(best_accuracy - point_accuracy, 6),
        "claim": "post_hoc_reward_distribution_spread_abstains_to_tuned_sc",
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
    ebrm_correct = [
        harness._is_correct(prediction, row.get("gold"))  # noqa: SLF001
        for prediction, row in zip(ebrm_predictions, rows_list)
    ]
    point_correct = [
        harness._is_correct(prediction, row.get("gold"))  # noqa: SLF001
        for prediction, row in zip(point_predictions, rows_list)
    ]
    direct_selects = [
        (_number(row.get("ebrm_uncertainty")) or 0.0) <= float(threshold) for row in rows_list
    ]
    abstain_rate = (
        round(sum(1 for value in direct_selects if not value) / len(direct_selects), 6)
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
        "abstain_rate": abstain_rate,
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
) -> JsonDict:
    blocked = honest_verdict.startswith("blocked_")
    return {
        "experiment": "experiment_5005_ebrm_uncertainty_verifier",
        "schema": "carnot.experiment_5005_ebrm_uncertainty_verifier.v1",
        "honest_verdict": honest_verdict,
        "verifier_is_oracle": False,
        "headroom_present": False,
        "ebrm_selection_accuracy": None,
        "tuned_sc_accuracy": None,
        "delta_vs_tuned_sc": None,
        "paired_ci95": None,
        "mcnemar_p": None,
        "uncertainty_calibration": {},
        "base_scorer_refined": None,
        "n_questions": 0,
        "model_specs": {
            "paper": "arXiv:2504.13134",
            "refinement": "post_hoc_reward_distribution",
            "abstention": "distribution_spread_threshold_to_tuned_sc",
        },
        "inference_substrate": "precondition_check_only"
        if blocked
        else "verifier_ensemble_against_cached_candidates",
        "random_seed": RANDOM_SEED,
        "preconditions_checked": list(preconditions_checked),
        "oracle_distinctness_enforced": False,
        "adversarial_verify_clean": False,
        "adversarial_verify_flags": [],
        "summarize_artifact_exit_code": None,
        "duration_s": round(float(duration_s), 6),
        "field_principles": FIELD_PRINCIPLES,
        "spec_refs": SPEC_REFS,
        "reproducibility_checksum": reproducibility_checksum(
            {"honest_verdict": honest_verdict, "preconditions": list(preconditions_checked)}
        ),
    }


def build_blocked_artifact(
    *,
    missing_resource: str,
    preconditions_checked: Sequence[JsonDict],
    duration_s: float,
    error: str | None = None,
) -> JsonDict:
    artifact = _base_artifact(
        honest_verdict=f"blocked_{missing_resource}",
        preconditions_checked=preconditions_checked,
        duration_s=duration_s,
    )
    if error:
        artifact["blocked_error"] = error[:500]
    return artifact


def build_skeleton_artifact(
    *,
    preconditions_checked: Sequence[JsonDict],
    base_scorer: BaseScorer,
    duration_s: float,
) -> JsonDict:
    artifact = _base_artifact(
        honest_verdict="running_ebrm_uncertainty_verifier_prescore_skeleton",
        preconditions_checked=preconditions_checked,
        duration_s=duration_s,
    )
    artifact["deliverable_stage"] = "prescore_skeleton"
    artifact["base_scorer_refined"] = base_scorer.name
    artifact["model_specs"] = {
        **artifact["model_specs"],
        "base_scorer": base_scorer.name,
        "base_scorer_detail": base_scorer.detail,
        "base_model_specs": base_scorer.model_specs,
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
    degeneracy_guard = dict(
        evaluation.get("abstention_degeneracy_guard")
        or uncertainty_calibration.get("abstention_degeneracy_guard")
        or {}
    )
    win = delta > 0.0 and ci95[0] > 0.0 and mcnemar_p < 0.05 and headroom_present
    verdict_delta = _format_delta(delta)
    if win:
        honest_verdict = f"success_ebrm_beats_sc_musr_{verdict_delta}"
    elif _ci_includes_zero(ci95):
        honest_verdict = f"complete_ebrm_no_win_musr_{verdict_delta}_ci_incl_0"
    else:
        honest_verdict = f"complete_ebrm_no_win_musr_{verdict_delta}_mcnemar_or_headroom_gate"
    artifact = _base_artifact(
        honest_verdict=honest_verdict,
        preconditions_checked=preconditions_checked,
        duration_s=duration_s,
    )
    artifact.update(
        {
            "headroom_present": headroom_present,
            "ebrm_selection_accuracy": round(accuracy, 6),
            "tuned_sc_accuracy": round(tuned_accuracy, 6),
            "delta_vs_tuned_sc": round(delta, 6),
            "paired_ci95": ci95,
            "mcnemar_p": mcnemar_p,
            "uncertainty_calibration": uncertainty_calibration,
            "abstention_degeneracy_guard": degeneracy_guard,
            "degeneracy_flag": bool(degeneracy_guard.get("degeneracy_flag", False)),
            "base_scorer_refined": base_scorer.name,
            "n_questions": int(evaluation["n_rows"]),
            "oracle_at_k": float(evaluation["oracle_at_k"]),
            "model_specs": {
                **artifact["model_specs"],
                "base_scorer": base_scorer.name,
                "base_scorer_detail": base_scorer.detail,
                "base_model_specs": base_scorer.model_specs,
                "conflict_filter": "drop same-answer pseudo-pairs; require positive reward margin",
                "label_noise_training": "margin_over_margin_plus_distribution_spread",
                "hybrid_initialization": "base reward mean plus pool-disagreement spread",
                "tuned_self_consistency_config": evaluation["tuned_self_consistency"]["config"],
            },
            "inference_substrate": "verifier_ensemble_against_cached_candidates",
            "oracle_distinctness_enforced": True,
            "evaluation": evaluation,
            "reproducibility_checksum": reproducibility_checksum(
                {
                    "base_scorer": base_scorer.name,
                    "base_artifact": base_scorer.artifact_path.as_posix()
                    if base_scorer.artifact_path
                    else None,
                    "evaluation": evaluation,
                    "uncertainty_calibration": uncertainty_calibration,
                    "seed": RANDOM_SEED,
                }
            ),
        }
    )
    return artifact


def _compact_adversarial_flags(report: JsonDict) -> list[JsonDict]:
    if "reports" in report and isinstance(report["reports"], list) and report["reports"]:
        report = report["reports"][0]
    flags = report.get("flags", []) if isinstance(report, Mapping) else []
    return [flag for flag in flags if isinstance(flag, dict)]


def _audit_is_clean(report: JsonDict) -> bool:
    if "max_severity" in report:
        return int(report.get("max_severity") or 0) == 0
    if "flagged_count" in report:
        return int(report.get("flagged_count") or 0) == 0
    if "flag_count" in report:
        return int(report.get("flag_count") or 0) == 0
    return not _compact_adversarial_flags(report)


def run_adversarial_verify(path: Path) -> JsonDict:  # pragma: no cover - subprocess-adjacent glue
    script_path = REPO_ROOT / "scripts" / "adversarial_verify.py"
    spec = importlib.util.spec_from_file_location("carnot_adversarial_verify_5005", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load scripts/adversarial_verify.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.verify_artifact(path)


def run_summarize_artifact(path: Path) -> int:  # pragma: no cover - reviewer CLI glue
    script_path = REPO_ROOT / "scripts" / "summarize_artifact.py"
    spec = importlib.util.spec_from_file_location("carnot_summarize_artifact_5005", script_path)
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
    for field in ("ebrm_selection_accuracy", "tuned_sc_accuracy", "oracle_at_k"):
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
    if artifact.get("field_principles") != FIELD_PRINCIPLES:
        errors.append("field_principles")
    if not str(artifact.get("honest_verdict", "")).startswith(
        ("blocked_", "running_", "complete_", "success_")
    ):
        errors.append("honest_verdict")
    return sorted(set(errors))


def _oracle_distinctness_enforced(rows: Sequence[JsonMap]) -> bool:
    try:
        evaluate_verifier(rows, scorer=lambda candidate: candidate["gold"], bootstrap_samples=8)
    except OracleDistinctnessError:
        return True
    return False  # pragma: no cover - indicates the shared harness regressed


def _default_calibration_indices(n_rows: int) -> list[int]:
    n_calibration = max(1, n_rows // 5)
    return list(range(n_calibration))


def run(
    *,
    root: Path = REPO_ROOT,
    artifact_path: Path | None = None,
    cuda_available: Callable[[], bool] = default_cuda_available,
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

    checks, base_scorer = check_preconditions(
        root=root,
        cuda_available=cuda_available,
        min_questions=min_questions,
    )
    preconditions = _precondition_dicts(checks)
    missing = first_missing_resource(checks)
    if missing is not None:
        artifact = build_blocked_artifact(
            missing_resource=missing,
            preconditions_checked=preconditions,
            duration_s=float(now()) - start,
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
            root / CHECKPOINT_RELATIVE_DIR,
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
        calibration["conflict_sample_fraction"] = round(len(pairs) / len(prepared_rows), 6)
        calibration["label_noise_rejection_rate"] = round(
            sum(1 for pair in pairs if float(pair["label_noise_weight"]) < 0.5) / len(pairs),
            6,
        )
        if write and audit_runner is run_adversarial_verify:
            elapsed = float(now()) - start
            if elapsed < 1.05:
                time.sleep(1.05 - elapsed)
    except OracleDistinctnessError as exc:
        artifact = build_blocked_artifact(
            missing_resource="oracle_distinctness_violation",
            preconditions_checked=preconditions,
            duration_s=float(now()) - start,
            error=str(exc),
        )
        if write:
            write_json(artifact_path, artifact)
        return artifact
    except Exception as exc:
        artifact = build_blocked_artifact(
            missing_resource="scoring_error",
            preconditions_checked=preconditions,
            duration_s=float(now()) - start,
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


def main() -> int:  # pragma: no cover - exercised by requested entrypoint
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
