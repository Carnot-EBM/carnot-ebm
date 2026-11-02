"""Exp 3682 discrimination-vs-selection gap diagnosis.

This module tests whether the Exp 3672 result is just a calibration problem or
the reward-model selection crisis pattern: good per-candidate discrimination
but poor best-of-N selection. It uses cached candidates only.

Spec: REQ-VERIFY-3682, SCENARIO-VERIFY-3682.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any

from carnot.phase3.p01_energy_vote_scoring import mcnemar_exact, paired_bootstrap_ci
from carnot.verify.ensemble_selection_sc_weak_3672 import (
    DEFAULT_CORPUS_PATHS,
    DEFAULT_MAX_MAJORITY_SUPPORTS,
    DEFAULT_MAX_SC_ACCURACY,
    DEFAULT_MIN_CANDIDATES,
    DEFAULT_MIN_EXAMPLES,
    DEFAULT_N_BOOT,
    Candidate,
    ProblemRecord,
    compute_regime_stats,
    load_multicandidate_records,
    majority_vote_with_support,
    make_default_energy_scorer,
    select_sc_weak_regime,
)


OUTPUT_REL_PATH = Path("results/experiment_3682_discrimination_vs_selection_gap.json")
RANDOM_SEED = int(
    hashlib.sha256(b"exp=3682;discrimination-vs-selection-gap").hexdigest()[:8],
    16,
) % (2**31)
INFERENCE_SUBSTRATE = (
    "verifier_ensemble_against_cached_candidates (principle: best-of-N over "
    "cached candidates; no LLM generation)."
)

CLOSED_VERDICT = "complete: selection_gap_closed_per_question_calibration_recovers_value"
FUNDAMENTAL_VERDICT = (
    "complete: selection_gap_fundamental_no_fix_beats_sc_discrimination_decoupled_as_2512_23067"
)
BLOCKED_VERDICT = "complete: blocked_no_multi_candidate_corpus"
TERMINAL_VERDICTS = (CLOSED_VERDICT, FUNDAMENTAL_VERDICT, BLOCKED_VERDICT)

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "inference_substrate",
    "per_candidate_auroc",
    "within_question_rank_corr",
    "sc_selection_accuracy",
    "oracle_bestofn_accuracy",
    "flip_count",
    "selection_accuracy_per_question_normalized",
    "selection_accuracy_ranking_calibrated",
    "self_certainty_selection_accuracy",
    "best_fix_vs_sc_delta_ci",
    "positive_control_valid",
    "selection_gap_closed",
    "n_examples",
    "random_seed",
    "reproducibility_checksum",
    "duration_s",
)

FIELD_PRINCIPLES = {
    "honest_verdict": "Terminal prefix for reconciler classification.",
    "inference_substrate": ("Best-of-N over cached candidates; no LLM generation."),
    "per_candidate_auroc": (
        "Cross-question discrimination -- confirms the ensemble discriminates "
        "(the half that works)."
    ),
    "within_question_rank_corr": (
        "Energy-vs-correctness rank correlation WITHIN questions -- the "
        "decoupling measure (arXiv:2512.23067)."
    ),
    "sc_selection_accuracy": "Majority-vote SC baseline accuracy -- the bar.",
    "oracle_bestofn_accuracy": (
        "Upper bound; must exceed SC for the positive control (selectable headroom)."
    ),
    "flip_count": (
        "Selections changed by the best fix vs SC -- flip_count==0 means a "
        "degenerate test (FALSE_NEGATIVE_RISK)."
    ),
    "selection_accuracy_per_question_normalized": (
        "FIX A: selection after within-question energy normalization."
    ),
    "selection_accuracy_ranking_calibrated": (
        "FIX B: selection after a within-question ranking calibration (held-out)."
    ),
    "self_certainty_selection_accuracy": (
        "FIX C baseline: self-certainty BoN selection (arXiv:2502.18581) -- "
        "the stronger free baseline."
    ),
    "best_fix_vs_sc_delta_ci": (
        "Paired delta + bootstrap CI + McNemar of the BEST fix vs SC -- the "
        "gap-closure magnitude + significance."
    ),
    "positive_control_valid": (
        "True iff oracle > SC AND flips > 0 -- without it the null is "
        "uninformative (the P0.1 lesson)."
    ),
    "selection_gap_closed": (
        "BARE bool. True iff at least one fix makes ensemble (or "
        "ensemble+self-certainty) selection beat SC with the delta CI excluding "
        "0, positive-control-valid -- the diagnosis verdict. STORE AS BARE "
        "true/false."
    ),
    "n_examples": "Sample-size rigor (>=50 multi-candidate items).",
    "random_seed": "Determinism precondition.",
    "reproducibility_checksum": "Drift detection.",
    "duration_s": "Plausibility floor.",
}


@dataclass(frozen=True)
class GapEvaluation:
    """Measured diagnostics and fix outcomes for the selected candidate rows."""

    n_examples: int
    mean_candidates_per_example: float
    per_candidate_auroc: float | None
    within_question_rank_corr: float | None
    within_question_rank_corr_details: dict[str, Any]
    sc_selection_accuracy: float
    oracle_bestofn_accuracy: float
    oracle_minus_sc_headroom: float
    ensemble_selection_accuracy: float
    selection_accuracy_per_question_normalized: float
    normalized_selection_details: dict[str, Any]
    selection_accuracy_ranking_calibrated: float
    ranking_calibration: dict[str, Any]
    self_certainty_selection_accuracy: float
    ensemble_self_certainty_fusion_accuracy: float
    best_fix_vs_sc_delta_ci: dict[str, Any]
    flip_count: int
    positive_control_valid: bool
    selection_gap_closed: bool
    best_fix_method: str
    random_seed: int
    reproducibility_checksum: str


@dataclass(frozen=True)
class OutcomeClassification:
    """Terminal verdict and bare bool selected from measured gates."""

    category: str
    terminal_verdict: str
    selection_gap_closed: bool


def classify_outcome(
    *,
    blocked: bool,
    positive_control_valid: bool,
    selection_gap_closed: bool,
) -> OutcomeClassification:
    """Map blocked/positive-control/gap-closed gates to terminal outcomes."""

    if blocked:
        return OutcomeClassification("blocked", BLOCKED_VERDICT, False)
    if positive_control_valid and selection_gap_closed:
        return OutcomeClassification(
            "fix_recovers_selection_value",
            CLOSED_VERDICT,
            True,
        )
    return OutcomeClassification(
        "decoupling_fundamental_no_fix_helps",
        FUNDAMENTAL_VERDICT,
        False,
    )


def _accuracy(correct: Sequence[bool]) -> float:
    return sum(1 for item in correct if item) / len(correct) if correct else 0.0


def tie_aware_auroc(labels: Sequence[int], scores: Sequence[float]) -> float | None:
    """Pure-Python AUROC with 0.5 credit for tied positive/negative scores."""

    positives = [score for label, score in zip(labels, scores, strict=True) if label == 1]
    negatives = [score for label, score in zip(labels, scores, strict=True) if label == 0]
    if not positives or not negatives:
        return None
    wins = 0.0
    for positive in positives:
        for negative in negatives:
            if positive > negative:
                wins += 1.0
            elif positive == negative:
                wins += 0.5
    return wins / (len(positives) * len(negatives))


def mean_within_question_rank_corr(
    energy_rows: Sequence[Sequence[float]],
    label_rows: Sequence[Sequence[int]],
) -> dict[str, Any]:
    """Compute a Kendall-tau-style within-question correctness rank signal."""

    taus: list[float] = []
    total_signed = 0.0
    total_pairs = 0
    for energies, labels in zip(energy_rows, label_rows, strict=True):
        signed = 0.0
        pairs = 0
        for i, (energy_i, label_i) in enumerate(zip(energies, labels, strict=True)):
            for energy_j, label_j in zip(energies[i + 1 :], labels[i + 1 :], strict=True):
                if label_i == label_j:
                    continue
                pairs += 1
                positive_energy = energy_i if label_i == 1 else energy_j
                negative_energy = energy_j if label_i == 1 else energy_i
                positive_score = -positive_energy
                negative_score = -negative_energy
                if positive_score > negative_score:
                    signed += 1.0
                elif positive_score < negative_score:
                    signed -= 1.0
        if pairs:
            taus.append(signed / pairs)
            total_signed += signed
            total_pairs += pairs
    return {
        "mean_tau": (sum(taus) / len(taus)) if taus else None,
        "weighted_tau": (total_signed / total_pairs) if total_pairs else None,
        "n_questions": len(taus),
        "n_comparable_pairs": total_pairs,
    }


def normalized_energy_scores(energies: Sequence[float], *, method: str) -> list[float]:
    """Return per-question normalized energy goodness scores; higher is better."""

    values = [float(energy) for energy in energies]
    if not values:
        return []
    if method == "minmax":
        lo = min(values)
        hi = max(values)
        if hi == lo:
            return [0.0 for _ in values]
        return [(hi - value) / (hi - lo) for value in values]
    if method == "zscore":
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / len(values)
        std = math.sqrt(variance)
        if std == 0.0:
            return [0.0 for _ in values]
        return [(mean - value) / std for value in values]
    raise ValueError(f"unknown normalization method: {method}")


def _minmax_high_good(values: Sequence[float]) -> list[float]:
    vals = [float(value) for value in values]
    if not vals:
        return []
    lo = min(vals)
    hi = max(vals)
    if hi == lo:
        return [0.0 for _ in vals]
    return [(value - lo) / (hi - lo) for value in vals]


def _select_by_scores(
    candidates: Sequence[Candidate],
    scores: Sequence[float | None],
) -> str | None:
    best_answer: str | None = None
    best_score = -math.inf
    for candidate, score in zip(candidates, scores, strict=True):
        if candidate.answer is None or score is None:
            continue
        value = float(score)
        if value > best_score:
            best_score = value
            best_answer = candidate.answer
    return best_answer


def _answer_correct(record: ProblemRecord, answer: str | None) -> bool:
    return answer is not None and answer == record.gold


def _confidence_scores(candidates: Sequence[Candidate]) -> list[float | None]:
    return [candidate.confidence for candidate in candidates]


def _fusion_scores(candidates: Sequence[Candidate], energies: Sequence[float]) -> list[float]:
    energy_good = normalized_energy_scores(energies, method="minmax")
    raw_confidences = [
        candidate.confidence if candidate.confidence is not None else -math.inf
        for candidate in candidates
    ]
    finite = [value for value in raw_confidences if math.isfinite(value)]
    floor = min(finite) if finite else 0.0
    confidences = [value if math.isfinite(value) else floor for value in raw_confidences]
    confidence_good = _minmax_high_good(confidences)
    return [energy + confidence for energy, confidence in zip(energy_good, confidence_good)]


def _lift_summary(
    method_correct: Sequence[bool],
    baseline_correct: Sequence[bool],
    *,
    method: str,
    seed: int,
    n_boot: int,
    flip_count: int,
) -> dict[str, Any]:
    method_acc = _accuracy(method_correct)
    baseline_acc = _accuracy(baseline_correct)
    return {
        "method": method,
        "comparison": f"{method}_vs_self_consistency",
        "accuracy": method_acc,
        "sc_accuracy_paired": baseline_acc,
        "delta": method_acc - baseline_acc,
        "ci95": list(
            paired_bootstrap_ci(
                list(method_correct),
                list(baseline_correct),
                seed=seed,
                n_boot=n_boot,
            )
        ),
        "mcnemar_exact_p": mcnemar_exact(list(baseline_correct), list(method_correct)),
        "flip_count": flip_count,
        "n": len(method_correct),
    }


def _split_indices(
    n_items: int, *, seed: int, train_fraction: float
) -> tuple[list[int], list[int]]:
    if n_items <= 1:
        return list(range(n_items)), list(range(n_items))
    keyed = [
        (
            hashlib.sha256(f"{seed}:{idx}".encode()).hexdigest(),
            idx,
        )
        for idx in range(n_items)
    ]
    ordered = [idx for _key, idx in sorted(keyed)]
    train_n = int(round(n_items * train_fraction))
    train_n = max(1, min(n_items - 1, train_n))
    return ordered[:train_n], ordered[train_n:]


def _fit_pairwise_orientation(
    records: Sequence[ProblemRecord],
    energy_rows: Sequence[Sequence[float]],
    train_indices: Sequence[int],
) -> dict[str, Any]:
    low_wins = 0.0
    high_wins = 0.0
    ties = 0
    pairs = 0
    for idx in train_indices:
        labels = [1 if candidate.correct else 0 for candidate in records[idx].candidates]
        energies = energy_rows[idx]
        for i, (energy_i, label_i) in enumerate(zip(energies, labels, strict=True)):
            for energy_j, label_j in zip(energies[i + 1 :], labels[i + 1 :], strict=True):
                if label_i == label_j:
                    continue
                pairs += 1
                correct_energy = energy_i if label_i == 1 else energy_j
                wrong_energy = energy_j if label_i == 1 else energy_i
                if correct_energy < wrong_energy:
                    low_wins += 1.0
                elif correct_energy > wrong_energy:
                    high_wins += 1.0
                else:
                    ties += 1
                    low_wins += 0.5
                    high_wins += 0.5
    orientation = "higher_energy_better" if high_wins > low_wins else "lower_energy_better"
    wins = high_wins if orientation == "higher_energy_better" else low_wins
    return {
        "orientation": orientation,
        "train_n": len(train_indices),
        "pairwise_train_pairs": pairs,
        "pairwise_train_ties": ties,
        "pairwise_train_accuracy": (wins / pairs) if pairs else None,
    }


def _checksum(
    records: Sequence[ProblemRecord],
    energy_rows: Sequence[Sequence[float]],
    *,
    seed: int,
) -> str:
    digest = hashlib.sha256()
    digest.update(f"exp=3682;seed={seed};substrate={INFERENCE_SUBSTRATE}".encode())
    for record, energies in zip(records, energy_rows, strict=True):
        digest.update(record.source_path.encode())
        digest.update(record.problem_id.encode())
        digest.update(record.gold.encode())
        for candidate, energy in zip(record.candidates, energies, strict=True):
            digest.update(str(candidate.answer).encode())
            digest.update(str(candidate.correct).encode())
            digest.update(str(candidate.confidence).encode())
            digest.update(f"{energy:.12g}".encode())
            digest.update(hashlib.sha256(candidate.text.encode()).hexdigest().encode())
    return digest.hexdigest()[:16]


def evaluate_gap(
    records: Sequence[ProblemRecord],
    *,
    energy_scorer: Callable[[Candidate], float],
    seed: int = RANDOM_SEED,
    n_boot: int = DEFAULT_N_BOOT,
    train_fraction: float = 0.67,
) -> GapEvaluation:
    """Evaluate discrimination, within-question rank signal, and fixes."""

    stats = compute_regime_stats(records)
    energy_rows = [
        [float(energy_scorer(candidate)) for candidate in record.candidates] for record in records
    ]
    label_rows = [
        [1 if candidate.correct else 0 for candidate in record.candidates] for record in records
    ]
    flat_labels = [label for labels in label_rows for label in labels]
    flat_scores = [-energy for energies in energy_rows for energy in energies]
    per_candidate_auroc = tie_aware_auroc(flat_labels, flat_scores)
    rank_details = mean_within_question_rank_corr(energy_rows, label_rows)

    sc_answers: list[str | None] = []
    sc_correct: list[bool] = []
    ensemble_correct: list[bool] = []
    normalized_z_correct: list[bool] = []
    normalized_minmax_correct: list[bool] = []
    self_certainty_correct: list[bool] = []
    fusion_correct: list[bool] = []
    normalized_answers: list[str | None] = []
    fusion_answers: list[str | None] = []

    for record, energies in zip(records, energy_rows, strict=True):
        sc_answer, _support = majority_vote_with_support(record)
        ensemble_answer = _select_by_scores(record.candidates, [-energy for energy in energies])
        normalized_z_answer = _select_by_scores(
            record.candidates,
            normalized_energy_scores(energies, method="zscore"),
        )
        normalized_minmax_answer = _select_by_scores(
            record.candidates,
            normalized_energy_scores(energies, method="minmax"),
        )
        self_certainty_answer = _select_by_scores(
            record.candidates,
            _confidence_scores(record.candidates),
        )
        fusion_answer = _select_by_scores(
            record.candidates, _fusion_scores(record.candidates, energies)
        )

        sc_answers.append(sc_answer)
        normalized_answers.append(normalized_z_answer)
        fusion_answers.append(fusion_answer)
        sc_correct.append(_answer_correct(record, sc_answer))
        ensemble_correct.append(_answer_correct(record, ensemble_answer))
        normalized_z_correct.append(_answer_correct(record, normalized_z_answer))
        normalized_minmax_correct.append(_answer_correct(record, normalized_minmax_answer))
        self_certainty_correct.append(_answer_correct(record, self_certainty_answer))
        fusion_correct.append(_answer_correct(record, fusion_answer))

    train_indices, heldout_indices = _split_indices(
        len(records),
        seed=seed,
        train_fraction=train_fraction,
    )
    ranking_calibration = _fit_pairwise_orientation(records, energy_rows, train_indices)
    high_good = ranking_calibration["orientation"] == "higher_energy_better"
    ranking_correct: list[bool] = []
    ranking_sc_correct: list[bool] = []
    ranking_answers: list[str | None] = []
    ranking_sc_answers: list[str | None] = []
    for idx in heldout_indices:
        record = records[idx]
        energies = energy_rows[idx]
        scores = [energy if high_good else -energy for energy in energies]
        answer = _select_by_scores(record.candidates, scores)
        sc_answer = sc_answers[idx]
        ranking_answers.append(answer)
        ranking_sc_answers.append(sc_answer)
        ranking_correct.append(_answer_correct(record, answer))
        ranking_sc_correct.append(sc_correct[idx])
    ranking_calibration = {
        **ranking_calibration,
        "heldout_n": len(heldout_indices),
        "train_fraction": train_fraction,
    }

    normalized_flips = sum(
        1
        for sc_answer, answer in zip(sc_answers, normalized_answers, strict=True)
        if sc_answer != answer
    )
    fusion_flips = sum(
        1
        for sc_answer, answer in zip(sc_answers, fusion_answers, strict=True)
        if sc_answer != answer
    )
    ranking_flips = sum(
        1
        for sc_answer, answer in zip(ranking_sc_answers, ranking_answers, strict=True)
        if sc_answer != answer
    )
    fix_summaries = [
        _lift_summary(
            normalized_z_correct,
            sc_correct,
            method="per_question_normalized",
            seed=seed + 11,
            n_boot=n_boot,
            flip_count=normalized_flips,
        ),
        _lift_summary(
            ranking_correct,
            ranking_sc_correct,
            method="ranking_calibrated",
            seed=seed + 12,
            n_boot=n_boot,
            flip_count=ranking_flips,
        ),
        _lift_summary(
            fusion_correct,
            sc_correct,
            method="ensemble_self_certainty_fusion",
            seed=seed + 13,
            n_boot=n_boot,
            flip_count=fusion_flips,
        ),
    ]
    best_fix = max(
        fix_summaries, key=lambda item: (item["accuracy"], item["delta"], item["flip_count"])
    )
    positive_control_valid = (
        stats.oracle_bestofn_accuracy > _accuracy(sc_correct) and best_fix["flip_count"] > 0
    )
    selection_gap_closed = bool(
        positive_control_valid
        and any(summary["delta"] > 0.0 and summary["ci95"][0] > 0.0 for summary in fix_summaries)
    )

    return GapEvaluation(
        n_examples=stats.n_examples,
        mean_candidates_per_example=stats.mean_candidates_per_example,
        per_candidate_auroc=per_candidate_auroc,
        within_question_rank_corr=rank_details["weighted_tau"],
        within_question_rank_corr_details=rank_details,
        sc_selection_accuracy=_accuracy(sc_correct),
        oracle_bestofn_accuracy=stats.oracle_bestofn_accuracy,
        oracle_minus_sc_headroom=stats.oracle_minus_sc_headroom,
        ensemble_selection_accuracy=_accuracy(ensemble_correct),
        selection_accuracy_per_question_normalized=_accuracy(normalized_z_correct),
        normalized_selection_details={
            "zscore_accuracy": _accuracy(normalized_z_correct),
            "minmax_accuracy": _accuracy(normalized_minmax_correct),
        },
        selection_accuracy_ranking_calibrated=_accuracy(ranking_correct),
        ranking_calibration=ranking_calibration,
        self_certainty_selection_accuracy=_accuracy(self_certainty_correct),
        ensemble_self_certainty_fusion_accuracy=_accuracy(fusion_correct),
        best_fix_vs_sc_delta_ci=best_fix,
        flip_count=int(best_fix["flip_count"]),
        positive_control_valid=positive_control_valid,
        selection_gap_closed=selection_gap_closed,
        best_fix_method=str(best_fix["method"]),
        random_seed=seed,
        reproducibility_checksum=_checksum(records, energy_rows, seed=seed),
    )


def _required_fields_present(artifact: dict[str, Any]) -> bool:
    return all(field in artifact for field in REQUIRED_ARTIFACT_FIELDS)


def _base_artifact(
    *,
    verdict: str,
    corpus_paths: Sequence[str],
    duration_s: float,
) -> dict[str, Any]:
    artifact: dict[str, Any] = {
        "honest_verdict": verdict,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "corpus_paths": list(corpus_paths),
        "per_candidate_auroc": None,
        "within_question_rank_corr": None,
        "within_question_rank_corr_details": None,
        "sc_selection_accuracy": None,
        "oracle_bestofn_accuracy": None,
        "oracle_minus_sc_headroom": None,
        "flip_count": 0,
        "ensemble_selection_accuracy": None,
        "selection_accuracy_per_question_normalized": None,
        "normalized_selection_details": None,
        "selection_accuracy_ranking_calibrated": None,
        "ranking_calibration": None,
        "self_certainty_selection_accuracy": None,
        "ensemble_self_certainty_fusion_accuracy": None,
        "best_fix_method": None,
        "best_fix_vs_sc_delta_ci": None,
        "positive_control_valid": False,
        "selection_gap_closed": False,
        "n_examples": 0,
        "mean_candidates_per_example": 0.0,
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": None,
        "duration_s": duration_s,
        "field_principles": dict(FIELD_PRINCIPLES),
    }
    artifact["acceptance_gate"] = {
        "condition": (
            "per_candidate_auroc present AND within_question_rank_corr present "
            "AND positive_control_valid == true AND flip_count > 0"
        ),
        "principle": (
            "A selection-gap verdict requires the discrimination measure, the "
            "within-question decoupling measure, a valid positive control and a "
            "fix that actually changed selections -- otherwise it repeats the "
            "degenerate-test trap."
        ),
        "required_fields_present": _required_fields_present(artifact),
        "per_candidate_auroc_present": artifact["per_candidate_auroc"] is not None,
        "within_question_rank_corr_present": artifact["within_question_rank_corr"] is not None,
        "positive_control_valid": artifact["positive_control_valid"],
        "flip_count_gt_0": artifact["flip_count"] > 0,
    }
    artifact["schema"] = sorted(artifact.keys())
    return artifact


def build_blocked_artifact(
    *,
    corpus_paths: Sequence[str],
    duration_s: float,
) -> dict[str, Any]:
    """Build the blocked artifact for a missing multi-candidate corpus."""

    return _base_artifact(
        verdict=BLOCKED_VERDICT,
        corpus_paths=corpus_paths,
        duration_s=duration_s,
    )


def build_measured_artifact(
    *,
    result: GapEvaluation,
    corpus_paths: Sequence[str],
    duration_s: float,
) -> dict[str, Any]:
    """Build the terminal measured artifact from an evaluation result."""

    classification = classify_outcome(
        blocked=False,
        positive_control_valid=result.positive_control_valid,
        selection_gap_closed=result.selection_gap_closed,
    )
    artifact = _base_artifact(
        verdict=classification.terminal_verdict,
        corpus_paths=corpus_paths,
        duration_s=duration_s,
    )
    artifact.update(
        {
            "honest_outcome": classification.category,
            "per_candidate_auroc": result.per_candidate_auroc,
            "within_question_rank_corr": result.within_question_rank_corr,
            "within_question_rank_corr_details": result.within_question_rank_corr_details,
            "sc_selection_accuracy": result.sc_selection_accuracy,
            "oracle_bestofn_accuracy": result.oracle_bestofn_accuracy,
            "oracle_minus_sc_headroom": result.oracle_minus_sc_headroom,
            "flip_count": result.flip_count,
            "ensemble_selection_accuracy": result.ensemble_selection_accuracy,
            "selection_accuracy_per_question_normalized": {
                "selected_accuracy": result.selection_accuracy_per_question_normalized,
                **result.normalized_selection_details,
            },
            "normalized_selection_details": result.normalized_selection_details,
            "selection_accuracy_ranking_calibrated": (result.selection_accuracy_ranking_calibrated),
            "ranking_calibration": result.ranking_calibration,
            "self_certainty_selection_accuracy": result.self_certainty_selection_accuracy,
            "ensemble_self_certainty_fusion_accuracy": (
                result.ensemble_self_certainty_fusion_accuracy
            ),
            "best_fix_method": result.best_fix_method,
            "best_fix_vs_sc_delta_ci": result.best_fix_vs_sc_delta_ci,
            "positive_control_valid": result.positive_control_valid,
            "selection_gap_closed": classification.selection_gap_closed,
            "n_examples": result.n_examples,
            "mean_candidates_per_example": result.mean_candidates_per_example,
            "random_seed": result.random_seed,
            "reproducibility_checksum": result.reproducibility_checksum,
        }
    )
    artifact["acceptance_gate"]["per_candidate_auroc_present"] = (
        artifact["per_candidate_auroc"] is not None
    )
    artifact["acceptance_gate"]["within_question_rank_corr_present"] = (
        artifact["within_question_rank_corr"] is not None
    )
    artifact["acceptance_gate"]["positive_control_valid"] = artifact["positive_control_valid"]
    artifact["acceptance_gate"]["flip_count_gt_0"] = artifact["flip_count"] > 0
    artifact["schema"] = sorted(artifact.keys())
    return artifact


def validate_artifact(artifact: dict[str, Any]) -> None:
    """Validate the Exp 3682 terminal artifact shape."""

    missing = set(REQUIRED_ARTIFACT_FIELDS) - set(artifact)
    if missing:
        raise AssertionError(f"missing required fields: {sorted(missing)}")
    missing_principles = set(REQUIRED_ARTIFACT_FIELDS) - set(artifact.get("field_principles", {}))
    if missing_principles:
        raise AssertionError(f"missing field principles: {sorted(missing_principles)}")
    if artifact["honest_verdict"] not in TERMINAL_VERDICTS:
        raise AssertionError(f"unknown terminal verdict: {artifact['honest_verdict']}")
    if type(artifact["selection_gap_closed"]) is not bool:
        raise AssertionError("selection_gap_closed must be a bare bool")
    if type(artifact["positive_control_valid"]) is not bool:
        raise AssertionError("positive_control_valid must be a bare bool")
    gate = artifact.get("acceptance_gate", {})
    if gate.get("required_fields_present") is not True:
        raise AssertionError("acceptance gate must record required_fields_present=true")


def _path_label(path: Path, repo_root: Path) -> str:
    try:
        return path.relative_to(repo_root).as_posix()
    except ValueError:
        return path.as_posix()


def run_experiment(
    *,
    repo_root: Path,
    output_path: Path | None = None,
    corpus_paths: Sequence[Path] = DEFAULT_CORPUS_PATHS,
    min_candidates: int = DEFAULT_MIN_CANDIDATES,
    min_examples: int = DEFAULT_MIN_EXAMPLES,
    max_sc_accuracy: float = DEFAULT_MAX_SC_ACCURACY,
    max_majority_supports: Sequence[float] = DEFAULT_MAX_MAJORITY_SUPPORTS,
    energy_scorer: Callable[[Candidate], float] | None = None,
    seed: int = RANDOM_SEED,
    n_boot: int = DEFAULT_N_BOOT,
) -> dict[str, Any]:
    """Run Exp 3682 and optionally write the terminal JSON artifact."""

    start = time.time()
    resolved_paths = [path if path.is_absolute() else repo_root / path for path in corpus_paths]
    path_labels = [_path_label(path, repo_root) for path in resolved_paths]
    records = load_multicandidate_records(resolved_paths, min_candidates=min_candidates)
    if not records:
        artifact = build_blocked_artifact(
            corpus_paths=path_labels,
            duration_s=time.time() - start,
        )
    else:
        selected = select_sc_weak_regime(
            records,
            min_examples=min_examples,
            max_sc_accuracy=max_sc_accuracy,
            max_majority_supports=max_majority_supports,
        )
        result = evaluate_gap(
            selected.records,
            energy_scorer=energy_scorer or make_default_energy_scorer(),
            seed=seed,
            n_boot=n_boot,
        )
        artifact = build_measured_artifact(
            result=result,
            corpus_paths=path_labels,
            duration_s=time.time() - start,
        )
        artifact["selected_stratum"] = selected.status
        artifact["stratum_rule"] = selected.rule
        artifact["max_majority_support"] = selected.max_majority_support
        artifact["schema"] = sorted(artifact.keys())
    validate_artifact(artifact)
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(artifact, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return artifact
