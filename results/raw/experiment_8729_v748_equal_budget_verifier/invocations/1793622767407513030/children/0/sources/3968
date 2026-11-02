"""PRM-guided selector for bounded repair candidates.

Exp 1429 can generate several schema-valid repair candidates for the same
repair-hint case. Exp 1430 tests the cheaper next step: freeze a process-reward
score for each candidate before semantic validation labels are consulted, pick
one candidate, and only then measure whether that selected candidate was
accepted by the existing validator.

Spec: REQ-VERIFY-1430, SCENARIO-VERIFY-1430
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from typing import Any


CandidateTextScorer = Callable[[str], float]


def candidate_process_text(candidate: Mapping[str, Any] | None) -> str:
    """Return the PRM-visible repair text without validation-result leakage."""

    if not isinstance(candidate, Mapping):
        return ""
    keys = (
        "draft_certificate",
        "draft_state",
        "final_certificate",
        "final_state",
        "repair_action_type",
        "repair_rationale",
    )
    parts = [str(candidate.get(key) or "").strip() for key in keys]
    return "\n".join(part for part in parts if part)


def rank_case_candidates(
    case_result: Mapping[str, Any],
    scorer: CandidateTextScorer,
) -> dict[str, Any]:
    """Score one case's candidates and return the highest PRM-ranked candidate."""

    candidate_scores: list[dict[str, Any]] = []
    for position, candidate_result in enumerate(case_result.get("candidate_results") or []):
        if not isinstance(candidate_result, Mapping):
            continue
        candidate_index = int(candidate_result.get("candidate_index", position))
        score = _finite_score(scorer(candidate_process_text(candidate_result.get("candidate"))))
        candidate_scores.append(
            {
                "candidate_index": candidate_index,
                "score": round(score, 6) if math.isfinite(score) else score,
                "accepted": bool(candidate_result.get("accepted")),
            }
        )

    if not candidate_scores:
        return {
            "case_id": str(case_result.get("case_id") or ""),
            "raw_best_of_n_success": bool(case_result.get("best_of_n_success")),
            "selected_candidate_index": None,
            "selected_score": None,
            "selected_accepted": False,
            "candidate_scores": [],
        }

    selected = max(
        candidate_scores,
        key=lambda row: (float(row["score"]), -int(row["candidate_index"])),
    )
    return {
        "case_id": str(case_result.get("case_id") or ""),
        "raw_best_of_n_success": bool(case_result.get("best_of_n_success")),
        "selected_candidate_index": selected["candidate_index"],
        "selected_score": selected["score"],
        "selected_accepted": bool(selected["accepted"]),
        "candidate_scores": candidate_scores,
    }


def evaluate_prm_guided_selection(
    candidate_search_results: Sequence[Mapping[str, Any]],
    scorer: CandidateTextScorer,
) -> dict[str, Any]:
    """Compare PRM top-1 selection with Exp 1429 raw best-of-N success."""

    case_selections = [
        rank_case_candidates(case_result, scorer)
        for case_result in candidate_search_results
        if isinstance(case_result, Mapping) and case_result.get("candidate_results")
    ]
    all_labels = [
        1 if candidate_score["accepted"] else 0
        for case_selection in case_selections
        for candidate_score in case_selection["candidate_scores"]
    ]
    all_scores = [
        float(candidate_score["score"])
        for case_selection in case_selections
        for candidate_score in case_selection["candidate_scores"]
    ]
    cases_evaluated = len(case_selections)
    raw_successes = sum(1 for item in case_selections if item["raw_best_of_n_success"])
    selected_successes = sum(1 for item in case_selections if item["selected_accepted"])
    raw_rate = _rate(raw_successes, cases_evaluated)
    selected_rate = _rate(selected_successes, cases_evaluated)
    return {
        "cases_evaluated": cases_evaluated,
        "selector_auroc": round(tie_aware_auroc(all_labels, all_scores), 6),
        "raw_best_of_n_repair_success_rate": raw_rate,
        "selected_repair_success_rate": selected_rate,
        "selection_improvement_pp": round((selected_rate - raw_rate) * 100.0, 6),
        "case_selections": case_selections,
    }


def tie_aware_auroc(labels: Sequence[int], scores: Sequence[float]) -> float:
    """Compute AUROC where label 1 and larger score mean accepted repair."""

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


def _finite_score(score: float) -> float:
    value = float(score)
    return value if math.isfinite(value) else float("-inf")


def _rate(numerator: int, denominator: int) -> float:
    return round(float(numerator) / float(denominator), 6) if denominator else 0.0
