"""Online process-reward repair selection for PRM v3.

Exp 1430 ranked each repair candidate as one final blob.  Exp 1448 keeps the
same bounded candidate pool but scores the candidate's visible repair process
one step at a time, then freezes that aggregate before semantic acceptance
labels are used for evaluation.

Spec: REQ-VERIFY-1448, SCENARIO-VERIFY-1448.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from typing import Any


StepTextScorer = Callable[[str], float]

_STEP_FIELDS = (
    ("draft_certificate", "draft_certificate"),
    ("repair_action", "repair_action_type"),
    ("repair_target", "repair_target"),
    ("repair_rationale", "repair_rationale"),
    ("final_certificate", "final_certificate"),
)


def candidate_process_steps(
    candidate_result: Mapping[str, Any] | None,
    *,
    max_steps: int = 6,
) -> list[dict[str, Any]]:
    """Return bounded PRM-visible repair steps without validation-label fields."""

    if not isinstance(candidate_result, Mapping):
        return []
    candidate = candidate_result.get("candidate")
    if not isinstance(candidate, Mapping):
        return []

    steps: list[dict[str, Any]] = []
    for step_type, field in _STEP_FIELDS:
        text = str(candidate.get(field) or "").strip()
        if not text:
            continue
        if step_type == "repair_action":
            target = str(candidate.get("repair_target") or "").strip()
            if target:
                text = f"{text}: {target}"
            steps.append(_step_row(len(steps), "repair_action", text))
            continue
        if step_type == "repair_target":
            continue
        steps.append(_step_row(len(steps), step_type, text))
        if len(steps) >= int(max_steps):
            break
    return steps[: max(0, int(max_steps))]


def bounded_pra_step_score(text: str, prm_v2_score: float) -> float:
    """Blend PRM v2 probability with simple local process-state signals."""

    base = _clamp01(_finite_score(prm_v2_score, default=0.0))
    lower = str(text or "").lower()
    adjustment = 0.0
    if any(marker in lower for marker in ("sat", "valid", "correct", "therefore")):
        adjustment += 0.14
    if any(marker in lower for marker in ("repair_hint", "invalid", "wrong", "incorrect")):
        adjustment -= 0.14
    if "low_energy" in lower:
        adjustment += 0.08
    if "high_energy" in lower:
        adjustment -= 0.08
    if "localized" in lower and "step" in lower:
        adjustment += 0.02
    return round(_clamp01(base + adjustment), 6)


def score_candidate_online(
    candidate_result: Mapping[str, Any],
    scorer: StepTextScorer,
) -> dict[str, Any]:
    """Score all visible process steps for one candidate and aggregate online."""

    step_scores: list[dict[str, Any]] = []
    for step in candidate_process_steps(candidate_result):
        raw_score = _finite_score(scorer(str(step["text"])), default=0.0)
        step_scores.append(
            {
                **step,
                "score": round(raw_score, 6),
            }
        )

    aggregate_score = _aggregate_step_scores([row["score"] for row in step_scores])
    candidate_index = int(candidate_result.get("candidate_index", 0))
    return {
        "candidate_index": candidate_index,
        "score": aggregate_score,
        "accepted": bool(candidate_result.get("accepted")),
        "false_acceptance": _candidate_false_acceptance(candidate_result),
        "step_scores": step_scores,
    }


def rank_case_candidates_online(
    case_result: Mapping[str, Any],
    scorer: StepTextScorer,
) -> dict[str, Any]:
    """Return the highest online process-reward candidate for one case."""

    candidate_scores = [
        score_candidate_online(candidate_result, scorer)
        for candidate_result in case_result.get("candidate_results") or []
        if isinstance(candidate_result, Mapping)
    ]
    if not candidate_scores:
        return {
            "case_id": str(case_result.get("case_id") or ""),
            "raw_best_of_n_success": bool(case_result.get("best_of_n_success")),
            "selected_candidate_index": None,
            "selected_score": None,
            "selected_accepted": False,
            "selected_false_acceptance": False,
            "step_scores_generated": 0,
            "candidate_scores": [],
        }

    selected = max(
        candidate_scores,
        key=lambda row: (float(row["score"]), -int(row["candidate_index"])),
    )
    return {
        "case_id": str(case_result.get("case_id") or ""),
        "raw_best_of_n_success": bool(case_result.get("best_of_n_success")),
        "raw_best_of_n_false_acceptance": _raw_best_of_n_false_acceptance(case_result),
        "selected_candidate_index": selected["candidate_index"],
        "selected_score": selected["score"],
        "selected_accepted": bool(selected["accepted"]),
        "selected_false_acceptance": bool(selected["false_acceptance"]),
        "step_scores_generated": sum(len(row["step_scores"]) for row in candidate_scores),
        "candidate_scores": candidate_scores,
    }


def evaluate_online_process_reward_selection(
    candidate_search_results: Sequence[Mapping[str, Any]],
    scorer: StepTextScorer,
    prm_v1_case_selections: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Compare raw best-of-N, PRM v1 top-1, and PRM v3 online selection."""

    case_selections = [
        rank_case_candidates_online(case_result, scorer)
        for case_result in candidate_search_results
        if isinstance(case_result, Mapping) and case_result.get("candidate_results")
    ]
    prmv1_by_case = {
        str(row.get("case_id") or ""): row
        for row in prm_v1_case_selections or []
        if isinstance(row, Mapping)
    }
    prmv1_rows = [
        _prmv1_selection_for_case(selection, prmv1_by_case.get(str(selection["case_id"])))
        for selection in case_selections
    ]
    cases_evaluated = len(case_selections)
    raw_rate = _rate(
        sum(1 for selection in case_selections if selection["raw_best_of_n_success"]),
        cases_evaluated,
    )
    prmv1_rate = _rate(sum(1 for row in prmv1_rows if row["selected_accepted"]), cases_evaluated)
    prmv3_rate = _rate(
        sum(1 for selection in case_selections if selection["selected_accepted"]),
        cases_evaluated,
    )
    raw_fa = _rate(
        sum(1 for selection in case_selections if selection["raw_best_of_n_false_acceptance"]),
        cases_evaluated,
    )
    prmv1_fa = _rate(
        sum(1 for row in prmv1_rows if row["selected_false_acceptance"]), cases_evaluated
    )
    prmv3_fa = _rate(
        sum(1 for selection in case_selections if selection["selected_false_acceptance"]),
        cases_evaluated,
    )
    labels = [
        1 if candidate["accepted"] else 0
        for selection in case_selections
        for candidate in selection["candidate_scores"]
    ]
    scores = [
        float(candidate["score"])
        for selection in case_selections
        for candidate in selection["candidate_scores"]
    ]
    regression = bool(prmv3_rate < prmv1_rate or prmv3_fa > prmv1_fa)
    return {
        "cases_evaluated": cases_evaluated,
        "traces_evaluated": sum(
            len(selection["candidate_scores"]) for selection in case_selections
        ),
        "step_scores_generated": sum(
            selection["step_scores_generated"] for selection in case_selections
        ),
        "selector_auroc": round(tie_aware_auroc(labels, scores), 6),
        "raw_best_of_n_repair_success_rate": raw_rate,
        "prm_v1_selected_repair_success_rate": prmv1_rate,
        "prm_v3_selected_repair_success_rate": prmv3_rate,
        "selection_improvement_pp": round((prmv3_rate - raw_rate) * 100.0, 6),
        "prm_v3_vs_prm_v1_selection_delta_pp": round((prmv3_rate - prmv1_rate) * 100.0, 6),
        "raw_best_of_n_false_acceptance_rate": raw_fa,
        "prm_v1_false_acceptance_rate": prmv1_fa,
        "prm_v3_false_acceptance_rate": prmv3_fa,
        "false_acceptance_rate_delta": round(prmv3_fa - prmv1_fa, 6),
        "regression_against_prm_v1": regression,
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


def _step_row(step_index: int, step_type: str, text: str) -> dict[str, Any]:
    return {"step_index": int(step_index), "step_type": step_type, "text": text}


def _aggregate_step_scores(scores: Sequence[float]) -> float:
    if not scores:
        return 0.0
    finite_scores = [_finite_score(score, default=0.0) for score in scores]
    online_belief = 0.5
    for score in finite_scores:
        online_belief = 0.65 * online_belief + 0.35 * _clamp01(score)
    trend = finite_scores[-1] - finite_scores[0]
    return round(_clamp01(online_belief + 0.05 * trend), 6)


def _candidate_false_acceptance(candidate_result: Mapping[str, Any]) -> bool:
    validation = candidate_result.get("validation_result")
    if isinstance(validation, Mapping) and "false_acceptance" in validation:
        return bool(validation.get("false_acceptance"))
    return False


def _raw_best_of_n_false_acceptance(case_result: Mapping[str, Any]) -> bool:
    if not bool(case_result.get("best_of_n_success")):
        return False
    candidate_results = [
        row for row in case_result.get("candidate_results") or [] if isinstance(row, Mapping)
    ]
    best_index = case_result.get("best_candidate_index")
    for row in candidate_results:
        if best_index is not None and int(row.get("candidate_index", -1)) == int(best_index):
            return _candidate_false_acceptance(row)
    return any(
        bool(row.get("accepted")) and _candidate_false_acceptance(row) for row in candidate_results
    )


def _prmv1_selection_for_case(
    prmv3_selection: Mapping[str, Any],
    prmv1_selection: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if not isinstance(prmv1_selection, Mapping):
        return {"selected_accepted": False, "selected_false_acceptance": False}
    selected_index = prmv1_selection.get("selected_candidate_index")
    selected_false_acceptance = False
    for candidate in prmv3_selection.get("candidate_scores") or []:
        if int(candidate.get("candidate_index", -1)) == int(selected_index):
            selected_false_acceptance = bool(candidate.get("false_acceptance"))
            break
    return {
        "selected_accepted": bool(prmv1_selection.get("selected_accepted")),
        "selected_false_acceptance": selected_false_acceptance,
    }


def _finite_score(score: float, *, default: float) -> float:
    try:
        value = float(score)
    except (TypeError, ValueError):
        return default
    return value if math.isfinite(value) else default


def _rate(numerator: int, denominator: int) -> float:
    return round(float(numerator) / float(denominator), 6) if denominator else 0.0


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))
