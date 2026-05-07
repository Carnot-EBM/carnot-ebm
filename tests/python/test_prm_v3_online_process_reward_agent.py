"""Tests for PRM v3 online process-reward repair selection.

Spec: REQ-VERIFY-1448, SCENARIO-VERIFY-1448.
"""

from __future__ import annotations

import pytest

from carnot.pipeline.prm_v3_online_process_reward_agent import (
    bounded_pra_step_score,
    candidate_process_steps,
    evaluate_online_process_reward_selection,
    rank_case_candidates_online,
    score_candidate_online,
    tie_aware_auroc,
)


def _candidate_result(
    index: int,
    *,
    accepted: bool,
    final_state: str,
    target: str,
    false_acceptance: bool = False,
) -> dict[str, object]:
    return {
        "candidate_index": index,
        "accepted": accepted,
        "validation_result": {
            "semantic_result": final_state,
            "constraint_passed": accepted,
            "false_acceptance": false_acceptance,
        },
        "candidate": {
            "draft_certificate": "<CARNOT_CERT_STATE:REPAIR_HINT>\nREPAIR_HINT: repair step.",
            "draft_state": "REPAIR_HINT",
            "final_certificate": f"<CARNOT_CERT_STATE:{final_state}>\n{final_state}",
            "final_state": final_state,
            "repair_action_type": "STEP_REWRITE",
            "repair_rationale": "Repair the localized reasoning step.",
            "repair_target": target,
            "validator_metadata": {
                "prototype_accept": accepted,
                "leaky_energy": 0.1 if accepted else 9.0,
            },
        },
    }


def _score(text: str) -> float:
    lower = text.lower()
    score = 0.35
    if "sat" in lower:
        score += 0.35
    if "repair_hint" in lower:
        score -= 0.2
    if "low_energy" in lower:
        score += 0.25
    if "high_energy" in lower:
        score -= 0.15
    return max(0.0, min(1.0, score))


def test_req1448_candidate_process_steps_are_step_level_and_non_leaky() -> None:
    """REQ-VERIFY-1448: candidate text is split into bounded process steps."""

    candidate = _candidate_result(
        0,
        accepted=True,
        final_state="SAT",
        target="accept_low_energy localized FoVer reasoning step",
    )

    steps = candidate_process_steps(candidate)

    assert [step["step_type"] for step in steps] == [
        "draft_certificate",
        "repair_action",
        "repair_rationale",
        "final_certificate",
    ]
    assert len(steps) == 4
    joined = "\n".join(step["text"] for step in steps)
    assert "validation_result" not in joined
    assert "accepted" not in joined
    assert "prototype_accept" not in joined
    assert "leaky_energy" not in joined


def test_scenario1448_online_scores_select_better_repair_trace() -> None:
    """SCENARIO-VERIFY-1448: step aggregation selects the best repair trace."""

    case = {
        "case_id": "case_1448",
        "best_of_n_success": True,
        "best_candidate_index": 2,
        "candidate_results": [
            _candidate_result(
                0,
                accepted=False,
                final_state="REPAIR_HINT",
                target="reject_first_candidate localized FoVer reasoning step",
            ),
            _candidate_result(
                1,
                accepted=True,
                final_state="SAT",
                target="accept_high_energy localized FoVer reasoning step",
                false_acceptance=True,
            ),
            _candidate_result(
                2,
                accepted=True,
                final_state="SAT",
                target="accept_low_energy localized FoVer reasoning step",
            ),
        ],
    }

    ranked = rank_case_candidates_online(case, _score)

    assert ranked["selected_candidate_index"] == 2
    assert ranked["selected_accepted"] is True
    assert ranked["selected_false_acceptance"] is False
    assert ranked["step_scores_generated"] == 12
    assert ranked["candidate_scores"][2]["step_scores"][0]["step_index"] == 0


def test_req1448_evaluation_compares_raw_prmv1_and_prmv3() -> None:
    """REQ-VERIFY-1448: evaluation reports selection and false-acceptance deltas."""

    cases = [
        {
            "case_id": "case_1",
            "best_of_n_success": True,
            "best_candidate_index": 2,
            "candidate_results": [
                _candidate_result(
                    0,
                    accepted=False,
                    final_state="REPAIR_HINT",
                    target="reject_first_candidate localized FoVer reasoning step",
                ),
                _candidate_result(
                    1,
                    accepted=True,
                    final_state="SAT",
                    target="accept_high_energy localized FoVer reasoning step",
                    false_acceptance=True,
                ),
                _candidate_result(
                    2,
                    accepted=True,
                    final_state="SAT",
                    target="accept_low_energy localized FoVer reasoning step",
                ),
            ],
        },
        {
            "case_id": "case_2",
            "best_of_n_success": False,
            "candidate_results": [
                _candidate_result(
                    0,
                    accepted=False,
                    final_state="REPAIR_HINT",
                    target="reject_first_candidate localized FoVer reasoning step",
                )
            ],
        },
    ]
    prmv1 = [
        {
            "case_id": "case_1",
            "selected_candidate_index": 1,
            "selected_accepted": True,
        },
        {
            "case_id": "case_2",
            "selected_candidate_index": 0,
            "selected_accepted": False,
        },
    ]

    aggregate = evaluate_online_process_reward_selection(cases, _score, prmv1)

    assert aggregate["cases_evaluated"] == 2
    assert aggregate["traces_evaluated"] == 4
    assert aggregate["step_scores_generated"] == 16
    assert aggregate["raw_best_of_n_repair_success_rate"] == pytest.approx(0.5)
    assert aggregate["prm_v1_selected_repair_success_rate"] == pytest.approx(0.5)
    assert aggregate["prm_v3_selected_repair_success_rate"] == pytest.approx(0.5)
    assert aggregate["selection_improvement_pp"] == pytest.approx(0.0)
    assert aggregate["false_acceptance_rate_delta"] == pytest.approx(-0.5)
    assert aggregate["regression_against_prm_v1"] is False


def test_req1448_scores_handle_empty_cases_and_auc_edges() -> None:
    """REQ-VERIFY-1448: empty inputs and one-class labels remain deterministic."""

    aggregate = evaluate_online_process_reward_selection([], _score, [])

    assert aggregate["cases_evaluated"] == 0
    assert aggregate["traces_evaluated"] == 0
    assert aggregate["step_scores_generated"] == 0
    assert aggregate["selector_auroc"] == pytest.approx(0.5)
    assert aggregate["selection_improvement_pp"] == pytest.approx(0.0)
    assert aggregate["regression_against_prm_v1"] is False
    assert candidate_process_steps(None) == []
    assert candidate_process_steps({"candidate": "not-a-map"}) == []
    assert candidate_process_steps({"candidate": {}}) == []
    assert (
        len(
            candidate_process_steps(
                _candidate_result(
                    0,
                    accepted=True,
                    final_state="SAT",
                    target="accept_low_energy localized FoVer reasoning step",
                ),
                max_steps=2,
            )
        )
        == 2
    )
    empty_ranked = rank_case_candidates_online(
        {"case_id": "empty", "best_of_n_success": True, "candidate_results": []},
        _score,
    )
    assert empty_ranked["selected_candidate_index"] is None
    assert empty_ranked["raw_best_of_n_success"] is True
    empty_candidate = score_candidate_online(
        {"candidate_index": 7, "accepted": False, "candidate": {}},
        _score,
    )
    assert empty_candidate["score"] == pytest.approx(0.0)
    assert empty_candidate["false_acceptance"] is False
    assert bounded_pra_step_score("SAT low_energy", 0.5) > bounded_pra_step_score(
        "REPAIR_HINT high_energy",
        0.5,
    )
    assert bounded_pra_step_score("plain", "bad") == pytest.approx(0.0)
    assert tie_aware_auroc([1, 0], [0.5, 0.5]) == pytest.approx(0.5)
    assert tie_aware_auroc([1, 0], [0.2, 0.8]) == pytest.approx(0.0)
    assert tie_aware_auroc([1, 1], [0.2, 0.8]) == pytest.approx(0.5)


def test_req1448_false_acceptance_and_missing_prmv1_edges() -> None:
    """REQ-VERIFY-1448: missing PRM v1 rows and raw fallback selection are safe."""

    case = {
        "case_id": "case_no_prmv1",
        "best_of_n_success": True,
        "candidate_results": [
            {
                "candidate_index": 0,
                "accepted": True,
                "validation_result": {"false_acceptance": True},
                "candidate": {
                    "repair_action_type": "STEP_REWRITE",
                    "repair_target": "accept_low_energy localized FoVer reasoning step",
                },
            }
        ],
    }

    aggregate = evaluate_online_process_reward_selection([case], lambda _text: "not-finite", [])

    assert aggregate["traces_evaluated"] == 1
    assert aggregate["step_scores_generated"] == 1
    assert aggregate["raw_best_of_n_false_acceptance_rate"] == pytest.approx(1.0)
    assert aggregate["prm_v1_selected_repair_success_rate"] == pytest.approx(0.0)
    assert aggregate["regression_against_prm_v1"] is True
