"""Tests for PRM-guided repair candidate selection.

Spec: REQ-VERIFY-1430, SCENARIO-VERIFY-1430
"""

from __future__ import annotations

import math

import pytest

from carnot.pipeline.prm_guided_repair_selector import (
    candidate_process_text,
    evaluate_prm_guided_selection,
    rank_case_candidates,
    tie_aware_auroc,
)


def _candidate_result(index: int, *, accepted: bool, final_state: str) -> dict[str, object]:
    return {
        "candidate_index": index,
        "accepted": accepted,
        "validation_result": {
            "semantic_result": "SAT" if accepted else "REPAIR_HINT",
            "constraint_passed": accepted,
        },
        "candidate": {
            "draft_certificate": "<CARNOT_CERT_STATE:REPAIR_HINT>\nREPAIR_HINT: repair step.",
            "draft_state": "REPAIR_HINT",
            "final_certificate": f"<CARNOT_CERT_STATE:{final_state}>\n{final_state}",
            "final_state": final_state,
            "repair_action_type": "STEP_REWRITE",
            "repair_rationale": "Repair the localized reasoning step.",
            "repair_target": "localized FoVer reasoning step",
            "validator_metadata": {
                "prototype_accept": accepted,
                "leaky_energy": 0.1 if accepted else 9.0,
            },
        },
    }


def test_scenario1430_prm_scores_before_labels_and_selects_later_candidate() -> None:
    """SCENARIO-VERIFY-1430: PRM ranking can select a later accepted repair."""

    case = {
        "case_id": "case_1430",
        "best_of_n_success": True,
        "candidate_results": [
            _candidate_result(0, accepted=False, final_state="REPAIR_HINT"),
            _candidate_result(1, accepted=True, final_state="SAT"),
        ],
    }
    scored_texts: list[str] = []

    def scorer(text: str) -> float:
        scored_texts.append(text)
        assert "validation_result" not in text
        assert "accepted" not in text
        assert "prototype_accept" not in text
        return 0.9 if "<CARNOT_CERT_STATE:SAT>" in text else 0.1

    aggregate = evaluate_prm_guided_selection([case], scorer)

    assert len(scored_texts) == 2
    assert aggregate["cases_evaluated"] == 1
    assert aggregate["selector_auroc"] == pytest.approx(1.0)
    assert aggregate["raw_best_of_n_repair_success_rate"] == pytest.approx(1.0)
    assert aggregate["selected_repair_success_rate"] == pytest.approx(1.0)
    assert aggregate["selection_improvement_pp"] == pytest.approx(0.0)
    assert aggregate["case_selections"][0]["selected_candidate_index"] == 1
    assert aggregate["case_selections"][0]["selected_accepted"] is True


def test_req1430_ranking_tie_breaks_by_candidate_index_and_sanitizes_scores() -> None:
    """REQ-VERIFY-1430: ranking is deterministic even for ties and bad scores."""

    case = {
        "case_id": "tie_case",
        "best_of_n_success": False,
        "candidate_results": [
            "not a candidate result",
            _candidate_result(2, accepted=False, final_state="REPAIR_HINT"),
            _candidate_result(1, accepted=False, final_state="REPAIR_HINT"),
            {"candidate_index": 3, "accepted": False, "candidate": None},
        ],
    }
    calls = 0

    def scorer(_text: str) -> float:
        nonlocal calls
        calls += 1
        return math.nan if calls == 3 else 0.5

    ranked = rank_case_candidates(case, scorer)

    assert ranked["selected_candidate_index"] == 1
    assert ranked["selected_accepted"] is False
    assert ranked["candidate_scores"][2]["score"] == float("-inf")
    assert candidate_process_text(None) == ""

    empty_ranked = rank_case_candidates({"case_id": "empty_case"}, lambda _text: 1.0)
    assert empty_ranked["selected_candidate_index"] is None
    assert empty_ranked["candidate_scores"] == []


def test_req1430_empty_candidate_pool_reports_zero_rates() -> None:
    """REQ-VERIFY-1430: empty candidate pools are deterministic and non-ready."""

    aggregate = evaluate_prm_guided_selection([], lambda _text: 0.0)

    assert aggregate["cases_evaluated"] == 0
    assert aggregate["selector_auroc"] == pytest.approx(0.5)
    assert aggregate["raw_best_of_n_repair_success_rate"] == pytest.approx(0.0)
    assert aggregate["selected_repair_success_rate"] == pytest.approx(0.0)
    assert aggregate["selection_improvement_pp"] == pytest.approx(0.0)


def test_req1430_tie_aware_auroc_edges() -> None:
    """REQ-VERIFY-1430: selector AUROC handles ties and one-class labels."""

    assert tie_aware_auroc([1, 0], [0.5, 0.5]) == pytest.approx(0.5)
    assert tie_aware_auroc([1, 0], [0.2, 0.8]) == pytest.approx(0.0)
    assert tie_aware_auroc([1, 1], [0.2, 0.8]) == pytest.approx(0.5)
