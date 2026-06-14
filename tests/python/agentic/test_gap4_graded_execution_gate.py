"""Tests for the GAP-4 guarded graded execution gate.

REQ-VERIFY-4187 / SCENARIO-VERIFY-4187: the production gate promotes only
demo-perfect executed predictions within tau, blocks high-vote vote overrides,
and leaves agreement as a confidence label rather than a selector.
"""

from carnot.agentic.gap4_graded_execution_gate import (
    non_exact_band_precision,
    normalized_hamming,
    select_guarded_graded_candidate,
)


def test_req_verify_4187_requires_demo_fit_exactly_one() -> None:
    candidates = [{"grid": [[1]], "votes": 1}]

    result = select_guarded_graded_candidate(
        candidates,
        prediction=[[1]],
        demo_fit=0.999,
        tau=0.005,
    )

    assert result["gate_fired"] is False
    assert result["selected_index"] is None
    assert result["reason"] == "demo_fit_not_exact"


def test_req_verify_4187_promotes_min_hamming_within_tau() -> None:
    candidates = [
        {"grid": [[9, 9], [9, 9]], "votes": 20},
        {"grid": [[1, 2], [3, 4]], "votes": 1},
    ]

    result = select_guarded_graded_candidate(
        candidates,
        prediction=[[1, 2], [3, 4]],
        demo_fit=1.0,
        tau=0.005,
    )

    assert result["gate_fired"] is True
    assert result["selected_index"] == 1
    assert result["min_hamming"] == 0.0
    assert result["agreement_confidence_label"] is False


def test_req_verify_4187_blocks_high_vote_override_without_gold_oracle() -> None:
    candidates = [
        {"grid": [[7]], "votes": 945},
        {"grid": [[3]], "votes": 32},
    ]

    result = select_guarded_graded_candidate(
        candidates,
        prediction=[[3]],
        demo_fit=1.0,
        task_id="25094a63",
        tau=0.005,
        high_vote_guard_threshold=900,
        agreement_confidence_label=True,
    )

    assert result["gate_fired"] is False
    assert result["guard_blocked"] is True
    assert result["selected_index"] is None
    assert result["would_select_index"] == 1
    assert result["agreement_confidence_label"] is True


def test_req_verify_4187_normalized_hamming_shape_mismatch_is_worse() -> None:
    pred = [[1, 2], [3, 4]]

    assert normalized_hamming(pred, pred) == 0.0
    assert normalized_hamming([[1, 2], [3, 0]], pred) == 0.25
    assert normalized_hamming([[1, 2]], pred) > 1.0
    assert normalized_hamming(None, pred) > 1.0


def test_req_verify_4187_abstains_on_missing_prediction_or_empty_pool() -> None:
    missing_pred = select_guarded_graded_candidate(
        [{"grid": [[1]], "votes": 1}],
        prediction=None,
        demo_fit=1.0,
    )
    empty_pool = select_guarded_graded_candidate(
        [],
        prediction=[[1]],
        demo_fit=1.0,
    )

    assert missing_pred["reason"] == "prediction_missing"
    assert empty_pool["reason"] == "candidate_pool_empty"


def test_req_verify_4187_band_precision_skips_ineligible_rows() -> None:
    entries = [
        {"candidates": [{"grid": [[1]], "correct": True}]},
        {"candidates": []},
    ]
    programs = [
        {"demo_fit": 0.5, "pred_grid": [[1]]},
        {"demo_fit": 1.0, "pred_grid": [[1]]},
    ]

    assert non_exact_band_precision(entries, programs) == {
        "tau": 0.02,
        "definition": "demo-perfect entries whose closest candidate has 0 < min_hamming <= tau",
        "correct": 0,
        "total": 0,
        "precision": None,
    }
