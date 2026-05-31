import pytest
from scripts.experiment_3552_p01_route2_headroom_corpus_greedy_wrong_construction_v3 import (
    has_selectable_headroom,
    compute_corpus_stats,
    classify_verdict_3552,
    _oracle_is_correct,
)

def test_has_selectable_headroom():
    # SCENARIO-AR-052-01: Headroom condition correctly identified
    # Case 1: Greedy is wrong, and at least one sample is correct
    record = {
        "gold_answer_norm": "42",
        "greedy": {"extracted_answer_norm": "41"},
        "sampled_answers": ["40", "42", "43"]
    }
    assert has_selectable_headroom(record) is True

    # Case 2: Greedy is correct (no headroom, SC tracks greedy)
    record = {
        "gold_answer_norm": "42",
        "greedy": {"extracted_answer_norm": "42"},
        "sampled_answers": ["40", "42", "43"]
    }
    assert has_selectable_headroom(record) is False

    # Case 3: Greedy is wrong, but all samples are wrong (no recoverable headroom)
    record = {
        "gold_answer_norm": "42",
        "greedy": {"extracted_answer_norm": "41"},
        "sampled_answers": ["40", "41", "43"]
    }
    assert has_selectable_headroom(record) is False

    # Case 4: No gold answer
    record = {
        "gold_answer_norm": None,
        "greedy": {"extracted_answer_norm": "41"},
        "sampled_answers": ["40", "42", "43"]
    }
    assert has_selectable_headroom(record) is False

def test_oracle_is_correct():
    record = {
        "gold_answer_norm": "42",
        "sampled_answers": ["40", "42", "43"]
    }
    assert _oracle_is_correct(record) is True

    record_false = {
        "gold_answer_norm": "42",
        "sampled_answers": ["40", "41", "43"]
    }
    assert _oracle_is_correct(record_false) is False

def test_compute_corpus_stats():
    # SCENARIO-AR-052-02: Oracle strictly exceeds SC
    # SCENARIO-AR-052-03: Artifact reports required headroom bounds
    kept_records = [
        {
            "gold_answer_norm": "42",
            "greedy_correct": False,
            "sampled_answers": ["41", "42", "41"],  # SC=41 (wrong), oracle=correct
            "mode": "sampled"
        },
        {
            "gold_answer_norm": "100",
            "greedy_correct": False,
            "sampled_answers": ["100", "100", "99"],  # SC=100 (correct), oracle=correct
            "mode": "sampled"
        }
    ]
    
    stats = compute_corpus_stats(kept_records)
    
    assert stats["greedy_accuracy"] == 0.0
    assert stats["oracle_accuracy"] == 1.0
    assert stats["self_consistency_accuracy"] == 0.5  # First record wrong (41), second record right (100)
    assert stats["selectable_headroom"] == 0.5
    assert stats["oracle_exceeds_sc"] is True

    empty_stats = compute_corpus_stats([])
    assert empty_stats["greedy_accuracy"] == 0.0
    assert empty_stats["oracle_accuracy"] == 0.0
    assert empty_stats["selectable_headroom"] == 0.0
    assert empty_stats["oracle_exceeds_sc"] is False

def test_classify_verdict_3552():
    # Target N is 40
    # Success
    assert classify_verdict_3552(40, 0.8, 0.4).startswith("complete: p01_greedy_wrong_headroom_corpus_built")
    # Partial
    assert classify_verdict_3552(20, 0.8, 0.4).startswith("complete: p01_greedy_wrong_headroom_corpus_partial")
    # Empty / tracks oracle
    assert classify_verdict_3552(0, 0.0, 0.0) == "complete: p01_sc_tracks_oracle_even_when_greedy_wrong_route2_premise_terminally_bounded_on_nl_math"
