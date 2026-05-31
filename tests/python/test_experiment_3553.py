import pytest
from scripts.experiment_3553_p01_route2_energy_vs_strong_sc_on_headroom_corpus_v3 import (
    compute_headroom_stats,
    compute_flip_metrics,
    compute_mcnemar_significance,
    build_strong_sc,
)

def test_compute_headroom_stats():
    # SCENARIO-AR-050-01: Headroom corpus oracle vs SC
    records = [
        {
            "gold_answer_norm": "42",
            "samples": [
                {"extracted_answer_norm": "41"},
                {"extracted_answer_norm": "42"},
                {"extracted_answer_norm": "41"},
                {"extracted_answer_norm": "41"},
            ]
        },
        {
            "gold_answer_norm": "100",
            "samples": [
                {"extracted_answer_norm": "100"},
                {"extracted_answer_norm": "100"},
                {"extracted_answer_norm": "99"},
                {"extracted_answer_norm": "99"},
            ]
        }
    ]
    stats = compute_headroom_stats(records)
    # problem 1: strong SC=41 (wrong), oracle=correct
    # problem 2: strong SC=100 (correct), oracle=correct
    assert stats["n"] == 2
    assert stats["oracle_accuracy"] == 1.0
    assert stats["strong_sc_accuracy"] == 0.5
    assert stats["selectable_headroom"] == 0.5
    assert stats["oracle_exceeds_sc"] is True

def test_compute_flip_metrics():
    cond = ["42", "100", "99", "1"]
    sc = ["41", "100", "100", "1"]
    gold = ["42", "100", "100", "2"]
    
    res = compute_flip_metrics(cond, sc, gold)
    # flips: problem 1 (41->42, correct), problem 3 (100->99, wrong)
    assert res["flip_count"] == 2
    assert res["flips_correct"] == 1
    assert res["flips_incorrect"] == 1
    assert res["net_correctness_gain"] == 0

def test_compute_mcnemar_significance():
    # Test significance calculation
    cond = [True, True, False, False, True]
    sc = [False, True, True, False, False]
    res = compute_mcnemar_significance(cond, sc, seed=42, n_boot=100)
    assert "mcnemar_p" in res
    assert "bootstrap_ci95" in res
    assert len(res["bootstrap_ci95"]) == 2

def test_build_strong_sc():
    records = [
        {
            "gold_answer_norm": "42",
            "samples": [
                {"extracted_answer_norm": "41"}, # weight 1
                {"extracted_answer_norm": "42"}, # weight 1/2
                {"extracted_answer_norm": "41"}, # weight 1/3
            ]
        }
    ]
    # "41" gets 1 + 1/3 = 1.33, "42" gets 0.5. SC is "41" (wrong).
    sc = build_strong_sc(records)
    assert sc[0][0] == "41"
    assert sc[0][1] is False
