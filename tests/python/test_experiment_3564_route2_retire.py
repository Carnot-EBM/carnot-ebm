"""Tests for experiment 3564 - Route 2 NL-Math Final Headroom or Retire.

Test traces to:
REQ-AR-051
SCENARIO-AR-051-01
"""
import json
import tempfile
from pathlib import Path

from scripts.experiment_3564_p01_route2_nlmath_final_headroom_or_retire_v4 import (
    _build_artifact,
    compute_multi_verifier_scores,
    classify_terminal_verdict
)

def test_build_artifact_generates_valid_schema():
    art = _build_artifact(
        verdict="complete: test_verdict",
        duration_s=10.0,
        n_kept=40,
        greedy_acc=0.0,
        sc_acc=0.5,
        oracle_acc=0.8,
        strong_sc_acc=0.55,
        multi_verifier_acc=0.6,
        multi_verifier_makes_distinct_selections=True,
        best_condition="multi_verifier",
        flip_count=5,
        net_correctness_gain=2,
        delta_best=0.05,
        paired_significance={"mcnemar_p": 0.04, "bootstrap_ci95": [0.01, 0.09]},
        route2_nlmath_terminal="positive",
        model_specs={"name": "test-model"},
        repro_checksum="abcd1234efgh5678",
    )
    assert art["honest_verdict"] == "complete: test_verdict"
    assert art["inference_substrate"] == "live_llm_inference"
    assert "data/p01_greedy_wrong_headroom_corpus.jsonl" in art["corpus_path"]
    assert "harder competition-grade" in art["construction_criterion"]
    assert art["k_candidates"] >= 16
    assert art["problem_pool_size"] == 0  # not passed in test but exist in schema
    assert art["route2_nlmath_terminal"] == "positive"
    assert art["multi_verifier_accuracy"] == 0.6
    assert art["multi_verifier_makes_distinct_selections"] is True

def test_compute_multi_verifier_scores():
    records = [
        {
            "gold_answer_norm": "1",
            "samples": [
                {"extracted_answer_norm": "1", "text": "1", "reasoning_steps": ["1"]},
                {"extracted_answer_norm": "2", "text": "2", "reasoning_steps": ["2"]},
            ]
        }
    ]
    # Simple verifier stub
    class MockVerifier:
        def energy(self, text): return 0.5
        def score(self, text): return 0.5
    
    class MockVerifiers:
        def __init__(self):
            self.ising = MockVerifier()
            self.ebmcot = MockVerifier()
            self.tier0r = MockVerifier()
            self.tier0u = MockVerifier()

    scores = compute_multi_verifier_scores(records, verifiers=MockVerifiers())
    assert len(scores) == 1
    assert len(scores[0]) == 2
    # Each score is the sum of 4 verifiers returning 0.5 -> 2.0
    assert scores[0][0] == 2.0
    assert scores[0][1] == 2.0

def test_classify_terminal_verdict():
    # G0 fair-test possible
    verdict, term = classify_terminal_verdict(
        n_kept=45, oracle_exceeds_sc=True, mv_distinct=True,
        net_gain=2, delta=0.05, p_val=0.04
    )
    assert "beats_strong_sc" in verdict
    assert term == "positive"
    
    # Informative negative
    verdict, term = classify_terminal_verdict(
        n_kept=45, oracle_exceeds_sc=True, mv_distinct=True,
        net_gain=-1, delta=-0.05, p_val=0.5
    )
    assert "informative_negative" in verdict
    assert term == "informative_negative_with_headroom"
    
    # Retire
    verdict, term = classify_terminal_verdict(
        n_kept=10, oracle_exceeds_sc=False, mv_distinct=True,
        net_gain=0, delta=0.0, p_val=1.0
    )
    assert "permanently_retired" in verdict
    assert term == "permanently_retired_no_headroom"
