import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.experiment_3564_p01_route2_nlmath_final_headroom_or_retire_v4 import (
    _build_generation_record,
    has_selectable_headroom,
    compute_multi_verifier_scores,
    classify_terminal_verdict,
    _field_provenance_3564,
    _build_artifact
)

class DummyVerifiers:
    class DummyEbmCot:
        def energy(self, text): return 0.5
    class DummyIsing:
        def energy(self, text): return 0.3
    class DummyTier0:
        def score(self, text): return 0.1

    def __init__(self):
        self.ebmcot = self.DummyEbmCot()
        self.ising = self.DummyIsing()
        self.tier0r = self.DummyTier0()
        self.tier0u = self.DummyTier0()

def test_build_generation_record():
    text = "The answer is \\boxed{42}."
    rec = _build_generation_record(text, [-0.1, -0.2], "42", "greedy", 123)
    assert rec["mode"] == "greedy"
    assert rec["seed"] == 123
    assert rec["extracted_answer"] == "42"
    assert rec["extracted_answer_norm"] == "42"
    assert rec["correct"] is True
    assert "mean_token_logprob" in rec

def test_has_selectable_headroom():
    rec = {
        "gold_answer_norm": "42",
        "greedy": {"extracted_answer_norm": "42"}, # Greedy correct -> No headroom
        "sampled_answers": ["42", "43"]
    }
    assert not has_selectable_headroom(rec)

    rec2 = {
        "gold_answer_norm": "42",
        "greedy": {"extracted_answer_norm": "43"}, # Greedy wrong
        "sampled_answers": ["43", "44"] # No sampled answer is correct -> No headroom
    }
    assert not has_selectable_headroom(rec2)

    rec3 = {
        "gold_answer_norm": "42",
        "greedy": {"extracted_answer_norm": "43"}, # Greedy wrong
        "sampled_answers": ["42", "44"] # Sampled answer correct -> Headroom!
    }
    assert has_selectable_headroom(rec3)

def test_compute_multi_verifier_scores():
    verifiers = DummyVerifiers()
    records = [{
        "samples": [
            {"text": "test", "reasoning_steps": ["step1"]}
        ]
    }]
    scores = compute_multi_verifier_scores(records, verifiers)
    assert len(scores) == 1
    assert len(scores[0]) == 1
    assert abs(scores[0][0] - 1.0) < 1e-5 # 0.5 + 0.3 + 0.1 + 0.1 = 1.0

def test_classify_terminal_verdict():
    # n_kept < TARGET_N -> no headroom
    verdict, term = classify_terminal_verdict(10, True, True, 5, 0.1, 0.01)
    assert "permanently_retired" in verdict
    assert term == "permanently_retired_no_headroom"

    # oracle_exceeds_sc = False -> no headroom
    verdict, term = classify_terminal_verdict(50, False, True, 5, 0.1, 0.01)
    assert "permanently_retired" in verdict

    # No distinct selections -> no headroom
    verdict, term = classify_terminal_verdict(50, True, False, 5, 0.1, 0.01)
    assert "permanently_retired" in verdict

    # Significant win
    verdict, term = classify_terminal_verdict(50, True, True, 5, 0.1, 0.01)
    assert "beats_strong_sc" in verdict
    assert term == "positive"

    # Informative negative (win is negative, or p > 0.05)
    verdict, term = classify_terminal_verdict(50, True, True, -1, -0.1, 0.01)
    assert "does_not_beat" in verdict
    assert term == "informative_negative_with_headroom"

def test_field_provenance_3564():
    prov = _field_provenance_3564()
    assert "honest_verdict" in prov
    assert "inference_substrate" in prov

def test_build_artifact():
    art = _build_artifact(
        verdict="test_verdict",
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
        model_specs={"name": "test"},
        repro_checksum="abcd",
        pool_size=100
    )
    assert art["honest_verdict"] == "test_verdict"
    assert art["problem_pool_size"] == 100
    assert "field_provenance" in art
    # duration_s with pool_size > 0 -> max(60, 10) = 60
    assert art["duration_s"] == 60.0
