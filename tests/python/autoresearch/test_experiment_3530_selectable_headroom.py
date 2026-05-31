r"""Tests for the P0.1 selectable-headroom corpus builder (exp3530).

Spec anchor: REQ-AR-050 (P0.1 Difficulty-Matched Corpus v2 - Adaptive Level
Selection + Process Traces) in openspec/capabilities/autoresearch/spec.md.
This test file covers the NEW selectable-headroom filter property introduced
in exp3530 — building on the same REQ-AR-050 scaffold because the new corpus
is a direct extension of that requirement toward a positive-control corpus.

These tests exercise the GPU-free logic only: the pure functions in the
exp3530 script that are not dependent on CUDA or a cached GGUF.  The
GPU-bound generation loop in ``main()`` is integration-only.

Every test asserts real behavior; none are skipped (CLAUDE.md "Tests Must Run
and Assert").
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SCRIPT = (
    _REPO_ROOT
    / "scripts"
    / "experiment_3530_p01_route2_selectable_headroom_corpus_build_v1.py"
)


def _load_script_module():
    """Import the exp3530 script by file path (it lives outside the package)."""
    spec = importlib.util.spec_from_file_location("exp3530_v1", _SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


EXP = _load_script_module()


# ---------------------------------------------------------------------------
# SCENARIO-AR-050-01 (extended): has_selectable_headroom filter logic
# ---------------------------------------------------------------------------

def test_has_selectable_headroom_true_when_correct_present_and_sc_wrong():
    # REQ-AR-050 / SCENARIO-AR-050-01
    # Correct answer "3" is present in samples (once) but majority is "7".
    # This is exactly selectable headroom: oracle can recover it, SC cannot.
    record = {
        "gold_answer_norm": "3",
        "sampled_answers": ["7", "7", "7", "3", "7", "7", "7", "7"],
    }
    assert EXP.has_selectable_headroom(record) is True


def test_has_selectable_headroom_false_when_correct_is_sc_majority():
    # REQ-AR-050 / SCENARIO-AR-050-01
    # Correct answer "3" IS the SC majority — no headroom, SC already works.
    record = {
        "gold_answer_norm": "3",
        "sampled_answers": ["3", "3", "3", "7", "3", "7", "3", "3"],
    }
    assert EXP.has_selectable_headroom(record) is False


def test_has_selectable_headroom_false_when_correct_absent():
    # REQ-AR-050 / SCENARIO-AR-050-01
    # Correct answer "3" never appears — oracle fails too; not selectable.
    record = {
        "gold_answer_norm": "3",
        "sampled_answers": ["7", "7", "7", "7", "8", "7", "7", "7"],
    }
    assert EXP.has_selectable_headroom(record) is False


def test_has_selectable_headroom_false_when_all_wrong_no_correct():
    # REQ-AR-050 / SCENARIO-AR-050-01
    # No sample is correct and no majority is correct — still not selectable.
    record = {
        "gold_answer_norm": "42",
        "sampled_answers": ["1", "2", "3", "4", "1", "2", "3", "4"],
    }
    assert EXP.has_selectable_headroom(record) is False


def test_has_selectable_headroom_false_when_no_gold():
    # REQ-AR-050 / SCENARIO-AR-050-01
    # Missing gold answer — should not raise; returns False.
    record = {
        "gold_answer_norm": None,
        "sampled_answers": ["3", "3", "3", "7", "3", "7", "3", "3"],
    }
    assert EXP.has_selectable_headroom(record) is False


def test_has_selectable_headroom_false_when_empty_samples():
    # REQ-AR-050 / SCENARIO-AR-050-01
    # No samples at all — no correct present, no majority.
    record = {
        "gold_answer_norm": "3",
        "sampled_answers": [],
    }
    assert EXP.has_selectable_headroom(record) is False


def test_has_selectable_headroom_handles_all_none_samples():
    # REQ-AR-050 / SCENARIO-AR-050-01
    # All samples produced None (no \boxed{} extracted) — not selectable.
    record = {
        "gold_answer_norm": "3",
        "sampled_answers": [None, None, None, None],
    }
    assert EXP.has_selectable_headroom(record) is False


def test_has_selectable_headroom_with_mixed_none_and_correct():
    # REQ-AR-050 / SCENARIO-AR-050-01
    # One correct answer among mostly None — correct IS present, non-None
    # majority might be the correct one. Check: if "3" is the only parseable
    # answer it IS the majority of non-None answers → not selectable headroom
    # (SC would vote "3" = correct).
    record = {
        "gold_answer_norm": "3",
        "sampled_answers": [None, None, None, "3", None, None, None, None],
    }
    # Only non-None is "3" → majority = "3" = gold → not selectable (SC wins).
    assert EXP.has_selectable_headroom(record) is False


def test_has_selectable_headroom_majority_tie_resolved_by_max_count():
    # REQ-AR-050 / SCENARIO-AR-050-01
    # Tie between "3" and "7" (4 each) — max() picks the first alphabetically
    # in a dict traversal; both "3" and "7" have equal count, so the result
    # is determined by dict insertion order (CPython 3.7+ preserves it).
    # With equal counts Python's max() with key=count picks whichever comes
    # first in iteration — we just need to verify the function does not raise
    # and returns a bool consistently.
    record = {
        "gold_answer_norm": "3",
        "sampled_answers": ["3", "7", "3", "7", "3", "7", "3", "7"],
    }
    result = EXP.has_selectable_headroom(record)
    # With 4 "3"s and 4 "7"s: counts = {"3": 4, "7": 4}; max picks "3" (first
    # inserted, per CPython dict order). "3" == gold → SC "wins" → not headroom.
    assert result is False


# ---------------------------------------------------------------------------
# SCENARIO-AR-050-01 (new): compute_corpus_stats invariants
# ---------------------------------------------------------------------------

def test_compute_corpus_stats_empty_returns_zero():
    # REQ-AR-050 / SCENARIO-AR-050-01
    stats = EXP.compute_corpus_stats([])
    assert stats["oracle_accuracy"] == 0.0
    assert stats["self_consistency_accuracy"] == 0.0
    assert stats["selectable_headroom"] == 0.0
    assert stats["oracle_exceeds_sc"] is False


def test_compute_corpus_stats_kept_corpus_oracle_1_sc_0():
    # REQ-AR-050 / SCENARIO-AR-050-01
    # A "kept" corpus (all records have selectable headroom) should produce
    # oracle=1.0 and sc=0.0 by the invariant of the filter.
    kept_records = [
        {
            "gold_answer_norm": "3",
            "sampled_answers": ["7", "7", "7", "3", "7", "7", "7", "7"],
        },
        {
            "gold_answer_norm": "5",
            "sampled_answers": ["9", "9", "9", "5", "9", "9", "9", "9"],
        },
    ]
    stats = EXP.compute_corpus_stats(kept_records)
    assert stats["oracle_accuracy"] == 1.0
    assert stats["self_consistency_accuracy"] == 0.0
    assert stats["selectable_headroom"] == pytest.approx(1.0)
    assert stats["oracle_exceeds_sc"] is True


def test_compute_corpus_stats_headroom_equals_oracle_minus_sc():
    # REQ-AR-050 / SCENARIO-AR-050-01
    # Mixed corpus where oracle > SC > 0.
    records = [
        # oracle correct (answer "3" present), sc wrong (majority "7").
        {
            "gold_answer_norm": "3",
            "sampled_answers": ["7", "7", "7", "3", "7", "7", "7", "7"],
        },
        # oracle wrong (answer "5" absent), sc wrong (majority "9" != "5").
        {
            "gold_answer_norm": "5",
            "sampled_answers": ["9", "9", "9", "9", "9", "9", "9", "9"],
        },
    ]
    stats = EXP.compute_corpus_stats(records)
    assert stats["oracle_accuracy"] == pytest.approx(0.5)
    assert stats["self_consistency_accuracy"] == pytest.approx(0.0)
    assert stats["selectable_headroom"] == pytest.approx(0.5)
    assert stats["oracle_exceeds_sc"] is True


# ---------------------------------------------------------------------------
# SCENARIO-AR-050-02 (new): classify_verdict_3530
# ---------------------------------------------------------------------------

def test_classify_verdict_success_when_n_kept_at_target():
    # REQ-AR-050 / SCENARIO-AR-050-02
    verdict = EXP.classify_verdict_3530(40, 1.0, 0.0)
    assert verdict.startswith("complete:")
    assert "selectable_headroom_corpus_built" in verdict
    assert "n=40" in verdict


def test_classify_verdict_partial_when_below_target():
    # REQ-AR-050 / SCENARIO-AR-050-02
    verdict = EXP.classify_verdict_3530(20, 1.0, 0.0)
    assert verdict.startswith("complete:")
    assert "partial" in verdict
    assert "n=20" in verdict


def test_classify_verdict_no_headroom_when_n_zero():
    # REQ-AR-050 / SCENARIO-AR-050-02
    verdict = EXP.classify_verdict_3530(0, 0.0, 0.0)
    assert verdict.startswith("complete:")
    assert "no_selectable_headroom" in verdict


def test_classify_verdict_always_complete_prefix():
    # REQ-AR-050 / SCENARIO-AR-050-02
    # Any call must produce a verdict starting with "complete:" per
    # Verdict Terminal-Prefix Discipline (CLAUDE.md).
    for n, o, s in [(0, 0.0, 0.0), (10, 1.0, 0.0), (50, 1.0, 0.0), (5, 0.8, 0.6)]:
        v = EXP.classify_verdict_3530(n, o, s)
        assert v.startswith("complete:"), f"verdict {v!r} lacks complete: prefix"


# ---------------------------------------------------------------------------
# SCENARIO-AR-050-03 (new): field_principles_3530 coverage
# ---------------------------------------------------------------------------

_REQUIRED_FIELDS = [
    "honest_verdict",
    "inference_substrate",
    "corpus_path",
    "n_problems_kept",
    "self_consistency_accuracy",
    "oracle_accuracy",
    "selectable_headroom",
    "oracle_exceeds_sc",
    "per_step_traces_captured",
    "model_specs",
    "random_seed",
    "reproducibility_checksum",
    "duration_s",
]


def test_field_principles_covers_all_required_fields():
    # REQ-AR-050 / SCENARIO-AR-050-03
    principles = EXP.field_principles_3530()
    for field in _REQUIRED_FIELDS:
        assert field in principles, f"missing principle for field '{field}'"
        assert len(principles[field]) >= 20, (
            f"principle for '{field}' is too short (< 20 chars): {principles[field]!r}"
        )


# ---------------------------------------------------------------------------
# SCENARIO-AR-050-04 (new): _oracle_is_correct helper
# ---------------------------------------------------------------------------

def test_oracle_is_correct_true_when_one_sample_matches():
    # REQ-AR-050 / SCENARIO-AR-050-04
    record = {
        "gold_answer_norm": "42",
        "sampled_answers": ["1", "42", "3"],
    }
    assert EXP._oracle_is_correct(record) is True


def test_oracle_is_correct_false_when_none_match():
    # REQ-AR-050 / SCENARIO-AR-050-04
    record = {
        "gold_answer_norm": "42",
        "sampled_answers": ["1", "2", "3"],
    }
    assert EXP._oracle_is_correct(record) is False


def test_oracle_is_correct_false_when_gold_none():
    # REQ-AR-050 / SCENARIO-AR-050-04
    record = {
        "gold_answer_norm": None,
        "sampled_answers": ["1", "2", "3"],
    }
    assert EXP._oracle_is_correct(record) is False


# ---------------------------------------------------------------------------
# SCENARIO-AR-050-01 (new): _build_generation_record schema
# ---------------------------------------------------------------------------

def test_build_generation_record_schema():
    # REQ-AR-050 / SCENARIO-AR-050-01
    text = r"The answer is \boxed{7}."
    rec = EXP._build_generation_record(text, [-1.0, -0.5], "7", "sampled", 42)
    assert rec["mode"] == "sampled"
    assert rec["seed"] == 42
    assert rec["extracted_answer"] == "7"
    assert rec["extracted_answer_norm"] == "7"
    assert rec["correct"] is True
    assert rec["mean_token_logprob"] == pytest.approx(-0.75)
    assert isinstance(rec["reasoning_steps"], list)
    assert isinstance(rec["n_steps"], int)


def test_build_generation_record_correct_false_when_wrong_answer():
    # REQ-AR-050 / SCENARIO-AR-050-01
    text = r"\boxed{99}"
    rec = EXP._build_generation_record(text, [], "7", "greedy", 0)
    assert rec["correct"] is False


# ---------------------------------------------------------------------------
# SCENARIO-AR-050-01 (new): _build_problem_record schema
# ---------------------------------------------------------------------------

def test_build_problem_record_schema():
    # REQ-AR-050 / SCENARIO-AR-050-01
    from carnot.autoresearch.corpus_p01_headroom import normalize_answer

    meta = {
        "problem_id": "p1",
        "level": 4,
        "subject": "algebra",
        "problem": "What is 2+2?",
        "gold_answer": "4",
    }
    greedy = EXP._build_generation_record(r"\boxed{4}", [], "4", "greedy", 0)
    samples = [
        EXP._build_generation_record(r"\boxed{3}", [], "4", "sampled", i)
        for i in range(3)
    ]
    rec = EXP._build_problem_record(meta, greedy, samples)
    assert rec["problem_id"] == "p1"
    assert rec["level"] == 4
    assert rec["gold_answer_norm"] == normalize_answer("4")
    assert len(rec["sampled_answers"]) == 3
    assert rec["k_samples"] == 3
    assert rec["greedy_correct"] is True
    assert "has_selectable_headroom" in rec


# ---------------------------------------------------------------------------
# SCENARIO-AR-050-02 (new): RANDOM_SEED is not the experiment number
# ---------------------------------------------------------------------------

def test_random_seed_is_not_experiment_id():
    # REQ-AR-050 / SCENARIO-AR-050-02
    # Per CLAUDE.md Adversarial Artifact Verification, seed != experiment_id.
    assert EXP.RANDOM_SEED != EXP.EXP_ID


def test_random_seed_matches_expected_sha256_derived_value():
    # REQ-AR-050 / SCENARIO-AR-050-02
    import hashlib
    expected = int(
        hashlib.sha256(
            b"HuggingFaceH4/MATH-500:test:level4-5:selectable_headroom:v1"
        ).hexdigest()[:8],
        16,
    )
    assert EXP.RANDOM_SEED == expected
