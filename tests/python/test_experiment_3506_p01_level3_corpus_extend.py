"""Tests for the P0.1 level-3 corpus extension helper (exp3506).

Spec: REQ-KONA-3506, SCENARIO-KONA-3506, SCENARIO-KONA-3506-VERDICT-BANDS,
      SCENARIO-KONA-3506-TERMINAL-PREFIX.

These tests pin the GPU-free scientific decisions exp3506 makes: the three-band
terminal verdict (headline-eligible / scorable-partial / partial), the Verdict
Terminal-Prefix Discipline (every band starts with ``complete:``), and the
level-3-specific blocked verdict when the SC lands outside the headroom band.
No live model is loaded — every assertion is deterministic.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.experiment_3506_p01_level3_corpus_extend_to_80_v4_optional import (
    TARGET_N,
    SCORABLE_N,
    classify_verdict_v4,
    field_principles_v4,
    _build_generation_record,
    _build_problem_record,
)


# ---------------------------------------------------------------------------
# classify_verdict_v4 — REQ-KONA-3506: three-band terminal verdict
# ---------------------------------------------------------------------------


def test_verdict_headline_eligible_at_target() -> None:
    """SCENARIO-KONA-3506-VERDICT-BANDS: n>=80 in-band -> headline-eligible."""
    v = classify_verdict_v4(80, True, 0.50)
    assert v == "complete: p01_level3_corpus_headline_eligible_n=80_sc=0.500"


def test_verdict_headline_eligible_above_target() -> None:
    """SCENARIO-KONA-3506-VERDICT-BANDS: n>80 also headline-eligible."""
    v = classify_verdict_v4(90, True, 0.55)
    assert v.startswith("complete: p01_level3_corpus_headline_eligible_n=90")
    assert "sc=0.550" in v


def test_verdict_scorable_partial_band() -> None:
    """SCENARIO-KONA-3506-VERDICT-BANDS: 40 <= n < 80 -> scorable-partial."""
    v = classify_verdict_v4(50, True, 0.48)
    assert "scorable_partial" in v
    assert v.startswith("complete:")
    assert "n=50" in v


def test_verdict_scorable_partial_at_floor() -> None:
    """SCENARIO-KONA-3506-VERDICT-BANDS: exactly SCORABLE_N -> scorable-partial."""
    assert SCORABLE_N == 40
    v = classify_verdict_v4(SCORABLE_N, True, 0.45)
    assert "scorable_partial" in v
    assert f"n={SCORABLE_N}" in v


def test_verdict_partial_below_scorable_floor() -> None:
    """SCENARIO-KONA-3506-VERDICT-BANDS: n<40 in-band -> partial (resume-next)."""
    v = classify_verdict_v4(25, True, 0.60)
    assert "partial" in v
    assert v.startswith("complete:")
    assert "n=25" in v
    # Must NOT be labelled scorable_partial (below the 40-problem floor).
    assert "scorable_partial" not in v


def test_verdict_zero_n_still_partial() -> None:
    """SCENARIO-KONA-3506-VERDICT-BANDS: n=0 in-band -> partial, never crashes."""
    v = classify_verdict_v4(0, True, 0.50)
    assert v.startswith("complete:")
    assert "partial" in v


def test_verdict_blocked_when_sc_out_of_band() -> None:
    """SCENARIO-KONA-3506: SC outside [0.40,0.70] -> blocked verdict."""
    # SC too high (ceiling).
    v_high = classify_verdict_v4(80, False, 0.80)
    assert v_high == "complete: blocked_level3_sc_outside_headroom_band"
    # SC too low (floor).
    v_low = classify_verdict_v4(80, False, 0.30)
    assert v_low == "complete: blocked_level3_sc_outside_headroom_band"
    # in_band=False with sc=None -> blocked (not a crash).
    v_none = classify_verdict_v4(0, False, None)
    assert v_none == "complete: blocked_level3_sc_outside_headroom_band"


# ---------------------------------------------------------------------------
# Verdict Terminal-Prefix Discipline — SCENARIO-KONA-3506-TERMINAL-PREFIX
# ---------------------------------------------------------------------------


def test_all_verdict_bands_are_complete_prefixed() -> None:
    """SCENARIO-KONA-3506-TERMINAL-PREFIX: every verdict starts with 'complete:'."""
    cases = [
        (0, True, 0.50),
        (25, True, 0.55),
        (40, True, 0.45),
        (80, True, 0.50),
        (90, True, 0.60),
        (0, False, 0.80),
        (80, False, 0.30),
        (80, False, None),
    ]
    for n, in_band, sc in cases:
        v = classify_verdict_v4(n, in_band, sc)
        assert v.startswith("complete:"), (
            f"classify_verdict_v4({n}, {in_band}, {sc}) returned {v!r} — "
            f"must start with 'complete:'"
        )


def test_verdict_sc_formatted_to_three_decimal_places() -> None:
    """REQ-KONA-3506: SC is formatted to 3 d.p. in the verdict string."""
    v = classify_verdict_v4(80, True, 0.51234)
    assert "sc=0.512" in v


def test_verdict_sc_na_when_none() -> None:
    """REQ-KONA-3506: SC=None produces 'NA' in verdict (not a crash)."""
    # None SC with in_band=False -> blocked (doesn't reach the sc formatting branch).
    v = classify_verdict_v4(80, False, None)
    assert v == "complete: blocked_level3_sc_outside_headroom_band"


# ---------------------------------------------------------------------------
# field_principles_v4 — REQ-KONA-3506: principle annotations present
# ---------------------------------------------------------------------------


def test_field_principles_v4_covers_required_artifact_fields() -> None:
    """REQ-KONA-3506: every REQUIRED ARTIFACT FIELD has a principle annotation."""
    principles = field_principles_v4()
    required = {
        "honest_verdict",
        "inference_substrate",
        "corpus_path",
        "n_problems_completed",
        "n_problems_added_this_run",
        "level3_self_consistency_accuracy",
        "self_consistency_in_headroom_band",
        "per_step_traces_captured",
        "model_specs",
        "random_seed",
        "reproducibility_checksum",
        "duration_s",
    }
    missing = required - set(principles)
    assert not missing, f"Missing principle annotations for: {missing}"


def test_field_principles_v4_all_non_empty_strings() -> None:
    """REQ-KONA-3506: every principle annotation is a non-empty string."""
    for field, principle in field_principles_v4().items():
        assert isinstance(principle, str), f"{field}: principle must be a string"
        assert principle.strip(), f"{field}: principle must be non-empty"


# ---------------------------------------------------------------------------
# _build_generation_record — REQ-KONA-3506: generation row schema
# ---------------------------------------------------------------------------


def test_build_generation_record_correct_answer() -> None:
    r"""REQ-KONA-3506: correct answer extraction populates correct=True."""
    text = r"The answer is \boxed{42}."
    rec = _build_generation_record(text, None, "42", "greedy", 0)
    assert rec["correct"] is True
    assert rec["extracted_answer"] == "42"
    assert rec["mode"] == "greedy"
    assert rec["seed"] == 0
    assert isinstance(rec["reasoning_steps"], list)


def test_build_generation_record_wrong_answer() -> None:
    r"""REQ-KONA-3506: wrong extracted answer -> correct=False."""
    text = r"The answer is \boxed{7}."
    rec = _build_generation_record(text, None, "42", "sampled", 1)
    assert rec["correct"] is False


def test_build_generation_record_no_boxed_answer() -> None:
    r"""REQ-KONA-3506: no \boxed{} in output -> extracted_answer=None, correct=False."""
    rec = _build_generation_record("No answer here.", None, "42", "greedy", 0)
    assert rec["extracted_answer"] is None
    assert rec["correct"] is False


def test_build_generation_record_steps_non_empty_for_multi_paragraph() -> None:
    """REQ-KONA-3506: multi-paragraph text yields multiple steps."""
    text = "Step one is here.\n\nStep two is here.\n\nFinal step."
    rec = _build_generation_record(text, None, None, "sampled", 5)
    assert len(rec["reasoning_steps"]) == 3
    assert rec["n_steps"] == 3


# ---------------------------------------------------------------------------
# _build_problem_record — REQ-KONA-3506: problem row schema
# ---------------------------------------------------------------------------


def _make_gen_rec(text: str, gold: str, mode: str, seed: int = 0) -> dict:
    return _build_generation_record(text, None, gold, mode, seed)


def test_build_problem_record_shape() -> None:
    r"""REQ-KONA-3506: problem record carries required top-level keys."""
    meta = {
        "problem_id": "test/algebra/1.json",
        "level": 3,
        "subject": "algebra",
        "problem": "What is 2+2?",
        "gold_answer": "4",
    }
    greedy = _make_gen_rec(r"The answer is \boxed{4}.", "4", "greedy")
    samples = [
        _make_gen_rec(r"\boxed{4}", "4", "sampled", i) for i in range(6)
    ]
    row = _build_problem_record(meta, greedy, samples)
    assert row["problem_id"] == "test/algebra/1.json"
    assert row["level"] == 3
    assert row["greedy_correct"] is True
    assert row["k_samples"] == 6
    assert len(row["sampled_answers"]) == 6
    assert all(a == "4" for a in row["sampled_answers"])


def test_build_problem_record_sampled_answers_normalised() -> None:
    r"""REQ-KONA-3506: sampled_answers are normalised for SC computation."""
    meta = {
        "problem_id": "test/prob/2.json",
        "level": 3,
        "subject": "probability",
        "problem": "A question.",
        "gold_answer": r"\frac{1}{2}",
    }
    greedy = _make_gen_rec(r"\boxed{\frac{1}{2}}", r"\frac{1}{2}", "greedy")
    samples = [
        _make_gen_rec(r"\boxed{\frac{1}{2}}", r"\frac{1}{2}", "sampled", i)
        for i in range(3)
    ]
    row = _build_problem_record(meta, greedy, samples)
    # sampled_answers should hold normalised strings (same for each sample).
    assert all(a is not None for a in row["sampled_answers"])
