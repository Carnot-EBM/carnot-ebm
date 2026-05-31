"""Tests for the P0.1 level-3 corpus extension helper (exp3516, v5).

Spec: REQ-KONA-3516, SCENARIO-KONA-3516, SCENARIO-KONA-3516-TERMINAL-PREFIX,
      SCENARIO-KONA-3516-SEED.

These tests pin the GPU-free scientific decisions exp3516 makes: the three-band
terminal verdict (headline-eligible / scorable-partial / partial), the Verdict
Terminal-Prefix Discipline (every band starts with ``complete:``), the content-derived
seed invariant (RANDOM_SEED != EXP_ID), and the principle annotations.
No live model is loaded — every assertion is deterministic.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.experiment_3516_p01_level3_corpus_extend_to_80_v5_optional import (
    EXP_ID,
    RANDOM_SEED,
    SCORABLE_N,
    TARGET_N,
    _build_generation_record,
    _build_problem_record,
    classify_verdict_v5,
    field_principles_v5,
)


# ---------------------------------------------------------------------------
# REQ-KONA-3516: content-derived seed invariant
# (SCENARIO-KONA-3516-SEED)
# ---------------------------------------------------------------------------


def test_random_seed_is_not_exp_id() -> None:
    """SCENARIO-KONA-3516-SEED: RANDOM_SEED != EXP_ID (no tautology seed)."""
    assert RANDOM_SEED != EXP_ID, (
        f"RANDOM_SEED={RANDOM_SEED} must not equal EXP_ID={EXP_ID}; "
        "the seed must be content-derived, not the experiment number."
    )


def test_exp_id_is_3516() -> None:
    """REQ-KONA-3516: EXP_ID must be 3516."""
    assert EXP_ID == 3516


def test_random_seed_is_not_v4_seed() -> None:
    """REQ-KONA-3516: v5 seed differs from v4 seed (3506) — distinct runs."""
    assert RANDOM_SEED != 3506


def test_random_seed_is_positive_integer() -> None:
    """REQ-KONA-3516: RANDOM_SEED is a positive integer usable by llama_cpp."""
    assert isinstance(RANDOM_SEED, int)
    assert RANDOM_SEED > 0


# ---------------------------------------------------------------------------
# classify_verdict_v5 — REQ-KONA-3516 / SCENARIO-KONA-3516 three-band verdict
# ---------------------------------------------------------------------------


def test_verdict_headline_eligible_at_target() -> None:
    """SCENARIO-KONA-3516: n>=80 in-band -> headline-eligible verdict."""
    v = classify_verdict_v5(80, True, 0.50)
    assert v == "complete: p01_level3_corpus_headline_eligible_n=80_sc=0.500"


def test_verdict_headline_eligible_above_target() -> None:
    """SCENARIO-KONA-3516: n>80 in-band also yields headline-eligible verdict."""
    v = classify_verdict_v5(90, True, 0.55)
    assert v.startswith("complete: p01_level3_corpus_headline_eligible_n=90")
    assert "sc=0.550" in v


def test_verdict_scorable_partial_middle_band() -> None:
    """SCENARIO-KONA-3516: 40 <= n < 80 in-band -> scorable-partial verdict."""
    v = classify_verdict_v5(50, True, 0.48)
    assert "scorable_partial" in v
    assert v.startswith("complete:")
    assert "n=50" in v


def test_verdict_scorable_partial_at_floor() -> None:
    """SCENARIO-KONA-3516: exactly SCORABLE_N -> scorable-partial verdict."""
    assert SCORABLE_N == 40
    v = classify_verdict_v5(SCORABLE_N, True, 0.45)
    assert "scorable_partial" in v
    assert f"n={SCORABLE_N}" in v


def test_verdict_partial_below_scorable_floor() -> None:
    """SCENARIO-KONA-3516: n<40 in-band -> partial (not scorable-partial)."""
    v = classify_verdict_v5(25, True, 0.60)
    assert "partial" in v
    assert v.startswith("complete:")
    assert "n=25" in v
    assert "scorable_partial" not in v


def test_verdict_zero_n_still_terminal() -> None:
    """SCENARIO-KONA-3516: n=0 in-band -> partial verdict, never crashes."""
    v = classify_verdict_v5(0, True, 0.50)
    assert v.startswith("complete:")
    assert "partial" in v


def test_verdict_blocked_when_sc_out_of_band() -> None:
    """SCENARIO-KONA-3516: SC outside [0.40, 0.70] -> blocked verdict."""
    v_high = classify_verdict_v5(80, False, 0.80)
    assert v_high == "complete: blocked_level3_sc_outside_headroom_band"
    v_low = classify_verdict_v5(80, False, 0.30)
    assert v_low == "complete: blocked_level3_sc_outside_headroom_band"
    v_none = classify_verdict_v5(0, False, None)
    assert v_none == "complete: blocked_level3_sc_outside_headroom_band"


# ---------------------------------------------------------------------------
# Verdict Terminal-Prefix Discipline — SCENARIO-KONA-3516-TERMINAL-PREFIX
# ---------------------------------------------------------------------------


def test_all_verdict_bands_are_complete_prefixed() -> None:
    """SCENARIO-KONA-3516-TERMINAL-PREFIX: every verdict starts with 'complete:'."""
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
        v = classify_verdict_v5(n, in_band, sc)
        assert v.startswith("complete:"), (
            f"classify_verdict_v5({n}, {in_band}, {sc}) returned {v!r} — "
            "must start with 'complete:'"
        )


def test_verdict_sc_formatted_to_three_decimal_places() -> None:
    """REQ-KONA-3516: SC is formatted to 3 d.p. in the verdict string."""
    v = classify_verdict_v5(80, True, 0.51234)
    assert "sc=0.512" in v


def test_verdict_sc_na_when_none_and_out_of_band() -> None:
    """REQ-KONA-3516: sc=None with in_band=False -> blocked verdict (no crash)."""
    v = classify_verdict_v5(80, False, None)
    assert v == "complete: blocked_level3_sc_outside_headroom_band"


def test_target_n_is_80() -> None:
    """REQ-KONA-3516: TARGET_N must be 80 (headline-eligibility threshold)."""
    assert TARGET_N == 80


# ---------------------------------------------------------------------------
# field_principles_v5 — REQ-KONA-3516: principle annotations present
# ---------------------------------------------------------------------------


def test_field_principles_v5_covers_required_artifact_fields() -> None:
    """REQ-KONA-3516: every REQUIRED ARTIFACT FIELD has a principle annotation."""
    principles = field_principles_v5()
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


def test_field_principles_v5_all_non_empty_strings() -> None:
    """REQ-KONA-3516: every principle annotation is a non-empty string."""
    for field, principle in field_principles_v5().items():
        assert isinstance(principle, str), f"{field}: principle must be a string"
        assert principle.strip(), f"{field}: principle must be non-empty"


def test_field_principles_v5_random_seed_mentions_content_derived() -> None:
    """REQ-KONA-3516: random_seed principle must mention content-derived provenance."""
    principle = field_principles_v5()["random_seed"]
    # The principle must warn against using the experiment number as the seed.
    assert "content" in principle.lower() or "sha" in principle.lower() or "derived" in principle.lower(), (
        "random_seed principle must document that the seed is content-derived, not the exp number"
    )


# ---------------------------------------------------------------------------
# _build_generation_record — REQ-KONA-3516: generation row schema
# ---------------------------------------------------------------------------


def test_build_generation_record_correct_answer() -> None:
    r"""REQ-KONA-3516: correct answer extraction populates correct=True."""
    text = r"The answer is \boxed{42}."
    rec = _build_generation_record(text, None, "42", "greedy", RANDOM_SEED)
    assert rec["correct"] is True
    assert rec["extracted_answer"] == "42"
    assert rec["mode"] == "greedy"
    assert rec["seed"] == RANDOM_SEED
    assert isinstance(rec["reasoning_steps"], list)


def test_build_generation_record_wrong_answer() -> None:
    r"""REQ-KONA-3516: wrong extracted answer -> correct=False."""
    text = r"The answer is \boxed{7}."
    rec = _build_generation_record(text, None, "42", "sampled", 1)
    assert rec["correct"] is False


def test_build_generation_record_no_boxed_answer() -> None:
    r"""REQ-KONA-3516: no \boxed{} in output -> extracted_answer=None, correct=False."""
    rec = _build_generation_record("No answer here.", None, "42", "greedy", 0)
    assert rec["extracted_answer"] is None
    assert rec["correct"] is False


def test_build_generation_record_step_parsing() -> None:
    """REQ-KONA-3516: multi-paragraph text yields multiple steps."""
    text = "Step one.\n\nStep two.\n\nFinal step."
    rec = _build_generation_record(text, None, None, "sampled", 5)
    assert len(rec["reasoning_steps"]) == 3
    assert rec["n_steps"] == 3


# ---------------------------------------------------------------------------
# _build_problem_record — REQ-KONA-3516: problem row schema
# ---------------------------------------------------------------------------


def _make_gen_rec(text: str, gold: str, mode: str, seed: int = 0) -> dict:
    return _build_generation_record(text, None, gold, mode, seed)


def test_build_problem_record_shape() -> None:
    r"""REQ-KONA-3516: problem record carries required top-level keys."""
    meta = {
        "problem_id": "test/algebra/1.json",
        "level": 3,
        "subject": "algebra",
        "problem": "What is 2+2?",
        "gold_answer": "4",
    }
    greedy = _make_gen_rec(r"The answer is \boxed{4}.", "4", "greedy")
    samples = [_make_gen_rec(r"\boxed{4}", "4", "sampled", i) for i in range(6)]
    row = _build_problem_record(meta, greedy, samples)
    assert row["problem_id"] == "test/algebra/1.json"
    assert row["level"] == 3
    assert row["greedy_correct"] is True
    assert row["k_samples"] == 6
    assert len(row["sampled_answers"]) == 6
    assert all(a == "4" for a in row["sampled_answers"])
