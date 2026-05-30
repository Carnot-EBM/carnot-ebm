"""Tests for the hard-math HEADROOM generation-corpus helpers (exp3471).

Spec: REQ-KONA-3471, SCENARIO-KONA-3471, SCENARIO-KONA-3471-RESUME,
SCENARIO-KONA-3471-NO-HEADROOM.

These tests pin the GPU-free scientific decisions the hard-math builder makes:
math ``\\boxed{}`` answer extraction + LaTeX normalisation + correctness, the NEW
per-step reasoning-trace parsing, the warm-up SC headroom-band self-check, the
four-band terminal verdict (partial / no-headroom-block / scorable / headline),
and the two acceptance gates (G1 HEADROOM-CONFIRMED, G2 SCORABLE). No live model
is loaded — every assertion runs deterministically so a reviewer/CI can re-derive
the verdict the live exp3471 run reports without a 26B GGUF.
"""

from __future__ import annotations

import json

from carnot.phase3.p01_headroom_corpus import (
    BAND_JUDGE_FLOOR,
    DEFAULT_BENCHMARK_ID,
    DEFAULT_N_TARGET,
    HEADLINE_FLOOR,
    SC_BAND_HI,
    SC_BAND_LO,
    SCORABLE_FLOOR,
    HeadroomSample,
    MathProblem,
    build_headroom_row,
    build_math_problems,
    corpus_problem_ids,
    derive_headroom_verdict,
    extract_boxed,
    extract_math_answer,
    headroom_acceptance_gates,
    headroom_reproducibility_checksum,
    headroom_warmup_check,
    make_headroom_sample,
    majority_vote_str,
    math_is_correct,
    normalize_math_answer,
    parse_reasoning_steps,
    sc_in_headroom_band,
)


# ---------------------------------------------------------------------------
# extract_boxed / extract_math_answer
# ---------------------------------------------------------------------------


def test_extract_boxed_handles_nested_braces() -> None:
    """REQ-KONA-3471: the balanced-brace walk captures a nested \\frac whole."""
    assert extract_boxed("The answer is \\boxed{\\frac{1}{2}} done.") == "\\frac{1}{2}"


def test_extract_boxed_takes_the_last_box() -> None:
    """REQ-KONA-3471: intermediate boxes are ignored; the final box is the answer."""
    assert extract_boxed("first \\boxed{3}, then \\boxed{7}") == "7"


def test_extract_boxed_allows_space_before_brace() -> None:
    """REQ-KONA-3471: '\\boxed {x}' (space) is still parsed."""
    assert extract_boxed("\\boxed {42}") == "42"


def test_extract_boxed_none_when_absent_or_no_brace() -> None:
    """REQ-KONA-3471: no box, empty text, or a brace-less '\\boxed 5' -> None."""
    assert extract_boxed("no box here") is None
    assert extract_boxed("") is None
    assert extract_boxed("\\boxed 5") is None


def test_extract_boxed_unbalanced_returns_remainder() -> None:
    """REQ-KONA-3471: a truncated '\\boxed{12' returns what follows the brace."""
    assert extract_boxed("answer \\boxed{12") == "12"


def test_extract_math_answer_prefers_box_then_coda_then_number() -> None:
    """REQ-KONA-3471: extraction preference order box > #### coda > last number."""
    assert extract_math_answer("blah \\boxed{9}") == "9"
    assert extract_math_answer("reasoning\n#### 15") == "15"
    assert extract_math_answer("the result is 7 and then 8") == "8"
    assert extract_math_answer("no numbers at all") is None
    assert extract_math_answer("") is None


def test_extract_math_answer_empty_box_falls_through() -> None:
    """REQ-KONA-3471: an empty '\\boxed{}' falls through to the next coda."""
    assert extract_math_answer("\\boxed{}\n#### 4") == "4"


# ---------------------------------------------------------------------------
# normalize_math_answer / math_is_correct
# ---------------------------------------------------------------------------


def test_normalize_folds_fraction_macros_and_strips_latex() -> None:
    """REQ-KONA-3471: \\dfrac/\\tfrac fold to \\frac and cosmetic LaTeX is stripped."""
    assert normalize_math_answer("\\dfrac{1}{2}") == normalize_math_answer("\\frac{1}{2}")
    assert normalize_math_answer("$ \\left( 3 \\right) $") == "(3)"
    assert normalize_math_answer("\\text{cm}") == "cm"


def test_normalize_strips_commas_braces_period_plus_and_lowercases() -> None:
    """REQ-KONA-3471: thousands commas, wrapping braces, trailing '.', leading '+'."""
    assert normalize_math_answer("1,000") == "1000"
    assert normalize_math_answer("{42}") == "42"
    assert normalize_math_answer("7.") == "7"
    assert normalize_math_answer("+3") == "3"
    assert normalize_math_answer("ABC") == "abc"


def test_normalize_none_is_none() -> None:
    """REQ-KONA-3471: a None answer normalises to None (never compares equal)."""
    assert normalize_math_answer(None) is None


def test_math_is_correct_matches_equivalent_surface_forms() -> None:
    """REQ-KONA-3471: equivalent LaTeX surface forms compare correct."""
    assert math_is_correct("\\dfrac{1}{2}", "\\frac{1}{2}") is True
    assert math_is_correct("1,000", "1000") is True
    assert math_is_correct("3", "4") is False


def test_math_is_correct_none_and_empty_are_never_correct() -> None:
    """REQ-KONA-3471: None prediction/gold, or an answer that normalises empty."""
    assert math_is_correct(None, "5") is False
    assert math_is_correct("5", None) is False
    # "$" normalises to the empty string -> not correct even against itself.
    assert math_is_correct("$", "$") is False


# ---------------------------------------------------------------------------
# majority_vote_str
# ---------------------------------------------------------------------------


def test_majority_vote_buckets_by_normalised_form_returns_raw() -> None:
    """REQ-KONA-3471: votes bucket on normalised form; the raw winner is returned."""
    # "1,000" and "1000" share a normalised bucket (commas stripped) -> 2 vs 1.
    answers = ["1,000", "1000", "7"]
    winner = majority_vote_str(answers)
    # The winning bucket has 2 votes; the FIRST raw form ("1,000") is returned.
    assert winner == "1,000"
    assert normalize_math_answer(winner) == "1000"


def test_majority_vote_tie_breaks_to_earliest() -> None:
    """REQ-KONA-3471: a tie breaks toward the bucket seen earliest (deterministic)."""
    assert majority_vote_str(["3", "5"]) == "3"


def test_majority_vote_all_none_returns_none() -> None:
    """REQ-KONA-3471: when no sample produced an answer the vote is None."""
    assert majority_vote_str([None, None]) is None


# ---------------------------------------------------------------------------
# parse_reasoning_steps (the NEW capability)
# ---------------------------------------------------------------------------


def test_parse_steps_splits_on_newlines() -> None:
    """REQ-KONA-3471: newline-delimited CoT yields one step per non-blank line."""
    text = "Step one.\nStep two.\n\nStep three."
    assert parse_reasoning_steps(text) == ["Step one.", "Step two.", "Step three."]


def test_parse_steps_single_line_falls_back_to_sentences() -> None:
    """REQ-KONA-3471: a single-line CoT segments on sentence terminators."""
    steps = parse_reasoning_steps("First we add. Then we multiply. Done!")
    assert steps == ["First we add.", "Then we multiply.", "Done!"]


def test_parse_steps_empty_and_cap() -> None:
    """REQ-KONA-3471: empty text -> []; the step list is capped at max_steps."""
    assert parse_reasoning_steps("") == []
    assert parse_reasoning_steps("   ") == []
    many = "\n".join(str(i) for i in range(100))
    assert len(parse_reasoning_steps(many, max_steps=10)) == 10


# ---------------------------------------------------------------------------
# HeadroomSample / make_headroom_sample / build_headroom_row
# ---------------------------------------------------------------------------


def test_make_headroom_sample_populates_answer_steps_logprobs() -> None:
    """REQ-KONA-3471: a sample carries answer, steps, and a finite mean logprob."""
    sample = make_headroom_sample(
        "Add 2 and 2.\n\\boxed{4}", [-0.1, -0.2, float("inf"), None]
    )
    assert sample.answer == "4"
    assert sample.steps == ["Add 2 and 2.", "\\boxed{4}"]
    # The inf and None are dropped; mean over the two finite values.
    assert sample.n_tokens == 2
    assert abs(sample.mean_token_logprob - (-0.15)) < 1e-9


def test_make_headroom_sample_no_logprobs_is_honest_null() -> None:
    """REQ-KONA-3471: an empty logprob list -> None mean, zero tokens."""
    sample = make_headroom_sample("text \\boxed{1}", None)
    assert sample.mean_token_logprob is None
    assert sample.n_tokens == 0
    assert sample.token_logprobs == []


def test_headroom_sample_to_dict_round_trips() -> None:
    """REQ-KONA-3471: to_dict carries steps + n_steps + the scoring fields."""
    sample = HeadroomSample(
        text="t",
        answer="4",
        steps=["a", "b"],
        token_logprobs=[-0.1],
        mean_token_logprob=-0.1,
        n_tokens=1,
    )
    d = sample.to_dict()
    assert d["steps"] == ["a", "b"]
    assert d["n_steps"] == 2
    assert d["answer"] == "4"
    assert d["mean_token_logprob"] == -0.1


def test_build_headroom_row_shape() -> None:
    """REQ-KONA-3471: a corpus row carries identity, gold, level, greedy + k samples."""
    greedy = make_headroom_sample("\\boxed{4}", [-0.1])
    samples = [make_headroom_sample("\\boxed{4}", [-0.2]) for _ in range(6)]
    row = build_headroom_row(
        problem_id="math-abc",
        question="2+2?",
        gold="4",
        level="Level 5",
        greedy=greedy,
        samples=samples,
        temperature=0.8,
    )
    assert row["k"] == 6
    assert row["level"] == "Level 5"
    assert row["gold"] == "4"
    assert len(row["samples"]) == 6
    assert row["greedy"]["answer"] == "4"


# ---------------------------------------------------------------------------
# sc_in_headroom_band / headroom_warmup_check
# ---------------------------------------------------------------------------


def test_sc_in_headroom_band_boundaries() -> None:
    """REQ-KONA-3471: the band is the closed interval [SC_BAND_LO, SC_BAND_HI]."""
    assert sc_in_headroom_band(SC_BAND_LO) is True
    assert sc_in_headroom_band(SC_BAND_HI) is True
    assert sc_in_headroom_band(0.55) is True
    assert sc_in_headroom_band(0.91) is False  # GSM8K ceiling -> no headroom
    assert sc_in_headroom_band(0.2) is False
    assert sc_in_headroom_band(0.9, lo=0.8, hi=0.95) is True  # custom band


def _row(gold: str, greedy: str | None, samples: list[str | None]) -> dict:
    return {
        "problem_id": gold + "|" + str(greedy),
        "gold": gold,
        "greedy": {"answer": greedy},
        "samples": [{"answer": a} for a in samples],
    }


def test_headroom_warmup_in_band_with_enough_problems() -> None:
    """SCENARIO-KONA-3471: SC ~0.5 over enough problems is in band."""
    rows = []
    # 10 problems: 5 where the majority vote is correct, 5 where it is wrong.
    for i in range(5):
        rows.append(_row("4", "4", ["4", "4", "9"]))  # SC correct
    for i in range(5):
        rows.append(_row("4", "9", ["9", "9", "4"]))  # SC wrong (votes 9)
    warm = headroom_warmup_check(rows)
    assert warm.n_problems == 10
    assert warm.self_consistency_accuracy == 0.5
    assert warm.in_band is True
    assert len(warm.examples) == 3


def test_headroom_warmup_ceiling_is_not_in_band() -> None:
    """SCENARIO-KONA-3471-NO-HEADROOM: SC at ceiling (1.0) is NOT in band."""
    rows = [_row("4", "4", ["4", "4", "4"]) for _ in range(10)]
    warm = headroom_warmup_check(rows)
    assert warm.self_consistency_accuracy == 1.0
    assert warm.in_band is False


def test_headroom_warmup_too_few_to_judge_not_in_band() -> None:
    """REQ-KONA-3471: below the judge floor, in_band is False even if SC fits."""
    rows = [_row("4", "4", ["4", "9", "9"]) for _ in range(3)]
    warm = headroom_warmup_check(rows)
    assert warm.n_problems == 3
    assert warm.in_band is False


def test_headroom_warmup_empty_corpus() -> None:
    """REQ-KONA-3471: an empty corpus reports zeros and is not in band."""
    warm = headroom_warmup_check([])
    assert warm.n_problems == 0
    assert warm.self_consistency_accuracy == 0.0
    assert warm.greedy_accuracy == 0.0
    assert warm.in_band is False
    assert warm.examples == []


# ---------------------------------------------------------------------------
# derive_headroom_verdict
# ---------------------------------------------------------------------------


def test_verdict_too_few_to_judge_is_partial() -> None:
    """REQ-KONA-3471: below the judge floor -> partial, resume next milestone."""
    v = derive_headroom_verdict(5, 0.5, False)
    assert v == "complete: p01_headroom_corpus_partial_n=5_resume_next_milestone"


def test_verdict_no_headroom_block_when_out_of_band() -> None:
    """SCENARIO-KONA-3471-NO-HEADROOM: enough problems but SC out of band -> block."""
    v = derive_headroom_verdict(60, 0.92, False)
    assert v == "complete: blocked_no_headroom_benchmark_sc_outside_band"


def test_verdict_headline_eligible_in_band() -> None:
    """SCENARIO-KONA-3471: n>=80 in band -> headline-eligible with the SC stamped."""
    v = derive_headroom_verdict(85, 0.523, True)
    assert v == "complete: p01_headroom_corpus_headline_eligible_n=85_sc=0.523"


def test_verdict_scorable_partial_in_band() -> None:
    """SCENARIO-KONA-3471: 40<=n<80 in band -> scorable-partial."""
    v = derive_headroom_verdict(50, 0.5, True)
    assert v == (
        "complete: p01_headroom_corpus_scorable_partial_n=50_resume_next_milestone"
    )


def test_verdict_in_band_but_below_scorable_floor_is_partial() -> None:
    """REQ-KONA-3471: in band but below the scorable floor -> partial."""
    v = derive_headroom_verdict(20, 0.5, True)
    assert v == "complete: p01_headroom_corpus_partial_n=20_resume_next_milestone"


def test_verdict_is_always_complete_prefixed() -> None:
    """REQ-KONA-3471: every band is `complete:`-prefixed (Verdict Terminal-Prefix)."""
    for n, sc, band in [(5, 0.5, False), (60, 0.9, False), (85, 0.5, True), (50, 0.5, True)]:
        assert derive_headroom_verdict(n, sc, band).startswith("complete:")


# ---------------------------------------------------------------------------
# headroom_acceptance_gates
# ---------------------------------------------------------------------------


def test_gates_g1_is_headroom_g2_needs_count_and_steps() -> None:
    """REQ-KONA-3471: G1 == in_band; G2 needs the scorable floor AND step traces."""
    gates = headroom_acceptance_gates(True, 50, True)
    assert gates == {"g1_headroom_confirmed": True, "g2_scorable": True}
    # Out of band -> G1 false.
    assert headroom_acceptance_gates(False, 50, True)["g1_headroom_confirmed"] is False
    # Below the scorable floor -> G2 false.
    assert headroom_acceptance_gates(True, 30, True)["g2_scorable"] is False
    # No step traces -> G2 false even with enough problems.
    assert headroom_acceptance_gates(True, 50, False)["g2_scorable"] is False


def test_gates_return_exactly_two_named_booleans() -> None:
    """REQ-KONA-3471: the gate dict carries exactly the two named gates."""
    gates = headroom_acceptance_gates(True, 80, True)
    assert set(gates) == {"g1_headroom_confirmed", "g2_scorable"}
    assert all(isinstance(v, bool) for v in gates.values())


# ---------------------------------------------------------------------------
# build_math_problems / checksum / resume alias
# ---------------------------------------------------------------------------


def _math_record(problem: str, level: str, boxed: str) -> dict:
    return {
        "problem": problem,
        "level": level,
        "type": "Algebra",
        "solution": f"work here \\boxed{{{boxed}}}",
    }


def test_build_math_problems_filters_levels_and_extracts_gold() -> None:
    """REQ-KONA-3471: only requested levels with a parseable \\boxed gold survive."""
    records = [
        _math_record("p5a", "Level 5", "4"),
        _math_record("p4", "Level 4", "9"),  # filtered out (wrong level)
        _math_record("p5b", "Level 5", "\\frac{1}{2}"),
        {"problem": "no sol", "level": "Level 5", "solution": "no box"},  # dropped
        {"problem": None, "level": "Level 5", "solution": "\\boxed{1}"},  # dropped
    ]
    problems = build_math_problems(records, levels={"Level 5"}, n=10, seed=7)
    assert len(problems) == 2
    assert {p.answer for p in problems} == {"4", "\\frac{1}{2}"}
    assert all(p.level == "Level 5" for p in problems)
    assert all(p.problem_id.startswith("math-") for p in problems)


def test_build_math_problems_dedupes_and_is_deterministic() -> None:
    """REQ-KONA-3471: duplicate questions dedupe; same seed -> same order."""
    records = [
        _math_record("dup", "Level 5", "1"),
        _math_record("dup", "Level 5", "1"),  # duplicate question -> dropped
        _math_record("other", "Level 5", "2"),
    ]
    a = build_math_problems(records, levels={"Level 5"}, n=10, seed=3)
    b = build_math_problems(records, levels={"Level 5"}, n=10, seed=3)
    assert len(a) == 2
    assert [p.problem_id for p in a] == [p.problem_id for p in b]


def test_build_math_problems_respects_n_cap() -> None:
    """REQ-KONA-3471: the slice never exceeds n."""
    records = [_math_record(f"q{i}", "Level 5", str(i)) for i in range(20)]
    assert len(build_math_problems(records, levels={"Level 5"}, n=5, seed=1)) == 5


def test_math_problem_dataclass_fields() -> None:
    """REQ-KONA-3471: MathProblem carries id/question/answer/level."""
    p = MathProblem(problem_id="math-x", question="q", answer="4", level="Level 5")
    assert (p.problem_id, p.question, p.answer, p.level) == ("math-x", "q", "4", "Level 5")


def test_reproducibility_checksum_is_stable_and_sensitive() -> None:
    """REQ-KONA-3471: same inputs -> same checksum; a changed seed -> different."""
    base = dict(
        benchmark_id=DEFAULT_BENCHMARK_ID,
        model_path="/models/gemma.gguf",
        n_target=80,
        k_samples=6,
        levels={"Level 5"},
    )
    c1 = headroom_reproducibility_checksum(seed=1, **base)
    c2 = headroom_reproducibility_checksum(seed=1, **base)
    c3 = headroom_reproducibility_checksum(seed=2, **base)
    assert c1 == c2 != c3
    assert len(c1) == 16


def test_corpus_problem_ids_alias_reads_completed_rows(tmp_path) -> None:
    """SCENARIO-KONA-3471-RESUME: the resume alias finds complete rows on disk."""
    path = tmp_path / "hardmath.jsonl"
    greedy = make_headroom_sample("\\boxed{4}", [-0.1])
    samples = [make_headroom_sample("\\boxed{4}", [-0.2]) for _ in range(6)]
    complete = build_headroom_row(
        problem_id="math-done",
        question="q",
        gold="4",
        level="Level 5",
        greedy=greedy,
        samples=samples,
        temperature=0.8,
    )
    partial = dict(complete)
    partial["problem_id"] = "math-partial"
    partial["samples"] = partial["samples"][:2]  # too few samples -> not complete
    path.write_text(json.dumps(complete) + "\n" + json.dumps(partial) + "\n")
    ids = corpus_problem_ids(path, k_samples=6)
    assert ids == {"math-done"}


def test_module_constants() -> None:
    """REQ-KONA-3471: the documented defaults match the spec bands."""
    assert (SC_BAND_LO, SC_BAND_HI) == (0.40, 0.70)
    assert DEFAULT_N_TARGET == HEADLINE_FLOOR == 80
    assert SCORABLE_FLOOR == 40
    assert BAND_JUDGE_FLOOR == 8
