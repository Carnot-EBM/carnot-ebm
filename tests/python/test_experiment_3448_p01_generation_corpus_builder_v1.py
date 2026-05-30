"""Tests for the P0.1 resumable generation-corpus builder helpers (exp3448).

Spec: REQ-KONA-3448, SCENARIO-KONA-3448, SCENARIO-KONA-3448-RESUME,
SCENARIO-KONA-3448-BLOCKED.

These tests pin the GPU-free scientific decisions the builder makes: the resume
contract (which problems are already done), the per-problem corpus-row shape the
scoring task consumes, the warm-up self-consistency self-check that guards the
exp3426 all-null-extraction bug, the terminal-verdict bands (a partial corpus is
a success), and the reproducibility checksum. No live model is loaded — every
assertion runs on synthetic rows so a reviewer/CI can re-derive the builder's
behaviour without a 26B GGUF.
"""

from __future__ import annotations

import json
from pathlib import Path

from carnot.phase3.p01_generation_corpus import (
    GenerationSample,
    build_corpus_row,
    completed_problem_ids,
    corpus_reproducibility_checksum,
    derive_corpus_verdict,
    make_sample,
    mean_logprob,
    read_corpus_rows,
    row_is_complete,
    warmup_self_consistency_check,
)


# ---------------------------------------------------------------------------
# mean_logprob / make_sample (SCENARIO-KONA-3448: per-generation logprobs)
# ---------------------------------------------------------------------------


def test_mean_logprob_averages_finite_values() -> None:
    """REQ-KONA-3448: mean-token logprob averages the finite chosen-token logprobs."""
    assert mean_logprob([-1.0, -2.0, -3.0]) == -2.0


def test_mean_logprob_none_on_empty_or_all_nonfinite() -> None:
    """REQ-KONA-3448: no scorable tokens -> None (not 0.0), the honest signal."""
    assert mean_logprob([]) is None
    assert mean_logprob(None) is None
    assert mean_logprob([float("nan"), float("inf")]) is None


def test_make_sample_extracts_answer_and_confidence() -> None:
    """REQ-KONA-3448: a sample carries text, extracted answer, logprobs + mean."""
    sample = make_sample("Reasoning here.\n#### 42", [-0.5, -1.5])
    assert sample.answer == 42
    assert sample.token_logprobs == [-0.5, -1.5]
    assert sample.mean_token_logprob == -1.0
    assert sample.n_tokens == 2


def test_make_sample_handles_no_logprobs_and_no_answer() -> None:
    """REQ-KONA-3448: empty generation -> None answer, empty logprobs, n_tokens 0."""
    sample = make_sample("", None)
    assert sample.answer is None
    assert sample.token_logprobs == []
    assert sample.mean_token_logprob is None
    assert sample.n_tokens == 0


def test_make_sample_drops_nonfinite_logprobs() -> None:
    """REQ-KONA-3448: non-finite logprobs are filtered before storage/mean."""
    sample = make_sample("#### 7", [-1.0, float("inf"), -3.0])
    assert sample.token_logprobs == [-1.0, -3.0]
    assert sample.mean_token_logprob == -2.0
    assert sample.n_tokens == 2


# ---------------------------------------------------------------------------
# build_corpus_row / row_is_complete (corpus row contract)
# ---------------------------------------------------------------------------


def _sample(answer_text: str, lps: list[float]) -> GenerationSample:
    return make_sample(answer_text, lps)


def test_build_corpus_row_shape() -> None:
    """REQ-KONA-3448: a row packs id, question, gold, greedy, and k samples."""
    greedy = _sample("#### 10", [-0.1])
    samples = [_sample("#### 10", [-0.2]), _sample("#### 11", [-0.3])]
    row = build_corpus_row(
        problem_id="gsm8k-1",
        question="What is 5+5?",
        gold=10,
        greedy=greedy,
        samples=samples,
        temperature=0.8,
    )
    assert row["problem_id"] == "gsm8k-1"
    assert row["gold"] == 10
    assert row["k"] == 2
    assert row["temperature"] == 0.8
    assert row["greedy"]["answer"] == 10
    assert [s["answer"] for s in row["samples"]] == [10, 11]
    # The row must be JSON-serialisable (it is appended to a JSONL file).
    json.dumps(row)


def test_row_is_complete_true_for_full_row() -> None:
    """REQ-KONA-3448: a row with greedy + >=k samples is complete."""
    row = build_corpus_row(
        problem_id="p1",
        question="q",
        gold=3,
        greedy=_sample("#### 3", [-0.1]),
        samples=[_sample("#### 3", [-0.1]) for _ in range(6)],
        temperature=0.8,
    )
    assert row_is_complete(row, k_samples=6) is True


def test_row_is_complete_false_for_short_or_malformed_rows() -> None:
    """REQ-KONA-3448: truncated rows are NOT complete -> regenerated on resume."""
    short = build_corpus_row(
        problem_id="p2",
        question="q",
        gold=3,
        greedy=_sample("#### 3", [-0.1]),
        samples=[_sample("#### 3", [-0.1]) for _ in range(2)],
        temperature=0.8,
    )
    assert row_is_complete(short, k_samples=6) is False
    assert row_is_complete({"problem_id": "p3"}, k_samples=6) is False
    assert row_is_complete({"greedy": {}, "samples": []}, k_samples=1) is False
    # greedy present but not a dict (e.g. a null from an interrupted write).
    assert (
        row_is_complete(
            {"problem_id": "p4", "gold": 1, "greedy": None, "samples": [{}]}, k_samples=1
        )
        is False
    )


# ---------------------------------------------------------------------------
# read_corpus_rows / completed_problem_ids (SCENARIO-KONA-3448-RESUME)
# ---------------------------------------------------------------------------


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def test_read_corpus_rows_missing_file_returns_empty(tmp_path: Path) -> None:
    """SCENARIO-KONA-3448-RESUME: a missing corpus is a fresh start (no rows)."""
    assert read_corpus_rows(tmp_path / "nope.jsonl") == []


def test_read_corpus_rows_skips_malformed_trailing_line(tmp_path: Path) -> None:
    """SCENARIO-KONA-3448-RESUME: an interrupted half-written line is skipped."""
    path = tmp_path / "corpus.jsonl"
    good = build_corpus_row(
        problem_id="p1",
        question="q",
        gold=1,
        greedy=_sample("#### 1", [-0.1]),
        samples=[_sample("#### 1", [-0.1])],
        temperature=0.8,
    )
    # Blank lines (e.g. a stray newline between appends) are ignored too.
    path.write_text(
        json.dumps(good) + "\n\n" + '{"problem_id": "p2", "trunc', encoding="utf-8"
    )
    rows = read_corpus_rows(path)
    assert len(rows) == 1
    assert rows[0]["problem_id"] == "p1"


def test_completed_problem_ids_only_counts_complete_rows(tmp_path: Path) -> None:
    """SCENARIO-KONA-3448-RESUME: only COMPLETE rows count as done (skip set)."""
    path = tmp_path / "corpus.jsonl"
    complete = build_corpus_row(
        problem_id="done-1",
        question="q",
        gold=1,
        greedy=_sample("#### 1", [-0.1]),
        samples=[_sample("#### 1", [-0.1]) for _ in range(6)],
        temperature=0.8,
    )
    partial = build_corpus_row(
        problem_id="partial-2",
        question="q",
        gold=2,
        greedy=_sample("#### 2", [-0.1]),
        samples=[_sample("#### 2", [-0.1])],  # only 1 of 6
        temperature=0.8,
    )
    _write_jsonl(path, [complete, partial])
    done = completed_problem_ids(path, k_samples=6)
    assert done == {"done-1"}


def test_completed_problem_ids_empty_for_missing_file(tmp_path: Path) -> None:
    """SCENARIO-KONA-3448-RESUME: no file -> empty done-set -> generate everything."""
    assert completed_problem_ids(tmp_path / "absent.jsonl", k_samples=6) == set()


# ---------------------------------------------------------------------------
# warmup_self_consistency_check (the exp3426 0.0-bug guard)
# ---------------------------------------------------------------------------


def _row(pid: str, gold: int, greedy_ans: str, sample_answers: list[str]) -> dict:
    return build_corpus_row(
        problem_id=pid,
        question="q",
        gold=gold,
        greedy=_sample(greedy_ans, [-0.1]),
        samples=[_sample(a, [-0.1]) for a in sample_answers],
        temperature=0.8,
    )


def test_warmup_non_degenerate_when_sc_beats_floor_and_greedy() -> None:
    """REQ-KONA-3448: SC >= greedy AND > 0.30 -> non_degenerate True."""
    # 20 problems: greedy right ~50%, self-consistency right 100% (majority right).
    rows = []
    for i in range(20):
        gold = i + 1
        greedy_ans = f"#### {gold}" if i % 2 == 0 else "#### 9999"
        # 3 of 4 samples agree on the gold -> majority vote is correct.
        sample_answers = [f"#### {gold}", f"#### {gold}", f"#### {gold}", "#### 9999"]
        rows.append(_row(f"p{i}", gold, greedy_ans, sample_answers))
    check = warmup_self_consistency_check(rows, min_problems=20)
    assert check.n_problems == 20
    assert check.self_consistency_accuracy == 1.0
    assert check.greedy_accuracy == 0.5
    assert check.non_degenerate is True
    assert check.examples == []  # examples only kept when the gate FAILS


def test_warmup_degenerate_records_examples_when_extraction_broken() -> None:
    """REQ-KONA-3448: all-null sample answers (exp3426 bug) -> degenerate + examples."""
    # Greedy extracts fine; every SAMPLE produced no parseable answer (the bug).
    rows = []
    for i in range(20):
        gold = i + 1
        rows.append(_row(f"p{i}", gold, f"#### {gold}", ["", "", "", ""]))
    check = warmup_self_consistency_check(rows, min_problems=20)
    assert check.self_consistency_accuracy == 0.0
    assert check.greedy_accuracy == 1.0
    assert check.non_degenerate is False
    # Three diagnosable examples so the broken extraction is debuggable.
    assert len(check.examples) == 3
    assert check.examples[0]["sample_answers"] == [None, None, None, None]
    assert check.examples[0]["majority_vote"] is None


def test_warmup_insufficient_data_is_not_yet_degenerate() -> None:
    """REQ-KONA-3448: < min_problems -> non_degenerate False, no examples (keep going)."""
    rows = [_row("p0", 1, "#### 1", ["#### 1", "#### 1"])]
    check = warmup_self_consistency_check(rows, min_problems=20)
    assert check.n_problems == 1
    assert check.non_degenerate is False
    assert check.examples == []


# ---------------------------------------------------------------------------
# derive_corpus_verdict (terminal bands — partial is a success)
# ---------------------------------------------------------------------------


def test_verdict_complete_when_target_met() -> None:
    """REQ-KONA-3448: n_completed >= target -> complete band."""
    v = derive_corpus_verdict(120, 120)
    assert v.startswith("complete: p01_generation_corpus_complete_n=120")


def test_verdict_partial_resumable_band() -> None:
    """REQ-KONA-3448: 30 <= n < target -> partial-resumable band (still success)."""
    v = derive_corpus_verdict(45, 120)
    assert v.startswith("complete: p01_generation_corpus_partial_resumable_n=45")


def test_verdict_seeded_band_below_clt_minimum() -> None:
    """REQ-KONA-3448: n < 30 -> seeded/resume-next-milestone band."""
    v = derive_corpus_verdict(12, 120)
    assert v.startswith("complete: p01_generation_corpus_seeded_n=12_resume_next_milestone")


def test_all_verdict_bands_are_terminal_complete_prefixed() -> None:
    """Verdict Terminal-Prefix Discipline: every band starts with `complete:`."""
    for n in (0, 29, 30, 119, 120, 200):
        assert derive_corpus_verdict(n, 120).startswith("complete:")


# ---------------------------------------------------------------------------
# corpus_reproducibility_checksum (audit trail)
# ---------------------------------------------------------------------------


def test_checksum_is_deterministic_and_content_sensitive(tmp_path: Path) -> None:
    """REQ-KONA-3448: same inputs -> same hash; changed corpus/seed -> changed hash."""
    src = tmp_path / "src.jsonl"
    src.write_text('{"original_question": "q", "original_answer": 1}\n', encoding="utf-8")
    base = corpus_reproducibility_checksum(
        corpus_path=src, model_path="/m/gemma.gguf", seed=42, n_target=120, k_samples=6
    )
    same = corpus_reproducibility_checksum(
        corpus_path=src, model_path="/m/gemma.gguf", seed=42, n_target=120, k_samples=6
    )
    assert base == same
    assert len(base) == 16

    diff_seed = corpus_reproducibility_checksum(
        corpus_path=src, model_path="/m/gemma.gguf", seed=43, n_target=120, k_samples=6
    )
    assert diff_seed != base

    src.write_text('{"original_question": "q2", "original_answer": 2}\n', encoding="utf-8")
    diff_corpus = corpus_reproducibility_checksum(
        corpus_path=src, model_path="/m/gemma.gguf", seed=42, n_target=120, k_samples=6
    )
    assert diff_corpus != base
