"""Tests for carnot.phase3.kona_trained_energy_hybrid.

Traces to REQ-KONA-3464 / SCENARIO-KONA-3464. Covers board_to_text,
score_boards, train_reranker_on_corpus, run_trained_energy_hybrid_arms,
paired_significance, reproducibility_checksum_3464, and derive_verdict.
All tests run fast: the JAX optimizer is stubbed, the corpus is synthetic,
and no live model is loaded.
"""

from __future__ import annotations

import json
import os
import tempfile

import jax
import pytest

from carnot.phase3.kona_trained_energy_hybrid import (
    _N_CANDIDATES,
    _FAST_N_STEPS,
    board_to_text,
    derive_verdict,
    paired_significance,
    reproducibility_checksum_3464,
    run_trained_energy_hybrid_arms,
    score_boards,
    train_reranker_on_corpus,
)
from carnot.phase3.p01_trained_energy_reranker import TrainedEnergyReranker, _Verifiers
from carnot.phase3.sudoku_global_opt import (
    OptimizeResult,
    SudokuPuzzle,
    generate_full_grid,
    dig_holes,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def clear_jax_cache():
    """Clear JAX compilation caches after each test to cap RSS growth."""
    yield
    jax.clear_caches()


@pytest.fixture
def minimal_puzzle() -> SudokuPuzzle:
    """One easy puzzle for fast tests (7-seed, 50 clues)."""
    full = generate_full_grid(7)
    clues = dig_holes(full, n_clues=50, seed=7)
    return SudokuPuzzle(
        puzzle_id="easy_0",
        difficulty="easy",
        clues=clues,
        solution=full,
        n_clues=50,
    )


@pytest.fixture
def fitted_reranker() -> TrainedEnergyReranker:
    """A minimal reranker fitted on two synthetic candidates (REQ-KONA-3464)."""
    r = TrainedEnergyReranker(n_iter=5)
    X = [[0.1, 0.0, 0.0, 0.0, -1.0, 0.5], [0.5, 0.2, 0.1, 0.1, -2.0, 0.3]]
    y = [1, 0]
    r.fit(X, y)
    return r


@pytest.fixture
def gsm8k_corpus_path(tmp_path) -> str:
    """A tiny synthetic GSM8K JSONL file with two problems, six samples each."""
    records = [
        {
            "problem_id": f"gsm8k-{i}",
            "gold": 42 + i,
            "greedy": {"text": f"So the answer is {42 + i}.", "answer": 42 + i},
            "samples": [
                {
                    "text": (
                        f"Step 1: compute 6 * 7 = {42 + i}. "
                        f"Therefore the answer is {42 + i}."
                    ),
                    "answer": 42 + i,
                    "mean_token_logprob": -0.3,
                },
                {
                    "text": "Step 1: compute 6 + 7 = 13. The answer is 13.",
                    "answer": 13,
                    "mean_token_logprob": -0.9,
                },
            ],
        }
        for i in range(3)
    ]
    p = tmp_path / "corpus.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in records))
    return str(p)


# ---------------------------------------------------------------------------
# board_to_text (REQ-KONA-3464)
# ---------------------------------------------------------------------------


def test_board_to_text_returns_string():
    """board_to_text converts a 9x9 grid to a non-empty string."""
    board = [[1] * 9 for _ in range(9)]
    result = board_to_text(board)
    assert isinstance(result, str)
    assert len(result) > 0


def test_board_to_text_contains_all_values():
    """Every cell value appears somewhere in the text output."""
    full = generate_full_grid(42)
    text = board_to_text(full)
    for row in full:
        for v in row:
            assert str(v) in text


def test_board_to_text_distinct_boards_produce_distinct_text():
    """Different boards produce different text strings."""
    b1 = generate_full_grid(1)
    b2 = generate_full_grid(2)
    assert board_to_text(b1) != board_to_text(b2)


# ---------------------------------------------------------------------------
# score_boards (REQ-KONA-3464)
# ---------------------------------------------------------------------------


def test_score_boards_returns_probabilities(fitted_reranker):
    """score_boards returns one float in [0, 1] per candidate board."""
    boards = [[[i + 1] * 9 for _ in range(9)] for i in range(3)]
    scores = score_boards(boards, fitted_reranker, _Verifiers())
    assert len(scores) == 3
    for s in scores:
        assert isinstance(s, float)
        assert 0.0 <= s <= 1.0


def test_score_boards_empty_input(fitted_reranker):
    """score_boards returns an empty list for zero boards."""
    scores = score_boards([], fitted_reranker, _Verifiers())
    assert scores == []


def test_score_boards_near_constant_on_sudoku(fitted_reranker):
    """Sudoku board strings produce near-identical scores (domain mismatch).

    All six verifier features are near-zero on board strings (no arithmetic
    equations, no reasoning steps), so the reranker's bias dominates and the
    spread across candidates is tiny.
    """
    boards = [generate_full_grid(seed) for seed in range(5)]
    scores = score_boards(boards, fitted_reranker, _Verifiers())
    spread = max(scores) - min(scores)
    # The spread should be small; 0.1 is a generous bound for any trained bias.
    assert spread < 0.1, f"Unexpectedly large score spread: {spread}"


# ---------------------------------------------------------------------------
# train_reranker_on_corpus (REQ-KONA-3464)
# ---------------------------------------------------------------------------


def test_train_reranker_on_corpus_returns_fitted_reranker(gsm8k_corpus_path):
    """train_reranker_on_corpus returns a fitted TrainedEnergyReranker."""
    reranker, n = train_reranker_on_corpus(gsm8k_corpus_path)
    assert isinstance(reranker, TrainedEnergyReranker)
    assert reranker._fitted
    assert n > 0


def test_train_reranker_on_corpus_candidate_count(gsm8k_corpus_path):
    """The returned candidate count matches total samples in the corpus."""
    _, n = train_reranker_on_corpus(gsm8k_corpus_path)
    # 3 problems × 2 samples each = 6 training candidates.
    assert n == 6


def test_train_reranker_on_empty_corpus(tmp_path):
    """An empty corpus produces an unfitted reranker without crashing."""
    p = tmp_path / "empty.jsonl"
    p.write_text("")
    reranker, n = train_reranker_on_corpus(str(p))
    assert isinstance(reranker, TrainedEnergyReranker)
    assert n == 0
    # Predict proba on an unfitted reranker raises RuntimeError.
    with pytest.raises(RuntimeError, match="not fitted"):
        reranker.predict_proba([[0.0] * 6])


# ---------------------------------------------------------------------------
# run_trained_energy_hybrid_arms (REQ-KONA-3464)
# ---------------------------------------------------------------------------


def test_run_trained_energy_hybrid_arms_schema(monkeypatch, minimal_puzzle, fitted_reranker):
    """run_trained_energy_hybrid_arms returns the required keys."""
    import carnot.phase3.kona_trained_energy_hybrid as kmod

    def _stub_opt(clues, *, seed, variant, n_steps, n_restarts):
        # Returns a non-valid board; the CP hybrid will still solve via CP.
        return OptimizeResult(
            board=[[1] * 9 for _ in range(9)],
            final_energy=10.0,
            solved=False,
            n_violated=20,
        )

    monkeypatch.setattr(kmod, "optimize_board", _stub_opt)

    result = run_trained_energy_hybrid_arms(
        [minimal_puzzle], fitted_reranker, seed=99
    )
    for key in (
        "trained_hybrid_solve_rate",
        "pure_trained_energy_descent_solve_rate",
        "per_puzzle_trained_hybrid",
        "per_puzzle_pure_trained_descent",
    ):
        assert key in result, f"Missing key: {key}"


def test_run_trained_energy_hybrid_arms_trained_hybrid_uses_cp(
    monkeypatch, minimal_puzzle, fitted_reranker
):
    """The trained_hybrid arm solves via the CP solver even when the optimizer fails.

    hybrid_solve ignores the energy proposal (``_ = energy_board``) and runs
    CP from the original clues.  The CP solver should find a valid completion
    for any well-formed puzzle, so trained_hybrid_solve_rate should be 1.0.
    """
    import carnot.phase3.kona_trained_energy_hybrid as kmod

    def _stub_opt(clues, *, seed, variant, n_steps, n_restarts):
        return OptimizeResult(
            board=[[1] * 9 for _ in range(9)],
            final_energy=999.0,
            solved=False,
            n_violated=27,
        )

    monkeypatch.setattr(kmod, "optimize_board", _stub_opt)

    result = run_trained_energy_hybrid_arms(
        [minimal_puzzle], fitted_reranker, seed=99
    )
    assert result["trained_hybrid_solve_rate"] == 1.0, (
        "Trained hybrid should always match untrained hybrid: the CP solver "
        "ignores the energy proposal and solves from the clues."
    )


def test_run_trained_energy_hybrid_arms_pure_descent_fails_bad_candidates(
    monkeypatch, minimal_puzzle, fitted_reranker
):
    """pure_trained_descent returns 0% when all candidates are invalid boards."""
    import carnot.phase3.kona_trained_energy_hybrid as kmod

    def _stub_opt(clues, *, seed, variant, n_steps, n_restarts):
        return OptimizeResult(
            board=[[1] * 9 for _ in range(9)],  # all-1 board is invalid
            final_energy=10.0,
            solved=False,
            n_violated=20,
        )

    monkeypatch.setattr(kmod, "optimize_board", _stub_opt)

    result = run_trained_energy_hybrid_arms(
        [minimal_puzzle], fitted_reranker, seed=99
    )
    assert result["pure_trained_energy_descent_solve_rate"] == 0.0


# ---------------------------------------------------------------------------
# paired_significance (REQ-KONA-3464)
# ---------------------------------------------------------------------------


def test_paired_significance_all_solved_gives_p1():
    """When both methods solve every puzzle, McNemar p = 1.0 (zero discordant pairs)."""
    untrained = [{"puzzle_id": f"p{i}", "solved": True} for i in range(5)]
    trained = [{"puzzle_id": f"p{i}", "solved": True} for i in range(5)]
    sig = paired_significance(untrained, trained)
    assert sig["mcnemar_exact_p"] == 1.0
    assert sig["discordant_trained_wins"] == 0
    assert sig["discordant_untrained_wins"] == 0


def test_paired_significance_two_disagreements():
    """With 2 discordant pairs (trained wins both), McNemar p = 0.5 < 1.0.

    McNemar exact: n=2, k=min(2,0)=0, cum=C(2,0)*0.25=0.25, p=2*0.25=0.5.
    """
    untrained = [{"solved": False}, {"solved": False}] + [{"solved": True}] * 4
    trained = [{"solved": True}] * 6
    sig = paired_significance(untrained, trained)
    assert sig["discordant_trained_wins"] == 2
    assert sig["discordant_untrained_wins"] == 0
    assert sig["mcnemar_exact_p"] == pytest.approx(0.5)


def test_paired_significance_schema():
    """paired_significance returns the required schema keys."""
    records = [{"solved": True}] * 3
    sig = paired_significance(records, records)
    for key in (
        "comparison",
        "discordant_trained_wins",
        "discordant_untrained_wins",
        "mcnemar_exact_p",
        "interpretation",
    ):
        assert key in sig, f"Missing key: {key}"
    assert sig["comparison"] == "trained_hybrid_vs_untrained_hybrid"


# ---------------------------------------------------------------------------
# reproducibility_checksum_3464 (REQ-KONA-3464)
# ---------------------------------------------------------------------------


def test_reproducibility_checksum_is_stable_and_seed_sensitive():
    """The checksum is deterministic for the same config and changes with seed."""
    a = reproducibility_checksum_3464(42, "data/corpus.jsonl", 3, 50)
    b = reproducibility_checksum_3464(42, "data/corpus.jsonl", 3, 50)
    assert a == b
    c = reproducibility_checksum_3464(99, "data/corpus.jsonl", 3, 50)
    assert a != c


def test_reproducibility_checksum_length():
    """The returned checksum is a 16-character hex string."""
    cs = reproducibility_checksum_3464(1, "x", 1, 1)
    assert isinstance(cs, str)
    assert len(cs) == 16
    assert all(c in "0123456789abcdef" for c in cs)


# ---------------------------------------------------------------------------
# derive_verdict (REQ-KONA-3464)
# ---------------------------------------------------------------------------


def test_derive_verdict_no_lift():
    """When trained <= untrained, verdict is 'no_lift'."""
    v = derive_verdict(
        trained_hybrid_solve_rate=1.0,
        untrained_hybrid_solve_rate=1.0,
        pure_trained_energy_descent_solve_rate=0.0,
        mcnemar_p=1.0,
    )
    assert v == "complete: trained_energy_no_lift_over_untrained_kona_hybrid"


def test_derive_verdict_strengthens():
    """When trained > untrained AND significance passes, verdict is 'strengthens'."""
    v = derive_verdict(
        trained_hybrid_solve_rate=0.9,
        untrained_hybrid_solve_rate=0.7,
        pure_trained_energy_descent_solve_rate=0.0,
        mcnemar_p=0.01,
    )
    assert v == "complete: trained_energy_strengthens_kona_hybrid_solve_rate"


def test_derive_verdict_pure_descent_solves():
    """When pure trained descent itself solves >50%, verdict is 'pure_descent_solves'."""
    v = derive_verdict(
        trained_hybrid_solve_rate=0.9,
        untrained_hybrid_solve_rate=0.7,
        pure_trained_energy_descent_solve_rate=0.6,
        mcnemar_p=0.01,
    )
    assert v == "complete: trained_energy_pure_descent_solves_hybrid_no_longer_required"


def test_derive_verdict_starts_with_complete_prefix():
    """All verdict branches start with 'complete:' (terminal prefix discipline)."""
    cases = [
        (1.0, 1.0, 0.0, 1.0),
        (0.9, 0.7, 0.0, 0.01),
        (0.9, 0.7, 0.6, 0.01),
        (0.5, 0.7, 0.0, 1.0),
    ]
    for args in cases:
        v = derive_verdict(*args)
        assert v.startswith("complete:"), f"Verdict does not start with complete:: {v!r}"
