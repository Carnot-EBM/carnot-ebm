"""Kona trained-energy hybrid: apply the GSM8K-trained reranker to Sudoku (exp 3464).

**Researcher summary:**
    exp3440 found that the Kona global-opt hybrid (untrained Sudoku energy
    proposes a board → constraint-propagation solver cleans up residuals) solves
    100% of the test puzzles. exp3460/3461 showed that a logistic-regression
    reranker TRAINED on GSM8K text-reasoning candidates reaches AUROC 0.629 —
    above the 0.55 threshold. The open question: does this trained energy lift the
    Kona hybrid beyond the untrained baseline?

    This module implements the comparison and documents the two reasons why the
    answer is NO LIFT:

    1. **Architectural ceiling**: ``sudoku_global_opt.hybrid_solve`` IGNORES the
       energy proposal (``_ = energy_board``) and hands the puzzle clues directly
       to the constraint-propagation solver. Any energy — trained or untrained —
       is irrelevant to the CP solver's outcome. The hybrid achieves 100% with or
       without a trained energy proposal.

    2. **Domain mismatch**: The reranker was trained on text-reasoning features
       (arithmetic violations, contradiction patterns, Curry-Howard type errors,
       logical inconsistencies). Sudoku board strings produce near-zero values on
       all six feature dimensions, so the reranker's P(correct) outputs collapse
       to its bias term — selection across candidates is effectively uniform and
       provides no optimisation signal.

**Engineers' guide:**
    ``board_to_text`` converts a 9×9 integer grid to a space-separated string.
    ``score_boards`` runs that string through the same six-feature extractor the
    trained reranker uses for GSM8K candidates, then calls ``predict_proba``.
    ``train_reranker_on_corpus`` fits the reranker on the full GSM8K corpus (no
    CV needed here — we are probing cross-domain transfer, not evaluating in-
    domain accuracy). ``run_trained_energy_hybrid_arms`` runs the two treatment
    arms (trained_hybrid + pure_trained_descent) on the puzzle set.

Spec: REQ-KONA-3464, SCENARIO-KONA-3464
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

from carnot.phase3.p01_energy_vote_scoring import mcnemar_exact
from carnot.phase3.p01_trained_energy_reranker import (
    TrainedEnergyReranker,
    _Verifiers,
    candidate_feature_vector,
)
from carnot.phase3.sudoku_global_opt import (
    SudokuPuzzle,
    board_is_valid_solution,
    hybrid_solve,
    optimize_board,
)

# Candidate generation budget for the treatment arm. We intentionally use a
# SMALL budget (not the exp3440 n_steps=3000) because:
#   a) The science question is about SELECTION quality, not optimizer depth.
#   b) A fast run demonstrates the domain-mismatch finding without the 5-minute
#      Langevin cost — the trained energy's scores collapse regardless of budget.
_N_CANDIDATES: int = 3
_FAST_N_STEPS: int = 50
_FAST_N_RESTARTS: int = 1


# ---------------------------------------------------------------------------
# Board → text encoding (the domain bridge that reveals the mismatch)
# ---------------------------------------------------------------------------


def board_to_text(board: list[list[int]]) -> str:
    """Serialise a 9×9 Sudoku board to a flat text string.

    The trained energy reranker expects chain-of-thought reasoning traces that
    contain arithmetic equations (``A + B = C``), logical steps, and type
    annotations. A Sudoku board has none of these, so every verifier score
    (arithmetic_energy, contradiction_energy, curry_howard_score,
    logical_inconsistency) will be near zero for any valid board string. The
    mean_logprob feature defaults to −10.0 (the ``None`` fallback in
    ``candidate_feature_vector``) and log_n_steps = log(1) = 0 because
    ``extract_steps`` finds no step delimiters. All features are therefore
    at or near their floor values for EVERY candidate board — the reranker
    cannot distinguish boards by their verifier scores.
    """
    return " ".join(" ".join(str(v) for v in row) for row in board)


# ---------------------------------------------------------------------------
# Candidate scoring
# ---------------------------------------------------------------------------


def score_boards(
    boards: list[list[list[int]]],
    reranker: TrainedEnergyReranker,
    verifiers: _Verifiers,
) -> list[float]:
    """Return the trained-energy P(correct) for each candidate board.

    Converts each board to a text string via ``board_to_text``, extracts the
    six verifier features, and calls the trained reranker's ``predict_proba``.
    Because all features are near-zero on board strings, the outputs converge
    to the reranker's bias term — the distribution of scores is essentially flat
    across all candidates. The returned list has the same length as ``boards``.
    Returns an empty list when ``boards`` is empty.
    """
    if not boards:
        return []
    feats = [
        candidate_feature_vector(board_to_text(b), mean_logprob=None, verifiers=verifiers)
        for b in boards
    ]
    return reranker.predict_proba(feats)


# ---------------------------------------------------------------------------
# Reranker training from the GSM8K corpus
# ---------------------------------------------------------------------------


def train_reranker_on_corpus(
    corpus_path: str,
    *,
    n_iter: int = 200,
) -> tuple[TrainedEnergyReranker, int]:
    """Fit a ``TrainedEnergyReranker`` on the full GSM8K corpus.

    Unlike exp3460 (which uses 5-fold CV for held-out accuracy evaluation),
    we train on the ENTIRE corpus here because we are testing CROSS-DOMAIN
    transfer to Sudoku — there is no risk of same-domain leakage. More training
    data gives the reranker the strongest possible GSM8K signal. If that strong
    signal still cannot guide Sudoku optimisation, the domain-mismatch finding
    is unambiguous.

    Parameters
    ----------
    corpus_path : str
        Path to the GSM8K generations JSONL file (one record per problem).
    n_iter : int
        Gradient-descent iterations for the logistic reranker.

    Returns
    -------
    (reranker, n_candidates_trained)
        The fitted reranker plus the total number of training candidates used.
    """
    records: list[dict] = []
    with open(corpus_path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    verifiers = _Verifiers()
    X: list[list[float]] = []
    y: list[int] = []
    for rec in records:
        gold = rec.get("gold")
        for s in rec.get("samples", []):
            text = s.get("text", "")
            mlp = s.get("mean_token_logprob")
            X.append(candidate_feature_vector(text, mlp, verifiers))
            y.append(1 if s.get("answer") == gold else 0)

    reranker = TrainedEnergyReranker(n_iter=n_iter)
    if X:
        reranker.fit(X, y)
    return reranker, len(X)


# ---------------------------------------------------------------------------
# Treatment arms
# ---------------------------------------------------------------------------


def _solve_rate(records: list[dict[str, Any]]) -> float:
    """Fraction of puzzle records with ``solved == True``."""
    if not records:
        return 0.0
    return float(sum(1 for r in records if r["solved"]) / len(records))


def run_trained_energy_hybrid_arms(
    puzzles: list[SudokuPuzzle],
    reranker: TrainedEnergyReranker,
    *,
    seed: int,
) -> dict[str, Any]:
    """Run trained_hybrid and pure_trained_descent arms on the puzzle set.

    For each puzzle:
      1. Generate ``_N_CANDIDATES`` candidate boards using a fast Langevin
         optimizer (n_steps=50, single restart — much cheaper than exp3440's
         3000 steps).
      2. Score each candidate with the trained energy reranker.
      3. Select the highest-P(correct) candidate as the "trained energy
         proposal."
      4. **trained_hybrid**: pass the proposal to ``hybrid_solve``. The CP
         solver IGNORES the proposal and solves from the original clues
         (``_ = energy_board`` in ``sudoku_global_opt.hybrid_solve``), so this
         arm achieves the same solve-rate as the untrained hybrid (100% on
         valid puzzles within the CP node budget).
      5. **pure_trained_descent**: accept the best-scored board without CP
         correction. At n_steps=50 the optimizer almost never reaches a valid
         board (exp3440 found 0% even at n_steps=3000 with restarts), so this
         arm is expected to report 0% solve-rate as well.

    Parameters
    ----------
    puzzles : list[SudokuPuzzle]
        The same puzzle set used in exp3440 (generated with seed=20260530).
    reranker : TrainedEnergyReranker
        Fitted reranker (trained on GSM8K corpus).
    seed : int
        Reproducibility seed for the fast optimizer; should differ from the
        puzzle-set seed to avoid tautology between experiment ID and seed.

    Returns
    -------
    dict with solve-rate scalars and per-puzzle records for both arms.
    """
    verifiers = _Verifiers()
    per_trained_hybrid: list[dict[str, Any]] = []
    per_pure_trained: list[dict[str, Any]] = []

    for p in puzzles:
        # Fast candidate generation — sufficient to exercise selection logic.
        candidates: list[list[list[int]]] = []
        for i in range(_N_CANDIDATES):
            res = optimize_board(
                p.clues,
                seed=seed + hash(p.puzzle_id) % 10_000 + i,
                variant="annealed",
                n_steps=_FAST_N_STEPS,
                n_restarts=_FAST_N_RESTARTS,
            )
            candidates.append(res.board)

        # Score candidates: all scores will be near the reranker's bias term
        # because Sudoku board text has near-zero verifier feature values.
        scores = score_boards(candidates, reranker, verifiers)
        best_idx = max(range(len(scores)), key=lambda idx: scores[idx])
        best_board = candidates[best_idx]

        # Trained hybrid: CP solver handles the clues; the energy proposal is
        # architecturally ignored inside hybrid_solve.
        _, hybrid_ok = hybrid_solve(p.clues, energy_board=best_board)
        per_trained_hybrid.append(
            {
                "puzzle_id": p.puzzle_id,
                "solved": hybrid_ok,
                "max_reranker_score": float(max(scores)),
                "score_range": float(max(scores) - min(scores)),
            }
        )

        # Pure trained descent: accept the reranker's best candidate directly.
        pure_ok = board_is_valid_solution(best_board, p.clues)
        per_pure_trained.append(
            {"puzzle_id": p.puzzle_id, "solved": pure_ok}
        )

    return {
        "trained_hybrid_solve_rate": _solve_rate(per_trained_hybrid),
        "pure_trained_energy_descent_solve_rate": _solve_rate(per_pure_trained),
        "per_puzzle_trained_hybrid": per_trained_hybrid,
        "per_puzzle_pure_trained_descent": per_pure_trained,
    }


# ---------------------------------------------------------------------------
# Paired significance (McNemar exact)
# ---------------------------------------------------------------------------


def paired_significance(
    untrained_hybrid_per_puzzle: list[dict[str, Any]],
    trained_hybrid_per_puzzle: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compute McNemar exact p for the trained-hybrid vs untrained-hybrid delta.

    Pairs are matched by index (both lists use the same puzzle ordering).  When
    both methods solve every puzzle (solve-rate = 1.0), there are zero discordant
    pairs and McNemar returns p = 1.0 — the null hypothesis of no difference
    cannot be rejected.

    Returns a dict with the discordant counts and the exact p-value, shaped to
    match the exp3460 ``paired_significance`` schema so downstream tooling can
    parse the artifact consistently.
    """
    a_correct = [bool(r["solved"]) for r in untrained_hybrid_per_puzzle]
    b_correct = [bool(r["solved"]) for r in trained_hybrid_per_puzzle]
    p = mcnemar_exact(a_correct, b_correct)
    b01 = sum(1 for a, b in zip(a_correct, b_correct) if (not a) and b)
    b10 = sum(1 for a, b in zip(a_correct, b_correct) if a and (not b))
    return {
        "comparison": "trained_hybrid_vs_untrained_hybrid",
        "discordant_trained_wins": b01,
        "discordant_untrained_wins": b10,
        "mcnemar_exact_p": p,
        "interpretation": (
            "p = 1.0 means zero discordant pairs — no statistical evidence "
            "that the trained energy changes the hybrid's outcome."
            if p == 1.0
            else f"p = {p:.4f} for the trained-hybrid vs untrained-hybrid delta."
        ),
    }


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


def reproducibility_checksum_3464(
    seed: int,
    corpus_path: str,
    n_candidates: int,
    fast_n_steps: int,
) -> str:
    """16-char content hash over the experimental configuration."""
    payload = {
        "experiment": 3464,
        "seed": seed,
        "corpus_path": str(Path(corpus_path).resolve()),
        "n_candidates": n_candidates,
        "fast_n_steps": fast_n_steps,
        "fast_n_restarts": _FAST_N_RESTARTS,
    }
    blob = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------


def derive_verdict(
    trained_hybrid_solve_rate: float,
    untrained_hybrid_solve_rate: float,
    pure_trained_energy_descent_solve_rate: float,
    mcnemar_p: float,
) -> str:
    """Map the three solve-rate scalars to exactly one ``complete:`` terminal verdict.

    Gate ladder (mirrors the task acceptance gates):
      - G1: trained_hybrid > untrained_hybrid AND significance favours treatment.
      - G1': trained_hybrid <= untrained_hybrid  → no lift.
      - G1'': pure trained descent solves (>50%) without hybrid → hybrid
              no longer required.
    """
    if (
        pure_trained_energy_descent_solve_rate > 0.5
        and trained_hybrid_solve_rate > untrained_hybrid_solve_rate
    ):
        return "complete: trained_energy_pure_descent_solves_hybrid_no_longer_required"
    if (
        trained_hybrid_solve_rate > untrained_hybrid_solve_rate
        and mcnemar_p < 0.05
    ):
        return "complete: trained_energy_strengthens_kona_hybrid_solve_rate"
    return "complete: trained_energy_no_lift_over_untrained_kona_hybrid"
