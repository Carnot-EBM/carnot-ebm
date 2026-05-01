"""Tests for experiment 1110 — RLVR + SSD v2 non-degenerate corpus.

Spec: REQ-PHI-001 (alpha_t selection signal), REQ-VERIFY-083 (live_gpu evidence).

Each test references the experiment's load-bearing claim:

- ``test_sota_model_corpus_has_nonzero_energy_distribution``
    SCENARIO: a freshly-generated corpus from a SOTA model produces
    continuous energy scores with non-zero spread, not a degenerate
    all-zero distribution.  This is the bug exp1099 hit and exp1110 fixes.

- ``test_rlvr_condition_uses_top_k_not_median``
    SCENARIO: the RLVR selection uses the top-30th-percentile by HIGHEST
    energy.  Median-threshold selection on a continuous distribution would
    accept everything ``<= median`` (50 %), not 30 %, and is what made
    exp1099 degenerate.

- ``test_ssd_condition_selects_low_energy_answers``
    SCENARIO: the SSD selection uses the top-30th-percentile by LOWEST
    energy (most-confident self-teacher signals).

- ``test_both_conditions_evaluated_on_held_out_set``
    SCENARIO: the experiment writes a held-out eval block alongside the
    training-corpus selection metrics, so both can be audited
    independently.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# Make the experiment script importable as ``exp_1110``.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "scripts"))
if str(_REPO_ROOT / "python") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "python"))

import experiment_1110_rlvr_ssd_v2_nondegenerate_live_gpu as exp_1110  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers — synthetic corpus generator
# ---------------------------------------------------------------------------


def _synthetic_record(qid: str, response: str, *, correct: bool, temperature: float = 0.7) -> dict:
    """Build a minimal record matching the shape ``compute_energy`` expects.

    The energy field is added by ``compute_energy`` so the record's
    ``response`` text is what determines the energy value.  We use real
    arithmetic-like text so SemEnergy's proxy and Z3 verifier both have
    something concrete to score.
    """
    return {
        "question_id": qid,
        "question": "How much is 2 + 3?",
        "answer": 5.0,
        "response": response,
        "temperature": float(temperature),
        "correct": bool(correct),
    }


# ---------------------------------------------------------------------------
# 1. Non-zero energy distribution from real-looking corpus
# ---------------------------------------------------------------------------


def test_sota_model_corpus_has_nonzero_energy_distribution() -> None:
    """A varied set of model-style responses must NOT produce all-zero energy.

    This is the load-bearing fix relative to exp1099.  The .85 corpus had
    every row at energy=0.0 because of upstream pre-filtering.  A fresh
    corpus scored by SemEnergy + Z3 must have genuine spread: any text-
    aware probe over diverse responses produces continuous values, never a
    degenerate single-point distribution.
    """
    responses = [
        "We need 2 + 3 = 5. The answer is 5.",
        "Adding 2 plus 3 gives 5. Answer: 5.",
        "Let me think. 2 + 3 = 6. So 6.",  # arithmetic violation
        "Maybe approximately around five or so, possibly six.",  # hedging, no math
        "",  # empty response
        "5",  # bare number
    ]
    records = [
        _synthetic_record(f"q{i}", r, correct=(i in {0, 1, 5})) for i, r in enumerate(responses)
    ]
    for rec in records:
        rec.update(exp_1110.compute_energy(rec["response"], rec["question"]))

    diag = exp_1110._energy_diagnostics(records)
    assert diag["all_zero"] is False, "energy distribution must not be all-zero"
    assert diag["nonzero_fraction"] > 0.5, (
        f"nonzero_fraction={diag['nonzero_fraction']} — expected most records to have non-zero energy"
    )
    assert diag["max"] > diag["min"], "distribution must have non-zero spread (max > min)"


# ---------------------------------------------------------------------------
# 2. RLVR uses top-k highest by energy, not median
# ---------------------------------------------------------------------------


def test_rlvr_condition_uses_top_k_not_median() -> None:
    """RLVR must select the highest-energy 30 %, not "<= median" (which is 50 %).

    Build 10 records with strictly-increasing energy values 0.0..9.0.  The
    RLVR top-30 % selection must pick the 3 highest (energies 7.0, 8.0,
    9.0), not the 5 records with energy <= median.  This is the exact
    regression that made exp1099's RLVR condition degenerate.
    """
    records = [
        {
            "question_id": f"q{i}",
            "response": f"resp{i}",
            "question": "?",
            "answer": 0.0,
            "correct": False,
            "energy": float(i),
        }
        for i in range(10)
    ]
    rlvr_subset = exp_1110._top_k_by_energy(records, fraction=exp_1110.TOP_K_FRACTION, highest=True)
    assert len(rlvr_subset) == 3, f"30%-of-10 should round to 3, got {len(rlvr_subset)}"
    selected_energies = sorted(r["energy"] for r in rlvr_subset)
    assert selected_energies == [7.0, 8.0, 9.0], (
        f"RLVR must pick the top-3 highest-energy records, got energies {selected_energies}"
    )


# ---------------------------------------------------------------------------
# 3. SSD uses top-k lowest by energy
# ---------------------------------------------------------------------------


def test_ssd_condition_selects_low_energy_answers() -> None:
    """SSD must select the lowest-energy 30 % (most-confident teacher signals).

    On the same monotone 0..9 corpus as above, SSD selects the 3 records
    with the LOWEST energy (0.0, 1.0, 2.0).
    """
    records = [
        {
            "question_id": f"q{i}",
            "response": f"resp{i}",
            "question": "?",
            "answer": 0.0,
            "correct": False,
            "energy": float(i),
        }
        for i in range(10)
    ]
    ssd_subset = exp_1110._top_k_by_energy(records, fraction=exp_1110.TOP_K_FRACTION, highest=False)
    assert len(ssd_subset) == 3
    selected_energies = sorted(r["energy"] for r in ssd_subset)
    assert selected_energies == [0.0, 1.0, 2.0], (
        f"SSD must pick the bottom-3 lowest-energy records, got energies {selected_energies}"
    )


# ---------------------------------------------------------------------------
# 4. Conditions evaluate on held-out eval set
# ---------------------------------------------------------------------------


def test_both_conditions_evaluated_on_held_out_set() -> None:
    """``evaluate_held_out`` must produce a baseline + count over eval records.

    The held-out eval is independent of the training corpus.  It must
    report ``eval_n_questions`` matching the size of the eval list and
    ``eval_baseline_fraction_correct`` matching the simple per-record
    correctness rate.  This guarantees the two evaluation surfaces (corpus
    selection vs held-out) stay independent in the artifact.
    """
    train_records = [_synthetic_record(f"t{i}", "2 + 3 = 5", correct=True) for i in range(4)]
    for rec in train_records:
        rec["energy"] = 0.5
    eval_records = [
        _synthetic_record("e0", "2 + 3 = 5", correct=True),
        _synthetic_record("e1", "2 + 3 = 7", correct=False),
        _synthetic_record("e2", "2 + 3 = 5", correct=True),
        _synthetic_record("e3", "", correct=False),
    ]
    block = exp_1110.evaluate_held_out(train_records, eval_records)
    assert block["eval_n_questions"] == 4
    assert block["eval_baseline_fraction_correct"] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Bonus — top_k handles tiny corpora gracefully (regression for empty edge case)
# ---------------------------------------------------------------------------


def test_top_k_returns_empty_on_empty_corpus() -> None:
    """``_top_k_by_energy`` must not crash on an empty input list."""
    assert exp_1110._top_k_by_energy([], fraction=0.3, highest=True) == []
    assert exp_1110._top_k_by_energy([], fraction=0.3, highest=False) == []
