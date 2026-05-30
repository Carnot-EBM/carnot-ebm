"""Tests for the energy-descent-vs-autoregressive premise helpers (exp3312).

Every test traces to REQ-KONA-3312 (the premise test) and the reasoning-mode
invariants REQ-KONA-001 (no token sampling in the latent loop) / REQ-KONA-002
(bounded-depth refinement). These cover the pure judgement logic that decides
whether the Phase-3 premise holds, so a reviewer can re-verify the gate without
a GPU.
"""

from __future__ import annotations

import json

import pytest
import torch

from carnot.phase3.energy_descent_premise import (
    _to_number,
    EnergyDescentResult,
    GsmProblem,
    derive_premise_verdict,
    energy_descent_select,
    extract_final_answer,
    is_correct,
    load_gsm8k_subset,
    majority_vote,
    mcnemar_test,
    paired_bootstrap_ci,
    reproducibility_checksum,
)


# --- REQ-KONA-3312: corpus loading is deterministic and CLT-valid ----------


def _write_corpus(path, n_rows):
    """Write a JSONL corpus shaped like exp281's GSM8K original-question file."""
    with open(path, "w", encoding="utf-8") as handle:
        for i in range(n_rows):
            handle.write(
                json.dumps(
                    {
                        "question_id": f"gsm8k-{i}",
                        "original_question": f"What is {i} plus {i}?",
                        "original_answer": 2 * i,
                        "variant_question": "ignored",
                        "variant_answer": -1,
                    }
                )
                + "\n"
            )
    return path


def test_load_gsm8k_subset_is_deterministic_and_paired(tmp_path):
    """SCENARIO-KONA-3312: same (path, n, seed) yields the identical ordered split."""
    corpus = _write_corpus(tmp_path / "gsm.jsonl", 300)
    a = load_gsm8k_subset(corpus, n=200, seed=3312)
    b = load_gsm8k_subset(corpus, n=200, seed=3312)
    assert len(a) == 200
    assert [p.problem_id for p in a] == [p.problem_id for p in b]
    assert all(isinstance(p, GsmProblem) and isinstance(p.answer, int) for p in a)
    # A different seed reshuffles, so the order must differ (paired-ness is per-seed).
    c = load_gsm8k_subset(corpus, n=200, seed=999)
    assert [p.problem_id for p in a] != [p.problem_id for p in c]


def test_load_gsm8k_subset_rejects_undersized_corpus(tmp_path):
    """REQ-KONA-3312: a corpus below n must raise, not silently truncate (CLT)."""
    corpus = _write_corpus(tmp_path / "small.jsonl", 50)
    with pytest.raises(ValueError, match="need n=200"):
        load_gsm8k_subset(corpus, n=200, seed=3312)


def test_load_gsm8k_subset_skips_malformed_and_dedupes(tmp_path):
    """Rows without a numeric original answer are skipped; ids dedupe."""
    path = tmp_path / "messy.jsonl"
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n")  # blank line
        handle.write(json.dumps({"question_id": "a", "original_question": "q", "original_answer": "notnum"}) + "\n")
        handle.write(json.dumps({"question_id": "b", "original_question": "q1", "original_answer": 3}) + "\n")
        handle.write(json.dumps({"question_id": "b", "original_question": "dup", "original_answer": 3}) + "\n")
        handle.write(json.dumps({"original_question": "q2", "original_answer": 4}) + "\n")  # no id -> row-N
        handle.write(json.dumps({"question_id": "c", "original_answer": 9}) + "\n")  # no question -> skipped
    rows = load_gsm8k_subset(path, n=2, seed=1)
    ids = {p.problem_id for p in rows}
    assert len(rows) == 2
    assert "a" not in ids  # malformed answer skipped
    assert "c" not in ids  # missing question skipped


def test_to_number_defensive_branches():
    """_to_number returns None on empty/dash and on un-floatable tokens."""
    assert _to_number("$") is None  # strips to "" -> None
    assert _to_number("-") is None
    assert _to_number("1.2.3") is None  # not floatable -> ValueError branch


# --- REQ-KONA-3312: answer extraction --------------------------------------


@pytest.mark.parametrize(
    "text,expected",
    [
        ("blah blah #### 42", 42),
        ("steps... #### 1,234 done", 1234),
        ("the answer is $18.0", 18),
        ("no numbers here", None),
        ("", None),
        ("first 5 then #### 7", 7),  # coda wins over earlier number
        ("ends with 9.", 9),
        ("#### -3", -3),
    ],
)
def test_extract_final_answer(text, expected):
    """REQ-KONA-3312: #### coda preferred, last-number fallback, None on no-number."""
    assert extract_final_answer(text) == expected


def test_is_correct_requires_exact_integer_match():
    assert is_correct(42, 42)
    assert not is_correct(41, 42)
    assert not is_correct(None, 42)


def test_majority_vote_self_consistency():
    """Equal-compute AR control: modal answer, deterministic tie-break, None-safe."""
    assert majority_vote([3, 3, 5, None]) == 3
    assert majority_vote([None, None]) is None
    # Tie between 1 and 2 -> earliest-seen (1) wins deterministically.
    assert majority_vote([1, 2, 2, 1]) == 1


# --- REQ-KONA-001/002: bounded-depth latent energy descent, no sampling ----


class _QuadEnergy(torch.nn.Module):
    """A trivial convex energy E(z) = ||z - target||^2 with a known minimum.

    Lets us assert the descent lowers energy and selects the candidate nearest
    the learned 'correct' manifold, without needing a trained Boltzmann-GPT.
    """

    def __init__(self, target):
        super().__init__()
        self.target = target

    def forward(self, z):
        return ((z - self.target) ** 2).sum(dim=1)


def _toy_embed(texts, *, visible_dim=2):
    """Deterministic embedding: map each candidate to a fixed point in R^2."""
    table = {"near": [0.1, 0.0], "far": [5.0, 5.0], "mid": [1.0, 1.0]}
    return torch.tensor([table[t] for t in texts], dtype=torch.float32)


def test_energy_descent_selects_lowest_energy_candidate():
    """REQ-KONA-001/002: pick the candidate whose refined latent has min energy."""
    energy = _QuadEnergy(torch.tensor([[0.0, 0.0]]))
    result = energy_descent_select(
        ["far", "near", "mid"],
        energy,
        visible_dim=2,
        n_steps=5,
        lr=0.1,
        embed_fn=_toy_embed,
    )
    assert isinstance(result, EnergyDescentResult)
    assert result.selected_index == 1  # "near" is closest to the energy minimum
    assert result.n_steps == 5
    # Descent must not increase energy on a convex bowl.
    for init, fin in zip(result.initial_energies, result.final_energies, strict=True):
        assert fin <= init + 1e-6


def test_energy_descent_requires_candidates():
    energy = _QuadEnergy(torch.tensor([[0.0, 0.0]]))
    with pytest.raises(ValueError, match="at least one candidate"):
        energy_descent_select([], energy, embed_fn=_toy_embed)


def test_energy_descent_uses_real_embed_default():
    """Default embed path wires to Boltzmann-GPT embed_texts (16-dim features)."""
    energy = _QuadEnergy(torch.zeros(1, 16))
    result = energy_descent_select(
        ["correct reasoning here", "wrong"],
        energy,
        visible_dim=16,
        n_steps=3,
        lr=0.05,
    )
    assert 0 <= result.selected_index < 2
    assert len(result.final_energies) == 2


# --- REQ-KONA-3312: paired significance ------------------------------------


def test_mcnemar_counts_discordant_pairs_and_direction():
    ar = [True, False, False, True, True]
    ed = [True, True, True, True, False]
    # discordant: idx1 (ed win), idx2 (ed win), idx4 (ar win) -> b=2, c=1
    res = mcnemar_test(ar, ed)
    assert res["energy_descent_wins"] == 2.0
    assert res["ar_wins"] == 1.0
    assert res["direction"] == 1.0
    assert 0.0 <= res["p_value"] <= 1.0


def test_mcnemar_ar_favoured_direction_is_negative():
    """When AR wins more discordant pairs, direction is -1."""
    ar = [True, True, False]
    ed = [False, False, False]
    res = mcnemar_test(ar, ed)
    assert res["ar_wins"] == 2.0
    assert res["energy_descent_wins"] == 0.0
    assert res["direction"] == -1.0


def test_mcnemar_no_discordance_is_p_one():
    res = mcnemar_test([True, False], [True, False])
    assert res["p_value"] == 1.0
    assert res["direction"] == 0.0


def test_mcnemar_strong_effect_is_significant():
    """20 energy-descent wins, 0 AR wins -> exact p well below 0.05."""
    ar = [False] * 20 + [True] * 5
    ed = [True] * 20 + [True] * 5
    res = mcnemar_test(ar, ed)
    assert res["energy_descent_wins"] == 20.0
    assert res["ar_wins"] == 0.0
    assert res["p_value"] < 0.05
    assert res["direction"] == 1.0


def test_mcnemar_length_mismatch_raises():
    with pytest.raises(ValueError, match="equal-length"):
        mcnemar_test([True], [True, False])


def test_paired_bootstrap_ci_positive_for_clear_win():
    ar = [False] * 40 + [True] * 10
    ed = [True] * 40 + [True] * 10
    lo, hi = paired_bootstrap_ci(ar, ed, n_boot=500, seed=1)
    assert lo > 0.0
    assert hi >= lo


def test_paired_bootstrap_ci_empty_and_mismatch():
    assert paired_bootstrap_ci([], []) == (0.0, 0.0)
    with pytest.raises(ValueError, match="equal-length"):
        paired_bootstrap_ci([True], [])


# --- REQ-KONA-3312: verdict mapping ----------------------------------------


def test_verdict_g2_validated_on_significant_win():
    v = derive_premise_verdict(0.40, 0.55, p_value=0.001, ci=(0.05, 0.25), direction=1.0)
    assert v.g1_premise_viable and v.g2_premise_validated
    assert v.verdict == "complete: energy_descent_beats_ar_premise_validated"


def test_verdict_g1_viable_when_matches_but_not_significant():
    v = derive_premise_verdict(0.50, 0.50, p_value=0.40, ci=(-0.05, 0.05), direction=0.0)
    assert v.g1_premise_viable and not v.g2_premise_validated
    assert v.verdict == "complete: energy_descent_viable_not_superior_at_scale"


def test_verdict_g1_viable_when_worse_but_not_significant():
    """Non-inferiority: slightly worse but the gap isn't significant -> still viable."""
    v = derive_premise_verdict(0.52, 0.50, p_value=0.30, ci=(-0.08, 0.04), direction=-1.0)
    assert v.g1_premise_viable and not v.g2_premise_validated


def test_verdict_unsupported_when_significantly_worse():
    v = derive_premise_verdict(0.55, 0.40, p_value=0.001, ci=(-0.25, -0.05), direction=-1.0)
    assert not v.g1_premise_viable and not v.g2_premise_validated
    assert v.verdict == "complete: energy_descent_below_ar_premise_unsupported_at_scale"


def test_verdict_g2_requires_positive_ci_lower_bound():
    """A significant-looking p but a CI touching 0 must not validate G2."""
    v = derive_premise_verdict(0.50, 0.53, p_value=0.04, ci=(-0.01, 0.10), direction=1.0)
    assert not v.g2_premise_validated
    assert v.g1_premise_viable


# --- REQ-KONA-3312: reproducibility checksum -------------------------------


def test_reproducibility_checksum_is_content_addressed(tmp_path):
    f = tmp_path / "c.jsonl"
    f.write_text("abc", encoding="utf-8")
    h1 = reproducibility_checksum(corpus_path=f, n_problems=200, seed=3312, substrate_signature="bgpt-v1")
    h2 = reproducibility_checksum(corpus_path=f, n_problems=200, seed=3312, substrate_signature="bgpt-v1")
    assert h1 == h2 and len(h1) == 16
    f.write_text("different", encoding="utf-8")
    h3 = reproducibility_checksum(corpus_path=f, n_problems=200, seed=3312, substrate_signature="bgpt-v1")
    assert h3 != h1  # content change -> checksum change
