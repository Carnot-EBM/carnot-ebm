"""Tests for Phase-1a adversarial verifier robustness audit (exp1106 v2).

Spec: REQ-VER-090 — Phase-1a adversarial verifier robustness audit acceptance.

Covers the four mandatory assertions for the adversarial corpus,
attack application, false-pass rate computation, and IPT consistency.

Spec: REQ-PHASE1A-ADV-001 — false-pass rate < 5% across all attack types.

Why these tests exist:
    CLAUDE.md "Phase Prototype + Empirical Validation + Adversarial Check
    Discipline" requires concrete pass/fail tests with explicit thresholds
    at every phase boundary.  These four tests are the minimal gate set
    specified in the task brief.
"""

from __future__ import annotations

import json
import os
import random
import re
import sys

import numpy as np
import pytest

# Add repo python/ to path so carnot modules resolve
_repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_repo_root, "python"))

from carnot.verify.semenergy_probe import SemEnergyProbe  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers shared by all four tests
# ---------------------------------------------------------------------------

_CORPUS_PATH = os.path.join(_repo_root, "data", "fover_corpus_v4.json")
_ATTACK_TYPES = [
    "stylistic_padding",
    "structured_formatting",
    "ipt_isomorphic_permutation",
    "verbosity_inflation",
    "confidence_signaling",
]
_RNG = random.Random(42)


def _load_50_correct_50_incorrect() -> tuple[list[dict], list[dict]]:
    """Load corpus and return exactly 50 correct + 50 incorrect examples."""
    with open(_CORPUS_PATH) as fh:
        corpus = json.load(fh)
    correct = [x for x in corpus if x["label"] == "correct"]
    incorrect = [x for x in corpus if x["label"] == "incorrect"]
    r = random.Random(42)
    r.shuffle(correct)
    r.shuffle(incorrect)
    return correct[:50], incorrect[:50]


def _apply_attack(name: str, text: str) -> str:
    """Return the attacked version of *text* using the named attack type."""
    if name == "stylistic_padding":
        fillers = [
            "Let me think step by step.",
            "I will carefully reason through this.",
            "First, let me re-read the problem.",
        ]
        return " ".join(fillers[:3]) + "\n\n" + text
    if name == "structured_formatting":
        lines = [ln for ln in text.splitlines() if ln.strip()]
        return "## Solution\n\n" + "\n".join(f"- {ln}" for ln in lines)
    if name == "ipt_isomorphic_permutation":
        result = text
        for pat, rep in [
            (r"\bx\b", "y"),
            (r"\ba\b", "b"),
            (r"\bJohn\b", "Alice"),
            (r"\bhe\b", "she"),
        ]:
            result = re.sub(pat, rep, result)
        return result
    if name == "verbosity_inflation":
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        inflated = []
        for s in sentences:
            inflated.append(s)
            inflated.append("In other words, " + s[0].lower() + s[1:] if s else s)
        return " ".join(inflated)
    if name == "confidence_signaling":
        return text + "\n\nI am certain this is the correct answer."
    raise ValueError(f"Unknown attack: {name}")


def _false_pass_rate(
    probe: SemEnergyProbe, originals: list[str], attacked_texts: list[str]
) -> float:
    """Compute false-pass rate for one attack type.

    A false pass occurs when the attacked wrong answer gets an energy score
    at least 20% lower (more confident/correct) than the original wrong
    answer.  Uses the sign-agnostic threshold:
        threshold = orig_e - 0.2 * |orig_e|
    which correctly handles the negative-energy regime returned by
    SemEnergyProbe.score_response_proxy.
    """
    count = 0
    for orig_text, atk_text in zip(originals, attacked_texts):
        orig_e = probe.score_response_proxy(orig_text)
        atk_e = probe.score_response_proxy(atk_text)
        threshold = orig_e - 0.2 * abs(orig_e)
        if atk_e < threshold:
            count += 1
    return count / max(len(originals), 1)


# ---------------------------------------------------------------------------
# Test 1 — adversarial corpus has exactly 100 examples
# ---------------------------------------------------------------------------


def test_adversarial_corpus_100_examples_generated() -> None:
    """Corpus must yield exactly 50 correct + 50 incorrect = 100 examples.

    REQ-PHASE1A-ADV-001 requires a 100-example adversarial corpus drawn
    from data/fover_corpus_v4.json.  This test checks that the corpus is
    large enough and that the 50/50 split is achievable.
    """
    correct, incorrect = _load_50_correct_50_incorrect()
    assert len(correct) == 50, f"expected 50 correct, got {len(correct)}"
    assert len(incorrect) == 50, f"expected 50 incorrect, got {len(incorrect)}"
    assert len(correct) + len(incorrect) == 100


# ---------------------------------------------------------------------------
# Test 2 — all 5 attack types are applied and produce distinct output
# ---------------------------------------------------------------------------


def test_all_5_attack_types_applied() -> None:
    """Each of the 5 APRM-style attack transforms must run and mutate text.

    A transform that returns the unchanged text offers no adversarial
    signal and must be caught here.
    """
    sample_text = (
        "Let x = 3. Then x + 2 = 5. John computed 5 * 3 = 15. He concluded the total is 15."
    )
    assert len(_ATTACK_TYPES) == 5, f"expected 5 attack types, got {len(_ATTACK_TYPES)}"
    for attack_name in _ATTACK_TYPES:
        attacked = _apply_attack(attack_name, sample_text)
        assert isinstance(attacked, str), f"{attack_name}: returned non-string"
        assert len(attacked) > 0, f"{attack_name}: returned empty string"
        # Every attack except IPT (which only renames variables not present
        # in all texts) must produce a different string.
        if attack_name != "ipt_isomorphic_permutation":
            assert attacked != sample_text, (
                f"{attack_name}: transform produced identical text — attack is a no-op"
            )


# ---------------------------------------------------------------------------
# Test 3 — false-pass rate is computed for each attack and stored as float
# ---------------------------------------------------------------------------


def test_false_pass_rate_computed_per_attack() -> None:
    """False-pass rate must be in [0, 1] for all 5 attack types.

    Also verifies the Phase-1a acceptance gate: max false-pass rate < 5%.
    Uses a small 10-example subset to keep test runtime < 1 s.
    """
    probe = SemEnergyProbe()
    _, incorrect = _load_50_correct_50_incorrect()
    # Use first 10 to stay fast
    subset = incorrect[:10]
    original_texts = [ex["step_text"] for ex in subset]

    rates: dict[str, float] = {}
    for attack_name in _ATTACK_TYPES:
        attacked_texts = [_apply_attack(attack_name, t) for t in original_texts]
        rate = _false_pass_rate(probe, original_texts, attacked_texts)
        assert 0.0 <= rate <= 1.0, f"{attack_name}: rate {rate} out of [0,1]"
        rates[attack_name] = rate

    # Phase-1a acceptance gate
    max_rate = max(rates.values())
    assert max_rate < 0.05, (
        f"Phase-1a gate FAILED: max false-pass rate = {max_rate:.3f} >= 0.05. "
        f"Vulnerable attacks: {[k for k, v in rates.items() if v >= 0.05]}"
    )


# ---------------------------------------------------------------------------
# Test 4 — IPT consistency on correct examples (score stable under rename)
# ---------------------------------------------------------------------------


def test_ipt_consistency_measured_correct_examples() -> None:
    """IPT permutation must not change energy by more than 10% on correct examples.

    arXiv 2604.15149 (IPT) requires that structure-preserving permutations
    yield near-identical scores.  For SemEnergyProbe this means the
    relative delta |E(permuted) - E(original)| / |E(original)| < 0.10.

    Uses a 10-example subset for speed.
    """
    probe = SemEnergyProbe()
    correct, _ = _load_50_correct_50_incorrect()
    subset = correct[:10]

    deltas: list[float] = []
    for ex in subset:
        orig_e = probe.score_response_proxy(ex["step_text"])
        perm_text = _apply_attack("ipt_isomorphic_permutation", ex["step_text"])
        perm_e = probe.score_response_proxy(perm_text)
        denom = abs(orig_e) if abs(orig_e) > 1e-12 else 1e-12
        deltas.append(abs(perm_e - orig_e) / denom)

    mean_delta = float(np.mean(deltas))
    assert mean_delta < 0.10, (
        f"IPT consistency FAILED: mean delta = {mean_delta:.4f} >= 0.10. "
        "Verifier is not invariant to structure-preserving isomorphic permutations."
    )
