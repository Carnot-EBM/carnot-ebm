"""Tests for Diffusion-of-Thought verifier-guided inference.

Spec: REQ-INFER-017, SCENARIO-INFER-017-001
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import pytest

from carnot.inference.diffusion_of_thought import DiffusionOfThought


class ArithmeticEnergy:
    """Small deterministic verifier for REQ-INFER-017 tests."""

    def energy(self, response: str, context: str = "") -> float:
        text = f"{context} {response}"
        if "[MASK]" in text:
            text = text.replace("[MASK]", "")
        return 1.0 if re.search(r"\b5\b", text) else 0.0


def test_compute_token_energies_identifies_maskable_violation_token() -> None:
    """REQ-INFER-017: token masking exposes the high-violation token."""

    dot = DiffusionOfThought(ArithmeticEnergy())

    energies = dot.compute_token_energies("2 + 2 = 5", context="")

    assert energies == [0.0, 0.0, 0.0, 0.0, 1.0]


def test_propose_correction_uses_arithmetic_context_with_punctuation() -> None:
    """REQ-INFER-017: deterministic fallback candidates include arithmetic repair."""

    dot = DiffusionOfThought(ArithmeticEnergy(), n_candidates_per_step=5)

    candidates = dot.propose_correction("5.", context="Check 2 + 2 = 5.", position=4)

    assert "4." in candidates
    assert 3 <= len(candidates) <= 5
    assert len(candidates) == len(set(candidates))


def test_refine_repairs_response_and_returns_non_increasing_trace() -> None:
    """SCENARIO-INFER-017-001: refinement picks the lowest-energy replacement."""

    dot = DiffusionOfThought(ArithmeticEnergy())

    refined, trace = dot.refine("2 + 2 = 5", context="", n_steps=1)

    assert refined == "2 + 2 = 4"
    assert trace == [1.0, 0.0]
    assert all(after <= before for before, after in zip(trace, trace[1:]))


def test_refine_zero_steps_returns_initial_response_and_energy() -> None:
    """REQ-INFER-017: T=0 baseline scoring is stable for experiments."""

    dot = DiffusionOfThought(ArithmeticEnergy())

    refined, trace = dot.refine("2 + 2 = 5", context="", n_steps=0)

    assert refined == "2 + 2 = 5"
    assert trace == [1.0]


def test_refine_rejects_negative_step_count() -> None:
    """REQ-INFER-017: timestep counts are bounded to non-negative values."""

    dot = DiffusionOfThought(ArithmeticEnergy())

    with pytest.raises(ValueError, match="n_steps"):
        dot.refine("2 + 2 = 5", context="", n_steps=-1)


def test_propose_correction_handles_text_case_and_decimals() -> None:
    """REQ-INFER-017: fallback candidates preserve text shape."""

    dot = DiffusionOfThought(ArithmeticEnergy(), n_candidates_per_step=5)

    assert "TRUE" in dot.propose_correction("FALSE", context="", position=0)
    assert "True" in dot.propose_correction("False", context="", position=0)
    assert "therefore" not in dot.propose_correction("therefore", context="", position=0)
    assert "2.5" in dot.propose_correction("3.5", context="", position=0)


@dataclass
class VerifyResult:
    per_verifier_scores: dict[str, float]


class VerifyStyleEnsemble:
    def verify(self, question: str, response: str) -> VerifyResult:
        assert question == "q"
        assert response == "r"
        return VerifyResult({"a": 0.2, "b": 0.4, "c": 0.6})


@dataclass
class VerifyEnergyResult:
    energy: float


class VerifyEnergyStyleEnsemble:
    def verify(self, question: str, response: str) -> VerifyEnergyResult:
        assert question == "q"
        assert response == "r"
        return VerifyEnergyResult(0.3)


class ScoreStyleVerifier:
    def score(self, text: str) -> float:
        assert text == "q\nr"
        return 0.25


class CallableVerifier:
    def __call__(self, response: str, context: str) -> float:
        assert response == "r"
        assert context == "q"
        return 0.75


def test_composite_energy_adapts_common_verifier_interfaces() -> None:
    """REQ-INFER-017: DoT can sit on top of k=5-style verifier interfaces."""

    assert DiffusionOfThought(VerifyStyleEnsemble()).composite_energy("r", "q") == pytest.approx(
        0.4
    )
    assert DiffusionOfThought(VerifyEnergyStyleEnsemble()).composite_energy("r", "q") == 0.3
    assert DiffusionOfThought(ScoreStyleVerifier()).composite_energy("r", "q") == 0.25
    assert DiffusionOfThought(CallableVerifier()).composite_energy("r", "q") == 0.75
