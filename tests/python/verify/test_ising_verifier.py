"""Tests for text arithmetic energy in ``carnot.verify.semantic_energy``.

Spec: REQ-LEARN-1209-2
"""

from __future__ import annotations

from carnot.verify.semantic_energy import IsingVerifier


def test_ising_verifier_text_energy_discriminates_arithmetic_claims() -> None:
    """REQ-LEARN-1209-2: arithmetic claims map to bounded violation energy."""
    verifier = IsingVerifier()

    assert verifier.energy("The sum 2+3=5") == 0.0
    assert verifier.energy("The sum 2+3=6") == 1.0
    assert verifier.energy("no arithmetic here") == 0.0
