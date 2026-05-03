"""Tests for BoltzmannGPTLayer — Exp 1226 Phase-3 seed integration.

Spec refs: REQ-PHASE3-BOLTZMANN-001
           SCENARIO-BOLTZMANN-001 (energy is scalar)
           SCENARIO-BOLTZMANN-002 (higher energy = lower score)
           SCENARIO-BOLTZMANN-003 (score is not constant across distinct inputs)

Background:
    arXiv 2601.17094 (Boltzmann-GPT, January 2026) proposes using a Boltzmann
    machine world model to score candidate continuations in an energy-weighted
    beam search. This test file verifies the seed implementation in
    carnot.phase3.continuous_ebm.BoltzmannGPTLayer behaves correctly before
    any contrastive training is applied.

    The seed experiment deliberately uses random (untrained) weights. We test
    *structural* properties of the layer — not that it achieves high AUROC.
    AUROC is measured separately in the Exp 1226 artifact.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_PYTHON_DIR = _PROJECT_ROOT / "python"
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
if str(_PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(_PYTHON_DIR))

from carnot.phase3.continuous_ebm import BoltzmannGPTLayer  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def layer() -> BoltzmannGPTLayer:
    """Default BoltzmannGPTLayer with small fixed seed."""
    return BoltzmannGPTLayer(hidden_dim=16, visible_dim=16, seed=42)


@pytest.fixture
def tiny_layer() -> BoltzmannGPTLayer:
    """Minimal 4-dim layer for numerical precision checks."""
    return BoltzmannGPTLayer(hidden_dim=4, visible_dim=4, seed=7)


# ---------------------------------------------------------------------------
# SCENARIO-BOLTZMANN-001 : energy returns a scalar
# ---------------------------------------------------------------------------

class TestEnergyIsScalar:
    """REQ-PHASE3-BOLTZMANN-001 / SCENARIO-BOLTZMANN-001."""

    def test_energy_is_scalar(self, layer: BoltzmannGPTLayer) -> None:
        """energy(v, h) must return a Python float, not an array."""
        v = np.zeros(16)
        h = np.zeros(16)
        result = layer.energy(v, h)
        # Must be a bare Python float (JSON-serialisable, not np.float64 subclass
        # that surprises callers doing `type(x) is float` checks).
        assert isinstance(result, float)

    def test_energy_is_finite(self, layer: BoltzmannGPTLayer) -> None:
        """energy must be finite for zero inputs (no overflow / NaN)."""
        v = np.zeros(16)
        h = np.zeros(16)
        assert np.isfinite(layer.energy(v, h))

    def test_energy_is_finite_for_random_inputs(self, layer: BoltzmannGPTLayer) -> None:
        """energy must stay finite for bounded random inputs."""
        rng = np.random.default_rng(0)
        for _ in range(10):
            v = rng.uniform(0, 1, 16)
            h = rng.uniform(0, 1, 16)
            assert np.isfinite(layer.energy(v, h))

    def test_energy_zero_for_zero_weights(self) -> None:
        """With all-zero weights and biases, energy is 0 regardless of inputs."""
        layer = BoltzmannGPTLayer(hidden_dim=4, visible_dim=4, seed=0)
        # Force zero weights (override after construction)
        layer.W = np.zeros((4, 4))
        layer.b = np.zeros(4)
        layer.c = np.zeros(4)
        v = np.ones(4)
        h = np.ones(4)
        assert layer.energy(v, h) == pytest.approx(0.0)

    def test_energy_formula_matches_manual(self, tiny_layer: BoltzmannGPTLayer) -> None:
        """energy(v, h) = -(v @ W @ h) - (b @ v) - (c @ h): verify against hand calc."""
        v = np.array([1.0, 0.0, 0.0, 0.0])
        h = np.array([1.0, 0.0, 0.0, 0.0])
        expected = -(v @ tiny_layer.W @ h) - (tiny_layer.b @ v) - (tiny_layer.c @ h)
        assert tiny_layer.energy(v, h) == pytest.approx(expected)


# ---------------------------------------------------------------------------
# SCENARIO-BOLTZMANN-002 : higher energy → lower score
# ---------------------------------------------------------------------------

class TestHigherEnergyLowerScore:
    """REQ-PHASE3-BOLTZMANN-001 / SCENARIO-BOLTZMANN-002.

    score(tokens) = -energy(v, h), so increasing the energy must decrease the score.
    We verify this by constructing two inputs with known energy ordering.
    """

    def test_higher_energy_lower_score(self) -> None:
        """Manually set W so high-energy input gets lower score than low-energy input."""
        layer = BoltzmannGPTLayer(hidden_dim=4, visible_dim=4, seed=0)
        # Make W a positive diagonal matrix so v@W@h = sum(v_i * h_i) is large when
        # both v and h are large — large coupling → low energy → high score.
        layer.W = np.eye(4) * 10.0  # strong coupling
        layer.b = np.zeros(4)
        layer.c = np.zeros(4)
        # Override _embed_tokens / _infer_hidden by setting W directly and
        # calling energy/score on concrete v, h.
        v_high = np.ones(4)  # high coupling → low energy
        h_high = np.ones(4)
        v_low = np.zeros(4)  # no coupling → zero energy (higher)
        h_low = np.zeros(4)

        e_high_coupling = layer.energy(v_high, h_high)  # should be very negative
        e_low_coupling = layer.energy(v_low, h_low)  # should be 0

        assert e_high_coupling < e_low_coupling, (
            f"Expected high-coupling energy ({e_high_coupling}) < "
            f"low-coupling energy ({e_low_coupling})"
        )
        # score = -energy, so high coupling → higher score
        score_high = -e_high_coupling
        score_low = -e_low_coupling
        assert score_high > score_low

    def test_score_sign_is_negative_of_energy(self, layer: BoltzmannGPTLayer) -> None:
        """score(tokens) must equal -energy(v, h) exactly (no extra transforms)."""
        tokens = ["The", "answer", "is", "42"]
        v = layer._embed_tokens(tokens)
        h = layer._infer_hidden(v)
        expected_score = -layer.energy(v, h)
        actual_score = layer.score(tokens)
        assert actual_score == pytest.approx(expected_score, abs=1e-12)

    def test_energy_decreases_with_stronger_coupling(self) -> None:
        """Scaling W by k scales energy by k; stronger coupling lowers energy for aligned v, h."""
        layer = BoltzmannGPTLayer(hidden_dim=4, visible_dim=4, seed=1)
        v = np.ones(4) * 0.5
        h = np.ones(4) * 0.5

        e_baseline = layer.energy(v, h)
        layer.W *= 2.0  # double the coupling strength
        e_scaled = layer.energy(v, h)

        # Doubling W doubles the v@W@h term; biases are zero so full energy doubles
        # (direction depends on sign of v@W@h, but ratio must hold).
        assert e_scaled == pytest.approx(2.0 * e_baseline, rel=1e-6)


# ---------------------------------------------------------------------------
# SCENARIO-BOLTZMANN-003 : score is not constant across distinct inputs
# ---------------------------------------------------------------------------

class TestScoreNotConstantAcrossInputs:
    """REQ-PHASE3-BOLTZMANN-001 / SCENARIO-BOLTZMANN-003.

    A random initialisation must produce non-degenerate scores.  Constant
    scores would mean the layer cannot distinguish any two inputs, making
    AUROC measurement meaningless.
    """

    def test_score_not_constant_across_inputs(self, layer: BoltzmannGPTLayer) -> None:
        """Score must vary across at least 3 distinct token sequences."""
        sequences = [
            ["The", "answer", "is", "42"],
            ["Compute", "2", "plus", "2", "equals", "4"],
            ["The", "quick", "brown", "fox"],
        ]
        scores = [layer.score(tokens) for tokens in sequences]
        assert len(set(scores)) > 1, (
            f"All scores are identical ({scores[0]:.6f}); layer produces degenerate output."
        )

    def test_score_varies_across_synthetic_fover_rows(self, layer: BoltzmannGPTLayer) -> None:
        """Score distribution across 20 synthetic FoVer rows is non-degenerate."""
        # Inline 5 synthetic rows to avoid import of nrgpt_energy in this test
        rows = [
            "Compute 47 + 26. The answer is 73.",
            "Compute 38 + 29. The answer is 65.",
            "Compute 26 + 47. The answer is 73.",
            "Compute 48 + 48. The answer is 97.",
            "Compute 17 + 29. The answer is 46.",
        ]
        scores = [layer.score(text.split()) for text in rows]
        score_range = max(scores) - min(scores)
        assert score_range > 1e-6, (
            f"Score range {score_range:.2e} is too small; layer is nearly constant."
        )

    def test_embed_tokens_returns_unit_vector_for_nonempty_input(
        self, layer: BoltzmannGPTLayer
    ) -> None:
        """_embed_tokens must return an L2-normalised vector for non-trivial input."""
        v = layer._embed_tokens(["hello", "world"])
        assert v.shape == (16,)
        assert np.linalg.norm(v) == pytest.approx(1.0, abs=1e-9)

    def test_embed_tokens_uniform_for_single_char_tokens(
        self, layer: BoltzmannGPTLayer
    ) -> None:
        """Single-char tokens have no bigrams → fallback to uniform distribution."""
        v = layer._embed_tokens(["a", "b", "c"])
        expected = np.full(16, 1.0 / 16)
        np.testing.assert_allclose(v, expected, atol=1e-12)

    def test_infer_hidden_values_in_unit_interval(self, layer: BoltzmannGPTLayer) -> None:
        """_infer_hidden must return values in (0, 1) — sigmoid outputs."""
        v = layer._embed_tokens(["The", "answer", "is", "42"])
        h = layer._infer_hidden(v)
        assert h.shape == (16,)
        assert np.all(h > 0.0) and np.all(h < 1.0)

    def test_score_empty_token_sequence_does_not_crash(
        self, layer: BoltzmannGPTLayer
    ) -> None:
        """score([]) must return a finite float (uniform fallback in _embed_tokens)."""
        result = layer.score([])
        assert isinstance(result, float)
        assert np.isfinite(result)
