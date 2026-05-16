"""Tests for the NRGPT continuous generation stub."""

import numpy as np

from carnot.phase3.nrgpt_stub import nrgpt_step, generate_sequence

def test_nrgpt_step_returns_negative_gradient():
    """Test that nrgpt_step computes J x + h."""
    state = np.array([0.5, -0.5])
    coupling = np.array([[1.0, 0.2], [0.2, 1.0]])
    bias = np.array([0.1, -0.1])

    expected_neg_grad = coupling @ state + bias
    actual_neg_grad = nrgpt_step(state, coupling, bias)

    np.testing.assert_allclose(actual_neg_grad, expected_neg_grad)

def test_generate_sequence_converges():
    """Test that the sequence of states converges to a fixed point."""
    rng = np.random.default_rng(42)
    dim = 5
    coupling = rng.standard_normal((dim, dim))
    coupling = (coupling + coupling.T) / 2.0  # Make symmetric
    bias = rng.standard_normal(dim)

    initial_state = rng.uniform(-0.5, 0.5, size=dim)

    sequence = generate_sequence(initial_state, coupling, bias, n_steps=100, lr=0.1)

    # Check that the last step change is very small (convergence)
    final_diff = np.linalg.norm(sequence[-1] - sequence[-2])
    assert final_diff < 1e-3, f"Sequence did not converge, final difference is {final_diff}"

    # Also verify that energy decreases along the sequence (mostly)
    def energy(x):
        return -0.5 * x @ coupling @ x - bias @ x

    energies = [energy(x) for x in sequence]
    assert energies[-1] < energies[0], "Energy did not decrease from start to end"
