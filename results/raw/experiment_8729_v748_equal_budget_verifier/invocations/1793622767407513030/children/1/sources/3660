"""NRGPT Continuous Generation Stub.

**Researcher summary:**
    Implements a generation step for NRGPT (arXiv:2602.15002) where
    the sequence generation is mapped to energy descent in the Phase 3
    continuous EBM space. The transformer generation step corresponds
    to computing the negative gradient of the energy function.
"""

import numpy as np

def nrgpt_step(state: np.ndarray, coupling: np.ndarray, bias: np.ndarray) -> np.ndarray:
    """Compute the NRGPT generation step as the negative gradient of the energy.

    For the Phase 3 ContinuousEBM, the energy is:
        E(x) = -0.5 * x^T J x - h^T x
    The gradient is:
        dE/dx = -J x - h
    The negative gradient (the descent direction) is:
        -dE/dx = J x + h

    Args:
        state: The current continuous latent state vector (shape (n,)).
        coupling: The coupling matrix J (shape (n, n)).
        bias: The bias vector h (shape (n,)).

    Returns:
        The negative gradient vector (shape (n,)).
    """
    return coupling @ state + bias

def generate_sequence(
    initial_state: np.ndarray,
    coupling: np.ndarray,
    bias: np.ndarray,
    n_steps: int = 50,
    lr: float = 0.1,
) -> list[np.ndarray]:
    """Generate a sequence of states via NRGPT steps (energy descent).

    At each step, the new state is computed by stepping in the direction of the
    negative gradient, with a tanh nonlinearity to keep states bounded in (-1, 1).

    Args:
        initial_state: The starting latent state.
        coupling: The coupling matrix.
        bias: The bias vector.
        n_steps: Number of generation steps.
        lr: Learning rate (step size).

    Returns:
        A list of state vectors over the generation sequence.
    """
    sequence = [initial_state.copy()]
    x = initial_state.copy()
    for _ in range(n_steps):
        neg_grad = nrgpt_step(x, coupling, bias)
        x = np.tanh(x + lr * neg_grad)
        sequence.append(x)
    return sequence
