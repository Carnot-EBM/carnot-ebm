"""Continuous-valued EBM — Phase 3 seed bridging discrete Ising to Kona's domain.

**Researcher summary:**
    Minimal continuous EBM that reuses an Ising model's coupling matrix J and
    bias h as initialisation.  Gradient descent with tanh squashing finds an
    energy minimum in the continuous [-1, 1]^n hypercube.  Used in Exp 435a to
    validate that discrete (simulated annealing) and continuous (gradient descent)
    minimisers agree on the same 10-variable problem.

**Why this matters for Phase 3:**
    Kona-style reasoning requires non-autoregressive inference over a continuous
    latent space.  Before we can train such a model we need to verify that our
    quadratic energy function E(x) = -0.5*x^T*J*x - h^T*x is consistent across
    the discrete↔continuous boundary.  If the gradient-descent minimiser lands in
    the same region as simulated annealing, the energy landscape is "trustworthy"
    and can serve as the foundation for a continuous latent-space reasoner.

**Why tanh squashing?**
    The Ising model's natural variable domain is {-1, +1} (discrete spins).  The
    continuous relaxation x ∈ ℝ^n has unbounded energy minima — adding tanh ensures
    x ∈ (-1, 1)^n, keeping the continuous and discrete problems comparable.

**Why JAX/numpy and NOT torch?**
    Phase 3 must be portable to future hardware (Extropic TSU, photonic computing).
    JAX's functional purity and XLA backend make this easier than PyTorch, which
    carries CUDA-centric assumptions.

Spec: REQ-KONA-001, SCENARIO-KONA-001, SCENARIO-KONA-002
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class ContinuousEBM:
    """Continuous-valued EBM with quadratic energy and tanh squashing.

    **Researcher summary:**
        Stores the coupling matrix J and bias h from an Ising model.
        Energy: E(x) = -0.5 * x^T * J * x - h^T * x, with x ∈ (-1, 1)^n
        enforced via tanh during sampling.

    **Detailed explanation for engineers:**
        This is intentionally minimal — no hidden layers, no learned parameters.
        It is the simplest possible continuous relaxation of the Ising energy
        function, used only to verify that gradient descent and simulated
        annealing agree on the energy landscape geometry.

    Attributes:
        variables: Number of variables (n).
        coupling: Symmetric coupling matrix J of shape (n, n).
        bias: Bias vector h of shape (n,).
    """

    variables: int
    coupling: np.ndarray
    bias: np.ndarray


def fit_continuous_ebm(ising_model: Any) -> ContinuousEBM:
    """Construct a ContinuousEBM directly from an Ising model's parameters.

    **Researcher summary:**
        Reuses J and h verbatim — no fitting, no gradient steps.  This is
        the "same problem, different solver" construction.

    **Detailed explanation for engineers:**
        The Ising model already has the coupling matrix (J) and bias (h) we
        need.  We just copy them into a ContinuousEBM so the continuous sampler
        can use the same energy function.  If the two minimisers agree, it
        means the energy landscape is not an artefact of the discrete domain.

    Args:
        ising_model: Any object with `coupling` (array, shape (n, n)) and
            `bias` (array, shape (n,)) attributes.  Typically an IsingModel.

    Returns:
        ContinuousEBM with the same coupling and bias as the Ising model.
    """
    coupling = np.asarray(ising_model.coupling, dtype=np.float64)
    bias = np.asarray(ising_model.bias, dtype=np.float64)
    n = coupling.shape[0]
    return ContinuousEBM(variables=n, coupling=coupling, bias=bias)


def sample_continuous(
    model: ContinuousEBM,
    n_steps: int = 1000,
    lr: float = 0.01,
    seed: int = 0,
) -> np.ndarray:
    """Find an approximate energy minimum via gradient descent with tanh squashing.

    **Researcher summary:**
        Vanilla gradient descent on E(x) = -0.5*x^T*J*x - h^T*x.  Each step:
        1. dE/dx = -J*x - h  (analytic gradient)
        2. x ← tanh(x - lr * dE/dx)   (step + squash to [-1, 1])

    **Why tanh squashing at each step (not just at the end)?**
        Applying tanh after each gradient step keeps x inside the hypercube
        throughout optimisation, not just at the final output.  This prevents
        the gradient from exploiting the unbounded ℝ^n extension (e.g. driving
        a single variable to ±∞ because the coupling is large).

    **Why is dE/dx = -J*x - h?**
        E(x) = -0.5 * x^T J x - h^T x
        ∂E/∂x = -0.5 * (J + J^T) x - h = -J x - h    (since J is symmetric)
        Gradient *descent* means x ← x - lr * ∂E/∂x = x + lr * (J x + h),
        which is the "uphill in J" direction — exactly what minimises E.

    Args:
        model: ContinuousEBM to minimise.
        n_steps: Number of gradient descent steps.
        lr: Learning rate (step size).
        seed: Random seed for initial point (uniform in [-1, 1]^n).

    Returns:
        Array of shape (n,) with values in (-1, 1) representing the
        approximate energy minimiser.
    """
    rng = np.random.default_rng(seed)
    # Initialise randomly in [-1, 1]^n — avoids bias toward any particular basin
    x = rng.uniform(-1.0, 1.0, size=model.variables)

    J = model.coupling  # shape (n, n)
    h = model.bias      # shape (n,)

    for _ in range(n_steps):
        # Analytic gradient of E(x) = -0.5 x^T J x - h^T x
        # dE/dx = -J x - h  (descent direction: negate to go downhill)
        grad = -J @ x - h
        # Gradient step then squash: keeps x in open (-1, 1)^n hypercube
        x = np.tanh(x - lr * grad)

    return x


def compare_minima(
    ising_sample: np.ndarray,
    continuous_sample: np.ndarray,
) -> dict[str, float]:
    """Compare discrete and continuous energy minimisers.

    **Researcher summary:**
        Two metrics: L2 distance (magnitude) and sign agreement (direction).
        Both are needed because L2 alone can be misleading for sparse Ising
        problems where the ground state has many near-zero components.

    **Detailed explanation for engineers:**
        - ``l2_distance``: ||ising - continuous||_2.  Small means the two
          minimisers are numerically close.  Threshold in REQ-KONA-001: ≤ 0.1.
        - ``sign_agreement``: fraction of indices where sign(ising_i) ==
          sign(continuous_i).  Robust to scale differences; measures whether
          the solvers agree on the *direction* of each variable.  Threshold: > 0.9.

    Args:
        ising_sample: Array of shape (n,) from Ising simulated annealing.
            Values should be in {-1, +1} or nearby floats.
        continuous_sample: Array of shape (n,) from ``sample_continuous``.
            Values are in (-1, 1) due to tanh squashing.

    Returns:
        Dict with keys:
            ``'l2_distance'`` (float): Euclidean distance between the two samples.
            ``'sign_agreement'`` (float): Fraction of coordinates with matching sign.
    """
    ising_arr = np.asarray(ising_sample, dtype=np.float64)
    cont_arr = np.asarray(continuous_sample, dtype=np.float64)

    l2 = float(np.linalg.norm(ising_arr - cont_arr))

    # Sign agreement: compare np.sign, treating 0 as +1 (convention consistent
    # with Ising {-1, +1} variables — 0 is not a valid Ising spin).
    ising_signs = np.sign(ising_arr)
    ising_signs = np.where(ising_signs == 0, 1.0, ising_signs)
    cont_signs = np.sign(cont_arr)
    cont_signs = np.where(cont_signs == 0, 1.0, cont_signs)
    agreement = float(np.mean(ising_signs == cont_signs))

    return {"l2_distance": l2, "sign_agreement": agreement}


def build_kona_artifact(
    comparison: dict[str, float],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a JSON-serialisable artifact for Exp 435a.

    **Researcher summary:**
        Standardised artifact schema ``carnot.kona_seed.v1``.  The
        ``honest_verdict`` field is the primary result — downstream tooling
        should read this rather than re-deriving thresholds.

    **Verdict derivation (from REQ-KONA-001 / SCENARIO-KONA-002):**
        - ``'continuous_matches_ising'``: L2 < 0.1 AND sign_agreement > 0.9
        - ``'partial_match'``: sign_agreement > 0.7 (but not the above)
        - ``'failed_to_match'``: otherwise

    Args:
        comparison: Dict returned by ``compare_minima``.
        extra: Optional extra fields merged into the artifact (e.g. energy
            values, model spec).  Must be JSON-serialisable.

    Returns:
        Dict with at minimum:
            ``'schema'``, ``'run_date'``, ``'honest_verdict'``,
            ``'l2_distance'``, ``'sign_agreement'``.
    """
    l2 = comparison["l2_distance"]
    sa = comparison["sign_agreement"]

    if l2 < 0.1 and sa > 0.9:
        verdict = "continuous_matches_ising"
    elif sa > 0.7:
        verdict = "partial_match"
    else:
        verdict = "failed_to_match"

    artifact: dict[str, Any] = {
        "schema": "carnot.kona_seed.v1",
        "run_date": datetime.date.today().isoformat(),
        "honest_verdict": verdict,
        "l2_distance": l2,
        "sign_agreement": sa,
    }
    if extra:
        artifact.update(extra)
    return artifact
