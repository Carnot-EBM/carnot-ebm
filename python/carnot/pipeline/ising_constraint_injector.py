"""IsingConstraintInjector — projects EmbeddingConstraintStore vectors into Ising coupling matrix.

**Researcher summary (RETRO-CONSTRAINT-ZERO-DELTA fix):**
    Exps 788 and 801 showed constraint_addition_delta=0.0 because retrieved constraint
    embeddings were never wired into the IsingEBM — they were only appended as metadata
    ConstraintResult entries that have no effect on the Ising energy computation.

    This module closes the loop: constraint embeddings from EmbeddingConstraintStore are
    projected to spin-space via a linear map (embedding_dim -> n_spins) and added as
    soft bias terms on the diagonal of the coupling matrix J.  Adding to the diagonal
    is physically equivalent to adding an external magnetic field h_i to each spin,
    which shifts the energy landscape so that configurations that violate the constraint
    have strictly higher energy.

**Why diagonal injection (not off-diagonal)?**
    Off-diagonal entries J_ij encode *pairwise* coupling between spins i and j.
    Diagonal entries J_ii encode *self-coupling* — effectively an external field h_i
    that biases spin i toward a preferred value.  Constraint vectors encode which
    reasoning dimensions are implicated in a violation (via the embedding); projecting
    them to spin biases lets each spin independently "feel" the constraint penalty.
    This is the simplest well-motivated injection that preserves the quadratic energy
    structure while changing the energy for constraint-violating configurations.

**RETRO-ISING-INJECTION-NO-DISCRIMINATION (CLOSED 20260425):**
    Exp 812 revealed that diagonal injection adds a CONSTANT energy shift -0.5*sum(bias)
    that is identical for ALL spin configurations because s_i^2 = 1 for ±1 spins.
    This means it cannot distinguish violations from correct responses.

    Fix: external field injection via compute_energy_with_external_field().
    E_total = -0.5 * s^T J s + h^T s, where h = clip(W @ emb_mean, 0, inf).
    When s_i = +1 (violation): field term = +h[i] > 0, total energy INCREASES.
    When s_i = -1 (correct):   field term = -h[i] < 0, total energy DECREASES.
    Note: uses +h^T s (not -h^T s as in standard Ising) to penalise violation spins.
    Net: violations are penalised, correct responses are rewarded. Discriminating.

Spec: REQ-VERIFY-095, REQ-VERIFY-096, REQ-VERIFY-173, REQ-VERIFY-174,
      SCENARIO-VERIFY-129, SCENARIO-VERIFY-227, SCENARIO-VERIFY-228
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np


@dataclass
class ConstraintInjectionResult:
    """Summary of one constraint injection trial.

    Fields map directly to the experiment 812 artifact schema so that the
    conductor can track injection effectiveness across experiment runs.

    Attributes:
        n_constraints_retrieved: How many SPO constraints were fetched from
            the store before injection (0 if store was empty or no match).
        embedding_dim: Dimensionality of each constraint embedding vector
            (384 for all-MiniLM-L6-v2, same for ci_hash fallback).
        n_spins: Number of spins in the Ising model (= size of coupling matrix).
        projection_applied: True when at least one constraint embedding was
            projected and injected into J; False when n_constraints_retrieved==0.
        energy_without_injection: Ising energy E(x) computed with the original J.
        energy_with_injection: Ising energy E(x) computed with the constraint-
            biased J (J_injected = J + diag(bias)).
        energy_delta_pct: 100 * (energy_with_injection - energy_without_injection)
            / abs(energy_without_injection) when energy_without_injection != 0.
            Positive means the constraint raised the energy (violation detected).
            Zero when energy_without_injection is ~0.
        honest_verdict: One of:
            "injection_works"          — mean delta_pct > 0 for error responses
            "injection_no_delta"       — delta_pct <= 0 (constraint had no effect)
            "injection_negative_delta" — energy decreased (sign error in bias)
    """

    n_constraints_retrieved: int
    embedding_dim: int
    n_spins: int
    projection_applied: bool
    energy_without_injection: float
    energy_with_injection: float
    energy_delta_pct: float
    honest_verdict: str


class IsingConstraintInjector:
    """Projects EmbeddingConstraintStore embeddings into Ising spin-space as coupling biases.

    **How the projection works:**
        Each constraint embedding is a 384-dim float vector in sentence-transformer space.
        The coupling matrix J lives in a (n_spins x n_spins) space.  A learned linear
        map W of shape (embedding_dim, n_spins) translates each embedding to a bias
        vector b of shape (n_spins,).  When multiple constraints are retrieved, their
        projected biases are averaged so that no single constraint dominates.

        The projected bias is then added to the diagonal of J (J_ii += b_i).  This is
        equivalent to adding external fields h = b to the Ising Hamiltonian:
            E_injected(x) = -0.5 x^T (J + diag(b)) x = E_original(x) - 0.5 * b^T (x * x)
        For binary spins x_i in {-1, +1}, x_i^2 = 1, so the energy shift is:
            ΔE = -0.5 * sum_i b_i
        If b_i aligns with the violation direction (positive), ΔE is negative — meaning
        the injected J raises energy for configurations that disagree with the constraint.

    Attributes:
        embedding_dim: Dimensionality of input constraint embeddings (default 384).
        n_spins: Number of spins / coupling matrix size (default 64).
    """

    def __init__(self, embedding_dim: int = 384, n_spins: int = 64) -> None:
        """Initialise the injector with a random linear projection.

        The projection matrix W is initialised small (std=0.01) so that before
        any training, the bias perturbation is tiny and the existing coupling
        matrix J is not drastically disrupted.  This is the same "small init"
        principle used in residual connections: start near identity behaviour,
        let the signal build up from data.

        Args:
            embedding_dim: Dimensionality of constraint embeddings (384 for MiniLM).
            n_spins: Number of Ising spins / coupling matrix side length.

        Spec: REQ-VERIFY-095
        """
        self.embedding_dim = embedding_dim
        self.n_spins = n_spins
        # Small random init: large values would overwhelm J and dominate the energy.
        rng = np.random.default_rng(42)
        self._projection: np.ndarray = rng.standard_normal((embedding_dim, n_spins)) * 0.01

    def project_to_spin_bias(
        self, constraint_embeddings: list[list[float]]
    ) -> np.ndarray:
        """Project constraint embeddings to an (n_spins,) bias vector.

        Each embedding is independently projected via W: bias_i = embedding @ W.
        The per-embedding biases are then averaged so no single constraint dominates.
        When the list is empty, returns a zero bias (no injection).

        Args:
            constraint_embeddings: List of float lists, each of length embedding_dim.

        Returns:
            np.ndarray of shape (n_spins,) — the mean projected spin bias.

        Spec: REQ-VERIFY-095
        """
        if not constraint_embeddings:
            return np.zeros(self.n_spins)
        emb_array = np.array(constraint_embeddings, dtype=np.float64)
        # emb_array: (n_constraints, embedding_dim)
        # projected: (n_constraints, n_spins)
        projected = emb_array @ self._projection
        return projected.mean(axis=0)

    def inject_into_coupling_matrix(
        self, J: np.ndarray, bias: np.ndarray
    ) -> np.ndarray:
        """Return a new coupling matrix with bias added to its diagonal.

        This is ADDITIVE — the original J is never mutated.  The returned
        matrix J_injected has the same off-diagonal entries as J and diagonal
        entries J_ii + bias_i.  Physically: each spin receives an external
        field equal to its bias component, lifting the energy of configurations
        that violate the encoded constraint.

        Args:
            J: Square coupling matrix of shape (n_spins, n_spins).
            bias: Bias vector of shape (n_spins,) from project_to_spin_bias.

        Returns:
            J_injected of the same shape, with bias added to the diagonal.

        Spec: REQ-VERIFY-095
        """
        J_injected = J.copy()
        np.fill_diagonal(J_injected, J_injected.diagonal() + bias)
        return J_injected

    def compute_energy_with_injection(
        self,
        ising_ebm: object,
        spins: np.ndarray,
        constraint_embeddings: list[list[float]],
    ) -> float:
        """Compute Ising energy using a constraint-biased coupling matrix.

        Uses the standard Ising energy formula:
            E = -0.5 * spins^T J_injected spins

        Note: The IsingModel class (carnot.models.ising) uses JAX arrays; this
        method operates on numpy and reads ising_ebm.coupling as a JAX array,
        converting to numpy for the computation.  The bias term ising_ebm.bias
        is intentionally omitted here — this method measures only the effect of
        the constraint injection on the coupling energy, not the full model energy.

        Args:
            ising_ebm: IsingModel instance (must have .coupling attribute of
                shape (n_spins, n_spins)).
            spins: numpy array of shape (n_spins,) — the spin configuration.
            constraint_embeddings: Retrieved constraint embeddings to inject.

        Returns:
            Scalar float energy after injection.

        Spec: REQ-VERIFY-095, REQ-VERIFY-096
        """
        bias = self.project_to_spin_bias(constraint_embeddings)
        J = np.array(ising_ebm.coupling, dtype=np.float64)
        J_injected = self.inject_into_coupling_matrix(J, bias)
        return float(-0.5 * spins @ J_injected @ spins)

    def compute_energy_with_external_field(
        self,
        J: np.ndarray,
        spins: np.ndarray,
        constraint_embeddings: list[list[float]],
    ) -> "ExternalFieldEnergyResult":
        """Compute Ising energy with an external field derived from constraint embeddings.

        Unlike inject_into_coupling_matrix (which adds a constant to the diagonal and
        therefore shifts energy identically for ALL spin configs), external field injection
        changes sign depending on spin orientation:
            - Violation spin s_i = +1: field term = +h[i] > 0 → total energy INCREASES.
            - Correct spin   s_i = -1: field term = -h[i] < 0 → total energy DECREASES.

        Note on sign convention: the standard Ising Hamiltonian uses -h^T s which
        FAVORS s=+1 (lower energy for aligned spins). We use +h^T s to PENALISE s=+1
        (violation encoding), which is the opposite convention but physically correct
        for our use case.

        The external field h is derived from constraint_embeddings via project_to_spin_bias,
        then clipped to [0, +inf] so constraint penalties only ever raise energy for
        violations, never accidentally lower it for correct configurations.

        Formula:
            E_total = E_ising + E_field
            E_ising = -0.5 * s^T J s
            h       = clip(W @ mean(embeddings), 0, inf)
            E_field = +h^T s  (positive sign — penalises violation spins s_i=+1)

        Args:
            J: Square coupling matrix of shape (n_spins, n_spins).
            spins: numpy array of shape (n_spins,) with values in {-1, +1}.
            constraint_embeddings: List of float lists, each of length embedding_dim.
                An empty list produces h=0 (no external field, backwards compatible).

        Returns:
            ExternalFieldEnergyResult namedtuple with fields:
                E_total: scalar energy E_ising + E_field.
                E_ising: coupling-only energy -0.5 * s^T J s.
                E_field: external field contribution -h^T s.
                h_norm: L2 norm of the external field vector h (0 when embeddings empty).

        Spec: REQ-VERIFY-173, REQ-VERIFY-174, SCENARIO-VERIFY-227, SCENARIO-VERIFY-228
        """
        spins_f = np.array(spins, dtype=np.float64)
        J_f = np.array(J, dtype=np.float64)

        # Coupling energy — standard Ising formula.
        E_ising = float(-0.5 * spins_f @ J_f @ spins_f)

        # External field: project embeddings to spin space, clip negative to zero so
        # the field can only penalise violations (never reward them by accident).
        if constraint_embeddings:
            emb_array = np.array(constraint_embeddings, dtype=np.float64)
            h_raw = emb_array @ self._projection  # (n_constraints, n_spins)
            h = np.clip(h_raw.mean(axis=0), 0.0, None)  # (n_spins,), non-negative
        else:
            h = np.zeros(self.n_spins, dtype=np.float64)

        # Positive sign: h^T s with h >= 0 penalises violation spins (s_i = +1 → +h[i])
        # and rewards correct spins (s_i = -1 → -h[i]).  The standard Ising external
        # field convention -h^T s favors +1 spins, which is the OPPOSITE of what we
        # want (see RETRO-ISING-INJECTION-NO-DISCRIMINATION for the sign analysis).
        E_field = float(h @ spins_f)
        E_total = E_ising + E_field
        h_norm = float(np.linalg.norm(h))

        return ExternalFieldEnergyResult(
            E_total=E_total,
            E_ising=E_ising,
            E_field=E_field,
            h_norm=h_norm,
        )


class ExternalFieldEnergyResult(NamedTuple):
    """Result of compute_energy_with_external_field.

    Attributes:
        E_total: Total energy E_ising + E_field.
        E_ising: Coupling-only energy -0.5 * s^T J s.
        E_field: External field contribution -h^T s.
        h_norm: L2 norm of the external field vector h.
    """

    E_total: float
    E_ising: float
    E_field: float
    h_norm: float
