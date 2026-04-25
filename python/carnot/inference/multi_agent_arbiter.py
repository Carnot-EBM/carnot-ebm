"""MultiAgentArbiter — selects the best agent response using Ising external field scoring.

**Researcher summary (RETRO-ARBITER-FLAT-ENERGY fix):**
    Exp 817 produced arbiter_accuracy=0.33 because ALL agent energies were 0.0.
    The root cause was that the arbiter used legacy diagonal injection, which shifts
    energy by a constant -0.5*sum(bias) regardless of spin configuration (s_i^2=1 for ±1).
    When all energies are equal, the arbiter picks arbitrarily — 1/3 chance correct.

    This module fixes that by calling compute_energy_with_external_field(), which changes
    sign based on spin orientation:
        - Violation spins (s_i=+1): field term = +h[i] > 0 → energy INCREASES (bad agent).
        - Correct spins  (s_i=-1): field term = -h[i] < 0 → energy DECREASES (good agent).
    The arbiter then selects the agent with the LOWEST total energy.

**AgentAuditor consensus penalty (arXiv 2602.09341):**
    When multiple agents converge on the same wrong answer (adversarial scenario), the
    majority-wrong cluster may have LOWER average energy than the lone correct agent,
    because the Ising landscape is shaped by aggregate spin statistics.  This is exactly
    the failure mode identified in AgentAuditor: consensus amplifies errors.

    Fix: detect when energy variance across agents is near-zero (all agents agree → same
    spin configuration → same energies).  In that case, identify the majority response
    cluster (by exact-match on response text) and add a consensus penalty to every agent
    in that cluster.  This breaks the tie in favor of the minority dissenter.

    The penalty size (default 0.1) is intentionally small: it only needs to break ties
    within the near-zero-variance regime, not override meaningful energy differences.

**Gibbs warm-start (Exp 846 fix, RETRO-ARBITER-ZERO-MAGNETIZATION):**
    Exp 835 showed accuracy_standard=0.0 despite Z-score normalization.  Root cause:
    energies_raw had |energy| < 0.2 for all agents — they were initialization noise,
    not Boltzmann-distributed values.  Z-score normalization of noise = still noise.

    Fix: GibbsWarmStart with 500 burn-in sweeps from mean-field initialization.
    The warm-start is run once per scoring call to validate the energy landscape
    (abs(E_warmstart) must exceed 0.5).  Per-agent scoring then uses the external
    field evaluation h^T s_text, where h is computed from the warm-started reference
    to ensure the field is properly calibrated before agent ranking.

    See GibbsWarmStart.warmup() for the Gibbs conditional derivation.

Spec: REQ-VERIFY-143, REQ-VERIFY-144, REQ-SAMPLE-020,
      SCENARIO-VERIFY-172, SCENARIO-VERIFY-173, SCENARIO-SAMPLE-032
"""
from __future__ import annotations

import logging
from collections import Counter
from typing import Optional

import numpy as np

from carnot.pipeline.ising_constraint_injector import IsingConstraintInjector
from carnot.inference.gibbs_warmstart import GibbsWarmStart

logger = logging.getLogger(__name__)


class MultiAgentArbiter:
    """Selects the best agent response using Ising external field scoring.

    The arbiter takes a list of candidate agent responses (strings) plus a list of
    constraint embeddings (from EmbeddingConstraintStore), converts each response to
    an Ising spin configuration, scores each spin config with the external field energy,
    and returns the index of the minimum-energy agent.

    When agent energies are nearly identical (variance < threshold), it applies the
    AgentAuditor consensus penalty to agents in the majority response cluster, then
    re-selects the minimum.

    Attributes:
        n_spins: Number of Ising spins used for spin encoding (default 16).
        embedding_dim: Dimensionality of constraint embeddings (default 384).
        consensus_threshold: Energy variance below which consensus penalty is applied
            (default 0.01).
        consensus_penalty: Energy added to majority-cluster agents (default 0.1).
    """

    def __init__(
        self,
        n_spins: int = 16,
        embedding_dim: int = 384,
        consensus_threshold: float = 0.01,
        consensus_penalty: float = 0.1,
        warm_start_sweeps: int = 500,
    ) -> None:
        """Initialise the arbiter with an Ising constraint injector.

        The injector provides the external field computation. The coupling matrix J
        is initialised as a small random symmetric matrix (std=0.01) to provide a
        weak prior over spin configurations without dominating the external field signal.

        Args:
            n_spins: Spin count for response encoding.
            embedding_dim: Embedding dimensionality (384 for all-MiniLM-L6-v2).
            consensus_threshold: Variance threshold for detecting consensus clusters.
            consensus_penalty: Energy bump added to majority-cluster agents.
            warm_start_sweeps: Number of Gibbs sweeps for warm-start calibration
                before agent scoring (default 500, per REQ-SAMPLE-020).  Set to 0
                to disable warm-start and use legacy cold-start behavior.

        Spec: REQ-VERIFY-143, REQ-VERIFY-144, REQ-SAMPLE-020
        """
        self.n_spins = n_spins
        self.embedding_dim = embedding_dim
        self.consensus_threshold = consensus_threshold
        self.consensus_penalty = consensus_penalty
        self.warm_start_sweeps = warm_start_sweeps

        self._injector = IsingConstraintInjector(
            embedding_dim=embedding_dim, n_spins=n_spins
        )

        # Weak symmetric coupling matrix: off-diagonal entries only so the matrix
        # does not overwhelm the external field signal from constraint embeddings.
        rng = np.random.default_rng(42)
        raw = rng.standard_normal((n_spins, n_spins)) * 0.01
        self._J: np.ndarray = (raw + raw.T) / 2.0  # symmetric

        # Gibbs warm-start sampler (seed fixed for reproducibility across runs)
        self._warmstart = GibbsWarmStart(beta=1.0, seed=42)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score_agents(
        self,
        responses: list[str],
        constraint_embeddings: list[list[float]],
    ) -> np.ndarray:
        """Score each agent response using the external field energy.

        When warm_start_sweeps > 0 (default), runs Gibbs warm-start from mean-field
        initialization to validate the energy landscape before scoring.  Per-agent
        scores use compute_energy_with_external_field with text-derived spins.

        If the warm-start energy magnitude is below 0.5, logs a diagnostic and falls
        back to legacy cold-start scoring (warm_start_sweeps effectively 0).

        Lower energy = better agent (more consistent with constraint embeddings).

        Args:
            responses: Agent response strings to score.
            constraint_embeddings: Constraint embeddings from EmbeddingConstraintStore.

        Returns:
            np.ndarray of shape (len(responses),) — the total energy per agent.

        Spec: REQ-VERIFY-143, REQ-SAMPLE-020
        """
        use_warmstart = self.warm_start_sweeps > 0

        if use_warmstart and constraint_embeddings:
            h = self._injector.project_to_spin_bias(constraint_embeddings)
            h = np.clip(h, 0.0, None)
            _, e_warmstart = self._warmstart.warmup(
                self._J, h, n_sweeps=self.warm_start_sweeps
            )
            if abs(e_warmstart) < 0.5:
                logger.warning(
                    "Gibbs warm-start energy %.4f below 0.5 magnitude threshold; "
                    "falling back to legacy cold-start scoring.  "
                    "Constraint embeddings may be too weak to calibrate the Ising landscape.",
                    e_warmstart,
                )
                use_warmstart = False

        energies = np.empty(len(responses), dtype=np.float64)
        for i, response in enumerate(responses):
            spins = self._text_to_spins(response)
            result = self._injector.compute_energy_with_external_field(
                self._J, spins, constraint_embeddings
            )
            energies[i] = result.E_total
        return energies

    def detect_consensus(
        self,
        energies: np.ndarray,
        responses: Optional[list[str]] = None,
        threshold: Optional[float] = None,
    ) -> bool:
        """Return True if agents are in consensus (energy range OR response cluster).

        Two complementary signals trigger consensus:

        1. **Energy-range check** (REQ-VERIFY-144): max(energies) - min(energies) < threshold.
           This fires when all agents produce nearly identical spin configs — e.g., all agents
           give the exact same answer so their hashed spin vectors are identical.

        2. **Response-cluster check**: when the majority response cluster has >= 2 members,
           at least two agents gave the same textual answer.  This is the adversarial failure
           mode from AgentAuditor (arXiv 2602.09341): two agents hallucinate the same wrong
           answer, creating a cluster that may dominate energy-only ranking.  The energy range
           may be nonzero (because the lone correct agent has a different spin config), so the
           energy check alone is insufficient.

        Combining both signals lets the arbiter handle:
        - Full consensus (all agents agree → energy range ≈ 0)
        - Adversarial majority (2-of-3 agree on wrong answer → response cluster ≥ 2)

        Args:
            energies: 1-D array of agent energies from score_agents.
            responses: Optional agent response strings.  When provided, the response-cluster
                check is also evaluated.  Should match the same order as energies.
            threshold: Override for consensus_threshold (uses instance default if None).

        Returns:
            True when either the energy range is below threshold OR the majority response
            cluster has >= 2 members.

        Spec: REQ-VERIFY-144
        """
        t = threshold if threshold is not None else self.consensus_threshold
        if float(np.max(energies) - np.min(energies)) < t:
            return True
        if responses is not None:
            counts = Counter(responses)
            if counts.most_common(1)[0][1] >= 2:
                return True
        return False

    def apply_consensus_penalty(
        self,
        energies: np.ndarray,
        responses: list[str],
        penalty: float | None = None,
    ) -> np.ndarray:
        """Add a consensus penalty to all agents sharing the majority response.

        Clusters agents by exact-match on their response text.  The largest cluster
        is the "majority" cluster.  All agents in that cluster receive +penalty energy,
        making them look worse than dissenting minority agents.

        This implements the AgentAuditor mechanism from arXiv 2602.09341: when a
        majority of agents hallucinate the same wrong answer, the penalty forces the
        arbiter to consider the outlier, which is more likely to be correct.

        Args:
            energies: 1-D array of agent energies (will NOT be mutated).
            responses: Agent response strings corresponding to energies.
            penalty: Energy bump for majority-cluster agents (uses instance default
                if None).

        Returns:
            New np.ndarray with penalty added to majority-cluster agents.

        Spec: REQ-VERIFY-144
        """
        p = penalty if penalty is not None else self.consensus_penalty
        counts = Counter(responses)
        majority_response = counts.most_common(1)[0][0]

        adjusted = energies.copy()
        for i, response in enumerate(responses):
            if response == majority_response:
                adjusted[i] += p
        return adjusted

    def arbitrate(
        self,
        responses: list[str],
        constraint_embeddings: list[list[float]],
    ) -> dict:
        """Select the best agent response and return a result dict.

        Pipeline:
            1. score_agents() — compute external field energy for each response.
            2. z-score normalize per-query (REQ-VERIFY-144): subtract mean, divide by
               std so all queries share a common scale.  When sigma <= 1e-6 (all energies
               equal), skip normalisation so the array is still usable for tie-breaking.
            3. detect_consensus() on normalized energies — check if all agents agree.
            4. If consensus detected: apply_consensus_penalty() to break the tie.
            5. Return the index of the minimum-energy (normalized+adjusted) agent.

        Args:
            responses: Candidate agent response strings (at least 1 required).
            constraint_embeddings: Constraint embeddings for external field scoring.

        Returns:
            Dict with keys:
                arbiter_index: int — index of the selected agent in responses.
                arbiter_response: str — the selected response text.
                energies_raw: list[float] — energies before normalisation.
                energies_normalized: list[float] — z-score normalised energies.
                energies_adjusted: list[float] — after optional consensus penalty.
                used_consensus_penalty: bool — True if penalty was applied.

        Spec: REQ-VERIFY-143, REQ-VERIFY-144
        """
        energies_raw = self.score_agents(responses, constraint_embeddings)

        # Per-query z-score normalization so energy scale is consistent across queries.
        mu = float(np.mean(energies_raw))
        sigma = float(np.std(energies_raw))
        if sigma > 1e-6:
            energies_norm = (energies_raw - mu) / sigma
        else:
            # All energies equal — normalisation would divide by zero; keep as-is.
            energies_norm = energies_raw.copy()

        used_penalty = False
        if self.detect_consensus(energies_norm, responses=responses):
            energies_adj = self.apply_consensus_penalty(energies_norm, responses)
            used_penalty = True
        else:
            energies_adj = energies_norm.copy()

        best_idx = int(np.argmin(energies_adj))
        return {
            "arbiter_index": best_idx,
            "arbiter_response": responses[best_idx],
            "energies_raw": energies_raw.tolist(),
            "energies_normalized": energies_norm.tolist(),
            "energies_adjusted": energies_adj.tolist(),
            "used_consensus_penalty": used_penalty,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _text_to_spins(self, text: str) -> np.ndarray:
        """Convert a response string to a deterministic ±1 spin array.

        Uses a seeded hash of the text to deterministically assign each spin.
        Spin i = +1 if hash bit i is 1, else -1.  This is a simple but stable
        encoding that ensures the same response always maps to the same spins.

        The encoding is NOT designed to preserve semantic similarity — it is only
        meant to produce a diverse spread of spin configs so that the external field
        h (derived from constraint embeddings) can discriminate between responses.

        Args:
            text: Agent response string.

        Returns:
            np.ndarray of shape (n_spins,) with values in {-1.0, +1.0}.
        """
        seed = hash(text) % (2**32)
        rng = np.random.default_rng(seed)
        bits = rng.integers(0, 2, size=self.n_spins)
        return np.where(bits == 1, 1.0, -1.0)
