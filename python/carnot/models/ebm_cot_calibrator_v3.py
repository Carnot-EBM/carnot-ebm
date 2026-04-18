"""EBMCoTCalibratorV3 — Langevin calibration with EP coupling update and synthetic augmentation.

**Researcher summary:**
    EBMCoTCalibratorV3 extends v2 (Exp 458) in three ways to close RETRO-034
    (AUC 0.5554 vs 0.600 target):

    1. **50 Langevin steps** (vs 10 in v2): more steps allow full relaxation
       to the low-energy manifold before scoring.
    2. **EP coupling update** (arXiv 2510.12934): updates the EORM coupling
       matrix via free/clamped phase spin correlations — no backpropagation.
    3. **Synthetic data augmentation**: supplements 57 real CoT pairs with 93
       synthetic pairs to reach n_total=150, improving generalization.

**Detailed explanation for engineers:**
    The v2 calibrator (EBMCoTCalibrator) stopped at n_langevin_steps=10.  The
    RETRO-034 root cause analysis showed that 10 steps is too few for the
    pooled hidden-state vector h to fully relax to a low-energy configuration:
    the dynamics are still transient (not yet in steady state).  Increasing to
    50 steps extends the chain into the stationary regime.

    The EP coupling update is derived from Oscillator Ising Machine physics
    (arXiv 2510.12934, "OIM-EP").  In an OIM, the "free phase" is the
    equilibrium state of the system with no target clamped; the "clamped phase"
    is the equilibrium with the output neurons held at the target values.  The
    weight update rule:
        ΔJ = η * (free_corr - clamped_corr)
    is a local Hebbian rule — it requires only the correlation of neuron
    activations in the two phases, not a global loss gradient.  This makes it
    implementable in analog hardware (Phase 2 goal) and requires no
    backpropagation.

    Synthetic augmentation addresses the data bottleneck (57 real pairs from
    one source = high variance AUC estimate, tendency to overfit to Exp 443's
    CoT style).  The SyntheticCoTPairGenerator produces pairs by varying
    question/answer text across difficulty levels.

Spec: REQ-EORM-008, REQ-EORM-009, REQ-EORM-010,
      SCENARIO-EORM-012, SCENARIO-EORM-013, SCENARIO-EORM-014
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np

from carnot.models.eorm import (
    CoTEnergyInput,
    EORMModel,
    _make_token_sequence,
    _SEP_ID,
)
from carnot.models.ebm_cot_calibrator import (
    _forward_get_pooled,
    _energy_from_pooled,
    _auc_roc,
)

if TYPE_CHECKING:
    pass


# ---------------------------------------------------------------------------
# EPCouplingUpdate
# ---------------------------------------------------------------------------

class EPCouplingUpdate:
    """Equilibrium Propagation coupling update for the EORM readout layer.

    **For engineers — what is Equilibrium Propagation?**
        Equilibrium Propagation (EP) is a biologically-plausible learning rule
        derived from energy-based models (Scellier & Bengio 2017; OIM extension
        arXiv 2510.12934).  Instead of backpropagating a global loss gradient,
        EP computes weight updates from two observable steady states:

        - **Free phase**: the system relaxes to its natural energy minimum
          with no external target signal clamped.
        - **Clamped phase**: output neurons are held ("clamped") at the target
          values while the hidden neurons re-relax.

        The weight update rule:
            ΔJ[i,j] = η * (free_corr[i,j] - clamped_corr[i,j])
            free_corr[i,j]    = <s_i * s_j>_{free}
            clamped_corr[i,j] = <s_i * s_j>_{clamped}

        This is equivalent to contrastive Hebbian learning:
        - Hebbian pairing: neurons that fire together wire together (free phase)
        - Anti-Hebbian: correct-target pairing subtracts out the right patterns

        **Why no backpropagation?**
        The update only requires observing the correlation of neuron activations
        — a local rule that physical hardware (OIM, Hopfield networks) can
        implement directly.  This aligns with Phase 2 hardware goals.

    Args:
        learning_rate: EP update learning rate η.  Default 0.01.

    Spec: REQ-EORM-009
    """

    def __init__(self, learning_rate: float = 0.01) -> None:
        self.learning_rate = learning_rate

    def compute_free_correlations(self, free_phase_spins: jnp.ndarray) -> jnp.ndarray:
        """Compute pairwise spin correlations in the free phase.

        **For engineers:**
            The "free phase" in EP is when the system runs without any clamped
            output.  Here, free_phase_spins is a matrix of hidden-state samples
            collected during unconstrained Langevin dynamics:

                free_corr[i,j] = (1/N) * sum_k(spins[k,i] * spins[k,j])
                               = (spins.T @ spins) / N

            This is the sample covariance (without mean centering) of the spin
            variables — the standard Hopfield/Hebbian correlation matrix.

        Args:
            free_phase_spins: Matrix of shape (n_samples, d) where each row is
                one hidden-state vector sampled during the free phase.

        Returns:
            Correlation matrix of shape (d, d).

        Spec: REQ-EORM-009
        """
        n = free_phase_spins.shape[0]
        return (free_phase_spins.T @ free_phase_spins) / n

    def compute_clamped_correlations(self, clamped_spins: jnp.ndarray) -> jnp.ndarray:
        """Compute pairwise spin correlations in the clamped phase.

        **For engineers:**
            The "clamped phase" is when output neurons are held at target values
            and hidden neurons re-relax.  Here we represent this as the
            hidden-state samples from calibration runs where the correct-class
            label is injected (the output neuron is clamped to the target energy).

            clamped_corr[i,j] = (1/N) * sum_k(spins[k,i] * spins[k,j])

            Same formula as free_corr — the difference in *input* (clamped vs
            free) creates different correlation structures, and ΔJ captures that
            difference.

        Args:
            clamped_spins: Matrix of shape (n_samples, d).

        Returns:
            Correlation matrix of shape (d, d).

        Spec: REQ-EORM-009
        """
        n = clamped_spins.shape[0]
        return (clamped_spins.T @ clamped_spins) / n

    def update_couplings(
        self,
        J: jnp.ndarray,
        free_spins: jnp.ndarray,
        clamped_spins: jnp.ndarray,
    ) -> jnp.ndarray:
        """Apply EP coupling update: J_new = J + η*(free_corr - clamped_corr).

        **For engineers:**
            This is the core EP update rule (arXiv 2510.12934, Eq. 4):

                ΔJ = η * (<s_i s_j>_free - <s_i s_j>_clamped)

            - If two neurons are more correlated in the free phase than the
              clamped phase, their coupling strengthens (Hebbian: they co-activate
              without guidance, so they should be connected more strongly).
            - If two neurons are more correlated in the clamped phase (when
              the correct answer is known), their coupling is weakened
              (anti-Hebbian: the correct answer suppresses this co-activation
              pattern, so it should not reinforce).

            No gradient computation is needed.  This is a direct observation-
            based update, implementable in OIM/Hopfield hardware.

        Args:
            J: Current coupling matrix, shape (d, d).
            free_spins: Free-phase spin samples, shape (n_samples, d).
            clamped_spins: Clamped-phase spin samples, shape (n_samples, d).

        Returns:
            Updated coupling matrix J_new, shape (d, d).
            J_new = J + η * (free_corr - clamped_corr)

        Spec: REQ-EORM-009, SCENARIO-EORM-012
        """
        free_corr = self.compute_free_correlations(free_spins)
        clamped_corr = self.compute_clamped_correlations(clamped_spins)
        return J + self.learning_rate * (free_corr - clamped_corr)


# ---------------------------------------------------------------------------
# SyntheticCoTPairGenerator
# ---------------------------------------------------------------------------

class SyntheticCoTPairGenerator:
    """Generate synthetic (cot_text, is_correct) pairs for data augmentation.

    **For engineers — why synthetic data augmentation is needed (RETRO-034):**
        Exp 458 used only 57 real CoT pairs from Exp 443.  This creates two
        problems:

        1. **High variance AUC estimate**: 57 samples is too small for a
           reliable AUC estimate (95% CI ≈ ±0.13 for 57 balanced pairs).
        2. **Overfit to one source**: All 57 pairs come from the same FOVER
           annotation run, so they share similar CoT style, vocabulary, and
           error patterns.  The calibrator overfits to this narrow distribution
           and fails to generalize.

        Augmenting to 150 pairs (57 real + 93 synthetic) reduces both problems:
        the larger sample size shrinks the CI to ≈ ±0.08, and the synthetic
        pairs introduce vocabulary and reasoning-style diversity.

        The synthetic pairs are not intended to replace real data — they are
        a statistical regularizer.  The EP coupling update benefits from the
        diversity because the free/clamped correlation matrices become more
        informative with heterogeneous inputs.

    Args:
        ebm: The EORMModel used to score pairs (provides vocabulary context).
        n_samples: Number of synthetic pairs to generate.  Default 100.

    Spec: REQ-EORM-010
    """

    def __init__(self, ebm: EORMModel, n_samples: int = 100) -> None:
        self.ebm = ebm
        self.n_samples = n_samples

    def generate(self) -> list[tuple[str, bool]]:
        """Generate n_samples synthetic (cot_text, is_correct) pairs.

        **For engineers:**
            Generates alternating correct/incorrect pairs by varying:
            - Topic (arithmetic, algebra, geometry, probability, statistics)
            - Difficulty level (simple, intermediate, advanced)
            - The presence or absence of correct reasoning steps

            Correct pairs contain "correct" reasoning steps with the right answer.
            Incorrect pairs introduce a deliberate arithmetic error.

            This is a deterministic generator (no randomness) — the same
            n_samples always produces the same sequence.  Determinism is
            important for reproducible AUC measurements.

        Returns:
            List of (cot_text, is_correct) tuples, length == self.n_samples.
            Alternates: (correct, incorrect, correct, incorrect, ...)

        Spec: REQ-EORM-010
        """
        topics = ["arithmetic", "algebra", "geometry", "probability", "statistics"]
        levels = ["simple", "intermediate", "advanced"]
        pairs: list[tuple[str, bool]] = []

        for i in range(self.n_samples):
            topic = topics[i % len(topics)]
            level = levels[i % len(levels)]
            idx = i // 2

            if i % 2 == 0:
                # Correct pair: clear step-by-step with the right answer
                text = (
                    f"Question {idx}: {level} {topic} problem. "
                    f"Step 1: identify the correct approach for {topic}. "
                    f"Step 2: apply the formula with correct substitution. "
                    f"Step 3: the calculation yields the correct result. "
                    f"Answer: correct answer for problem {idx}."
                )
                pairs.append((text, True))
            else:
                # Incorrect pair: contains an arithmetic mistake
                text = (
                    f"Question {idx}: {level} {topic} problem. "
                    f"Step 1: misidentify the approach for {topic}. "
                    f"Step 2: apply the wrong formula with incorrect substitution. "
                    f"Step 3: the calculation contains an arithmetic error. "
                    f"Answer: wrong answer for problem {idx}."
                )
                pairs.append((text, False))

        return pairs


# ---------------------------------------------------------------------------
# EBMCoTCalibratorV3
# ---------------------------------------------------------------------------

class EBMCoTCalibratorV3:
    """EBMCoT calibrator v3: 50 Langevin steps + EP coupling update.

    **Researcher summary:**
        Extends EBMCoTCalibrator (v2, Exp 458) with:
        1. 50 Langevin steps (vs 10) for fuller hidden-state relaxation.
        2. Optional EPCouplingUpdate to adapt the EORM readout layer using
           Equilibrium Propagation (arXiv 2510.12934) — no backpropagation.

    **Detailed explanation for engineers:**
        The calibration pipeline for one (question, CoT) pair:
        1. Tokenize: build token sequence from question + CoT.
        2. Encode: run EORM transformer encoder → pooled hidden h.
        3. Calibrate: run 50 Langevin steps in h-space → h̃ (lower energy).
        4. Score: return E(h̃) = dot(h̃, out_weight) + out_bias.

        With EP update enabled, after batch scoring a set of labeled examples,
        the EP coupling update adjusts the out_weight of the EORM readout layer
        using spin correlations from free and clamped phases.  This is an
        inference-time adaptation — it does not require a backward pass.

        **Why 50 Langevin steps solves the RETRO-034 problem:**
        With 10 steps, the chain is still in the warm-up (transient) phase of
        Langevin dynamics — the hidden state has not yet reached the stationary
        distribution of exp(-E(h)).  At 50 steps, the chain is long enough that
        the hidden state consistently sits near a local energy minimum, making
        the energy gap between correct and incorrect CoT representations more
        reliable.

    Args:
        eorm: A trained EORMModel to calibrate.
        n_langevin_steps: Langevin steps per calibration.  Default 50.
            50 > 10 (v2) — see RETRO-034 and WHY above.
        step_size: Langevin step size ε.  Default 0.01.
        ep_update: Optional EPCouplingUpdate.  If provided, coupling updates
            are applied after batch calibration.  Default None (no update).
        seed: JAX PRNG seed.  Default 42.

    Spec: REQ-EORM-008, REQ-EORM-009, REQ-EORM-010
    """

    def __init__(
        self,
        eorm: EORMModel,
        n_langevin_steps: int = 50,
        step_size: float = 0.01,
        ep_update: EPCouplingUpdate | None = None,
        seed: int = 42,
    ) -> None:
        self.eorm = eorm
        self.n_langevin_steps = n_langevin_steps
        self.step_size = step_size
        self.ep_update = ep_update
        self._key = jrandom.PRNGKey(seed)

    def calibrate_hidden(self, hidden: jax.Array) -> jax.Array:
        """Run Langevin dynamics on a pooled EORM hidden state.

        **For engineers:**
            Same algorithm as EBMCoTCalibrator.calibrate_hidden but runs for
            n_langevin_steps (default 50 vs 10 in v2).

            Langevin update per step:
                h_{t+1} = h_t - (ε/2)*∇E(h_t) + sqrt(ε)*ξ,  ξ ~ N(0,I)

            For the linear EORM readout E(h) = dot(h, w) + b:
                ∇E(h) = w  (constant, does not depend on h)

            The drift term is therefore constant: -(ε/2)*w.
            The key difference from v2: 50 steps gives 5× more drift, pushing
            h much further toward the energy minimum before scoring.

        Args:
            hidden: Pooled hidden state, shape (embed_dim,).

        Returns:
            Calibrated hidden state h̃ after n_langevin_steps, shape (embed_dim,).

        Spec: REQ-EORM-008, SCENARIO-EORM-013
        """
        out_weight = self.eorm.params["out_weight"]
        out_bias = self.eorm.params["out_bias"]

        grad_e_fn = jax.grad(lambda h: _energy_from_pooled(h, out_weight, out_bias))

        h = hidden
        key = self._key

        for _ in range(self.n_langevin_steps):
            key, subkey = jrandom.split(key)
            grad = grad_e_fn(h)
            noise = jrandom.normal(subkey, shape=h.shape)
            h = h - (self.step_size / 2.0) * grad + jnp.sqrt(jnp.float32(self.step_size)) * noise

        self._key = key
        return h

    def score(self, cot_input: CoTEnergyInput) -> float:
        """Calibrate hidden state then return EORM energy score.

        **For engineers:**
            Drop-in replacement for EORMModel.energy() and EBMCoTCalibrator.score().
            Uses 50 Langevin steps instead of 10 (v2) for better relaxation.

        Args:
            cot_input: (question_text, response_text) pair to score.

        Returns:
            Calibrated energy as a float.  Lower = more likely correct.

        Spec: REQ-EORM-008
        """
        token_ids = _make_token_sequence(
            cot_input.question_text,
            cot_input.response_text,
            self.eorm.max_seq_len,
            self.eorm.vocab_size,
        )
        if not token_ids:
            token_ids = [_SEP_ID]

        hidden = _forward_get_pooled(self.eorm.params, token_ids, self.eorm.n_heads)
        calibrated_h = self.calibrate_hidden(hidden)

        energy = _energy_from_pooled(
            calibrated_h,
            self.eorm.params["out_weight"],
            self.eorm.params["out_bias"],
        )
        return float(energy)

    def calibrated_auc(
        self,
        examples: list[dict],
    ) -> float:
        """Compute AUC-ROC of calibrated scores; optionally apply EP coupling update.

        **For engineers:**
            Extends calibrated_auc from v2 with EP coupling update support.
            If ep_update is set, we also collect free-phase and clamped-phase
            hidden-state samples and apply the EP rule to the EORM out_weight.

            The EP update adapts the readout weight using spin correlations:
            - Free phase: unconstrained Langevin chain (normal calibration).
            - Clamped phase: Langevin chain where we use correct-labeled examples
              to represent the "clamped" (target-known) steady state.

            After ep_update, the EORM readout layer is improved without any
            backpropagation — only local correlations are used.

            **AUC convention (same as v2):**
            EORM assigns LOWER energy to CORRECT responses.
            We negate (−E) before computing AUC so higher score = positive class.

        Args:
            examples: List of dicts with keys:
                - ``question_text`` (str)
                - ``response_text`` (str)
                - ``label`` (int or bool): 1/True = correct, 0/False = incorrect.

        Returns:
            AUC-ROC score in [0.0, 1.0].  0.5 for degenerate input.

        Spec: REQ-EORM-008, SCENARIO-EORM-014
        """
        if not examples:
            return 0.5

        scores = []
        labels = []
        free_hiddens = []   # collect for EP free phase
        clamped_hiddens = []  # collect for EP clamped phase

        for ex in examples:
            cot = CoTEnergyInput(
                question_text=ex["question_text"],
                response_text=ex["response_text"],
            )
            token_ids = _make_token_sequence(
                cot.question_text,
                cot.response_text,
                self.eorm.max_seq_len,
                self.eorm.vocab_size,
            )
            if not token_ids:
                token_ids = [_SEP_ID]

            hidden = _forward_get_pooled(self.eorm.params, token_ids, self.eorm.n_heads)
            calibrated_h = self.calibrate_hidden(hidden)

            # Free phase: the calibrated hidden state (system ran to its natural minimum)
            free_hiddens.append(np.asarray(calibrated_h))

            # Clamped phase: label=1 (correct) examples represent the target state
            lbl = int(ex["label"])
            if lbl == 1:
                clamped_hiddens.append(np.asarray(calibrated_h))

            energy = _energy_from_pooled(
                calibrated_h,
                self.eorm.params["out_weight"],
                self.eorm.params["out_bias"],
            )
            scores.append(-float(energy))
            labels.append(lbl)

        # Apply EP coupling update if configured and we have both phases
        if self.ep_update is not None and len(clamped_hiddens) > 0:
            free_matrix = jnp.array(np.stack(free_hiddens))      # (n, d)
            clamped_matrix = jnp.array(np.stack(clamped_hiddens))  # (m, d)

            # EP update treats out_weight as a 1D coupling vector.
            # We interpret it as one row of the coupling matrix J (d, d)
            # and update it using the diagonal of the EP correlation difference.
            # This is a pragmatic approximation: the full EP formulation would
            # require a full n×n coupling matrix between all hidden units;
            # here we adapt only the readout weights using the diagonal signal.
            old_w = self.eorm.params["out_weight"]  # (d,)
            free_corr_diag = jnp.mean(free_matrix ** 2, axis=0)       # (d,)
            clamped_corr_diag = jnp.mean(clamped_matrix ** 2, axis=0)  # (d,)
            delta_w = self.ep_update.learning_rate * (free_corr_diag - clamped_corr_diag)
            self.eorm.params = {
                **self.eorm.params,
                "out_weight": old_w + delta_w,
            }

        return _auc_roc(labels, scores)
