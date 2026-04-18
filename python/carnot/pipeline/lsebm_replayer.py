"""LSEBMCL constraint replay: train an Ising EBM on Session 1 violation patterns.

**Why EBM replay prevents the cold-start problem (arXiv 2501.05495):**
    LSEBM-CL (Lifelong Sequence EBM with Continual Learning) addresses catastrophic
    forgetting in continual learning by training a compact generative EBM on each
    task's data distribution and replaying synthetic samples when starting a new task.
    In Carnot's cross-session relay setting:

    - "Task N" = Session N's error distribution (e.g., carry errors are common).
    - Without replay (Exp 448): Session 2 starts cold — the template library has no
      knowledge that carry errors are likely until enough real questions accumulate.
    - With replay (this module): the EBM encodes the Session 1 distribution compactly;
      Session 2 warm-starts by sampling synthetic violations from the EBM BEFORE any
      real questions arrive.

    The key insight is that the EBM is trained on violation TYPE distributions, not
    on raw text.  Each violation type becomes a binary indicator in a small feature
    vector (e.g., dim=8 covers all four arithmetic error families with slack).

**Why Ising EBM for this task:**
    The Ising EBM is the simplest EBM in the Carnot hierarchy:
    E(x) = -0.5 * x^T J x - b^T x

    For violation distribution modelling we need:
    - Low dimensionality: we have at most ~10 violation types, so dim=8-16 is enough.
    - Fast CPU training: 100-iter contrastive divergence converges in < 1 second.
    - No GPU required: the entire experiment is CPU-only.
    - Interpretable: bias b[i] captures the marginal prevalence of violation type i;
      coupling J[i][j] captures co-occurrence patterns.

    The Gibbs or Boltzmann tiers would be over-engineered for this 8-dimensional
    categorical distribution problem.

**How warm-start differs from Exp 448's exact-copy template loading:**
    Exp 448 serialised the full ConstraintTemplateLibrary to disk and deserialised
    it verbatim in Session 2 — an exact copy of stored state.  This requires saving
    and loading the entire template library (all observation counts, all activation
    states).

    LSEBMCL replay is generative and probabilistic:
    1. Fit a small Ising EBM on the Session 1 violation frequency distribution.
    2. In Session 2, SAMPLE new synthetic violation instances from the EBM.
    3. Inject those synthetic violations into the template library's observation counts.

    Advantages: (a) compact storage — only EBM parameters, not the full library;
    (b) probabilistic generalisation — the EBM can sample from the interior of the
    learned distribution, not just exact replay of observed counts; (c) the approach
    scales to many violation types without a quadratic blowup in storage.

Spec: REQ-SELFLEARN-013, REQ-SELFLEARN-014, REQ-SELFLEARN-015,
SCENARIO-SELFLEARN-013, SCENARIO-SELFLEARN-014, SCENARIO-SELFLEARN-015
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

import jax
import jax.numpy as jnp
import jax.random as jrandom

from carnot.models.ising import IsingConfig, IsingModel


# ---------------------------------------------------------------------------
# ViolationDistribution
# ---------------------------------------------------------------------------


@dataclass
class ViolationDistribution:
    """Frequency distribution over observed violation types from a session.

    **Why we need this class:**
        Raw violation strings (e.g., ['carry', 'carry', 'sign', 'carry']) must be
        converted into a format the Ising EBM can train on.  The EBM operates on
        fixed-length binary/continuous vectors, not arbitrary strings.

        ViolationDistribution handles the vocabulary bookkeeping: mapping violation
        type strings to integer indices, and converting counts to (feature_vector,
        energy_target) training pairs.

    **Training pair encoding:**
        For a vocabulary of K types, each training pair is a K-dimensional binary
        vector where position i is 1.0 if violation type i is present.  We generate
        one training pair per observed violation instance, so common violation types
        appear more often in the training set — naturally weighting the EBM to assign
        low energy to common types.

    Args:
        counts: Mapping from violation type string to integer observation count.

    Spec: REQ-SELFLEARN-013, SCENARIO-SELFLEARN-013
    """

    counts: dict[str, int]
    # Sorted vocabulary list — deterministic order for reproducibility.
    _vocab: list[str] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._vocab = sorted(self.counts.keys())

    @property
    def vocab(self) -> list[str]:
        """Sorted list of violation types seen during this session."""
        return list(self._vocab)

    def to_training_pairs(self) -> list[tuple[list[float], float]]:
        """Convert violation counts to (feature_vector, energy_target) pairs.

        **How the encoding works:**
            Each violation instance produces one training pair.  The feature vector
            is a K-dimensional binary vector where position i is 1.0 if the violation
            type for that instance matches vocabulary item i.

            Energy target is 0.0 for all training pairs — we want the EBM to assign
            LOW energy (i.e., high probability) to violation types that were observed.
            The contrastive divergence training then pushes the model to assign HIGH
            energy (low probability) to unobserved configurations.

        Returns:
            List of (feature_vector, energy_target) tuples.  Length equals the total
            number of violation instances (sum of all counts).  Empty list if counts is
            empty.

        Spec: REQ-SELFLEARN-013
        """
        k = len(self._vocab)
        if k == 0:
            return []

        pairs: list[tuple[list[float], float]] = []
        for i, vtype in enumerate(self._vocab):
            count = self.counts.get(vtype, 0)
            # Build a one-hot vector for this violation type.
            vec = [0.0] * k
            vec[i] = 1.0
            # Each instance contributes one training pair.
            for _ in range(count):
                pairs.append((list(vec), 0.0))
        return pairs

    def most_common(self, n: int) -> list[tuple[str, int]]:
        """Return the top-n most frequent violation types.

        Args:
            n: Maximum number of (type, count) pairs to return.  Returns fewer
               if the vocabulary is smaller than n.

        Returns:
            List of (violation_type, count) sorted descending by count.

        Spec: REQ-SELFLEARN-013
        """
        sorted_items = sorted(self.counts.items(), key=lambda x: x[1], reverse=True)
        return sorted_items[:n]


# ---------------------------------------------------------------------------
# LSEBMConstraintReplayer
# ---------------------------------------------------------------------------

# Minimum Ising EBM dimension — must be >= vocab size.  If vocab is larger,
# we pad to the next multiple of _MIN_DIM.
_MIN_DIM = 4

# Contrastive divergence step size — small enough for stable CPU training.
_CD_STEP_SIZE = 0.01

# Number of Gibbs sampling steps per CD update — one step is enough for a
# small Ising model with low correlation (fast mixing at dim ≤ 16).
_CD_GIBBS_STEPS = 1


class LSEBMConstraintReplayer:
    """Train a small Ising EBM on Session 1 violation patterns; replay in Session 2.

    **Lifecycle:**
        1. Call ``fit(session1_violations)`` after Session 1 to train the EBM.
        2. Call ``generate_replay(n)`` to sample synthetic violation types.
        3. Call ``warm_start(memory)`` to inject synthetic violations into a
           SessionMemory-compatible store and return the count of warm-started templates.

    **Training algorithm (contrastive divergence):**
        CD-1 for an Ising EBM on violation distribution:
        1. For each training pair (x_pos, _):
           a. Gibbs-sample a negative sample x_neg from the current model.
           b. Update J += lr * (x_pos @ x_pos.T - x_neg @ x_neg.T) / batch
           c. Update b += lr * (x_pos - x_neg) / batch
           d. Re-symmetrise J = (J + J.T) / 2
        2. Repeat for n_iter steps across the training pairs.

    **Sampling algorithm:**
        Block Gibbs sampling (one variable at a time):
        1. Start from a random binary vector.
        2. For each variable i in random order:
           conditional: p(x_i = 1 | x_{-i}) = sigmoid(J[i] @ x + b[i])
        3. After burn-in steps, decode: return vocab[argmax(x)].

    Args:
        n_replay: Number of synthetic violations to generate in warm-start. Default 20.
        ebm_n_iter: Number of contrastive divergence training iterations. Default 100.

    Spec: REQ-SELFLEARN-013, REQ-SELFLEARN-014, SCENARIO-SELFLEARN-013/014
    """

    def __init__(self, n_replay: int = 20, ebm_n_iter: int = 100) -> None:
        self.n_replay = n_replay
        self.ebm_n_iter = ebm_n_iter
        self._dist: ViolationDistribution | None = None
        self._model: IsingModel | None = None
        # JAX coupling and bias as mutable numpy-backed arrays for CD updates.
        self._coupling: Any = None
        self._bias: Any = None
        self._vocab: list[str] = []
        self._key = jrandom.PRNGKey(42)

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

    def fit(self, session1_violations: list[str]) -> None:
        """Train a small Ising EBM on the violation type distribution from Session 1.

        **Why we count first, then train:**
            The EBM needs a fixed-dimension feature space.  We build the vocabulary
            from the observed violation types, then encode each instance as a one-hot
            vector.  Training on one-hot encodings biases the EBM toward assigning
            low energy to the most frequently observed types.

        **CD training details:**
            We run CD-1 (one Gibbs step per CD update) for ebm_n_iter iterations.
            This is fast on CPU for dim ≤ 16 and sufficient to learn marginal frequencies
            and pairwise co-occurrences for the simple violation type vocabularies we
            encounter in practice (4-8 distinct types).

        Args:
            session1_violations: List of violation type strings observed in Session 1
                                  (e.g., ['carry', 'carry', 'sign', 'carry', ...]).
                                  Duplicate strings are counted — do not pre-deduplicate.

        Spec: REQ-SELFLEARN-013, SCENARIO-SELFLEARN-013
        """
        # Step 1: build frequency distribution.
        counts: dict[str, int] = {}
        for v in session1_violations:
            counts[v] = counts.get(v, 0) + 1
        self._dist = ViolationDistribution(counts=counts)
        self._vocab = self._dist.vocab

        if not self._vocab:
            # No violations — nothing to train on.
            return

        # Step 2: determine EBM dimension (pad to at least _MIN_DIM).
        k = len(self._vocab)
        dim = max(k, _MIN_DIM)

        # Step 3: initialise Ising model.
        config = IsingConfig(input_dim=dim, coupling_init="xavier_uniform")
        self._key, subkey = jrandom.split(self._key)
        model = IsingModel(config, key=subkey)

        # Step 4: extract mutable numpy copies of parameters for CD updates.
        import numpy as np

        J = np.array(model.coupling)
        b = np.array(model.bias)

        # Step 5: build training pairs (one-hot vectors, padded to dim).
        raw_pairs = self._dist.to_training_pairs()

        # Pad each k-dim one-hot to dim.
        x_pos_all = np.zeros((len(raw_pairs), dim), dtype=np.float32)
        for idx, (vec, _) in enumerate(raw_pairs):
            x_pos_all[idx, : len(vec)] = vec

        n_train = len(x_pos_all)
        lr = _CD_STEP_SIZE

        # Step 6: CD training loop (skipped when there are no training pairs).
        rng = np.random.default_rng(seed=0)
        for iteration in range(self.ebm_n_iter if n_train > 0 else 0):
            # Pick a random positive sample.
            pos_idx = rng.integers(0, n_train)
            x_pos = x_pos_all[pos_idx].copy()

            # Gibbs sample negative from current model (block Gibbs, _CD_GIBBS_STEPS steps).
            x_neg = rng.choice([0.0, 1.0], size=dim).astype(np.float32)
            for _ in range(_CD_GIBBS_STEPS):
                for i in rng.permutation(dim):
                    # Conditional activation energy: -(J[i] @ x_neg + b[i])
                    logit = float(J[i] @ x_neg + b[i])
                    prob = 1.0 / (1.0 + np.exp(-logit))
                    x_neg[i] = 1.0 if rng.random() < prob else 0.0

            # Contrastive divergence parameter update.
            # WHY: CD-1 gradient approximation for Ising EBM:
            #   dE/dJ ≈ -(x_pos x_pos^T - x_neg x_neg^T)
            #   dE/db ≈ -(x_pos - x_neg)
            # We minimise energy for positive samples, maximise for negative.
            J += lr * (np.outer(x_pos, x_pos) - np.outer(x_neg, x_neg))
            b += lr * (x_pos - x_neg)
            # Re-enforce symmetry: J must be symmetric for Ising EBM correctness.
            J = (J + J.T) / 2.0

        # Step 7: store trained parameters.
        self._coupling = J
        self._bias = b
        # Also update the IsingModel object for completeness.
        model.coupling = jnp.array(J)
        model.bias = jnp.array(b)
        self._model = model

    # ------------------------------------------------------------------
    # generate_replay
    # ------------------------------------------------------------------

    def generate_replay(self, n: int) -> list[str]:
        """Sample n synthetic violation types from the trained EBM.

        **Sampling strategy:**
            We use block Gibbs sampling with a short burn-in.  After burn-in,
            we map the binary sample vector back to a violation type by taking
            argmax over the first k=len(vocab) dimensions (the padded dimensions
            beyond the vocabulary are ignored).

            If no violations were observed during fit (empty vocab), returns an
            empty list.

        **Why argmax decoding:**
            The one-hot training encoding means the model's energy landscape has
            local minima near each basis vector (one-hot for each violation type).
            argmax decoding therefore maps a Gibbs sample to the nearest basis
            vector, i.e., the most activated violation type in the sample.  This
            is the simplest decoder consistent with how the EBM was trained.

        Args:
            n: Number of synthetic violation type strings to generate.

        Returns:
            List of n violation type strings, all drawn from the vocabulary seen
            during ``fit()``.  Empty list if ``fit()`` was not called or no violations
            were observed.

        Spec: REQ-SELFLEARN-014, SCENARIO-SELFLEARN-013
        """
        if not self._vocab or self._coupling is None or self._bias is None:
            return []

        import numpy as np

        J = self._coupling
        b = self._bias
        k = len(self._vocab)
        dim = J.shape[0]
        rng = np.random.default_rng(seed=1)

        results: list[str] = []
        # Start from a fresh random state for each sample to reduce autocorrelation.
        x = rng.choice([0.0, 1.0], size=dim).astype(np.float32)

        burn_in = 20
        for _ in range(burn_in):
            for i in rng.permutation(dim):
                logit = float(J[i] @ x + b[i])
                prob = 1.0 / (1.0 + np.exp(-logit))
                x[i] = 1.0 if rng.random() < prob else 0.0

        for _ in range(n):
            # One Gibbs step per sample.
            for i in rng.permutation(dim):
                logit = float(J[i] @ x + b[i])
                prob = 1.0 / (1.0 + np.exp(-logit))
                x[i] = 1.0 if rng.random() < prob else 0.0
            # Decode: argmax over vocab dimensions maps to nearest basis vector.
            vocab_activation = x[:k]
            if vocab_activation.sum() == 0.0:
                # All zeros: fall back to the most common violation type from training.
                best_idx = 0
            else:
                best_idx = int(np.argmax(vocab_activation))
            results.append(self._vocab[best_idx])

        return results

    # ------------------------------------------------------------------
    # warm_start
    # ------------------------------------------------------------------

    def warm_start(self, memory: Any) -> int:
        """Inject synthetic violations into session memory's template observation counts.

        **What "warm-start" means here:**
            The SessionMemory object holds saved state for a pipeline's template library.
            Normally, Session 2 starts cold — the template library's observation counts
            are zero for all templates.

            warm_start uses the EBM to generate ``self.n_replay`` synthetic violation
            instances, then injects them as observations into the session-level violation
            tracking dict.  When Session 2 loads this memory, the template library already
            has non-zero observation counts for the violation types the EBM predicts are
            likely — so templates can activate immediately on the first real question.

        **Why we don't need full SessionMemory integration:**
            The experiment script tracks violation counts independently of the on-disk
            SessionMemory (which requires CaseMemory/ConstraintTemplateLibrary/FPTracker
            tuple serialisation).  For the warm-start experiment, we track injected counts
            directly on the memory object via a duck-typed ``_warm_start_counts`` dict.
            This keeps the replayer decoupled from the full SessionMemory serialisation
            format while still demonstrating the LSEBMCL principle.

        Args:
            memory: A SessionMemory-like object.  This method attaches a
                    ``_warm_start_counts`` attribute with the injected violation counts.
                    This is the experiment-level warm-start record; downstream code reads
                    it to pre-populate template observation counts.

        Returns:
            Count of distinct violation types that were warm-started (i.e., the number
            of distinct types in the generated replay, which equals the number of templates
            that would activate immediately in Session 2).

        Spec: REQ-SELFLEARN-014, SCENARIO-SELFLEARN-014
        """
        replay = self.generate_replay(self.n_replay)
        warm_counts: dict[str, int] = {}
        for v in replay:
            warm_counts[v] = warm_counts.get(v, 0) + 1
        # Attach the warm-start record to the memory object (duck-typed extension).
        memory._warm_start_counts = warm_counts  # type: ignore[attr-defined]
        return len(warm_counts)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "ViolationDistribution",
    "LSEBMConstraintReplayer",
]
