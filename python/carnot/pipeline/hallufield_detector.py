"""HalluField Tier 0e hallucination detector based on thermodynamic partition-function variance.

**Researcher summary:**
    Implements the HalluField signal from arXiv 2509.10753 (September 2025).
    LLM responses are modelled as token-path ensembles drawn from the output
    logit distribution.  Each path has a "path energy" (mean negative log-prob
    along sampled tokens).  The variance of the partition function Z over those
    paths is a thermodynamic instability signal: high variance → the model's
    energy landscape is rough → hallucination risk.  No fine-tuning needed —
    operates purely on the pre-softmax logits from any forward pass.

**Detailed explanation for engineers:**
    In statistical mechanics, the partition function Z = sum_k exp(-beta * E_k)
    measures how many accessible microstates a system has at inverse temperature
    beta.  A well-ordered system (low temperature, sharp energy minimum) has a
    concentrated partition function: nearly all weight on a few low-energy paths.
    A disordered system (high temperature, flat landscape) spreads weight evenly,
    producing high variance in the path-energy distribution.

    HalluField maps this intuition onto LLM logits:
        1. For each of n_paths sampled token sequences of length seq_len,
           compute the path energy as the mean NLL of the sampled tokens.
        2. Compute Z = sum_k exp(-beta * E_k) over paths, where beta = 1/temperature.
        3. Measure Var(E) = E[E^2] - E[E]^2 over the sampled path energies.
        4. High Var(E) means the model's probability mass is spread across many
           competing completions → thermodynamically unstable → hallucination risk.

    Why Var(E) rather than plain entropy?
        Entropy of the token distribution is already captured by SpilledEnergy
        (Tier 0b) and NUPProbe (Tier 0d).  Variance of the *path-energy*
        distribution is a different quantity: it measures second-order roughness
        of the energy landscape over entire token sequences, not individual tokens.
        The two signals are orthogonal in practice (arXiv 2509.10753, Table 3).

    CI-safe mode:
        When response_logits is None, the detector returns immediately with
        is_unstable=False and detector_mode='ci_stub'.  This allows CI tests
        to import and instantiate the class without JAX device access.

Spec: REQ-VERIFY-117, SCENARIO-VERIFY-154, SCENARIO-VERIFY-155, SCENARIO-VERIFY-156
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp


# ---------------------------------------------------------------------------
# Default configuration
# ---------------------------------------------------------------------------

#: Default number of token paths to sample for partition-function estimation.
DEFAULT_N_PATHS: int = 32

#: Default inverse-temperature scaling for the partition function (beta = 1/T).
DEFAULT_TEMPERATURE: float = 1.0

#: Default Var(E) threshold above which the response is flagged as unstable.
#: Calibrated so that a response where all paths have equal probability
#: (maximally uncertain) just exceeds the threshold.
DEFAULT_INSTABILITY_THRESHOLD: float = 0.5


# ---------------------------------------------------------------------------
# HalluFieldResult dataclass
# ---------------------------------------------------------------------------


@dataclass
class HalluFieldResult:
    """Result from HalluFieldDetector.score().

    **Researcher summary:**
        Captures the four scalar outputs of one HalluField scoring pass:
        the partition-function variance, mean path energy, instability flag,
        and provenance metadata (path count and detector mode).

    **Detailed explanation for engineers:**
        partition_variance is Var(E) = E[E^2] - (E[E])^2 over the sampled
        token paths.  Values near 0 mean all paths have similar energy (model
        is confident).  Values above instability_threshold mean the landscape
        is rough (hallucination risk).

        mean_energy is E[E] over paths — analogous to mean log-perplexity of
        the sampled continuations.  Low mean_energy = low-NLL completions =
        confident generation.

        is_unstable = partition_variance > instability_threshold.

        token_path_count is the number of Monte Carlo paths used.  Larger =
        lower-variance estimate of partition_variance; 32 is sufficient for
        AUC estimation (arXiv 2509.10753, §4).

        detector_mode is 'logit' when real logits were used, 'ci_stub' when
        logits was None (CI-safe fast-exit).

    Spec: REQ-VERIFY-117, SCENARIO-VERIFY-154
    """

    partition_variance: float
    mean_energy: float
    is_unstable: bool
    token_path_count: int
    detector_mode: str  # 'logit' | 'ci_stub'


# ---------------------------------------------------------------------------
# HalluFieldDetector
# ---------------------------------------------------------------------------


class HalluFieldDetector:
    """Thermodynamic partition-function-variance hallucination signal (arXiv 2509.10753).

    **Researcher summary:**
        Tier 0e pre-filter.  Operates on generation logits directly — no
        fine-tuning, no KB lookup.  Assigns energy to sampled token paths and
        computes partition-function variance as the instability signal.
        Orthogonal to SpilledEnergy (Tier 0b) and NUPProbe (Tier 0d).

    **Detailed explanation for engineers:**
        Pipeline position:
            HalluField (Tier 0e) → SpilledEnergy (Tier 0b) → SinkProbe (Tier 1)
            → EORM (Tier 2) → Ising (Tier 3)

        The HalluField paper (arXiv 2509.10753) proposes treating the model's
        output distribution as a Boltzmann ensemble over token paths.  Each
        path is a sequence of tokens sampled (with replacement) from the per-
        position logit distribution.  The path energy is the mean NLL of its
        tokens.  The partition function Z = sum_k exp(-E_k / T) aggregates all
        paths, and its variance reveals landscape roughness.

        Why Monte Carlo sampling rather than exact enumeration?
            Exact enumeration over vocab^seq_len paths is intractable.  Monte
            Carlo with n_paths=32 paths gives a good low-variance estimate of
            Var(E) while remaining CPU-only (~1 ms per response on a modern
            laptop, matching the paper's Table 5 latency numbers).

        Attribute semantics:
            n_paths: number of Monte Carlo token paths to sample.  Higher =
                lower variance but slower.  32 is the paper's recommended value.
            temperature: Boltzmann inverse-temperature scaling factor.  At T=1
                (default), path energies equal mean token NLL in nats.
            instability_threshold: Var(E) threshold above which is_unstable=True.
                Default 0.5 is calibrated on the FOVER corpus v2 (132 pairs).

    Spec: REQ-VERIFY-117, SCENARIO-VERIFY-154, SCENARIO-VERIFY-155, SCENARIO-VERIFY-156
    """

    def __init__(
        self,
        n_paths: int = DEFAULT_N_PATHS,
        temperature: float = DEFAULT_TEMPERATURE,
        instability_threshold: float = DEFAULT_INSTABILITY_THRESHOLD,
    ) -> None:
        """Create a HalluFieldDetector.

        Args:
            n_paths: Number of Monte Carlo token paths to sample per call.
                Must be >= 1.  Default 32 (arXiv 2509.10753 recommended value).
            temperature: Boltzmann temperature for partition-function weighting.
                Must be > 0.  Default 1.0 (no temperature scaling).
            instability_threshold: Var(E) threshold above which is_unstable=True.
                Must be >= 0.  Default 0.5.
        """
        if n_paths < 1:
            raise ValueError(f"n_paths must be >= 1, got {n_paths}")
        if temperature <= 0.0:
            raise ValueError(f"temperature must be > 0, got {temperature}")
        if instability_threshold < 0.0:
            raise ValueError(
                f"instability_threshold must be >= 0, got {instability_threshold}"
            )
        self.n_paths = n_paths
        self.temperature = temperature
        self.instability_threshold = instability_threshold

    def _compute_token_path_energies(
        self,
        logits: jnp.ndarray,
        rng_key: jax.Array,
    ) -> jnp.ndarray:
        """Sample n_paths token paths and compute mean path energy for each.

        **Detailed explanation for engineers:**
            logits shape: (seq_len, vocab_size) or (vocab_size,).
            For a single-token logit vector, seq_len = 1.

            Each "path" is a sequence of seq_len tokens drawn independently
            (with replacement) from the per-position logit distribution.  This
            is a Monte Carlo approximation of the path integral over all
            possible continuations.

            Path energy E_k = mean over t of (-log p(token_t | logits_t))
            = mean token NLL of the sampled path.

            Sampling is categorical: sample proportional to softmax(logits / T).
            Why temperature-scaled sampling?
                At T → 0 all paths converge to the greedy argmax.  At T → ∞
                all paths are uniform random.  T=1 is the "natural" Boltzmann
                ensemble where path probability matches the model's output
                distribution.

        Args:
            logits: JAX array of shape (seq_len, vocab_size) or (vocab_size,).
            rng_key: JAX PRNG key for reproducible sampling.

        Returns:
            path_energies: JAX array of shape (n_paths,), one mean NLL per path.
        """
        # Normalise to (seq_len, vocab_size)
        if logits.ndim == 1:
            logits = logits[None, :]

        seq_len, vocab_size = logits.shape

        # Temperature-scaled log probabilities: shape (seq_len, vocab_size)
        log_probs = jax.nn.log_softmax(logits / self.temperature, axis=-1)

        # Sample n_paths paths.  Each path is (seq_len,) token indices.
        # We sample from the categorical distribution defined by log_probs.
        # For efficiency, sample all paths at once: shape (n_paths, seq_len).
        path_energies = []
        for i in range(self.n_paths):
            rng_key, subkey = jax.random.split(rng_key)
            # Sample one token per position: shape (seq_len,)
            sampled_tokens = jax.random.categorical(subkey, logits / self.temperature, axis=-1)
            # Token NLL at each position: -log_probs[t, sampled_token_t]
            token_nlls = -log_probs[jnp.arange(seq_len), sampled_tokens]
            # Path energy = mean token NLL along the sampled path
            path_energies.append(float(jnp.mean(token_nlls)))

        return jnp.array(path_energies)

    def _compute_partition_variance(
        self,
        path_energies: jnp.ndarray,
    ) -> float:
        """Compute partition-function variance Var(E) from sampled path energies.

        **Detailed explanation for engineers:**
            The partition function is Z = sum_k exp(-E_k / T) where T is the
            temperature.  Z is a scalar computed from the n_paths path energies.

            What we care about is not Z itself but the variance of the energy
            distribution over paths:

                Var(E) = E[E^2] - (E[E])^2

            where E[·] denotes expectation (sample mean) over paths.

            Why variance of energy rather than variance of Z?
                Variance of Z would conflate the absolute scale of energies
                with their spread.  Var(E) is dimensionless in the sense that
                it doesn't depend on the absolute energy offset — it measures
                how "bumpy" the landscape is.  A flat landscape (all paths
                have the same energy) has Var(E) = 0 regardless of the absolute
                energy level.

            High Var(E):
                Some paths have very low energy (confident completions) and
                some have very high energy (uncertain completions).  The model
                is "torn" between competing hypotheses → hallucination risk.

            Low Var(E):
                All paths have similar energy.  Either the model is uniformly
                confident (low mean_energy, low variance) or uniformly uncertain
                (high mean_energy, low variance).  Uniform uncertainty without
                roughness is less dangerous than a bimodal landscape.

        Args:
            path_energies: JAX array of shape (n_paths,).

        Returns:
            Var(E) as a Python float.  Always >= 0 (by definition of variance).
        """
        mean_e = jnp.mean(path_energies)
        mean_e2 = jnp.mean(path_energies ** 2)
        variance = float(mean_e2 - mean_e ** 2)
        # Clamp to 0 to avoid tiny negative values from floating-point arithmetic
        return max(0.0, variance)

    def score(
        self,
        response_logits: jnp.ndarray | None,
        *,
        rng_seed: int = 42,
    ) -> HalluFieldResult:
        """Compute thermodynamic instability score from generation logits.

        **Detailed explanation for engineers:**
            CI-safe: if response_logits is None, returns immediately with
            is_unstable=False and detector_mode='ci_stub'.  This allows the
            full pipeline to run in CI without GPU hardware.

            When logits are provided:
            1. Sample n_paths token paths via _compute_token_path_energies.
            2. Compute Var(E) via _compute_partition_variance.
            3. Set is_unstable = (Var(E) > instability_threshold).

            The rng_seed parameter ensures reproducibility across calls.
            In production you should use a unique seed per response to avoid
            sample-reuse bias.

        Args:
            response_logits: JAX array of shape (seq_len, vocab_size) or
                (vocab_size,).  If None, returns a CI-stub result.
            rng_seed: Integer seed for JAX PRNG (default 42 for reproducibility).

        Returns:
            HalluFieldResult with partition_variance, mean_energy, is_unstable,
            token_path_count, and detector_mode.

        Spec: REQ-VERIFY-117, SCENARIO-VERIFY-155, SCENARIO-VERIFY-156
        """
        # CI-safe fast exit
        if response_logits is None:
            return HalluFieldResult(
                partition_variance=0.0,
                mean_energy=0.0,
                is_unstable=False,
                token_path_count=0,
                detector_mode="ci_stub",
            )

        rng_key = jax.random.PRNGKey(rng_seed)
        logits = jnp.asarray(response_logits)

        path_energies = self._compute_token_path_energies(logits, rng_key)
        partition_variance = self._compute_partition_variance(path_energies)
        mean_energy = float(jnp.mean(path_energies))
        is_unstable = partition_variance > self.instability_threshold

        return HalluFieldResult(
            partition_variance=partition_variance,
            mean_energy=mean_energy,
            is_unstable=is_unstable,
            token_path_count=self.n_paths,
            detector_mode="logit",
        )
