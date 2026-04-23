"""Adaptive PSV sampler with variance-based early stopping (arXiv 2603.22812).

**Researcher summary:**
    The PSV (self-play verify-repair) loop normally draws a fixed K=4 parallel
    samples per question before selecting the best response via minimum energy.
    This module implements the adaptive stopping strategy from "Efficient
    Hallucination Detection: Adaptive Bayesian Estimation with Guided Semantic
    Exploration" (arXiv 2603.22812, March 2026):

    The key insight: when the distribution of energy scores across K samples has
    low variance, the samples have already converged — drawing more samples will
    not change the selected response (the minimum-energy sample is stable).
    We can stop early and save oracle calls without harming detection quality.

    Target: 30-50% reduction in samples used (sample_reduction_fraction >= 0.30)
    while maintaining detection AUC within 0.02 of the fixed-K baseline.

**How early stopping works:**
    1. Draw K_min samples (minimum — guarantees a meaningful variance estimate).
    2. After each additional sample, compute population variance of all energy
       scores collected so far.
    3. If variance < variance_threshold: stop (distribution converged, minimum
       energy sample is stable).
    4. Hard stop at K_max regardless (prevents unbounded sampling on noisy inputs).

**Why population variance instead of sample variance:**
    For small K (2-8 samples), Bessel's correction (n-1 denominator) inflates
    the variance estimate and causes the sampler to over-sample.  Population
    variance (n denominator) gives a tighter bound on convergence, which is
    the correct behavior when we want to STOP early.

**CPU-only, no real LLM required:**
    AdaptivePSVSampler accepts any callable for generate and compute_energy.
    The experiment uses synthetic energy functions to isolate the stopping
    algorithm from model quality.

Spec: REQ-SAMPLE-020, REQ-SAMPLE-021, SCENARIO-SAMPLE-030, SCENARIO-SAMPLE-031
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable


# ---------------------------------------------------------------------------
# AdaptiveSamplerConfig
# ---------------------------------------------------------------------------


@dataclass
class AdaptiveSamplerConfig:
    """Configuration for variance-based early stopping.

    Parameters
    ----------
    K_min : int
        Minimum number of samples to collect before evaluating the stopping
        criterion.  Must be >= 2 to compute a meaningful variance.
        Corresponds to REQ-SAMPLE-020-2.
    K_max : int
        Maximum number of samples to collect per question.  Sampling always
        stops at K_max even when variance stays above the threshold.
        Corresponds to REQ-SAMPLE-020-1.
    variance_threshold : float
        Population variance threshold below which sampling stops early.
        A lower threshold requires tighter convergence (more samples used).
        A higher threshold stops earlier (fewer samples, potentially noisier).
        Default 0.05 is calibrated to target >= 30% sample reduction on
        typical arithmetic reasoning distributions (arXiv 2603.22812 Fig. 3).
        Corresponds to REQ-SAMPLE-020-3.
    """

    K_min: int = 2
    K_max: int = 8
    variance_threshold: float = 0.05


# ---------------------------------------------------------------------------
# AdaptiveSampleResult
# ---------------------------------------------------------------------------


@dataclass
class AdaptiveSampleResult:
    """Result from one call to sample_until_convergent.

    Fields
    ------
    samples : list[str]
        The candidate responses collected (length == k_used).
    energy_scores : list[float]
        Energy score for each candidate (lower energy = more likely correct).
        Parallel to ``samples``.
    k_used : int
        Number of samples actually collected.  Always in [K_min, K_max].
        Corresponds to REQ-SAMPLE-020-4.
    stopped_early : bool
        True when the variance criterion triggered before K_max was reached.
    best_sample : str
        The candidate with the lowest energy score.  This is the response
        the pipeline would select for this question.
    best_energy : float
        Energy of the best (minimum-energy) sample.
    """

    samples: list[str]
    energy_scores: list[float]
    k_used: int
    stopped_early: bool
    best_sample: str
    best_energy: float


# ---------------------------------------------------------------------------
# AdaptivePSVSampler
# ---------------------------------------------------------------------------


class AdaptivePSVSampler:
    """Variance-based adaptive sampler for the PSV verify-repair loop.

    Wraps any generate/compute_energy pair and applies variance-based early
    stopping per REQ-SAMPLE-020.  The sampler is intentionally decoupled from
    VerifyRepairPipeline so it can be tested with synthetic functions and
    later wired into the real pipeline without changes.

    Parameters
    ----------
    generate_fn : Callable[[str], str]
        Given a question string, returns one candidate response string.
        In production this is VerifyRepairPipeline._generate; in tests it
        is a synthetic function returning deterministic strings.
    compute_energy_fn : Callable[[str, str], float]
        Given (question, candidate_response), returns a scalar energy score.
        Lower energy == better (more constraint-satisfying) response.
        In production this is the EBM energy evaluation; in tests it is
        a synthetic function that returns controlled values.
    config : AdaptiveSamplerConfig
        Stopping criterion parameters (K_min, K_max, variance_threshold).
    """

    def __init__(
        self,
        generate_fn: Callable[[str], str],
        compute_energy_fn: Callable[[str, str], float],
        config: AdaptiveSamplerConfig | None = None,
    ) -> None:
        self._generate = generate_fn
        self._compute_energy = compute_energy_fn
        self.config = config if config is not None else AdaptiveSamplerConfig()

    # ------------------------------------------------------------------
    # sample_until_convergent
    # ------------------------------------------------------------------

    def sample_until_convergent(self, question: str) -> AdaptiveSampleResult:
        """Draw samples until energy variance converges or K_max is reached.

        This is the core adaptive loop from arXiv 2603.22812.  Each iteration
        generates one candidate, evaluates its energy, then checks whether
        the population variance of ALL energy scores collected so far has
        dropped below variance_threshold.  If so, stops early.

        The early-stop check is only evaluated AFTER K_min samples have been
        collected (REQ-SAMPLE-020-2) — variance of fewer than 2 points is
        degenerate (variance of a single point is always 0, which would
        cause immediate false-positive stopping on the first sample).

        Parameters
        ----------
        question : str
            The question to generate and evaluate candidates for.

        Returns
        -------
        AdaptiveSampleResult
            Contains all collected samples, their energy scores, k_used,
            the best sample, its energy, and whether early stopping fired.
        """
        cfg = self.config
        samples: list[str] = []
        energy_scores: list[float] = []
        stopped_early = False

        for k in range(cfg.K_max):
            candidate = self._generate(question)
            energy = self._compute_energy(question, candidate)
            samples.append(candidate)
            energy_scores.append(energy)

            # Only check stopping criterion after K_min samples.
            # k is 0-indexed so k+1 == number of samples collected so far.
            if k + 1 >= cfg.K_min:
                variance = _population_variance(energy_scores)
                if variance < cfg.variance_threshold:
                    stopped_early = True
                    break

        best_idx = energy_scores.index(min(energy_scores))
        return AdaptiveSampleResult(
            samples=samples,
            energy_scores=energy_scores,
            k_used=len(samples),
            stopped_early=stopped_early,
            best_sample=samples[best_idx],
            best_energy=energy_scores[best_idx],
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _population_variance(values: list[float]) -> float:
    """Compute population variance (n denominator, not n-1).

    Uses n denominator rather than Bessel-corrected n-1 because:
    - For K_min=2, Bessel gives (sum of squares) / 1 which is always 2x the
      actual spread — consistently over-estimates variance and delays stopping.
    - We want to STOP early when distribution converges; tighter (smaller)
      variance estimate achieves that goal without sacrificing correctness on
      the detection side.

    Parameters
    ----------
    values : list[float]
        Energy scores for which to compute variance.  Must have >= 1 element.

    Returns
    -------
    float
        Population variance.  Returns 0.0 for a single-element list.
    """
    n = len(values)
    if n <= 1:
        return 0.0
    mean = sum(values) / n
    return sum((v - mean) ** 2 for v in values) / n


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------


def compute_sample_reduction_fraction(k_used_list: list[int], K_max: int) -> float:
    """Compute sample_reduction_fraction per REQ-SAMPLE-021.

    sample_reduction_fraction = 1 - mean(k_used) / K_max

    A value of 0.0 means every question required K_max samples (no savings).
    A value of 1 - K_min/K_max is the theoretical maximum (all stopped at K_min).
    Target for efficiency claim: >= 0.30 (REQ-SAMPLE-021).

    Parameters
    ----------
    k_used_list : list[int]
        Number of samples used per question (from AdaptiveSampleResult.k_used).
    K_max : int
        Maximum samples the adaptive sampler could have drawn.

    Returns
    -------
    float
        Fraction of samples saved, in [0.0, 1.0].
    """
    if not k_used_list or K_max <= 0:
        return 0.0
    mean_used = sum(k_used_list) / len(k_used_list)
    return 1.0 - mean_used / K_max
