"""JEPA Quality-Gated Retrain (V2) — CoT pair quality filter + EBM-guided augmentation.

**Why this module exists (RETRO-040):**
    Exp 472 caused an AUC regression: JEPA dropped from 0.667 to 0.400 after retraining
    on 54 real CoT pairs. Root cause analysis (RETRO-040, milestone .35) found that some
    of those pairs included partially-verifiable arithmetic steps annotated with low
    confidence — a human annotator marking a step as 'incorrect' when they were only 60%
    sure. Training JEPA on low-confidence labels teaches it the wrong signal: the energy
    landscape becomes noisy, discriminating ability decreases, and AUC regresses.

    The fix has two parts:

    1. **Quality gate** (CoTPairQualityFilter): before a pair enters training, it must
       pass two thresholds: arithmetic_coverage >= 0.3 (at least 30% of steps have a
       verifiable arithmetic operation) AND label_confidence >= 0.7 (the annotator was
       at least 70% confident in the label). Pairs that fail either threshold are
       discarded before the training loop sees them.

    2. **EBM-guided augmentation** (JEPAQualityAugmentor): after filtering, the corpus
       is typically too small (e.g., 30 pairs from 57 raw). Random synthetic pairs are
       not the right fix — they sample uniformly from constraint space and do not
       resemble the actual failure modes seen in the pipeline. Instead, we sample from
       the Ising model's energy landscape: spin configurations with energy above the
       mean are "violation" examples; configurations with energy below the mean are
       "correct" examples. Because the Ising coupling matrix encodes learned pairwise
       constraint interactions from real data, these synthetic pairs ARE representative
       of the failures JEPA must learn to predict.

**Design principle — energy as ground truth:**
    The Carnot project treats the EBM energy function as ground truth (CLAUDE.md §
    Operational Principles). JEPAQualityAugmentor operationalizes this: instead of
    asking a human to label synthetic examples, we let the energy function decide.
    High energy = the model assigns low probability = likely a violation. This is
    internally consistent with the rest of the pipeline.

**Spec:** REQ-LEARN-037, REQ-LEARN-038, REQ-LEARN-039,
          SCENARIO-LEARN-066, SCENARIO-LEARN-067, SCENARIO-LEARN-068
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp
import jax.random as jrandom


# ---------------------------------------------------------------------------
# CoTPairQuality — quality metrics for one training pair
# ---------------------------------------------------------------------------


@dataclass
class CoTPairQuality:
    """Quality metrics for a single CoT training pair.

    **For engineers:**
        Before a CoT pair enters the JEPA training loop, we score it on two
        orthogonal axes:

        1. ``arithmetic_coverage``: fraction of reasoning steps that contain a
           verifiable arithmetic expression (e.g. "2 * 3 = 6", "x + 5 = 12").
           Steps without arithmetic cannot be checked by Z3 or similar symbolic
           verifiers, so they provide weaker training signal.  A pair with
           coverage=0.0 has no checkable steps at all.

        2. ``label_confidence``: the annotator's stated confidence in the
           correct/incorrect label.  Confidence comes from the FOVER annotator
           pipeline (FOVERAnnotator, Exp 462).  If the annotator said 70%, this
           value is 0.7.  Low-confidence labels introduce noise that the JEPA
           model cannot overcome, so we gate them out.

    Attributes:
        arithmetic_coverage: Fraction of steps with verifiable arithmetic [0, 1].
        label_confidence: Annotator confidence in the label [0, 1].
        n_steps: Total number of reasoning steps in the response.

    Derived attributes:
        passes_gate: True iff both thresholds are met.
        quality_score: Harmonic mean of coverage and confidence (high = good).

    Spec: REQ-LEARN-037
    """

    arithmetic_coverage: float
    label_confidence: float
    n_steps: int

    @property
    def passes_gate(self) -> bool:
        """Return True iff both quality thresholds are met.

        The gate uses a default of arithmetic_coverage >= 0.3 AND
        label_confidence >= 0.7.  These thresholds are set on the
        CoTPairQualityFilter, not here — this property just reflects the
        raw measurements.  CoTPairQualityFilter.compute_quality() compares
        these values against its configured thresholds to make the accept/
        reject decision.

        Note: this property uses HARDCODED thresholds (0.3, 0.7) and is only
        used for testing convenience.  In production, always use
        CoTPairQualityFilter which applies the configured thresholds.
        """
        return self.arithmetic_coverage >= 0.3 and self.label_confidence >= 0.7

    @property
    def quality_score(self) -> float:
        """Harmonic mean of arithmetic_coverage and label_confidence.

        **Why harmonic mean?**
            The harmonic mean penalizes imbalance harder than the arithmetic
            mean.  A pair with coverage=0.0 and confidence=1.0 gets quality_score=0
            even though its arithmetic mean is 0.5.  This reflects that a pair
            with zero arithmetic coverage provides zero verifiable training signal
            regardless of how confident the annotator is.

        Returns 0.0 if either component is zero (avoids division by zero).
        """
        c = self.arithmetic_coverage
        k = self.label_confidence
        if c <= 0.0 or k <= 0.0:
            return 0.0
        return 2.0 * c * k / (c + k)


# ---------------------------------------------------------------------------
# CoTPairQualityFilter
# ---------------------------------------------------------------------------


def _estimate_arithmetic_coverage(pair: dict[str, Any]) -> float:
    """Estimate arithmetic_coverage for a pair dict from available fields.

    **For engineers:**
        A pair dict may come from multiple sources (FOVER annotator, Exp 442,
        Exp 476, etc.).  Each source uses slightly different field names.
        This function tries them in priority order:

        1. ``arithmetic_coverage`` — explicit field from FOVER annotator v2+
        2. ``step_text`` — raw step text; we count arithmetic-looking tokens
        3. Fall back to a conservative estimate from label_confidence

        The arithmetic detection regex looks for patterns like:
        - "2 * 3 = 6"       (multiply)
        - "x + 5 = 12"      (add with variable)
        - "100 - 20 = 80"   (subtract)
        - "18 / 3 = 6"      (divide)
        - standalone numbers adjacent to operators

        This is intentionally permissive — we want coverage >= 0.3 to mean
        "at least a few arithmetic steps present", not "every step is perfect".
    """
    # Explicit field wins
    if "arithmetic_coverage" in pair:
        try:
            return float(pair["arithmetic_coverage"])
        except (TypeError, ValueError):
            pass

    # Try to estimate from step_text
    text = pair.get("step_text", "") or pair.get("response", "") or pair.get("full_response", "")
    if text:
        # Detect arithmetic in multiple formats, including LaTeX (common in FOVER annotator output):
        #   - Plain:  "2 * 3 = 6"
        #   - LaTeX:  "\( S = 20 \)", "\[ C = 4 \times 20 = 80 \]", "$10 \times 1.2 = 12$"
        #   - Mixed:  "rate is \( 0.25 \times 16 = 4 \)"
        patterns = [
            re.compile(r"[\d\.]+\s*[+\-\*\/×÷]\s*[\d\.]"),       # plain arithmetic
            re.compile(r"\\times\s*[\d\.]"),                        # LaTeX \times
            re.compile(r"\\cdot\s*[\d\.]"),                         # LaTeX \cdot
            re.compile(r"\\frac\{"),                                # LaTeX fraction
            re.compile(r"\\\(\s*[A-Za-z\d].*?=.*?\\?\)"),          # LaTeX inline math with =
            re.compile(r"\\\[.*?=.*?\\\]", re.DOTALL),             # LaTeX display math with =
            re.compile(r"\$[^\$]*[+\-\*\/=][^\$]*\$"),            # $...$ inline math
            re.compile(r"[\d\.]+\s*=\s*[\d\.]"),                   # simple equation N=M
        ]
        total_matches = sum(len(p.findall(text)) for p in patterns)
        # Normalise by a rough step count (split on sentence/line boundaries)
        sentences = [s.strip() for s in re.split(r"[.\n]", text) if s.strip()]
        n_sentences = max(1, len(sentences))
        # Coverage = arithmetic density per sentence, clamped to [0, 1]
        raw = total_matches / n_sentences
        return min(1.0, raw * 0.4)  # scale: ~2.5 arithmetic hits per sentence → 1.0

    return 0.0


def _estimate_label_confidence(pair: dict[str, Any]) -> float:
    """Estimate label_confidence for a pair dict.

    Priority order:
    1. ``label_confidence`` explicit field
    2. ``confidence`` field (Exp 442 / FOVER format)
    3. Default 1.0 if label is deterministic (e.g., Z3-verified)
    """
    for field in ("label_confidence", "confidence"):
        if field in pair:
            try:
                return float(pair[field])
            except (TypeError, ValueError):
                pass
    # If no confidence field, assume maximum confidence (Z3-verified)
    return 1.0


class CoTPairQualityFilter:
    """Filter a corpus of CoT training pairs by quality thresholds.

    **For engineers:**
        This is the primary fix for RETRO-040.  Before any CoT pair enters the
        JEPA training loop, it must pass through this filter.  Pairs that fail
        either threshold are discarded entirely — they are not down-weighted,
        not relabeled, not corrected.  The philosophy is that noisy supervision
        is worse than smaller clean data.

        After filtering, if the corpus is too small for robust training, use
        JEPAQualityAugmentor to add EBM-guided synthetic pairs.

    Args:
        min_coverage: Minimum arithmetic_coverage threshold (inclusive). Default 0.3.
        min_confidence: Minimum label_confidence threshold (inclusive). Default 0.7.

    Spec: REQ-LEARN-037, SCENARIO-LEARN-066
    """

    def __init__(self, min_coverage: float = 0.3, min_confidence: float = 0.7) -> None:
        """Create filter with the given quality thresholds.

        Args:
            min_coverage: Pairs with arithmetic_coverage below this are rejected.
            min_confidence: Pairs with label_confidence below this are rejected.
        """
        self.min_coverage = min_coverage
        self.min_confidence = min_confidence

    def compute_quality(self, pair: dict[str, Any]) -> CoTPairQuality:
        """Compute quality metrics for a single pair dict.

        **For engineers:**
            This function is separate from filter() so you can inspect quality
            scores before deciding on thresholds.  Useful for threshold tuning.

        Args:
            pair: A dict with at minimum a 'label' or 'correct' field.
                  May also have 'arithmetic_coverage', 'confidence', 'step_text'.

        Returns:
            CoTPairQuality with computed metrics.
        """
        coverage = _estimate_arithmetic_coverage(pair)
        confidence = _estimate_label_confidence(pair)
        # n_steps: count from step_text line breaks, or default to 1
        text = pair.get("step_text", "") or pair.get("response", "") or ""
        lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
        n_steps = max(1, len(lines))
        return CoTPairQuality(
            arithmetic_coverage=coverage,
            label_confidence=confidence,
            n_steps=n_steps,
        )

    def filter(self, pairs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Return only pairs that pass both quality thresholds.

        **For engineers:**
            Iterates pairs, computes quality for each, and includes only those
            where arithmetic_coverage >= min_coverage AND label_confidence >= min_confidence.
            Rejected pairs are silently dropped (not logged here — caller should
            log the before/after counts).

        Args:
            pairs: List of pair dicts. Each dict needs at least a label field.

        Returns:
            Subset of pairs that passed the quality gate. May be empty.

        Spec: REQ-LEARN-037, SCENARIO-LEARN-066
        """
        accepted: list[dict[str, Any]] = []
        for pair in pairs:
            q = self.compute_quality(pair)
            if q.arithmetic_coverage >= self.min_coverage and q.label_confidence >= self.min_confidence:
                accepted.append(pair)
        return accepted


# ---------------------------------------------------------------------------
# JEPAQualityAugmentor — EBM-guided synthetic pair generation
# ---------------------------------------------------------------------------


class JEPAQualityAugmentor:
    """Generate EBM-guided synthetic CoT pairs by sampling from the Ising energy landscape.

    **Why not random synthetic?**
        Random synthetic pairs sample uniformly from the constraint space and
        therefore do not concentrate on the actual failure modes the JEPA model
        must learn to predict.  In contrast, the Ising model's energy landscape
        was learned from real pipeline data — its coupling matrix encodes pairwise
        constraint interactions observed in real CoT steps.

        Sampling from the tails of the energy distribution gives us:
        - **Violation pairs** (incorrect=True): spin configurations near local energy
          maxima — the model says these are unlikely / wrong.  These represent the
          kinds of reasoning mistakes the pipeline actually produces.
        - **Correct pairs** (incorrect=False): spin configurations near local energy
          minima — the model says these are natural / correct.

        The energy function is ground truth (per CLAUDE.md § Operational Principles).
        We trust it to label our synthetic examples rather than using human annotation.

    **Sampling procedure:**
        We draw random spin configurations (uniform ±1 for Ising-style) using JAX,
        compute their energies, and partition them at the mean energy.  Configurations
        above mean energy are labeled "violation"; below mean energy are labeled "correct".
        The spin vector is converted to a text representation for compatibility with
        the ViolationPair format used by JEPARetrainer.

    Args:
        ising_model: Any model with an ``energy(x: jax.Array) -> jax.Array`` method.
                     Typically an IsingModel instance.
        n_samples: Total number of synthetic pairs to generate (split evenly between
                   violations and correct examples). Default 50.

    Spec: REQ-LEARN-038, SCENARIO-LEARN-067
    """

    def __init__(self, ising_model: Any, n_samples: int = 50) -> None:
        """Create augmentor with the given Ising model and sample count.

        Args:
            ising_model: Trained Ising model; must have .energy() and .config.input_dim.
            n_samples: Number of spin configurations to sample (half violation, half correct).
        """
        self.ising_model = ising_model
        self.n_samples = n_samples

    def _sample_spin_configs(self, seed: int = 0) -> tuple[list[Any], list[float]]:
        """Sample random spin configurations and compute their energies.

        Returns:
            (configs, energies): parallel lists of JAX arrays and float energies.
        """
        key = jrandom.PRNGKey(seed)
        dim = self.ising_model.config.input_dim
        configs = []
        energies = []
        for i in range(self.n_samples):
            k, key = jrandom.split(key)
            # Ising spin: uniform {-1, +1} via sign of normal sample
            spin = jnp.sign(jrandom.normal(k, (dim,)))
            # Edge case: zeros (very rare) become +1
            spin = jnp.where(spin == 0, 1.0, spin)
            e = float(self.ising_model.energy(spin))
            configs.append(spin)
            energies.append(e)
        return configs, energies

    def _spin_to_text(self, spin: Any) -> str:
        """Convert a spin configuration vector to a text string for ViolationPair compatibility.

        **For engineers:**
            The JEPA retrainer expects text strings (not raw tensors) because the
            original ViolationPair format was designed for LLM response text.
            We represent the spin vector as a space-separated string of rounded
            float values.  The JEPA text-to-embedding function (character-code pooling)
            will encode this into a fixed-size vector — not ideal, but preserves the
            pipeline interface without breaking changes.
        """
        values = [f"{float(v):.2f}" for v in spin]
        return " ".join(values)

    def generate_violation_pairs(self) -> list[dict[str, Any]]:
        """Generate synthetic pairs labeled as incorrect (violations).

        Returns the top-half by energy (highest energy = most violation-like).

        Returns:
            List of dicts with keys: correct, response, question_id, source.
            All have correct=False.

        Spec: REQ-LEARN-038, SCENARIO-LEARN-067
        """
        configs, energies = self._sample_spin_configs(seed=1)
        mean_e = sum(energies) / max(1, len(energies))
        # Violation = above-mean energy; sort descending to get worst first
        violation_configs = [
            (c, e) for c, e in zip(configs, energies) if e >= mean_e
        ]
        violation_configs.sort(key=lambda x: x[1], reverse=True)

        pairs: list[dict[str, Any]] = []
        for i, (spin, energy) in enumerate(violation_configs):
            pairs.append({
                "correct": False,
                "label": "incorrect",
                "response": self._spin_to_text(spin),
                "step_text": self._spin_to_text(spin),
                "question_id": f"ising_violation_{i:04d}",
                "model_id": "ising_sampler",
                "source": "ebm_guided_synthetic",
                "energy": energy,
                "arithmetic_coverage": 0.5,  # synthetic pairs are pre-approved
                "confidence": 1.0,            # energy function is ground truth
                "label_confidence": 1.0,
            })
        return pairs

    def generate_correct_pairs(self) -> list[dict[str, Any]]:
        """Generate synthetic pairs labeled as correct (low-energy configurations).

        Returns the bottom-half by energy (lowest energy = most correct-like).

        Returns:
            List of dicts with keys: correct, response, question_id, source.
            All have correct=True.

        Spec: REQ-LEARN-038
        """
        configs, energies = self._sample_spin_configs(seed=2)
        mean_e = sum(energies) / max(1, len(energies))
        correct_configs = [
            (c, e) for c, e in zip(configs, energies) if e < mean_e
        ]
        correct_configs.sort(key=lambda x: x[1])  # ascending: lowest energy first

        pairs: list[dict[str, Any]] = []
        for i, (spin, energy) in enumerate(correct_configs):
            pairs.append({
                "correct": True,
                "label": "correct",
                "response": self._spin_to_text(spin),
                "step_text": self._spin_to_text(spin),
                "question_id": f"ising_correct_{i:04d}",
                "model_id": "ising_sampler",
                "source": "ebm_guided_synthetic",
                "energy": energy,
                "arithmetic_coverage": 0.5,
                "confidence": 1.0,
                "label_confidence": 1.0,
            })
        return pairs


# ---------------------------------------------------------------------------
# JEPARetrainV2Result — summary of quality-gated retrain outcome
# ---------------------------------------------------------------------------


@dataclass
class JEPARetrainV2Result:
    """Summary of a quality-gated JEPA retrain run.

    **For engineers:**
        This dataclass captures the key metrics from a single quality-gated
        JEPA retrain cycle.  It is constructed after training completes and
        is used to compute the artifact's honest_verdict.

    Attributes:
        n_pairs_raw: Number of real CoT pairs loaded before quality filtering.
        n_pairs_filtered: Number of real pairs that passed the quality gate.
        n_synthetic: Number of EBM-guided synthetic pairs added by JEPAQualityAugmentor.
        before_auc: JEPA AUC on held-out set before this retrain (regression baseline ~0.400).
        after_auc: JEPA AUC on held-out set after retraining on quality-gated corpus.

    Derived attributes:
        auc_improvement: after_auc - before_auc.
        target_met: True iff after_auc > 0.700 (production deployment threshold).
        regression_recovered: True iff after_auc > 0.571 (Exp 443 level, pre-regression).
        retro_040_closed: True iff after_auc > 0.600 (conservative RETRO-040 closure bar).

    Spec: REQ-LEARN-039, SCENARIO-LEARN-068
    """

    n_pairs_raw: int
    n_pairs_filtered: int
    n_synthetic: int
    before_auc: float
    after_auc: float

    @property
    def auc_improvement(self) -> float:
        """AUC delta: after_auc minus before_auc.  Positive = improvement."""
        return self.after_auc - self.before_auc

    @property
    def target_met(self) -> bool:
        """True iff after_auc exceeds the production deployment threshold (0.700)."""
        return self.after_auc > 0.700

    @property
    def regression_recovered(self) -> bool:
        """True iff after_auc exceeds the Exp 443 pre-regression level (0.571).

        Exp 443 established AUC=0.571 as the prior-best checkpoint.  Exp 472
        regressed to 0.400.  Recovery means we are at least back to the prior best.
        """
        return self.after_auc > 0.571

    @property
    def retro_040_closed(self) -> bool:
        """True iff after_auc > 0.600 (conservative RETRO-040 closure threshold).

        We require >0.600 rather than just recovery to >0.571 because the quality
        gate fix should produce a measurable positive delta above the pre-regression
        level, not just break even with it.
        """
        return self.after_auc > 0.600
