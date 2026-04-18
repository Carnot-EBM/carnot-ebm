"""AdversarialBenchmarkResult — three-condition adversarial benchmark metric.

**Why this exists (GSM-Symbolic thesis, Exp 468):**
    Apple researchers (arXiv 2410.05229) showed that ALL tested LLMs drop accuracy
    on adversarial GSM-Symbolic variants (same logic, different numbers + irrelevant
    sentences), because LLMs pattern-match keywords rather than reason about arithmetic.
    Even o1-preview dropped from 92.7% → 77.4%.

    Carnot's thesis: EBM constraint verification (Ising) should MAINTAIN accuracy on
    adversarial variants because it verifies arithmetic constraints independently of
    irrelevant context.  The headline result is:

        adversarial improvement (B→C) > standard improvement (A→baseline Carnot)

    If this holds, Carnot fixes the failure mode that breaks ALL other approaches.

**Three conditions:**
    A: standard GSM8K baseline (LLM only, no pipeline)
    B: adversarial variant baseline (LLM only, no pipeline)
    C: adversarial variant + Carnot verify-repair (IntegratedExtractor)

    adversarial_drop = standard_acc - adversarial_baseline_acc  (how much does LLM regress?)
    carnot_adversarial_improvement = adversarial_carnot_acc - adversarial_baseline_acc
    carnot_standard_improvement = standard_carnot_acc - standard_acc  (optional; from Exp 464)
    thesis_confirmed = carnot_adversarial_improvement > carnot_standard_improvement

**Wilson CI:**
    ci_95_adversarial is computed on adversarial_carnot_acc (condition C accuracy),
    using the Wilson score interval so the bound stays in [0,1] even near extremes.

Spec: REQ-BENCH-020, REQ-BENCH-021, REQ-BENCH-022,
      SCENARIO-BENCH-039, SCENARIO-BENCH-040, SCENARIO-BENCH-041
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# z-score for 95% two-sided confidence interval (Agresti & Coull, 1998)
_Z95 = 1.959963984540054


def _wilson_ci(p: float, n: int) -> tuple[float, float]:
    """Return Wilson score 95% CI for proportion p with sample size n.

    Stays in [0, 1] even when p is near 0 or 1.  Unlike the normal approximation,
    this never produces negative bounds.  Standard for NLP proportion benchmarking.

    Formula (Agresti & Coull, 1998):
        center = (p̂ + z²/(2n)) / (1 + z²/n)
        margin = z * sqrt(p̂*(1-p̂)/n + z²/(4n²)) / (1 + z²/n)
    """
    n = max(n, 1)
    z = _Z95
    z2 = z * z
    center = (p + z2 / (2 * n)) / (1 + z2 / n)
    margin = (z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n))) / (1 + z2 / n)
    return (max(0.0, center - margin), min(1.0, center + margin))


@dataclass
class AdversarialBenchmarkResult:
    """Per-model result from the three-condition adversarial benchmark (Exp 468).

    Parameters
    ----------
    model_id : str
        Human-readable model name, e.g. 'Gemma4-E4B-it'.
    standard_acc : float
        Condition A accuracy: LLM on standard GSM8K, no pipeline.  Range [0, 1].
    adversarial_baseline_acc : float
        Condition B accuracy: LLM on adversarial GSM-Symbolic, no pipeline.  Range [0, 1].
    adversarial_carnot_acc : float
        Condition C accuracy: LLM on adversarial GSM-Symbolic WITH Carnot verify-repair.
    n_questions : int
        Number of questions evaluated per condition.
    carnot_standard_improvement : float
        Carnot's improvement on STANDARD questions (condition A Carnot - condition A baseline).
        Loaded from Exp 464/467 results when available; 0.0 if not available.
        This is the comparison baseline for the thesis: adversarial improvement must
        exceed this to confirm that Carnot is MORE useful on adversarial than standard.

    Computed properties
    -------------------
    adversarial_drop : float
        standard_acc - adversarial_baseline_acc.  Positive means LLM regressed on adversarial.
        Replicates the Apple finding: ALL models should show a positive drop.
    carnot_adversarial_improvement : float
        adversarial_carnot_acc - adversarial_baseline_acc.  How much Carnot recovers.
    thesis_confirmed : bool
        True when carnot_adversarial_improvement > carnot_standard_improvement.
        This is the headline result: Carnot's benefit is LARGER on adversarial variants.
    ci_95_adversarial : tuple[float, float]
        Wilson score 95% CI on adversarial_carnot_acc.

    Spec: REQ-BENCH-020, REQ-BENCH-021, REQ-BENCH-022,
          SCENARIO-BENCH-039, SCENARIO-BENCH-040, SCENARIO-BENCH-041
    """

    model_id: str
    standard_acc: float
    adversarial_baseline_acc: float
    adversarial_carnot_acc: float
    n_questions: int
    carnot_standard_improvement: float = 0.0

    @property
    def adversarial_drop(self) -> float:
        """How much accuracy the LLM loses on adversarial vs standard questions.

        Positive = regressed (as Apple found for ALL LLMs).
        Zero or negative = unusually robust LLM (rare; still valid to report).
        """
        return self.standard_acc - self.adversarial_baseline_acc

    @property
    def carnot_adversarial_improvement(self) -> float:
        """Carnot's accuracy recovery on adversarial questions (condition C - condition B).

        This is the primary metric: how much does verify-repair help on adversarial variants?
        """
        return self.adversarial_carnot_acc - self.adversarial_baseline_acc

    @property
    def thesis_confirmed(self) -> bool:
        """Return True when Carnot's adversarial improvement exceeds its standard improvement.

        The Carnot thesis: verify-repair should help MORE on adversarial variants than on
        standard ones, because Ising arithmetic verification is immune to the irrelevant
        context sentences that fool LLMs.

        If this is True, Carnot fixes the failure mode that breaks every other approach.
        """
        return self.carnot_adversarial_improvement > self.carnot_standard_improvement

    @property
    def ci_95_adversarial(self) -> tuple[float, float]:
        """Wilson score 95% CI on adversarial_carnot_acc (condition C).

        Used to assess statistical credibility of the adversarial improvement claim.
        """
        return _wilson_ci(self.adversarial_carnot_acc, self.n_questions)

    def to_dict(self) -> dict:
        """Return a JSON-serializable dict with all fields and computed properties."""
        lo, hi = self.ci_95_adversarial
        return {
            "model_id": self.model_id,
            "standard_acc": self.standard_acc,
            "adversarial_baseline_acc": self.adversarial_baseline_acc,
            "adversarial_carnot_acc": self.adversarial_carnot_acc,
            "n_questions": self.n_questions,
            "carnot_standard_improvement": self.carnot_standard_improvement,
            "adversarial_drop": self.adversarial_drop,
            "carnot_adversarial_improvement": self.carnot_adversarial_improvement,
            "thesis_confirmed": self.thesis_confirmed,
            "ci_95_adversarial": [lo, hi],
        }
