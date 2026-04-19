"""AdversarialV4Result — per-model robustness metric for the adversarial v4 benchmark (Exp 504).

**What this measures (RETRO-039, Apple arXiv 2410.05229):**
    Apple showed that ALL major LLMs drop accuracy when irrelevant sentences are injected
    into math problems.  o1-preview dropped 92.7% -> 77.4% (-15.3pp).  GPT-4o dropped
    95% -> 88% (-7pp).  The finding: LLMs attend to ALL context, so distractors derail
    their reasoning chain.

    Carnot's ROBUSTNESS CLAIM: the Ising arithmetic verifier checks arithmetic constraints
    over extracted equation tokens only.  It ignores surrounding context words entirely.
    Therefore, the Carnot pipeline's adversarial drop should be SMALLER than the baseline
    drop — i.e., Carnot is MORE robust to irrelevant-sentence injection than the raw LLM.

**Four conditions for each model:**
    standard_baseline:   LLM only, standard GSM8K questions (no distractors, no pipeline)
    standard_pipeline:   Carnot verify-repair, standard GSM8K questions (no distractors)
    adversarial_baseline: LLM only, adversarial questions (distractors injected)
    adversarial_pipeline: Carnot verify-repair, adversarial questions (distractors injected)

**The headline metric — robustness_delta:**
    standard_drop_baseline = standard_baseline - adversarial_baseline
        How much the raw LLM degrades under adversarial injection.

    standard_drop_pipeline = standard_pipeline - adversarial_pipeline
        How much the Carnot pipeline degrades under adversarial injection.

    robustness_delta = standard_drop_baseline - standard_drop_pipeline
        Positive means Carnot dropped LESS than the baseline under adversarial injection.
        This is the RETRO-039 credibility result: Carnot is more robust.
        Negative means the pipeline somehow amplified the adversarial effect (unexpected).
        Zero means both dropped by the same amount (no robustness benefit).

    carnot_more_robust = robustness_delta > 0.

Spec: REQ-BENCH-049, REQ-BENCH-050, REQ-BENCH-051,
      SCENARIO-BENCH-068, SCENARIO-BENCH-069, SCENARIO-BENCH-070
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class AdversarialV4Result:
    """Per-model robustness result for the adversarial v4 benchmark (Exp 504).

    Parameters
    ----------
    model_id : str
        Human-readable model name (e.g. 'Gemma4-Q4KM', 'Qwen3.5-0.8B').
    standard_baseline : float
        Fraction correct: LLM only, standard (non-adversarial) questions.  Range [0, 1].
    standard_pipeline : float
        Fraction correct: Carnot verify-repair, standard questions.  Range [0, 1].
    adversarial_baseline : float
        Fraction correct: LLM only, adversarial (distractor-injected) questions.
    adversarial_pipeline : float
        Fraction correct: Carnot verify-repair, adversarial questions.
    n : int
        Number of questions in each condition (same n for all four conditions).

    Computed properties
    -------------------
    standard_improvement : float
        Pipeline benefit on standard questions: standard_pipeline - standard_baseline.
        Positive means Carnot helps even without distractors.

    adversarial_improvement : float
        Pipeline benefit on adversarial questions: adversarial_pipeline - adversarial_baseline.
        Positive means Carnot recovers some accuracy lost to distractors.

    standard_drop_baseline : float
        How much the RAW LLM drops from standard to adversarial:
        standard_baseline - adversarial_baseline.
        Positive = LLM regresses under distractors (expected per Apple finding).

    standard_drop_pipeline : float
        How much the Carnot PIPELINE drops from standard to adversarial:
        standard_pipeline - adversarial_pipeline.
        This should be smaller than standard_drop_baseline if Carnot is robust.

    robustness_delta : float
        standard_drop_baseline - standard_drop_pipeline.
        Positive = Carnot dropped LESS than baseline under adversarial injection.
        This is the headline RETRO-039 credibility metric.

    carnot_more_robust : bool
        True when robustness_delta > 0.  The primary boolean verdict.

    Spec: REQ-BENCH-049, REQ-BENCH-050, REQ-BENCH-051
    """

    model_id: str
    standard_baseline: float
    standard_pipeline: float
    adversarial_baseline: float
    adversarial_pipeline: float
    n: int

    @property
    def standard_improvement(self) -> float:
        """Carnot's benefit on standard (non-adversarial) questions.

        Positive means the pipeline helps even without distractor sentences.
        Zero or negative means the pipeline did not help on standard questions.
        """
        return self.standard_pipeline - self.standard_baseline

    @property
    def adversarial_improvement(self) -> float:
        """Carnot's benefit on adversarial (distractor-injected) questions.

        Positive means Carnot recovered some of the accuracy lost to distractors.
        This is a secondary metric; robustness_delta is the primary one.
        """
        return self.adversarial_pipeline - self.adversarial_baseline

    @property
    def standard_drop_baseline(self) -> float:
        """How much the RAW LLM loses on adversarial vs standard questions.

        Replicates the Apple paper finding: ALL tested LLMs showed a positive drop.
        A large positive value confirms that distractors hurt the baseline model.
        """
        return self.standard_baseline - self.adversarial_baseline

    @property
    def standard_drop_pipeline(self) -> float:
        """How much the Carnot PIPELINE loses on adversarial vs standard questions.

        If Carnot is robust to irrelevant context, this should be near zero even
        when standard_drop_baseline is large — the Ising verifier ignores distractor
        sentences so the pipeline accuracy should not regress.
        """
        return self.standard_pipeline - self.adversarial_pipeline

    @property
    def robustness_delta(self) -> float:
        """The headline RETRO-039 credibility metric.

        Positive = Carnot's adversarial drop is SMALLER than the baseline's adversarial
        drop.  This is the robustness claim: Carnot loses less accuracy under distractor
        injection than the raw LLM does.

        Formula: standard_drop_baseline - standard_drop_pipeline.
        """
        return self.standard_drop_baseline - self.standard_drop_pipeline

    @property
    def carnot_more_robust(self) -> bool:
        """True when robustness_delta > 0.

        This is the single boolean verdict: did Carnot prove more robust than
        the baseline to adversarial distractor injection?
        """
        return self.robustness_delta > 0

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dict with all fields and computed properties."""
        return {
            "model_id": self.model_id,
            "n": self.n,
            "standard_baseline": self.standard_baseline,
            "standard_pipeline": self.standard_pipeline,
            "adversarial_baseline": self.adversarial_baseline,
            "adversarial_pipeline": self.adversarial_pipeline,
            "standard_improvement": self.standard_improvement,
            "adversarial_improvement": self.adversarial_improvement,
            "standard_drop_baseline": self.standard_drop_baseline,
            "standard_drop_pipeline": self.standard_drop_pipeline,
            "robustness_delta": self.robustness_delta,
            "carnot_more_robust": self.carnot_more_robust,
        }
