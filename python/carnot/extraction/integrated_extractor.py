"""IntegratedExtractor — runs VeriCoT + VPRM in sequence and merges violations.

**Why a combined extractor (arXiv 2511.04662 + arXiv 2601.17223):**

    VeriCoT (vericot_validator.py) catches multi-step LOGICAL inconsistency by
    formalizing CoT steps to FOL and running Z3.  Its weak point is the LLM
    extraction step, which adds latency and can hallucinate premises.

    VPRM (vprm_verifier.py) catches single-step ARITHMETIC errors with zero LLM
    overhead via deterministic regex + identity checks.  Its weak point is pattern
    coverage: it only detects claims matching its six rule families.

    Running both in sequence gives complementary coverage:
    1. VeriCoT runs first (catches logical inconsistency across steps)
    2. VPRM runs second (catches arithmetic errors within individual steps)
    3. Optional ArithmeticExtractor fallback (equation-style regex, base-model output)

    Together these three layers cover the full spectrum from prose-style IT model
    output (VeriCoT + VPRM) to equation-style base model output (ArithmeticExtractor),
    with no single point of failure.

Spec: REQ-BENCH-015, SCENARIO-BENCH-035
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from carnot.extraction.vericot_validator import StepVerdict, VeriCoTStepValidator
from carnot.extraction.vprm_verifier import RuleVerdict, VPRMArithmeticVerifier

if TYPE_CHECKING:
    from carnot.pipeline.verify_repair import ArithmeticExtractor

# ---------------------------------------------------------------------------
# Violation — unified violation type for the integrated extractor
# ---------------------------------------------------------------------------


@dataclass
class Violation:
    """A single violation detected by the integrated extraction stack.

    Attributes
    ----------
    source : str
        Which extractor produced this violation: 'vericot', 'vprm', or 'arithmetic'.
        Used to populate the extractor_used field in the experiment artifact.
    step_text : str
        The natural-language reasoning step where the violation was found.
    detail : dict[str, Any]
        Raw verdict data serialized for the artifact.  For VeriCoT verdicts this
        contains step_idx, status, fol_premises.  For VPRM verdicts it contains
        rule_name, computed_value, stated_value, error_magnitude.

    Why a unified type?
        Downstream repair logic and artifact builders need a single list of
        violations to iterate over without knowing which extractor produced them.
        A unified type also makes it easy to deduplicate if both extractors flag
        the same step.
    """

    source: str
    step_text: str
    detail: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# IntegratedExtractor
# ---------------------------------------------------------------------------


class IntegratedExtractor:
    """Run VeriCoT + VPRM in sequence and return merged violations.

    This is the primary extraction stack for Exp 464 (REQ-BENCH-015).  It applies
    VeriCoTStepValidator first (slower, LLM-assisted, catches logical inconsistency),
    then VPRMArithmeticVerifier (fast, deterministic, catches arithmetic errors), and
    optionally an ArithmeticExtractor fallback for equation-style text.

    Parameters
    ----------
    vericot : VeriCoTStepValidator
        Pre-constructed VeriCoT validator.  Pass with use_mock=True for tests.
    vprm : VPRMArithmeticVerifier
        Pre-constructed VPRM verifier.  Uses default rule set when constructed
        with VPRMArithmeticVerifier().
    fallback : ArithmeticExtractor | None
        Optional equation-style regex extractor (base-model output fallback).
        When None, the fallback step is skipped.

    Spec: REQ-BENCH-015, SCENARIO-BENCH-035
    """

    def __init__(
        self,
        vericot: VeriCoTStepValidator,
        vprm: VPRMArithmeticVerifier,
        fallback: Any | None = None,
    ) -> None:
        self.vericot = vericot
        self.vprm = vprm
        self.fallback = fallback

    def extract(self, cot_text: str) -> list[Violation]:
        """Run all extractors on a CoT text and return merged violations.

        Order of execution:
        1. VeriCoT: detect_violations() — returns list[StepVerdict] (unsat only)
        2. VPRM: detect_violations()    — returns list[RuleVerdict] (failed only)
        3. Fallback (optional): extract() — returns list of Violation objects

        Each result is converted to a Violation with the appropriate source tag.

        Parameters
        ----------
        cot_text : str
            Full chain-of-thought text from an IT model response.

        Returns
        -------
        list[Violation]
            All violations detected by any extractor, in source order
            (vericot first, vprm second, fallback last).
        """
        violations: list[Violation] = []

        # Step 1: VeriCoT (FOL + Z3, catches logical inconsistency)
        vericot_verdicts: list[StepVerdict] = self.vericot.detect_violations(cot_text)
        for verdict in vericot_verdicts:
            violations.append(
                Violation(
                    source="vericot",
                    step_text=verdict.step_text,
                    detail=verdict.to_dict(),
                )
            )

        # Step 2: VPRM (deterministic arithmetic rules, catches computation errors)
        vprm_verdicts: list[RuleVerdict] = self.vprm.detect_violations(cot_text)
        for verdict in vprm_verdicts:
            violations.append(
                Violation(
                    source="vprm",
                    step_text="",
                    detail={
                        "rule_name": verdict.rule_name,
                        "passed": verdict.passed,
                        "computed_value": verdict.computed_value,
                        "stated_value": verdict.stated_value,
                        "error_magnitude": verdict.error_magnitude,
                    },
                )
            )

        # Step 3: Optional fallback (ArithmeticExtractor, equation-style regex)
        if self.fallback is not None:
            fallback_results = self.fallback.extract(cot_text, "arithmetic")
            for result in fallback_results:
                violations.append(
                    Violation(
                        source="arithmetic",
                        step_text=cot_text,
                        detail={"result": str(result)},
                    )
                )

        return violations

    def extractor_names_used(self, violations: list[Violation]) -> str:
        """Return comma-separated names of extractors that produced violations.

        Used to populate the extractor_used field in the experiment artifact.
        Returns 'none' when no violations were found.

        Parameters
        ----------
        violations : list[Violation]
            The violations returned by extract().
        """
        sources = sorted({v.source for v in violations})
        return ",".join(sources) if sources else "none"

    def detection_rate(self, test_samples: list[dict]) -> float:
        """Compute the fraction of samples where at least one violation was detected.

        Each sample must have a 'cot_text' key.  Samples without that key are skipped.
        Returns 0.0 for empty input to avoid division by zero.

        Parameters
        ----------
        test_samples : list[dict]
            Each dict must contain at least {'cot_text': str}.

        Returns
        -------
        float
            Fraction of samples (0.0–1.0) where extract() returned ≥1 violation.
        """
        if not test_samples:
            return 0.0
        detected = sum(
            1
            for s in test_samples
            if s.get("cot_text") and self.extract(s["cot_text"])
        )
        return detected / len(test_samples)
