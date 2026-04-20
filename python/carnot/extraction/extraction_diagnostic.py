"""Extraction diagnostic — per-extractor false-positive and true-positive rate measurement.

**Why this module exists (RETRO-033 miss #10 root cause investigation):**

    Exp 538 ran verify-repair on 25 live GSM8K questions and found
    honest_verdict='live_no_improvement_25q' — the pipeline made no measurable
    improvement.  Three hypotheses explain this:

        H1. VeriCoT or VPRM produces false positives on CORRECT IT-model responses,
            triggering spurious repair of valid answers (FP rate too high).
        H2. Violations are detected correctly, but the repair generates wrong fixes
            (low TP rate is not the problem; the repair step is).
        H3. Scoring bug elsewhere in the pipeline.

    This module tests H1 by running each extractor independently on a set of
    known-labeled responses (correct vs incorrect) and computing a confusion matrix.
    If the FP rate > 0.3, that is strong evidence for H1.

Spec: REQ-EXTRACT-030, SCENARIO-EXTRACT-055, SCENARIO-EXTRACT-056, SCENARIO-EXTRACT-057
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


# ---------------------------------------------------------------------------
# Protocol: what a "violation extractor" must look like for diagnostics
# ---------------------------------------------------------------------------


class ViolationExtractor(Protocol):
    """Minimal protocol for any extractor we want to diagnose.

    An extractor must expose one method: detect_violations(text) -> list[Any].
    The returned list is interpreted as: empty = no violations found,
    non-empty = at least one violation detected.

    Why a Protocol instead of ABC?
        VeriCoTStepValidator and VPRMArithmeticVerifier already exist with
        detect_violations() methods.  Using a Protocol avoids forcing a
        re-implementation or inheritance change on those classes while still
        letting type checkers verify compatibility.
    """

    def detect_violations(self, text: str) -> list[Any]:
        """Detect violations in *text* and return a (possibly empty) list."""
        ...  # pragma: no cover


# ---------------------------------------------------------------------------
# ExtractionDiagnosticResult — confusion matrix + derived rates
# ---------------------------------------------------------------------------


@dataclass
class ExtractionDiagnosticResult:
    """Per-extractor false-positive and true-positive rates from labeled responses.

    The confusion matrix uses the convention:
        - Positive class  = "has a violation" (i.e. the response is INCORRECT)
        - Negative class  = "no violation"    (i.e. the response is CORRECT)

    Attributes
    ----------
    extractor_name : str
        Human-readable name of the extractor (e.g. 'VeriCoTStepValidator',
        'VPRMArithmeticVerifier').
    n_tested : int
        Total number of labeled responses processed.
    n_violations_found : int
        Total responses where the extractor detected at least one violation,
        regardless of whether the response was actually correct.
    n_true_positive : int
        INCORRECT responses where the extractor correctly flagged a violation.
        (Extractor said "violation" AND response was wrong — good.)
    n_false_positive : int
        CORRECT responses where the extractor incorrectly flagged a violation.
        (Extractor said "violation" BUT response was right — bad, causes spurious repair.)
    n_true_negative : int
        CORRECT responses where the extractor correctly found no violation.
        (Extractor said "no violation" AND response was right — good.)
    n_false_negative : int
        INCORRECT responses where the extractor missed the violation.
        (Extractor said "no violation" BUT response was wrong — bad, silent failure.)
    tp_rate : float
        True positive rate = n_true_positive / n_actual_incorrect.
        Also called recall or sensitivity.  0.0 to 1.0.
        0.0 if there are no incorrect responses in the sample.
    fp_rate : float
        False positive rate = n_false_positive / n_actual_correct.
        Proportion of CORRECT answers incorrectly flagged for repair.
        0.0 to 1.0.  0.0 if there are no correct responses in the sample.

    Why include both raw counts AND rates?
        The rates are the primary diagnostic signal (H1: FP rate > 0.3).
        The raw counts are essential for interpreting the rates — an FP rate of
        0.5 on 2 correct responses is very different from 0.5 on 20 correct
        responses.  Both are needed to judge confidence in the measurement.

    Spec: REQ-EXTRACT-030, SCENARIO-EXTRACT-055
    """

    extractor_name: str
    n_tested: int
    n_violations_found: int
    n_true_positive: int
    n_false_positive: int
    n_true_negative: int
    n_false_negative: int
    tp_rate: float
    fp_rate: float
    per_question_flags: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dict for JSON artifact embedding."""
        return {
            "extractor_name": self.extractor_name,
            "n_tested": self.n_tested,
            "n_violations_found": self.n_violations_found,
            "n_true_positive": self.n_true_positive,
            "n_false_positive": self.n_false_positive,
            "n_true_negative": self.n_true_negative,
            "n_false_negative": self.n_false_negative,
            "tp_rate": self.tp_rate,
            "fp_rate": self.fp_rate,
            "per_question_flags": self.per_question_flags,
        }


# ---------------------------------------------------------------------------
# run_extractor_diagnostic — main diagnostic function
# ---------------------------------------------------------------------------


def run_extractor_diagnostic(
    extractor: ViolationExtractor,
    extractor_name: str,
    labeled_responses: list[dict[str, Any]],
) -> ExtractionDiagnosticResult:
    """Measure per-extractor FP and TP rates on labeled responses.

    For each response in *labeled_responses*, the extractor's detect_violations()
    method is called on the ``'response'`` field.  The result is compared against
    the ``'is_correct'`` label to build a confusion matrix.

    Confusion matrix logic per response:
        is_correct=False + violation_found=True  → True Positive  (TP)
        is_correct=True  + violation_found=True  → False Positive (FP) ← H1
        is_correct=True  + violation_found=False → True Negative  (TN)
        is_correct=False + violation_found=False → False Negative (FN)

    Parameters
    ----------
    extractor : ViolationExtractor
        Any object implementing detect_violations(text: str) -> list[Any].
        VeriCoTStepValidator and VPRMArithmeticVerifier both satisfy this protocol.
    extractor_name : str
        Human-readable label for the extractor (included in the result).
    labeled_responses : list[dict]
        Each entry must have:
          - ``'response'``: str — the full IT model response text.
          - ``'is_correct'``: bool — True iff the response was graded correct.
        Additional fields are ignored.

    Returns
    -------
    ExtractionDiagnosticResult
        Confusion matrix plus TP rate and FP rate for the extractor.

    Why pass extractor_name separately instead of reading it from the extractor?
        The extractor classes do not have a standardized .name attribute.
        Passing the name explicitly avoids relying on fragile __class__.__name__
        strings that could change if the class is renamed or wrapped.

    Spec: REQ-EXTRACT-030, SCENARIO-EXTRACT-055, SCENARIO-EXTRACT-056
    """
    n_tp = 0
    n_fp = 0
    n_tn = 0
    n_fn = 0
    n_violations_found = 0
    per_question_flags: list[dict[str, Any]] = []

    for entry in labeled_responses:
        response_text: str = entry["response"]
        is_correct: bool = bool(entry["is_correct"])

        violations = extractor.detect_violations(response_text)
        violation_found = len(violations) > 0
        if violation_found:
            n_violations_found += 1

        if not is_correct and violation_found:
            n_tp += 1
            cell = "TP"
        elif is_correct and violation_found:
            n_fp += 1
            cell = "FP"
        elif is_correct and not violation_found:
            n_tn += 1
            cell = "TN"
        else:  # not is_correct and not violation_found
            n_fn += 1
            cell = "FN"

        per_question_flags.append(
            {
                "is_correct": is_correct,
                "violation_found": violation_found,
                "cell": cell,
            }
        )

    n_actual_incorrect = n_tp + n_fn  # total responses that were wrong
    n_actual_correct = n_fp + n_tn    # total responses that were right

    # tp_rate = recall: of all wrong answers, how many did we catch?
    tp_rate = n_tp / n_actual_incorrect if n_actual_incorrect > 0 else 0.0

    # fp_rate: of all correct answers, how many did we incorrectly flag?
    fp_rate = n_fp / n_actual_correct if n_actual_correct > 0 else 0.0

    return ExtractionDiagnosticResult(
        extractor_name=extractor_name,
        n_tested=len(labeled_responses),
        n_violations_found=n_violations_found,
        n_true_positive=n_tp,
        n_false_positive=n_fp,
        n_true_negative=n_tn,
        n_false_negative=n_fn,
        tp_rate=tp_rate,
        fp_rate=fp_rate,
        per_question_flags=per_question_flags,
    )
