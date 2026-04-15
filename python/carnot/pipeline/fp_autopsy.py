"""FP Autopsy: systematic categorization of broken verify-repair cases.

**Researcher summary:**
    Exp 184 (3B scaling study) showed verify-repair has 0% net improvement at 3B:
    6 cases fixed, 6 cases broken.  This module provides the data types and
    classification logic to understand *why* the pipeline hurt on those broken cases.

    The central question: when verify-repair makes the answer WORSE than baseline,
    which of five root-cause patterns explains the degradation?

**Detailed explanation for engineers:**
    A "broken case" is defined operationally:
        baseline_answer == correct_answer   (LLM got it right without help)
        vr_answer != correct_answer          (verify-repair made it wrong)

    Five root causes are modelled:

    1. VALID_INTERMEDIATE — the extractor flagged an arithmetic expression that IS
       correct, but is an intermediate step, not the final answer.  E.g. the model
       wrote "10 - 3 = 7 ... so the answer is 14" and ArithmeticExtractor matched
       "10 - 3 = 7" as a violation because it lost context of which equation was
       the conclusion.  This is the most common expected FP type on chain-of-thought.

    2. PRECISION_LIMIT — the correct computation involves rounding or approximation
       (e.g. "1/3 ≈ 0.33") and ArithmeticExtractor's integer-only regex flags it as
       wrong.  The model's answer is actually correct to the problem's precision.

    3. REGEX_ARTIFACT — the regex pattern matched a substring that is not an
       arithmetic claim at all: a year ("2024 - 1 = 2023"), a phone extension, a
       list index, etc.  The violation is a pure regex false positive.

    4. REPAIR_DEGRADATION — the violation flagged was real (the model did make an
       arithmetic mistake) but the repair step produced a response that was even
       worse than the original.  The violation detection worked; the repair failed.

    5. UNCATEGORIZED — none of the above patterns match.  Manual review is needed.

    Classification is deterministic given the fields of AutopsyCase.  Tests
    exercise all five branches and the load_broken_cases / compute_category_distribution
    helpers so the module reaches 100% coverage in CI.

Spec: REQ-EXTRACT-013, REQ-EXTRACT-014,
      SCENARIO-EXTRACT-027, SCENARIO-EXTRACT-028,
      SCENARIO-EXTRACT-029, SCENARIO-EXTRACT-030
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


# ---------------------------------------------------------------------------
# FPCategory — the five root-cause buckets
# ---------------------------------------------------------------------------


class FPCategory(str, Enum):
    """Root-cause category for a broken verify-repair case (FP = false positive).

    **Detailed explanation for engineers:**
        Each value corresponds to a distinct failure mode of the verify-repair
        pipeline.  The string values are used directly in JSON artifacts so they
        are human-readable without needing an enum lookup table.

        VALID_INTERMEDIATE  — correct expression flagged because context was lost
        PRECISION_LIMIT     — rounding/approximation triggered integer regex
        REGEX_ARTIFACT      — regex matched non-arithmetic substring
        REPAIR_DEGRADATION  — real violation but repair made answer worse
        UNCATEGORIZED       — none of the above; needs manual inspection

    Spec: REQ-EXTRACT-013
    """

    VALID_INTERMEDIATE = "VALID_INTERMEDIATE"
    PRECISION_LIMIT = "PRECISION_LIMIT"
    REGEX_ARTIFACT = "REGEX_ARTIFACT"
    REPAIR_DEGRADATION = "REPAIR_DEGRADATION"
    UNCATEGORIZED = "UNCATEGORIZED"


# ---------------------------------------------------------------------------
# AutopsyCase — one broken verify-repair case
# ---------------------------------------------------------------------------


@dataclass
class AutopsyCase:
    """One broken verify-repair case ready for root-cause analysis.

    **Detailed explanation for engineers:**
        A case is "broken" when verify-repair produced a wrong answer on a question
        where the baseline model (no verification) produced the correct answer.
        The delta — baseline correct, VR wrong — is the signal that something in the
        pipeline introduced an error rather than fixed one.

        ``violations_flagged`` is a list of human-readable strings describing the
        constraints that the extractor(s) flagged as violations.  These are the
        raw description strings from ConstraintResult objects.

        ``fp_category`` starts as UNCATEGORIZED and is updated by ``categorize_fp``.

        ``evidence`` is a free-text field explaining which observation led to the
        assigned category.  It is empty until categorize_fp fills it in.

    Attributes:
        question:          The original question posed to the LLM.
        baseline_answer:   The answer produced by the model without verification.
        vr_answer:         The answer produced by the verify-repair pipeline.
        correct_answer:    The ground-truth answer.
        violations_flagged: Descriptions of the violations the extractor(s) found.
        fp_category:       Root-cause category (set by categorize_fp).
        evidence:          Human-readable explanation of the category assignment.

    Spec: REQ-EXTRACT-013
    """

    question: str
    baseline_answer: str
    vr_answer: str
    correct_answer: str
    violations_flagged: list[str] = field(default_factory=list)
    fp_category: FPCategory = FPCategory.UNCATEGORIZED
    evidence: str = ""


# ---------------------------------------------------------------------------
# categorize_fp — deterministic root-cause assignment
# ---------------------------------------------------------------------------

# Patterns that suggest VALID_INTERMEDIATE: the violation description contains a
# phrase indicating it refers to an intermediate step or sub-result.
_INTERMEDIATE_KEYWORDS = (
    "intermediate",
    "step",
    "sub",
    "partial",
    "first",
    "then",
    "so",
    "thus",
    "therefore",
)

# Patterns that suggest PRECISION_LIMIT: the violation text mentions approximate
# values, decimal fractions, or rounding.
_PRECISION_KEYWORDS = (
    "approx",
    "about",
    "roughly",
    "≈",
    "~",
    "0.",
    ".0",
    "fraction",
    "decimal",
    "round",
)

# Regex to detect likely REGEX_ARTIFACT patterns: years (4-digit numbers >= 1900),
# or cases where the claimed arithmetic result is a 4-digit number that looks like
# a year range (e.g. "2024 - 1 = 2023").
_YEAR_PATTERN = re.compile(r"\b(19|20)\d{2}\b")


def _looks_like_regex_artifact(violation: str) -> bool:
    """Return True when the violation string looks like a regex substring match.

    **Detailed explanation for engineers:**
        ArithmeticExtractor uses a simple pattern ``(-?\\d+)\\s*([+\\-])\\s*(-?\\d+)\\s*=\\s*(-?\\d+)``.
        This pattern matches ANY "a OP b = c" sequence in the text, including:
        - Years: "2024 - 1 = 2023" (calendar arithmetic in a word problem)
        - Indices: "item 1 + 0 = 1" (list counting)
        - Phone extensions, zip codes, etc.

        Heuristic: if either operand or the result is a 4-digit year-like number
        (1900–2099), classify as REGEX_ARTIFACT.  This is conservative: it will
        miss non-year artifacts, but those are rarer and fall into UNCATEGORIZED.
    """
    return bool(_YEAR_PATTERN.search(violation))


def categorize_fp(case: AutopsyCase) -> FPCategory:
    """Assign a root-cause FPCategory to a broken verify-repair case.

    **Detailed explanation for engineers:**
        Classification is a decision tree applied to the AutopsyCase fields:

        1. No violations flagged: either repair itself degraded a correct response
           (REPAIR_DEGRADATION) or nothing was detected (UNCATEGORIZED — should not
           happen for a broken case, but handled defensively).

        2. Any violation contains year-like numbers: REGEX_ARTIFACT (ArithmeticExtractor
           matched a non-arithmetic substring).

        3. Any violation mentions approximation/rounding keywords: PRECISION_LIMIT.

        4. Any violation mentions intermediate-step keywords AND the vr_answer
           differs from the correct_answer (which is always true for a broken case):
           VALID_INTERMEDIATE.  The extractor flagged a correct step as an error.

        5. Violations were found but vr_answer is still wrong: REPAIR_DEGRADATION
           (the violation was real but repair made it worse).

        6. None of the above: UNCATEGORIZED.

        The function also writes a short ``evidence`` string into the case so that
        the autopsy artifact can explain the assignment without re-running the
        classification.

    Args:
        case: An AutopsyCase with violations_flagged populated.

    Returns:
        The assigned FPCategory (also stored in case.fp_category).

    Spec: REQ-EXTRACT-013, SCENARIO-EXTRACT-027, SCENARIO-EXTRACT-028
    """
    violations = case.violations_flagged

    # No violations flagged at all: the pipeline ran but flagged nothing.
    # If VR still hurt, the repair step itself was the problem.
    if not violations:
        case.fp_category = FPCategory.REPAIR_DEGRADATION
        case.evidence = (
            "No violations were flagged, yet verify-repair changed the answer. "
            "The repair step degraded an already-correct baseline response."
        )
        return case.fp_category

    combined = " ".join(violations).lower()

    # REGEX_ARTIFACT: year-like numbers in violation text.
    if _looks_like_regex_artifact(" ".join(violations)):
        case.fp_category = FPCategory.REGEX_ARTIFACT
        case.evidence = (
            "Violation text contains year-like 4-digit numbers, suggesting "
            "ArithmeticExtractor matched a calendar expression rather than a "
            "genuine arithmetic claim."
        )
        return case.fp_category

    # PRECISION_LIMIT: approximation or decimal keywords present.
    if any(kw in combined for kw in _PRECISION_KEYWORDS):
        case.fp_category = FPCategory.PRECISION_LIMIT
        case.evidence = (
            "Violation text mentions approximation or decimal values, suggesting "
            "ArithmeticExtractor's integer-only matching flagged a correct "
            "rounded result as an error."
        )
        return case.fp_category

    # VALID_INTERMEDIATE: intermediate-step keywords in violation text.
    if any(kw in combined for kw in _INTERMEDIATE_KEYWORDS):
        case.fp_category = FPCategory.VALID_INTERMEDIATE
        case.evidence = (
            "Violation text contains intermediate-step language (e.g. 'step', "
            "'then', 'so'), suggesting the extractor flagged a correct intermediate "
            "result rather than the final answer."
        )
        return case.fp_category

    # Violations were flagged and no specific pattern matched: the violation was
    # real but the repair made it worse (REPAIR_DEGRADATION).
    case.fp_category = FPCategory.REPAIR_DEGRADATION
    case.evidence = (
        "Violations were flagged and appear genuine, but the repair step "
        "produced a worse answer than the original baseline response."
    )
    return case.fp_category


# ---------------------------------------------------------------------------
# load_broken_cases — extract broken VR cases from a benchmark result file
# ---------------------------------------------------------------------------


def load_broken_cases(results_path: str) -> list[AutopsyCase]:
    """Load cases where verify-repair hurt from a fullscale benchmark result file.

    **Detailed explanation for engineers:**
        A "broken case" satisfies:
            baseline_answer == correct_answer   (baseline was right)
            vr_answer       != correct_answer   (verify-repair was wrong)

        The function handles three shapes of result file:
        1. Files with a top-level ``cases`` list where each item has the four
           required fields directly.
        2. Files with a top-level ``per_question_results`` list.
        3. Files that only store aggregate accuracy numbers (like the Exp 316/328
           artifacts) — in this case, no per-question data is available and an
           empty list is returned without raising an error.  This is not a failure:
           the autopsy experiment handles the empty list by emitting an
           ``inconclusive`` artifact.

        The function never raises on missing or malformed files; it returns [] so
        the caller can decide how to handle the absence of data.

    Args:
        results_path: Path to a JSON file produced by a fullscale benchmark run.

    Returns:
        List of AutopsyCase objects (may be empty).

    Spec: REQ-EXTRACT-013
    """
    try:
        with open(results_path) as fh:
            data: dict[str, Any] = json.load(fh)
    except (FileNotFoundError, json.JSONDecodeError):
        return []

    # Try known per-question list keys.
    for key in ("cases", "per_question_results", "questions"):
        rows = data.get(key)
        if isinstance(rows, list) and rows:
            return _extract_broken_from_rows(rows)

    # No per-question data found — aggregate-only artifact.
    return []


def _extract_broken_from_rows(rows: list[dict[str, Any]]) -> list[AutopsyCase]:
    """Pull broken cases from a list of per-question result dicts.

    **Detailed explanation for engineers:**
        Expected keys per row:
            question, baseline_answer, vr_answer, correct_answer
        Optional keys:
            violations_flagged (list[str]) — extractor output descriptions
            evidence (str)

        Rows missing required keys are skipped silently to avoid crashing on
        partially-written checkpoint files.
    """
    broken: list[AutopsyCase] = []
    for row in rows:
        try:
            baseline = str(row["baseline_answer"])
            vr = str(row["vr_answer"])
            correct = str(row["correct_answer"])
        except KeyError:
            continue

        if baseline == correct and vr != correct:
            broken.append(
                AutopsyCase(
                    question=str(row.get("question", "")),
                    baseline_answer=baseline,
                    vr_answer=vr,
                    correct_answer=correct,
                    violations_flagged=list(row.get("violations_flagged", [])),
                    evidence=str(row.get("evidence", "")),
                )
            )
    return broken


# ---------------------------------------------------------------------------
# compute_category_distribution — summarise cases by category
# ---------------------------------------------------------------------------


def compute_category_distribution(
    cases: list[AutopsyCase],
) -> dict[FPCategory, int]:
    """Count the number of broken cases per FPCategory.

    **Detailed explanation for engineers:**
        Returns a dict with ALL five FPCategory values as keys (counts default
        to 0 for categories with no cases).  This ensures downstream code can
        always index into the result by category without a KeyError.

    Args:
        cases: List of AutopsyCase objects (fp_category should already be set).

    Returns:
        Dict mapping each FPCategory to its count.

    Spec: REQ-EXTRACT-013
    """
    dist: dict[FPCategory, int] = {cat: 0 for cat in FPCategory}
    for case in cases:
        dist[case.fp_category] += 1
    return dist
