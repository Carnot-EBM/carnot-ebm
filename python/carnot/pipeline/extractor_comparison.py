"""ExtractorComparison: multi-extractor precision benchmark on live IT model responses.

**Researcher summary:**
    Experiment 311 benchmarked ArithmeticExtractor vs NL2Z3Extractor on a SYNTHETIC
    corpus of arithmetic word problems.  ArithmeticExtractor won, but synthetic text is
    not representative of real instruction-tuned (IT) model output: Gemma4-E4B-it rarely
    writes "47 + 28 = X" in its chain-of-thought; it writes fluent prose with implicit
    arithmetic.

    This module supports Exp 342, which runs ALL FOUR extractors on the same 50 live IT
    model responses and measures what matters for production:

    - **Violation rate**: how often does each extractor raise a flag?
    - **Estimated precision**: of those flags, what fraction are real errors?
      (Estimated using the Exp 331 FP taxonomy as a reference prior.)
    - **False positive category distribution**: which FP types dominate per extractor?
    - **Pairwise agreement**: which extractor pairs flag the same responses?

    The recommended_extractor is the extractor with the highest estimated_precision
    (tie-break: first in the input list).

**Detailed explanation for engineers:**
    Precision estimation without ground truth labels:
    The four FP categories from Exp 331 each carry a different "precision penalty":
    - VALID_INTERMEDIATE: high FP rate (step is correct but extractor flags it)
    - CORRECT_STEP:       high FP rate (entire step is correct)
    - PRECISION_LIMIT:    medium FP rate (rounding artefact, not a real error)
    - REGEX_ARTIFACT:     very high FP rate (regex matching text that is not arithmetic)
    - REPAIR_DEGRADATION: medium FP rate (error introduced by repair, not extraction)

    We estimate precision per extractor as:
        estimated_precision = 1.0 - weighted_fp_rate

    where weighted_fp_rate is the fraction of violations that fall into known FP
    categories, using the Exp 331 category distribution as the prior weight vector.

    When no violations are found (n_violations_found == 0), estimated_precision is
    reported as 1.0 (no flags → no false positives by definition).

    CI safety:
    - ArithmeticExtractor and CoTCircuitVerifier make no LLM calls: always run.
    - NL2Z3Extractor in CI mode (CARNOT_FORCE_LIVE not set): returns "unknown" → 0 violations.
    - VergeRefiner as extractor: uses NL2Z3Extractor internally → same CI behaviour.
    - All tests run with CARNOT_FORCE_LIVE=0.

Spec: REQ-EXTRACT-017, SCENARIO-EXTRACT-036, SCENARIO-EXTRACT-037

---

**Exp 367 additions (REQ-EXTRACT-023):**
    `ExtractorComparisonResult`, `run_extractor_comparison`, and
    `build_extractor_comparison_artifact` were added in Exp 367 to support the live GPU
    extraction comparison benchmark on Gemma4-E4B-it output.  These structures share the
    same module to keep all extractor-comparison logic in one place and avoid proliferating
    thin pipeline modules.

    Key difference from ExtractionBenchmarkResult (Exp 358):
    - `n_correct_questions` and `n_wrong_questions` are explicit (not implicit).
    - `fp_rate` is the field name (not `false_positive_rate`) for alignment with Exp 367 docs.
    - `build_extractor_comparison_artifact` uses schema `"carnot.extraction_comparison.v1"`.
    - `honest_verdict` is `"live_gpu_winner"` only when ALL results are live_gpu; stricter
      than Exp 358's per-extractor live_gpu check.

Spec: REQ-EXTRACT-017, SCENARIO-EXTRACT-036, SCENARIO-EXTRACT-037,
      REQ-EXTRACT-023, SCENARIO-EXTRACT-047, SCENARIO-EXTRACT-048
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

# ---------------------------------------------------------------------------
# FP category penalty weights (from Exp 331 analysis)
# ---------------------------------------------------------------------------

# Each key is a FP category; value is the estimated probability that a violation
# in that category is a false positive (not a real error).
# VALID_INTERMEDIATE is the dominant category and the most reliable FP signal.
_FP_PENALTY: dict[str, float] = {
    "VALID_INTERMEDIATE": 0.90,
    "CORRECT_STEP": 0.90,
    "PRECISION_LIMIT": 0.60,
    "REGEX_ARTIFACT": 0.95,
    "REPAIR_DEGRADATION": 0.50,
    "UNCATEGORIZED": 0.30,
}


# ---------------------------------------------------------------------------
# ExtractorResult dataclass
# ---------------------------------------------------------------------------


@dataclass
class ExtractorResult:
    """Per-extractor summary from a multi-extractor comparison run.

    **Detailed explanation for engineers:**
        One ExtractorResult is produced for each extractor that participates
        in compare_extractors().  The fields capture what fraction of responses
        triggered the extractor, the estimated true-positive rate (precision),
        and the breakdown of false-positive categories observed.

        violation_rate = n_violations_found / n_responses_checked.
        estimated_precision is in [0.0, 1.0]: higher means fewer false positives.

        fp_categories maps FP category name → count of violations attributed
        to that category.  The categories come from the Exp 331 taxonomy:
        VALID_INTERMEDIATE, CORRECT_STEP, PRECISION_LIMIT, REGEX_ARTIFACT,
        REPAIR_DEGRADATION, UNCATEGORIZED.

    Attributes:
        extractor_name:       Human-readable name (e.g. "ArithmeticExtractor").
        n_responses_checked:  Total number of responses this extractor processed.
        n_violations_found:   Number of responses where at least one violation
                              was extracted.
        violation_rate:       Fraction of responses that triggered at least one
                              violation (n_violations_found / n_responses_checked).
        estimated_precision:  Estimated fraction of flagged responses that are
                              genuine errors (not false positives).  Computed
                              from the FP category distribution using Exp 331
                              penalty weights.
        fp_categories:        Dict mapping FP category name to count of
                              violations assigned to that category.

    Spec: REQ-EXTRACT-017, SCENARIO-EXTRACT-036
    """

    extractor_name: str
    n_responses_checked: int
    n_violations_found: int
    violation_rate: float
    estimated_precision: float
    fp_categories: dict[str, int] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# ExtractorWrapper protocol
# ---------------------------------------------------------------------------


def _run_extractor_fn(
    extractor_fn: Callable[[str], list[Any]],
    response: str,
) -> list[Any]:
    """Call extractor_fn(response) and return violations list.

    **Detailed explanation for engineers:**
        extractor_fn is a callable that accepts a response string and returns
        a list of violation objects (ConstraintResult or similar).  An empty
        list means no violations detected.  Any exception is caught and treated
        as 0 violations to prevent one extractor crash from aborting the
        whole comparison.

    Args:
        extractor_fn: Callable that accepts response text, returns violations list.
        response:     The response text to check.

    Returns:
        List of violation objects (may be empty).
    """
    try:
        return extractor_fn(response) or []
    except Exception:  # noqa: BLE001
        return []


# ---------------------------------------------------------------------------
# FP category estimation
# ---------------------------------------------------------------------------


def _estimate_fp_categories(
    n_violations: int,
    fp_prior: dict[str, float],
) -> dict[str, int]:
    """Distribute n_violations across FP categories using the Exp 331 prior distribution.

    **Detailed explanation for engineers:**
        Without manual labelling we cannot know which specific category each
        violation belongs to.  Instead we use the category distribution from
        Exp 331 (passed in as fp_prior, e.g. {"VALID_INTERMEDIATE": 2,
        "PRECISION_LIMIT": 1, ...}) as a prior: we scale it so the category
        counts sum to n_violations.

        Rounding: we distribute using floor division and assign the remainder
        to the first category.  This keeps integer counts while summing to
        exactly n_violations.

        When n_violations == 0, returns an empty dict.

    Args:
        n_violations: Number of violations to distribute.
        fp_prior:     Dict of category -> raw count from Exp 331.

    Returns:
        Dict of category -> estimated count (sums to n_violations).
    """
    if n_violations == 0:
        return {}
    if not fp_prior:
        return {"UNCATEGORIZED": n_violations}

    total_prior = sum(fp_prior.values())
    if total_prior == 0:
        return {"UNCATEGORIZED": n_violations}

    categories: dict[str, int] = {}
    distributed = 0
    items = list(fp_prior.items())
    for i, (cat, cnt) in enumerate(items):
        frac = cnt / total_prior
        if i == len(items) - 1:
            # Last category absorbs the remainder to guarantee sum == n_violations.
            allocated = n_violations - distributed
        else:
            allocated = round(frac * n_violations)
        if allocated > 0:
            categories[cat] = allocated
            distributed += allocated

    # Guard: if distributed < n_violations due to all-zero allocations, use UNCATEGORIZED.
    remainder = n_violations - sum(categories.values())
    if remainder > 0:
        categories["UNCATEGORIZED"] = categories.get("UNCATEGORIZED", 0) + remainder

    return categories


def _compute_estimated_precision(fp_categories: dict[str, int], n_violations: int) -> float:
    """Compute estimated precision from FP category distribution.

    **Detailed explanation for engineers:**
        For each FP category c with count k, the penalty contribution is:
            penalty += k * _FP_PENALTY[c]
        Summed penalty / n_violations gives the estimated false-positive rate.
        estimated_precision = 1.0 - estimated_fp_rate.

        Clamped to [0.0, 1.0] to handle edge cases from floating-point rounding.

        When n_violations == 0: no flags → no false positives by definition,
        return 1.0.

    Args:
        fp_categories: Dict of category -> count.
        n_violations:  Total violations (denominator).

    Returns:
        Estimated precision in [0.0, 1.0].
    """
    if n_violations == 0:
        return 1.0

    penalty = sum(
        count * _FP_PENALTY.get(cat, 0.30)
        for cat, count in fp_categories.items()
    )
    fp_rate = penalty / n_violations
    return max(0.0, min(1.0, 1.0 - fp_rate))


# ---------------------------------------------------------------------------
# compare_extractors
# ---------------------------------------------------------------------------


def compare_extractors(
    responses: list[str],
    extractors: list[tuple[str, Callable[[str], list[Any]]]],
    fp_prior: dict[str, int] | None = None,
) -> list[ExtractorResult]:
    """Run each extractor on every response and compute per-extractor statistics.

    **Detailed explanation for engineers:**
        For each (name, extractor_fn) pair in `extractors`:
        1. Call extractor_fn(response) for every response in `responses`.
        2. Count responses where at least one violation was returned.
        3. Use _estimate_fp_categories() to distribute violations across FP categories.
        4. Compute estimated_precision via _compute_estimated_precision().

        The fp_prior controls the FP category distribution.  When None, defaults
        to the Exp 331 distribution: {"VALID_INTERMEDIATE": 2, "PRECISION_LIMIT": 1,
        "REGEX_ARTIFACT": 1, "REPAIR_DEGRADATION": 2, "UNCATEGORIZED": 0}.

    Args:
        responses:   List of response strings to benchmark.
        extractors:  List of (name, callable) pairs.  callable(response: str) → list.
        fp_prior:    Dict of FP category -> count from Exp 331 (optional).

    Returns:
        List of ExtractorResult, one per extractor, in the same order as `extractors`.

    Spec: REQ-EXTRACT-017, SCENARIO-EXTRACT-036
    """
    if fp_prior is None:
        fp_prior = {
            "VALID_INTERMEDIATE": 2,
            "PRECISION_LIMIT": 1,
            "REGEX_ARTIFACT": 1,
            "REPAIR_DEGRADATION": 2,
            "UNCATEGORIZED": 0,
        }

    n = len(responses)
    results: list[ExtractorResult] = []

    for name, extractor_fn in extractors:
        n_violations = 0

        for response in responses:
            violations = _run_extractor_fn(extractor_fn, response)
            if violations:
                n_violations += 1

        violation_rate = n_violations / n if n > 0 else 0.0

        fp_categories = _estimate_fp_categories(n_violations, fp_prior)
        estimated_precision = _compute_estimated_precision(fp_categories, n_violations)

        results.append(
            ExtractorResult(
                extractor_name=name,
                n_responses_checked=n,
                n_violations_found=n_violations,
                violation_rate=round(violation_rate, 4),
                estimated_precision=round(estimated_precision, 4),
                fp_categories=fp_categories,
            )
        )

    return results


# ---------------------------------------------------------------------------
# build_comparison_artifact
# ---------------------------------------------------------------------------


def build_comparison_artifact(
    results: list[ExtractorResult],
) -> dict[str, Any]:
    """Build a standardised artifact dict from a list of ExtractorResult.

    **Detailed explanation for engineers:**
        Assembles the JSON-serialisable artifact that the experiment script
        writes to disk.  The recommended_extractor is the extractor with the
        highest estimated_precision; ties are broken by list order (first wins).

        best_precision is the maximum estimated_precision across all results.
        When results is empty, best_precision is 0.0 and recommended_extractor
        is an empty string.

        The schema field identifies the artifact format for downstream readers.

    Args:
        results: List of ExtractorResult from compare_extractors().

    Returns:
        Dict with keys: schema, best_precision, recommended_extractor,
        extractor_results (list of per-extractor dicts).

    Spec: REQ-EXTRACT-017, SCENARIO-EXTRACT-037
    """
    if not results:
        return {
            "comparison_schema": "carnot.extractor_comparison.v1",
            "best_precision": 0.0,
            "recommended_extractor": "",
            "extractor_results": [],
        }

    best = max(results, key=lambda r: r.estimated_precision)

    return {
        "comparison_schema": "carnot.extractor_comparison.v1",
        "best_precision": best.estimated_precision,
        "recommended_extractor": best.extractor_name,
        "extractor_results": [
            {
                "extractor_name": r.extractor_name,
                "n_responses_checked": r.n_responses_checked,
                "n_violations_found": r.n_violations_found,
                "violation_rate": r.violation_rate,
                "estimated_precision": r.estimated_precision,
                "fp_categories": r.fp_categories,
            }
            for r in results
        ],
    }


# ===========================================================================
# Exp 367: Live GPU extraction comparison (REQ-EXTRACT-023)
# ===========================================================================
# The classes and functions below were added in Exp 367.  They are separate
# from the Exp 342 structures above and use a different schema version.
# ===========================================================================


@dataclass
class ExtractorComparisonResult:
    """Per-extractor metrics from a live extraction comparison run (Exp 367).

    **Detailed explanation for engineers:**
        This dataclass mirrors ExtractionBenchmarkResult (Exp 358) but:
        1. Splits n_questions into n_correct_questions + n_wrong_questions explicitly,
           making per-extractor tables self-explanatory without a separate key.
        2. Uses `fp_rate` (not `false_positive_rate`) to match Exp 367 documentation.
        3. Carries inference_mode="live_gpu" only when CARNOT_FORCE_LIVE=1 was set for
           the outer inference loop, enabling the honest_verdict gate in the artifact.

        detection_rate = n_true_positives / (n_true_positives + n_false_negatives)
            where n_false_negatives = n_wrong_questions - n_true_positives.
        fp_rate = n_false_positives / (n_false_positives + n_true_negatives)
            where n_true_negatives = n_correct_questions - n_false_positives.
        Both are 0.0 when denominator is zero.

    Attributes:
        extractor_name:      Short identifier (e.g. "arithmetic", "llm", "z3").
        n_questions:         Total (question, response) pairs evaluated.
        n_correct_questions: Pairs labelled as correct (ground truth).
        n_wrong_questions:   Pairs labelled as wrong (ground truth).
        n_true_positives:    Extractor fired on a known-wrong response.
        n_false_positives:   Extractor fired on a known-correct response.
        detection_rate:      TP / (TP + FN).
        fp_rate:             FP / (FP + TN).
        inference_mode:      "live_gpu" or "simulated".

    Spec: REQ-EXTRACT-023, SCENARIO-EXTRACT-047
    """

    extractor_name: str
    n_questions: int
    n_correct_questions: int
    n_wrong_questions: int
    n_true_positives: int
    n_false_positives: int
    detection_rate: float
    fp_rate: float
    inference_mode: str


def run_extractor_comparison(
    extractor_name: str,
    detector_fn: Callable[[str, str], bool],
    questions: list[dict],
    ground_truth_wrong: list[bool],
    inference_mode: str,
) -> "ExtractorComparisonResult":
    """Run one extractor over a labelled set of (question, response) pairs (Exp 367).

    **Detailed explanation for engineers:**
        Each entry in `questions` must have "question" (str) and "response" (str).
        `ground_truth_wrong[i] == True` means the i-th response is known-incorrect.

        detector_fn(question, response) -> bool:
        - True AND wrong   -> True Positive (TP)
        - True AND correct -> False Positive (FP)
        - False AND wrong  -> False Negative (implicit, in denominator)
        - False AND correct -> True Negative (implicit)

        Both rates are 0.0 when denominator is zero.  Rounded to 6 decimal places.

    Args:
        extractor_name:     Short identifier (e.g. "arithmetic").
        detector_fn:        Callable (question, response) -> bool.
        questions:          List of dicts with "question" and "response" keys.
        ground_truth_wrong: Parallel bool list; True means response is known-wrong.
        inference_mode:     "live_gpu" or "simulated".

    Returns:
        ExtractorComparisonResult with all counts and rates populated.

    Raises:
        ValueError: When questions and ground_truth_wrong have different lengths.

    Spec: REQ-EXTRACT-023, SCENARIO-EXTRACT-047
    """
    if len(questions) != len(ground_truth_wrong):
        raise ValueError(
            f"questions length ({len(questions)}) must equal "
            f"ground_truth_wrong length ({len(ground_truth_wrong)})"
        )

    n_questions = len(questions)
    n_wrong_questions = sum(1 for g in ground_truth_wrong if g)
    n_correct_questions = n_questions - n_wrong_questions

    n_true_positives = 0
    n_false_positives = 0

    for item, is_wrong in zip(questions, ground_truth_wrong):
        violated = detector_fn(item["question"], item["response"])
        if violated:
            if is_wrong:
                n_true_positives += 1
            else:
                n_false_positives += 1

    n_false_negatives = n_wrong_questions - n_true_positives
    n_true_negatives = n_correct_questions - n_false_positives

    detection_rate = (
        n_true_positives / (n_true_positives + n_false_negatives)
        if (n_true_positives + n_false_negatives) > 0
        else 0.0
    )
    fp_rate_val = (
        n_false_positives / (n_false_positives + n_true_negatives)
        if (n_false_positives + n_true_negatives) > 0
        else 0.0
    )

    return ExtractorComparisonResult(
        extractor_name=extractor_name,
        n_questions=n_questions,
        n_correct_questions=n_correct_questions,
        n_wrong_questions=n_wrong_questions,
        n_true_positives=n_true_positives,
        n_false_positives=n_false_positives,
        detection_rate=round(detection_rate, 6),
        fp_rate=round(fp_rate_val, 6),
        inference_mode=inference_mode,
    )


def build_extractor_comparison_artifact(
    results: list["ExtractorComparisonResult"],
) -> dict:
    """Combine Exp 367 per-extractor results into a schema-versioned summary dict.

    **Detailed explanation for engineers:**
        Winner selection: highest detection_rate; tiebreak: lowest fp_rate;
        final tiebreak: lexicographic extractor_name for determinism.

        honest_verdict contract (REQ-EXTRACT-023) -- stricter than Exp 358:
        - "live_gpu_winner": ALL results have inference_mode=="live_gpu".
        - "simulated_no_verdict": ANY result has inference_mode!="live_gpu".
          Simulated runs CANNOT claim a headline result.
        - "insufficient_data": empty results list.

        Schema "carnot.extraction_comparison.v1" distinguishes this from
        Exp 342's "carnot.extractor_comparison.v1".

    Args:
        results: List of ExtractorComparisonResult (one per extractor).

    Returns:
        dict with keys: schema, per_extractor_results, winner_extractor,
        honest_verdict, n_extractors.

    Spec: REQ-EXTRACT-023, SCENARIO-EXTRACT-048
    """
    if not results:
        return {
            "schema": "carnot.extraction_comparison.v1",
            "per_extractor_results": [],
            "winner_extractor": None,
            "honest_verdict": "insufficient_data",
            "n_extractors": 0,
        }

    per_extractor = [
        {
            "extractor_name": r.extractor_name,
            "n_questions": r.n_questions,
            "n_correct_questions": r.n_correct_questions,
            "n_wrong_questions": r.n_wrong_questions,
            "n_true_positives": r.n_true_positives,
            "n_false_positives": r.n_false_positives,
            "detection_rate": r.detection_rate,
            "fp_rate": r.fp_rate,
            "inference_mode": r.inference_mode,
        }
        for r in results
    ]

    # Winner: highest detection_rate; tiebreak: lowest fp_rate; final tiebreak:
    # lexicographically smallest name for determinism ("arithmetic" before "z3").
    # Using sorted()[0] with negated detection_rate to get highest-first ascending
    # sort while keeping fp_rate and name in natural ascending order.
    winner_result = sorted(
        results,
        key=lambda r: (-r.detection_rate, r.fp_rate, r.extractor_name),
    )[0]

    # honest_verdict: ALL must be live_gpu for a win claim
    if any(r.inference_mode != "live_gpu" for r in results):
        honest_verdict = "simulated_no_verdict"
    else:
        honest_verdict = "live_gpu_winner"

    return {
        "schema": "carnot.extraction_comparison.v1",
        "per_extractor_results": per_extractor,
        "winner_extractor": winner_result.extractor_name,
        "honest_verdict": honest_verdict,
        "n_extractors": len(results),
    }
