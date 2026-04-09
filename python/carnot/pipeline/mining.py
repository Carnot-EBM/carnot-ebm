"""Failure mining: analyze pipeline false negatives to discover missing extractors.

**Researcher summary:**
    When the verify-repair pipeline says a wrong answer is correct, that's a
    false negative — the pipeline failed to extract a constraint that would
    have caught the error. This module analyzes those failures to find
    CATEGORIES of claims the pipeline consistently misses, and proposes new
    regex/AST patterns to cover them.

**Detailed explanation for engineers:**
    The FailureAnalyzer class takes a batch of (question, response, ground_truth)
    triples, runs them through the pipeline in verify-only mode, and identifies
    false negatives: cases where the response is wrong but the pipeline reported
    no violations (verified=True).

    For each false negative, it examines the response text to classify which
    types of claims were present but NOT extracted as constraints. The six
    uncovered claim categories are:

    1. arithmetic_chain — Multi-step calculations where intermediate results
       aren't expressed as "A + B = C" (e.g., "first add 3 and 5 to get 8,
       then multiply by 2"). The current ArithmeticExtractor only catches
       explicit "X op Y = Z" patterns.

    2. implicit_logic — Logical relationships embedded in prose without
       explicit "if/then" markers (e.g., "since X is true, Y must follow").
       The LogicExtractor requires "if...then" or "but not" patterns.

    3. world_knowledge — Factual claims that require external knowledge to
       verify (e.g., "Python was created in 1991", "the speed of light is
       3e8 m/s"). The NLExtractor can parse "X is Y" but can't check truth.

    4. code_semantics — Claims about what code DOES rather than its syntactic
       structure (e.g., "this function runs in O(n) time", "the loop
       terminates"). The CodeExtractor checks types/bounds but not behavior.

    5. comparison — Relative claims like "X is greater than Y", "A is the
       largest", "B comes before C". No current extractor handles orderings.

    6. negation — Claims about what is NOT true or what doesn't happen (e.g.,
       "the function never returns None", "there are no duplicates"). The
       LogicExtractor handles some negation but misses many patterns.

    The FailureReport includes frequency counts per category and suggests
    new regex patterns for the top categories. These patterns are tested
    against the false negatives to estimate how much coverage they'd add.

Spec: REQ-VERIFY-001, REQ-VERIFY-002, REQ-VERIFY-003, SCENARIO-VERIFY-005
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from carnot.pipeline.extract import AutoExtractor, ConstraintResult
from carnot.pipeline.verify_repair import VerificationResult, VerifyRepairPipeline


# ---------------------------------------------------------------------------
# Claim category detection patterns
# ---------------------------------------------------------------------------

# Each category maps to a list of regex patterns that detect claims in that
# category. These are DETECTION patterns (does the response contain this type
# of claim?) not EXTRACTION patterns (can we verify the claim?). The gap
# between detection and extraction is exactly what we're measuring.

CATEGORY_DETECTORS: dict[str, list[re.Pattern[str]]] = {
    "arithmetic_chain": [
        # "first ... then ..." multi-step calculation language
        re.compile(r"first\s+.+?\s+then\s+", re.IGNORECASE),
        # "step 1: ... step 2: ..." numbered steps with numbers
        re.compile(r"step\s+\d+\s*:", re.IGNORECASE),
        # "which gives us N" or "resulting in N" intermediate results
        re.compile(
            r"(?:which\s+gives\s+(?:us\s+)?|resulting\s+in\s+|"
            r"that\s+(?:gives|makes|equals)\s+)-?\d+",
            re.IGNORECASE,
        ),
        # "N * M" or "N / M" without "= Z" (implicit result)
        re.compile(r"-?\d+\s*[*/]\s*-?\d+(?!\s*=)"),
    ],
    "implicit_logic": [
        # "since X, Y" or "because X, Y" causal reasoning
        re.compile(r"(?:since|because|as)\s+.+?,\s+.+", re.IGNORECASE),
        # "therefore" / "thus" / "hence" / "so" conclusions
        re.compile(
            r"(?:therefore|thus|hence|consequently|so)\s+.+",
            re.IGNORECASE,
        ),
        # "X means Y" or "X implies Y" without "if"
        re.compile(r"\b(?:means|implies|indicates)\s+(?:that\s+)?.+", re.IGNORECASE),
        # "must be" / "has to be" / "should be" modal necessity
        re.compile(r"(?:must|has\s+to|should|would)\s+be\s+", re.IGNORECASE),
    ],
    "world_knowledge": [
        # Year claims: "in 1991", "created in 2005", etc.
        re.compile(r"in\s+(?:19|20)\d{2}\b"),
        # Named entity + "is/was" + property (not already caught by NL "X is Y"
        # because the claim requires KB lookup to verify)
        re.compile(
            r"[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:is|was|are|were)\s+"
            r"(?:a|an|the)\s+",
        ),
        # Numeric constants: "speed of light is 3e8", "pi is 3.14"
        re.compile(r"(?:speed|constant|value|rate|population)\s+.+?\d+", re.IGNORECASE),
        # "invented by" / "discovered by" / "founded by"
        re.compile(r"(?:invented|discovered|founded|created|written)\s+by\s+", re.IGNORECASE),
    ],
    "code_semantics": [
        # Big-O notation: "O(n)", "O(n log n)", "O(1)"
        re.compile(r"O\(\s*(?:1|n|n\s*(?:log\s*n|\*\s*n|²|\^2))\s*\)"),
        # "the function/algorithm/loop does/returns/handles ..."
        re.compile(
            r"(?:the\s+)?(?:function|algorithm|loop|code|method)\s+"
            r"(?:does|returns|handles|computes|calculates|produces)",
            re.IGNORECASE,
        ),
        # "terminates" / "halts" / "converges"
        re.compile(r"\b(?:terminates|halts|converges|diverges)\b", re.IGNORECASE),
        # "correctly" / "correctly handles" behavioral claims
        re.compile(r"\bcorrectly\s+(?:handles|computes|returns|processes)\b", re.IGNORECASE),
    ],
    "comparison": [
        # "X is greater/less/larger/smaller than Y"
        re.compile(
            r"\b(?:greater|less|larger|smaller|higher|lower|more|fewer|bigger)"
            r"\s+than\b",
            re.IGNORECASE,
        ),
        # "X is the largest/smallest/best/worst"
        re.compile(
            r"\bis\s+the\s+(?:largest|smallest|biggest|best|worst|highest|lowest|"
            r"most|least|fastest|slowest)\b",
            re.IGNORECASE,
        ),
        # "X > Y" or "X < Y" or "X >= Y" symbolic comparisons
        re.compile(r"-?\d+\.?\d*\s*[<>]=?\s*-?\d+\.?\d*"),
        # "before/after X" temporal ordering
        re.compile(r"\b(?:before|after|prior\s+to|following)\s+", re.IGNORECASE),
    ],
    "negation": [
        # "never" / "no" + noun claims
        re.compile(r"\bnever\s+\w+", re.IGNORECASE),
        re.compile(r"\bno\s+(?:duplicate|error|overflow|exception|bug|issue)s?\b", re.IGNORECASE),
        # "not possible" / "impossible" / "cannot happen"
        re.compile(r"\b(?:not\s+possible|impossible|cannot\s+happen)\b", re.IGNORECASE),
        # "without" + gerund: "without causing", "without losing"
        re.compile(r"\bwithout\s+\w+ing\b", re.IGNORECASE),
        # "none of" claims
        re.compile(r"\bnone\s+of\b", re.IGNORECASE),
    ],
}

# Categories in canonical order for consistent reporting.
CLAIM_CATEGORIES = list(CATEGORY_DETECTORS.keys())


# ---------------------------------------------------------------------------
# Suggested extraction patterns — new regexes that COULD extract constraints
# ---------------------------------------------------------------------------

# For each category, these are candidate patterns that could be added to the
# pipeline as new extractors. They attempt to not just DETECT but EXTRACT
# verifiable structure from the claim.

SUGGESTED_PATTERNS: dict[str, list[dict[str, str]]] = {
    "arithmetic_chain": [
        {
            "name": "intermediate_result",
            "pattern": r"(?:gives|makes|equals|results?\s+in|is)\s+(-?\d+\.?\d*)",
            "description": "Extract intermediate calculation results for chain verification",
        },
        {
            "name": "implicit_multiplication",
            "pattern": r"(-?\d+\.?\d*)\s*[*/]\s*(-?\d+\.?\d*)",
            "description": "Extract multiplication/division without explicit '=' result",
        },
    ],
    "implicit_logic": [
        {
            "name": "causal_therefore",
            "pattern": r"(?:therefore|thus|hence|so)\s+(.+?)(?:\.|$)",
            "description": "Extract conclusions from causal reasoning chains",
        },
        {
            "name": "since_because",
            "pattern": r"(?:since|because)\s+(.+?),\s+(.+?)(?:\.|$)",
            "description": "Extract premise-conclusion pairs from 'since/because' sentences",
        },
    ],
    "world_knowledge": [
        {
            "name": "entity_property",
            "pattern": (
                r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:is|was)\s+"
                r"(?:a|an|the)\s+(.+?)(?:\.|$)"
            ),
            "description": "Extract entity-property claims for KB lookup verification",
        },
    ],
    "code_semantics": [
        {
            "name": "complexity_claim",
            "pattern": r"O\(\s*((?:1|n|n\s*(?:log\s*n|\*\s*n|²|\^2)))\s*\)",
            "description": "Extract Big-O complexity claims for AST-based verification",
        },
    ],
    "comparison": [
        {
            "name": "numeric_comparison",
            "pattern": r"(-?\d+\.?\d*)\s*([<>]=?)\s*(-?\d+\.?\d*)",
            "description": "Extract and verify numeric comparison claims",
        },
        {
            "name": "superlative_claim",
            "pattern": (
                r"(\w+(?:\s+\w+)?)\s+is\s+the\s+"
                r"(largest|smallest|biggest|best|worst|highest|lowest|"
                r"most|least|fastest|slowest)\b"
            ),
            "description": "Extract superlative claims about entities",
        },
    ],
    "negation": [
        {
            "name": "never_claim",
            "pattern": r"(\w+(?:\s+\w+)?)\s+never\s+(\w+(?:\s+\w+)?)",
            "description": "Extract 'X never Y' claims as negated constraints",
        },
        {
            "name": "no_entity_claim",
            "pattern": r"\bno\s+(duplicate|error|overflow|exception|bug|issue)s?\b",
            "description": "Extract 'no X' absence claims",
        },
    ],
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class FalseNegative:
    """A single false negative: wrong answer the pipeline said was correct.

    **Detailed explanation for engineers:**
        Captures all the context needed to understand WHY the pipeline missed
        this error. The ``uncovered_categories`` dict maps each claim category
        to the number of detected-but-not-extracted claims of that type. The
        ``extracted_constraints`` list shows what the pipeline DID catch (which
        wasn't enough to flag the wrong answer).

    Attributes:
        question: The original question.
        response: The LLM's (wrong) response.
        ground_truth: The correct answer.
        verification: The pipeline's (incorrect) verification result.
        extracted_constraints: What the pipeline did extract.
        uncovered_categories: Claim types detected in response but not
            covered by extracted constraints. Maps category -> count.

    Spec: REQ-VERIFY-003, SCENARIO-VERIFY-005
    """

    question: str
    response: str
    ground_truth: str
    verification: VerificationResult
    extracted_constraints: list[ConstraintResult]
    uncovered_categories: dict[str, int] = field(default_factory=dict)


@dataclass
class FailureReport:
    """Aggregated analysis of pipeline false negatives.

    **Detailed explanation for engineers:**
        The main output of FailureAnalyzer.analyze(). Contains the list of
        individual false negatives, aggregate category frequency counts, and
        suggested extraction patterns ranked by potential impact (how many
        false negatives each pattern would help catch).

    Attributes:
        total_questions: Total questions analyzed.
        total_wrong: Questions where the response was wrong.
        false_negatives: Wrong answers the pipeline said were correct.
        false_negative_rate: false_negatives / total_wrong (0 if no wrong).
        category_counts: How many false negatives had uncovered claims in
            each category. Higher = more important gap to fill.
        suggested_patterns: Top suggested patterns sorted by estimated
            coverage improvement. Each entry has name, pattern, category,
            estimated_catch_count (how many false negatives it would match).

    Spec: REQ-VERIFY-003, SCENARIO-VERIFY-005
    """

    total_questions: int
    total_wrong: int
    false_negatives: list[FalseNegative]
    false_negative_rate: float
    category_counts: dict[str, int]
    suggested_patterns: list[dict[str, Any]]


# ---------------------------------------------------------------------------
# FailureAnalyzer
# ---------------------------------------------------------------------------


class FailureAnalyzer:
    """Analyze verify-repair pipeline false negatives to discover missing extractors.

    **Researcher summary:**
        Takes a batch of questions with known-wrong responses, runs them through
        the pipeline, finds false negatives, categorizes what types of claims
        the pipeline missed, and suggests new patterns to improve coverage.
        This is the first step toward FR-11 (autonomous self-improvement).

    **Detailed explanation for engineers:**
        The analysis proceeds in four phases:

        Phase 1 — Identify false negatives: Run each (question, response) pair
        through the pipeline's verify() method. If the response is wrong (per
        ground_truth) but the pipeline says verified=True, that's a false
        negative.

        Phase 2 — Categorize uncovered claims: For each false negative, scan
        the response text with CATEGORY_DETECTORS to find what types of claims
        are present. Then check which of those categories are NOT represented
        in the extracted constraints. The difference = uncovered claims.

        Phase 3 — Aggregate: Count how many false negatives have uncovered
        claims in each category. Categories with the highest counts are the
        biggest gaps in the pipeline.

        Phase 4 — Suggest patterns: For the top 3 categories, pull candidate
        extraction patterns from SUGGESTED_PATTERNS and test them against the
        false negative responses. Report how many false negatives each pattern
        would match (estimated catch count).

    Spec: REQ-VERIFY-001, REQ-VERIFY-002, REQ-VERIFY-003, SCENARIO-VERIFY-005
    """

    def __init__(self, pipeline: VerifyRepairPipeline | None = None) -> None:
        """Initialize with an optional pipeline instance.

        **Detailed explanation for engineers:**
            If no pipeline is provided, creates a default VerifyRepairPipeline
            in verify-only mode (no model). The pipeline is used to run
            constraint extraction and verification on each response.

        Args:
            pipeline: Pre-configured pipeline, or None for default.

        Spec: REQ-VERIFY-001
        """
        self._pipeline = pipeline or VerifyRepairPipeline()

    def analyze(
        self,
        questions: list[str],
        responses: list[str],
        ground_truths: list[str],
        checkers: list[Any] | None = None,
    ) -> FailureReport:
        """Run full failure analysis on a batch of question-response pairs.

        **Detailed explanation for engineers:**
            Main entry point. For each triple (question, response, ground_truth),
            determines if the response is wrong, runs pipeline verification,
            and if the pipeline missed the error (false negative), categorizes
            the uncovered claims.

            The ``checkers`` parameter is an optional list of callables, one per
            question, that take (response_text) -> bool. If provided, these are
            used to determine correctness instead of simple string matching
            against ground_truth. This is important because LLM answers rarely
            match ground truth exactly — a checker can extract the numeric
            answer and compare it.

        Args:
            questions: List of question strings.
            responses: List of LLM response strings (one per question).
            ground_truths: List of correct answer strings.
            checkers: Optional list of callable(response) -> bool for
                determining correctness. If None, uses string containment.

        Returns:
            FailureReport with false negatives, category counts, and
            suggested patterns.

        Spec: REQ-VERIFY-001, REQ-VERIFY-002, REQ-VERIFY-003
        """
        assert len(questions) == len(responses) == len(ground_truths), (
            "questions, responses, and ground_truths must have the same length"
        )

        false_negatives: list[FalseNegative] = []
        total_wrong = 0

        for i, (q, r, gt) in enumerate(zip(questions, responses, ground_truths)):
            # Determine if the response is correct.
            if checkers and i < len(checkers) and checkers[i] is not None:
                is_correct = checkers[i](r)
            else:
                is_correct = self._default_check(r, gt)

            if is_correct:
                # Response is correct — not interesting for failure analysis.
                continue

            total_wrong += 1

            # Run pipeline verification.
            verification = self._pipeline.verify(question=q, response=r)

            if not verification.verified:
                # Pipeline correctly flagged this as wrong — true positive.
                continue

            # FALSE NEGATIVE: wrong answer but pipeline said verified=True.
            extracted = list(verification.constraints)
            uncovered = self._categorize_uncovered(r, extracted)

            false_negatives.append(
                FalseNegative(
                    question=q,
                    response=r,
                    ground_truth=gt,
                    verification=verification,
                    extracted_constraints=extracted,
                    uncovered_categories=uncovered,
                )
            )

        # Aggregate category counts across all false negatives.
        category_counts = self._aggregate_categories(false_negatives)

        # Suggest patterns for top 3 categories.
        suggested = self._suggest_patterns(false_negatives, category_counts)

        fn_rate = len(false_negatives) / total_wrong if total_wrong > 0 else 0.0

        return FailureReport(
            total_questions=len(questions),
            total_wrong=total_wrong,
            false_negatives=false_negatives,
            false_negative_rate=fn_rate,
            category_counts=category_counts,
            suggested_patterns=suggested,
        )

    def _default_check(self, response: str, ground_truth: str) -> bool:
        """Default correctness check: ground truth string appears in response.

        **Detailed explanation for engineers:**
            Simple containment check (case-insensitive). This is a rough
            heuristic — for real benchmarks, callers should provide domain-
            specific checker functions via the ``checkers`` parameter.
        """
        return ground_truth.lower() in response.lower()

    def _categorize_uncovered(
        self, response: str, extracted: list[ConstraintResult]
    ) -> dict[str, int]:
        """Identify claim categories present in response but not extracted.

        **Detailed explanation for engineers:**
            For each category in CATEGORY_DETECTORS, counts how many detection
            patterns match the response text. Then subtracts claims that ARE
            covered by extracted constraints (based on constraint_type mapping).
            The remainder is the uncovered claim count per category.

            The mapping from constraint_type to category is:
            - "arithmetic" -> arithmetic_chain (partially covered)
            - "implication", "exclusion", "disjunction" -> implicit_logic (partially)
            - "factual", "factual_relation" -> world_knowledge (partially)
            - "type_check", "bound", "return_type" etc. -> code_semantics (partially)

            "Partially covered" means the extractor catches SOME claims in that
            category but not all — the uncovered count reflects the gap.

        Spec: REQ-VERIFY-003
        """
        # Count detected claims per category.
        detected: dict[str, int] = {}
        for category, patterns in CATEGORY_DETECTORS.items():
            count = 0
            for pattern in patterns:
                count += len(pattern.findall(response))
            detected[category] = count

        # Count extracted constraints that partially cover each category.
        # This maps constraint types to the categories they partially cover.
        type_to_category: dict[str, str] = {
            "arithmetic": "arithmetic_chain",
            "implication": "implicit_logic",
            "exclusion": "implicit_logic",
            "disjunction": "implicit_logic",
            "negation": "negation",
            "universal": "implicit_logic",
            "factual": "world_knowledge",
            "factual_relation": "world_knowledge",
            "quantity": "world_knowledge",
            "type_check": "code_semantics",
            "return_type": "code_semantics",
            "return_value_type": "code_semantics",
            "bound": "code_semantics",
            "initialization": "code_semantics",
        }

        covered: dict[str, int] = {cat: 0 for cat in CLAIM_CATEGORIES}
        for constraint in extracted:
            cat = type_to_category.get(constraint.constraint_type)
            if cat:
                covered[cat] += 1

        # Uncovered = detected minus what's already covered, floored at 0.
        uncovered: dict[str, int] = {}
        for category in CLAIM_CATEGORIES:
            gap = detected.get(category, 0) - covered.get(category, 0)
            if gap > 0:
                uncovered[category] = gap

        return uncovered

    def _aggregate_categories(
        self, false_negatives: list[FalseNegative]
    ) -> dict[str, int]:
        """Count how many false negatives have uncovered claims per category.

        **Detailed explanation for engineers:**
            This is a FREQUENCY count, not a total claim count. If 15 out of
            30 false negatives had uncovered arithmetic_chain claims, the
            count is 15. This tells us: "if we added an arithmetic_chain
            extractor, we could potentially catch up to 15 more errors."
        """
        counts: dict[str, int] = {cat: 0 for cat in CLAIM_CATEGORIES}
        for fn in false_negatives:
            for category, count in fn.uncovered_categories.items():
                if count > 0:
                    counts[category] += 1
        return counts

    def _suggest_patterns(
        self,
        false_negatives: list[FalseNegative],
        category_counts: dict[str, int],
    ) -> list[dict[str, Any]]:
        """Suggest new extraction patterns for the top 3 uncovered categories.

        **Detailed explanation for engineers:**
            Ranks categories by frequency count (descending), takes the top 3,
            and for each retrieves candidate patterns from SUGGESTED_PATTERNS.
            Each pattern is tested against all false negative responses to
            estimate how many it would catch (match count).

            Returns a flat list of pattern suggestions sorted by estimated
            catch count (descending), so the most impactful pattern is first.

        Spec: REQ-VERIFY-003
        """
        # Rank categories by count, take top 3.
        ranked = sorted(category_counts.items(), key=lambda x: x[1], reverse=True)
        top_categories = [cat for cat, count in ranked[:3] if count > 0]

        suggestions: list[dict[str, Any]] = []
        for category in top_categories:
            patterns = SUGGESTED_PATTERNS.get(category, [])
            for pat_info in patterns:
                # Test this pattern against false negative responses.
                catch_count = 0
                compiled = re.compile(pat_info["pattern"], re.IGNORECASE)
                for fn in false_negatives:
                    if compiled.search(fn.response):
                        catch_count += 1

                suggestions.append({
                    "category": category,
                    "name": pat_info["name"],
                    "pattern": pat_info["pattern"],
                    "description": pat_info["description"],
                    "estimated_catch_count": catch_count,
                    "estimated_catch_rate": (
                        catch_count / len(false_negatives)
                        if false_negatives
                        else 0.0
                    ),
                })

        # Sort by estimated catch count descending.
        suggestions.sort(key=lambda x: x["estimated_catch_count"], reverse=True)
        return suggestions
