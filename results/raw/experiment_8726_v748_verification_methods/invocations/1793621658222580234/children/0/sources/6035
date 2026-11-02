"""EstimationVerifier — arithmetic plausibility check for single-step word problems.

**Why this exists (the problem it solves):**
    FoVer (Formal Step Verifier) works by labeling each intermediate reasoning step
    in a chain-of-thought. GSM8K problems average 4–6 CoT steps, so FoVer has plenty
    of steps to label and the signal is rich.

    SVAMP problems are SINGLE-STEP arithmetic word problems (mean CoT depth < 2 as
    confirmed by Exp 893). FoVer has nothing to label — it assigns "labeling failed"
    to every SVAMP response, producing 100% label noise and AUC ≈ 0.5 (random).

    EstimationVerifier takes a completely different approach: instead of checking the
    reasoning PROCESS (step-by-step), it checks whether the final ANSWER is within a
    plausible arithmetic range given the numbers in the question. No CoT needed.

**How it works:**
    1. Extract all numbers from the question text (regex-based).
    2. Identify which arithmetic operation the question is asking about (add/subtract/
       multiply/divide) by scanning for keyword signals in the question text.
    3. Compute a plausible answer range based on that operation and the extracted numbers.
       These ranges are intentionally generous (e.g., add → [min(nums), sum(nums)*2])
       to tolerate minor phrasing variation without being so wide they accept nonsense.
    4. Extract the numerical answer from the model's response (trailing number, or
       patterns like "= X", "answer is X", "result is X").
    5. Return a structured dict with the extracted numbers, operation type, plausible
       range, extracted answer, whether it falls in-range, and a confidence score.

**Spec:** REQ-VER-085, SCENARIO-VER-085 (Exp 896)
"""

from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# Keyword dictionaries for operation detection.
# Each set contains English words that signal the corresponding operation.
# ---------------------------------------------------------------------------

_ADD_KEYWORDS: set[str] = {
    "more",
    "total",
    "sum",
    "combined",
    "together",
    "added",
    "plus",
    "increased",
}
_SUB_KEYWORDS: set[str] = {
    "left",
    "remaining",
    "fewer",
    "difference",
    "less",
    "lost",
    "spent",
    "removed",
    "decreased",
}
_MUL_KEYWORDS: set[str] = {
    "times",
    "each",
    "per",
    "product",
    "every",
    "multiplied",
    "rows",
    "columns",
}
_DIV_KEYWORDS: set[str] = {"split", "share", "average", "divided", "portions", "equally", "groups"}

# Regex to extract numbers including decimals.
_NUMBER_RE: re.Pattern[str] = re.compile(r"\d+\.?\d*")

# Regex patterns for extracting the answer from the model response.
# Checked in order; first match wins. The last pattern (last number in text)
# acts as a broad fallback when none of the explicit patterns match.
_ANSWER_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"[Aa]nswer\s+is\s+([+-]?\d+\.?\d*)"),
    re.compile(r"[Rr]esult\s+is\s+([+-]?\d+\.?\d*)"),
    re.compile(r"=\s*([+-]?\d+\.?\d*)\s*$", re.MULTILINE),
    re.compile(r"([+-]?\d+\.?\d*)\s*[^\d]*$"),  # last number before trailing non-digits
]


class EstimationVerifier:
    """Verify single-step arithmetic word-problem responses by plausibility range.

    Unlike FoVer (which labels step-by-step CoT), EstimationVerifier checks whether
    the extracted numerical answer falls within a plausible range computed from the
    numbers present in the question and the detected operation type.

    This approach is correct for SVAMP-style problems where there is no meaningful
    intermediate reasoning chain to label.

    Usage::

        ev = EstimationVerifier()
        result = ev.verify("Tom has 5 apples and buys 3 more. How many total?", "Tom has 8 apples total.")
        label = ev.label_pair("Tom has 5 apples...", "Tom has 8 apples total.", ground_truth=8.0)
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def verify(self, question: str, response: str) -> dict:
        """Check whether the response answer is arithmetically plausible for the question.

        Args:
            question: The word-problem question text.
            response: The model's textual response.

        Returns:
            A dict with keys:
                ``extracted_numbers``  — list of floats found in the question
                ``operation_type``     — "add", "subtract", "multiply", "divide", or "unknown"
                ``plausible_range``    — [low, high] inclusive plausibility window
                ``extracted_answer``   — float answer from response, or None if not found
                ``in_range``           — True if extracted_answer is within plausible_range
                ``confidence``         — 1.0 when answer extracted, 0.5 when answer is None
        """
        numbers = self._extract_numbers(question)
        op = self._detect_operation(question)
        plausible_range = self._compute_range(numbers, op)
        extracted_answer = self._extract_answer(response)

        if extracted_answer is not None and plausible_range is not None:
            in_range = plausible_range[0] <= extracted_answer <= plausible_range[1]
            confidence = 1.0
        elif extracted_answer is None:
            in_range = False
            confidence = 0.5
        else:
            # No numbers in question — cannot compute range, give benefit of doubt.
            in_range = False
            confidence = 0.5

        return {
            "extracted_numbers": numbers,
            "operation_type": op,
            "plausible_range": plausible_range if plausible_range is not None else [0.0, 0.0],
            "extracted_answer": extracted_answer,
            "in_range": in_range,
            "confidence": confidence,
        }

    def label_pair(
        self,
        question: str,
        response: str,
        ground_truth: float | None = None,
    ) -> int:
        """Return 1 (correct) or 0 (wrong) for a question/response pair.

        When ``ground_truth`` is provided, uses exact comparison (within 0.01
        tolerance) against the ground truth value — this is the gold-standard
        label used for building training corpora.

        When ``ground_truth`` is None, falls back to the plausibility range
        check from ``verify()``. This is a weaker signal but requires no
        ground-truth annotation, enabling self-supervised labeling on new problems.

        Args:
            question: The word-problem question text.
            response: The model's textual response.
            ground_truth: Optional known correct answer (float).

        Returns:
            1 if the response is judged correct, 0 if wrong.
        """
        result = self.verify(question, response)
        extracted = result["extracted_answer"]

        if ground_truth is not None:
            if extracted is None:
                return 0
            return 1 if abs(extracted - ground_truth) < 0.01 else 0

        return 1 if result["in_range"] else 0

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _extract_numbers(self, text: str) -> list[float]:
        """Return all numbers found in text as a sorted list of floats."""
        matches = _NUMBER_RE.findall(text)
        return [float(m) for m in matches]

    def _detect_operation(self, question: str) -> str:
        """Identify the arithmetic operation from keyword signals in the question.

        Tokenises the question into lowercase words, then counts how many
        keywords from each operation set are present. The operation with the
        most hits wins. Ties favour the order: add > subtract > multiply > divide.
        Returns "unknown" when no keywords match.
        """
        words = set(re.findall(r"[a-z]+", question.lower()))
        scores = {
            "add": len(words & _ADD_KEYWORDS),
            "subtract": len(words & _SUB_KEYWORDS),
            "multiply": len(words & _MUL_KEYWORDS),
            "divide": len(words & _DIV_KEYWORDS),
        }
        best_score = max(scores.values())
        if best_score == 0:
            return "unknown"
        # Prefer in priority order when tied.
        for op in ("add", "subtract", "multiply", "divide"):
            if scores[op] == best_score:
                return op
        return "unknown"

    def _compute_range(self, numbers: list[float], operation: str) -> list[float] | None:
        """Compute a generous plausible answer range given the numbers and operation.

        Ranges are deliberately wide — the goal is to reject clearly wrong answers
        (off by orders of magnitude) while accepting valid responses even when phrased
        in unexpected ways.

        Returns None if there are no numbers in the question (cannot compute range).
        """
        if not numbers:
            return None

        lo: float
        hi: float

        if operation == "add":
            lo = min(numbers)
            hi = sum(numbers) * 2.0
        elif operation == "subtract":
            lo = 0.0
            hi = max(numbers)
        elif operation == "multiply":
            lo = min(numbers)
            hi = max(numbers) ** 2
        elif operation == "divide":
            lo = 0.0
            hi = max(numbers)
        else:
            # Unknown operation — use a broad range spanning all numbers.
            lo = 0.0
            hi = sum(numbers) * 2.0

        return [lo, hi]

    def _extract_answer(self, response: str) -> float | None:
        """Extract a numerical answer from the model's response text.

        Tries each answer-extraction pattern in priority order. Returns the
        first match as a float, or None if no number is found.
        """
        for pattern in _ANSWER_PATTERNS:
            m = pattern.search(response)
            if m:
                try:
                    return float(m.group(1))
                except (ValueError, IndexError):
                    continue
        return None
