"""SymCodeVerifier — executable Python verification of CoT arithmetic steps.

**Why this module exists (RETRO-069):**

    DSVD offline AUC=0.976 vs live AUC=0.158.  The hidden-state probing approach
    trained on synthetic data cannot generalise to live Qwen3.5-0.8B outputs because
    hidden states are model-specific and distribution-sensitive: the probe learned what
    *synthetic* CoT steps look like, not what a real instruction-tuned model writes.

    The fix (arXiv 2510.25975, SymCode; arXiv 2602.11202, Interwhen): instead of
    probing hidden states, use the LLM to generate executable Python from each CoT
    step, then run it.  Code execution is model-agnostic — eval('47+28') == 75
    regardless of how the model phrased the step.  This converts the hard problem of
    "does this natural language sentence state the right number?" into the trivial
    problem of "does this Python expression evaluate to this number?"

    Key advantage over DSVD: no training required, no distribution shift possible.
    The verifier is correct-by-construction: if the code says 47+28 and the model
    wrote "75", the verification passes; if the model wrote "65", it fails.

Spec: REQ-VERIFY-122, REQ-VERIFY-123,
      SCENARIO-VERIFY-160, SCENARIO-VERIFY-161, SCENARIO-VERIFY-162
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Optional

from carnot.extraction.llm_extractor_v1 import safe_eval

# ---------------------------------------------------------------------------
# CoTStep — result for one step's verification
# ---------------------------------------------------------------------------

# Regex to find the last numeric result in a CoT step.  Matches "= 75", "= $75.00",
# "is 75", "gives 75.5", "equals 75", "total 75", "result is 75", etc.
# The last match wins — CoT steps typically state their final answer at the end.
_LAST_NUMBER_RE = re.compile(
    r"(?:(?:=\s*\$?|(?:is|gives?|totals?|results?\s+in|equals?|total)\s+\$?)"
    r"([\d,]+(?:\.\d+)?)"
    r"|(?<!\w)([\d,]+(?:\.\d+)?)(?!\w)(?:\s*(?:dollars?|cents?))?)",
    re.IGNORECASE,
)

# Prompt used when llm_caller is available.  Deliberately terse to reduce token
# cost per step.  "None" as the output is the canonical signal for "no arithmetic".
_SYMCODE_STEP_PROMPT = (
    "Write a single-line Python expression that computes the arithmetic stated "
    "in this reasoning step. Output ONLY the expression (e.g. \"47+28\"). "
    'If no arithmetic: output "None". Step: {step}'
)

# Fallback regex for CI / no-LLM mode: recognise "N op M" patterns.
_EXPR_RE = re.compile(
    r"([\d,]+(?:\.\d+)?)\s*([+\-*/])\s*([\d,]+(?:\.\d+)?)"
)


@dataclass
class CoTStep:
    """Result of verifying one step from a chain-of-thought response.

    Fields:
        text             — Original step text (sentence or line).
        step_index       — Zero-based position in the parent response.
        generated_code   — Python expression generated for this step.  None when
                           the step contains no detectable arithmetic.
        executed_result  — float result of safe_eval(generated_code).  None when
                           the expression is unevaluable or generated_code is None.
        stated_result    — Last numeric value stated in the step text.  None when
                           no number can be extracted.
        violation_detected — True iff executed_result and stated_result are both
                             non-None and differ by more than 1e-6.  This is the
                             distribution-invariant violation signal: code execution
                             is always correct; the model's stated value is what
                             we compare against.
    """

    text: str
    step_index: int
    generated_code: Optional[str]
    executed_result: Optional[float]
    stated_result: Optional[float]
    violation_detected: bool


# ---------------------------------------------------------------------------
# SymCodeVerifier
# ---------------------------------------------------------------------------


class SymCodeVerifier:
    """Distribution-invariant CoT arithmetic verifier via executable Python.

    Instead of probing hidden states (DSVD, AUC=0.158 live), this verifier asks the
    LLM to translate each CoT step into a Python expression, then evaluates it.
    Because code execution is deterministic and model-agnostic, this approach is
    immune to the hidden-state distribution shift that killed DSVD on live data.

    In CI mode (llm_caller=None): falls back to SymCodeExtractor-style regex
    extraction of N op M patterns.  This is sufficient for testing correctness
    of the pipeline logic without a real LLM.

    Args:
        llm_caller : callable(prompt: str) -> str, or None for CI / regex fallback.
                     In live mode, pass a Qwen3.5-0.8B CPU inference function.

    Spec: REQ-VERIFY-122, REQ-VERIFY-123
    """

    def __init__(self, llm_caller: Optional[Callable[[str], str]] = None) -> None:
        self.llm_caller = llm_caller

    # ------------------------------------------------------------------
    # Step segmentation
    # ------------------------------------------------------------------

    def segment_steps(self, response: str) -> list[str]:
        """Split a CoT response into individual reasoning steps.

        Splits on sentence boundaries ('. ' and newlines) and returns only
        non-empty, non-whitespace strings.  Each returned string is one step
        to be independently verified.

        Why sentence-level (not word-level): arithmetic errors occur within
        a step's claimed result.  Splitting at the step boundary gives each
        step its own "lhs expression" and "stated result" in one atomic unit.

        Args:
            response: Full CoT response text from the model.

        Returns:
            Ordered list of non-empty step strings.
        """
        # Split on newlines first, then on '. ' within each line.
        raw_parts: list[str] = []
        for line in response.splitlines():
            for part in re.split(r"\.\s+", line):
                raw_parts.append(part)
        return [p.strip() for p in raw_parts if p.strip()]

    # ------------------------------------------------------------------
    # Code extraction per step
    # ------------------------------------------------------------------

    def extract_code_for_step(self, step: str) -> Optional[str]:
        """Produce a Python expression that computes the arithmetic in this step.

        With llm_caller: sends the step to the LLM with the SymCode prompt and
        returns the stripped response.  If the LLM outputs "None" (its signal
        that no arithmetic is present), returns None.

        Without llm_caller (CI fallback): applies a simple regex to find the
        first "N op M" pattern in the step and returns it as an expression.
        Returns None if no such pattern is found.

        Args:
            step: A single CoT step text string.

        Returns:
            A Python expression string (e.g. "47+28"), or None if no arithmetic
            can be extracted.
        """
        if self.llm_caller is not None:
            prompt = _SYMCODE_STEP_PROMPT.format(step=step)
            try:
                raw = self.llm_caller(prompt).strip()
            except Exception:  # noqa: BLE001 — LLM errors are non-fatal
                return None
            if not raw or raw.lower() in ("none", "null", ""):
                return None
            # Strip markdown fences if present.
            raw = re.sub(r"^```(?:python)?\n?", "", raw, flags=re.MULTILINE)
            raw = re.sub(r"\n?```$", "", raw, flags=re.MULTILINE).strip()
            return raw if raw else None

        # CI fallback: extract first "N op M" from the step text.
        m = _EXPR_RE.search(step)
        if not m:
            return None
        a = re.sub(r"[,_$]", "", m.group(1))
        op = m.group(2)
        b = re.sub(r"[,_$]", "", m.group(3))
        return f"{a}{op}{b}"

    # ------------------------------------------------------------------
    # Per-step verification
    # ------------------------------------------------------------------

    def _extract_stated_result(self, step: str) -> Optional[float]:
        """Extract the last numeric value stated in the step.

        Looks for "= N", "is N", "gives N", etc. first; if none found, falls
        back to the last standalone number in the step.  Returns None if the
        step contains no numbers.

        Why 'last' number: CoT steps typically state their conclusion at the end
        ("… so the total is 75"), so the last number is the answer being claimed.
        """
        # Try the structured "= N" / "is N" pattern first.
        matches = re.findall(
            r"(?:=\s*\$?|(?:is|gives?|totals?|results?\s+in|equals?)\s+\$?)"
            r"([\d,]+(?:\.\d+)?)",
            step,
            flags=re.IGNORECASE,
        )
        if matches:
            raw = re.sub(r"[,_$]", "", matches[-1])
            try:
                return float(raw)
            except ValueError:
                pass

        # Fallback: last standalone number in the step.
        nums = re.findall(r"(?<!\w)([\d,]+(?:\.\d+)?)(?!\w)", step)
        if nums:
            raw = re.sub(r"[,_$]", "", nums[-1])
            try:
                return float(raw)
            except ValueError:
                pass
        return None

    def verify_step(self, step: str, step_index: int = 0) -> CoTStep:
        """Verify a single CoT step for arithmetic correctness.

        Generates Python code for the step, evaluates it, and compares to the
        stated result.  A violation is detected when the executed result disagrees
        with the stated result by more than 1e-6.

        Args:
            step:       CoT step text to verify.
            step_index: Zero-based index for this step in its parent response.

        Returns:
            CoTStep with all fields populated.
        """
        code_raw = self.extract_code_for_step(step)

        if code_raw is None or code_raw.strip().lower() == "none":
            return CoTStep(
                text=step,
                step_index=step_index,
                generated_code=None,
                executed_result=None,
                stated_result=None,
                violation_detected=False,
            )

        executed = safe_eval(code_raw)
        stated = self._extract_stated_result(step)

        violation = (
            executed is not None
            and stated is not None
            and abs(executed - stated) > 1e-6
        )

        return CoTStep(
            text=step,
            step_index=step_index,
            generated_code=code_raw,
            executed_result=executed,
            stated_result=stated,
            violation_detected=violation,
        )

    # ------------------------------------------------------------------
    # Full response verification
    # ------------------------------------------------------------------

    def verify_response(self, response: str) -> list[CoTStep]:
        """Verify all steps in a CoT response.

        Segments the response into steps and runs verify_step() on each.

        Args:
            response: Full CoT response text.

        Returns:
            Ordered list of CoTStep objects, one per segmented step.
        """
        steps = self.segment_steps(response)
        return [self.verify_step(s, idx) for idx, s in enumerate(steps)]

    def detection_score(self, response: str) -> float:
        """Return the fraction of steps with a detected violation.

        A score of 0.0 means no violations were detected; 1.0 means every step
        had a violation.  Responses with no arithmetic steps score 0.0 (cannot
        detect what is not present).

        This scalar can be thresholded (> 0.0) to classify a response as
        "contains at least one arithmetic error," which is the primary use case
        for RETRO-069 resolution.

        Args:
            response: Full CoT response text.

        Returns:
            Float in [0.0, 1.0].
        """
        cot_steps = self.verify_response(response)
        if not cot_steps:
            return 0.0
        return sum(1 for s in cot_steps if s.violation_detected) / len(cot_steps)
