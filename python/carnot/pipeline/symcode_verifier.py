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

**RETRO-SYMCODE-SERIAL fix (batch_verify):**

    verify_response() processes each paragraph as a separate verify_step() call.
    For 10+ paragraph responses (Exp 627 style) this creates N separate exec()
    invocations, each paying the module-import and namespace-init cost (~50 ms each).
    batch_verify() collects all expressions in one pass and evaluates them in a single
    shared exec() namespace, reducing total latency from ~N×50ms to ~1×50ms.

Spec: REQ-VERIFY-122, REQ-VERIFY-123, REQ-VERIFY-148,
      SCENARIO-VERIFY-160, SCENARIO-VERIFY-161, SCENARIO-VERIFY-162,
      SCENARIO-VERIFY-173
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
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
# SymCodeBatchResult — aggregate result for batch_verify()
# ---------------------------------------------------------------------------


@dataclass
class SymCodeBatchResult:
    """Aggregate result from a single-exec() batch verification of N paragraphs.

    **Why this exists (RETRO-SYMCODE-SERIAL):**
        Processing 10+ paragraphs one at a time with verify_step() pays ~50ms of
        module-import and namespace-init overhead per paragraph.  batch_verify()
        collects all arithmetic expressions in one regex pass, then evaluates them
        in a single shared exec() namespace, bringing total overhead back to ~1×50ms.

    Fields:
        per_paragraph_results — one CoTStep per input paragraph (same order as input).
        total_violations      — count of paragraphs where violation_detected=True.
        batch_latency_ms      — wall-clock time for the entire batch_verify() call.
        n_paragraphs          — number of paragraphs that were processed.

    Spec: REQ-VERIFY-148, SCENARIO-VERIFY-173
    """

    per_paragraph_results: list[CoTStep]
    total_violations: int
    batch_latency_ms: float
    n_paragraphs: int = field(init=False)

    def __post_init__(self) -> None:
        # n_paragraphs is always derived from per_paragraph_results, never set by caller.
        self.n_paragraphs = len(self.per_paragraph_results)


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

    # ------------------------------------------------------------------
    # Batch verification (RETRO-SYMCODE-SERIAL)
    # ------------------------------------------------------------------

    def batch_verify(self, paragraphs: list[str]) -> SymCodeBatchResult:
        """Verify all paragraphs in a single exec() call.  Faster than N serial verify() calls.

        **Why batching helps:**
            verify_step() calls safe_eval() once per paragraph.  safe_eval() (and any
            exec()-backed evaluator) pays a namespace-creation cost every invocation.
            For 10 paragraphs this is ~10× that cost.  batch_verify() builds one shared
            namespace dict, stuffs all expressions into it as '_expr_N = <expr>', then
            calls exec() once.  The per-paragraph overhead is then O(1) not O(N).

        **Algorithm:**
            1. For each paragraph, extract the code expression (same logic as verify_step).
            2. Collect all non-None expressions into a single exec script:
                   _expr_0 = 47+28
                   _expr_1 = 3*4
                   ...
            3. exec() the script once in a shared namespace.
            4. Read back _expr_N values from the namespace.
            5. For each paragraph, compare the namespace result to the stated value
               (extracted by _extract_stated_result) and set violation_detected.

        **Spec:** REQ-VERIFY-148, SCENARIO-VERIFY-173

        Args:
            paragraphs: List of paragraph strings to verify in one batch.

        Returns:
            SymCodeBatchResult with per-paragraph CoTStep results, violation count,
            and wall-clock latency for the full batch call.
        """
        t_start = time.perf_counter()

        # Step 1: extract code expressions for all paragraphs.
        # We use the same extract_code_for_step logic; None means no arithmetic.
        codes: list[Optional[str]] = [
            self.extract_code_for_step(p) for p in paragraphs
        ]

        # Step 2: build a single exec script with all non-None expressions.
        # Variable names are _expr_0, _expr_1, … so they never clash.
        script_lines: list[str] = []
        for idx, code in enumerate(codes):
            if code is not None and code.strip().lower() != "none":
                script_lines.append(f"_expr_{idx} = {code}")

        # Step 3: exec once into a shared namespace if there is anything to run.
        namespace: dict = {}
        if script_lines:
            script = "\n".join(script_lines)
            try:
                exec(script, {"__builtins__": {}}, namespace)  # noqa: S102
            except Exception:  # noqa: BLE001 — eval errors are non-fatal; results stay None
                pass

        # Step 4 & 5: build per-paragraph CoTStep results using namespace values.
        results: list[CoTStep] = []
        for idx, (paragraph, code) in enumerate(zip(paragraphs, codes)):
            if code is None or code.strip().lower() == "none":
                results.append(
                    CoTStep(
                        text=paragraph,
                        step_index=idx,
                        generated_code=None,
                        executed_result=None,
                        stated_result=None,
                        violation_detected=False,
                    )
                )
                continue

            # Retrieve the evaluated result from the shared namespace.
            raw_result = namespace.get(f"_expr_{idx}")
            try:
                executed: Optional[float] = float(raw_result) if raw_result is not None else None
            except (TypeError, ValueError):
                executed = None

            stated = self._extract_stated_result(paragraph)
            violation = (
                executed is not None
                and stated is not None
                and abs(executed - stated) > 1e-6
            )

            results.append(
                CoTStep(
                    text=paragraph,
                    step_index=idx,
                    generated_code=code,
                    executed_result=executed,
                    stated_result=stated,
                    violation_detected=violation,
                )
            )

        t_end = time.perf_counter()
        batch_latency_ms = (t_end - t_start) * 1000.0
        total_violations = sum(1 for r in results if r.violation_detected)

        return SymCodeBatchResult(
            per_paragraph_results=results,
            total_violations=total_violations,
            batch_latency_ms=batch_latency_ms,
        )

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
