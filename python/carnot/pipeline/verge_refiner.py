"""VergeRefiner: iterative SMT-guided step-level repair of chain-of-thought responses.

**Researcher summary:**
    VERGE (arXiv 2601.20055) demonstrated near-perfect accuracy on multi-step math
    problems by identifying the SPECIFIC reasoning step that contains an arithmetic
    contradiction and repairing only that step — not the whole response.

    This is superior to whole-response Z3-gated repair (Exp 312) because:
    - Step isolation: we surface the exact assertion that Z3 found inconsistent,
      giving the LLM a precise fix target rather than a vague "your reasoning is wrong".
    - Minimal repair surface: only the broken step is rewritten, preserving all
      correct steps in the surrounding context.
    - Closed-loop verification: each repair is immediately re-verified by Z3, so
      we know instantly whether the fix was sufficient.

    Architecture:
    - ``VergeIteration``: dataclass logging one iteration's evidence.
    - ``extract_failed_assertion``: parse z3_code for the first ``s.add(...)``
      body when Z3 returned UNSAT — this is the assertion that triggered the
      contradiction.
    - ``build_step_repair_prompt``: returns a targeted prompt asking the LLM to
      fix only the step that produced the failed assertion.
    - ``VergeRefiner``: orchestrates the iterative loop.
      refine(question, response) → (final_response, list[VergeIteration])

    LLM caller contract:
    - ``llm_caller(prompt: str) -> str`` — any callable that accepts a string
      prompt and returns the LLM's response.  In production: wrap the
      generate() function from carnot.inference.model_loader.  In CI tests:
      pass a MagicMock.

    Timeout:
    - Each iteration inherits the NL2Z3Extractor's per-Z3-call timeout.
    - The caller controls the total iteration budget via max_iterations.

Spec: REQ-REPAIR-012, REQ-REPAIR-013,
      SCENARIO-REPAIR-024, SCENARIO-REPAIR-025, SCENARIO-REPAIR-026, SCENARIO-REPAIR-027
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable

from carnot.pipeline.nl2z3_extractor import Z3Result


# ---------------------------------------------------------------------------
# VergeIteration dataclass
# ---------------------------------------------------------------------------


@dataclass
class VergeIteration:
    """Log record for one VERGE refinement iteration.

    **Detailed explanation for engineers:**
        Every time VergeRefiner attempts to repair a failing step, it records one
        VergeIteration so callers can audit the full repair history and compute
        aggregate metrics (e.g., mean iterations to convergence).

    Attributes:
        iteration_n:      1-based iteration counter.
        assertion_failed: The assertion body string extracted from z3_code that
                          triggered UNSAT, or None if no specific assertion was
                          parseable.
        step_text:        Text passed to the LLM as the "problematic step" context.
                          Usually the full response (we let the LLM identify the
                          specific step given the assertion as a pointer).
        repair_prompt:    The full prompt string sent to the LLM for this iteration.
        repaired_step:    The LLM's response — the corrected step text.
        new_z3_result:    Z3Result from re-verifying the patched response.
        resolved:         True when new_z3_result.sat_status == "sat" (contradiction
                          was eliminated).  False when still UNSAT or unknown.

    Spec: REQ-REPAIR-012
    """

    iteration_n: int
    assertion_failed: str | None
    step_text: str
    repair_prompt: str
    repaired_step: str
    new_z3_result: Z3Result
    resolved: bool


# ---------------------------------------------------------------------------
# extract_failed_assertion
# ---------------------------------------------------------------------------

# Match s.add(<body>) where <body> may contain nested parentheses.
# We use a simple "find matching close paren" approach rather than a pure regex
# so nested calls like s.add(z3.And(x > 0, y < 5)) are handled correctly.
_SADD_START = re.compile(r"s\.add\(")


def extract_failed_assertion(z3_result: Z3Result) -> str | None:
    """Parse z3_code for the first s.add(...) body when Z3 returned UNSAT.

    **Detailed explanation for engineers:**
        Z3's Python API does not natively report which assertion triggered UNSAT
        unless you use assert_and_track() with named labels.  The code generated
        by NL2Z3Extractor uses the simpler s.add() form, so we cannot ask Z3 for
        the culprit assertion directly.

        Instead we use a heuristic that's very effective in practice: take the
        FIRST assertion in the code.  This is correct often enough because:
        1. NL2Z3Extractor generates code that mirrors the reasoning steps in order.
        2. The first incorrect step is usually the earliest arithmetic mistake.
        3. Z3 only returns UNSAT when the whole system is inconsistent — the first
           assertion is almost always part of the minimal UNSAT core.

        Parsing strategy:
        1. Locate "s.add(" in z3_code using a regex.
        2. Walk forward from the opening paren, tracking depth, until the matching
           closing paren is found.
        3. Return the substring between the outer parens.

        Returns None when:
        - sat_status is not "unsat" (no contradiction to surface).
        - z3_code is empty.
        - No "s.add(" call is found in z3_code.

    Args:
        z3_result: Z3Result from NL2Z3Extractor or run_z3_code.

    Returns:
        The assertion body string (content of s.add(...)), or None.

    Spec: REQ-REPAIR-013, SCENARIO-REPAIR-026, SCENARIO-REPAIR-027
    """
    if z3_result.sat_status != "unsat":
        return None
    if not z3_result.z3_code:
        return None

    code = z3_result.z3_code
    match = _SADD_START.search(code)
    if not match:
        return None

    # Walk forward from the character after "s.add(" to find the matching ")".
    start = match.end()  # index of the first char after "s.add("
    depth = 1
    i = start
    while i < len(code) and depth > 0:
        if code[i] == "(":
            depth += 1
        elif code[i] == ")":
            depth -= 1
        i += 1

    if depth != 0:
        # Unbalanced parentheses — malformed code, cannot parse.
        return None

    # i is now one past the closing ")" — the body is code[start : i-1].
    body = code[start : i - 1].strip()
    return body if body else None


# ---------------------------------------------------------------------------
# build_step_repair_prompt
# ---------------------------------------------------------------------------


def build_step_repair_prompt(
    assertion_failed: str | None,
    full_response: str,
) -> str:
    """Build a targeted LLM prompt to correct the step that produced a failed assertion.

    **Detailed explanation for engineers:**
        The VERGE approach is effective because it gives the LLM a precise repair
        target — the specific assertion that Z3 found inconsistent — rather than
        asking "is there anything wrong with this response?".

        When assertion_failed is None (no specific assertion could be extracted),
        we fall back to a general "please correct any arithmetic inconsistency"
        prompt.  This is less targeted but still provides the full response context
        so the LLM can attempt a repair.

        The prompt structure:
        1. State the problem precisely (the assertion that failed).
        2. Include the full response for context (the LLM needs to see all steps
           to understand which step produced the failing assertion).
        3. Ask for a corrected version of ONLY that specific step — not a rewrite
           of the whole response.  This minimises repair artifacts.

    Args:
        assertion_failed: The assertion body from extract_failed_assertion, or None.
        full_response:    The full chain-of-thought response being repaired.

    Returns:
        A string prompt suitable for passing to an LLM.

    Spec: REQ-REPAIR-012
    """
    if assertion_failed is not None:
        return (
            "The following step in your reasoning is arithmetically inconsistent:\n"
            f"    {assertion_failed}\n\n"
            "Please correct only that specific step and rewrite the corrected step.\n\n"
            f"Full reasoning context:\n{full_response}"
        )
    # Fallback: no specific assertion found
    return (
        "Your reasoning contains an arithmetic inconsistency that could not be "
        "precisely located.  Please review the following response and correct any "
        "step that contains an arithmetic error.  Rewrite only the corrected step.\n\n"
        f"Full reasoning context:\n{full_response}"
    )


# ---------------------------------------------------------------------------
# VergeRefiner
# ---------------------------------------------------------------------------

_LLMCallerFn = Callable[[str], str]


class VergeRefiner:
    """Iterative SMT-guided step-level repair of chain-of-thought responses.

    **Detailed explanation for engineers:**
        VergeRefiner orchestrates the three-phase VERGE loop for each question:

        Phase 1 — Initial Z3 check:
            Run NL2Z3Extractor on the response.  If SAT: return immediately (the
            response is self-consistent; no repair needed).  This is the fast path
            and the most common outcome for correct responses.

        Phase 2 — Step isolation (UNSAT):
            Z3 returned UNSAT — there is a provable arithmetic contradiction.
            Call extract_failed_assertion() to identify which s.add(...) body
            triggered it.  Build a targeted repair prompt.

        Phase 3 — LLM repair + re-verify:
            Call llm_caller(prompt) to get a corrected step.  Patch the response
            by appending or replacing with the corrected text.  Re-run Z3 on the
            patched response.  If SAT: done (resolved=True).  If still UNSAT:
            repeat from Phase 2, up to max_iterations times.

        Iteration log:
            Every Phase 2–3 cycle produces one VergeIteration record.  The log
            captures all evidence needed for post-hoc analysis and benchmark
            comparison against the Exp 312 Z3-gated baseline.

    Args:
        nl2z3_extractor: Any object with extract(q, r, domain) and last_z3_result
                         — matches NL2Z3Extractor's public API.
        llm_caller:      Callable(prompt: str) -> str.  Called with the targeted
                         repair prompt; returns the corrected step text.
        max_iterations:  Maximum number of repair attempts.  Default 3.

    Spec: REQ-REPAIR-012, SCENARIO-REPAIR-024, SCENARIO-REPAIR-025
    """

    def __init__(
        self,
        nl2z3_extractor: object,
        llm_caller: _LLMCallerFn,
        max_iterations: int = 3,
    ) -> None:
        self._extractor = nl2z3_extractor
        self._llm_caller = llm_caller
        self.max_iterations = max_iterations

    def refine(
        self,
        question: str,
        response: str,
    ) -> tuple[str, list[VergeIteration]]:
        """Run iterative VERGE refinement on a chain-of-thought response.

        **Detailed explanation for engineers:**
            The response is patched in-place across iterations: each LLM repair
            produces a new "patched_response" which becomes the input to the next
            Z3 check.  This allows the LLM to build on its previous corrections
            rather than starting from scratch each time.

            Patching strategy: we append the repaired step as a "Correction" note
            at the end of the current response.  This is deliberately simple —
            it is sufficient for Z3 to evaluate whether the new arithmetic is
            consistent, and avoids complex NLP text surgery that could introduce
            its own errors.

        Args:
            question: The original question (passed to the extractor for context).
            response: The initial chain-of-thought response to refine.

        Returns:
            (final_response, iteration_log) where:
            - final_response: the response after all applied repairs.
            - iteration_log: list of VergeIteration (empty if initial check was SAT).

        Spec: REQ-REPAIR-012, SCENARIO-REPAIR-024, SCENARIO-REPAIR-025
        """
        # Phase 1: initial Z3 check.
        self._extractor.extract(question, response, "reasoning")
        z3_result: Z3Result | None = getattr(self._extractor, "last_z3_result", None)

        # If initial check is SAT (or no result), take the fast path.
        if z3_result is None or z3_result.sat_status != "unsat":
            return response, []

        # Phase 2–3: iterative repair loop.
        iterations: list[VergeIteration] = []
        current_response = response

        for n in range(1, self.max_iterations + 1):
            # Extract which assertion failed.
            assertion_failed = extract_failed_assertion(z3_result)

            # Build targeted repair prompt.
            repair_prompt = build_step_repair_prompt(assertion_failed, current_response)

            # Call LLM for repair.
            repaired_step = self._llm_caller(repair_prompt)

            # Patch the response: append the correction.
            patched_response = (
                f"{current_response}\n\nCorrection (iteration {n}): {repaired_step}"
            )

            # Re-verify the patched response.
            self._extractor.extract(question, patched_response, "reasoning")
            new_z3_result: Z3Result | None = getattr(self._extractor, "last_z3_result", None)
            if new_z3_result is None:
                new_z3_result = Z3Result(
                    sat_status="unknown", z3_code="", runtime_ms=0.0
                )

            resolved = new_z3_result.sat_status == "sat"

            iterations.append(
                VergeIteration(
                    iteration_n=n,
                    assertion_failed=assertion_failed,
                    step_text=current_response,
                    repair_prompt=repair_prompt,
                    repaired_step=repaired_step,
                    new_z3_result=new_z3_result,
                    resolved=resolved,
                )
            )

            current_response = patched_response

            if resolved:
                break

            # Update z3_result for the next iteration's assertion extraction.
            z3_result = new_z3_result

        return current_response, iterations
