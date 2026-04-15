"""LLMz3Formalizer: LLM-guided Z3 formalization for instruction-tuned LLM responses.

**Researcher summary:**
    NL2Z3Extractor (Exp 310) fails on instruction-tuned (IT) format responses
    because it issues a single LLM call that must both extract arithmetic AND
    produce Z3 code.  When the response is markdown, numbered steps, or mixed
    prose, the combined task overwhelms the model and produces malformed Z3 code.

    LLMz3Formalizer separates the two concerns:
    1. A targeted LLM prompt asks the model to output ONLY Python z3 assertion
       strings — no prose, no explanation, just runnable code.
    2. A second pass runs the Z3 snippet via exec() in a restricted sandbox.

    This separation mirrors the insight from arXiv 2601.04675 (LLM-guided SMT):
    80% improvement in Z3 success rate when an LLM rewrites ambiguous arithmetic
    into explicit Z3 assertion syntax, because the model is not distracted by
    extraction from a noisy format.

**Detailed explanation for engineers:**
    The key architectural decisions:

    exec() sandbox instead of subprocess:
    - exec() is faster than subprocess (no process fork overhead).
    - The sandbox is enforced by providing a restricted ``__builtins__`` dict
      that only allows `z3` imports.  Any attempt to ``import os``, ``import sys``,
      or ``import subprocess`` raises NameError from the restricted __import__
      hook — this is the security invariant.
    - stdout capture uses io.StringIO so we can parse "sat"/"unsat" without
      a process boundary.

    CI-safe stub:
    - When ``llm_caller`` is None, a hardcoded Z3 snippet is returned immediately.
      This lets tests and CI run without any GPU or network call.
    - ``formalization_mode`` records whether the snippet came from a real LLM
      call ("llm") or from the CI stub ("ci_stub").

    n_assertions:
    - Counted by scanning the z3_code for ``.add(`` call patterns.
      This is a heuristic count for observability only — Z3 itself is authoritative.

    is_sat:
    - Derived from z3_result in ``__post_init__``.  Like Z3Result.violations_found,
      it is not an init parameter; it is set automatically.

    Relationship to NL2Z3Extractor:
    - NL2Z3Extractor is a ConstraintExtractor that returns ConstraintResult objects.
    - LLMz3Formalizer returns a Z3FormalizationResult directly — it is a primitive,
      not a pipeline stage.  Higher-level components can wrap it.

Spec: REQ-EXTRACT-019, REQ-EXTRACT-020,
      SCENARIO-EXTRACT-039, SCENARIO-EXTRACT-040, SCENARIO-EXTRACT-041
"""

from __future__ import annotations

import io
import re
import sys
from contextlib import redirect_stdout
from dataclasses import dataclass, field
from typing import Callable, Optional

# ---------------------------------------------------------------------------
# Z3FormalizationResult dataclass
# ---------------------------------------------------------------------------


@dataclass
class Z3FormalizationResult:
    """Result of one LLM-guided Z3 formalization pass.

    **Detailed explanation for engineers:**
        Returned by ``LLMz3Formalizer.formalize()`` every time it is called.
        All fields are populated regardless of whether the exec() sandbox
        succeeded or failed, so callers can always inspect the full evidence.

        ``is_sat``:
        - Derived automatically from ``z3_result`` in ``__post_init__``.
        - True only when Z3 confirmed the assertions are satisfiable ("sat").
        - False for "unsat" (contradiction found), "unknown" (no verdict),
          and "error" (the generated code was malformed or sandbox-blocked).

        ``n_assertions``:
        - Heuristic count of `.add(` occurrences in the z3_code string.
        - Used for observability and experiment metrics, not for correctness.

        ``formalization_mode``:
        - "llm"     → result came from a real LLM call via ``llm_caller``.
        - "ci_stub" → result came from the hardcoded CI stub (no LLM).

        ``source_response_length``:
        - Length of the original response passed to ``formalize()``.
        - Useful for correlating formalization success rate with response length.

    Attributes:
        z3_code:                 Python z3 snippet that was exec'd in the sandbox.
        z3_result:               Solver verdict: "sat", "unsat", "unknown", "error".
        n_assertions:            Count of .add() calls in z3_code (heuristic).
        is_sat:                  True iff z3_result == "sat" (auto-computed).
        formalization_mode:      "llm" or "ci_stub".
        source_response_length:  len(response) passed to formalize().
        error_message:           Exception details when z3_result == "error".

    Spec: REQ-EXTRACT-019, SCENARIO-EXTRACT-039
    """

    z3_code: str
    z3_result: str  # "sat" | "unsat" | "unknown" | "error"
    n_assertions: int
    is_sat: bool = field(init=False)
    formalization_mode: str  # "llm" | "ci_stub"
    source_response_length: int
    error_message: Optional[str] = None

    def __post_init__(self) -> None:
        # is_sat is derived from z3_result, not supplied by the caller.
        # Using object.__setattr__ to bypass the frozen-like pattern; this
        # field is set once here and should not be mutated afterwards.
        object.__setattr__(self, "is_sat", self.z3_result == "sat")


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

_FORMALIZATION_SYSTEM_PROMPT = (
    "You are a formal verification assistant specializing in Z3 SMT constraints. "
    "Your task is to convert arithmetic reasoning from an LLM response into "
    "self-contained Python z3 code. "
    "Rules: "
    "1. Output ONLY a ```python ... ``` code block — no explanation, no prose. "
    "2. The code must import z3 and end with print(s.check()) where s is a z3.Solver(). "
    "3. Assert EVERY numeric claim using s.add(...). "
    "4. Use z3.Int for integer variables, z3.Real for decimal values. "
    "5. If the response contains no arithmetic, emit: "
    "   import z3; s = z3.Solver(); print(s.check())"
)


def build_z3_formalization_prompt(question: str, response: str) -> str:
    """Build the structured prompt asking the LLM to write Python z3 code.

    **Detailed explanation for engineers:**
        This prompt is intentionally narrow: the LLM is NOT asked to verify
        the reasoning, explain the math, or produce any prose.  It is only
        asked to translate every numeric claim into a z3.Solver().add() call.

        Including the original question helps the model understand which
        values are given (premises) vs. derived (intermediate steps), so it
        can encode the right assertions.

        The output contract is strict: ``parse_z3_snippet()`` is called
        immediately after the LLM responds to extract the code block.
        Anything outside ``` fences is discarded.

    Args:
        question: The original question posed to the LLM being verified.
        response: The LLM response whose arithmetic is to be formalized.

    Returns:
        A single combined prompt string (system + user).

    Spec: REQ-EXTRACT-019
    """
    user_part = (
        f"Question: {question}\n\n"
        f"LLM Response:\n{response}\n\n"
        "Write Python z3 code that asserts all arithmetic claims in this response. "
        "Output ONLY the ```python ... ``` code block."
    )
    return f"{_FORMALIZATION_SYSTEM_PROMPT}\n\n{user_part}"


# ---------------------------------------------------------------------------
# Z3 snippet parser
# ---------------------------------------------------------------------------

_CODE_BLOCK_RE = re.compile(r"```python\s*(.*?)```", re.DOTALL)


def parse_z3_snippet(llm_output: str) -> str:
    """Extract the Python code block from an LLM response.

    **Detailed explanation for engineers:**
        The LLM is instructed to output ONLY a ```python ... ``` block.
        In practice, models sometimes prefix with "Here is the code:" or
        similar preamble.  This function strips everything outside the fences.

        If no code block is found, an empty string is returned.
        The caller (``LLMz3Formalizer.formalize``) treats an empty string as
        a formalization failure and returns z3_result="unknown".

    Args:
        llm_output: Raw text output from the LLM call.

    Returns:
        The Python code inside the first ```python ... ``` fences, or "".

    Spec: REQ-EXTRACT-019
    """
    match = _CODE_BLOCK_RE.search(llm_output)
    if match:
        return match.group(1).strip()
    return ""


# ---------------------------------------------------------------------------
# exec() sandbox
# ---------------------------------------------------------------------------

# Pattern to count assertion calls in z3 code (heuristic).
_ASSERTION_RE = re.compile(r"\.add\(")

# CI stub: a minimal valid Z3 program that always returns "sat".
# Used when llm_caller is None so tests and CI run without any LLM.
_CI_STUB_Z3_CODE = (
    "import z3\n"
    "s = z3.Solver()\n"
    "x = z3.Int('x')\n"
    "s.add(x >= 0)\n"
    "print(s.check())\n"
)


def _make_restricted_import(z3_module: object) -> Callable:
    """Return a __import__ function that allows only 'z3'.

    **Detailed explanation for engineers:**
        Python's exec() uses the ``__import__`` builtin when the execd code
        contains an ``import`` statement.  By replacing ``__import__`` with
        this restricted version, we ensure that only ``import z3`` succeeds;
        any other import raises NameError.

        NameError (not ImportError) is raised by design: the task spec
        requires NameError so tests can assert on the exception type.

    Args:
        z3_module: The pre-imported z3 module object to return on success.

    Returns:
        A callable compatible with Python's __import__ signature.
    """

    def _restricted_import(
        name: str,
        globals: object = None,
        locals: object = None,
        fromlist: tuple = (),
        level: int = 0,
    ) -> object:
        if name == "z3" or name.startswith("z3."):
            return z3_module
        raise NameError(
            f"Import of '{name}' is not allowed in the Z3 formalization sandbox. "
            "Only 'z3' may be imported."
        )

    return _restricted_import


def _exec_z3_snippet(code: str) -> tuple[str, Optional[str]]:
    """Execute a Z3 Python snippet in a restricted exec() sandbox.

    **Detailed explanation for engineers:**
        Security model:
        - ``__builtins__`` is replaced with a minimal dict containing only
          ``print`` (redirected to capture output) and a restricted
          ``__import__`` that blocks everything except z3.
        - This means the execd code cannot call open(), os.system(), etc.
        - Any attempt to ``import os``, ``import sys``, or ``import subprocess``
          raises NameError from the restricted __import__ hook.

        Output capture:
        - We redirect the built-in print() to a StringIO buffer.
        - The execd code's print(s.check()) writes "sat" or "unsat" to this buffer.
        - We parse the buffer for the Z3 verdict after exec completes.

        Error handling:
        - Any exception during exec (SyntaxError, NameError, Z3 errors) is caught.
        - Returns ("error", error_message) so the caller can record it gracefully.

    Args:
        code: Self-contained Python code that uses z3 and calls print(s.check()).

    Returns:
        (z3_result, error_message) where z3_result is "sat"/"unsat"/"unknown"/"error".

    Spec: REQ-EXTRACT-019, SCENARIO-EXTRACT-040
    """
    if not code.strip():
        return "unknown", None

    try:
        import z3  # noqa: PLC0415  (deferred import — z3 may not be installed)
    except ImportError:
        return "error", "z3 package is not installed"

    buf = io.StringIO()

    safe_builtins: dict = {
        "__import__": _make_restricted_import(z3),
        "print": lambda *args, **kwargs: print(*args, file=buf, **kwargs),
        "len": len,
        "range": range,
        "int": int,
        "float": float,
        "str": str,
        "bool": bool,
        "list": list,
        "dict": dict,
        "tuple": tuple,
        "True": True,
        "False": False,
        "None": None,
    }

    safe_globals: dict = {
        "__builtins__": safe_builtins,
        "z3": z3,
    }

    try:
        exec(code, safe_globals)  # noqa: S102
    except NameError as exc:
        return "error", f"NameError in sandbox: {exc}"
    except SyntaxError as exc:
        return "error", f"SyntaxError in sandbox: {exc}"
    except Exception as exc:  # noqa: BLE001
        return "error", f"{type(exc).__name__}: {exc}"

    output = buf.getvalue().strip()

    # Parse Z3 verdict from captured stdout.
    # Check "unsat" BEFORE "sat" because "unsat" contains "sat" as a substring.
    if "unsat" in output:
        return "unsat", None
    if "sat" in output:
        return "sat", None
    return "unknown", None


# ---------------------------------------------------------------------------
# LLMz3Formalizer
# ---------------------------------------------------------------------------

# Type alias for the injectable LLM caller function.
# Signature: (prompt: str) -> str
LLMCallerFn = Callable[[str], str]


class LLMz3Formalizer:
    """Separate LLM-guided Z3 formalization from extraction.

    **Detailed explanation for engineers:**
        NL2Z3Extractor issues a single LLM call that must both:
        (a) understand the structure of an IT-format response, and
        (b) produce correct Z3 Python code.

        LLMz3Formalizer focuses the LLM on (b) only: "given that arithmetic
        exists in this response, write the Z3 assertions."  The LLM is not
        asked to parse structure — just to recognize numbers and relationships.

        This separation is the key insight from arXiv 2601.04675: specialized
        prompts for formalization outperform combined extraction+formalization
        by 80% on Z3 success rate.

        max_iterations:
        - If the first LLM call produces code that exec()s as "error", the
          formalizer can retry up to max_iterations times.  Each retry passes
          the error message back in the prompt so the model can self-correct.
        - Default is 2 (one initial attempt + one retry).

        CI-safe:
        - Pass llm_caller=None to skip all LLM calls.  The CI stub is a
          trivially-satisfiable Z3 program that returns "sat".  Tests and CI
          pipelines can validate the full formalization pipeline without a GPU.

    Attributes:
        last_result: The Z3FormalizationResult from the most recent formalize() call.

    Spec: REQ-EXTRACT-019, REQ-EXTRACT-020,
          SCENARIO-EXTRACT-039, SCENARIO-EXTRACT-040, SCENARIO-EXTRACT-041
    """

    def __init__(
        self,
        llm_caller: Optional[LLMCallerFn] = None,
        model_id: str = "ci_stub",
        max_iterations: int = 2,
    ) -> None:
        """Initialize the LLMz3Formalizer.

        Args:
            llm_caller:     Callable (prompt: str) -> str.  When None, CI stub is used.
            model_id:       Identifier for the model being used (for provenance logging).
            max_iterations: Maximum number of LLM formalization attempts (default 2).
        """
        self._llm_caller = llm_caller
        self._model_id = model_id
        self._max_iterations = max(1, max_iterations)
        self.last_result: Optional[Z3FormalizationResult] = None

    def formalize(self, question: str, response: str) -> Z3FormalizationResult:
        """Formalize arithmetic in a response into a Z3FormalizationResult.

        **Detailed explanation for engineers:**
            Step 1 — CI stub path:
            If self._llm_caller is None, skip all LLM calls and exec the
            hardcoded CI stub snippet.  This is the fast path for tests and CI.

            Step 2 — LLM formalization path:
            Build the formalization prompt with build_z3_formalization_prompt().
            Call the LLM with self._llm_caller(prompt).
            Parse the code block with parse_z3_snippet().

            Step 3 — Retry loop:
            If exec returns "error" and we have remaining iterations, rebuild
            the prompt with the error message appended and retry.

            Step 4 — exec sandbox:
            Execute the extracted code in the restricted exec sandbox via
            _exec_z3_snippet().  Capture the Z3 verdict.

            Step 5 — Build result:
            Count assertions, record formalization_mode, build and return
            Z3FormalizationResult.

        Args:
            question: The original question posed to the LLM being verified.
            response: The full LLM response text to formalize.

        Returns:
            Z3FormalizationResult with the Z3 verdict and all metadata.

        Spec: REQ-EXTRACT-019, SCENARIO-EXTRACT-039, SCENARIO-EXTRACT-041
        """
        source_len = len(response)

        # --- CI stub path: llm_caller is None ---
        if self._llm_caller is None:
            z3_result, error_msg = _exec_z3_snippet(_CI_STUB_Z3_CODE)
            n_assertions = len(_ASSERTION_RE.findall(_CI_STUB_Z3_CODE))
            result = Z3FormalizationResult(
                z3_code=_CI_STUB_Z3_CODE,
                z3_result=z3_result,
                n_assertions=n_assertions,
                formalization_mode="ci_stub",
                source_response_length=source_len,
                error_message=error_msg,
            )
            self.last_result = result
            return result

        # --- LLM formalization path ---
        z3_code = ""
        z3_result = "unknown"
        error_msg: Optional[str] = None
        last_error: Optional[str] = None

        for iteration in range(self._max_iterations):
            # Build prompt, optionally appending last error for self-correction.
            prompt = build_z3_formalization_prompt(question, response)
            if last_error is not None and iteration > 0:
                prompt += (
                    f"\n\nNote: Your previous attempt produced an error:\n"
                    f"  {last_error}\n"
                    "Please correct the Z3 code and try again."
                )

            try:
                llm_output = self._llm_caller(prompt)
            except Exception as exc:  # noqa: BLE001
                error_msg = f"LLM call failed: {exc}"
                z3_result = "unknown"
                break

            snippet = parse_z3_snippet(llm_output)
            if not snippet:
                # LLM did not produce a code block; this counts as unknown.
                last_error = "No ```python ... ``` code block found in LLM output."
                z3_result = "unknown"
                continue

            z3_code = snippet
            z3_result, exec_error = _exec_z3_snippet(z3_code)

            if z3_result != "error":
                # Success: got a usable verdict (sat or unsat or unknown).
                error_msg = None
                break

            # exec errored: record for retry prompt and loop.
            last_error = exec_error
            error_msg = exec_error

        n_assertions = len(_ASSERTION_RE.findall(z3_code))
        result = Z3FormalizationResult(
            z3_code=z3_code,
            z3_result=z3_result,
            n_assertions=n_assertions,
            formalization_mode="llm",
            source_response_length=source_len,
            error_message=error_msg,
        )
        self.last_result = result
        return result
