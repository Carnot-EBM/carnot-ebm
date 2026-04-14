"""NL2Z3Extractor: translate chain-of-thought reasoning to Z3 assertions.

**Researcher summary:**
    Addresses the constraint-extraction bottleneck (Exp 203/207): regex-based
    extractors found 0 violations on Gemma4-E4B-it because they cannot capture
    the logical structure of chain-of-thought responses.

    NL2Z3Extractor makes a second LLM call asking it to translate the
    reasoning steps into self-contained Z3 Python code, then runs that code
    in a sandboxed subprocess.  If Z3 reports "unsat", the reasoning chain is
    internally inconsistent and a constraint violation is raised.

    Inspired by:
    - Emergent Formal Verification (arXiv 2603.21149): LLM → Z3 assertions
    - Z3 Security Verification (arXiv 2604.05292): 3,500 formal artifact corpus

**Detailed explanation for engineers:**
    The key insight: Z3 cannot be fooled by surface-level fluency.  If the LLM
    generates code that encodes an arithmetic contradiction, Z3.check() returns
    "unsat" and we surface the violation regardless of how confident the prose
    sounds.

    Architecture:
    - ``Z3Result``: dataclass capturing sat_status, z3_code, runtime_ms,
      violations_found, and error_message from one subprocess run.
    - ``build_z3_prompt(response)``: returns (system, user) messages for the
      LLM call that generates Z3 Python code.
    - ``run_z3_code(code, timeout_s)``: executes the code in a fresh subprocess
      with a hard timeout.  Parses stdout for "unsat"/"sat".
    - ``NL2Z3Extractor``: implements ConstraintExtractor protocol.  In CI mode
      (CARNOT_FORCE_LIVE not set), skips the LLM call and returns "unknown".
      Accepts an injectable ``generate_fn`` for testing.

    LLM guard:
    - Production: set CARNOT_FORCE_LIVE=1 to enable real LLM calls.
    - CI default: no LLM call, Z3Result(sat_status="unknown", z3_code="").

    Subprocess sandbox:
    - ``subprocess.run`` with ``timeout=timeout_s``, ``capture_output=True``.
    - SyntaxError and NameError from the generated code surface as
      sat_status="error" (never crashes the caller).
    - Timeout kills the process and returns sat_status="unknown".

Spec: REQ-EXTRACT-010, REQ-EXTRACT-011,
      SCENARIO-EXTRACT-020, SCENARIO-EXTRACT-021, SCENARIO-EXTRACT-022,
      SCENARIO-EXTRACT-023, SCENARIO-EXTRACT-024
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from carnot.pipeline.extract import ConstraintResult

# ---------------------------------------------------------------------------
# Z3Result dataclass
# ---------------------------------------------------------------------------

_VALID_SAT_STATUSES = frozenset({"sat", "unsat", "unknown", "error"})


@dataclass
class Z3Result:
    """Result of one NL2Z3 check: SMT solver verdict + metadata.

    **Detailed explanation for engineers:**
        Returned by both ``run_z3_code`` and ``NL2Z3Extractor.extract`` so
        callers can inspect the raw Z3 evidence even when no ConstraintResult
        is raised.

        ``violations_found`` is a computed property: True only when Z3 returned
        "unsat" (internally inconsistent reasoning).  "sat", "unknown", and
        "error" do NOT generate violations because:
        - "sat"    → reasoning is consistent; no problem.
        - "unknown"→ LLM unavailable or code ran with no output; we cannot
                     conclude either way, so we conservatively skip.
        - "error"  → the generated code itself was malformed; we blame the
                     LLM output quality rather than the reasoning being wrong.

    Attributes:
        sat_status:       Z3 verdict — "sat", "unsat", "unknown", or "error".
        z3_code:          The Z3 Python code that was generated and executed.
        runtime_ms:       Wall-clock time the subprocess ran, in milliseconds.
        violations_found: True iff sat_status == "unsat".
        error_message:    If sat_status is "error", the exception message.

    Spec: REQ-EXTRACT-011
    """

    sat_status: str  # "sat" | "unsat" | "unknown" | "error"
    z3_code: str
    runtime_ms: float
    violations_found: bool = field(init=False)
    error_message: str | None = None

    def __post_init__(self) -> None:
        # violations_found is derived from sat_status; not an init parameter.
        # We set it here so callers can rely on it without calling a method.
        object.__setattr__(self, "violations_found", self.sat_status == "unsat")


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = (
    "You are a formal verification assistant. "
    "Translate arithmetic reasoning steps from the response below into Z3 Python code "
    "that checks consistency. "
    "Use z3.solve() or z3.Solver().check() at the end. "
    "The code must be self-contained and runnable with `import z3`. "
    "Output ONLY a Python code block delimited by ```python ... ```. "
    "Do not include any explanation outside the code block."
)


def build_z3_prompt(response: str) -> tuple[str, str]:
    """Build the (system, user) messages for the NL2Z3 LLM call.

    **Detailed explanation for engineers:**
        The system prompt instructs the model to output ONLY a runnable Python
        code block containing Z3 constraints derived from the arithmetic in the
        response.  The user message is the raw response text being checked.

        The two-message structure is compatible with both the transformers
        chat-template API and with simple string concatenation for models that
        only accept a single prompt string.

    Args:
        response: The chain-of-thought response whose arithmetic is being checked.

    Returns:
        (system_prompt, user_message) — both strings.

    Spec: REQ-EXTRACT-010
    """
    user_message = (
        "Translate the arithmetic in this reasoning response to Z3 Python code "
        "that checks whether the arithmetic steps are internally consistent. "
        "The code must end with print(s.check()) where s is a z3.Solver().\n\n"
        f"Response:\n{response}"
    )
    return _SYSTEM_PROMPT, user_message


# ---------------------------------------------------------------------------
# Z3 subprocess runner
# ---------------------------------------------------------------------------

_CODE_BLOCK_PATTERN = re.compile(r"```python\s*(.*?)```", re.DOTALL)


def run_z3_code(code: str, timeout_s: float = 2.0) -> Z3Result:
    """Execute Z3 Python code in a subprocess sandbox and parse the verdict.

    **Detailed explanation for engineers:**
        We use a fresh subprocess rather than exec() for two reasons:
        1. Safety: malformed generated code cannot affect the parent process.
        2. Timeout: subprocess.run(..., timeout=timeout_s) kills the child
           after ``timeout_s`` seconds, preventing runaway solvers.

        Stdout parsing:
        - "unsat" anywhere in stdout → sat_status="unsat" (contradiction found)
        - "sat" anywhere in stdout   → sat_status="sat"   (consistent)
        - neither                    → sat_status="unknown"

        Note: "unsat" is checked before "sat" because "unsat" contains "sat"
        as a substring.

        Error handling:
        - subprocess.TimeoutExpired → sat_status="unknown", runtime_ms >= timeout.
        - Non-zero exit code with stderr → sat_status="error", error_message set.
        - Empty code → runs as no-op; stdout will be empty → "unknown".

    Args:
        code:      Self-contained Python code to execute.
        timeout_s: Hard time limit in seconds (default 2.0).

    Returns:
        Z3Result with the solver verdict and execution metadata.

    Spec: REQ-EXTRACT-011, SCENARIO-EXTRACT-023
    """
    if not code.strip():
        return Z3Result(sat_status="unknown", z3_code=code, runtime_ms=0.0)

    start = time.monotonic()
    try:
        proc = subprocess.run(
            [sys.executable, "-c", code],
            timeout=timeout_s,
            capture_output=True,
            text=True,
        )
        runtime_ms = (time.monotonic() - start) * 1000.0
        stdout = proc.stdout.strip()
        stderr = proc.stderr.strip()

        if proc.returncode != 0:
            # The generated code crashed (SyntaxError, NameError, ImportError …)
            error_msg = stderr if stderr else f"exit code {proc.returncode}"
            return Z3Result(
                sat_status="error",
                z3_code=code,
                runtime_ms=runtime_ms,
                error_message=error_msg,
            )

        # Parse Z3 verdict from stdout.
        # Check "unsat" FIRST because "unsat" contains "sat" as a substring.
        if "unsat" in stdout:
            return Z3Result(sat_status="unsat", z3_code=code, runtime_ms=runtime_ms)
        if "sat" in stdout:
            return Z3Result(sat_status="sat", z3_code=code, runtime_ms=runtime_ms)
        return Z3Result(sat_status="unknown", z3_code=code, runtime_ms=runtime_ms)

    except subprocess.TimeoutExpired:
        runtime_ms = (time.monotonic() - start) * 1000.0
        return Z3Result(
            sat_status="unknown",
            z3_code=code,
            runtime_ms=runtime_ms,
            error_message=f"subprocess timed out after {timeout_s}s",
        )


# ---------------------------------------------------------------------------
# NL2Z3Extractor
# ---------------------------------------------------------------------------

# Type alias for the injectable generate function (matches LLMConstraintExtractor pattern)
_GenerateFn = Callable[[str], str]


def _default_generate(prompt: str) -> str:
    """Call the real LLM.  Only invoked when CARNOT_FORCE_LIVE=1."""
    from carnot.inference.model_loader import generate, load_model  # deferred import

    model, tokenizer = load_model("Qwen/Qwen2.5-0.5B-Instruct")
    return generate(model, tokenizer, prompt, max_new_tokens=256)


class NL2Z3Extractor:
    """Detect internally inconsistent chain-of-thought via LLM → Z3 translation.

    **Detailed explanation for engineers:**
        This extractor is the answer to the constraint-extraction bottleneck
        documented in Exp 203/207: regex patterns find 0 violations because
        they cannot parse the logical structure of a multi-step reasoning trace.

        NL2Z3Extractor delegates that understanding to the same class of model
        that produced the trace, but asks it to output formal Z3 code instead
        of prose.  Z3 then checks the math rigorously.

        CI / production split:
        - CI (CARNOT_FORCE_LIVE not set): no LLM call; returns empty list with
          sat_status="unknown".  Tests run in under 1 s with zero GPU usage.
        - Production (CARNOT_FORCE_LIVE=1): calls the injected or default
          generate_fn, parses the code block, runs Z3 via subprocess.

        Injectable generate_fn:
        - In tests: pass a MagicMock that returns canned Z3 code.
        - In production: leave as None to use _default_generate.

        last_z3_result:
        - Set after every extract() call so callers can inspect the raw verdict.

    Attributes:
        last_z3_result: The Z3Result from the most recent extract() call.

    Spec: REQ-EXTRACT-010, SCENARIO-EXTRACT-020, SCENARIO-EXTRACT-021,
          SCENARIO-EXTRACT-024
    """

    def __init__(
        self,
        generate_fn: _GenerateFn | None = None,
        timeout_s: float = 2.0,
    ) -> None:
        self._generate_fn = generate_fn or _default_generate
        self._timeout_s = timeout_s
        self.last_z3_result: Z3Result | None = None

    @property
    def supported_domains(self) -> list[str]:
        """Domains this extractor handles: chain-of-thought reasoning traces."""
        return ["reasoning"]

    def extract(
        self,
        question: str,
        response: str,
        domain: str | None = None,
    ) -> list[ConstraintResult]:
        """Extract Z3 violations from a chain-of-thought response.

        **Detailed explanation for engineers:**
            1. Domain filter: skip if caller specified a domain that is not
               "reasoning".  This lets AutoExtractor route correctly.
            2. LLM guard: if CARNOT_FORCE_LIVE is not set, return [] immediately
               with a "unknown" Z3Result — safe for CI.
            3. LLM call: call generate_fn with the Z3 prompt.
            4. Code extraction: pull out the ```python … ``` block from the LLM
               output.  If none found, return [].
            5. Z3 execution: run_z3_code() with self._timeout_s.
            6. Violation emission: if sat_status=="unsat", return one
               ConstraintResult with constraint_type="z3_unsat".

        Args:
            question: The original question posed to the LLM.
            response: The chain-of-thought response to verify.
            domain:   Optional domain hint; non-"reasoning" domains are skipped.

        Returns:
            List of ConstraintResult (empty when no violation, or CI mode).

        Spec: REQ-EXTRACT-010
        """
        # Domain filter: skip if domain is set and is not "reasoning".
        if domain is not None and domain not in self.supported_domains:
            return []

        # CI guard: skip LLM call when not in live mode.
        if not os.environ.get("CARNOT_FORCE_LIVE"):
            self.last_z3_result = Z3Result(
                sat_status="unknown", z3_code="", runtime_ms=0.0
            )
            return []

        # Build the combined prompt (system + user as a single string for
        # models that don't support multi-turn chat templates).
        system, user = build_z3_prompt(response)
        full_prompt = f"{system}\n\n{user}"

        # Call the LLM and parse the Z3 code block.
        try:
            llm_output = self._generate_fn(full_prompt)
        except Exception as exc:  # noqa: BLE001
            self.last_z3_result = Z3Result(
                sat_status="unknown",
                z3_code="",
                runtime_ms=0.0,
                error_message=str(exc),
            )
            return []

        z3_code = self._extract_code_block(llm_output)
        if not z3_code:
            self.last_z3_result = Z3Result(
                sat_status="unknown", z3_code="", runtime_ms=0.0
            )
            return []

        z3_result = run_z3_code(z3_code, timeout_s=self._timeout_s)
        self.last_z3_result = z3_result

        if z3_result.sat_status == "unsat":
            return [
                ConstraintResult(
                    constraint_type="z3_unsat",
                    description=(
                        "Z3 found the reasoning chain internally inconsistent (UNSAT). "
                        "The arithmetic constraints derived from the response contradict "
                        "each other."
                    ),
                    metadata={
                        "z3_code": z3_code,
                        "runtime_ms": z3_result.runtime_ms,
                        "sat_status": "unsat",
                    },
                )
            ]
        return []

    @staticmethod
    def _extract_code_block(text: str) -> str:
        """Pull the first ```python ... ``` block from LLM output.

        Returns the code inside the fences, or empty string if none found.
        This is intentionally lenient: it does not validate that the code is
        valid Python — that is Z3's job.
        """
        match = _CODE_BLOCK_PATTERN.search(text)
        if match:
            return match.group(1).strip()
        return ""
