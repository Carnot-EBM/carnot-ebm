"""Two-round iterative code repair pipeline for LLM-generated code.

**Researcher summary (arXiv 2604.10508):**
    "How Many Tries Does It Take?" shows self-repair universally improves pass
    rates by +4.9 to +17.1pp on HumanEval, with most gains in the FIRST TWO
    ROUNDS.  This module implements exactly two repair rounds so Exp 744 can
    measure whether Carnot's execution-based code path replicates that result.

**Why execution-based instead of regex extraction?**
    The paper's gains rely on feeding real execution errors back to the model.
    Regex extraction guesses at errors from text patterns — it frequently
    misses assertion failures where the code *runs* but produces the wrong
    answer.  Carnot's CodeExtractor already exercises actual Python execution,
    so we extend that path here rather than adding a regex layer on top.

**Error classification rationale:**
    The paper finds assertion errors are hardest to repair (~45% repair rate)
    while syntax and name errors are easiest.  We track error type per problem
    so Exp 744 can verify the same distribution holds for Qwen3.5-0.8B.

**Repair prompt design:**
    The key signal is the full traceback plus expected vs actual output.  We
    include:
    1. The original problem statement (so the model knows the intent).
    2. The failing code (so the model can see what went wrong).
    3. The full traceback (so the model sees the exact error type and line).
    4. Expected vs actual output (so the model understands the correctness gap).
    A plain instruction "The code above has a bug. Fix it." follows the error
    context — the paper shows this simple phrasing outperforms elaborate
    meta-instructions.

Spec: REQ-CODE-031, REQ-CODE-032, SCENARIO-CODE-029, SCENARIO-CODE-030
"""

from __future__ import annotations

import signal
import sys
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable

_EXECUTION_TIMEOUT_S = 10
"""Hard timeout in seconds for a single code execution attempt.

Why 10 s: HumanEval problems are algorithmic, not I/O bound.  Any correct
solution runs in under 1 s.  10 s catches infinite loops with a comfortable
margin while keeping the benchmark tractable at 50 problems × 3 rounds.
"""


# ---------------------------------------------------------------------------
# ExecutionResult
# ---------------------------------------------------------------------------


@dataclass
class ExecutionResult:
    """Outcome of running generated code against a set of test cases.

    Fields
    ------
    passed : bool
        True iff ALL test cases executed without raising an exception.
    stdout : str
        Captured standard output during execution (may be empty).
    stderr : str
        Captured standard error during execution (may be empty).
    traceback_str : str
        Full traceback string when an exception occurred; empty string on pass.
    error_type : str
        Classified error category — one of:
        ``"syntax_error"``, ``"assertion_error"``, ``"name_error"``,
        ``"timeout"``, ``"other"``, or ``""`` when all tests passed.
    actual_output : str
        String representation of the first failing call's actual return value,
        or ``"<no return value>"`` when an exception prevented a return.
    expected_output : str
        String representation of the expected value from the first failing test
        case, or ``"<unknown>"`` when the test did not specify one.

    Spec: REQ-CODE-031
    """

    passed: bool
    stdout: str = ""
    stderr: str = ""
    traceback_str: str = ""
    error_type: str = ""
    actual_output: str = ""
    expected_output: str = ""


# ---------------------------------------------------------------------------
# TwoRoundResult
# ---------------------------------------------------------------------------


@dataclass
class TwoRoundResult:
    """Per-problem outcome across all three code generation rounds.

    Fields
    ------
    round0_pass : bool
        True iff the initial (pre-repair) code passed all test cases.
    round1_pass : bool
        True iff the round-1 repaired code passed all test cases.
        Meaningful only when ``round0_pass`` is False.
    round2_pass : bool
        True iff the round-2 repaired code passed all test cases.
        Meaningful only when ``round0_pass`` and ``round1_pass`` are False.
    round0_code : str
        Code string from round 0 (initial generation).
    round1_code : str
        Code string from round 1 (first repair attempt); empty if round 0 passed.
    round2_code : str
        Code string from round 2 (second repair attempt); empty if round 0 or 1 passed.
    error_types : list[str]
        Error type from each round that failed.  At most two entries
        (round 0 error, round 1 error).  Empty if round 0 passed immediately.

    Spec: REQ-CODE-031, REQ-CODE-032
    """

    round0_pass: bool
    round1_pass: bool
    round2_pass: bool
    round0_code: str = ""
    round1_code: str = ""
    round2_code: str = ""
    error_types: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# TwoRoundCodeRepairPipeline
# ---------------------------------------------------------------------------


class TwoRoundCodeRepairPipeline:
    """Generate code, run it, and iteratively repair up to two rounds on failure.

    This class is deliberately stateless between calls to ``run()`` so that
    it is safe to reuse across many HumanEval problems without state leakage.

    **Round logic:**
    - Round 0: generate code from the problem prompt and execute.
    - Round 1: if round 0 failed, build a repair prompt with the traceback and
      re-generate.  Execute the new code.
    - Round 2: if round 1 failed, build another repair prompt with the new
      traceback and re-generate.  Execute.  No further repair is attempted.

    **Why only two rounds?**
    arXiv 2604.10508 shows gains concentrate almost entirely in rounds 1 and 2.
    Round 3+ adds noise without measurable improvement and triples wall-clock
    time.  Two rounds is the empirical sweet spot.

    Spec: REQ-CODE-031, REQ-CODE-032
    """

    def generate(self, problem: str, llm_caller: Callable[[str], str]) -> str:
        """Generate initial code from a problem statement using the provided LLM.

        Parameters
        ----------
        problem : str
            The full HumanEval-style problem prompt (docstring + signature).
        llm_caller : callable
            Function accepting a prompt string and returning the model's response.

        Returns
        -------
        str
            Raw model response; may contain markdown code fences.

        Spec: REQ-CODE-031
        """
        prompt = (
            "You are an expert Python programmer.  Write a correct Python function "
            "that solves the following problem.  Return ONLY the function definition "
            "with no extra explanation.\n\n"
            f"{problem}"
        )
        return llm_caller(prompt)

    def _extract_code(self, response: str) -> str:
        """Strip markdown code fences from a model response.

        Why this is needed: LLMs often wrap their answer in ```python ... ```
        fences even when instructed to return only a function.  We strip the
        outermost fence pair and return the inner code so exec() can parse it.

        Args:
            response: Raw model response, possibly fenced.

        Returns:
            Clean Python source code string.
        """
        lines = response.strip().splitlines()
        # Find and strip the first ``` fence pair.
        start = 0
        end = len(lines)
        if lines and lines[0].startswith("```"):
            start = 1
        if lines and lines[-1].strip() == "```":
            end = len(lines) - 1
        return "\n".join(lines[start:end]).strip()

    def _classify_error(self, tb_str: str) -> str:
        """Map a traceback string to one of the canonical error-type labels.

        We check for the most specific error names first (SyntaxError, etc.)
        because a traceback can contain multiple lines and we want the primary
        exception class, not an intermediate one.

        Args:
            tb_str: Full traceback string from traceback.format_exc().

        Returns:
            One of ``"syntax_error"``, ``"assertion_error"``, ``"name_error"``,
            ``"timeout"``, or ``"other"``.

        Spec: REQ-CODE-031
        """
        if not tb_str:
            return ""
        if "TimeoutError" in tb_str or "timeout" in tb_str.lower():
            return "timeout"
        if "SyntaxError" in tb_str:
            return "syntax_error"
        if "AssertionError" in tb_str:
            return "assertion_error"
        if "NameError" in tb_str:
            return "name_error"
        return "other"

    def execute(self, code: str, test_cases: list[dict[str, Any]]) -> ExecutionResult:
        """Run code against test cases and capture any execution errors.

        The code is executed in an isolated namespace so that each call starts
        from a clean global state.  A per-call timeout of ``_EXECUTION_TIMEOUT_S``
        seconds is enforced via SIGALRM (Unix only); when the signal fires we
        raise TimeoutError and classify it as ``"timeout"``.

        Test case format:
            Each dict in ``test_cases`` must have:
            - ``"call"`` (str): Python expression that calls the generated function.
            - ``"expected"`` (Any, optional): Expected return value for display.

        Args:
            code: Python source code string to execute.
            test_cases: List of test-case dicts.

        Returns:
            ExecutionResult with ``passed=True`` iff all cases ran without error.

        Spec: REQ-CODE-031
        """
        namespace: dict[str, Any] = {}

        # --- Compile first to catch SyntaxError before running anything ---
        try:
            compiled = compile(code, "<generated>", "exec")
        except SyntaxError:
            tb = traceback.format_exc()
            return ExecutionResult(
                passed=False,
                traceback_str=tb,
                error_type="syntax_error",
            )

        # --- Execute the function definition in isolated namespace ---
        try:
            exec(compiled, namespace)  # noqa: S102 — intentional sandboxed exec
        except Exception:
            tb = traceback.format_exc()
            return ExecutionResult(
                passed=False,
                traceback_str=tb,
                error_type=self._classify_error(tb),
            )

        # --- Run each test case ---
        for tc in test_cases:
            call_expr = tc.get("call", "")
            expected = tc.get("expected", "<unknown>")

            # Set up SIGALRM timeout on Unix platforms.
            _alarm_available = hasattr(signal, "SIGALRM")
            if _alarm_available:
                def _timeout_handler(signum: int, frame: Any) -> None:  # noqa: ARG001
                    raise TimeoutError(f"execution exceeded {_EXECUTION_TIMEOUT_S}s")
                signal.signal(signal.SIGALRM, _timeout_handler)
                signal.alarm(_EXECUTION_TIMEOUT_S)

            try:
                actual = eval(call_expr, namespace)  # noqa: S307 — test case expression
                if _alarm_available:
                    signal.alarm(0)  # cancel alarm
            except Exception:
                if _alarm_available:
                    signal.alarm(0)
                tb = traceback.format_exc()
                return ExecutionResult(
                    passed=False,
                    traceback_str=tb,
                    error_type=self._classify_error(tb),
                    actual_output="<no return value>",
                    expected_output=str(expected),
                )

            # Check equality when an expected value is provided.
            if "expected" in tc and actual != expected:
                return ExecutionResult(
                    passed=False,
                    traceback_str="",
                    error_type="assertion_error",
                    actual_output=repr(actual),
                    expected_output=repr(expected),
                )

        return ExecutionResult(passed=True)

    def build_repair_prompt(
        self,
        original_problem: str,
        failed_code: str,
        traceback_str: str,
        expected_output: str,
        actual_output: str,
    ) -> str:
        """Build a repair prompt that includes all context needed to fix the bug.

        The prompt structure (from arXiv 2604.10508 ablation):
        1. Original problem statement — the model needs the intent to fix the bug.
        2. Failing code — the model must see what it wrote.
        3. Full traceback — the most informative signal for the model.
        4. Expected vs actual output — explicit correctness gap.
        5. Repair instruction — simple directive outperforms elaborate prompting.

        Args:
            original_problem: The original HumanEval problem text.
            failed_code: The code that failed the test cases.
            traceback_str: Full traceback string, or error description.
            expected_output: String representation of the expected output.
            actual_output: String representation of the actual (wrong) output.

        Returns:
            Formatted repair prompt string ready for the LLM caller.

        Spec: REQ-CODE-031
        """
        parts = [
            "You are an expert Python programmer.  The code below has a bug.",
            "",
            "## Original Problem",
            original_problem.strip(),
            "",
            "## Failing Code",
            "```python",
            failed_code.strip(),
            "```",
            "",
            "## Execution Error",
            traceback_str.strip() if traceback_str.strip() else "(no traceback — wrong output)",
            "",
            "## Expected Output",
            expected_output if expected_output else "<unknown>",
            "",
            "## Actual Output",
            actual_output if actual_output else "<no return value>",
            "",
            "Fix the bug in the code above.  Return ONLY the corrected function "
            "definition with no extra explanation.",
        ]
        return "\n".join(parts)

    def repair(self, repair_prompt: str, llm_caller: Callable[[str], str]) -> str:
        """Generate a repaired version of the code using the repair prompt.

        Args:
            repair_prompt: Prompt built by ``build_repair_prompt()``.
            llm_caller: Function accepting a prompt string and returning model response.

        Returns:
            Raw model response containing the repaired code.

        Spec: REQ-CODE-031
        """
        return llm_caller(repair_prompt)

    def run(
        self,
        problem: str,
        test_cases: list[dict[str, Any]],
        llm_caller: Callable[[str], str],
    ) -> TwoRoundResult:
        """Execute the full two-round repair loop for a single problem.

        Rounds:
        - Round 0: generate and test initial code.
        - Round 1: repair if round 0 failed; test repaired code.
        - Round 2: repair if round 1 failed; test repaired code.  Stop here.

        Args:
            problem: Full problem prompt string.
            test_cases: Test cases list (see ``execute()`` for format).
            llm_caller: LLM inference function.

        Returns:
            TwoRoundResult with pass/fail per round and all code strings.

        Spec: REQ-CODE-031, REQ-CODE-032
        """
        error_types: list[str] = []

        # --- Round 0 ---
        raw0 = self.generate(problem, llm_caller)
        code0 = self._extract_code(raw0)
        result0 = self.execute(code0, test_cases)

        if result0.passed:
            return TwoRoundResult(
                round0_pass=True,
                round1_pass=False,
                round2_pass=False,
                round0_code=code0,
                round1_code="",
                round2_code="",
                error_types=[],
            )

        error_types.append(result0.error_type)

        # --- Round 1 ---
        repair_prompt1 = self.build_repair_prompt(
            original_problem=problem,
            failed_code=code0,
            traceback_str=result0.traceback_str,
            expected_output=result0.expected_output,
            actual_output=result0.actual_output,
        )
        raw1 = self.repair(repair_prompt1, llm_caller)
        code1 = self._extract_code(raw1)
        result1 = self.execute(code1, test_cases)

        if result1.passed:
            return TwoRoundResult(
                round0_pass=False,
                round1_pass=True,
                round2_pass=False,
                round0_code=code0,
                round1_code=code1,
                round2_code="",
                error_types=error_types,
            )

        error_types.append(result1.error_type)

        # --- Round 2 ---
        repair_prompt2 = self.build_repair_prompt(
            original_problem=problem,
            failed_code=code1,
            traceback_str=result1.traceback_str,
            expected_output=result1.expected_output,
            actual_output=result1.actual_output,
        )
        raw2 = self.repair(repair_prompt2, llm_caller)
        code2 = self._extract_code(raw2)
        result2 = self.execute(code2, test_cases)

        return TwoRoundResult(
            round0_pass=False,
            round1_pass=False,
            round2_pass=result2.passed,
            round0_code=code0,
            round1_code=code1,
            round2_code=code2,
            error_types=error_types,
        )
