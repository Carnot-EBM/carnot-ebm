"""Iterative Self-Repair pipeline — execution-feedback-driven code repair.

**Researcher summary (Exp 905):**
    Code repair experiments Exps 850-881 all failed with honest_verdict='zero_constraints'
    because ArithmeticExtractor's regex patterns ('a+b=c' style) never match the natural
    language responses that instruction-tuned LLMs produce.  The extractor extracts zero
    constraints, so VerifyRepairPipeline has no repair signal.

    This module bypasses ArithmeticExtractor entirely.  Instead of trying to extract
    symbolic constraints from free-form text, we:

        1. Generate code with an LLM (first attempt, attempt 0).
        2. Execute the code in a subprocess sandbox.
        3. If execution raises an exception, feed the FULL traceback back to the LLM
           as a correction prompt (attempt 1, 2, … up to max_retries).
        4. Carnot's energy scorer ranks all attempts; the lowest-energy attempt is
           returned as the best solution regardless of whether it passed.

    This is the approach from arXiv 2604.10508 ("Iterative Self-Repair"), which showed
    +4.9pp to +17.1pp improvement on HumanEval and MBPP with GPT-4 and Claude 3.
    Here we implement it for local Qwen3 MoE (unsloth/Qwen3.6-35B-A3B-GGUF).

    Why execution error IS the constraint signal:
        The execution traceback tells the model exactly which line failed, what
        exception was raised, and what the expected vs. actual values were.  This
        is far richer than any symbolic constraint we could extract from free text.
        The model already knows how to interpret tracebacks — it was trained on
        millions of them.

    Energy scorer role:
        After all attempts are collected, the Ising energy scorer ranks them.
        The energy score reflects how "thermodynamically plausible" the code
        structure is — lower energy = more regular, more likely to be correct.
        When multiple attempts pass the tests, we return the lowest-energy passer.
        When no attempt passes, we return the lowest-energy attempt overall as
        the best guess.

Spec: REQ-CODE-033 (IterativeSelfRepair pipeline),
      SCENARIO-CODE-031 (retry with execution feedback until passing or budget)
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from dataclasses import dataclass, field
from typing import Any, Protocol


# ---------------------------------------------------------------------------
# Protocols — let callers inject any LLM runner or energy scorer
# ---------------------------------------------------------------------------


class LLMRunner(Protocol):
    """Minimal protocol for an LLM that can generate text from a prompt.

    Why a Protocol instead of an abstract base class: Python structural typing
    lets the caller inject ANY object that has a generate() method without
    forcing it to inherit from a Carnot base class.  This keeps the pipeline
    decoupled from the specific LLM backend (llama.cpp, HuggingFace, OpenAI).
    """

    def generate(self, prompt: str) -> str:
        """Return the model's text completion of *prompt*."""
        ...


class EnergyScorer(Protocol):
    """Minimal protocol for a component that scores text energy.

    Lower return value means the scorer believes the text is more correct /
    thermodynamically stable.  The Ising pipeline's score() method fits this
    protocol directly.
    """

    def score(self, text: str) -> float:
        """Return a scalar energy for *text*.  Lower = more plausible."""
        ...


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ExecResult:
    """Outcome of executing a code candidate against test cases.

    Fields
    ------
    passed : bool
        True if the code ran without raising any exception and all test
        assertions succeeded.
    error : str | None
        The full traceback string when execution failed, or None on success.
        The full traceback is essential for repair prompts — arXiv 2604.10508
        shows that traceback quality is the primary driver of repair gains.
    timed_out : bool
        True if the subprocess was killed because it exceeded the timeout.
        Timeout is reported as a separate field so the repair prompt can
        tell the model "your code timed out" rather than "your code errored".
    """

    passed: bool
    error: str | None = None
    timed_out: bool = False


@dataclass
class RepairAttempt:
    """One attempt in an iterative self-repair loop.

    Fields
    ------
    attempt_index : int
        Zero-based index.  Attempt 0 is the initial generation; attempts
        1+ are repair rounds.
    response : str
        The raw LLM response for this attempt (may be wrapped in markdown
        fences — _extract_code() strips them).
    exec_passed : bool
        Whether the extracted code passed all test cases.
    exec_error : str | None
        Full traceback if execution failed, None if it passed.
    energy_score : float
        Ising energy for this attempt.  Lower is more plausible.
    """

    attempt_index: int
    response: str
    exec_passed: bool
    exec_error: str | None
    energy_score: float


@dataclass
class RepairResult:
    """Final outcome of an IterativeSelfRepair.repair() call.

    Fields
    ------
    best_attempt : RepairAttempt
        The attempt selected by the energy scorer as the best solution.
        When any attempt passes tests, best_attempt is the lowest-energy
        passer.  When no attempt passes, best_attempt is the lowest-energy
        attempt overall.
    all_attempts : list[RepairAttempt]
        Every attempt in order, including the initial generation.
    n_retries : int
        Number of repair rounds executed (= len(all_attempts) - 1).
    energy_selected_passing : bool
        True when best_attempt.exec_passed == True.  Tracks whether the
        energy scorer successfully identified a passing solution.
    """

    best_attempt: RepairAttempt
    all_attempts: list[RepairAttempt]
    n_retries: int
    energy_selected_passing: bool


# ---------------------------------------------------------------------------
# IterativeSelfRepair
# ---------------------------------------------------------------------------


class IterativeSelfRepair:
    """Iterative self-repair loop using execution error as correction signal.

    The approach (arXiv 2604.10508):
        1. Generate code with LLM.
        2. Execute against test cases.
        3. On failure: build a correction prompt containing the original code,
           the full traceback, and the test cases that failed.
        4. Re-generate and repeat up to max_retries times.
        5. Select the best attempt by Carnot energy score.

    This bypasses ArithmeticExtractor entirely — the execution traceback IS the
    constraint signal.  No regex pattern matching needed.

    Parameters
    ----------
    llm_runner : LLMRunner
        Any object with a generate(prompt: str) -> str method.
    energy_scorer : EnergyScorer
        Any object with a score(text: str) -> float method.  Lower energy =
        more plausible.  The IsingPipeline fits this interface directly.
    max_retries : int
        Maximum number of repair rounds after the initial generation.
        0 means only the initial attempt is made (no repair).
        Default 3, matching arXiv 2604.10508's experimental setup.
    sandbox : bool
        When True AND CARNOT_USE_SANDBOX=1 is set, execute code inside a
        gVisor-sandboxed subprocess.  When False, use a plain subprocess.
        Sandbox mode adds latency but protects against malicious code.
    exec_timeout_s : float
        Per-attempt execution timeout in seconds.  Default 10s, enough for
        any HumanEval problem but short enough to detect infinite loops.
    """

    def __init__(
        self,
        llm_runner: LLMRunner,
        energy_scorer: EnergyScorer,
        max_retries: int = 3,
        sandbox: bool = True,
        exec_timeout_s: float = 10.0,
    ) -> None:
        self.llm = llm_runner
        self.energy = energy_scorer
        self.max_retries = max_retries
        self.sandbox = sandbox
        self.exec_timeout_s = exec_timeout_s
        self._attempt_log: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def repair(self, problem: str, test_cases: list[str]) -> RepairResult:
        """Run the iterative self-repair loop for one problem.

        The loop generates an initial solution, executes it, and if it fails,
        feeds the traceback back to the LLM as a correction prompt.  This
        repeats up to max_retries times.

        The best attempt is selected by energy score (lower = better).
        When any attempt passes tests, we restrict the selection to passing
        attempts before applying the energy criterion.

        Parameters
        ----------
        problem : str
            The problem description / function signature + docstring.
            Passed to the LLM as the initial generation prompt.
        test_cases : list[str]
            Python assert statements or function calls to verify the solution.
            Each string is executed in the same namespace as the generated code.

        Returns
        -------
        RepairResult with the best attempt and the full attempt log.
        """
        self._attempt_log = []
        attempts: list[RepairAttempt] = []

        # Attempt 0: initial generation
        response = self.llm.generate(problem)

        for i in range(self.max_retries + 1):
            exec_result = self._sandbox_exec(response, test_cases)
            score = self.energy.score(response)

            attempt = RepairAttempt(
                attempt_index=i,
                response=response,
                exec_passed=exec_result.passed,
                exec_error=exec_result.error,
                energy_score=score,
            )
            attempts.append(attempt)
            self._attempt_log.append(
                {
                    "attempt": i,
                    "exec_passed": exec_result.passed,
                    "energy_score": score,
                    "timed_out": exec_result.timed_out,
                }
            )

            if exec_result.passed:
                # Found a passing solution — no need to continue.
                break

            if i < self.max_retries:
                # Build a correction prompt: give the model its own code back
                # plus the full traceback.  This is the core insight of
                # arXiv 2604.10508 — the error message IS the repair signal.
                error_text = exec_result.error or "unknown error (no traceback)"
                correction_prompt = (
                    f"{problem}\n\n"
                    f"Your previous attempt produced the following code:\n"
                    f"```python\n{response}\n```\n\n"
                    f"It failed with this error:\n"
                    f"```\n{error_text}\n```\n\n"
                    f"Fix the code so that it passes all the tests. "
                    f"Return ONLY the corrected Python code."
                )
                response = self.llm.generate(correction_prompt)

        # Select best attempt: prefer passing attempts; within passing (or
        # when none pass), pick lowest energy score.
        passing = [a for a in attempts if a.exec_passed]
        candidate_pool = passing if passing else attempts
        best = min(candidate_pool, key=lambda a: a.energy_score)

        return RepairResult(
            best_attempt=best,
            all_attempts=attempts,
            n_retries=len(attempts) - 1,
            energy_selected_passing=best.exec_passed,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _sandbox_exec(self, code: str, test_cases: list[str]) -> ExecResult:
        """Execute *code* + *test_cases* and return pass/fail with traceback.

        Execution strategy:
            - If CARNOT_USE_SANDBOX=1 AND self.sandbox=True: run inside gVisor
              via `runsc --platform=kvm run` wrapping a Docker container.
              This provides strong isolation against malicious code.
            - Otherwise: run in a subprocess with a timeout.  The subprocess
              runs in a fresh Python interpreter so generated code cannot
              affect the parent process's state.

        Why subprocess instead of exec() in-process:
            HumanEval solutions can call sys.exit(), import arbitrary modules,
            or run infinite loops.  A subprocess is killed cleanly by timeout;
            an in-process exec() would require threading + signal gymnastics.

        Parameters
        ----------
        code : str
            Python source code to execute.  May be wrapped in markdown fences
            (```python ... ```) — these are stripped before execution.
        test_cases : list[str]
            Python assert statements.  Each is appended to the code before
            execution so the test can call functions defined in *code*.

        Returns
        -------
        ExecResult(passed=bool, error=str|None, timed_out=bool)
        """
        clean_code = _extract_code(code)
        test_block = "\n".join(test_cases)
        full_script = f"{clean_code}\n\n# --- test cases ---\n{test_block}\n"

        use_gvisor = (
            self.sandbox
            and os.environ.get("CARNOT_USE_SANDBOX", "0") == "1"
        )

        if use_gvisor:
            return self._exec_gvisor(full_script)
        return self._exec_subprocess(full_script)

    def _exec_subprocess(self, script: str) -> ExecResult:
        """Run *script* in a fresh subprocess with timeout protection.

        Uses sys.executable so the same Python interpreter runs the script —
        important when the experiment is running inside a venv.
        """
        try:
            result = subprocess.run(
                [sys.executable, "-c", script],
                capture_output=True,
                text=True,
                timeout=self.exec_timeout_s,
            )
        except subprocess.TimeoutExpired:
            return ExecResult(
                passed=False,
                error=f"TimeoutError: execution exceeded {self.exec_timeout_s}s",
                timed_out=True,
            )
        except Exception as exc:
            return ExecResult(passed=False, error=str(exc))

        if result.returncode == 0:
            return ExecResult(passed=True)

        # Combine stdout + stderr for the most informative traceback.
        error_text = (result.stderr or "") + (result.stdout or "")
        return ExecResult(passed=False, error=error_text.strip() or "non-zero exit")

    def _exec_gvisor(self, script: str) -> ExecResult:
        """Run *script* inside a gVisor sandbox via Docker with runsc runtime.

        gVisor intercepts syscalls in userspace, providing strong isolation
        without requiring hardware virtualisation.  It plays nicely with
        nvidia-container-toolkit so GPU access is preserved when needed.

        Falls back to _exec_subprocess() if Docker is not available on the
        current host — this ensures CI machines without gVisor still work.
        """
        try:
            result = subprocess.run(
                [
                    "docker",
                    "run",
                    "--rm",
                    "--runtime=runsc",
                    "--network=none",
                    "python:3.11-slim",
                    "python",
                    "-c",
                    script,
                ],
                capture_output=True,
                text=True,
                timeout=self.exec_timeout_s + 30,  # docker startup overhead
            )
        except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
            # Docker not available or timed out — fall back to subprocess.
            if isinstance(exc, subprocess.TimeoutExpired):
                return ExecResult(
                    passed=False,
                    error=f"TimeoutError: gVisor execution exceeded timeout",
                    timed_out=True,
                )
            return self._exec_subprocess(script)
        except Exception as exc:
            return self._exec_subprocess(script)

        if result.returncode == 0:
            return ExecResult(passed=True)
        error_text = (result.stderr or "") + (result.stdout or "")
        return ExecResult(passed=False, error=error_text.strip() or "non-zero exit")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_code(response: str) -> str:
    """Strip markdown fences from an LLM response to get raw Python source.

    LLMs typically wrap their code in ```python ... ``` blocks.  This
    function extracts the content between the fences.  If no fences are
    found, the original string is returned unchanged (some models respond
    with bare code).

    Examples
    --------
    >>> _extract_code("```python\\nreturn 42\\n```")
    'return 42'
    >>> _extract_code("def foo(): return 1")
    'def foo(): return 1'
    """
    stripped = response.strip()

    # Try ```python ... ``` first, then ``` ... ```.
    for fence in ("```python", "```"):
        if stripped.startswith(fence):
            rest = stripped[len(fence):]
            end_idx = rest.rfind("```")
            if end_idx != -1:
                return rest[:end_idx].strip()

    return stripped
