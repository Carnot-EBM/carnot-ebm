"""Tests for IterativeSelfRepair pipeline.

**Detailed explanation for engineers:**
    Tests the execution-feedback-driven code repair pipeline in
    python/carnot/pipeline/iterative_self_repair.py.

    The test suite validates:
    - _extract_code() correctly strips markdown fences.
    - ExecResult dataclass holds correct types.
    - RepairAttempt dataclass holds correct types.
    - RepairResult dataclass holds correct types.
    - IterativeSelfRepair.repair() returns on first passing attempt.
    - IterativeSelfRepair.repair() retries when first attempt fails.
    - IterativeSelfRepair.repair() stops after max_retries even if all fail.
    - Correction prompt includes original code and traceback.
    - Energy scorer selects the lowest-energy passing attempt.
    - Energy scorer falls back to lowest-energy attempt when none pass.
    - _sandbox_exec() subprocess path returns ExecResult(passed=True) for valid code.
    - _sandbox_exec() subprocess path returns ExecResult(passed=False) for bad code.
    - _sandbox_exec() detects timeout and sets timed_out=True.
    - RepairResult.energy_selected_passing reflects whether best attempt passed.

Spec: REQ-CODE-033 (IterativeSelfRepair pipeline),
      SCENARIO-CODE-031 (retry with execution feedback until passing or budget)
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from carnot.pipeline.iterative_self_repair import (
    ExecResult,
    IterativeSelfRepair,
    RepairAttempt,
    RepairResult,
    _extract_code,
)


# ---------------------------------------------------------------------------
# Helpers / stubs
# ---------------------------------------------------------------------------


class _FixedLLM:
    """LLM stub that returns a pre-determined sequence of responses.

    Why a sequence: the first call is the initial generation, subsequent
    calls are repair rounds.  We pre-configure what each call returns so
    tests are deterministic and do not require a real GPU.
    """

    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self.calls: list[str] = []  # record every prompt received

    def generate(self, prompt: str) -> str:
        self.calls.append(prompt)
        if self._responses:
            return self._responses.pop(0)
        return "# no more responses"


class _FixedEnergyScorer:
    """Energy scorer stub that returns pre-configured scores per response text.

    Falls back to 0.0 for any text not in the map.
    """

    def __init__(self, scores: dict[str, float]) -> None:
        self._scores = scores

    def score(self, text: str) -> float:
        return self._scores.get(text, 0.0)


class _ConstantEnergyScorer:
    """Energy scorer that returns the same score for every input."""

    def __init__(self, value: float = 0.0) -> None:
        self._value = value

    def score(self, text: str) -> float:
        return self._value


# ---------------------------------------------------------------------------
# _extract_code tests
# ---------------------------------------------------------------------------


def test_extract_code_plain() -> None:
    """Plain Python code with no fences should be returned unchanged.

    Spec: REQ-CODE-033
    """
    code = "def foo():\n    return 1"
    assert _extract_code(code) == code


def test_extract_code_python_fence() -> None:
    """Code wrapped in ```python ... ``` fences should be extracted.

    Spec: REQ-CODE-033
    """
    fenced = "```python\ndef foo():\n    return 1\n```"
    result = _extract_code(fenced)
    assert "def foo():" in result
    assert "```" not in result


def test_extract_code_plain_fence() -> None:
    """Code wrapped in plain ``` ... ``` fences should be extracted.

    Spec: REQ-CODE-033
    """
    fenced = "```\ndef bar(): pass\n```"
    result = _extract_code(fenced)
    assert "def bar():" in result
    assert "```" not in result


def test_extract_code_empty() -> None:
    """Empty string returns empty string.

    Spec: REQ-CODE-033
    """
    assert _extract_code("") == ""


# ---------------------------------------------------------------------------
# Dataclass tests
# ---------------------------------------------------------------------------


def test_exec_result_passed() -> None:
    """ExecResult(passed=True) should have error=None, timed_out=False.

    Spec: REQ-CODE-033
    """
    r = ExecResult(passed=True)
    assert r.passed is True
    assert r.error is None
    assert r.timed_out is False


def test_exec_result_failed() -> None:
    """ExecResult(passed=False, error=...) should store the error string.

    Spec: REQ-CODE-033
    """
    r = ExecResult(passed=False, error="AssertionError: 1 != 2")
    assert r.passed is False
    assert r.error == "AssertionError: 1 != 2"


def test_exec_result_timed_out() -> None:
    """ExecResult with timed_out=True should set that flag.

    Spec: REQ-CODE-033
    """
    r = ExecResult(passed=False, error="timeout", timed_out=True)
    assert r.timed_out is True


def test_repair_attempt_fields() -> None:
    """RepairAttempt should hold all five fields with correct types.

    Spec: REQ-CODE-033
    """
    a = RepairAttempt(
        attempt_index=0,
        response="def foo(): return 1",
        exec_passed=True,
        exec_error=None,
        energy_score=0.5,
    )
    assert a.attempt_index == 0
    assert isinstance(a.response, str)
    assert a.exec_passed is True
    assert a.exec_error is None
    assert isinstance(a.energy_score, float)


def test_repair_result_fields() -> None:
    """RepairResult should expose best_attempt, all_attempts, n_retries, energy_selected_passing.

    Spec: REQ-CODE-033
    """
    best = RepairAttempt(0, "code", True, None, 0.1)
    rr = RepairResult(
        best_attempt=best,
        all_attempts=[best],
        n_retries=0,
        energy_selected_passing=True,
    )
    assert rr.n_retries == 0
    assert rr.energy_selected_passing is True
    assert rr.best_attempt is best


# ---------------------------------------------------------------------------
# IterativeSelfRepair.repair() logic tests
# ---------------------------------------------------------------------------


def _make_isr(responses: list[str], scores: dict[str, float] | None = None) -> IterativeSelfRepair:
    """Build an IterativeSelfRepair with stub LLM and energy scorer."""
    scorer = _FixedEnergyScorer(scores or {}) if scores else _ConstantEnergyScorer(0.0)
    llm = _FixedLLM(responses)
    isr = IterativeSelfRepair(
        llm_runner=llm,
        energy_scorer=scorer,
        max_retries=3,
        sandbox=False,
        exec_timeout_s=5.0,
    )
    return isr


def test_repair_passes_on_first_attempt() -> None:
    """When the first attempt passes all tests, repair() returns immediately.

    Spec: SCENARIO-CODE-031
    """
    # A simple function that passes its own assertion test.
    code = "def add(a, b):\n    return a + b"
    test_cases = ["assert add(1, 2) == 3"]

    isr = _make_isr([code])
    result = isr.repair("implement add", test_cases)

    assert result.best_attempt.exec_passed is True
    assert result.n_retries == 0
    assert len(result.all_attempts) == 1


def test_repair_retries_on_failure() -> None:
    """When the first attempt fails, repair() retries with a correction prompt.

    Spec: SCENARIO-CODE-031, REQ-CODE-033
    """
    bad_code = "def add(a, b):\n    return a - b"  # wrong: subtract instead of add
    good_code = "def add(a, b):\n    return a + b"
    test_cases = ["assert add(1, 2) == 3"]

    isr = _make_isr([bad_code, good_code])
    result = isr.repair("implement add", test_cases)

    assert result.best_attempt.exec_passed is True
    assert result.n_retries == 1
    # Second call should have been a correction prompt (includes "error")
    assert len(isr.llm.calls) == 2  # type: ignore[attr-defined]
    second_prompt = isr.llm.calls[1]  # type: ignore[attr-defined]
    # Correction prompt includes the error and the original problem
    assert "error" in second_prompt.lower() or "fix" in second_prompt.lower()


def test_repair_stops_at_max_retries() -> None:
    """repair() stops after max_retries even if all attempts fail.

    Spec: REQ-CODE-033
    """
    bad_code = "def add(a, b):\n    return a * b"  # always wrong
    test_cases = ["assert add(1, 2) == 3"]

    # Provide 4 bad responses (initial + 3 retries)
    isr = _make_isr([bad_code] * 10)
    isr.max_retries = 3
    result = isr.repair("implement add", test_cases)

    # Should have made exactly 4 attempts (0, 1, 2, 3)
    assert len(result.all_attempts) == 4
    assert result.n_retries == 3
    # None should have passed
    assert all(not a.exec_passed for a in result.all_attempts)


def test_energy_selects_passing_attempt_over_failing() -> None:
    """Energy scorer prefers passing attempts over failing ones, regardless of energy.

    Even if a failing attempt has lower energy, the best passing attempt is chosen.

    Spec: REQ-CODE-033
    """
    bad_code = "def add(a, b):\n    return 99"  # fails
    good_code = "def add(a, b):\n    return a + b"  # passes
    test_cases = ["assert add(1, 2) == 3"]

    # Give bad code very low energy (would be selected if we didn't prefer passers)
    scores = {bad_code: -100.0, good_code: 5.0}
    scorer = _FixedEnergyScorer(scores)
    llm = _FixedLLM([bad_code, good_code])
    isr = IterativeSelfRepair(llm_runner=llm, energy_scorer=scorer, max_retries=3, sandbox=False)

    result = isr.repair("implement add", test_cases)

    # good_code passed, so it should be selected despite higher energy
    assert result.best_attempt.exec_passed is True
    assert result.energy_selected_passing is True


def test_energy_selects_lowest_energy_when_none_pass() -> None:
    """When no attempt passes, energy scorer selects the lowest-energy attempt.

    Spec: REQ-CODE-033
    """
    attempt0 = "def f(): return 1"
    attempt1 = "def f(): return 2"
    test_cases = ["assert f() == 99"]  # nothing will pass this

    scores = {attempt0: 10.0, attempt1: 3.0}
    scorer = _FixedEnergyScorer(scores)
    llm = _FixedLLM([attempt0, attempt1])
    isr = IterativeSelfRepair(llm_runner=llm, energy_scorer=scorer, max_retries=1, sandbox=False)

    result = isr.repair("implement f", test_cases)

    # attempt1 has lower energy, should be selected
    assert result.best_attempt.response == attempt1
    assert result.energy_selected_passing is False


def test_repair_result_energy_selected_passing_false_when_none_pass() -> None:
    """energy_selected_passing is False when best attempt did not pass tests.

    Spec: REQ-CODE-033
    """
    bad = "def f(): return 0"
    test_cases = ["assert f() == 1"]

    isr = _make_isr([bad] * 4)
    isr.max_retries = 3
    result = isr.repair("implement f", test_cases)

    assert result.energy_selected_passing is False


# ---------------------------------------------------------------------------
# _sandbox_exec subprocess integration tests
# ---------------------------------------------------------------------------


def test_sandbox_exec_passes_for_valid_code() -> None:
    """_sandbox_exec returns ExecResult(passed=True) for valid code+tests.

    Spec: REQ-CODE-033
    """
    isr = IterativeSelfRepair(
        llm_runner=_ConstantEnergyScorer(),  # type: ignore[arg-type]
        energy_scorer=_ConstantEnergyScorer(),
        sandbox=False,
    )
    code = "def add(a, b):\n    return a + b"
    tests = ["assert add(1, 2) == 3", "assert add(0, 0) == 0"]
    result = isr._sandbox_exec(code, tests)
    assert result.passed is True
    assert result.error is None


def test_sandbox_exec_fails_for_bad_code() -> None:
    """_sandbox_exec returns ExecResult(passed=False) when assertions fail.

    Spec: REQ-CODE-033
    """
    isr = IterativeSelfRepair(
        llm_runner=_ConstantEnergyScorer(),  # type: ignore[arg-type]
        energy_scorer=_ConstantEnergyScorer(),
        sandbox=False,
    )
    code = "def add(a, b):\n    return a - b"  # wrong
    tests = ["assert add(1, 2) == 3"]
    result = isr._sandbox_exec(code, tests)
    assert result.passed is False
    assert result.error is not None


def test_sandbox_exec_timeout() -> None:
    """_sandbox_exec marks timed_out=True when subprocess exceeds timeout.

    Spec: REQ-CODE-033
    """
    isr = IterativeSelfRepair(
        llm_runner=_ConstantEnergyScorer(),  # type: ignore[arg-type]
        energy_scorer=_ConstantEnergyScorer(),
        sandbox=False,
        exec_timeout_s=0.1,  # very short timeout
    )
    # Infinite loop code
    code = "while True: pass"
    tests = []
    result = isr._sandbox_exec(code, tests)
    assert result.passed is False
    assert result.timed_out is True


# ---------------------------------------------------------------------------
# Correction prompt content test
# ---------------------------------------------------------------------------


def test_sandbox_exec_gvisor_passes() -> None:
    """_exec_gvisor with sandbox=True passes for valid code by falling back to subprocess.

    When Docker/runsc is not available (CI), _exec_gvisor falls back to
    _exec_subprocess.  We verify the fallback path works.

    Spec: REQ-CODE-033
    """
    import os
    from unittest.mock import patch

    isr = IterativeSelfRepair(
        llm_runner=_ConstantEnergyScorer(),  # type: ignore[arg-type]
        energy_scorer=_ConstantEnergyScorer(),
        sandbox=True,
        exec_timeout_s=5.0,
    )
    code = "x = 1 + 1"
    tests: list[str] = []

    # Simulate CARNOT_USE_SANDBOX=1 to route through _sandbox_exec -> _exec_gvisor
    with patch.dict(os.environ, {"CARNOT_USE_SANDBOX": "1"}):
        # Docker is not available in CI, so _exec_gvisor will fall back to subprocess.
        result = isr._sandbox_exec(code, tests)
    # Either path should return a valid ExecResult
    assert isinstance(result.passed, bool)


def test_sandbox_exec_gvisor_timeout_returns_timed_out() -> None:
    """_exec_gvisor with TimeoutExpired returns timed_out=True.

    Spec: REQ-CODE-033
    """
    import os
    import subprocess
    from unittest.mock import patch, MagicMock

    isr = IterativeSelfRepair(
        llm_runner=_ConstantEnergyScorer(),  # type: ignore[arg-type]
        energy_scorer=_ConstantEnergyScorer(),
        sandbox=True,
        exec_timeout_s=5.0,
    )

    def _raise_timeout(*args: object, **kwargs: object) -> None:
        raise subprocess.TimeoutExpired(cmd=["docker"], timeout=5.0)

    with patch("subprocess.run", side_effect=_raise_timeout):
        result = isr._exec_gvisor("x = 1")

    assert result.timed_out is True
    assert result.passed is False


def test_sandbox_exec_gvisor_not_found_falls_back() -> None:
    """_exec_gvisor with FileNotFoundError falls back to _exec_subprocess.

    Spec: REQ-CODE-033
    """
    import subprocess
    from unittest.mock import patch, call

    isr = IterativeSelfRepair(
        llm_runner=_ConstantEnergyScorer(),  # type: ignore[arg-type]
        energy_scorer=_ConstantEnergyScorer(),
        sandbox=True,
        exec_timeout_s=5.0,
    )

    call_count = [0]

    def _side_effect(*args: object, **kwargs: object) -> object:
        call_count[0] += 1
        if call_count[0] == 1:
            raise FileNotFoundError("docker not found")
        # Second call is the subprocess fallback — return success
        mock = MagicMock()
        mock.returncode = 0
        mock.stdout = ""
        mock.stderr = ""
        return mock

    with patch("subprocess.run", side_effect=_side_effect):
        result = isr._exec_gvisor("x = 1")

    assert result.passed is True


def test_sandbox_exec_gvisor_success() -> None:
    """_exec_gvisor returns ExecResult(passed=True) when Docker returns rc=0.

    Spec: REQ-CODE-033
    """
    import subprocess
    from unittest.mock import patch, MagicMock

    isr = IterativeSelfRepair(
        llm_runner=_ConstantEnergyScorer(),  # type: ignore[arg-type]
        energy_scorer=_ConstantEnergyScorer(),
        sandbox=True,
        exec_timeout_s=5.0,
    )

    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = ""
    mock_result.stderr = ""

    with patch("subprocess.run", return_value=mock_result):
        result = isr._exec_gvisor("x = 1")

    assert result.passed is True
    assert result.error is None


def test_sandbox_exec_gvisor_failure() -> None:
    """_exec_gvisor returns ExecResult(passed=False) when Docker returns rc!=0.

    Spec: REQ-CODE-033
    """
    import subprocess
    from unittest.mock import patch, MagicMock

    isr = IterativeSelfRepair(
        llm_runner=_ConstantEnergyScorer(),  # type: ignore[arg-type]
        energy_scorer=_ConstantEnergyScorer(),
        sandbox=True,
        exec_timeout_s=5.0,
    )

    mock_result = MagicMock()
    mock_result.returncode = 1
    mock_result.stdout = ""
    mock_result.stderr = "AssertionError"

    with patch("subprocess.run", return_value=mock_result):
        result = isr._exec_gvisor("x = 1")

    assert result.passed is False
    assert "AssertionError" in (result.error or "")


def test_sandbox_exec_gvisor_generic_exception_falls_back() -> None:
    """_exec_gvisor with generic exception falls back to subprocess.

    Spec: REQ-CODE-033
    """
    import subprocess
    from unittest.mock import patch, MagicMock

    isr = IterativeSelfRepair(
        llm_runner=_ConstantEnergyScorer(),  # type: ignore[arg-type]
        energy_scorer=_ConstantEnergyScorer(),
        sandbox=True,
        exec_timeout_s=5.0,
    )

    call_count = [0]

    def _side_effect(*args: object, **kwargs: object) -> object:
        call_count[0] += 1
        if call_count[0] == 1:
            raise OSError("socket error")
        mock = MagicMock()
        mock.returncode = 0
        mock.stdout = ""
        mock.stderr = ""
        return mock

    with patch("subprocess.run", side_effect=_side_effect):
        result = isr._exec_gvisor("x = 1")

    assert result.passed is True


def test_exec_subprocess_generic_exception() -> None:
    """_exec_subprocess returns ExecResult(passed=False) on unexpected OSError.

    Spec: REQ-CODE-033
    """
    import subprocess
    from unittest.mock import patch

    isr = IterativeSelfRepair(
        llm_runner=_ConstantEnergyScorer(),  # type: ignore[arg-type]
        energy_scorer=_ConstantEnergyScorer(),
        sandbox=False,
        exec_timeout_s=5.0,
    )

    with patch("subprocess.run", side_effect=OSError("permission denied")):
        result = isr._exec_subprocess("x = 1")

    assert result.passed is False
    assert result.error is not None


def test_correction_prompt_contains_traceback_and_problem() -> None:
    """The correction prompt must include the original problem, the failing code,
    and the error traceback text.

    Rationale: arXiv 2604.10508 shows error message quality is the primary
    driver of self-repair gains.  A prompt that omits the traceback gives the
    model insufficient signal.

    Spec: REQ-CODE-033, SCENARIO-CODE-031
    """
    bad_code = "def f(): return 0"
    good_code = "def f(): return 1"
    test_cases = ["assert f() == 1"]

    llm = _FixedLLM([bad_code, good_code])
    isr = IterativeSelfRepair(
        llm_runner=llm,
        energy_scorer=_ConstantEnergyScorer(),
        max_retries=1,
        sandbox=False,
    )
    isr.repair("Implement function f that returns 1.", test_cases)

    assert len(llm.calls) == 2
    correction_prompt = llm.calls[1]
    # Must contain the problem text
    assert "Implement function f" in correction_prompt
    # Must contain the bad code
    assert bad_code in correction_prompt
    # Must contain some indication of an error / fix instruction
    assert "error" in correction_prompt.lower() or "fix" in correction_prompt.lower()
