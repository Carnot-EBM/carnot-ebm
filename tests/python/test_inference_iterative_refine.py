"""Tests for EBM-guided iterative code refinement.

Spec coverage: REQ-INFER-013
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from carnot.inference.llm_solver import (
    RefinementResult,
    _build_code_violation_feedback,
    iterative_refine_code,
    LLMSolverConfig,
)


def _make_config() -> LLMSolverConfig:
    return LLMSolverConfig(api_base="http://test:8080/v1", model="test")


# ---------------------------------------------------------------------------
# Tests: RefinementResult
# ---------------------------------------------------------------------------


class TestRefinementResult:
    """Tests for the result dataclass."""

    def test_defaults(self) -> None:
        """REQ-INFER-013: sensible defaults."""
        r = RefinementResult()
        assert r.iterations == 0
        assert r.final_verified is False
        assert r.energy_trajectory == []
        assert r.final_code == ""


# ---------------------------------------------------------------------------
# Tests: _build_code_violation_feedback
# ---------------------------------------------------------------------------


class TestBuildViolationFeedback:
    """Tests for feedback prompt construction."""

    def test_no_failures_empty(self) -> None:
        """REQ-INFER-013: no failures → empty feedback."""
        results = [{"passed": True, "input": (1,), "expected": 1, "actual": 1, "error": None}]
        assert _build_code_violation_feedback(results) == ""

    def test_failures_listed(self) -> None:
        """REQ-INFER-013: failures are listed with details."""
        results = [
            {"passed": False, "input": (2, 3), "expected": 5, "actual": 6, "error": None},
            {"passed": True, "input": (1, 1), "expected": 2, "actual": 2, "error": None},
        ]
        feedback = _build_code_violation_feedback(results)
        assert "failed 1 test" in feedback
        assert "expected 5, got 6" in feedback

    def test_error_shown(self) -> None:
        """REQ-INFER-013: exceptions are shown in feedback."""
        results = [
            {
                "passed": False,
                "input": (0,),
                "expected": 1,
                "actual": None,
                "error": "ZeroDivisionError",
            },
        ]
        feedback = _build_code_violation_feedback(results)
        assert "ZeroDivisionError" in feedback


# ---------------------------------------------------------------------------
# Tests: iterative_refine_code
# ---------------------------------------------------------------------------


class TestIterativeRefineCode:
    """Tests for the refinement loop."""

    @patch("openai.OpenAI")
    def test_passes_first_try(self, mock_cls: MagicMock) -> None:
        """REQ-INFER-013: correct code on first attempt → 1 iteration."""
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="def add(a, b):\n    return a + b"))]
        )

        result = iterative_refine_code(
            _make_config(),
            "Write a function add(a, b) that returns a + b",
            "add",
            [((1, 2), 3), ((0, 0), 0)],
            expected_type=int,
        )
        assert result.final_verified
        assert result.iterations == 1
        assert result.energy_trajectory == [0.0]

    @patch("openai.OpenAI")
    def test_fixes_on_second_try(self, mock_cls: MagicMock) -> None:
        """REQ-INFER-013: wrong first attempt, corrected after feedback."""
        mock_client = MagicMock()
        mock_cls.return_value = mock_client

        # First call: buggy code (a - b instead of a + b)
        # Second call: fixed code
        mock_client.chat.completions.create.side_effect = [
            MagicMock(
                choices=[MagicMock(message=MagicMock(content="def add(a, b):\n    return a - b"))]
            ),
            MagicMock(
                choices=[MagicMock(message=MagicMock(content="def add(a, b):\n    return a + b"))]
            ),
        ]

        result = iterative_refine_code(
            _make_config(),
            "Write add(a, b)",
            "add",
            [((2, 3), 5), ((0, 0), 0)],
            expected_type=int,
        )
        assert result.final_verified
        assert result.iterations == 2
        assert len(result.energy_trajectory) == 2
        assert result.energy_trajectory[0] > 0  # First attempt wrong
        assert result.energy_trajectory[1] == 0.0  # Second attempt correct

    @patch("openai.OpenAI")
    def test_max_iterations_reached(self, mock_cls: MagicMock) -> None:
        """REQ-INFER-013: gives up after max_iterations."""
        mock_client = MagicMock()
        mock_cls.return_value = mock_client

        # Always return wrong code
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="def add(a, b):\n    return 999"))]
        )

        result = iterative_refine_code(
            _make_config(),
            "Write add(a, b)",
            "add",
            [((1, 2), 3)],
            max_iterations=3,
        )
        assert not result.final_verified
        assert result.iterations == 3
        assert len(result.energy_trajectory) == 3

    @patch("openai.OpenAI")
    def test_api_failure_stops(self, mock_cls: MagicMock) -> None:
        """REQ-INFER-013: API failure stops the loop gracefully."""
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        mock_client.chat.completions.create.side_effect = RuntimeError("API down")

        result = iterative_refine_code(
            _make_config(),
            "Write add(a, b)",
            "add",
            [((1, 2), 3)],
        )
        assert not result.final_verified
        assert result.iterations == 1

    def test_missing_openai(self) -> None:
        """REQ-INFER-013: missing openai returns empty result."""
        with patch.dict("sys.modules", {"openai": None}):
            import importlib

            import carnot.inference.llm_solver as mod

            importlib.reload(mod)
            result = mod.iterative_refine_code(_make_config(), "test", "f", [((1,), 1)])
            assert result.iterations == 0
            importlib.reload(mod)

    @patch("openai.OpenAI")
    def test_energy_decreases_over_iterations(self, mock_cls: MagicMock) -> None:
        """REQ-INFER-013: energy should decrease as LLM incorporates feedback."""
        mock_client = MagicMock()
        mock_cls.return_value = mock_client

        # First: 2/3 tests fail. Second: 1/3 fails. Third: all pass.
        mock_client.chat.completions.create.side_effect = [
            MagicMock(
                choices=[
                    MagicMock(
                        message=MagicMock(
                            content="def f(x):\n    return 0"  # Only f(0)=0 passes
                        )
                    )
                ]
            ),
            MagicMock(
                choices=[
                    MagicMock(
                        message=MagicMock(
                            content="def f(x):\n    return x if x >= 0 else 0"  # f(-1) still wrong
                        )
                    )
                ]
            ),
            MagicMock(
                choices=[
                    MagicMock(
                        message=MagicMock(
                            content="def f(x):\n    return abs(x)"  # All pass
                        )
                    )
                ]
            ),
        ]

        result = iterative_refine_code(
            _make_config(),
            "Write f(x) returning abs(x)",
            "f",
            [((0,), 0), ((3,), 3), ((-1,), 1)],
        )
        assert result.final_verified
        assert result.iterations == 3
        # Energy should be non-increasing
        for i in range(1, len(result.energy_trajectory)):
            assert result.energy_trajectory[i] <= result.energy_trajectory[i - 1]

    @patch("openai.OpenAI")
    def test_code_block_extraction(self, mock_cls: MagicMock) -> None:
        """REQ-INFER-013: extracts code from markdown code blocks."""
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[
                MagicMock(
                    message=MagicMock(
                        content="Here's the function:\n```python\ndef add(a, b):\n    return a + b\n```"
                    )
                )
            ]
        )

        result = iterative_refine_code(_make_config(), "add", "add", [((1, 2), 3)])
        assert result.final_verified
        assert "def add" in result.final_code
