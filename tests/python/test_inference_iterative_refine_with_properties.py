"""Tests for iterative_refine_with_properties — combined test cases + property tests.

Spec coverage: REQ-INFER-013
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from carnot.inference.llm_solver import (
    LLMSolverConfig,
    iterative_refine_with_properties,
)


def _make_config() -> LLMSolverConfig:
    return LLMSolverConfig(api_base="http://test:8080/v1", model="test")


def _abs_properties() -> list[dict]:
    """Properties for an abs(x) function: non-negative output, idempotent."""
    return [
        {
            "name": "non_negative",
            "gen_args": lambda rng: (rng.randint(-100, 100),),
            "check": lambda result, x: result >= 0,
        },
        {
            "name": "idempotent",
            "gen_args": lambda rng: (rng.randint(-100, 100),),
            "check": lambda result, x: result == abs(x),
        },
    ]


def _add_properties() -> list[dict]:
    """Properties for add(a, b): commutative, identity."""
    return [
        {
            "name": "commutative",
            "gen_args": lambda rng: (rng.randint(-100, 100), rng.randint(-100, 100)),
            "check": lambda result, a, b: True,  # Always passes (checked via test cases)
        },
    ]


# ---------------------------------------------------------------------------
# Tests: iterative_refine_with_properties
# ---------------------------------------------------------------------------


class TestIterativeRefineWithProperties:
    """Tests for the combined refinement loop.

    REQ-INFER-013: iterative refinement using both deterministic
    test cases and random property-based testing.
    """

    @patch("openai.OpenAI")
    def test_passes_first_try_both_checks(self, mock_cls: MagicMock) -> None:
        """REQ-INFER-013: correct code passes both test cases and properties."""
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="def myabs(x):\n    return abs(x)"))]
        )

        result = iterative_refine_with_properties(
            _make_config(),
            "Write myabs(x) returning absolute value of x",
            "myabs",
            test_cases=[((5,), 5), ((-3,), 3), ((0,), 0)],
            properties=_abs_properties(),
            property_samples=50,
        )
        assert result.final_verified
        assert result.iterations == 1
        assert result.energy_trajectory == [0.0]
        assert result.violation_trajectory == [0]

    @patch("openai.OpenAI")
    def test_test_cases_pass_but_properties_fail(self, mock_cls: MagicMock) -> None:
        """REQ-INFER-013: test cases pass but property tests catch bugs."""
        mock_client = MagicMock()
        mock_cls.return_value = mock_client

        # First attempt: returns x (passes positive test cases, fails non_negative property
        # for negative inputs). Second attempt: correct abs.
        mock_client.chat.completions.create.side_effect = [
            MagicMock(
                choices=[MagicMock(message=MagicMock(content="def myabs(x):\n    return x"))]
            ),
            MagicMock(
                choices=[MagicMock(message=MagicMock(content="def myabs(x):\n    return abs(x)"))]
            ),
        ]

        result = iterative_refine_with_properties(
            _make_config(),
            "Write myabs(x) returning absolute value",
            "myabs",
            test_cases=[((5,), 5), ((0,), 0)],  # These pass with identity too
            properties=_abs_properties(),
            property_samples=50,
        )
        assert result.iterations == 2
        assert result.final_verified
        # First iteration should have nonzero energy from property failures
        assert result.energy_trajectory[0] > 0
        assert result.energy_trajectory[1] == 0.0

    @patch("openai.OpenAI")
    def test_both_test_cases_and_properties_fail(self, mock_cls: MagicMock) -> None:
        """REQ-INFER-013: both test cases and properties fail, combined feedback sent."""
        mock_client = MagicMock()
        mock_cls.return_value = mock_client

        # Always returns 999 — fails test cases AND properties
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="def myabs(x):\n    return 999"))]
        )

        result = iterative_refine_with_properties(
            _make_config(),
            "Write myabs(x)",
            "myabs",
            test_cases=[((-1,), 1), ((0,), 0)],
            properties=_abs_properties(),
            property_samples=10,
            max_iterations=2,
        )
        assert not result.final_verified
        assert result.iterations == 2
        assert all(e > 0 for e in result.energy_trajectory)

        # Check that conversation includes combined feedback
        calls = mock_client.chat.completions.create.call_args_list
        assert len(calls) == 2
        # Second call should have feedback messages appended:
        # system, user(task), assistant(resp1), user(feedback1), assistant(resp2?), ...
        # At minimum the last user message should contain combined feedback
        second_messages = calls[1].kwargs.get("messages") or calls[1][1].get("messages", [])
        feedback_msg = second_messages[3]["content"]  # First feedback message
        # Should contain both test case and property feedback
        assert "failed" in feedback_msg.lower()
        assert "Property" in feedback_msg

    @patch("openai.OpenAI")
    def test_max_iterations_reached(self, mock_cls: MagicMock) -> None:
        """REQ-INFER-013: gives up after max_iterations."""
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="def myabs(x):\n    return -1"))]
        )

        result = iterative_refine_with_properties(
            _make_config(),
            "Write myabs(x)",
            "myabs",
            test_cases=[((1,), 1)],
            properties=_abs_properties(),
            property_samples=10,
            max_iterations=3,
        )
        assert not result.final_verified
        assert result.iterations == 3
        assert len(result.energy_trajectory) == 3

    @patch("openai.OpenAI")
    def test_api_failure_stops(self, mock_cls: MagicMock) -> None:
        """REQ-INFER-013: API failure stops loop gracefully."""
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        mock_client.chat.completions.create.side_effect = RuntimeError("API down")

        result = iterative_refine_with_properties(
            _make_config(),
            "Write myabs(x)",
            "myabs",
            test_cases=[((1,), 1)],
            properties=_abs_properties(),
        )
        assert not result.final_verified
        assert result.iterations == 1

    def test_missing_openai(self) -> None:
        """REQ-INFER-013: missing openai returns empty result."""
        with patch.dict("sys.modules", {"openai": None}):
            import importlib

            import carnot.inference.llm_solver as mod

            importlib.reload(mod)
            result = mod.iterative_refine_with_properties(
                _make_config(),
                "test",
                "f",
                test_cases=[((1,), 1)],
                properties=[],
            )
            assert result.iterations == 0
            importlib.reload(mod)

    @patch("openai.OpenAI")
    def test_energy_combines_both_sources(self, mock_cls: MagicMock) -> None:
        """REQ-INFER-013: energy reflects combined test case + property failures."""
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="def add(a, b):\n    return a + b"))]
        )

        # All test cases pass, all properties pass (trivial property)
        result = iterative_refine_with_properties(
            _make_config(),
            "Write add(a, b)",
            "add",
            test_cases=[((1, 2), 3), ((0, 0), 0)],
            properties=_add_properties(),
            property_samples=20,
        )
        assert result.final_verified
        assert result.final_energy == 0.0
        # violation_trajectory should have a single 0
        assert result.violation_trajectory == [0]

    @patch("openai.OpenAI")
    def test_code_block_extraction(self, mock_cls: MagicMock) -> None:
        """REQ-INFER-013: extracts code from markdown code blocks."""
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[
                MagicMock(
                    message=MagicMock(
                        content="Here:\n```python\ndef add(a, b):\n    return a + b\n```"
                    )
                )
            ]
        )

        result = iterative_refine_with_properties(
            _make_config(),
            "add",
            "add",
            test_cases=[((1, 2), 3)],
            properties=_add_properties(),
            property_samples=10,
        )
        assert result.final_verified
        assert "def add" in result.final_code

    @patch("openai.OpenAI")
    def test_only_property_failures_in_feedback(self, mock_cls: MagicMock) -> None:
        """REQ-INFER-013: when only properties fail, only property feedback is sent."""
        mock_client = MagicMock()
        mock_cls.return_value = mock_client

        # Returns x (not abs(x)): test case (5,)->5 passes, but non_negative property fails
        mock_client.chat.completions.create.side_effect = [
            MagicMock(
                choices=[MagicMock(message=MagicMock(content="def myabs(x):\n    return x"))]
            ),
            MagicMock(
                choices=[MagicMock(message=MagicMock(content="def myabs(x):\n    return abs(x)"))]
            ),
        ]

        result = iterative_refine_with_properties(
            _make_config(),
            "Write myabs(x)",
            "myabs",
            test_cases=[((5,), 5)],  # passes with identity
            properties=_abs_properties(),
            property_samples=50,
        )
        # Should converge on second try
        assert result.final_verified
        assert result.iterations == 2

        # Check feedback in second call contains property info
        calls = mock_client.chat.completions.create.call_args_list
        second_messages = calls[1].kwargs.get("messages") or calls[1][1].get("messages", [])
        feedback_msg = second_messages[-1]["content"]
        assert "Property" in feedback_msg or "property" in feedback_msg.lower()

    @patch("openai.OpenAI")
    def test_empty_properties_list(self, mock_cls: MagicMock) -> None:
        """REQ-INFER-013: empty properties list degrades to test-case-only refinement."""
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="def add(a, b):\n    return a + b"))]
        )

        result = iterative_refine_with_properties(
            _make_config(),
            "Write add(a, b)",
            "add",
            test_cases=[((1, 2), 3)],
            properties=[],
        )
        assert result.final_verified
        assert result.iterations == 1
