"""Tests for LLM-powered solver and prompt construction.

Spec coverage: REQ-INFER-006, SCENARIO-INFER-007
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from carnot.inference.llm_solver import (
    LLMSolverConfig,
    _build_coloring_prompt,
    _build_sat_prompt,
    run_llm_coloring_experiment,
    run_llm_sat_experiment,
    solve_coloring_with_llm,
    solve_sat_with_llm,
)
from carnot.verify.sat import SATClause


def _make_config() -> LLMSolverConfig:
    return LLMSolverConfig(api_base="http://test:8080/v1", model="test")


# ---------------------------------------------------------------------------
# Tests: prompt construction
# ---------------------------------------------------------------------------


class TestBuildSATPrompt:
    """Tests for SAT prompt building."""

    def test_includes_variables(self) -> None:
        """REQ-INFER-006: prompt mentions variable count."""
        clauses = [SATClause([(0, False), (1, True)])]
        prompt = _build_sat_prompt(clauses, n_vars=3)
        assert "3 variables" in prompt
        assert "x1" in prompt

    def test_includes_clauses(self) -> None:
        """REQ-INFER-006: prompt includes all clauses."""
        clauses = [
            SATClause([(0, False), (1, True)]),
            SATClause([(2, False)]),
        ]
        prompt = _build_sat_prompt(clauses, n_vars=3)
        assert "Clause 1" in prompt
        assert "Clause 2" in prompt
        assert "NOT x2" in prompt  # var_idx=1 -> x2 (1-based)

    def test_format_instructions(self) -> None:
        """REQ-INFER-006: prompt asks for parseable format."""
        prompt = _build_sat_prompt([SATClause([(0, False)])], 2)
        assert "x1=True" in prompt


class TestBuildColoringPrompt:
    """Tests for coloring prompt building."""

    def test_includes_graph_info(self) -> None:
        """REQ-INFER-006: prompt includes nodes and edges."""
        edges = [(0, 1), (1, 2)]
        prompt = _build_coloring_prompt(edges, n_nodes=3, n_colors=3)
        assert "3 nodes" in prompt
        assert "(0, 1)" in prompt

    def test_includes_colors(self) -> None:
        """REQ-INFER-006: prompt mentions available colors."""
        prompt = _build_coloring_prompt([(0, 1)], 2, 3)
        assert "3 colors" in prompt


# ---------------------------------------------------------------------------
# Tests: LLM API calls
# ---------------------------------------------------------------------------


class TestSolveSATWithLLM:
    """Tests for SAT LLM solver."""

    @patch("openai.OpenAI")
    def test_calls_api_and_returns_response(self, mock_cls: MagicMock) -> None:
        """REQ-INFER-006: calls LLM API and returns text."""
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="x1=True\nx2=False"))]
        )

        result = solve_sat_with_llm(_make_config(), [SATClause([(0, False)])], n_vars=2)
        assert "x1=True" in result
        mock_client.chat.completions.create.assert_called_once()

    @patch("openai.OpenAI")
    def test_empty_response(self, mock_cls: MagicMock) -> None:
        """REQ-INFER-006: handles empty LLM response."""
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content=None))]
        )
        result = solve_sat_with_llm(_make_config(), [SATClause([(0, False)])], 1)
        assert result == ""


class TestSolveColoringWithLLM:
    """Tests for coloring LLM solver."""

    @patch("openai.OpenAI")
    def test_calls_api(self, mock_cls: MagicMock) -> None:
        """REQ-INFER-006: calls LLM API for coloring."""
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="0 1 2"))]
        )
        result = solve_coloring_with_llm(_make_config(), [(0, 1)], 3, 3)
        assert "0 1 2" in result


# ---------------------------------------------------------------------------
# Tests: full pipeline
# ---------------------------------------------------------------------------


class TestRunLLMSATExperiment:
    """Tests for end-to-end LLM SAT experiment."""

    @patch("openai.OpenAI")
    def test_full_pipeline(self, mock_cls: MagicMock) -> None:
        """SCENARIO-INFER-007: full pipeline with LLM response."""
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        # LLM returns a correct assignment
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="x1=True\nx2=True"))]
        )

        clauses = [
            SATClause([(0, False), (1, False)]),  # x1 OR x2
        ]
        result = run_llm_sat_experiment(_make_config(), clauses, n_vars=2)
        assert result.initial_verification is not None

    @patch("openai.OpenAI")
    def test_api_failure(self, mock_cls: MagicMock) -> None:
        """REQ-INFER-006: handles API failure gracefully."""
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        mock_client.chat.completions.create.side_effect = RuntimeError("down")

        result = run_llm_sat_experiment(_make_config(), [SATClause([(0, False)])], n_vars=1)
        assert result.initial_verification is None

    @patch("openai.OpenAI")
    def test_parse_failure(self, mock_cls: MagicMock) -> None:
        """REQ-INFER-006: handles unparseable LLM response."""
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="I don't know"))]
        )
        result = run_llm_sat_experiment(_make_config(), [SATClause([(0, False)])], n_vars=1)
        assert result.initial_verification is None

    def test_missing_openai(self) -> None:
        """REQ-INFER-006: handles missing openai package."""
        with patch.dict("sys.modules", {"openai": None}):
            import importlib

            import carnot.inference.llm_solver as mod

            importlib.reload(mod)
            result = mod.run_llm_sat_experiment(_make_config(), [SATClause([(0, False)])], 1)
            assert result.initial_verification is None
            importlib.reload(mod)

    @patch("openai.OpenAI")
    def test_multi_start_repair_incorrect_assignment(self, mock_cls: MagicMock) -> None:
        """REQ-INFER-009: multi-start repair runs when LLM assignment is wrong."""
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        # LLM returns an incorrect assignment (x1=False, x2=False fails x1 OR x2)
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="x1=False\nx2=False"))]
        )

        clauses = [
            SATClause([(0, False), (1, False)]),  # x1 OR x2
        ]
        result = run_llm_sat_experiment(
            _make_config(), clauses, n_vars=2, n_starts=3,
        )
        assert result.initial_verification is not None
        # Multi-start should have attempted repair
        assert result.rounded_verification is not None

    @patch("openai.OpenAI")
    def test_multi_start_correct_assignment_skips_repair(self, mock_cls: MagicMock) -> None:
        """REQ-INFER-009: multi-start skips repair when LLM is already correct."""
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        # LLM returns a correct assignment
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="x1=True\nx2=True"))]
        )

        clauses = [
            SATClause([(0, False), (1, False)]),  # x1 OR x2
        ]
        result = run_llm_sat_experiment(
            _make_config(), clauses, n_vars=2, n_starts=5,
        )
        assert result.initial_verification is not None
        assert result.initial_verification.verdict.verified
        assert result.n_repair_steps == 0


class TestRunLLMColoringExperiment:
    """Tests for end-to-end LLM coloring experiment."""

    @patch("openai.OpenAI")
    def test_full_pipeline(self, mock_cls: MagicMock) -> None:
        """SCENARIO-INFER-007: full coloring pipeline."""
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="0 1 2"))]
        )

        result = run_llm_coloring_experiment(
            _make_config(), [(0, 1), (1, 2)], n_nodes=3, n_colors=3
        )
        assert result.initial_verification is not None

    @patch("openai.OpenAI")
    def test_api_failure(self, mock_cls: MagicMock) -> None:
        """REQ-INFER-006: handles API failure."""
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        mock_client.chat.completions.create.side_effect = RuntimeError("down")

        result = run_llm_coloring_experiment(_make_config(), [(0, 1)], n_nodes=2, n_colors=2)
        assert result.initial_verification is None

    @patch("openai.OpenAI")
    def test_parse_failure(self, mock_cls: MagicMock) -> None:
        """REQ-INFER-006: handles unparseable response."""
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="no idea"))]
        )
        result = run_llm_coloring_experiment(_make_config(), [(0, 1)], n_nodes=2, n_colors=2)
        assert result.initial_verification is None

    def test_missing_openai(self) -> None:
        """REQ-INFER-006: handles missing openai."""
        with patch.dict("sys.modules", {"openai": None}):
            import importlib

            import carnot.inference.llm_solver as mod

            importlib.reload(mod)
            result = mod.run_llm_coloring_experiment(_make_config(), [(0, 1)], 2, 2)
            assert result.initial_verification is None
            importlib.reload(mod)
