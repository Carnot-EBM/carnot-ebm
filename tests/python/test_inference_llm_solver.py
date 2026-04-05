"""Tests for LLM-powered solver and prompt construction.

Spec coverage: REQ-INFER-006, REQ-INFER-008, SCENARIO-INFER-007
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from carnot.inference.llm_solver import (
    LLMSolverConfig,
    RejectionSampleResult,
    _build_coloring_prompt,
    _build_sat_prompt,
    _generate_with_logprobs,
    logprob_rejection_sample,
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


# ---------------------------------------------------------------------------
# Tests: logprob-based rejection sampling
# ---------------------------------------------------------------------------


def _make_mock_torch_outputs(
    generated_token_ids: list[int],
    scores_logprobs: list[list[float]],
    prompt_len: int = 3,
) -> MagicMock:
    """Build a mock torch model.generate() output with scores.

    Args:
        generated_token_ids: Token IDs that were "generated".
        scores_logprobs: For each step, a list of logprobs (one per vocab token).
            The generated_token_ids[step] entry is the one that matters.
        prompt_len: Length of the prompt tokens (prepended to sequences).
    """
    import torch

    # Build sequences tensor: [batch=1, prompt_len + gen_len]
    prompt_ids = list(range(prompt_len))  # dummy prompt token IDs
    all_ids = prompt_ids + generated_token_ids
    sequences = torch.tensor([all_ids], dtype=torch.long)

    # Build score tensors: one per generation step, shape (1, vocab_size)
    # We provide raw logits; _generate_with_logprobs applies log_softmax
    score_tensors = []
    for step_logprobs in scores_logprobs:
        # Convert logprobs back to logits (log_softmax(logits) ≈ logprobs
        # if logprobs are already normalized). For simplicity, use logprobs
        # directly as logits — log_softmax will shift them but relative
        # ordering is preserved. For exact control, use unnormalized logits.
        score_tensors.append(torch.tensor([step_logprobs], dtype=torch.float32))

    outputs = MagicMock()
    outputs.sequences = sequences
    outputs.scores = tuple(score_tensors)
    return outputs


class TestGenerateWithLogprobs:
    """Tests for _generate_with_logprobs helper.

    REQ-INFER-008: per-token logprob computation from model scores.
    """

    def test_computes_mean_logprob(self) -> None:
        """REQ-INFER-008: mean logprob is average of per-token logprobs."""
        import torch

        # 2 generated tokens, vocab size 4
        # Token 0 selected at step 0, token 2 selected at step 1
        generated_ids = [0, 2]
        # Raw logits — log_softmax will normalize these
        scores_step0 = [10.0, 1.0, 1.0, 1.0]  # token 0 has highest logit
        scores_step1 = [1.0, 1.0, 10.0, 1.0]  # token 2 has highest logit
        mock_outputs = _make_mock_torch_outputs(generated_ids, [scores_step0, scores_step1])

        mock_model = MagicMock()
        mock_model.generate.return_value = mock_outputs

        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {"input_ids": torch.tensor([[0, 1, 2]])}
        mock_tokenizer.decode.return_value = "hello world"

        response, mean_lp = _generate_with_logprobs(
            mock_model, mock_tokenizer, "test prompt"
        )

        assert response == "hello world"
        # Both tokens had high logits, so mean logprob should be close to 0
        # (log_softmax of dominant logit ≈ 0)
        assert mean_lp < 0.0  # logprobs are always negative
        assert mean_lp > -1.0  # but close to 0 for confident predictions

    def test_empty_scores(self) -> None:
        """REQ-INFER-008: handles model with no scores gracefully."""
        import torch

        mock_outputs = MagicMock()
        mock_outputs.sequences = torch.tensor([[0, 1, 2, 5]])  # prompt=3, gen=1
        mock_outputs.scores = ()  # empty scores

        mock_model = MagicMock()
        mock_model.generate.return_value = mock_outputs

        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {"input_ids": torch.tensor([[0, 1, 2]])}
        mock_tokenizer.decode.return_value = "ok"

        response, mean_lp = _generate_with_logprobs(
            mock_model, mock_tokenizer, "test"
        )
        assert response == "ok"
        assert mean_lp == 0.0  # no scores → total_logprob=0, n_tokens=0, fallback 0/1

    def test_sampling_kwargs_passed(self) -> None:
        """REQ-INFER-008: do_sample and temperature are forwarded to generate."""
        import torch

        mock_outputs = _make_mock_torch_outputs([0], [[5.0, 1.0]], prompt_len=2)

        mock_model = MagicMock()
        mock_model.generate.return_value = mock_outputs

        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {"input_ids": torch.tensor([[0, 1]])}
        mock_tokenizer.decode.return_value = "sampled"

        _generate_with_logprobs(
            mock_model, mock_tokenizer, "test",
            do_sample=True, temperature=0.7,
        )

        call_kwargs = mock_model.generate.call_args[1]
        assert call_kwargs["do_sample"] is True
        assert call_kwargs["temperature"] == 0.7
        assert call_kwargs["output_scores"] is True


class TestLogprobRejectionSample:
    """Tests for logprob_rejection_sample.

    REQ-INFER-008: rejection sampling selects highest-confidence candidate.
    """

    def _make_mock_model_returning(
        self, responses: list[tuple[str, list[int], list[list[float]]]]
    ) -> tuple[MagicMock, MagicMock]:
        """Create mock model+tokenizer that return different outputs per call.

        Args:
            responses: List of (text, token_ids, scores_per_step) for each call.
        """
        import torch

        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {"input_ids": torch.tensor([[0, 1, 2]])}

        outputs_list = []
        decode_returns = []
        for text, token_ids, scores in responses:
            outputs_list.append(
                _make_mock_torch_outputs(token_ids, scores, prompt_len=3)
            )
            decode_returns.append(text)

        mock_model.generate.side_effect = outputs_list
        mock_tokenizer.decode.side_effect = decode_returns

        return mock_model, mock_tokenizer

    def test_selects_highest_logprob_candidate(self) -> None:
        """REQ-INFER-008: picks candidate with highest mean logprob."""
        # Candidate 0: low confidence (uniform-ish logits)
        # Candidate 1: high confidence (dominant logit)
        # Candidate 2: medium confidence
        mock_model, mock_tokenizer = self._make_mock_model_returning([
            ("low conf", [0], [[2.0, 2.0, 2.0, 2.0]]),      # ~uniform → low logprob
            ("high conf", [0], [[20.0, 0.0, 0.0, 0.0]]),     # dominant → high logprob
            ("med conf", [0], [[5.0, 1.0, 1.0, 1.0]]),       # moderate
        ])

        result = logprob_rejection_sample(
            _make_config(),
            prompt="test question",
            n_candidates=3,
            model=mock_model,
            tokenizer=mock_tokenizer,
        )

        assert result.best_response == "high conf"
        assert len(result.all_candidates) == 3
        # all_candidates sorted descending by logprob
        assert result.all_candidates[0][0] == "high conf"
        assert result.all_candidates[0][1] == result.mean_logprob
        # Verify descending order
        for i in range(len(result.all_candidates) - 1):
            assert result.all_candidates[i][1] >= result.all_candidates[i + 1][1]

    def test_single_candidate_degrades_gracefully(self) -> None:
        """REQ-INFER-008: n_candidates=1 → single greedy generation."""
        mock_model, mock_tokenizer = self._make_mock_model_returning([
            ("only answer", [1], [[1.0, 10.0, 1.0]]),
        ])

        result = logprob_rejection_sample(
            _make_config(),
            prompt="single test",
            n_candidates=1,
            model=mock_model,
            tokenizer=mock_tokenizer,
        )

        assert result.best_response == "only answer"
        assert len(result.all_candidates) == 1
        # With n_candidates=1, do_sample should be False (greedy)
        call_kwargs = mock_model.generate.call_args[1]
        assert "do_sample" not in call_kwargs or call_kwargs.get("do_sample") is not True

    def test_uses_sampling_for_multiple_candidates(self) -> None:
        """REQ-INFER-008: n_candidates>1 enables sampling."""
        mock_model, mock_tokenizer = self._make_mock_model_returning([
            ("a", [0], [[5.0, 1.0]]),
            ("b", [1], [[1.0, 5.0]]),
        ])

        logprob_rejection_sample(
            _make_config(),
            prompt="multi test",
            n_candidates=2,
            temperature=0.9,
            model=mock_model,
            tokenizer=mock_tokenizer,
        )

        # Both calls should have do_sample=True
        for call in mock_model.generate.call_args_list:
            assert call[1]["do_sample"] is True
            assert call[1]["temperature"] == 0.9

    def test_transformers_import_error(self) -> None:
        """REQ-INFER-008: raises ImportError when transformers not installed."""
        with patch.dict("sys.modules", {"transformers": None}):
            with pytest.raises(ImportError):
                logprob_rejection_sample(
                    _make_config(),
                    prompt="test",
                    n_candidates=3,
                    # no model/tokenizer → triggers transformers import
                )

    def test_result_dataclass_defaults(self) -> None:
        """REQ-INFER-008: RejectionSampleResult has sensible defaults."""
        r = RejectionSampleResult()
        assert r.best_response == ""
        assert r.mean_logprob == 0.0
        assert r.all_candidates == []
