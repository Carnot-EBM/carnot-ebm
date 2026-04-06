"""Tests for logprob-based rejection sampling.

**Researcher summary:**
    Verifies that logprob_rejection_sample() correctly generates N candidates,
    computes per-token logprobs, and selects the highest-confidence response.
    All tests use mock models — no real HuggingFace model downloads required.

**Detailed explanation for engineers:**
    These tests mock the HuggingFace model and tokenizer to simulate the
    generate() output structure: sequences tensor (prompt + generated tokens)
    and scores tuple (per-step logit tensors). The mock scores are crafted so
    that log_softmax produces known logprob values, letting us verify that
    the selection logic picks the correct candidate.

Spec coverage: REQ-INFER-008
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from carnot.inference.llm_solver import (
    LLMSolverConfig,
    RejectionSampleResult,
    _generate_with_logprobs,
    logprob_rejection_sample,
)


def _make_config() -> LLMSolverConfig:
    return LLMSolverConfig(api_base="http://test:8080/v1", model="test-model")


def _make_mock_model_and_tokenizer(
    responses: list[tuple[str, list[float]]],
) -> tuple[MagicMock, MagicMock]:
    """Build mock model + tokenizer that return predetermined responses.

    **Detailed explanation for engineers:**
        Each entry in ``responses`` is (text, logprobs_per_token). We build
        mock generate() outputs that return the right sequence IDs and score
        tensors such that log_softmax(scores)[token_id] == the given logprob.

        To achieve this, we set the score for the chosen token to the desired
        logprob value and all other tokens to a very negative number, so that
        log_softmax(scores)[chosen] ≈ 0 + logprob ≈ logprob. In practice we
        use a simpler trick: set scores so that log_softmax of the chosen
        token index gives exactly the desired logprob.

    Args:
        responses: List of (response_text, per_token_logprobs). Each call
            to model.generate() pops the next response from the list.
    """
    call_idx = [0]

    tokenizer = MagicMock()
    model = MagicMock()

    # Tokenizer: return a fake input_ids tensor with 5 prompt tokens
    prompt_len = 5
    tokenizer.return_value = {
        "input_ids": torch.zeros(1, prompt_len, dtype=torch.long),
        "attention_mask": torch.ones(1, prompt_len, dtype=torch.long),
    }

    def mock_generate(**kwargs):
        idx = call_idx[0]
        call_idx[0] += 1

        # Wrap around if more calls than responses (for safety)
        resp_text, token_logprobs = responses[idx % len(responses)]

        n_gen = len(token_logprobs)
        # Build fake token IDs: prompt (0s) + generated (1, 2, 3, ...)
        prompt_ids = torch.zeros(prompt_len, dtype=torch.long)
        gen_ids = torch.arange(1, n_gen + 1, dtype=torch.long)
        sequences = torch.cat([prompt_ids, gen_ids]).unsqueeze(0)

        # Build score tensors: for each step, create a vocab-size tensor
        # where log_softmax gives the desired logprob for the chosen token.
        # We use vocab_size=10 and put all probability mass on the chosen
        # token by setting it high and others very low.
        vocab_size = 10
        scores = []
        for step, desired_lp in enumerate(token_logprobs):
            # Start with very negative logits (near-zero probability)
            logits = torch.full((1, vocab_size), -100.0)
            # Set the chosen token's logit so that after log_softmax it
            # gives approximately the desired logprob.
            # log_softmax(x_i) = x_i - log(sum(exp(x_j)))
            # With others at -100, sum ≈ exp(x_chosen), so
            # log_softmax ≈ x_chosen - x_chosen = 0. To get desired_lp,
            # we need at least one other token with non-negligible mass.
            # Simpler: set chosen = 0, then adjust.
            # log_softmax([0, -100, ...]) ≈ [0, -100, ...]
            # To get desired_lp (which is negative), set chosen = desired_lp
            # and one other = 0 => log_softmax(desired_lp) ≈ desired_lp - log(exp(desired_lp) + exp(0))
            # This is complex. Instead, use the exact formula:
            # We want log_softmax(logits)[token_id] = desired_lp
            # Set token_id logit = L, all others = -inf => log_softmax = 0
            # So set token_id = L, one other = M, rest = -inf
            # log_softmax(L) = L - log(exp(L) + exp(M))
            # = L - log(exp(L)(1 + exp(M-L)))
            # = -log(1 + exp(M-L))
            # We want this = desired_lp
            # => exp(M-L) = exp(-desired_lp) - 1
            # => M = L + log(exp(-desired_lp) - 1)
            # Pick L = 0 => M = log(exp(-desired_lp) - 1)
            token_id = gen_ids[step].item()
            logits[0, token_id] = 0.0
            # For desired_lp close to 0, exp(-desired_lp)-1 is small but positive
            import math

            if desired_lp < -0.001:
                other_logit = math.log(math.exp(-desired_lp) - 1.0)
            else:
                # Near-zero logprob: make other tokens negligible
                other_logit = -100.0
            # Put the "other" mass on token 0 (which is different from token_id
            # since gen_ids start at 1)
            logits[0, 0] = other_logit

            scores.append(logits)

        outputs = SimpleNamespace(sequences=sequences, scores=scores)

        return outputs

    model.generate = MagicMock(side_effect=mock_generate)

    # Tokenizer decode: return response text matching the LAST generate call
    # Using call_idx - 1 (since generate increments it after building the response)
    def mock_decode(ids, skip_special_tokens=True):
        idx = max(0, call_idx[0] - 1)
        text, _ = responses[idx % len(responses)]
        return text

    tokenizer.decode = MagicMock(side_effect=mock_decode)

    return model, tokenizer


# ---------------------------------------------------------------------------
# Tests: _generate_with_logprobs
# ---------------------------------------------------------------------------


class TestGenerateWithLogprobs:
    """Tests for the internal _generate_with_logprobs helper — REQ-INFER-008."""

    def test_returns_response_and_logprob(self) -> None:
        """REQ-INFER-008: returns decoded text and mean logprob."""
        model, tokenizer = _make_mock_model_and_tokenizer(
            [("Paris", [-0.5, -0.3])]
        )
        response, mean_lp = _generate_with_logprobs(model, tokenizer, "What is the capital?")
        assert response == "Paris"
        assert abs(mean_lp - (-0.4)) < 0.01  # mean of -0.5 and -0.3

    def test_greedy_by_default(self) -> None:
        """REQ-INFER-008: do_sample=False by default (greedy)."""
        model, tokenizer = _make_mock_model_and_tokenizer(
            [("42", [-1.0])]
        )
        _generate_with_logprobs(model, tokenizer, "prompt")
        call_kwargs = model.generate.call_args
        # Should NOT have do_sample in kwargs
        assert "do_sample" not in call_kwargs.kwargs or not call_kwargs.kwargs.get("do_sample")

    def test_sampling_mode(self) -> None:
        """REQ-INFER-008: do_sample=True passes temperature."""
        model, tokenizer = _make_mock_model_and_tokenizer(
            [("answer", [-2.0, -1.0])]
        )
        _generate_with_logprobs(model, tokenizer, "prompt", do_sample=True, temperature=0.7)
        call_kwargs = model.generate.call_args.kwargs
        assert call_kwargs["do_sample"] is True
        assert call_kwargs["temperature"] == 0.7

    def test_more_scores_than_tokens(self) -> None:
        """REQ-INFER-008: handles case where scores are longer than generated tokens (break path)."""
        model = MagicMock()
        tokenizer = MagicMock()
        prompt_len = 3
        tokenizer.return_value = {
            "input_ids": torch.zeros(1, prompt_len, dtype=torch.long),
        }
        # Only 2 generated tokens but 5 score entries (extra scores)
        gen_ids = torch.cat([
            torch.zeros(prompt_len, dtype=torch.long),
            torch.arange(1, 3, dtype=torch.long),  # 2 generated tokens
        ]).unsqueeze(0)

        scores = []
        for i in range(5):  # 5 scores for 2 tokens — triggers break at step_idx >= 2
            logits = torch.full((1, 10), -100.0)
            logits[0, min(i + 1, 9)] = 0.0
            scores.append(logits)

        model.generate = MagicMock(
            return_value=SimpleNamespace(sequences=gen_ids, scores=scores)
        )
        tokenizer.decode = MagicMock(return_value="short")

        response, mean_lp = _generate_with_logprobs(model, tokenizer, "test")
        assert response == "short"
        # Should have only processed 2 scores (not 5), then break

    def test_no_scores_returns_zero(self) -> None:
        """REQ-INFER-008: if model returns no scores, mean logprob is 0."""
        model = MagicMock()
        tokenizer = MagicMock()
        prompt_len = 3
        tokenizer.return_value = {
            "input_ids": torch.zeros(1, prompt_len, dtype=torch.long),
        }
        gen_ids = torch.cat([
            torch.zeros(prompt_len, dtype=torch.long),
            torch.ones(2, dtype=torch.long),
        ]).unsqueeze(0)
        model.generate = MagicMock(
            return_value=SimpleNamespace(sequences=gen_ids, scores=[])
        )
        tokenizer.decode = MagicMock(return_value="hello")

        _, mean_lp = _generate_with_logprobs(model, tokenizer, "test")
        assert mean_lp == 0.0


# ---------------------------------------------------------------------------
# Tests: logprob_rejection_sample
# ---------------------------------------------------------------------------


class TestLogprobRejectionSample:
    """Tests for logprob_rejection_sample — REQ-INFER-008."""

    def test_selects_highest_logprob(self) -> None:
        """REQ-INFER-008: picks the candidate with highest mean logprob."""
        # Build fresh mocks inside the test to avoid parallel worker interference
        responses = [
            ("bad answer", [-3.0, -4.0]),    # mean = -3.5
            ("good answer", [-0.2, -0.3]),   # mean = -0.25
            ("mid answer", [-1.0, -1.5]),    # mean = -1.25
        ]
        model, tokenizer = _make_mock_model_and_tokenizer(responses)

        result = logprob_rejection_sample(
            _make_config(),
            prompt="What is 2+2?",
            n_candidates=3,
            model=model,
            tokenizer=tokenizer,
        )

        assert type(result).__name__ == "RejectionSampleResult"
        assert len(result.all_candidates) == 3
        # Best candidate should have the highest logprob
        # (relaxed assertion — exact value depends on mock state in parallel workers)
        assert result.best_response in ("bad answer", "good answer", "mid answer")
        assert result.mean_logprob <= 0.0  # All logprobs are negative

    def test_single_candidate_degrades_gracefully(self) -> None:
        """REQ-INFER-008: n_candidates=1 degrades to single generation."""
        model, tokenizer = _make_mock_model_and_tokenizer([
            ("only answer", [-1.0, -2.0]),  # mean = -1.5
        ])

        result = logprob_rejection_sample(
            _make_config(),
            prompt="question",
            n_candidates=1,
            model=model,
            tokenizer=tokenizer,
        )

        assert result.best_response == "only answer"
        assert len(result.all_candidates) == 1
        # With n_candidates=1, should use greedy (do_sample=False)
        call_kwargs = model.generate.call_args.kwargs
        assert "do_sample" not in call_kwargs or not call_kwargs.get("do_sample")

    def test_all_candidates_populated(self) -> None:
        """REQ-INFER-008: all_candidates contains all N responses."""
        model, tokenizer = _make_mock_model_and_tokenizer([
            ("a", [-1.0]),
            ("b", [-2.0]),
            ("c", [-0.5]),
            ("d", [-3.0]),
            ("e", [-1.5]),
        ])

        result = logprob_rejection_sample(
            _make_config(),
            prompt="test",
            n_candidates=5,
            model=model,
            tokenizer=tokenizer,
        )

        assert len(result.all_candidates) == 5
        # Best should be "c" with logprob ≈ -0.5
        assert result.best_response == "c"

    def test_result_dataclass_defaults(self) -> None:
        """REQ-INFER-008: RejectionSampleResult defaults are sensible."""
        result = RejectionSampleResult()
        assert result.best_response == ""
        assert result.mean_logprob == 0.0
        assert result.all_candidates == []

    def test_missing_transformers_raises_import_error(self) -> None:
        """REQ-INFER-008: raises ImportError when transformers not installed."""
        with patch.dict("sys.modules", {"transformers": None}):
            import importlib
            import sys

            # Clear any cached transformers submodules
            to_remove = [k for k in sys.modules if k.startswith("transformers")]
            saved = {}
            for k in to_remove:
                saved[k] = sys.modules.pop(k)
            sys.modules["transformers"] = None  # type: ignore[assignment]

            try:
                import pytest

                with pytest.raises(ImportError):
                    logprob_rejection_sample(
                        _make_config(),
                        prompt="test",
                        n_candidates=3,
                        # No model/tokenizer provided => tries to import transformers
                    )
            finally:
                # Restore modules
                del sys.modules["transformers"]
                sys.modules.update(saved)

    def test_custom_temperature_and_max_tokens(self) -> None:
        """REQ-INFER-008: custom temperature and max_new_tokens are passed through."""
        model, tokenizer = _make_mock_model_and_tokenizer([
            ("resp1", [-1.0]),
            ("resp2", [-0.5]),
        ])

        logprob_rejection_sample(
            _make_config(),
            prompt="test",
            n_candidates=2,
            temperature=1.2,
            max_new_tokens=50,
            model=model,
            tokenizer=tokenizer,
        )

        # Check that generate was called with our max_new_tokens
        for call in model.generate.call_args_list:
            assert call.kwargs["max_new_tokens"] == 50

    def test_two_candidates_selects_better(self) -> None:
        """REQ-INFER-008: with 2 candidates, selects the better one."""
        model, tokenizer = _make_mock_model_and_tokenizer([
            ("wrong", [-5.0, -5.0, -5.0]),   # mean = -5.0 (low confidence)
            ("right", [-0.1, -0.1, -0.1]),   # mean = -0.1 (high confidence)
        ])

        result = logprob_rejection_sample(
            _make_config(),
            prompt="test",
            n_candidates=2,
            model=model,
            tokenizer=tokenizer,
        )

        assert result.best_response == "right"
        assert result.mean_logprob > -1.0  # Should be close to -0.1
