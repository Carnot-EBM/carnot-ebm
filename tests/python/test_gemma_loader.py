"""Tests for carnot.pipeline.gemma_loader — GemmaTransformersLoader.

RETRO-028: Exp 439 scored 0% on GSM8K because llama.cpp tokenizer bug
(llama.cpp#21516) caused Gemma4 to emit infinite <unused8> tokens (token_id=14).
This test suite verifies:
- The loader uses HuggingFace transformers (REQ-LOADER-001)
- Invalid model_id raises ValueError (REQ-LOADER-001)
- is_valid_output correctly detects all-unused-token output (REQ-LOADER-002)
- generate() raises RuntimeError if called before load() (REQ-LOADER-001)
- load() and generate() delegate to AutoModelForCausalLM (REQ-LOADER-001)

Spec: REQ-LOADER-001, REQ-LOADER-002,
      SCENARIO-LOADER-001, SCENARIO-LOADER-002
"""

from __future__ import annotations

from unittest.mock import MagicMock, call, patch

import pytest

from carnot.pipeline.gemma_loader import GemmaTransformersLoader


# ---------------------------------------------------------------------------
# is_valid_output tests (SCENARIO-LOADER-001, SCENARIO-LOADER-002)
# ---------------------------------------------------------------------------


class TestIsValidOutput:
    """REQ-LOADER-002: output validation catches all-unused-token garbage."""

    def test_all_unused_returns_false(self) -> None:
        # SCENARIO-LOADER-001: bare <unused> tokens are invalid
        assert GemmaTransformersLoader.is_valid_output("<unused><unused><unused>") is False

    def test_unused_with_numbers_returns_false(self) -> None:
        # The llama.cpp#21516 bug produces <unused8> (token_id=14 in Gemma4 vocab)
        assert GemmaTransformersLoader.is_valid_output("<unused8><unused8><unused8>") is False

    def test_single_unused_token_returns_false(self) -> None:
        assert GemmaTransformersLoader.is_valid_output("<unused8>") is False

    def test_mixed_unused_numbers_returns_false(self) -> None:
        assert GemmaTransformersLoader.is_valid_output("<unused1><unused2><unused8>") is False

    def test_normal_text_returns_true(self) -> None:
        # SCENARIO-LOADER-002: valid text passes
        assert GemmaTransformersLoader.is_valid_output("The answer is 42.") is True

    def test_normal_text_with_numbers_returns_true(self) -> None:
        assert GemmaTransformersLoader.is_valid_output("Step 1: 2 + 2 = 4") is True

    def test_empty_string_returns_false(self) -> None:
        # Empty output means no answer was generated
        assert GemmaTransformersLoader.is_valid_output("") is False

    def test_whitespace_only_returns_false(self) -> None:
        assert GemmaTransformersLoader.is_valid_output("   ") is False

    def test_unused_mixed_with_real_text_returns_true(self) -> None:
        # Partial contamination: text contains unused tokens but also real content.
        # The regex only rejects strings where EVERY non-whitespace char is an unused token.
        assert GemmaTransformersLoader.is_valid_output("<unused8> The answer is 42.") is True

    def test_unused_uppercase_returns_false(self) -> None:
        # Case-insensitive match (regex uses re.IGNORECASE)
        assert GemmaTransformersLoader.is_valid_output("<UNUSED8><UNUSED8>") is False


# ---------------------------------------------------------------------------
# Constructor validation (REQ-LOADER-001)
# ---------------------------------------------------------------------------


class TestConstructorValidation:
    """The loader must reject non-Gemma model IDs at construction time."""

    def test_gemma_model_id_accepted(self) -> None:
        loader = GemmaTransformersLoader("google/gemma-4-E4B-it")
        assert loader.model_id == "google/gemma-4-E4B-it"

    def test_gemma_uppercase_accepted(self) -> None:
        loader = GemmaTransformersLoader("google/Gemma-4-E4B-it")
        assert loader.model_id == "google/Gemma-4-E4B-it"

    def test_non_gemma_raises_value_error(self) -> None:
        # Qwen, Llama, etc. must be rejected — this loader is Gemma-only
        with pytest.raises(ValueError, match="GemmaTransformersLoader only supports Gemma"):
            GemmaTransformersLoader("Qwen/Qwen3.5-0.8B")

    def test_non_gemma_llama_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="GemmaTransformersLoader only supports Gemma"):
            GemmaTransformersLoader("meta-llama/Llama-3-8B")

    def test_empty_model_id_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="GemmaTransformersLoader only supports Gemma"):
            GemmaTransformersLoader("")

    def test_default_device_is_auto(self) -> None:
        loader = GemmaTransformersLoader("google/gemma-4-E4B-it")
        assert loader.device == "auto"

    def test_device_override(self) -> None:
        loader = GemmaTransformersLoader("google/gemma-4-E4B-it", device="cpu")
        assert loader.device == "cpu"


# ---------------------------------------------------------------------------
# generate() before load() raises RuntimeError (REQ-LOADER-001)
# ---------------------------------------------------------------------------


class TestGenerateBeforeLoad:
    """generate() must raise RuntimeError if the model is not loaded."""

    def test_generate_without_load_raises(self) -> None:
        loader = GemmaTransformersLoader("google/gemma-4-E4B-it")
        with pytest.raises(RuntimeError, match="Model not loaded"):
            loader.generate("Hello")


# ---------------------------------------------------------------------------
# load() and generate() use AutoModelForCausalLM (REQ-LOADER-001)
# ---------------------------------------------------------------------------


class TestLoadUsesTransformers:
    """load() must call AutoModelForCausalLM.from_pretrained, NOT llama.cpp."""

    def test_load_calls_auto_model_from_pretrained(self) -> None:
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()

        with patch(
            "carnot.pipeline.gemma_loader.AutoModelForCausalLM",
            create=True,
        ) as mock_auto_model, patch(
            "carnot.pipeline.gemma_loader.AutoTokenizer",
            create=True,
        ) as mock_auto_tok:
            # Patch the transformers import inside the module
            mock_auto_model.from_pretrained.return_value = mock_model
            mock_auto_tok.from_pretrained.return_value = mock_tokenizer

            with patch.dict(
                "sys.modules",
                {
                    "transformers": MagicMock(
                        AutoModelForCausalLM=mock_auto_model,
                        AutoTokenizer=mock_auto_tok,
                    )
                },
            ):
                loader = GemmaTransformersLoader("google/gemma-4-E4B-it")
                loader.load()

            mock_auto_tok.from_pretrained.assert_called_once_with("google/gemma-4-E4B-it")
            mock_auto_model.from_pretrained.assert_called_once_with(
                "google/gemma-4-E4B-it",
                device_map="auto",
            )

    def test_generate_decodes_new_tokens_only(self) -> None:
        """generate() must return only the new tokens, not the prompt."""
        import torch

        mock_tokenizer = MagicMock()
        mock_model = MagicMock()

        # Simulate tokenizer returning input_ids of length 3
        prompt_ids = torch.tensor([[1, 2, 3]])
        mock_tokenizer.return_value = {"input_ids": prompt_ids}
        mock_tokenizer.__call__ = mock_tokenizer

        # Model generates 5 total tokens (3 prompt + 2 new)
        output_ids = torch.tensor([[1, 2, 3, 4, 5]])
        mock_model.generate.return_value = output_ids

        # Simulate parameters() to provide a device
        dummy_param = torch.zeros(1)
        mock_model.parameters.return_value = iter([dummy_param])

        mock_tokenizer.decode.return_value = "4 and 5"

        loader = GemmaTransformersLoader("google/gemma-4-E4B-it")
        loader._model = mock_model
        loader._tokenizer = mock_tokenizer

        result = loader.generate("What is 2+2?", max_new_tokens=128)

        assert result == "4 and 5"
        # decode was called with only the 2 new token ids
        decoded_call = mock_tokenizer.decode.call_args
        new_ids = decoded_call[0][0]
        assert list(new_ids.numpy()) == [4, 5]
