"""Tests for carnot.inference.model_loader — robust HuggingFace model loading.

Verifies that load_model and generate:
1. Skip loading when CARNOT_SKIP_LLM is set (REQ-VERIFY-001)
2. Check available memory before attempting to load (REQ-VERIFY-001)
3. Retry on OOM errors up to max_retries times (REQ-VERIFY-001)
4. Use float32 dtype on CPU by default (REQ-VERIFY-001)
5. Raise ModelLoadError when CARNOT_FORCE_LIVE=1 and load fails (REQ-VERIFY-001)
6. Return (None, None) without raising when CARNOT_FORCE_LIVE is unset (REQ-VERIFY-001)
7. Apply chat template with enable_thinking fallback (REQ-VERIFY-002)
8. Strip <think>...</think> tokens from Qwen output (REQ-VERIFY-002)
9. Fall back to raw prompt when chat template is unavailable (REQ-VERIFY-002)
10. Export load_model and generate from carnot.inference (REQ-VERIFY-001)

Spec coverage: REQ-VERIFY-001, REQ-VERIFY-002, SCENARIO-VERIFY-003
"""

from __future__ import annotations

import gc
from typing import Any
from unittest.mock import MagicMock, call, patch

import pytest

from carnot.inference.model_loader import (
    ModelLoadError,
    _MIN_FREE_RAM_BYTES,
    _available_ram_bytes,
    _check_memory,
    generate,
    load_model,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def clear_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reset all CARNOT_* env vars before each test to avoid cross-test pollution.

    Ensures that tests that set CARNOT_FORCE_LIVE or CARNOT_SKIP_LLM do not
    bleed into the next test.
    """
    monkeypatch.delenv("CARNOT_FORCE_LIVE", raising=False)
    monkeypatch.delenv("CARNOT_SKIP_LLM", raising=False)
    monkeypatch.delenv("CARNOT_FORCE_CPU", raising=False)


def _make_fake_model(device_str: str = "cpu") -> MagicMock:
    """Build a minimal mock that quacks like a HuggingFace model.

    The mock's parameters() method returns a single tensor whose .device
    attribute matches device_str, which is what generate() uses to detect
    where to place input tensors.
    """
    import torch

    fake_model = MagicMock()
    fake_param = MagicMock()
    fake_param.device = torch.device(device_str)
    fake_model.parameters.return_value = iter([fake_param])
    # model.generate() returns a 2-D tensor: [batch, input_len + new_tokens].
    # We simulate: input was 3 tokens, model generated 2 new tokens.
    fake_model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5]])
    return fake_model


def _make_fake_tokenizer(
    chat_template_text: str = "formatted_prompt",
    enable_thinking_support: bool = True,
    decoded_output: str = "Hello world",
) -> MagicMock:
    """Build a minimal mock that quacks like a HuggingFace tokenizer.

    Args:
        chat_template_text: The string returned by apply_chat_template.
        enable_thinking_support: If False, apply_chat_template raises TypeError
            when called with enable_thinking kwarg (simulates older tokenizer).
        decoded_output: String returned by tokenizer.decode().
    """
    import torch

    fake_tok = MagicMock()
    fake_tok.eos_token_id = 2

    if enable_thinking_support:
        fake_tok.apply_chat_template.return_value = chat_template_text
    else:
        # First call (with enable_thinking) raises TypeError; second call succeeds.
        fake_tok.apply_chat_template.side_effect = [
            TypeError("unexpected keyword argument 'enable_thinking'"),
            chat_template_text,
        ]

    # tokenizer("text", return_tensors="pt") → {"input_ids": tensor([[1,2,3]])}
    fake_tok.return_value = {"input_ids": torch.tensor([[1, 2, 3]])}
    fake_tok.decode.return_value = decoded_output
    return fake_tok


# ---------------------------------------------------------------------------
# Tests: load_model
# ---------------------------------------------------------------------------


class TestLoadModelSkipLLM:
    """load_model returns (None, None) immediately when CARNOT_SKIP_LLM is set.

    SCENARIO-VERIFY-003: CI environments set CARNOT_SKIP_LLM so model loading
    is skipped entirely without errors.
    """

    def test_skip_returns_none_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """REQ-VERIFY-001: CARNOT_SKIP_LLM=1 skips load and returns (None, None)."""
        monkeypatch.setenv("CARNOT_SKIP_LLM", "1")
        model, tokenizer = load_model("Qwen/Qwen3.5-0.8B")
        assert model is None
        assert tokenizer is None

    def test_skip_never_calls_from_pretrained(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """REQ-VERIFY-001: CARNOT_SKIP_LLM set means from_pretrained is never called."""
        monkeypatch.setenv("CARNOT_SKIP_LLM", "1")
        with patch("carnot.inference.model_loader.AutoModelForCausalLM") as MockModel:
            model, tok = load_model("any_model")
            MockModel.from_pretrained.assert_not_called()
        assert model is None


class TestLoadModelMemoryCheck:
    """load_model checks available memory before attempting to load.

    SCENARIO-VERIFY-003: Prevent OOM crashes in conductor subprocesses by
    refusing to start when RAM is insufficient.
    """

    def test_low_memory_returns_none_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """REQ-VERIFY-001: returns (None, None) when available RAM < minimum."""
        monkeypatch.setenv("CARNOT_FORCE_CPU", "1")
        # Simulate only 512 MiB available — well below the 2 GiB minimum.
        with patch(
            "carnot.inference.model_loader._available_ram_bytes",
            return_value=512 * 1024 ** 2,
        ):
            with patch("carnot.inference.model_loader.AutoTokenizer"), \
                 patch("carnot.inference.model_loader.AutoModelForCausalLM"):
                model, tok = load_model("Qwen/Qwen3.5-0.8B")
        assert model is None
        assert tok is None

    def test_low_memory_raises_when_force_live(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """REQ-VERIFY-001: raises ModelLoadError when CARNOT_FORCE_LIVE=1 and RAM low."""
        monkeypatch.setenv("CARNOT_FORCE_LIVE", "1")
        monkeypatch.setenv("CARNOT_FORCE_CPU", "1")
        with patch(
            "carnot.inference.model_loader._available_ram_bytes",
            return_value=512 * 1024 ** 2,
        ):
            with patch("carnot.inference.model_loader.AutoTokenizer"), \
                 patch("carnot.inference.model_loader.AutoModelForCausalLM"):
                with pytest.raises(ModelLoadError, match="Insufficient memory"):
                    load_model("Qwen/Qwen3.5-0.8B")

    def test_sufficient_memory_proceeds(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """REQ-VERIFY-001: proceeds to load when RAM is above minimum."""
        monkeypatch.setenv("CARNOT_FORCE_CPU", "1")
        fake_model = _make_fake_model()
        fake_tok = _make_fake_tokenizer()
        with patch(
            "carnot.inference.model_loader._available_ram_bytes",
            return_value=_MIN_FREE_RAM_BYTES + 1,
        ):
            with patch("carnot.inference.model_loader.AutoTokenizer") as MockTok, \
                 patch("carnot.inference.model_loader.AutoModelForCausalLM") as MockModel:
                MockTok.from_pretrained.return_value = fake_tok
                MockModel.from_pretrained.return_value = fake_model
                model, tok = load_model("Qwen/Qwen3.5-0.8B")
        assert model is fake_model
        assert tok is fake_tok


class TestLoadModelDtype:
    """load_model selects safe dtypes by default.

    float16 on CPU triggers AVX2 crashes on older kernels. The safe default
    is float32 on CPU, float16 on CUDA.
    """

    def test_default_dtype_is_float32_on_cpu(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """REQ-VERIFY-001: dtype defaults to torch.float32 on CPU."""
        import torch

        monkeypatch.setenv("CARNOT_FORCE_CPU", "1")
        fake_model = _make_fake_model()
        fake_tok = _make_fake_tokenizer()
        with patch(
            "carnot.inference.model_loader._available_ram_bytes",
            return_value=_MIN_FREE_RAM_BYTES + 1,
        ):
            with patch("carnot.inference.model_loader.AutoTokenizer") as MockTok, \
                 patch("carnot.inference.model_loader.AutoModelForCausalLM") as MockModel:
                MockTok.from_pretrained.return_value = fake_tok
                MockModel.from_pretrained.return_value = fake_model
                load_model("Qwen/Qwen3.5-0.8B")
                # Verify that from_pretrained was called with float32.
                call_kwargs = MockModel.from_pretrained.call_args[1]
                assert call_kwargs.get("torch_dtype") == torch.float32

    def test_explicit_dtype_is_passed_through(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """REQ-VERIFY-001: caller-supplied dtype overrides the default."""
        import torch

        monkeypatch.setenv("CARNOT_FORCE_CPU", "1")
        fake_model = _make_fake_model()
        fake_tok = _make_fake_tokenizer()
        with patch(
            "carnot.inference.model_loader._available_ram_bytes",
            return_value=_MIN_FREE_RAM_BYTES + 1,
        ):
            with patch("carnot.inference.model_loader.AutoTokenizer") as MockTok, \
                 patch("carnot.inference.model_loader.AutoModelForCausalLM") as MockModel:
                MockTok.from_pretrained.return_value = fake_tok
                MockModel.from_pretrained.return_value = fake_model
                load_model("Qwen/Qwen3.5-0.8B", dtype=torch.bfloat16)
                call_kwargs = MockModel.from_pretrained.call_args[1]
                assert call_kwargs.get("torch_dtype") == torch.bfloat16


class TestLoadModelRetryOnOOM:
    """load_model retries up to max_retries times on out-of-memory errors.

    SCENARIO-VERIFY-003: Conductor subprocess may have stale tensors in memory
    from a previous attempt. Retrying after gc.collect() reclaims that memory.
    """

    def test_retries_on_oom_and_succeeds(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """REQ-VERIFY-001: succeeds on second attempt after OOM on first."""
        monkeypatch.setenv("CARNOT_FORCE_CPU", "1")
        fake_model = _make_fake_model()
        fake_tok = _make_fake_tokenizer()

        # First call to from_pretrained raises OOM; second succeeds.
        call_count = {"n": 0}

        def fake_from_pretrained(*args: Any, **kwargs: Any) -> Any:
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise RuntimeError("CUDA out of memory. Tried to allocate ...")
            return fake_model

        with patch(
            "carnot.inference.model_loader._available_ram_bytes",
            return_value=_MIN_FREE_RAM_BYTES + 1,
        ):
            with patch("carnot.inference.model_loader.AutoTokenizer") as MockTok, \
                 patch("carnot.inference.model_loader.AutoModelForCausalLM") as MockModel, \
                 patch("carnot.inference.model_loader.time") as mock_time, \
                 patch("carnot.inference.model_loader.gc") as mock_gc:
                MockTok.from_pretrained.return_value = fake_tok
                MockModel.from_pretrained.side_effect = fake_from_pretrained
                model, tok = load_model("Qwen/Qwen3.5-0.8B", max_retries=3)

        assert model is fake_model
        assert tok is fake_tok
        assert call_count["n"] == 2  # Failed once, succeeded on second attempt.
        mock_gc.collect.assert_called()  # gc.collect() must have been called.
        mock_time.sleep.assert_called()  # sleep between retries.

    def test_exhausts_retries_returns_none_none(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """REQ-VERIFY-001: returns (None, None) after all retries fail with OOM."""
        monkeypatch.setenv("CARNOT_FORCE_CPU", "1")
        with patch(
            "carnot.inference.model_loader._available_ram_bytes",
            return_value=_MIN_FREE_RAM_BYTES + 1,
        ):
            with patch("carnot.inference.model_loader.AutoTokenizer") as MockTok, \
                 patch("carnot.inference.model_loader.AutoModelForCausalLM") as MockModel, \
                 patch("carnot.inference.model_loader.time"), \
                 patch("carnot.inference.model_loader.gc"):
                MockTok.from_pretrained.return_value = _make_fake_tokenizer()
                MockModel.from_pretrained.side_effect = RuntimeError(
                    "out of memory"
                )
                model, tok = load_model("Qwen/Qwen3.5-0.8B", max_retries=2)

        assert model is None
        assert tok is None

    def test_non_oom_failure_does_not_retry(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """REQ-VERIFY-001: non-OOM errors break immediately without retrying."""
        monkeypatch.setenv("CARNOT_FORCE_CPU", "1")
        call_count = {"n": 0}

        def fake_from_pretrained(*args: Any, **kwargs: Any) -> Any:
            call_count["n"] += 1
            raise OSError("model not found on HuggingFace Hub")

        with patch(
            "carnot.inference.model_loader._available_ram_bytes",
            return_value=_MIN_FREE_RAM_BYTES + 1,
        ):
            with patch("carnot.inference.model_loader.AutoTokenizer") as MockTok, \
                 patch("carnot.inference.model_loader.AutoModelForCausalLM") as MockModel, \
                 patch("carnot.inference.model_loader.time"), \
                 patch("carnot.inference.model_loader.gc"):
                MockTok.from_pretrained.return_value = _make_fake_tokenizer()
                MockModel.from_pretrained.side_effect = fake_from_pretrained
                model, tok = load_model("nonexistent/model", max_retries=3)

        assert model is None
        assert tok is None
        # Only one attempt should have been made — OSError is not retried.
        assert call_count["n"] == 1


class TestLoadModelForceLive:
    """CARNOT_FORCE_LIVE=1 causes load_model to raise instead of silent fallback.

    Benchmark scripts must not silently produce simulated results when asked
    for live inference. Setting CARNOT_FORCE_LIVE=1 converts a silent (None, None)
    return into a hard ModelLoadError that the benchmark can surface.
    """

    def test_raises_on_import_error_when_force_live(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """REQ-VERIFY-001: raises ModelLoadError when transformers unavailable + FORCE_LIVE."""
        monkeypatch.setenv("CARNOT_FORCE_LIVE", "1")
        # Simulate transformers not installed by setting module-level sentinel to None.
        with patch("carnot.inference.model_loader.AutoTokenizer", None), \
             patch("carnot.inference.model_loader.AutoModelForCausalLM", None):
            with pytest.raises(ModelLoadError, match="not installed"):
                load_model("Qwen/Qwen3.5-0.8B")

    def test_raises_on_load_failure_when_force_live(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """REQ-VERIFY-001: raises ModelLoadError when model load fails + FORCE_LIVE."""
        monkeypatch.setenv("CARNOT_FORCE_LIVE", "1")
        monkeypatch.setenv("CARNOT_FORCE_CPU", "1")
        with patch(
            "carnot.inference.model_loader._available_ram_bytes",
            return_value=_MIN_FREE_RAM_BYTES + 1,
        ):
            with patch("carnot.inference.model_loader.AutoTokenizer") as MockTok, \
                 patch("carnot.inference.model_loader.AutoModelForCausalLM") as MockModel, \
                 patch("carnot.inference.model_loader.time"), \
                 patch("carnot.inference.model_loader.gc"):
                MockTok.from_pretrained.return_value = _make_fake_tokenizer()
                MockModel.from_pretrained.side_effect = OSError("not found")
                with pytest.raises(ModelLoadError, match="Failed to load"):
                    load_model("nonexistent/model", max_retries=1)

    def test_no_raise_without_force_live(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """REQ-VERIFY-001: returns (None, None) silently without FORCE_LIVE."""
        monkeypatch.setenv("CARNOT_FORCE_CPU", "1")
        with patch(
            "carnot.inference.model_loader._available_ram_bytes",
            return_value=_MIN_FREE_RAM_BYTES + 1,
        ):
            with patch("carnot.inference.model_loader.AutoTokenizer") as MockTok, \
                 patch("carnot.inference.model_loader.AutoModelForCausalLM") as MockModel, \
                 patch("carnot.inference.model_loader.time"), \
                 patch("carnot.inference.model_loader.gc"):
                MockTok.from_pretrained.return_value = _make_fake_tokenizer()
                MockModel.from_pretrained.side_effect = OSError("not found")
                model, tok = load_model("nonexistent/model", max_retries=1)
        assert model is None
        assert tok is None

    def test_no_raise_on_import_error_without_force_live(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """REQ-VERIFY-001: returns (None, None) when torch/transformers unavailable without FORCE_LIVE."""
        # Simulate transformers not installed by setting module-level sentinels to None.
        with patch("carnot.inference.model_loader.AutoTokenizer", None), \
             patch("carnot.inference.model_loader.AutoModelForCausalLM", None):
            model, tok = load_model("Qwen/Qwen3.5-0.8B")
        assert model is None
        assert tok is None


# ---------------------------------------------------------------------------
# Tests: generate
# ---------------------------------------------------------------------------


class TestGenerateChatTemplate:
    """generate() applies chat template and handles Qwen3 enable_thinking kwarg.

    SCENARIO-VERIFY-003: Qwen3 tokenizers require enable_thinking=False to
    suppress chain-of-thought prefixes. Older tokenizers raise TypeError on
    this kwarg and need a retry without it.
    """

    def test_generate_with_enable_thinking_support(self) -> None:
        """REQ-VERIFY-002: apply_chat_template called with enable_thinking=False."""
        import torch

        fake_model = _make_fake_model()
        # Tokenizer supports enable_thinking (Qwen3).
        fake_tok = _make_fake_tokenizer(
            chat_template_text="formatted_prompt",
            enable_thinking_support=True,
            decoded_output="The answer is 42",
        )

        result = generate(fake_model, fake_tok, "What is 6*7?")

        # Verify chat template was called once with enable_thinking=False.
        fake_tok.apply_chat_template.assert_called_once_with(
            [{"role": "user", "content": "What is 6*7?"}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        assert result == "The answer is 42"

    def test_generate_falls_back_when_no_enable_thinking(self) -> None:
        """REQ-VERIFY-002: falls back to call without enable_thinking on TypeError."""
        import torch

        fake_model = _make_fake_model()
        fake_tok = _make_fake_tokenizer(
            chat_template_text="raw_formatted",
            enable_thinking_support=False,  # raises TypeError on first call
            decoded_output="42",
        )

        result = generate(fake_model, fake_tok, "What is 6*7?")

        # Two calls to apply_chat_template: first with enable_thinking, then without.
        assert fake_tok.apply_chat_template.call_count == 2
        assert result == "42"

    def test_generate_falls_back_to_raw_prompt_on_template_error(self) -> None:
        """REQ-VERIFY-002: uses raw prompt when apply_chat_template raises non-TypeError."""
        import torch

        fake_model = _make_fake_model()
        fake_tok = _make_fake_tokenizer()
        fake_tok.apply_chat_template.side_effect = Exception("jinja template error")
        fake_tok.decode.return_value = "direct response"

        result = generate(fake_model, fake_tok, "Hello")

        # The raw prompt "Hello" should have been passed to the tokenizer.
        # (We can't directly assert the exact call because the mock call chain
        # goes through __call__, but we verify generation succeeded.)
        assert result == "direct response"


class TestGenerateThinkingTokenStrip:
    """generate() strips <think>...</think> tokens from Qwen3 output.

    Qwen3 models emit chain-of-thought between <think>...</think> tags before
    the actual answer. These must be stripped before returning to callers.
    """

    def test_strips_think_tokens(self) -> None:
        """REQ-VERIFY-002: <think>...</think> prefix is stripped from output."""
        import torch

        fake_model = _make_fake_model()
        fake_tok = _make_fake_tokenizer(
            decoded_output="<think>Let me compute 6*7 = 42</think> The answer is 42.",
        )

        result = generate(fake_model, fake_tok, "What is 6*7?")

        assert result == "The answer is 42."
        assert "<think>" not in result
        assert "</think>" not in result

    def test_no_think_tokens_passes_through(self) -> None:
        """REQ-VERIFY-002: output without think tokens is returned unchanged."""
        fake_model = _make_fake_model()
        fake_tok = _make_fake_tokenizer(decoded_output="The answer is 42.")

        result = generate(fake_model, fake_tok, "What is 6*7?")

        assert result == "The answer is 42."

    def test_strips_only_after_last_end_think(self) -> None:
        """REQ-VERIFY-002: splits on last </think> when multiple blocks present."""
        fake_model = _make_fake_model()
        fake_tok = _make_fake_tokenizer(
            decoded_output="<think>Step 1</think> mid <think>Step 2</think> Final answer",
        )

        result = generate(fake_model, fake_tok, "any")

        assert result == "Final answer"


class TestGenerateNoneModel:
    """generate() raises RuntimeError when called without a loaded model."""

    def test_raises_on_none_model(self) -> None:
        """REQ-VERIFY-001: RuntimeError raised when model is None."""
        with pytest.raises(RuntimeError, match="model=None"):
            generate(None, _make_fake_tokenizer(), "prompt")

    def test_raises_on_none_tokenizer(self) -> None:
        """REQ-VERIFY-001: RuntimeError raised when tokenizer is None."""
        with pytest.raises(RuntimeError, match="tokenizer=None"):
            generate(_make_fake_model(), None, "prompt")


# ---------------------------------------------------------------------------
# Tests: public exports
# ---------------------------------------------------------------------------


class TestPublicExports:
    """load_model and generate are exported from carnot.inference."""

    def test_load_model_exported(self) -> None:
        """REQ-VERIFY-001: load_model is importable from carnot.inference."""
        from carnot.inference import load_model as _load_model

        assert callable(_load_model)

    def test_generate_exported(self) -> None:
        """REQ-VERIFY-002: generate is importable from carnot.inference."""
        from carnot.inference import generate as _generate

        assert callable(_generate)

    def test_model_load_error_exported(self) -> None:
        """REQ-VERIFY-001: ModelLoadError is importable from carnot.inference."""
        from carnot.inference import ModelLoadError as _Err

        assert issubclass(_Err, Exception)


# ---------------------------------------------------------------------------
# Tests: memory helpers (unit)
# ---------------------------------------------------------------------------


class TestMemoryHelpers:
    """_check_memory and _available_ram_bytes behave correctly."""

    def test_check_memory_passes_when_sufficient(self) -> None:
        """REQ-VERIFY-001: no exception when available RAM >= minimum."""
        with patch(
            "carnot.inference.model_loader._available_ram_bytes",
            return_value=_MIN_FREE_RAM_BYTES + 1,
        ):
            _check_memory("test-model")  # Should not raise.

    def test_check_memory_raises_when_insufficient(self) -> None:
        """REQ-VERIFY-001: ModelLoadError raised when available RAM < minimum."""
        with patch(
            "carnot.inference.model_loader._available_ram_bytes",
            return_value=_MIN_FREE_RAM_BYTES - 1,
        ):
            with pytest.raises(ModelLoadError, match="Insufficient memory"):
                _check_memory("test-model")

    def test_available_ram_returns_int(self) -> None:
        """REQ-VERIFY-001: _available_ram_bytes returns a positive integer."""
        result = _available_ram_bytes()
        assert isinstance(result, int)
        assert result > 0

    def test_available_ram_without_psutil(self) -> None:
        """REQ-VERIFY-001: returns large sentinel value when psutil not installed.

        The psutil import inside _available_ram_bytes is mocked to raise
        ImportError, simulating an environment where psutil is absent. The
        function must return 2**63 (effectively unlimited) rather than
        crashing, so the memory check is effectively skipped.
        """
        import builtins
        real_import = builtins.__import__

        def import_blocker(name: str, *args: Any, **kwargs: Any) -> Any:
            if name == "psutil":
                raise ImportError("No module named 'psutil'")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=import_blocker):
            result = _available_ram_bytes()

        assert result == 2 ** 63


# ---------------------------------------------------------------------------
# Tests: additional coverage for uncovered branches
# ---------------------------------------------------------------------------


class TestLoadModelTorchUnavailable:
    """load_model returns (None, None) when torch/transformers are patched out.

    These tests cover the runtime sentinel check (lines 226-235) which fires
    when _TORCH_AVAILABLE or _TRANSFORMERS_AVAILABLE flags are False — the
    state the module is in when torch/transformers are not installed.
    """

    def test_returns_none_none_when_torch_flag_false(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """REQ-VERIFY-001: returns (None, None) silently when _TORCH_AVAILABLE=False."""
        with patch("carnot.inference.model_loader._TORCH_AVAILABLE", False):
            model, tok = load_model("any_model")
        assert model is None
        assert tok is None

    def test_returns_none_none_when_transformers_flag_false(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """REQ-VERIFY-001: returns (None, None) silently when _TRANSFORMERS_AVAILABLE=False."""
        with patch("carnot.inference.model_loader._TRANSFORMERS_AVAILABLE", False):
            model, tok = load_model("any_model")
        assert model is None
        assert tok is None


class TestLoadModelForceCPUDisabled:
    """load_model device resolution when CARNOT_FORCE_CPU is explicitly 0.

    When CARNOT_FORCE_CPU=0, load_model respects the ``device`` argument
    and falls back to CPU if CUDA is unavailable.
    """

    def test_force_cpu_disabled_falls_back_to_cpu_when_no_cuda(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """REQ-VERIFY-001: device=cpu used when FORCE_CPU=0 and CUDA unavailable."""
        import torch

        monkeypatch.setenv("CARNOT_FORCE_CPU", "0")
        fake_model = _make_fake_model()
        fake_tok = _make_fake_tokenizer()
        with patch(
            "carnot.inference.model_loader._available_ram_bytes",
            return_value=_MIN_FREE_RAM_BYTES + 1,
        ), patch("carnot.inference.model_loader.torch") as mock_torch, \
           patch("carnot.inference.model_loader.AutoTokenizer") as MockTok, \
           patch("carnot.inference.model_loader.AutoModelForCausalLM") as MockModel:
            mock_torch.cuda.is_available.return_value = False
            mock_torch.float32 = torch.float32
            MockTok.from_pretrained.return_value = fake_tok
            MockModel.from_pretrained.return_value = fake_model
            model, tok = load_model("any_model", device="cuda")
        # CUDA unavailable → should fall back to CPU load (no .cuda() called).
        assert model is fake_model
        fake_model.cuda.assert_not_called()


class TestLoadModelCUDAPath:
    """load_model calls model.cuda() when effective_device is cuda."""

    def test_model_moved_to_cuda_when_available(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """REQ-VERIFY-001: model.cuda() is called when CUDA is available."""
        import torch

        monkeypatch.setenv("CARNOT_FORCE_CPU", "0")
        fake_model = _make_fake_model()
        # model.cuda() must return a value (chained assignment).
        cuda_model = _make_fake_model()
        fake_model.cuda.return_value = cuda_model
        fake_tok = _make_fake_tokenizer()

        with patch(
            "carnot.inference.model_loader._available_ram_bytes",
            return_value=_MIN_FREE_RAM_BYTES + 1,
        ), patch("carnot.inference.model_loader.torch") as mock_torch, \
           patch("carnot.inference.model_loader.AutoTokenizer") as MockTok, \
           patch("carnot.inference.model_loader.AutoModelForCausalLM") as MockModel:
            mock_torch.cuda.is_available.return_value = True
            mock_torch.float16 = torch.float16
            MockTok.from_pretrained.return_value = fake_tok
            MockModel.from_pretrained.return_value = fake_model
            model, tok = load_model("any_model", device="cuda")

        fake_model.cuda.assert_called_once()
        assert model is cuda_model


class TestLoadModelOOMEmptyCacheException:
    """OOM retry swallows cuda.empty_cache() exceptions gracefully.

    If torch.cuda.empty_cache() itself raises (e.g., on a machine where CUDA
    is partially initialised), the exception must be swallowed so the retry
    loop continues.
    """

    def test_empty_cache_exception_is_swallowed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """REQ-VERIFY-001: cuda.empty_cache() exception during OOM retry is ignored."""
        import torch

        monkeypatch.setenv("CARNOT_FORCE_CPU", "1")
        fake_model = _make_fake_model()
        fake_tok = _make_fake_tokenizer()
        call_count = {"n": 0}

        def oom_then_ok(*args: Any, **kwargs: Any) -> Any:
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise RuntimeError("out of memory")
            return fake_model

        with patch(
            "carnot.inference.model_loader._available_ram_bytes",
            return_value=_MIN_FREE_RAM_BYTES + 1,
        ), patch("carnot.inference.model_loader.AutoTokenizer") as MockTok, \
           patch("carnot.inference.model_loader.AutoModelForCausalLM") as MockModel, \
           patch("carnot.inference.model_loader.time"), \
           patch("carnot.inference.model_loader.gc"), \
           patch("carnot.inference.model_loader.torch") as mock_torch:
            mock_torch.float32 = torch.float32
            mock_torch.cuda.empty_cache.side_effect = RuntimeError("CUDA not init")
            MockTok.from_pretrained.return_value = fake_tok
            MockModel.from_pretrained.side_effect = oom_then_ok
            # Should NOT raise despite empty_cache() failing.
            model, tok = load_model("any_model", max_retries=3)
        assert model is fake_model


class TestGenerateEdgeCases:
    """Additional generate() coverage for edge-case branches."""

    def test_empty_parameters_falls_back_to_cpu_device(self) -> None:
        """REQ-VERIFY-002: StopIteration from model.parameters() → uses cpu device.

        Some model wrappers may have no parameters() (e.g., a mock that
        returns an empty iterator). generate() must fall back to cpu rather
        than crashing.
        """
        fake_model = MagicMock()
        fake_model.parameters.return_value = iter([])  # empty iterator → StopIteration
        fake_tok = _make_fake_tokenizer()

        result = generate(fake_model, fake_tok, "Hello")

        # Should complete without raising despite StopIteration.
        assert isinstance(result, str)

    def test_both_chat_template_calls_fail_uses_raw_prompt(self) -> None:
        """REQ-VERIFY-002: raw prompt used when both apply_chat_template calls raise.

        First call raises TypeError (enable_thinking not supported), second
        call also raises (jinja template broken). generate() must fall back
        to the raw prompt string.
        """
        fake_model = _make_fake_model()
        fake_tok = _make_fake_tokenizer()
        # Both calls raise — first TypeError, second a generic Exception.
        fake_tok.apply_chat_template.side_effect = [
            TypeError("enable_thinking not supported"),
            RuntimeError("jinja broken"),
        ]
        fake_tok.decode.return_value = "response from raw prompt"

        result = generate(fake_model, fake_tok, "My raw prompt")

        assert result == "response from raw prompt"
