"""Tests for carnot.inference.tensorrt_backend.

Spec: REQ-VERIFY-039, REQ-VERIFY-040,
SCENARIO-VERIFY-039, SCENARIO-VERIFY-040, SCENARIO-VERIFY-041
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from carnot.inference.tensorrt_backend import (
    TRTBackendStatus,
    TRTLLMBackend,
    benchmark_huggingface_vs_tensorrt,
    load_trt_backend,
)


class _FakeClock:
    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


class _FakeBuildConfig:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs


class _FakeSamplingParams:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs


class _FakeQuantConfig:
    def __init__(
        self,
        *,
        quant_algo: str | None = None,
        kv_cache_quant_algo: str | None = None,
    ) -> None:
        self.quant_algo = quant_algo
        self.kv_cache_quant_algo = kv_cache_quant_algo


class _FakeCompletionOutput:
    def __init__(self, text: str) -> None:
        self.text = text


class _FakeRequestOutput:
    def __init__(self, text: str) -> None:
        self.outputs = [_FakeCompletionOutput(text)]


class _FakeLLMInstance:
    def __init__(self, model: str, **kwargs: Any) -> None:
        self.model = model
        self.kwargs = kwargs
        self.saved_paths: list[str] = []
        self.generate_calls: list[tuple[list[str], Any]] = []
        self.shutdown_calls = 0

    def save(self, path: str) -> None:
        output_dir = Path(path)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "engine.ok").write_text("ok")
        self.saved_paths.append(path)

    def generate(self, prompts: list[str], sampling_params: Any = None) -> list[_FakeRequestOutput]:
        self.generate_calls.append((list(prompts), sampling_params))
        return [
            _FakeRequestOutput(f"<think>hidden</think>{self.model}::{prompt}")
            for prompt in prompts
        ]

    def shutdown(self) -> None:
        self.shutdown_calls += 1


class _FakeLLMFactory:
    def __init__(self) -> None:
        self.instances: list[_FakeLLMInstance] = []
        self.build_instances: list[_FakeLLMInstance] = []
        self.runtime_instances: list[_FakeLLMInstance] = []

    def __call__(self, model: str, **kwargs: Any) -> _FakeLLMInstance:
        instance = _FakeLLMInstance(model, **kwargs)
        self.instances.append(instance)
        if Path(model).exists():
            self.runtime_instances.append(instance)
        else:
            self.build_instances.append(instance)
        return instance


@pytest.fixture(autouse=True)
def clear_force_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-VERIFY-040: availability tests isolate `CARNOT_FORCE_CPU` behavior."""
    monkeypatch.delenv("CARNOT_FORCE_CPU", raising=False)


def _patch_trt_api(
    monkeypatch: pytest.MonkeyPatch,
    *,
    llm_factory: Any,
    cuda_available: bool = True,
) -> None:
    import carnot.inference.tensorrt_backend as trt_module

    monkeypatch.setattr(trt_module, "LLM", llm_factory)
    monkeypatch.setattr(trt_module, "BuildConfig", _FakeBuildConfig)
    monkeypatch.setattr(trt_module, "SamplingParams", _FakeSamplingParams)
    monkeypatch.setattr(trt_module, "QuantConfig", _FakeQuantConfig)
    monkeypatch.setattr(
        trt_module,
        "QuantAlgo",
        SimpleNamespace(INT8="INT8"),
    )
    monkeypatch.setattr(
        trt_module,
        "torch",
        SimpleNamespace(
            cuda=SimpleNamespace(is_available=lambda: cuda_available),
        ),
    )


class TestLoadTRTBackend:
    """Tests for backend loading, cache management, and structured fallback."""

    def test_returns_structured_unavailable_status_when_tensorrt_is_missing(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """SCENARIO-VERIFY-040: missing TensorRT-LLM returns status instead of raising."""
        import carnot.inference.tensorrt_backend as trt_module

        monkeypatch.setattr(trt_module, "LLM", None)
        monkeypatch.setattr(
            trt_module,
            "torch",
            SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True)),
        )

        backend, status = load_trt_backend(
            "Qwen/Qwen3.5-0.8B",
            engine_root=tmp_path,
        )

        assert backend is None
        assert status.available is False
        assert status.reason is not None
        assert "tensorrt_llm" in status.reason

    def test_respects_force_cpu_before_attempting_cuda_backend(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """REQ-VERIFY-040: `CARNOT_FORCE_CPU=1` disables TensorRT preference."""
        _patch_trt_api(monkeypatch, llm_factory=_FakeLLMFactory())
        monkeypatch.setenv("CARNOT_FORCE_CPU", "1")

        backend, status = load_trt_backend(
            "Qwen/Qwen3.5-0.8B",
            engine_root=tmp_path,
        )

        assert backend is None
        assert status.available is False
        assert status.reason == "CARNOT_FORCE_CPU=1 disables TensorRT-LLM"

    def test_returns_structured_unavailable_status_when_cuda_is_missing(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """SCENARIO-VERIFY-040: missing CUDA returns structured fallback metadata."""
        _patch_trt_api(monkeypatch, llm_factory=_FakeLLMFactory(), cuda_available=False)

        backend, status = load_trt_backend(
            "Qwen/Qwen3.5-0.8B",
            engine_root=tmp_path,
        )

        assert backend is None
        assert status.available is False
        assert status.reason == "CUDA is unavailable for TensorRT-LLM"

    def test_builds_engine_writes_metadata_and_generates_from_runtime(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """REQ-VERIFY-039: cache miss builds and persists an engine with metadata."""
        llm_factory = _FakeLLMFactory()
        _patch_trt_api(monkeypatch, llm_factory=llm_factory)

        backend, status = load_trt_backend(
            "Qwen/Qwen3.5-0.8B",
            engine_root=tmp_path,
            quantization="int8",
            max_batch_size=4,
            max_input_len=128,
            max_seq_len=256,
            max_num_tokens=512,
        )

        assert isinstance(backend, TRTLLMBackend)
        assert status.available is True
        assert status.used_cached_engine is False
        assert status.built_engine is True
        assert status.engine_dir is not None
        assert len(llm_factory.build_instances) == 1
        assert len(llm_factory.runtime_instances) == 1
        build_instance = llm_factory.build_instances[0]
        runtime_instance = llm_factory.runtime_instances[0]
        assert isinstance(build_instance.kwargs["build_config"], _FakeBuildConfig)
        assert build_instance.kwargs["build_config"].kwargs["max_batch_size"] == 4
        assert isinstance(build_instance.kwargs["quant_config"], _FakeQuantConfig)
        assert build_instance.kwargs["quant_config"].quant_algo == "INT8"
        assert build_instance.kwargs["quant_config"].kv_cache_quant_algo == "INT8"

        metadata = json.loads((status.engine_dir / "carnot_trt_engine.json").read_text())
        assert metadata["model_name"] == "Qwen/Qwen3.5-0.8B"
        assert metadata["quantization"] == "int8"
        assert metadata["max_batch_size"] == 4

        responses = backend.generate_batch(["one", "two"], max_new_tokens=32)
        assert responses == [
            f"{runtime_instance.model}::one",
            f"{runtime_instance.model}::two",
        ]
        prompts, sampling_params = runtime_instance.generate_calls[0]
        assert prompts == ["one", "two"]
        assert isinstance(sampling_params, _FakeSamplingParams)
        assert sampling_params.kwargs["max_tokens"] == 32
        backend.shutdown()
        assert runtime_instance.shutdown_calls == 1

    def test_reuses_matching_cached_engine_without_rebuilding(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """SCENARIO-VERIFY-039: cache hit loads the existing engine directly."""
        first_factory = _FakeLLMFactory()
        _patch_trt_api(monkeypatch, llm_factory=first_factory)
        first_backend, first_status = load_trt_backend(
            "google/gemma-4-E4B-it",
            engine_root=tmp_path,
            quantization="fp16",
        )
        assert first_backend is not None
        assert first_status.engine_dir is not None
        first_backend.shutdown()

        second_factory = _FakeLLMFactory()
        _patch_trt_api(monkeypatch, llm_factory=second_factory)
        backend, status = load_trt_backend(
            "google/gemma-4-E4B-it",
            engine_root=tmp_path,
            quantization="fp16",
        )

        assert isinstance(backend, TRTLLMBackend)
        assert status.available is True
        assert status.used_cached_engine is True
        assert status.built_engine is False
        assert len(second_factory.build_instances) == 0
        assert len(second_factory.runtime_instances) == 1

    def test_captures_build_failures_without_raising(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """SCENARIO-VERIFY-040: build failures return structured fallback metadata."""

        def failing_llm_factory(model: str, **kwargs: Any) -> Any:
            del kwargs
            if not Path(model).exists():
                raise RuntimeError("builder exploded")
            return _FakeLLMInstance(model)

        _patch_trt_api(monkeypatch, llm_factory=failing_llm_factory)

        backend, status = load_trt_backend(
            "Qwen/Qwen3.5-0.8B",
            engine_root=tmp_path,
        )

        assert backend is None
        assert status.available is False
        assert status.reason == "builder exploded"

    def test_rebuilds_when_cached_engine_load_fails(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """SCENARIO-VERIFY-039: cached-load failure falls through to a rebuild."""
        warm_factory = _FakeLLMFactory()
        _patch_trt_api(monkeypatch, llm_factory=warm_factory)
        backend, status = load_trt_backend(
            "Qwen/Qwen3.5-0.8B",
            engine_root=tmp_path,
        )
        assert backend is not None
        assert status.engine_dir is not None
        backend.shutdown()

        cache_failures = {"remaining": 1}

        def flaky_cache_factory(model: str, **kwargs: Any) -> _FakeLLMInstance:
            del kwargs
            if Path(model).exists() and cache_failures["remaining"] > 0:
                cache_failures["remaining"] -= 1
                raise RuntimeError("cached engine broken")
            return _FakeLLMInstance(model)

        _patch_trt_api(monkeypatch, llm_factory=flaky_cache_factory)
        rebuilt_backend, rebuilt_status = load_trt_backend(
            "Qwen/Qwen3.5-0.8B",
            engine_root=tmp_path,
        )

        assert rebuilt_backend is not None
        assert rebuilt_status.available is True
        assert rebuilt_status.built_engine is True
        assert rebuilt_status.used_cached_engine is False

    def test_rejects_unknown_quantization_mode(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """REQ-VERIFY-039: only fp16 and int8 quantization modes are accepted."""
        _patch_trt_api(monkeypatch, llm_factory=_FakeLLMFactory())

        with pytest.raises(ValueError, match="quantization"):
            load_trt_backend(
                "Qwen/Qwen3.5-0.8B",
                engine_root=tmp_path,
                quantization="fp8",  # type: ignore[arg-type]
            )

    def test_int8_requires_quant_config_support(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """REQ-VERIFY-039: int8 mode reports unavailable when QuantConfig is missing."""
        import carnot.inference.tensorrt_backend as trt_module

        _patch_trt_api(monkeypatch, llm_factory=_FakeLLMFactory())
        monkeypatch.setattr(trt_module, "QuantConfig", None)

        backend, status = load_trt_backend(
            "Qwen/Qwen3.5-0.8B",
            engine_root=tmp_path,
            quantization="int8",
        )

        assert backend is None
        assert status.available is False
        assert status.reason == "tensorrt_llm QuantConfig is unavailable"


class TestTensorRTBackendCoveragePaths:
    """Coverage-oriented tests for helper and guard branches."""

    def test_default_engine_root_honors_environment_override(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """REQ-VERIFY-039: engine root uses the environment override when provided."""
        import carnot.inference.tensorrt_backend as trt_module

        monkeypatch.setenv("CARNOT_TRT_ENGINE_ROOT", "~/custom-trt-cache")
        assert trt_module._default_engine_root() == Path("~/custom-trt-cache").expanduser()
        monkeypatch.delenv("CARNOT_TRT_ENGINE_ROOT", raising=False)
        assert trt_module._default_engine_root().name == "tensorrt_llm"

    def test_generate_wrapper_and_empty_batch_guard_paths(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """REQ-VERIFY-039: generate() delegates and empty batches are no-ops."""
        import carnot.inference.tensorrt_backend as trt_module

        _patch_trt_api(monkeypatch, llm_factory=_FakeLLMFactory())
        backend = TRTLLMBackend(
            llm=_FakeLLMInstance(str(tmp_path / "engine")),
            model_name="Qwen/Qwen3.5-0.8B",
            engine_dir=tmp_path,
        )

        assert backend.generate_batch([]) == []
        assert backend.generate("prompt", max_new_tokens=12).endswith("::prompt")

        monkeypatch.setattr(trt_module, "SamplingParams", None)
        with pytest.raises(RuntimeError, match="SamplingParams is unavailable"):
            backend.generate_batch(["prompt"])

    def test_engine_record_match_helpers_cover_missing_and_mismatch_paths(
        self,
        tmp_path: Path,
    ) -> None:
        """REQ-VERIFY-039: cache matching rejects missing and mismatched metadata."""
        import carnot.inference.tensorrt_backend as trt_module

        engine_dir = tmp_path / "engine"
        record = trt_module._EngineCacheRecord(
            model_name="Qwen/Qwen3.5-0.8B",
            quantization="fp16",
            max_batch_size=8,
            max_input_len=1024,
            max_seq_len=2048,
            max_num_tokens=4096,
            tensor_parallel_size=1,
        )
        assert trt_module._read_engine_record(engine_dir) is None
        assert trt_module._engine_record_matches(engine_dir, record) is False

        trt_module._write_engine_record(engine_dir, record)
        assert trt_module._engine_record_matches(engine_dir, record) is False
        (engine_dir / "engine.ok").write_text("ok")
        assert trt_module._engine_record_matches(engine_dir, record) is True

        mismatched = trt_module._EngineCacheRecord(
            model_name="Qwen/Qwen3.5-0.8B",
            quantization="int8",
            max_batch_size=8,
            max_input_len=1024,
            max_seq_len=2048,
            max_num_tokens=4096,
            tensor_parallel_size=1,
        )
        assert trt_module._engine_record_matches(engine_dir, mismatched) is False

    def test_extract_text_and_quant_helpers_cover_fallback_paths(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """REQ-VERIFY-039: helper fallbacks handle text-only and string-like outputs."""
        import carnot.inference.tensorrt_backend as trt_module

        text_only = SimpleNamespace(text="<think>x</think>answer")
        assert trt_module._extract_text(text_only) == "answer"

        class _StringLike:
            def __str__(self) -> str:
                return "<think>x</think>fallback"

        assert trt_module._extract_text(_StringLike()) == "fallback"
        monkeypatch.setattr(trt_module, "QuantAlgo", None)
        assert trt_module._int8_quant_algo() == "INT8"

    def test_default_hf_runner_factory_uses_model_loader(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """REQ-VERIFY-040: the default HF benchmark runner delegates through model_loader."""
        import carnot.inference.model_loader as model_loader
        import carnot.inference.tensorrt_backend as trt_module

        calls: list[tuple[str, str, int]] = []

        def fake_load_model(model_name: str, device: str = "cpu", **kwargs: Any) -> tuple[str, str]:
            del kwargs
            return f"model::{model_name}", f"tokenizer::{device}"

        def fake_generate(model: str, tokenizer: str, prompt: str, max_new_tokens: int) -> str:
            calls.append((model, tokenizer, max_new_tokens))
            return f"hf::{prompt}"

        monkeypatch.setattr(model_loader, "load_model", fake_load_model)
        monkeypatch.setattr(model_loader, "generate", fake_generate)

        runner = trt_module._default_hf_runner_factory("google/gemma-4-E4B-it")
        results = runner(["q1", "q2"], 24)

        assert results == ["hf::q1", "hf::q2"]
        assert calls == [
            ("model::google/gemma-4-E4B-it", "tokenizer::cuda", 24),
            ("model::google/gemma-4-E4B-it", "tokenizer::cuda", 24),
        ]


class TestTensorRTBenchmark:
    """Tests for deterministic HF-vs-TRT benchmark reporting."""

    def test_reports_speedup_when_tensorrt_backend_is_available(self) -> None:
        """SCENARIO-VERIFY-041: benchmark returns reproducible HF and TRT timings."""
        clock = _FakeClock()
        questions = [f"question-{index}" for index in range(50)]

        def hf_runner_factory(model_name: str) -> Any:
            assert model_name == "Qwen/Qwen3.5-0.8B"

            def runner(prompts: list[str], max_new_tokens: int) -> list[str]:
                assert max_new_tokens == 64
                clock.advance(12.5)
                return [f"hf::{prompt}" for prompt in prompts]

            return runner

        class _FakeBackend:
            def __init__(self) -> None:
                self.calls: list[tuple[list[str], int]] = []
                self.shutdown_calls = 0

            def generate_batch(self, prompts: list[str], max_new_tokens: int = 256) -> list[str]:
                self.calls.append((list(prompts), max_new_tokens))
                clock.advance(3.125)
                return [f"trt::{prompt}" for prompt in prompts]

            def shutdown(self) -> None:
                self.shutdown_calls += 1

        fake_backend = _FakeBackend()

        def fake_trt_loader(
            model_name: str,
            **kwargs: Any,
        ) -> tuple[_FakeBackend, TRTBackendStatus]:
            del kwargs
            assert model_name == "Qwen/Qwen3.5-0.8B"
            return fake_backend, TRTBackendStatus(
                available=True,
                reason=None,
                engine_dir=Path("/tmp/fake"),
                used_cached_engine=True,
                built_engine=False,
                quantization="fp16",
            )

        result = benchmark_huggingface_vs_tensorrt(
            "Qwen/Qwen3.5-0.8B",
            questions,
            max_new_tokens=64,
            clock=clock,
            hf_runner_factory=hf_runner_factory,
            trt_loader_fn=fake_trt_loader,
        )

        assert result.n_questions == 50
        assert result.available is True
        assert result.huggingface_elapsed_seconds == pytest.approx(12.5)
        assert result.tensorrt_elapsed_seconds == pytest.approx(3.125)
        assert result.speedup == pytest.approx(4.0)
        assert result.fallback_reason is None
        assert fake_backend.calls == [(questions, 64)]
        assert fake_backend.shutdown_calls == 1

    def test_reports_unavailable_result_when_tensorrt_cannot_be_used(self) -> None:
        """REQ-VERIFY-040: benchmark degrades to an unavailable result instead of raising."""
        clock = _FakeClock()

        def hf_runner_factory(model_name: str) -> Any:
            del model_name

            def runner(prompts: list[str], max_new_tokens: int) -> list[str]:
                del max_new_tokens
                clock.advance(5.0)
                return [f"hf::{prompt}" for prompt in prompts]

            return runner

        def unavailable_trt_loader(
            model_name: str,
            **kwargs: Any,
        ) -> tuple[None, TRTBackendStatus]:
            del model_name, kwargs
            return None, TRTBackendStatus(
                available=False,
                reason="tensorrt_llm not installed",
                engine_dir=None,
                used_cached_engine=False,
                built_engine=False,
                quantization="fp16",
            )

        result = benchmark_huggingface_vs_tensorrt(
            "google/gemma-4-E4B-it",
            [f"question-{index}" for index in range(50)],
            clock=clock,
            hf_runner_factory=hf_runner_factory,
            trt_loader_fn=unavailable_trt_loader,
        )

        assert result.available is False
        assert result.huggingface_elapsed_seconds == pytest.approx(5.0)
        assert result.tensorrt_elapsed_seconds is None
        assert result.speedup is None
        assert result.fallback_reason == "tensorrt_llm not installed"
