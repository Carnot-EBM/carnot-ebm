"""Optional TensorRT-LLM backend for Carnot inference.

Spec: REQ-VERIFY-039, REQ-VERIFY-040,
SCENARIO-VERIFY-039, SCENARIO-VERIFY-040, SCENARIO-VERIFY-041
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Literal, cast

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]

try:
    from tensorrt_llm.llmapi import (  # type: ignore[import-not-found]
        LLM,
        BuildConfig,
        QuantAlgo,
        QuantConfig,
        SamplingParams,
    )
except ImportError:  # pragma: no cover
    BuildConfig = None
    LLM = None
    QuantAlgo = None
    QuantConfig = None
    SamplingParams = None

QuantizationMode = Literal["fp16", "int8"]
HFRunner = Callable[[list[str], int], list[str]]
HFRunnerFactory = Callable[[str], HFRunner]


def _default_engine_root() -> Path:
    configured = os.environ.get("CARNOT_TRT_ENGINE_ROOT")
    if configured:
        return Path(configured).expanduser()
    return Path.home() / ".cache" / "carnot" / "tensorrt_llm"


@dataclass(frozen=True)
class _EngineCacheRecord:
    model_name: str
    quantization: QuantizationMode
    max_batch_size: int
    max_input_len: int
    max_seq_len: int
    max_num_tokens: int
    tensor_parallel_size: int


@dataclass(frozen=True)
class TRTBackendStatus:
    """Structured backend availability and cache status."""

    available: bool
    reason: str | None
    engine_dir: Path | None
    used_cached_engine: bool
    built_engine: bool
    quantization: QuantizationMode


@dataclass(frozen=True)
class TRTLLMBenchmarkResult:
    """Deterministic HuggingFace versus TensorRT benchmark summary."""

    model_name: str
    n_questions: int
    quantization: QuantizationMode
    available: bool
    huggingface_elapsed_seconds: float
    tensorrt_elapsed_seconds: float | None
    speedup: float | None
    fallback_reason: str | None


@dataclass
class TRTLLMBackend:
    """Thin wrapper around a TensorRT-LLM runtime instance."""

    llm: Any
    model_name: str
    engine_dir: Path
    quantization: QuantizationMode = "fp16"

    def generate(self, prompt: str, max_new_tokens: int = 256) -> str:
        return self.generate_batch([prompt], max_new_tokens=max_new_tokens)[0]

    def generate_batch(
        self,
        prompts: list[str],
        max_new_tokens: int = 256,
    ) -> list[str]:
        if not prompts:
            return []
        if SamplingParams is None:
            raise RuntimeError("TensorRT-LLM SamplingParams is unavailable")

        sampling_params = SamplingParams(
            max_tokens=max_new_tokens,
            temperature=0.0,
            top_k=1,
            top_p=1.0,
        )
        outputs = list(self.llm.generate(prompts, sampling_params=sampling_params))
        return [_extract_text(output) for output in outputs]

    def shutdown(self) -> None:
        for method_name in ("shutdown", "close"):
            method = getattr(self.llm, method_name, None)
            if callable(method):
                method()
                return


def _slugify_model_name(model_name: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", model_name).strip("-")
    return slug or "model"


def _engine_dir(
    model_name: str,
    quantization: QuantizationMode,
    engine_root: Path,
) -> Path:
    return engine_root / _slugify_model_name(model_name) / quantization


def _engine_record_path(engine_dir: Path) -> Path:
    return engine_dir / "carnot_trt_engine.json"


def _read_engine_record(engine_dir: Path) -> _EngineCacheRecord | None:
    path = _engine_record_path(engine_dir)
    if not path.exists():
        return None
    payload = json.loads(path.read_text())
    return _EngineCacheRecord(
        model_name=str(payload["model_name"]),
        quantization=cast("QuantizationMode", payload["quantization"]),
        max_batch_size=int(payload["max_batch_size"]),
        max_input_len=int(payload["max_input_len"]),
        max_seq_len=int(payload["max_seq_len"]),
        max_num_tokens=int(payload["max_num_tokens"]),
        tensor_parallel_size=int(payload["tensor_parallel_size"]),
    )


def _write_engine_record(engine_dir: Path, record: _EngineCacheRecord) -> None:
    engine_dir.mkdir(parents=True, exist_ok=True)
    _engine_record_path(engine_dir).write_text(
        json.dumps(asdict(record), indent=2, sort_keys=True) + "\n"
    )


def _engine_record_matches(engine_dir: Path, expected: _EngineCacheRecord) -> bool:
    existing = _read_engine_record(engine_dir)
    if existing is None:
        return False
    if existing != expected:
        return False
    return any(path.name != "carnot_trt_engine.json" for path in engine_dir.iterdir())


def _strip_thinking_tokens(response: str) -> str:
    if "</think>" in response:
        response = response.split("</think>")[-1]
    return response.strip()


def _extract_text(output: Any) -> str:
    completions = getattr(output, "outputs", None)
    if isinstance(completions, list) and completions:
        candidate = completions[0]
        text = getattr(candidate, "text", None)
        if isinstance(text, str):
            return _strip_thinking_tokens(text)
    text = getattr(output, "text", None)
    if isinstance(text, str):
        return _strip_thinking_tokens(text)
    return _strip_thinking_tokens(str(output))


def _int8_quant_algo() -> Any:
    if QuantAlgo is None:
        return "INT8"
    return getattr(QuantAlgo, "INT8", "INT8")


def _default_hf_runner_factory(model_name: str) -> HFRunner:
    from carnot.inference.model_loader import generate, load_model

    model, tokenizer = load_model(model_name, device="cuda")

    def run(prompts: list[str], max_new_tokens: int) -> list[str]:
        return [generate(model, tokenizer, prompt, max_new_tokens) for prompt in prompts]

    return run


def load_trt_backend(
    model_name: str,
    *,
    quantization: QuantizationMode = "fp16",
    engine_root: Path | None = None,
    max_batch_size: int = 8,
    max_input_len: int = 1024,
    max_seq_len: int = 2048,
    max_num_tokens: int = 4096,
    tensor_parallel_size: int = 1,
) -> tuple[TRTLLMBackend | None, TRTBackendStatus]:
    """Load or build a TensorRT-LLM backend, returning structured status on failure."""
    if os.environ.get("CARNOT_FORCE_CPU", "0") == "1":
        return None, TRTBackendStatus(
            available=False,
            reason="CARNOT_FORCE_CPU=1 disables TensorRT-LLM",
            engine_dir=None,
            used_cached_engine=False,
            built_engine=False,
            quantization=quantization,
        )

    if torch is None or not bool(torch.cuda.is_available()):
        return None, TRTBackendStatus(
            available=False,
            reason="CUDA is unavailable for TensorRT-LLM",
            engine_dir=None,
            used_cached_engine=False,
            built_engine=False,
            quantization=quantization,
        )

    if LLM is None or BuildConfig is None or SamplingParams is None:
        return None, TRTBackendStatus(
            available=False,
            reason="tensorrt_llm is not installed",
            engine_dir=None,
            used_cached_engine=False,
            built_engine=False,
            quantization=quantization,
        )

    if quantization not in {"fp16", "int8"}:
        raise ValueError("quantization must be 'fp16' or 'int8'")

    engine_root = (engine_root or _default_engine_root()).expanduser().resolve()
    engine_dir = _engine_dir(model_name, quantization, engine_root)
    record = _EngineCacheRecord(
        model_name=model_name,
        quantization=quantization,
        max_batch_size=max_batch_size,
        max_input_len=max_input_len,
        max_seq_len=max_seq_len,
        max_num_tokens=max_num_tokens,
        tensor_parallel_size=tensor_parallel_size,
    )

    if _engine_record_matches(engine_dir, record):
        try:
            runtime = LLM(str(engine_dir), tokenizer=model_name)
            return TRTLLMBackend(
                llm=runtime,
                model_name=model_name,
                engine_dir=engine_dir,
                quantization=quantization,
            ), TRTBackendStatus(
                available=True,
                reason=None,
                engine_dir=engine_dir,
                used_cached_engine=True,
                built_engine=False,
                quantization=quantization,
            )
        except Exception:
            pass

    build_kwargs: dict[str, Any] = {
        "model": model_name,
        "tokenizer": model_name,
        "dtype": "float16",
        "tensor_parallel_size": tensor_parallel_size,
        "build_config": BuildConfig(
            max_batch_size=max_batch_size,
            max_input_len=max_input_len,
            max_seq_len=max_seq_len,
            max_num_tokens=max_num_tokens,
        ),
    }

    if quantization == "int8":
        if QuantConfig is None:
            return None, TRTBackendStatus(
                available=False,
                reason="tensorrt_llm QuantConfig is unavailable",
                engine_dir=engine_dir,
                used_cached_engine=False,
                built_engine=False,
                quantization=quantization,
            )
        int8_algo = _int8_quant_algo()
        build_kwargs["quant_config"] = QuantConfig(
            quant_algo=int8_algo,
            kv_cache_quant_algo=int8_algo,
        )

    try:
        builder = LLM(**build_kwargs)
        engine_dir.mkdir(parents=True, exist_ok=True)
        builder.save(str(engine_dir))
        _write_engine_record(engine_dir, record)
        if hasattr(builder, "shutdown") and callable(builder.shutdown):
            builder.shutdown()
        runtime = LLM(str(engine_dir), tokenizer=model_name)
        return TRTLLMBackend(
            llm=runtime,
            model_name=model_name,
            engine_dir=engine_dir,
            quantization=quantization,
        ), TRTBackendStatus(
            available=True,
            reason=None,
            engine_dir=engine_dir,
            used_cached_engine=False,
            built_engine=True,
            quantization=quantization,
        )
    except Exception as exc:
        return None, TRTBackendStatus(
            available=False,
            reason=str(exc),
            engine_dir=engine_dir,
            used_cached_engine=False,
            built_engine=False,
            quantization=quantization,
        )


def benchmark_huggingface_vs_tensorrt(
    model_name: str,
    questions: list[str],
    *,
    quantization: QuantizationMode = "fp16",
    batch_size: int = 8,
    engine_root: Path | None = None,
    max_new_tokens: int = 256,
    clock: Callable[[], float] = perf_counter,
    hf_runner_factory: HFRunnerFactory = _default_hf_runner_factory,
    trt_loader_fn: Callable[..., tuple[Any | None, TRTBackendStatus]] = load_trt_backend,
) -> TRTLLMBenchmarkResult:
    """Benchmark warm HuggingFace generation against TensorRT-LLM generation."""
    hf_runner = hf_runner_factory(model_name)
    hf_start = clock()
    hf_runner(list(questions), max_new_tokens)
    hf_elapsed = clock() - hf_start

    backend, status = trt_loader_fn(
        model_name,
        quantization=quantization,
        engine_root=engine_root,
        max_batch_size=batch_size,
    )
    if backend is None:
        return TRTLLMBenchmarkResult(
            model_name=model_name,
            n_questions=len(questions),
            quantization=quantization,
            available=False,
            huggingface_elapsed_seconds=hf_elapsed,
            tensorrt_elapsed_seconds=None,
            speedup=None,
            fallback_reason=status.reason,
        )

    trt_start = clock()
    try:
        cast("Any", backend).generate_batch(list(questions), max_new_tokens=max_new_tokens)
    finally:
        shutdown = getattr(backend, "shutdown", None)
        if callable(shutdown):
            shutdown()
    trt_elapsed = clock() - trt_start
    speedup = hf_elapsed / trt_elapsed if trt_elapsed > 0 else float("inf")
    return TRTLLMBenchmarkResult(
        model_name=model_name,
        n_questions=len(questions),
        quantization=quantization,
        available=True,
        huggingface_elapsed_seconds=hf_elapsed,
        tensorrt_elapsed_seconds=trt_elapsed,
        speedup=speedup,
        fallback_reason=None,
    )


__all__ = [
    "TRTBackendStatus",
    "TRTLLMBackend",
    "TRTLLMBenchmarkResult",
    "benchmark_huggingface_vs_tensorrt",
    "load_trt_backend",
]
