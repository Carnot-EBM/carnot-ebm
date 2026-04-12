"""Warm multi-model inference server and deterministic benchmark helpers.

Spec: REQ-VERIFY-036, REQ-VERIFY-037, REQ-VERIFY-038,
SCENARIO-VERIFY-036, SCENARIO-VERIFY-037, SCENARIO-VERIFY-038
"""

from __future__ import annotations

import gc
import threading
from collections import deque
from collections.abc import Callable, Sequence
from contextlib import nullcontext
from dataclasses import dataclass, field
from queue import Empty, Queue
from time import perf_counter
from typing import Any, cast

import carnot.inference.model_loader as model_loader_module
from carnot.inference.model_loader import (
    _model_device,
    _render_generation_prompt,
    _strip_thinking_tokens,
    generate,
    load_model,
)
from carnot.inference.tensorrt_backend import load_trt_backend

LoaderFn = Callable[[str], tuple[Any, Any]]
BatchGenerateFn = Callable[[Any, Any, list[str], int], list[str]]


@dataclass(frozen=True)
class WarmServerBenchmarkResult:
    """Deterministic cold-load versus warm-server timing summary."""

    model_name: str
    n_questions: int
    cold_elapsed_seconds: float
    warm_elapsed_seconds: float
    speedup: float


@dataclass
class _QueuedRequest:
    model_name: str
    prompts: tuple[str, ...]
    max_new_tokens: int
    done: threading.Event = field(default_factory=threading.Event)
    responses: list[str] | None = None
    error: BaseException | None = None


@dataclass
class _BatchStats:
    total_requests: int = 0
    total_batches: int = 0
    max_observed_batch_size: int = 0

    def record(self, batch_size: int) -> None:
        self.total_requests += batch_size
        self.total_batches += 1
        self.max_observed_batch_size = max(self.max_observed_batch_size, batch_size)

    def snapshot(self) -> dict[str, float | int]:
        average = self.total_requests / self.total_batches if self.total_batches > 0 else 0.0
        return {
            "total_requests": self.total_requests,
            "total_batches": self.total_batches,
            "average_batch_size": average,
            "max_observed_batch_size": self.max_observed_batch_size,
        }


def _default_batch_generate(
    model: Any,
    tokenizer: Any,
    prompts: list[str],
    max_new_tokens: int,
) -> list[str]:
    if model is tokenizer and hasattr(model, "generate_batch"):
        return cast("list[str]", model.generate_batch(prompts, max_new_tokens=max_new_tokens))
    if model is None or tokenizer is None:
        raise RuntimeError(
            "_default_batch_generate() called with model=None or tokenizer=None. "
            "Warm loading must succeed before batched generation."
        )
    if not prompts:
        return []

    rendered_prompts = [_render_generation_prompt(tokenizer, prompt) for prompt in prompts]
    device = _model_device(model)

    if (
        getattr(tokenizer, "pad_token_id", None) is None
        and getattr(tokenizer, "eos_token", None) is not None
    ):
        tokenizer.pad_token = tokenizer.eos_token

    inputs = tokenizer(
        rendered_prompts,
        return_tensors="pt",
        padding=True,
    )
    inputs = {key: value.to(device) for key, value in inputs.items()}
    input_length = int(inputs["input_ids"].shape[1])

    torch_module = getattr(model_loader_module, "torch", None)
    no_grad = torch_module.no_grad if torch_module is not None else nullcontext
    with no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    responses: list[str] = []
    for index in range(len(prompts)):
        decoded = tokenizer.decode(
            outputs[index, input_length:],
            skip_special_tokens=True,
        )
        responses.append(_strip_thinking_tokens(decoded))
    return responses


def _default_loader(model_name: str, *, batch_size: int = 8) -> tuple[Any, Any]:
    """Prefer TensorRT-LLM and otherwise request CUDA through the HF loader."""
    backend, status = load_trt_backend(
        model_name,
        max_batch_size=batch_size,
    )
    if backend is not None:
        return backend, backend
    del status
    return load_model(model_name, device="cuda")


class ModelServer:
    """Warm, queued multi-model inference server with deterministic batching."""

    def __init__(
        self,
        model_names: Sequence[str],
        *,
        batch_size: int = 8,
        loader: LoaderFn | None = None,
        batch_generate_fn: BatchGenerateFn = _default_batch_generate,
        torch_module: Any | None = None,
        clock: Callable[[], float] = perf_counter,
    ) -> None:
        if batch_size < 1 or batch_size > 16:
            raise ValueError("batch_size must be within the inclusive range 1..16")

        self.model_names = tuple(model_names)
        self.batch_size = batch_size
        self._loader = loader or (
            lambda model_name: _default_loader(model_name, batch_size=batch_size)
        )
        self._batch_generate_fn = batch_generate_fn
        self._torch = torch_module
        self._clock = clock

        self._running = False
        self._loaded_models: dict[str, tuple[Any, Any]] = {}
        self._request_queue: Queue[_QueuedRequest | None] = Queue()
        self._deferred_requests: deque[_QueuedRequest] = deque()
        self._stop_event = threading.Event()
        self._worker: threading.Thread | None = None
        self._stats = _BatchStats()
        self._state_lock = threading.Lock()

    def __enter__(self) -> ModelServer:
        return self.start()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: Any,
    ) -> None:
        del exc_type, exc, tb
        self.shutdown()

    def start(self) -> ModelServer:
        """Load configured models and start the batching worker."""
        if self._running:
            return self

        loaded_models: dict[str, tuple[Any, Any]] = {}
        for model_name in self.model_names:
            model, tokenizer = self._loader(model_name)
            if model is None or tokenizer is None:
                raise RuntimeError(f"Failed to warm-load model '{model_name}'")
            loaded_models[model_name] = (model, tokenizer)

        with self._state_lock:
            self._loaded_models = loaded_models
            self._running = True
            self._stats = _BatchStats()
            self._deferred_requests.clear()

        self._stop_event.clear()
        self._worker = threading.Thread(
            target=self._worker_loop,
            name="carnot-model-server",
            daemon=True,
        )
        self._worker.start()
        return self

    def shutdown(self) -> None:
        """Stop the worker, release warm models, and clear CUDA cache when available."""
        with self._state_lock:
            was_running = self._running
            self._running = False

        if was_running:
            self._stop_event.set()
            self._request_queue.put(None)

        worker = self._worker
        if worker is not None and worker.is_alive():
            worker.join(timeout=5.0)
        self._worker = None

        self._fail_pending_requests(RuntimeError("ModelServer shut down before request completion"))

        with self._state_lock:
            self._loaded_models.clear()
            self._deferred_requests.clear()

        gc.collect()
        self._maybe_empty_cuda_cache()

    def serves_model(self, model_name: str) -> bool:
        """Return whether this running server currently serves the requested model."""
        with self._state_lock:
            return self._running and model_name in self._loaded_models

    def generate(self, prompt: str, *, model: str, max_new_tokens: int = 256) -> str:
        """Generate a single response through the warm server."""
        return self.generate_batch([prompt], model=model, max_new_tokens=max_new_tokens)[0]

    def generate_batch(
        self,
        prompts: Sequence[str],
        *,
        model: str,
        max_new_tokens: int = 256,
    ) -> list[str]:
        """Queue prompts for batched generation and preserve caller-visible order."""
        with self._state_lock:
            running = self._running
            loaded = model in self._loaded_models

        if not running:
            raise RuntimeError("ModelServer is not running")
        if not loaded:
            raise KeyError(model)
        if not prompts:
            return []

        responses: list[str] = []
        for start in range(0, len(prompts), self.batch_size):
            request = _QueuedRequest(
                model_name=model,
                prompts=tuple(prompts[start : start + self.batch_size]),
                max_new_tokens=max_new_tokens,
            )
            self._request_queue.put(request)
            request.done.wait()
            if request.error is not None:
                raise RuntimeError(str(request.error)) from request.error
            assert request.responses is not None
            responses.extend(request.responses)
        return responses

    def health_check(self) -> dict[str, Any]:
        """Return the current running state, queue depth, batch stats, and GPU snapshot."""
        with self._state_lock:
            loaded_models = list(self._loaded_models)
            running = self._running
            deferred_depth = len(self._deferred_requests)
            batch_stats = self._stats.snapshot()

        return {
            "running": running,
            "loaded_models": loaded_models,
            "queue_depth": self._request_queue.qsize() + deferred_depth,
            "batch_stats": batch_stats,
            "gpu_memory": self._gpu_memory_snapshot(),
        }

    def _worker_loop(self) -> None:
        while True:
            if self._stop_event.is_set():
                return

            request = self._next_request()
            if request is None:
                continue

            batch_requests = [request]
            batch_prompt_count = len(request.prompts)

            while batch_prompt_count < self.batch_size:
                queued = self._next_request(non_blocking=True)
                if queued is None:
                    break
                if self._compatible_with(batch_requests[0], queued) and (
                    batch_prompt_count + len(queued.prompts) <= self.batch_size
                ):
                    batch_requests.append(queued)
                    batch_prompt_count += len(queued.prompts)
                else:
                    with self._state_lock:
                        self._deferred_requests.append(queued)

            self._execute_batch(batch_requests)

    def _next_request(self, *, non_blocking: bool = False) -> _QueuedRequest | None:
        if not non_blocking:
            with self._state_lock:
                if self._deferred_requests:
                    return self._deferred_requests.popleft()

        try:
            item = (
                self._request_queue.get_nowait()
                if non_blocking
                else self._request_queue.get(timeout=0.05)
            )
        except Empty:
            return None

        if item is None:
            self._stop_event.set()
            return None
        return item

    @staticmethod
    def _compatible_with(first: _QueuedRequest, other: _QueuedRequest) -> bool:
        return first.model_name == other.model_name and first.max_new_tokens == other.max_new_tokens

    def _execute_batch(self, requests: list[_QueuedRequest]) -> None:
        first = requests[0]
        with self._state_lock:
            model, tokenizer = self._loaded_models[first.model_name]

        prompts: list[str] = []
        for request in requests:
            prompts.extend(request.prompts)

        try:
            outputs = self._batch_generate_fn(model, tokenizer, prompts, first.max_new_tokens)
            cursor = 0
            for request in requests:
                prompt_count = len(request.prompts)
                request.responses = outputs[cursor : cursor + prompt_count]
                cursor += prompt_count
                request.done.set()

            with self._state_lock:
                self._stats.record(len(prompts))
        except BaseException as exc:
            for request in requests:
                request.error = exc
                request.done.set()

    def _fail_pending_requests(self, error: BaseException) -> None:
        with self._state_lock:
            deferred = list(self._deferred_requests)

        for request in deferred:
            request.error = error
            request.done.set()

        while True:
            try:
                queued = self._request_queue.get_nowait()
            except Empty:
                break
            if queued is None:
                continue
            queued.error = error
            queued.done.set()

    def _gpu_memory_snapshot(self) -> dict[str, int | bool]:
        cuda = getattr(self._torch, "cuda", None) if self._torch is not None else None
        cuda_available = bool(cuda is not None and cuda.is_available())
        allocated_bytes = int(cuda.memory_allocated()) if cuda_available and cuda is not None else 0
        reserved_bytes = int(cuda.memory_reserved()) if cuda_available and cuda is not None else 0
        return {
            "cuda_available": cuda_available,
            "allocated_bytes": allocated_bytes,
            "reserved_bytes": reserved_bytes,
        }

    def _maybe_empty_cuda_cache(self) -> None:
        cuda = getattr(self._torch, "cuda", None) if self._torch is not None else None
        if cuda is not None and cuda.is_available():
            cuda.empty_cache()


def benchmark_cold_load_vs_warm_server(
    model_name: str,
    questions: Sequence[str],
    *,
    batch_size: int = 8,
    load_model_fn: LoaderFn = load_model,
    generate_fn: Callable[[Any, Any, str, int], str] = generate,
    server_factory: Callable[[], ModelServer] | None = None,
    clock: Callable[[], float] = perf_counter,
    max_new_tokens: int = 256,
) -> WarmServerBenchmarkResult:
    """Compare repeated cold-load generation against a warm batching server."""
    cold_start = clock()
    for question in questions:
        model, tokenizer = load_model_fn(model_name)
        generate_fn(model, tokenizer, question, max_new_tokens)
    cold_elapsed = clock() - cold_start

    factory = server_factory or (lambda: ModelServer([model_name], batch_size=batch_size))
    warm_start = clock()
    with factory() as server:
        server.generate_batch(
            list(questions),
            model=model_name,
            max_new_tokens=max_new_tokens,
        )
    warm_elapsed = clock() - warm_start

    speedup = cold_elapsed / warm_elapsed if warm_elapsed > 0 else float("inf")
    return WarmServerBenchmarkResult(
        model_name=model_name,
        n_questions=len(questions),
        cold_elapsed_seconds=cold_elapsed,
        warm_elapsed_seconds=warm_elapsed,
        speedup=speedup,
    )


__all__ = [
    "ModelServer",
    "WarmServerBenchmarkResult",
    "benchmark_cold_load_vs_warm_server",
]
