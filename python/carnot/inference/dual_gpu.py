"""Dual-GPU helpers for paired live benchmark execution.

Spec: REQ-VERIFY-041, SCENARIO-VERIFY-042
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from time import perf_counter
from typing import Any

_MODEL_SIZE_RE = re.compile(r"(\d+(?:\.\d+)?)B", re.IGNORECASE)

DualGPUTaskFn = Callable[["DualGPUExecutionContext"], Any]
LoadModelFn = Callable[..., tuple[Any, Any]]
UnloadModelFn = Callable[[Any, Any], None]


@dataclass(frozen=True)
class DualGPUExecutionContext:
    """Loaded model context passed to each per-model benchmark task."""

    model_name: str
    model_hf_id: str
    device_assignment: str
    uses_device_map_auto: bool
    model: Any
    tokenizer: Any


@dataclass(frozen=True)
class DualGPUExecutionResult:
    """Ordered result wrapper returned by DualGPURunner.run_model_tasks()."""

    model_name: str
    model_hf_id: str
    device_assignment: str
    uses_device_map_auto: bool
    elapsed_seconds: float
    payload: Any


def estimate_model_size_billions(model_name: str) -> float | None:
    """Best-effort parameter-count parser from common HuggingFace model IDs."""
    matches = [float(match.group(1)) for match in _MODEL_SIZE_RE.finditer(model_name)]
    if not matches:
        return None
    return max(matches)


def requires_device_map_auto(model_name: str, *, threshold_b: float = 7.0) -> bool:
    """Return whether the model should be loaded with device_map='auto'."""
    size_b = estimate_model_size_billions(model_name)
    return size_b is not None and size_b >= threshold_b


class DualGPURunner:
    """Run paired benchmark tasks on dedicated GPUs or sharded sequential fallback."""

    def __init__(
        self,
        model_specs: Sequence[Mapping[str, str]],
        *,
        load_model_fn: LoadModelFn | None = None,
        unload_fn: UnloadModelFn | None = None,
        torch_module: Any | None = None,
        clock: Callable[[], float] = perf_counter,
        large_model_threshold_b: float = 7.0,
    ) -> None:
        if len(model_specs) != 2:
            raise ValueError("DualGPURunner requires exactly two model specs.")
        self.model_specs = tuple(
            {"name": str(spec["name"]), "hf_id": str(spec["hf_id"])} for spec in model_specs
        )
        if load_model_fn is None:
            from carnot.inference.model_loader import load_model

            self._load_model_fn = load_model
        else:
            self._load_model_fn = load_model_fn
        self._unload_fn = unload_fn
        if torch_module is None:
            from carnot.inference import model_loader as model_loader_module

            self._torch = model_loader_module.torch
        else:
            self._torch = torch_module
        self._clock = clock
        self._large_model_threshold_b = large_model_threshold_b

    def has_two_gpus(self) -> bool:
        """Return whether two visible CUDA devices are available."""
        torch_module = self._torch
        if torch_module is None:
            return False
        cuda = getattr(torch_module, "cuda", None)
        if cuda is None:
            return False
        return bool(cuda.is_available() and cuda.device_count() >= 2)

    def execution_mode(self) -> str:
        """Return the planned execution mode for the configured model pair."""
        if not self.has_two_gpus():
            return "unavailable"
        if any(
            requires_device_map_auto(
                spec["hf_id"],
                threshold_b=self._large_model_threshold_b,
            )
            for spec in self.model_specs
        ):
            return "sharded"
        return "parallel"

    def _load_context(self, spec: Mapping[str, str], index: int) -> DualGPUExecutionContext:
        uses_device_map_auto = requires_device_map_auto(
            spec["hf_id"],
            threshold_b=self._large_model_threshold_b,
        )
        if uses_device_map_auto:
            device = "cuda"
            device_map = "auto"
            device_assignment = "device_map:auto"
        else:
            device = f"cuda:{index}" if self.execution_mode() == "parallel" else "cuda:0"
            device_map = None
            device_assignment = device

        model, tokenizer = self._load_model_fn(
            spec["hf_id"],
            device=device,
            device_map=device_map,
        )
        if model is None or tokenizer is None:
            raise RuntimeError(f"Failed to load dual-GPU model '{spec['hf_id']}'")
        return DualGPUExecutionContext(
            model_name=spec["name"],
            model_hf_id=spec["hf_id"],
            device_assignment=device_assignment,
            uses_device_map_auto=uses_device_map_auto,
            model=model,
            tokenizer=tokenizer,
        )

    def _run_task(
        self,
        task: DualGPUTaskFn,
        context: DualGPUExecutionContext,
    ) -> DualGPUExecutionResult:
        started = self._clock()
        payload = task(context)
        return DualGPUExecutionResult(
            model_name=context.model_name,
            model_hf_id=context.model_hf_id,
            device_assignment=context.device_assignment,
            uses_device_map_auto=context.uses_device_map_auto,
            elapsed_seconds=self._clock() - started,
            payload=payload,
        )

    def _unload_context(self, context: DualGPUExecutionContext) -> None:
        if self._unload_fn is not None:
            self._unload_fn(context.model, context.tokenizer)

    def run_model_tasks(
        self,
        tasks: Mapping[str, DualGPUTaskFn],
    ) -> list[DualGPUExecutionResult]:
        """Execute one task per configured model in deterministic model order."""
        if not self.has_two_gpus():
            raise RuntimeError("DualGPURunner requires at least two CUDA devices.")

        missing = [spec["name"] for spec in self.model_specs if spec["name"] not in tasks]
        if missing:
            joined = ", ".join(missing)
            raise ValueError(f"Missing task for model(s): {joined}")

        if self.execution_mode() == "parallel":
            contexts: list[DualGPUExecutionContext] = []
            try:
                for index, spec in enumerate(self.model_specs):
                    contexts.append(self._load_context(spec, index))
                with ThreadPoolExecutor(max_workers=2) as executor:
                    futures = [
                        executor.submit(self._run_task, tasks[context.model_name], context)
                        for context in contexts
                    ]
                    return [future.result() for future in futures]
            finally:
                for context in contexts:
                    self._unload_context(context)

        results: list[DualGPUExecutionResult] = []
        for index, spec in enumerate(self.model_specs):
            context = self._load_context(spec, index)
            try:
                results.append(self._run_task(tasks[context.model_name], context))
            finally:
                self._unload_context(context)
        return results


__all__ = [
    "DualGPUExecutionContext",
    "DualGPUExecutionResult",
    "DualGPURunner",
    "estimate_model_size_billions",
    "requires_device_map_auto",
]
