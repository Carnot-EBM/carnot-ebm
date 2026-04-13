"""Tests for carnot.inference.dual_gpu.

Spec coverage: REQ-VERIFY-041, SCENARIO-VERIFY-042.
"""

from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest
from carnot.inference import (
    DualGPUExecutionResult,
    DualGPURunner,
    estimate_model_size_billions,
    requires_device_map_auto,
)

MODEL_SPECS = [
    {"name": "Qwen3.5-0.8B", "hf_id": "Qwen/Qwen3.5-0.8B"},
    {"name": "Gemma4-E4B-it", "hf_id": "google/gemma-4-E4B-it"},
]


class _FakeCuda:
    def __init__(self, *, available: bool, device_count: int) -> None:
        self._available = available
        self._device_count = device_count

    def is_available(self) -> bool:
        return self._available

    def device_count(self) -> int:
        return self._device_count


class _FakeTorch:
    def __init__(self, *, available: bool = True, device_count: int = 2) -> None:
        self.cuda = _FakeCuda(available=available, device_count=device_count)


def test_model_size_helpers_parse_small_and_large_model_ids() -> None:
    """REQ-VERIFY-041: model size heuristics detect small vs large HuggingFace IDs."""
    assert estimate_model_size_billions("Qwen/Qwen3.5-0.8B") == pytest.approx(0.8)
    assert estimate_model_size_billions("google/gemma-4-E4B-it") == pytest.approx(4.0)
    assert estimate_model_size_billions("meta-llama/Llama-3.1-8B-Instruct") == pytest.approx(8.0)
    assert estimate_model_size_billions("org/mystery-model") is None

    assert requires_device_map_auto("Qwen/Qwen3.5-0.8B") is False
    assert requires_device_map_auto("meta-llama/Llama-3.1-8B-Instruct") is True


def test_runner_requires_exactly_two_model_specs() -> None:
    """REQ-VERIFY-041: DualGPURunner is limited to the paired-model Exp 218 contract."""
    with pytest.raises(ValueError, match="exactly two model specs"):
        DualGPURunner([MODEL_SPECS[0]], torch_module=_FakeTorch())


def test_runner_requires_two_visible_cuda_devices() -> None:
    """REQ-VERIFY-041: parallel dual-GPU execution requires at least two CUDA devices."""
    runner = DualGPURunner(
        MODEL_SPECS,
        load_model_fn=lambda *args, **kwargs: (object(), object()),
        torch_module=_FakeTorch(device_count=1),
    )

    with pytest.raises(RuntimeError, match="at least two CUDA devices"):
        runner.run_model_tasks(
            {spec["name"]: (lambda context: context.model_name) for spec in MODEL_SPECS}
        )


def test_runner_reports_unavailable_without_torch_module() -> None:
    """REQ-VERIFY-041: execution mode becomes unavailable when torch is missing."""
    runner = DualGPURunner(
        MODEL_SPECS,
        load_model_fn=lambda *args, **kwargs: (object(), object()),
        torch_module=_FakeTorch(),
    )
    runner._torch = None

    assert runner.has_two_gpus() is False
    assert runner.execution_mode() == "unavailable"


def test_runner_reports_unavailable_without_cuda_attribute() -> None:
    """REQ-VERIFY-041: execution mode becomes unavailable when torch.cuda is absent."""
    runner = DualGPURunner(
        MODEL_SPECS,
        load_model_fn=lambda *args, **kwargs: (object(), object()),
        torch_module=SimpleNamespace(),
    )

    assert runner.has_two_gpus() is False
    assert runner.execution_mode() == "unavailable"


def test_parallel_runner_assigns_one_gpu_per_model_and_preserves_order() -> None:
    """REQ-VERIFY-041, SCENARIO-VERIFY-042: small paired models run on cuda:0/cuda:1."""
    load_calls: list[tuple[str, str, str | None]] = []
    unload_calls: list[str] = []
    barrier = threading.Barrier(2, timeout=1.0)

    def fake_load(
        model_name: str,
        *,
        device: str = "cuda",
        device_map: str | None = None,
    ) -> tuple[SimpleNamespace, SimpleNamespace]:
        load_calls.append((model_name, device, device_map))
        return SimpleNamespace(name=model_name), SimpleNamespace(name=model_name)

    def fake_unload(model: SimpleNamespace, tokenizer: SimpleNamespace) -> None:
        del tokenizer
        unload_calls.append(model.name)

    def make_task(expected_name: str):
        def _task(context: SimpleNamespace) -> dict[str, object]:
            assert context.model_name == expected_name
            barrier.wait()
            return {
                "device_assignment": context.device_assignment,
                "uses_device_map_auto": context.uses_device_map_auto,
            }

        return _task

    runner = DualGPURunner(
        MODEL_SPECS,
        load_model_fn=fake_load,
        unload_fn=fake_unload,
        torch_module=_FakeTorch(),
    )

    results = runner.run_model_tasks(
        {spec["name"]: make_task(spec["name"]) for spec in MODEL_SPECS}
    )

    assert runner.execution_mode() == "parallel"
    assert load_calls == [
        ("Qwen/Qwen3.5-0.8B", "cuda:0", None),
        ("google/gemma-4-E4B-it", "cuda:1", None),
    ]
    assert [result.model_name for result in results] == [spec["name"] for spec in MODEL_SPECS]
    assert [result.device_assignment for result in results] == ["cuda:0", "cuda:1"]
    assert [result.payload["device_assignment"] for result in results] == ["cuda:0", "cuda:1"]
    assert [result.payload["uses_device_map_auto"] for result in results] == [False, False]
    assert unload_calls == ["Qwen/Qwen3.5-0.8B", "google/gemma-4-E4B-it"]


def test_runner_uses_model_loader_defaults_when_dependencies_not_injected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-VERIFY-041: default runner wiring comes from carnot.inference.model_loader."""
    import carnot.inference.model_loader as model_loader_module

    load_calls: list[tuple[str, str, str | None]] = []

    def fake_load(
        model_name: str,
        *,
        device: str = "cuda",
        device_map: str | None = None,
    ) -> tuple[SimpleNamespace, SimpleNamespace]:
        load_calls.append((model_name, device, device_map))
        return SimpleNamespace(name=model_name), SimpleNamespace(name=model_name)

    monkeypatch.setattr(model_loader_module, "load_model", fake_load)
    monkeypatch.setattr(model_loader_module, "torch", _FakeTorch())

    runner = DualGPURunner(MODEL_SPECS)
    results = runner.run_model_tasks(
        {spec["name"]: (lambda context: context.model_name) for spec in MODEL_SPECS}
    )

    assert load_calls == [
        ("Qwen/Qwen3.5-0.8B", "cuda:0", None),
        ("google/gemma-4-E4B-it", "cuda:1", None),
    ]
    assert [result.payload for result in results] == ["Qwen3.5-0.8B", "Gemma4-E4B-it"]


def test_runner_uses_model_server_context_when_one_is_registered() -> None:
    """REQ-VERIFY-041: DualGPURunner reuses a warm model server when it serves the model."""

    class _FakeModelServer:
        def serves_model(self, hf_id: str) -> bool:
            return hf_id == "Qwen/Qwen3.5-0.8B"

    load_calls: list[tuple[str, str, str | None]] = []

    def fake_load(
        model_name: str,
        *,
        device: str = "cuda",
        device_map: str | None = None,
    ) -> tuple[SimpleNamespace, SimpleNamespace]:
        load_calls.append((model_name, device, device_map))
        return SimpleNamespace(name=model_name), SimpleNamespace(name=model_name)

    runner = DualGPURunner(
        MODEL_SPECS,
        load_model_fn=fake_load,
        unload_fn=lambda model, tokenizer: None,
        torch_module=_FakeTorch(),
        model_server=_FakeModelServer(),
    )

    results = runner.run_model_tasks(
        {
            "Qwen3.5-0.8B": lambda context: context.device_assignment,
            "Gemma4-E4B-it": lambda context: context.device_assignment,
        }
    )

    assert results[0].device_assignment == "model_server"
    assert results[0].payload == "model_server"
    assert results[1].device_assignment == "cuda:1"
    assert load_calls == [("google/gemma-4-E4B-it", "cuda:1", None)]


def test_large_model_falls_back_to_device_map_auto_and_sequential_execution() -> None:
    """REQ-VERIFY-041, SCENARIO-VERIFY-042: 7B+ models use device_map='auto'."""
    specs = [
        {"name": "Small-4B", "hf_id": "org/small-4B"},
        {"name": "Large-7B", "hf_id": "org/large-7B"},
    ]
    load_calls: list[tuple[str, str, str | None]] = []
    task_order: list[str] = []

    def fake_load(
        model_name: str,
        *,
        device: str = "cuda",
        device_map: str | None = None,
    ) -> tuple[SimpleNamespace, SimpleNamespace]:
        load_calls.append((model_name, device, device_map))
        return SimpleNamespace(name=model_name), SimpleNamespace(name=model_name)

    runner = DualGPURunner(
        specs,
        load_model_fn=fake_load,
        unload_fn=lambda model, tokenizer: None,
        torch_module=_FakeTorch(),
    )

    results = runner.run_model_tasks(
        {
            spec["name"]: (
                lambda context, spec_name=spec["name"]: (
                    task_order.append(spec_name)
                    or {
                        "device_assignment": context.device_assignment,
                        "uses_device_map_auto": context.uses_device_map_auto,
                    }
                )
            )
            for spec in specs
        }
    )

    assert runner.execution_mode() == "sharded"
    assert load_calls == [
        ("org/small-4B", "cuda:0", None),
        ("org/large-7B", "cuda", "auto"),
    ]
    assert task_order == ["Small-4B", "Large-7B"]
    assert [result.uses_device_map_auto for result in results] == [False, True]
    assert [result.payload["device_assignment"] for result in results] == [
        "cuda:0",
        "device_map:auto",
    ]


def test_runner_raises_when_a_task_is_missing() -> None:
    """REQ-VERIFY-041: callers must provide one task per configured model."""
    runner = DualGPURunner(
        MODEL_SPECS,
        load_model_fn=lambda *args, **kwargs: (object(), object()),
        torch_module=_FakeTorch(),
    )

    with pytest.raises(ValueError, match="Gemma4-E4B-it"):
        runner.run_model_tasks({"Qwen3.5-0.8B": lambda context: context.model_name})


def test_runner_raises_when_model_load_fails() -> None:
    """REQ-VERIFY-041: load failures surface as runtime errors before task execution."""
    runner = DualGPURunner(
        MODEL_SPECS,
        load_model_fn=lambda *args, **kwargs: (None, None),
        torch_module=_FakeTorch(),
    )

    with pytest.raises(RuntimeError, match="Failed to load dual-GPU model"):
        runner.run_model_tasks(
            {spec["name"]: (lambda context: context.model_name) for spec in MODEL_SPECS}
        )


def test_runner_exports_are_available_from_carnot_inference() -> None:
    """REQ-VERIFY-041: new dual-GPU symbols are exported from carnot.inference."""
    from carnot.inference import DualGPURunner as ExportedRunner

    assert ExportedRunner is DualGPURunner
    assert DualGPUExecutionResult.__name__ == "DualGPUExecutionResult"
