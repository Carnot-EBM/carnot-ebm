"""Tests for carnot.inference.model_server and warm-server integration.

Spec: REQ-VERIFY-036, REQ-VERIFY-037, REQ-VERIFY-038,
SCENARIO-VERIFY-036, SCENARIO-VERIFY-037, SCENARIO-VERIFY-038
"""

from __future__ import annotations

from typing import Any

import pytest

from carnot.inference import ModelServer, benchmark_cold_load_vs_warm_server
from carnot.inference.model_loader import (
    ServerBackedModelHandle,
    clear_model_server,
    generate,
    load_model,
    register_model_server,
)


class _FakeClock:
    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


class _FakeCuda:
    def __init__(self, *, available: bool, allocated: int = 0, reserved: int = 0) -> None:
        self._available = available
        self._allocated = allocated
        self._reserved = reserved
        self.empty_cache_calls = 0

    def is_available(self) -> bool:
        return self._available

    def memory_allocated(self) -> int:
        return self._allocated

    def memory_reserved(self) -> int:
        return self._reserved

    def empty_cache(self) -> None:
        self.empty_cache_calls += 1


class _FakeTorch:
    def __init__(self, *, cuda_available: bool, allocated: int = 0, reserved: int = 0) -> None:
        self.cuda = _FakeCuda(
            available=cuda_available,
            allocated=allocated,
            reserved=reserved,
        )


@pytest.fixture(autouse=True)
def clear_registered_server() -> None:
    """REQ-VERIFY-038: tests isolate model-loader server registration."""
    clear_model_server()
    yield
    clear_model_server()


def _make_loader(load_calls: list[str], clock: _FakeClock | None = None) -> Any:
    def fake_loader(model_name: str) -> tuple[dict[str, str], dict[str, str]]:
        load_calls.append(model_name)
        if clock is not None:
            clock.advance(3.0)
        return {"model_name": model_name}, {"tokenizer_name": model_name}

    return fake_loader


def _make_batch_generate(
    batch_calls: list[list[str]],
    clock: _FakeClock | None = None,
) -> Any:
    def fake_batch_generate(
        model: dict[str, str],
        tokenizer: dict[str, str],
        prompts: list[str],
        max_new_tokens: int,
    ) -> list[str]:
        del tokenizer, max_new_tokens
        batch_calls.append(list(prompts))
        if clock is not None:
            clock.advance(0.25)
        return [f"{model['model_name']}::{prompt}" for prompt in prompts]

    return fake_batch_generate


class TestModelServerLifecycle:
    """Tests for server startup, shutdown, and exports."""

    def test_rejects_batch_sizes_outside_supported_range(self) -> None:
        """REQ-VERIFY-036: ModelServer validates batch_size in the inclusive 1..16 range."""
        with pytest.raises(ValueError, match="batch_size"):
            ModelServer(["Qwen/Qwen3.5-0.8B"], batch_size=0)
        with pytest.raises(ValueError, match="batch_size"):
            ModelServer(["Qwen/Qwen3.5-0.8B"], batch_size=17)

    def test_context_manager_loads_models_once_and_releases_resources(self) -> None:
        """SCENARIO-VERIFY-037: entering loads eagerly and exiting clears warm resources."""
        load_calls: list[str] = []
        batch_calls: list[list[str]] = []
        fake_torch = _FakeTorch(cuda_available=True, allocated=123, reserved=456)

        with ModelServer(
            ["Qwen/Qwen3.5-0.8B", "google/gemma-4-E4B-it"],
            loader=_make_loader(load_calls),
            batch_generate_fn=_make_batch_generate(batch_calls),
            torch_module=fake_torch,
        ) as server:
            health = server.health_check()
            assert health["running"] is True
            assert health["loaded_models"] == [
                "Qwen/Qwen3.5-0.8B",
                "google/gemma-4-E4B-it",
            ]
            assert load_calls == [
                "Qwen/Qwen3.5-0.8B",
                "google/gemma-4-E4B-it",
            ]
            assert batch_calls == []

        post_shutdown = server.health_check()
        assert post_shutdown["running"] is False
        assert post_shutdown["loaded_models"] == []
        assert fake_torch.cuda.empty_cache_calls == 1

    def test_exports_are_available_from_carnot_inference(self) -> None:
        """REQ-VERIFY-036: ModelServer and benchmark helper are exported from carnot.inference."""
        assert ModelServer is not None
        assert callable(benchmark_cold_load_vs_warm_server)


class TestModelServerBatching:
    """Tests for queued generation, batching, and health stats."""

    def test_generate_batch_preserves_result_order_and_records_stats(self) -> None:
        """SCENARIO-VERIFY-036: one forward pass returns per-question results in order."""
        load_calls: list[str] = []
        batch_calls: list[list[str]] = []

        with ModelServer(
            ["Qwen/Qwen3.5-0.8B"],
            loader=_make_loader(load_calls),
            batch_generate_fn=_make_batch_generate(batch_calls),
        ) as server:
            results = server.generate_batch(
                ["What is 2+2?", "What is 3+3?", "What is 4+4?"],
                model="Qwen/Qwen3.5-0.8B",
            )
            health = server.health_check()

        assert results == [
            "Qwen/Qwen3.5-0.8B::What is 2+2?",
            "Qwen/Qwen3.5-0.8B::What is 3+3?",
            "Qwen/Qwen3.5-0.8B::What is 4+4?",
        ]
        assert load_calls == ["Qwen/Qwen3.5-0.8B"]
        assert batch_calls == [["What is 2+2?", "What is 3+3?", "What is 4+4?"]]
        assert health["batch_stats"]["total_requests"] == 3
        assert health["batch_stats"]["total_batches"] == 1
        assert health["batch_stats"]["average_batch_size"] == pytest.approx(3.0)
        assert health["batch_stats"]["max_observed_batch_size"] == 3

    def test_generate_batch_splits_large_inputs_at_batch_limit(self) -> None:
        """REQ-VERIFY-037: queued requests are split into batches no larger than batch_size."""
        load_calls: list[str] = []
        batch_calls: list[list[str]] = []
        prompts = [f"question-{index}" for index in range(10)]

        with ModelServer(
            ["Qwen/Qwen3.5-0.8B"],
            batch_size=4,
            loader=_make_loader(load_calls),
            batch_generate_fn=_make_batch_generate(batch_calls),
        ) as server:
            results = server.generate_batch(prompts, model="Qwen/Qwen3.5-0.8B")
            health = server.health_check()

        assert results == [f"Qwen/Qwen3.5-0.8B::{prompt}" for prompt in prompts]
        assert batch_calls == [
            ["question-0", "question-1", "question-2", "question-3"],
            ["question-4", "question-5", "question-6", "question-7"],
            ["question-8", "question-9"],
        ]
        assert health["batch_stats"]["total_requests"] == 10
        assert health["batch_stats"]["total_batches"] == 3
        assert health["batch_stats"]["average_batch_size"] == pytest.approx(10 / 3)
        assert health["batch_stats"]["max_observed_batch_size"] == 4

    def test_health_check_reports_gpu_memory_snapshot(self) -> None:
        """REQ-VERIFY-037: health_check includes CUDA allocation and reservation metrics."""
        fake_torch = _FakeTorch(cuda_available=True, allocated=2048, reserved=4096)

        with ModelServer(
            ["Qwen/Qwen3.5-0.8B"],
            loader=_make_loader([]),
            batch_generate_fn=_make_batch_generate([]),
            torch_module=fake_torch,
        ) as server:
            health = server.health_check()

        assert health["gpu_memory"] == {
            "cuda_available": True,
            "allocated_bytes": 2048,
            "reserved_bytes": 4096,
        }

    def test_unknown_model_raises_key_error(self) -> None:
        """REQ-VERIFY-036: generate_batch rejects model ids that were not loaded by the server."""
        with ModelServer(
            ["Qwen/Qwen3.5-0.8B"],
            loader=_make_loader([]),
            batch_generate_fn=_make_batch_generate([]),
        ) as server:
            with pytest.raises(KeyError, match="google/gemma-4-E4B-it"):
                server.generate_batch(["hello"], model="google/gemma-4-E4B-it")


class TestModelLoaderServerIntegration:
    """Tests for model_loader using a registered warm ModelServer."""

    def test_registered_server_returns_server_backed_handle_and_routes_generate(self) -> None:
        """SCENARIO-VERIFY-038: load_model returns a server handle and generate uses the server."""
        load_calls: list[str] = []
        batch_calls: list[list[str]] = []

        with ModelServer(
            ["Qwen/Qwen3.5-0.8B"],
            loader=_make_loader(load_calls),
            batch_generate_fn=_make_batch_generate(batch_calls),
        ) as server:
            register_model_server(server)
            model, tokenizer = load_model("Qwen/Qwen3.5-0.8B")
            result = generate(model, tokenizer, "Explain batching")
            health = server.health_check()

        assert isinstance(model, ServerBackedModelHandle)
        assert tokenizer is model
        assert result == "Qwen/Qwen3.5-0.8B::Explain batching"
        assert load_calls == ["Qwen/Qwen3.5-0.8B"]
        assert batch_calls == [["Explain batching"]]
        assert health["batch_stats"]["total_requests"] == 1

    def test_registered_server_falls_back_to_direct_load_for_unserved_model(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """REQ-VERIFY-038: load_model cold-loads when the registered server lacks that model."""
        import carnot.inference.model_loader as model_loader

        with ModelServer(
            ["Qwen/Qwen3.5-0.8B"],
            loader=_make_loader([]),
            batch_generate_fn=_make_batch_generate([]),
        ) as server:
            register_model_server(server)
            monkeypatch.setattr(model_loader, "_available_ram_bytes", lambda: 3 * 1024 ** 3)

            class _DummyAutoTokenizer:
                @staticmethod
                def from_pretrained(model_name: str, trust_remote_code: bool = True) -> dict[str, str]:
                    assert model_name == "google/gemma-4-E4B-it"
                    assert trust_remote_code is True
                    return {"tokenizer_name": model_name}

            class _DummyModel:
                def __init__(self) -> None:
                    self.eval_calls = 0

                def eval(self) -> None:
                    self.eval_calls += 1

            dummy_model = _DummyModel()

            class _DummyAutoModel:
                @staticmethod
                def from_pretrained(
                    model_name: str,
                    trust_remote_code: bool = True,
                    torch_dtype: Any = None,
                ) -> _DummyModel:
                    assert model_name == "google/gemma-4-E4B-it"
                    assert trust_remote_code is True
                    assert torch_dtype is not None
                    return dummy_model

            monkeypatch.setattr(model_loader, "AutoTokenizer", _DummyAutoTokenizer)
            monkeypatch.setattr(model_loader, "AutoModelForCausalLM", _DummyAutoModel)

            model, tokenizer = load_model("google/gemma-4-E4B-it")

        assert not isinstance(model, ServerBackedModelHandle)
        assert tokenizer == {"tokenizer_name": "google/gemma-4-E4B-it"}
        assert dummy_model.eval_calls == 1


class TestWarmServerBenchmark:
    """Tests for the deterministic cold-load versus warm-server benchmark."""

    def test_benchmark_reports_speedup_for_fifty_questions(self) -> None:
        """SCENARIO-VERIFY-038: benchmark reports reproducible timings and speedup over 50 prompts."""
        clock = _FakeClock()
        cold_load_calls: list[str] = []
        warm_batch_calls: list[list[str]] = []
        questions = [f"question-{index}" for index in range(50)]

        def cold_load_model(model_name: str) -> tuple[dict[str, str], dict[str, str]]:
            cold_load_calls.append(model_name)
            clock.advance(3.0)
            return {"model_name": model_name}, {"tokenizer_name": model_name}

        def cold_generate(
            model: dict[str, str],
            tokenizer: dict[str, str],
            prompt: str,
            max_new_tokens: int,
        ) -> str:
            del model, tokenizer, prompt, max_new_tokens
            clock.advance(0.05)
            return "cold"

        result = benchmark_cold_load_vs_warm_server(
            "Qwen/Qwen3.5-0.8B",
            questions,
            batch_size=10,
            load_model_fn=cold_load_model,
            generate_fn=cold_generate,
            server_factory=lambda: ModelServer(
                ["Qwen/Qwen3.5-0.8B"],
                batch_size=10,
                loader=_make_loader([], clock),
                batch_generate_fn=_make_batch_generate(warm_batch_calls, clock),
                clock=clock,
            ),
            clock=clock,
        )

        assert result.n_questions == 50
        assert result.cold_elapsed_seconds == pytest.approx(152.5)
        assert result.warm_elapsed_seconds == pytest.approx(4.25)
        assert result.speedup == pytest.approx(152.5 / 4.25)
        assert cold_load_calls == ["Qwen/Qwen3.5-0.8B"] * 50
        assert warm_batch_calls == [
            [f"question-{index}" for index in range(0, 10)],
            [f"question-{index}" for index in range(10, 20)],
            [f"question-{index}" for index in range(20, 30)],
            [f"question-{index}" for index in range(30, 40)],
            [f"question-{index}" for index in range(40, 50)],
        ]
