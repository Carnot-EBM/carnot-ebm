"""Tests for carnot.inference.model_server and warm-server integration.

Spec: REQ-VERIFY-036, REQ-VERIFY-037, REQ-VERIFY-038,
SCENARIO-VERIFY-036, SCENARIO-VERIFY-037, SCENARIO-VERIFY-038
"""

from __future__ import annotations

import threading
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
        with (
            ModelServer(
                ["Qwen/Qwen3.5-0.8B"],
                loader=_make_loader([]),
                batch_generate_fn=_make_batch_generate([]),
            ) as server,
            pytest.raises(KeyError, match="google/gemma-4-E4B-it"),
        ):
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
            monkeypatch.setattr(model_loader, "_available_ram_bytes", lambda: 3 * 1024**3)

            class _DummyAutoTokenizer:
                @staticmethod
                def from_pretrained(
                    model_name: str,
                    trust_remote_code: bool = True,
                ) -> dict[str, str]:
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
        """SCENARIO-VERIFY-038: benchmark reports reproducible timings over 50 prompts."""
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


class TestModelServerCoveragePaths:
    """Extra coverage for default helpers and defensive server paths."""

    def test_start_is_idempotent(self) -> None:
        """REQ-VERIFY-036: repeated start() calls do not reload already-warm models."""
        load_calls: list[str] = []
        server = ModelServer(
            ["Qwen/Qwen3.5-0.8B"],
            loader=_make_loader(load_calls),
            batch_generate_fn=_make_batch_generate([]),
        )
        server.start()
        server.start()
        try:
            assert server.serves_model("Qwen/Qwen3.5-0.8B") is True
        finally:
            server.shutdown()

        assert load_calls == ["Qwen/Qwen3.5-0.8B"]

    def test_default_loader_requests_cuda(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """REQ-VERIFY-036: the default loader path requests CUDA.

        Fallback and opt-out behavior still belong to load_model().
        """
        import carnot.inference.model_server as model_server_module

        load_calls: list[tuple[str, str]] = []

        def fake_load_model(
            model_name: str,
            device: str = "cpu",
            dtype: Any = None,
            max_retries: int = 3,
        ) -> tuple[dict[str, str], dict[str, str]]:
            del dtype, max_retries
            load_calls.append((model_name, device))
            return {"model_name": model_name}, {"tokenizer_name": model_name}

        monkeypatch.setattr(model_server_module, "load_model", fake_load_model)

        with model_server_module.ModelServer(
            ["Qwen/Qwen3.5-0.8B"],
            batch_generate_fn=_make_batch_generate([]),
        ) as server:
            assert server.serves_model("Qwen/Qwen3.5-0.8B") is True

        assert load_calls == [("Qwen/Qwen3.5-0.8B", "cuda")]

    def test_model_device_returns_string_cpu_when_torch_is_unavailable(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """REQ-VERIFY-037: shared device detection still falls back cleanly when torch is absent."""

        class _EmptyModel:
            def parameters(self) -> Any:
                return iter(())

        import carnot.inference.model_loader as model_loader_module

        monkeypatch.setattr(model_loader_module, "torch", None)
        assert model_loader_module._model_device(_EmptyModel()) == "cpu"

    def test_default_batch_generate_issues_one_batched_model_call(self) -> None:
        """SCENARIO-VERIFY-036: default batching uses one padded model.generate() call."""
        import torch

        class _FakeTokenizer:
            eos_token_id = 99
            eos_token = "<eos>"
            pad_token_id = None

            def __init__(self) -> None:
                self.chat_prompts: list[str] = []
                self.tokenized_texts: list[list[str]] = []
                self.decode_inputs: list[list[int]] = []
                self.pad_token: str | None = None

            def apply_chat_template(
                self,
                messages: list[dict[str, str]],
                *,
                tokenize: bool = False,
                add_generation_prompt: bool = True,
                enable_thinking: bool = False,
            ) -> str:
                del tokenize, add_generation_prompt, enable_thinking
                prompt = messages[0]["content"]
                self.chat_prompts.append(prompt)
                return f"chat::{prompt}"

            def __call__(
                self,
                texts: list[str],
                *,
                return_tensors: str = "pt",
                padding: bool = False,
            ) -> dict[str, torch.Tensor]:
                assert return_tensors == "pt"
                assert padding is True
                self.tokenized_texts.append(list(texts))
                return {
                    "input_ids": torch.tensor([[10, 11, 0], [20, 21, 22]]),
                    "attention_mask": torch.tensor([[1, 1, 0], [1, 1, 1]]),
                }

            def decode(self, token_ids: torch.Tensor, *, skip_special_tokens: bool = True) -> str:
                assert skip_special_tokens is True
                values = token_ids.tolist()
                self.decode_inputs.append(values)
                mapping = {
                    (30, 31): "first-response",
                    (40, 41): "<think>ignored</think>second-response",
                }
                return mapping[tuple(values)]

        class _FakeModel:
            def __init__(self) -> None:
                self.generate_calls: list[dict[str, Any]] = []

            def parameters(self) -> Any:
                fake_param = type("Param", (), {"device": torch.device("cpu")})()
                return iter([fake_param])

            def generate(self, **kwargs: Any) -> torch.Tensor:
                self.generate_calls.append(kwargs)
                return torch.tensor(
                    [
                        [10, 11, 0, 30, 31],
                        [20, 21, 22, 40, 41],
                    ]
                )

        fake_model = _FakeModel()
        fake_tokenizer = _FakeTokenizer()

        with ModelServer(
            ["Qwen/Qwen3.5-0.8B"],
            loader=lambda _name: (fake_model, fake_tokenizer),
        ) as server:
            results = server.generate_batch(
                ["question-1", "question-2"],
                model="Qwen/Qwen3.5-0.8B",
            )
            health = server.health_check()

        assert results == ["first-response", "second-response"]
        assert fake_tokenizer.chat_prompts == ["question-1", "question-2"]
        assert fake_tokenizer.tokenized_texts == [["chat::question-1", "chat::question-2"]]
        assert fake_tokenizer.decode_inputs == [[30, 31], [40, 41]]
        assert fake_tokenizer.pad_token == fake_tokenizer.eos_token
        assert len(fake_model.generate_calls) == 1
        assert fake_model.generate_calls[0]["max_new_tokens"] == 256
        assert fake_model.generate_calls[0]["do_sample"] is False
        assert fake_model.generate_calls[0]["pad_token_id"] == fake_tokenizer.eos_token_id
        assert health["batch_stats"]["total_batches"] == 1

    def test_default_batch_generate_guard_paths_and_generate_wrapper(self) -> None:
        """REQ-VERIFY-037: empty-batch guard paths return early and generate() delegates."""
        import carnot.inference.model_server as model_server_module

        assert model_server_module._default_batch_generate(object(), object(), [], 8) == []
        with pytest.raises(RuntimeError, match="called with model=None or tokenizer=None"):
            model_server_module._default_batch_generate(None, object(), ["hello"], 8)

        with ModelServer(
            ["Qwen/Qwen3.5-0.8B"],
            loader=_make_loader([]),
            batch_generate_fn=_make_batch_generate([]),
        ) as server:
            assert (
                server.generate("question-1", model="Qwen/Qwen3.5-0.8B")
                == "Qwen/Qwen3.5-0.8B::question-1"
            )

    def test_start_raises_when_loader_cannot_warm_load(self) -> None:
        """REQ-VERIFY-036: startup fails fast when eager warm loading yields no model."""
        server = ModelServer(
            ["Qwen/Qwen3.5-0.8B"],
            loader=lambda _name: (None, None),
            batch_generate_fn=_make_batch_generate([]),
        )

        with pytest.raises(RuntimeError, match="Failed to warm-load model"):
            server.start()

    def test_generate_batch_handles_empty_prompts_and_requires_running_server(self) -> None:
        """REQ-VERIFY-037: empty prompt batches are no-ops and stopped servers reject work."""
        with ModelServer(
            ["Qwen/Qwen3.5-0.8B"],
            loader=_make_loader([]),
            batch_generate_fn=_make_batch_generate([]),
        ) as server:
            assert server.generate_batch([], model="Qwen/Qwen3.5-0.8B") == []

        with pytest.raises(RuntimeError, match="not running"):
            server.generate_batch(["hello"], model="Qwen/Qwen3.5-0.8B")

    def test_generate_batch_surfaces_worker_errors(self) -> None:
        """REQ-VERIFY-037: worker exceptions propagate back to the caller."""

        def failing_batch_generate(
            model: dict[str, str],
            tokenizer: dict[str, str],
            prompts: list[str],
            max_new_tokens: int,
        ) -> list[str]:
            del model, tokenizer, prompts, max_new_tokens
            raise ValueError("boom")

        with (
            ModelServer(
                ["Qwen/Qwen3.5-0.8B"],
                loader=_make_loader([]),
                batch_generate_fn=failing_batch_generate,
            ) as server,
            pytest.raises(RuntimeError, match="boom"),
        ):
            server.generate_batch(["hello"], model="Qwen/Qwen3.5-0.8B")

    def test_worker_defers_incompatible_requests_until_the_current_batch_finishes(self) -> None:
        """REQ-VERIFY-037: incompatible queued work is deferred and then replayed in order."""
        import carnot.inference.model_server as model_server_module

        batch_calls: list[list[str]] = []
        server = ModelServer(
            ["Qwen/Qwen3.5-0.8B", "google/gemma-4-E4B-it"],
            loader=_make_loader([]),
            batch_generate_fn=_make_batch_generate(batch_calls),
        )
        server._loaded_models = {
            "Qwen/Qwen3.5-0.8B": (
                {"model_name": "Qwen/Qwen3.5-0.8B"},
                {"tokenizer_name": "Qwen/Qwen3.5-0.8B"},
            ),
            "google/gemma-4-E4B-it": (
                {"model_name": "google/gemma-4-E4B-it"},
                {"tokenizer_name": "google/gemma-4-E4B-it"},
            ),
        }
        server._running = True

        req1 = model_server_module._QueuedRequest(
            model_name="Qwen/Qwen3.5-0.8B",
            prompts=("question-1",),
            max_new_tokens=32,
        )
        req2 = model_server_module._QueuedRequest(
            model_name="google/gemma-4-E4B-it",
            prompts=("question-2",),
            max_new_tokens=32,
        )
        server._request_queue.put(req1)
        server._request_queue.put(req2)

        worker = threading.Thread(target=server._worker_loop, daemon=True)
        worker.start()
        assert req1.done.wait(timeout=1.0)
        assert req2.done.wait(timeout=1.0)
        server._stop_event.set()
        worker.join(timeout=1.0)

        assert req1.responses == ["Qwen/Qwen3.5-0.8B::question-1"]
        assert req2.responses == ["google/gemma-4-E4B-it::question-2"]
        assert batch_calls == [["question-1"], ["question-2"]]

    def test_worker_coalesces_compatible_requests_into_one_forward_pass(self) -> None:
        """SCENARIO-VERIFY-036: compatible queued requests are coalesced up to batch_size."""
        import carnot.inference.model_server as model_server_module

        batch_calls: list[list[str]] = []
        server = ModelServer(
            ["Qwen/Qwen3.5-0.8B"],
            loader=_make_loader([]),
            batch_generate_fn=_make_batch_generate(batch_calls),
        )
        server._loaded_models = {
            "Qwen/Qwen3.5-0.8B": (
                {"model_name": "Qwen/Qwen3.5-0.8B"},
                {"tokenizer_name": "Qwen/Qwen3.5-0.8B"},
            ),
        }
        server._running = True

        req1 = model_server_module._QueuedRequest(
            model_name="Qwen/Qwen3.5-0.8B",
            prompts=("question-1",),
            max_new_tokens=32,
        )
        req2 = model_server_module._QueuedRequest(
            model_name="Qwen/Qwen3.5-0.8B",
            prompts=("question-2",),
            max_new_tokens=32,
        )
        server._request_queue.put(req1)
        server._request_queue.put(req2)

        worker = threading.Thread(target=server._worker_loop, daemon=True)
        worker.start()
        assert req1.done.wait(timeout=1.0)
        assert req2.done.wait(timeout=1.0)
        server._stop_event.set()
        worker.join(timeout=1.0)

        assert req1.responses == ["Qwen/Qwen3.5-0.8B::question-1"]
        assert req2.responses == ["Qwen/Qwen3.5-0.8B::question-2"]
        assert batch_calls == [["question-1", "question-2"]]

    def test_fail_pending_requests_marks_deferred_and_queued_work(self) -> None:
        """REQ-VERIFY-037: shutdown cleanup wakes callers waiting on deferred or queued work."""
        import carnot.inference.model_server as model_server_module

        server = ModelServer(
            ["Qwen/Qwen3.5-0.8B"],
            loader=_make_loader([]),
            batch_generate_fn=_make_batch_generate([]),
        )
        deferred = model_server_module._QueuedRequest(
            model_name="Qwen/Qwen3.5-0.8B",
            prompts=("deferred",),
            max_new_tokens=8,
        )
        queued = model_server_module._QueuedRequest(
            model_name="Qwen/Qwen3.5-0.8B",
            prompts=("queued",),
            max_new_tokens=8,
        )
        error = RuntimeError("shutting down")

        server._deferred_requests.append(deferred)
        server._request_queue.put(queued)
        server._request_queue.put(None)
        server._fail_pending_requests(error)

        assert deferred.error is error
        assert deferred.done.is_set()
        assert queued.error is error
        assert queued.done.is_set()
