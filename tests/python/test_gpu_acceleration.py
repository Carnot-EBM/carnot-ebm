"""Tests for GPU-acceleration wiring: ModelServer + DualGPURunner in ExperimentTemplate,
three-way benchmark, CPU fallback, and CARNOT_NO_SERVER env-var.

Spec coverage:
  REQ-INFRA-007  — DualGPU auto-assignment
  REQ-INFRA-014  — Explicit failure when CARNOT_FORCE_LIVE=1 and GPU unavailable
  REQ-VERIFY-036 — ModelServer warm cache + batching
  REQ-VERIFY-037 — ModelServer batching worker
  REQ-VERIFY-038 — ModelServer integration with model_loader
  REQ-VERIFY-041 — DualGPURunner paired benchmark tasks
  SCENARIO-VERIFY-036 — one forward pass through warm server
  SCENARIO-VERIFY-038 — server-backed model handle routing
  SCENARIO-VERIFY-110 — setup_gpu() health_status dict
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Import ExperimentTemplate from scripts/
# ---------------------------------------------------------------------------

_SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "experiment_template.py"
sys.path.insert(0, str(_SCRIPT_PATH.parent))

_spec = importlib.util.spec_from_file_location("experiment_template", _SCRIPT_PATH)
assert _spec is not None and _spec.loader is not None
_mod = importlib.util.module_from_spec(_spec)
sys.modules["experiment_template"] = _mod
_spec.loader.exec_module(_mod)  # type: ignore[union-attr]

ExperimentTemplate = _mod.ExperimentTemplate
_cuda_is_available = _mod._cuda_is_available


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _healthy_prewarm(model_name: str, hf_id: str, gpu_id: int, **kwargs: Any) -> Any:
    """Mock prewarm that always reports healthy."""
    r = MagicMock()
    r.health_ok = True
    r.load_time_s = 0.05
    r.stall_root_cause = None
    return r


def _make_model_specs(n: int = 2) -> list[dict[str, Any]]:
    return [{"name": f"Model{i}", "hf_id": f"mock/model{i}", "gpu": i} for i in range(n)]


# ---------------------------------------------------------------------------
# TestCudaIsAvailable
# ---------------------------------------------------------------------------


class TestCudaIsAvailable:
    """_cuda_is_available() returns False safely when torch is unavailable."""

    def test_returns_bool(self) -> None:
        """_cuda_is_available() always returns a bool.  REQ-INFRA-007"""
        result = _cuda_is_available()
        assert isinstance(result, bool)

    def test_returns_false_when_torch_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Returns False when torch is not importable.  REQ-INFRA-007"""
        import builtins

        real_import = builtins.__import__

        def _patched_import(name: str, *args: Any, **kwargs: Any) -> Any:
            if name == "torch":
                raise ImportError("torch not available")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _patched_import)
        # Re-call the helper; it must not raise.
        result = _mod._cuda_is_available()
        assert result is False

    def test_returns_false_when_cuda_unavailable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Returns False when torch.cuda.is_available() returns False.  REQ-INFRA-007"""
        fake_cuda = MagicMock()
        fake_cuda.is_available.return_value = False
        fake_torch = MagicMock()
        fake_torch.cuda = fake_cuda

        with patch.dict(sys.modules, {"torch": fake_torch}):
            # Force re-evaluation inside the helper
            import importlib

            importlib.reload(_mod)
            result = _mod._cuda_is_available()
        assert result is False


# ---------------------------------------------------------------------------
# TestSetupGpuCpuFallback — SCENARIO-VERIFY-110
# ---------------------------------------------------------------------------


class TestSetupGpuCpuFallback:
    """setup_gpu() degrades gracefully when no CUDA GPUs are detected."""

    def test_cpu_fallback_returns_all_healthy(self, tmp_path: Path) -> None:
        """CPU fallback marks all models healthy so the experiment can proceed.
        SCENARIO-VERIFY-110 / CPU fallback contract.
        """
        t = ExperimentTemplate(999, "T", "results/out.json", requires_gpu=True, repo_root=tmp_path)
        specs = _make_model_specs(2)

        # Force CPU-only by patching _cuda_is_available inside the template module.
        with patch.object(_mod, "_cuda_is_available", return_value=False):
            status = t.setup_gpu(specs, prewarm_fn=_healthy_prewarm)

        assert status["all_healthy"] is True

    def test_cpu_fallback_sets_cpu_fallback_flag(self, tmp_path: Path) -> None:
        """cpu_fallback=True in status when no GPU detected.  SCENARIO-VERIFY-110"""
        t = ExperimentTemplate(999, "T", "results/out.json", repo_root=tmp_path)
        with patch.object(_mod, "_cuda_is_available", return_value=False):
            status = t.setup_gpu(_make_model_specs(1), prewarm_fn=_healthy_prewarm)
        assert status["cpu_fallback"] is True

    def test_cpu_fallback_model_server_active_false(self, tmp_path: Path) -> None:
        """ModelServer is NOT started in CPU fallback mode.  REQ-VERIFY-036"""
        t = ExperimentTemplate(999, "T", "results/out.json", repo_root=tmp_path)
        with patch.object(_mod, "_cuda_is_available", return_value=False):
            status = t.setup_gpu(_make_model_specs(1), prewarm_fn=_healthy_prewarm)
        assert status["model_server_active"] is False
        assert t.model_server is None

    def test_cpu_fallback_gpu_runner_active_false(self, tmp_path: Path) -> None:
        """DualGPURunner is NOT created in CPU fallback mode.  REQ-VERIFY-041"""
        t = ExperimentTemplate(999, "T", "results/out.json", repo_root=tmp_path)
        with patch.object(_mod, "_cuda_is_available", return_value=False):
            status = t.setup_gpu(_make_model_specs(2), prewarm_fn=_healthy_prewarm)
        assert status["gpu_runner_active"] is False
        assert t.gpu_runner is None

    def test_cpu_fallback_no_prewarm_fn_still_succeeds(self, tmp_path: Path) -> None:
        """CPU fallback with no explicit prewarm_fn uses no-op and returns healthy.
        This verifies the CPU-safe default prewarm path.  REQ-INFRA-007
        """
        t = ExperimentTemplate(999, "T", "results/out.json", repo_root=tmp_path)
        with patch.object(_mod, "_cuda_is_available", return_value=False):
            # No prewarm_fn — must not try to import experiment_294 (which requires GPU)
            status = t.setup_gpu(_make_model_specs(2))
        assert status["all_healthy"] is True
        assert status["cpu_fallback"] is True

    def test_cpu_fallback_gpu_monitor_results_present(self, tmp_path: Path) -> None:
        """cpu_fallback status dict still has gpu_monitor_results key.  SCENARIO-VERIFY-110"""
        t = ExperimentTemplate(999, "T", "results/out.json", repo_root=tmp_path)
        with patch.object(_mod, "_cuda_is_available", return_value=False):
            status = t.setup_gpu(_make_model_specs(1), prewarm_fn=_healthy_prewarm)
        assert "gpu_monitor_results" in status
        assert status["gpu_monitor_results"]["n_gpus_detected"] == 0

    def test_cpu_fallback_prewarm_fn_still_called_when_provided(
        self, tmp_path: Path
    ) -> None:
        """Explicitly provided prewarm_fn is still called in CPU fallback mode.
        This preserves backward compatibility for tests that mock prewarm_fn.
        SCENARIO-VERIFY-110
        """
        calls: list[str] = []

        def _tracking_prewarm(model_name: str, hf_id: str, gpu_id: int, **kwargs: Any) -> Any:
            calls.append(model_name)
            r = MagicMock()
            r.health_ok = True
            r.load_time_s = 0.0
            r.stall_root_cause = None
            return r

        t = ExperimentTemplate(999, "T", "results/out.json", repo_root=tmp_path)
        specs = _make_model_specs(2)
        with patch.object(_mod, "_cuda_is_available", return_value=False):
            t.setup_gpu(specs, prewarm_fn=_tracking_prewarm)
        assert calls == ["Model0", "Model1"]

    def test_cpu_fallback_unhealthy_prewarm_propagates(self, tmp_path: Path) -> None:
        """all_healthy=False propagates even in CPU fallback when prewarm_fn reports failure.
        SCENARIO-VERIFY-110
        """

        def _failing_prewarm(model_name: str, hf_id: str, gpu_id: int, **kwargs: Any) -> Any:
            r = MagicMock()
            r.health_ok = False
            r.load_time_s = 0.0
            r.stall_root_cause = "mock_failure"
            return r

        t = ExperimentTemplate(999, "T", "results/out.json", repo_root=tmp_path)
        with patch.object(_mod, "_cuda_is_available", return_value=False):
            status = t.setup_gpu(_make_model_specs(1), prewarm_fn=_failing_prewarm)
        assert status["all_healthy"] is False


# ---------------------------------------------------------------------------
# TestSetupGpuNoServerFlag — REQ-INFRA-007
# ---------------------------------------------------------------------------


class TestSetupGpuNoServerFlag:
    """use_server=False and CARNOT_NO_SERVER=1 disable ModelServer startup."""

    def _make_mock_model_server(self) -> MagicMock:
        """Return a mock ModelServer that records start() calls."""
        ms = MagicMock()
        ms.start.return_value = None
        ms.shutdown.return_value = None
        ms.serves_model.return_value = True
        return ms

    def test_use_server_false_skips_model_server(self, tmp_path: Path) -> None:
        """use_server=False means ModelServer is not started.  REQ-INFRA-007"""
        t = ExperimentTemplate(999, "T", "results/out.json", repo_root=tmp_path)
        with (
            patch.object(_mod, "_cuda_is_available", return_value=True),
            patch("carnot.inference.model_server.ModelServer") as MockMS,
        ):
            status = t.setup_gpu(
                _make_model_specs(2),
                prewarm_fn=_healthy_prewarm,
                use_server=False,
            )
        MockMS.assert_not_called()
        assert status["model_server_active"] is False
        assert t.model_server is None

    def test_carnot_no_server_env_skips_model_server(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """CARNOT_NO_SERVER=1 disables ModelServer (--no-server env-var equivalent).
        REQ-INFRA-007
        """
        monkeypatch.setenv("CARNOT_NO_SERVER", "1")
        t = ExperimentTemplate(999, "T", "results/out.json", repo_root=tmp_path)
        with (
            patch.object(_mod, "_cuda_is_available", return_value=True),
            patch("carnot.inference.model_server.ModelServer") as MockMS,
        ):
            status = t.setup_gpu(_make_model_specs(2), prewarm_fn=_healthy_prewarm)
        MockMS.assert_not_called()
        assert status["model_server_active"] is False

    def test_carnot_no_server_env_0_does_not_skip(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """CARNOT_NO_SERVER=0 (default) does NOT disable ModelServer.  REQ-INFRA-007"""
        monkeypatch.setenv("CARNOT_NO_SERVER", "0")
        ms_instance = MagicMock()
        ms_instance.start.return_value = None
        ms_instance.serves_model.return_value = True

        t = ExperimentTemplate(999, "T", "results/out.json", repo_root=tmp_path)
        with (
            patch.object(_mod, "_cuda_is_available", return_value=True),
            patch("carnot.inference.model_server.ModelServer", return_value=ms_instance),
            patch("carnot.inference.dual_gpu.DualGPURunner"),
        ):
            status = t.setup_gpu(_make_model_specs(2), prewarm_fn=_healthy_prewarm)
        ms_instance.start.assert_called_once()
        assert status["model_server_active"] is True


# ---------------------------------------------------------------------------
# TestSetupGpuModelServerIntegration — REQ-VERIFY-036
# ---------------------------------------------------------------------------


class TestSetupGpuModelServerIntegration:
    """setup_gpu() starts ModelServer and stores it on self.model_server."""

    def test_model_server_started_when_cuda_available(self, tmp_path: Path) -> None:
        """ModelServer.start() called when CUDA is available and use_server=True.
        REQ-VERIFY-036 / SCENARIO-VERIFY-110
        """
        ms_instance = MagicMock()
        ms_instance.start.return_value = None
        ms_instance.serves_model.return_value = True

        t = ExperimentTemplate(999, "T", "results/out.json", repo_root=tmp_path)
        with (
            patch.object(_mod, "_cuda_is_available", return_value=True),
            patch("carnot.inference.model_server.ModelServer", return_value=ms_instance),
            patch("carnot.inference.dual_gpu.DualGPURunner"),
        ):
            status = t.setup_gpu(_make_model_specs(2), prewarm_fn=_healthy_prewarm)

        ms_instance.start.assert_called_once()
        assert t.model_server is ms_instance
        assert status["model_server_active"] is True

    def test_model_server_receives_all_hf_ids(self, tmp_path: Path) -> None:
        """ModelServer is constructed with the hf_id of every model_spec.
        REQ-VERIFY-036
        """
        constructor_calls: list[list[str]] = []

        class _TrackingMS(MagicMock):
            def __init__(self, model_names: list[str], **kwargs: Any) -> None:
                super().__init__()
                constructor_calls.append(list(model_names))

            def start(self) -> None:
                pass

            def serves_model(self, _name: str) -> bool:
                return True

        t = ExperimentTemplate(999, "T", "results/out.json", repo_root=tmp_path)
        specs = [
            {"name": "Qwen", "hf_id": "Qwen/Qwen3.5-0.8B", "gpu": 0},
            {"name": "Gemma", "hf_id": "google/gemma-4-E4B-it", "gpu": 1},
        ]
        with (
            patch.object(_mod, "_cuda_is_available", return_value=True),
            patch("carnot.inference.model_server.ModelServer", _TrackingMS),
            patch("carnot.inference.dual_gpu.DualGPURunner"),
        ):
            t.setup_gpu(specs, prewarm_fn=_healthy_prewarm)

        assert constructor_calls == [["Qwen/Qwen3.5-0.8B", "google/gemma-4-E4B-it"]]

    def test_model_server_failure_falls_back_gracefully(self, tmp_path: Path) -> None:
        """ModelServer startup failure falls back to cold-load (model_server_active=False).
        REQ-VERIFY-036 / graceful degradation contract.
        """

        def _exploding_start(self: Any) -> None:
            raise RuntimeError("GPU OOM")

        ms_instance = MagicMock()
        ms_instance.start.side_effect = RuntimeError("GPU OOM")

        t = ExperimentTemplate(999, "T", "results/out.json", repo_root=tmp_path)
        with (
            patch.object(_mod, "_cuda_is_available", return_value=True),
            patch("carnot.inference.model_server.ModelServer", return_value=ms_instance),
        ):
            status = t.setup_gpu(_make_model_specs(2), prewarm_fn=_healthy_prewarm)

        assert status["model_server_active"] is False
        assert t.model_server is None
        # The experiment should still have valid health results (all_healthy from prewarm)
        assert status["all_healthy"] is True


# ---------------------------------------------------------------------------
# TestSetupGpuDualGPURunnerIntegration — REQ-VERIFY-041
# ---------------------------------------------------------------------------


class TestSetupGpuDualGPURunnerIntegration:
    """setup_gpu() creates DualGPURunner with model_server when >=2 specs."""

    def test_dual_gpu_runner_created_with_model_server(self, tmp_path: Path) -> None:
        """DualGPURunner is constructed with model_server=<ModelServer instance>.
        REQ-VERIFY-041 / SCENARIO-VERIFY-042
        """
        ms_instance = MagicMock()
        ms_instance.start.return_value = None
        ms_instance.serves_model.return_value = True

        runner_kwargs: list[dict[str, Any]] = []

        class _TrackingRunner(MagicMock):
            def __init__(self, specs: Any, **kwargs: Any) -> None:
                super().__init__()
                runner_kwargs.append(kwargs)

            def execution_mode(self) -> str:
                return "parallel"

        t = ExperimentTemplate(999, "T", "results/out.json", repo_root=tmp_path)
        with (
            patch.object(_mod, "_cuda_is_available", return_value=True),
            patch("carnot.inference.model_server.ModelServer", return_value=ms_instance),
            patch("carnot.inference.dual_gpu.DualGPURunner", _TrackingRunner),
        ):
            status = t.setup_gpu(_make_model_specs(2), prewarm_fn=_healthy_prewarm)

        assert status["gpu_runner_active"] is True
        assert t.gpu_runner is not None
        assert len(runner_kwargs) == 1
        assert runner_kwargs[0]["model_server"] is ms_instance

    def test_no_dual_gpu_runner_for_single_spec(self, tmp_path: Path) -> None:
        """DualGPURunner is NOT created when only 1 model spec is provided.
        REQ-VERIFY-041
        """
        ms_instance = MagicMock()
        ms_instance.start.return_value = None
        ms_instance.serves_model.return_value = True

        t = ExperimentTemplate(999, "T", "results/out.json", repo_root=tmp_path)
        with (
            patch.object(_mod, "_cuda_is_available", return_value=True),
            patch("carnot.inference.model_server.ModelServer", return_value=ms_instance),
            patch("carnot.inference.dual_gpu.DualGPURunner") as MockRunner,
        ):
            status = t.setup_gpu(_make_model_specs(1), prewarm_fn=_healthy_prewarm)

        MockRunner.assert_not_called()
        assert status["gpu_runner_active"] is False

    def test_dual_gpu_runner_failure_does_not_block_experiment(self, tmp_path: Path) -> None:
        """DualGPURunner creation failure is non-fatal; experiment continues.
        REQ-VERIFY-041
        """
        ms_instance = MagicMock()
        ms_instance.start.return_value = None
        ms_instance.serves_model.return_value = True

        t = ExperimentTemplate(999, "T", "results/out.json", repo_root=tmp_path)
        with (
            patch.object(_mod, "_cuda_is_available", return_value=True),
            patch("carnot.inference.model_server.ModelServer", return_value=ms_instance),
            patch(
                "carnot.inference.dual_gpu.DualGPURunner",
                side_effect=RuntimeError("CUDA out of memory"),
            ),
        ):
            status = t.setup_gpu(_make_model_specs(2), prewarm_fn=_healthy_prewarm)

        assert status["gpu_runner_active"] is False
        assert t.gpu_runner is None
        # ModelServer still started successfully
        assert status["model_server_active"] is True


# ---------------------------------------------------------------------------
# TestSetupGpuReturnKeys — SCENARIO-VERIFY-110
# ---------------------------------------------------------------------------


class TestSetupGpuReturnKeys:
    """setup_gpu() return dict contains all expected keys in both GPU and CPU modes."""

    EXPECTED_KEYS = {
        "all_healthy",
        "models",
        "prewarm_time_s",
        "dual_gpu_auto_assigned",
        "model_server_active",
        "gpu_runner_active",
        "cpu_fallback",
        "gpu_monitor_results",
    }

    def test_all_keys_present_in_cpu_fallback_mode(self, tmp_path: Path) -> None:
        """All expected keys present in CPU fallback mode.  SCENARIO-VERIFY-110"""
        t = ExperimentTemplate(999, "T", "results/out.json", repo_root=tmp_path)
        with patch.object(_mod, "_cuda_is_available", return_value=False):
            status = t.setup_gpu(_make_model_specs(2), prewarm_fn=_healthy_prewarm)
        for key in self.EXPECTED_KEYS:
            assert key in status, f"Missing key: {key}"

    def test_all_keys_present_when_server_disabled(self, tmp_path: Path) -> None:
        """All expected keys present when use_server=False.  SCENARIO-VERIFY-110"""
        t = ExperimentTemplate(999, "T", "results/out.json", repo_root=tmp_path)
        with patch.object(_mod, "_cuda_is_available", return_value=False):
            status = t.setup_gpu(
                _make_model_specs(2),
                prewarm_fn=_healthy_prewarm,
                use_server=False,
            )
        for key in self.EXPECTED_KEYS:
            assert key in status, f"Missing key: {key}"


# ---------------------------------------------------------------------------
# TestThreeWayBenchmark — REQ-VERIFY-036, REQ-VERIFY-037
# ---------------------------------------------------------------------------


class TestThreeWayBenchmark:
    """benchmark_three_way() measures cold, warm, and TRT tiers deterministically."""

    def test_three_way_benchmark_structure(self) -> None:
        """benchmark_three_way() returns ThreeWayBenchmarkResult with correct fields.
        REQ-VERIFY-036
        """
        from carnot.inference.model_server import ThreeWayBenchmarkResult, benchmark_three_way

        class _FakeClock:
            def __init__(self) -> None:
                self.value = 0.0

            def __call__(self) -> float:
                return self.value

            def advance(self, seconds: float) -> None:
                self.value += seconds

        clock = _FakeClock()
        questions = [f"q{i}" for i in range(10)]

        cold_load_calls: list[str] = []
        warm_batch_calls: list[list[str]] = []

        def _cold_load(model_name: str) -> tuple[dict, dict]:
            cold_load_calls.append(model_name)
            clock.advance(3.0)  # 3 s per cold load
            return {"model": model_name}, {"tok": model_name}

        def _cold_generate(
            model: dict, tokenizer: dict, prompt: str, max_new_tokens: int
        ) -> str:
            clock.advance(0.1)  # 0.1 s per question
            return "cold"

        def _batch_generate(
            model: dict, tokenizer: dict, prompts: list[str], max_new_tokens: int
        ) -> list[str]:
            warm_batch_calls.append(list(prompts))
            clock.advance(0.5)  # 0.5 s per batch
            return [f"warm::{p}" for p in prompts]

        from carnot.inference.model_server import ModelServer

        def _hf_factory() -> ModelServer:
            return ModelServer(
                ["Qwen/Qwen3.5-0.8B"],
                loader=lambda _: ({"model": "fake"}, {"tok": "fake"}),
                batch_generate_fn=_batch_generate,
                clock=clock,
            )

        result = benchmark_three_way(
            "Qwen/Qwen3.5-0.8B",
            questions,
            load_model_fn=_cold_load,
            generate_fn=_cold_generate,
            hf_server_factory=_hf_factory,
            clock=clock,
        )

        assert isinstance(result, ThreeWayBenchmarkResult)
        assert result.model_name == "Qwen/Qwen3.5-0.8B"
        assert result.n_questions == 10
        # Cold: 10 * (3.0 + 0.1) = 31.0 s
        assert result.cold_elapsed_seconds == pytest.approx(31.0)
        # warm_speedup > 1 (server is faster than cold-load)
        assert result.warm_speedup > 1.0
        # trt_speedup > 1
        assert result.trt_speedup > 1.0

    def test_three_way_result_fields_populated(self) -> None:
        """ThreeWayBenchmarkResult has all expected fields.  REQ-VERIFY-036"""
        from carnot.inference.model_server import ThreeWayBenchmarkResult

        r = ThreeWayBenchmarkResult(
            model_name="test/model",
            n_questions=10,
            cold_elapsed_seconds=30.0,
            warm_elapsed_seconds=5.0,
            trt_elapsed_seconds=2.0,
            warm_speedup=6.0,
            trt_speedup=15.0,
            trt_available=True,
        )
        assert r.warm_speedup == pytest.approx(6.0)
        assert r.trt_speedup == pytest.approx(15.0)
        assert r.trt_available is True

    def test_three_way_speedup_zero_guard(self) -> None:
        """Speedup is inf when elapsed time is zero (no div-by-zero).  REQ-VERIFY-036"""
        from carnot.inference.model_server import ModelServer, benchmark_three_way

        clock_val = [0.0]

        def _clock() -> float:
            return clock_val[0]

        def _zero_loader(model_name: str) -> tuple[dict, dict]:
            return {"model": model_name}, {"tok": model_name}

        def _zero_generate(model: dict, tok: dict, prompt: str, max_new_tokens: int) -> str:
            return "x"

        def _zero_batch(
            model: dict, tok: dict, prompts: list[str], max_new_tokens: int
        ) -> list[str]:
            return ["x"] * len(prompts)

        def _factory() -> ModelServer:
            return ModelServer(
                ["test/model"],
                loader=_zero_loader,
                batch_generate_fn=_zero_batch,
                clock=_clock,
            )

        result = benchmark_three_way(
            "test/model",
            ["q1"],
            load_model_fn=_zero_loader,
            generate_fn=_zero_generate,
            hf_server_factory=_factory,
            clock=_clock,
        )
        # When all times are zero the speedup should be inf (no crash)
        assert result.warm_speedup == float("inf") or result.warm_speedup > 0
        assert result.trt_speedup == float("inf") or result.trt_speedup > 0

    def test_three_way_exported_from_model_server(self) -> None:
        """benchmark_three_way and ThreeWayBenchmarkResult are in __all__.
        REQ-VERIFY-036
        """
        from carnot.inference import model_server as ms_module

        assert "benchmark_three_way" in ms_module.__all__
        assert "ThreeWayBenchmarkResult" in ms_module.__all__


# ---------------------------------------------------------------------------
# TestExp218NoServerFlag — integration smoke test for build_parser
# ---------------------------------------------------------------------------


class TestExp218NoServerFlag:
    """Exp 218 build_parser() includes --no-server flag."""

    def test_no_server_flag_parseable(self) -> None:
        """build_parser() accepts --no-server without error.  REQ-INFRA-007"""
        import importlib.util as ilu

        harness_path = (
            Path(__file__).resolve().parents[2]
            / "scripts"
            / "experiment_218_live_dual_model_suite.py"
        )
        spec218 = ilu.spec_from_file_location("exp218", harness_path)
        assert spec218 is not None and spec218.loader is not None
        mod218 = ilu.module_from_spec(spec218)
        spec218.loader.exec_module(mod218)  # type: ignore[union-attr]

        parser = mod218.build_parser()
        # --benchmark is required; supply it along with --no-server
        args = parser.parse_args(["--benchmark", "gsm8k_semantic", "--no-server"])
        assert args.no_server is True

    def test_no_server_default_false(self) -> None:
        """--no-server defaults to False (ModelServer enabled by default).  REQ-INFRA-007"""
        import importlib.util as ilu

        harness_path = (
            Path(__file__).resolve().parents[2]
            / "scripts"
            / "experiment_218_live_dual_model_suite.py"
        )
        spec218 = ilu.spec_from_file_location("exp218_2", harness_path)
        assert spec218 is not None and spec218.loader is not None
        mod218 = ilu.module_from_spec(spec218)
        spec218.loader.exec_module(mod218)  # type: ignore[union-attr]

        parser = mod218.build_parser()
        args = parser.parse_args(["--benchmark", "humaneval_property"])
        assert args.no_server is False
