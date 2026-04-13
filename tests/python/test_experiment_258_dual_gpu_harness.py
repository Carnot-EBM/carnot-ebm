"""Tests for scripts/experiment_258_dual_gpu_harness.py.

All tests run under CARNOT_FORCE_LIVE=0 (simulated / mock mode).

Spec coverage: REQ-VERIFY-041, REQ-VERIFY-036, REQ-VERIFY-037, REQ-VERIFY-038,
SCENARIO-VERIFY-042, SCENARIO-VERIFY-036, SCENARIO-VERIFY-037
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

# ---------------------------------------------------------------------------
# Helpers to load the experiment script as a module without executing main()
# ---------------------------------------------------------------------------

_SCRIPT_PATH = Path(__file__).parent.parent.parent / "scripts" / "experiment_258_dual_gpu_harness.py"


def _load_script() -> Any:
    """Import experiment_258 as a module (mock mode, no live GPU required)."""
    spec = importlib.util.spec_from_file_location("experiment_258", _SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    # Register in sys.modules before exec so @dataclass can resolve the module dict.
    sys.modules["experiment_258"] = mod
    # Ensure simulated mode so imports do not try to hit real CUDA.
    os.environ.setdefault("CARNOT_FORCE_LIVE", "0")
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


# Load once at collection time.
_mod = _load_script()

ThroughputMeasurement = _mod.ThroughputMeasurement
GPUAssignmentVerifier = _mod.GPUAssignmentVerifier
DualGPUBenchmarkHarness = _mod.DualGPUBenchmarkHarness
write_harness_report = _mod.write_harness_report
MODEL_SPECS = _mod.MODEL_SPECS
TARGET_SECONDS_PER_CASE = _mod.TARGET_SECONDS_PER_CASE
MIN_FREE_VRAM_GB = _mod.MIN_FREE_VRAM_GB


# ---------------------------------------------------------------------------
# Shared fakes
# ---------------------------------------------------------------------------


class _FakeCudaMem:
    """Fake torch.cuda that reports configurable free VRAM per device."""

    def __init__(
        self,
        *,
        available: bool = True,
        device_count: int = 2,
        free_per_device: list[int] | None = None,
        empty_cache_calls: list[int] | None = None,
    ) -> None:
        self._available = available
        self._device_count = device_count
        # Express free memory in bytes; default 24 GiB per device.
        _24gib = 24 * (1024**3)
        self._free_per_device: list[int] = free_per_device or [_24gib] * device_count
        self._empty_cache_calls: list[int] = empty_cache_calls if empty_cache_calls is not None else []

    def is_available(self) -> bool:
        return self._available

    def device_count(self) -> int:
        return self._device_count

    def mem_get_info(self, device_index: int) -> tuple[int, int]:
        """Return (free_bytes, total_bytes) matching torch.cuda.mem_get_info()."""
        free = self._free_per_device[device_index]
        return (free, free)  # total == free for simplicity

    def empty_cache(self) -> None:
        self._empty_cache_calls.append(1)


class _FakeTorch:
    def __init__(
        self,
        *,
        available: bool = True,
        device_count: int = 2,
        free_per_device: list[int] | None = None,
        empty_cache_calls: list[int] | None = None,
    ) -> None:
        self.cuda = _FakeCudaMem(
            available=available,
            device_count=device_count,
            free_per_device=free_per_device,
            empty_cache_calls=empty_cache_calls,
        )


# ---------------------------------------------------------------------------
# ThroughputMeasurement tests
# ---------------------------------------------------------------------------


class TestThroughputMeasurement:
    """REQ-VERIFY-041: throughput helper tracks cases/sec and flags ≤3s/case target."""

    def test_empty_report_shows_no_models(self) -> None:
        """Report with no recorded batches returns empty per_model dict."""
        tm = ThroughputMeasurement()
        report = tm.report()
        assert report["per_model"] == {}
        assert report["target_seconds_per_case"] == TARGET_SECONDS_PER_CASE

    def test_single_model_target_met(self) -> None:
        """Target is met when mean seconds/case is below threshold."""
        tm = ThroughputMeasurement()
        tm.record_batch("Qwen3.5-0.8B", n_cases=10, elapsed_seconds=20.0)  # 2.0 s/case
        report = tm.report()
        model_entry = report["per_model"]["Qwen3.5-0.8B"]
        assert model_entry["cases_per_sec"] == pytest.approx(0.5)
        assert model_entry["mean_seconds_per_case"] == pytest.approx(2.0)
        assert model_entry["target_met"] is True

    def test_single_model_target_not_met(self) -> None:
        """Target is not met when mean seconds/case exceeds threshold."""
        tm = ThroughputMeasurement()
        tm.record_batch("Gemma4-E4B-it", n_cases=5, elapsed_seconds=25.0)  # 5.0 s/case
        report = tm.report()
        model_entry = report["per_model"]["Gemma4-E4B-it"]
        assert model_entry["target_met"] is False

    def test_multiple_batches_accumulate(self) -> None:
        """Multiple record_batch calls accumulate for the same model."""
        tm = ThroughputMeasurement()
        tm.record_batch("Qwen3.5-0.8B", n_cases=4, elapsed_seconds=8.0)
        tm.record_batch("Qwen3.5-0.8B", n_cases=4, elapsed_seconds=8.0)
        report = tm.report()
        entry = report["per_model"]["Qwen3.5-0.8B"]
        # 8 cases / 16 seconds = 0.5 cases/sec, 2.0 s/case
        assert entry["cases_per_sec"] == pytest.approx(0.5)
        assert entry["mean_seconds_per_case"] == pytest.approx(2.0)
        assert entry["target_met"] is True

    def test_exactly_at_target_boundary_is_met(self) -> None:
        """Exactly 3.0 s/case is considered target met (≤ threshold)."""
        tm = ThroughputMeasurement()
        tm.record_batch("Qwen3.5-0.8B", n_cases=1, elapsed_seconds=3.0)
        assert tm.report()["per_model"]["Qwen3.5-0.8B"]["target_met"] is True

    def test_two_models_tracked_independently(self) -> None:
        """Two different models are tracked in separate per_model buckets."""
        tm = ThroughputMeasurement()
        tm.record_batch("Qwen3.5-0.8B", n_cases=6, elapsed_seconds=12.0)
        tm.record_batch("Gemma4-E4B-it", n_cases=3, elapsed_seconds=18.0)
        report = tm.report()
        assert report["per_model"]["Qwen3.5-0.8B"]["target_met"] is True   # 2 s/case
        assert report["per_model"]["Gemma4-E4B-it"]["target_met"] is False  # 6 s/case


# ---------------------------------------------------------------------------
# GPUAssignmentVerifier tests
# ---------------------------------------------------------------------------


class TestGPUAssignmentVerifier:
    """REQ-VERIFY-041: GPU assignment verifier checks device count and VRAM thresholds."""

    def test_passes_when_both_gpus_have_sufficient_vram(self) -> None:
        """SCENARIO-VERIFY-042: both GPUs with >20 GiB free — no error raised."""
        torch_mod = _FakeTorch()
        verifier = GPUAssignmentVerifier(min_free_vram_gb=MIN_FREE_VRAM_GB)
        # Should not raise.
        verifier.verify(torch_mod)

    def test_raises_when_gpu0_has_insufficient_vram(self) -> None:
        """SCENARIO-VERIFY-042: GPU 0 with only 10 GiB free triggers error."""
        _10gib = 10 * (1024**3)
        _24gib = 24 * (1024**3)
        torch_mod = _FakeTorch(free_per_device=[_10gib, _24gib])
        verifier = GPUAssignmentVerifier(min_free_vram_gb=MIN_FREE_VRAM_GB)
        with pytest.raises(RuntimeError, match="GPU 0"):
            verifier.verify(torch_mod)

    def test_raises_when_gpu1_has_insufficient_vram(self) -> None:
        """SCENARIO-VERIFY-042: GPU 1 with only 8 GiB free triggers error."""
        _24gib = 24 * (1024**3)
        _8gib = 8 * (1024**3)
        torch_mod = _FakeTorch(free_per_device=[_24gib, _8gib])
        verifier = GPUAssignmentVerifier(min_free_vram_gb=MIN_FREE_VRAM_GB)
        with pytest.raises(RuntimeError, match="GPU 1"):
            verifier.verify(torch_mod)

    def test_raises_when_cuda_unavailable(self) -> None:
        """SCENARIO-VERIFY-042: CUDA not available raises RuntimeError."""
        torch_mod = _FakeTorch(available=False)
        verifier = GPUAssignmentVerifier(min_free_vram_gb=MIN_FREE_VRAM_GB)
        with pytest.raises(RuntimeError, match="CUDA"):
            verifier.verify(torch_mod)

    def test_raises_when_fewer_than_two_gpus(self) -> None:
        """SCENARIO-VERIFY-042: only one GPU present raises RuntimeError."""
        torch_mod = _FakeTorch(device_count=1, free_per_device=[24 * 1024**3])
        verifier = GPUAssignmentVerifier(min_free_vram_gb=MIN_FREE_VRAM_GB)
        with pytest.raises(RuntimeError, match="two"):
            verifier.verify(torch_mod)

    def test_custom_vram_threshold(self) -> None:
        """GPUAssignmentVerifier respects a custom min_free_vram_gb threshold."""
        _5gib = 5 * (1024**3)
        torch_mod = _FakeTorch(free_per_device=[_5gib, _5gib])
        # With default 20 GiB threshold this would fail; with 4 GiB it should pass.
        verifier = GPUAssignmentVerifier(min_free_vram_gb=4.0)
        verifier.verify(torch_mod)  # Should not raise.


# ---------------------------------------------------------------------------
# DualGPUBenchmarkHarness construction tests
# ---------------------------------------------------------------------------


class TestDualGPUBenchmarkHarnessConstruction:
    """REQ-VERIFY-041: harness picks up batch_size from environment."""

    def test_default_batch_size_is_eight(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Default batch_size is 8 when CARNOT_DUAL_GPU_BATCH_SIZE is not set."""
        monkeypatch.delenv("CARNOT_DUAL_GPU_BATCH_SIZE", raising=False)
        harness = DualGPUBenchmarkHarness(torch_module=_FakeTorch())
        assert harness.batch_size == 8

    def test_batch_size_from_environment(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """CARNOT_DUAL_GPU_BATCH_SIZE env var overrides the default."""
        monkeypatch.setenv("CARNOT_DUAL_GPU_BATCH_SIZE", "4")
        harness = DualGPUBenchmarkHarness(torch_module=_FakeTorch())
        assert harness.batch_size == 4

    def test_explicit_batch_size_overrides_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Explicit batch_size constructor arg takes priority over env var."""
        monkeypatch.setenv("CARNOT_DUAL_GPU_BATCH_SIZE", "16")
        harness = DualGPUBenchmarkHarness(batch_size=2, torch_module=_FakeTorch())
        assert harness.batch_size == 2

    def test_model_specs_default_to_exp218_pair(self) -> None:
        """Default model_specs match the Exp 218 Qwen + Gemma pairing."""
        harness = DualGPUBenchmarkHarness(torch_module=_FakeTorch())
        names = [spec["name"] for spec in harness.model_specs]
        assert "Qwen3.5-0.8B" in names
        assert "Gemma4-E4B-it" in names

    def test_gpu0_assigned_qwen_gpu1_assigned_gemma(self) -> None:
        """Qwen goes to GPU 0, Gemma goes to GPU 1 per spec order."""
        harness = DualGPUBenchmarkHarness(torch_module=_FakeTorch())
        assert harness.model_specs[0]["hf_id"] == "Qwen/Qwen3.5-0.8B"
        assert harness.model_specs[1]["hf_id"] == "google/gemma-4-E4B-it"


# ---------------------------------------------------------------------------
# DualGPUBenchmarkHarness.verify_gpu_assignments tests
# ---------------------------------------------------------------------------


class TestHarnessVerifyGPUAssignments:
    """REQ-VERIFY-041: verify_gpu_assignments delegates to GPUAssignmentVerifier."""

    def test_passes_with_sufficient_vram(self) -> None:
        """No error when both GPUs report sufficient free VRAM."""
        harness = DualGPUBenchmarkHarness(torch_module=_FakeTorch())
        harness.verify_gpu_assignments()  # Should not raise.

    def test_raises_with_insufficient_vram(self) -> None:
        """RuntimeError propagates from GPUAssignmentVerifier on low VRAM."""
        low_vram = [1 * (1024**3), 24 * (1024**3)]
        harness = DualGPUBenchmarkHarness(torch_module=_FakeTorch(free_per_device=low_vram))
        with pytest.raises(RuntimeError, match="GPU 0"):
            harness.verify_gpu_assignments()


# ---------------------------------------------------------------------------
# DualGPUBenchmarkHarness.empty_cache_between_runs tests
# ---------------------------------------------------------------------------


class TestHarnessEmptyCache:
    """REQ-VERIFY-036: GPU memory cleanup is triggered between benchmark runs."""

    def test_empty_cache_calls_torch_cuda_empty_cache(self) -> None:
        """empty_cache_between_runs() invokes torch.cuda.empty_cache() exactly once."""
        calls: list[int] = []
        harness = DualGPUBenchmarkHarness(
            torch_module=_FakeTorch(empty_cache_calls=calls)
        )
        harness.empty_cache_between_runs()
        assert len(calls) == 1

    def test_empty_cache_no_op_when_cuda_unavailable(self) -> None:
        """empty_cache_between_runs() is a no-op when CUDA is not available."""
        calls: list[int] = []
        harness = DualGPUBenchmarkHarness(
            torch_module=_FakeTorch(available=False, empty_cache_calls=calls)
        )
        harness.empty_cache_between_runs()
        assert len(calls) == 0


# ---------------------------------------------------------------------------
# DualGPUBenchmarkHarness interface compatibility with Exp 218
# ---------------------------------------------------------------------------


class TestHarnessExp218Compatibility:
    """REQ-VERIFY-041: harness exposes checkpoint_path/load_checkpoint/save_checkpoint
    with the same signatures as Exp 218 so existing runners opt in unchanged.
    """

    def test_checkpoint_path_returns_expected_filename(self, tmp_path: Path) -> None:
        """checkpoint_path() produces the Exp-218-style <benchmark>__<model>__<mode>.json path."""
        harness = DualGPUBenchmarkHarness(torch_module=_FakeTorch())
        p = harness.checkpoint_path(
            tmp_path,
            benchmark="constraint_ir",
            model_name="Qwen3.5-0.8B",
            mode="baseline",
        )
        assert p.name == "constraint_ir__qwen3_5-0_8b__baseline.json"
        assert p.parent == tmp_path

    def test_load_checkpoint_returns_fresh_when_file_missing(self, tmp_path: Path) -> None:
        """load_checkpoint returns a fresh empty result dict when no file exists."""
        harness = DualGPUBenchmarkHarness(torch_module=_FakeTorch())
        result = harness.load_checkpoint(tmp_path / "missing.json", ["case-1", "case-2"])
        assert result["results_by_case"] == {}
        assert result["case_ids"] == ["case-1", "case-2"]

    def test_save_and_reload_checkpoint(self, tmp_path: Path) -> None:
        """save_checkpoint writes and load_checkpoint reloads matching payload."""
        harness = DualGPUBenchmarkHarness(torch_module=_FakeTorch())
        path = tmp_path / "ckpt.json"
        case_ids = ["c1", "c2"]
        payload = {
            "benchmark": "gsm8k_semantic",
            "model_name": "Qwen3.5-0.8B",
            "mode": "baseline",
            "case_ids": case_ids,
            "results_by_case": {"c1": {"score": 1}},
        }
        harness.save_checkpoint(path, payload)
        assert path.exists()
        reloaded = harness.load_checkpoint(path, case_ids)
        assert reloaded["results_by_case"] == {"c1": {"score": 1}}

    def test_load_checkpoint_discards_stale_cohort(self, tmp_path: Path) -> None:
        """load_checkpoint returns fresh state when stored case_ids don't match."""
        harness = DualGPUBenchmarkHarness(torch_module=_FakeTorch())
        path = tmp_path / "ckpt.json"
        harness.save_checkpoint(
            path,
            {
                "case_ids": ["old-case"],
                "results_by_case": {"old-case": {"score": 0}},
            },
        )
        result = harness.load_checkpoint(path, ["new-case"])
        assert result["results_by_case"] == {}

    def test_run_mode_executes_missing_cases_only(self, tmp_path: Path) -> None:
        """run_mode() skips already-checkpointed cases and executes only missing ones."""
        harness = DualGPUBenchmarkHarness(torch_module=_FakeTorch())
        executed: list[str] = []

        def execute(case: dict[str, Any]) -> dict[str, Any]:
            executed.append(str(case["case_id"]))
            return {"result": "ok"}

        cases = [{"case_id": "c1"}, {"case_id": "c2"}, {"case_id": "c3"}]
        # Pre-populate c1 in the checkpoint.
        ckpt_path = harness.checkpoint_path(
            tmp_path, benchmark="gsm8k_semantic", model_name="Qwen3.5-0.8B", mode="baseline"
        )
        harness.save_checkpoint(
            ckpt_path,
            {
                "benchmark": "gsm8k_semantic",
                "model_name": "Qwen3.5-0.8B",
                "mode": "baseline",
                "case_ids": ["c1", "c2", "c3"],
                "results_by_case": {"c1": {"case_id": "c1", "mode": "baseline", "result": "ok"}},
            },
        )
        results = harness.run_mode(
            benchmark="gsm8k_semantic",
            model_name="Qwen3.5-0.8B",
            mode="baseline",
            cases=cases,
            checkpoint_dir=tmp_path,
            execute_case=execute,
        )
        # Only c2 and c3 should have been executed.
        assert set(executed) == {"c2", "c3"}
        assert len(results) == 3

    def test_run_mode_records_throughput(self, tmp_path: Path) -> None:
        """run_mode() populates harness throughput measurements."""
        harness = DualGPUBenchmarkHarness(torch_module=_FakeTorch())

        def execute(case: dict[str, Any]) -> dict[str, Any]:
            return {"result": "ok"}

        cases = [{"case_id": "c1"}, {"case_id": "c2"}]
        harness.run_mode(
            benchmark="gsm8k_semantic",
            model_name="Qwen3.5-0.8B",
            mode="baseline",
            cases=cases,
            checkpoint_dir=tmp_path,
            execute_case=execute,
        )
        report = harness.throughput.report()
        assert "Qwen3.5-0.8B" in report["per_model"]


# ---------------------------------------------------------------------------
# DualGPUBenchmarkHarness.run_suite (mock mode) tests
# ---------------------------------------------------------------------------


class TestHarnessRunSuiteMockMode:
    """REQ-VERIFY-041: run_suite wires DualGPURunner with injected model tasks."""

    def _make_fake_runner(self) -> SimpleNamespace:
        """Return a fake DualGPURunner that records calls."""
        calls: list[dict[str, Any]] = []

        def run_model_tasks(tasks: dict[str, Any]) -> list[SimpleNamespace]:
            calls.append({"tasks": tasks})
            results = []
            for model_name, task_fn in tasks.items():
                # Simulate a minimal context.
                ctx = SimpleNamespace(
                    model_name=model_name,
                    model_hf_id="hf/" + model_name,
                    device_assignment="cuda:0",
                    model=object(),
                    tokenizer=object(),
                )
                result_payload = task_fn(ctx)
                results.append(
                    SimpleNamespace(
                        model_name=model_name,
                        elapsed_seconds=1.0,
                        payload=result_payload,
                    )
                )
            return results

        ns = SimpleNamespace(
            run_model_tasks=run_model_tasks,
            has_two_gpus=lambda: True,
            execution_mode=lambda: "parallel",
            calls=calls,
        )
        return ns

    def test_run_suite_calls_run_model_tasks(self, tmp_path: Path) -> None:
        """run_suite() invokes DualGPURunner.run_model_tasks once per call."""
        harness = DualGPUBenchmarkHarness(torch_module=_FakeTorch())
        fake_runner = self._make_fake_runner()

        def mock_suite_fn(
            *,
            benchmark: str,
            model_spec: dict[str, str],
            model: Any,
            tokenizer: Any,
            **kwargs: Any,
        ) -> dict[str, Any]:
            return {"model_name": model_spec["name"], "paired_runs": []}

        results = harness.run_suite(
            benchmark="constraint_ir",
            cohort=[{"case_id": "c1"}],
            checkpoint_dir=tmp_path,
            policy={},
            max_repairs=1,
            runner=fake_runner,
            suite_fn=mock_suite_fn,
        )
        assert len(fake_runner.calls) == 1
        assert len(results) == 2  # one entry per model

    def test_run_suite_empties_cache_after_completion(self, tmp_path: Path) -> None:
        """run_suite() calls empty_cache_between_runs once after tasks complete."""
        cache_calls: list[int] = []
        harness = DualGPUBenchmarkHarness(
            torch_module=_FakeTorch(empty_cache_calls=cache_calls)
        )
        fake_runner = self._make_fake_runner()

        results = harness.run_suite(
            benchmark="constraint_ir",
            cohort=[{"case_id": "c1"}],
            checkpoint_dir=tmp_path,
            policy={},
            max_repairs=1,
            runner=fake_runner,
            suite_fn=lambda **kwargs: {"model_name": kwargs["model_spec"]["name"], "paired_runs": []},
        )
        assert len(cache_calls) >= 1

    def test_run_suite_returns_one_entry_per_model(self, tmp_path: Path) -> None:
        """run_suite() result list has exactly two entries — one per configured model."""
        harness = DualGPUBenchmarkHarness(torch_module=_FakeTorch())
        fake_runner = self._make_fake_runner()
        results = harness.run_suite(
            benchmark="gsm8k_semantic",
            cohort=[{"case_id": "x1"}],
            checkpoint_dir=tmp_path,
            policy={},
            max_repairs=0,
            runner=fake_runner,
            suite_fn=lambda **kwargs: {"model_name": kwargs["model_spec"]["name"], "paired_runs": []},
        )
        model_names = [r["model_name"] for r in results]
        assert "Qwen3.5-0.8B" in model_names
        assert "Gemma4-E4B-it" in model_names


# ---------------------------------------------------------------------------
# write_harness_report tests
# ---------------------------------------------------------------------------


class TestWriteHarnessReport:
    """REQ-VERIFY-037: harness report artifact is written with expected fields."""

    def test_report_written_to_json(self, tmp_path: Path) -> None:
        """write_harness_report creates a valid JSON file at the given path."""
        out_path = tmp_path / "experiment_258_harness_report.json"
        throughput = ThroughputMeasurement()
        throughput.record_batch("Qwen3.5-0.8B", n_cases=10, elapsed_seconds=20.0)
        throughput.record_batch("Gemma4-E4B-it", n_cases=10, elapsed_seconds=50.0)
        write_harness_report(out_path, throughput=throughput, run_date="20260413")
        assert out_path.exists()
        data = json.loads(out_path.read_text())
        assert data["experiment"] == 258
        assert data["run_date"] == "20260413"
        assert "throughput" in data
        assert "Qwen3.5-0.8B" in data["throughput"]["per_model"]

    def test_report_contains_target_met_summary(self, tmp_path: Path) -> None:
        """Report includes overall target_met flag derived from per-model results."""
        out_path = tmp_path / "report.json"
        tm = ThroughputMeasurement()
        tm.record_batch("Qwen3.5-0.8B", n_cases=5, elapsed_seconds=10.0)  # 2 s/case -> met
        tm.record_batch("Gemma4-E4B-it", n_cases=5, elapsed_seconds=10.0)  # 2 s/case -> met
        write_harness_report(out_path, throughput=tm, run_date="20260413")
        data = json.loads(out_path.read_text())
        # All models met target → overall target_met should be True.
        assert data["target_met"] is True

    def test_report_overall_target_not_met_when_any_model_slow(self, tmp_path: Path) -> None:
        """Overall target_met is False when at least one model misses the threshold."""
        out_path = tmp_path / "report.json"
        tm = ThroughputMeasurement()
        tm.record_batch("Qwen3.5-0.8B", n_cases=5, elapsed_seconds=10.0)   # 2 s/case -> met
        tm.record_batch("Gemma4-E4B-it", n_cases=5, elapsed_seconds=100.0)  # 20 s/case -> not met
        write_harness_report(out_path, throughput=tm, run_date="20260413")
        data = json.loads(out_path.read_text())
        assert data["target_met"] is False

    def test_report_without_any_measurements_is_not_met(self, tmp_path: Path) -> None:
        """When no batches have been recorded, target_met defaults to False."""
        out_path = tmp_path / "report.json"
        write_harness_report(out_path, throughput=ThroughputMeasurement(), run_date="20260413")
        data = json.loads(out_path.read_text())
        assert data["target_met"] is False

    def test_report_creates_parent_directory(self, tmp_path: Path) -> None:
        """write_harness_report creates missing parent directories."""
        out_path = tmp_path / "nested" / "dir" / "report.json"
        write_harness_report(out_path, throughput=ThroughputMeasurement(), run_date="20260413")
        assert out_path.exists()
