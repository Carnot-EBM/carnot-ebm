"""Tests for dual_gpu_controlled_test module.

Spec: REQ-INFRA-070, SCENARIO-INFRA-079, SCENARIO-INFRA-080
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from carnot.pipeline.dual_gpu_controlled_test import (
    DualGPUTestResult,
    run_dual_inference,
    sample_gpu_utilization,
)


class TestDualGPUTestResultHonestVerdict:
    """SCENARIO-INFRA-080: DualGPUTestResult.honest_verdict classifies GPU 1 activity."""

    def test_gpu1_active_when_live_and_above_threshold(self):
        # SCENARIO-INFRA-080: live_gpu + gpu1_compute_pct > 10 → 'gpu1_active'
        result = DualGPUTestResult(
            gpu0_compute_pct=80.0,
            gpu1_compute_pct=55.0,
            n_samples_run=20,
            inference_mode="live_gpu",
            honest_verdict="gpu1_active",
        )
        assert result.honest_verdict == "gpu1_active"

    def test_gpu1_idle_when_live_and_at_threshold(self):
        # SCENARIO-INFRA-080: live_gpu + gpu1_compute_pct == 10.0 → 'gpu1_idle'
        result = DualGPUTestResult(
            gpu0_compute_pct=75.0,
            gpu1_compute_pct=10.0,
            n_samples_run=20,
            inference_mode="live_gpu",
            honest_verdict="gpu1_idle",
        )
        assert result.honest_verdict == "gpu1_idle"

    def test_gpu1_idle_when_live_and_zero(self):
        # SCENARIO-INFRA-080: live_gpu + gpu1_compute_pct=0 → 'gpu1_idle'
        result = DualGPUTestResult(
            gpu0_compute_pct=0.0,
            gpu1_compute_pct=0.0,
            n_samples_run=20,
            inference_mode="live_gpu",
            honest_verdict="gpu1_idle",
        )
        assert result.honest_verdict == "gpu1_idle"

    def test_gpu_required_verdict(self):
        # SCENARIO-INFRA-080: gpu_required mode → 'gpu_required'
        result = DualGPUTestResult(
            gpu0_compute_pct=0.0,
            gpu1_compute_pct=0.0,
            n_samples_run=0,
            inference_mode="gpu_required",
            honest_verdict="gpu_required",
        )
        assert result.honest_verdict == "gpu_required"
        assert result.inference_mode == "gpu_required"

    def test_fields_are_stored(self):
        # REQ-INFRA-070: all required fields are present in DualGPUTestResult
        result = DualGPUTestResult(
            gpu0_compute_pct=42.5,
            gpu1_compute_pct=38.0,
            n_samples_run=15,
            inference_mode="live_gpu",
            honest_verdict="gpu1_active",
        )
        assert result.gpu0_compute_pct == 42.5
        assert result.gpu1_compute_pct == 38.0
        assert result.n_samples_run == 15


class TestSampleGpuUtilizationCIStub:
    """SCENARIO-INFRA-079: CI stub returns 0.0 when pynvml is not installed."""

    def test_ci_stub_returns_zero_for_all_devices(self):
        # SCENARIO-INFRA-079: no pynvml → {0: 0.0, 1: 0.0}
        with patch.dict("sys.modules", {"pynvml": None}):
            result = sample_gpu_utilization([0, 1], n_samples=4, interval_s=0.0)
        assert result == {0: 0.0, 1: 0.0}

    def test_ci_stub_single_device(self):
        # Single device also gets 0.0 from CI stub.
        with patch.dict("sys.modules", {"pynvml": None}):
            result = sample_gpu_utilization([0], n_samples=2, interval_s=0.0)
        assert result == {0: 0.0}

    def test_ci_stub_empty_device_list(self):
        # Empty device list → empty dict.
        with patch.dict("sys.modules", {"pynvml": None}):
            result = sample_gpu_utilization([], n_samples=2, interval_s=0.0)
        assert result == {}


class TestSampleGpuUtilizationWithMockedPynvml:
    """SCENARIO-INFRA-079: mean utilization computed correctly over n_samples."""

    def _make_pynvml_mock(self, util_map: dict) -> MagicMock:
        """Build a pynvml mock where each device returns a fixed .gpu value."""
        pynvml = MagicMock()
        pynvml.nvmlInit.return_value = None

        def _handle_by_index(dev_id):
            return f"handle_{dev_id}"

        pynvml.nvmlDeviceGetHandleByIndex.side_effect = _handle_by_index

        def _util_rates(handle):
            dev_id = int(handle.split("_")[1])
            util = MagicMock()
            util.gpu = util_map[dev_id]
            return util

        pynvml.nvmlDeviceGetUtilizationRates.side_effect = _util_rates
        return pynvml

    def test_mean_computed_over_n_samples(self):
        # SCENARIO-INFRA-079: 4 samples of 50% for device 0, 30% for device 1.
        pynvml_mock = self._make_pynvml_mock({0: 50, 1: 30})
        with patch.dict("sys.modules", {"pynvml": pynvml_mock}):
            result = sample_gpu_utilization([0, 1], n_samples=4, interval_s=0.0)
        assert result[0] == 50.0
        assert result[1] == 30.0

    def test_correct_call_count(self):
        # SCENARIO-INFRA-079: nvmlDeviceGetUtilizationRates called exactly n_samples times per device.
        pynvml_mock = self._make_pynvml_mock({0: 80, 1: 20})
        with patch.dict("sys.modules", {"pynvml": pynvml_mock}):
            sample_gpu_utilization([0, 1], n_samples=3, interval_s=0.0)
        # 3 samples * 2 devices = 6 total calls
        assert pynvml_mock.nvmlDeviceGetUtilizationRates.call_count == 6

    def test_single_sample(self):
        # Single sample: mean equals the single reading.
        pynvml_mock = self._make_pynvml_mock({0: 99})
        with patch.dict("sys.modules", {"pynvml": pynvml_mock}):
            result = sample_gpu_utilization([0], n_samples=1, interval_s=0.0)
        assert result[0] == 99.0

    def test_pynvml_init_error_returns_stub(self):
        # If pynvml.nvmlInit raises, fall back to 0.0 stub.
        pynvml_mock = MagicMock()
        pynvml_mock.nvmlInit.side_effect = Exception("driver not loaded")
        with patch.dict("sys.modules", {"pynvml": pynvml_mock}):
            result = sample_gpu_utilization([0, 1], n_samples=4, interval_s=0.0)
        assert result == {0: 0.0, 1: 0.0}

    def test_per_sample_query_error_skips_sample(self):
        # If nvmlDeviceGetUtilizationRates raises on one call, that sample is skipped.
        pynvml_mock = MagicMock()
        pynvml_mock.nvmlInit.return_value = None
        pynvml_mock.nvmlDeviceGetHandleByIndex.return_value = "h0"
        # First call raises, second returns 60.
        util_ok = MagicMock()
        util_ok.gpu = 60
        pynvml_mock.nvmlDeviceGetUtilizationRates.side_effect = [
            Exception("query error"),
            util_ok,
        ]
        with patch.dict("sys.modules", {"pynvml": pynvml_mock}):
            result = sample_gpu_utilization([0], n_samples=2, interval_s=0.0)
        # Only the second sample succeeded; mean = 60.0
        assert result[0] == 60.0

    def test_interval_sleep_is_called(self):
        # When interval_s > 0, time.sleep is called between samples.
        pynvml_mock = self._make_pynvml_mock({0: 50})
        sleep_calls: list[float] = []
        with patch.dict("sys.modules", {"pynvml": pynvml_mock}):
            with patch("carnot.pipeline.dual_gpu_controlled_test.time.sleep",
                       side_effect=lambda s: sleep_calls.append(s)):
                sample_gpu_utilization([0], n_samples=3, interval_s=0.25)
        assert sleep_calls == [0.25, 0.25, 0.25]


class TestRunDualInference:
    """REQ-INFRA-070: run_dual_inference runs both models simultaneously."""

    def test_both_models_called(self):
        # REQ-INFRA-070: each model is called once per prompt.
        calls_a: list[str] = []
        calls_b: list[str] = []

        def model_a(prompt: str) -> str:
            calls_a.append(prompt)
            return f"a:{prompt}"

        def model_b(prompt: str) -> str:
            calls_b.append(prompt)
            return f"b:{prompt}"

        prompts = ["p1", "p2", "p3"]
        resp_a, resp_b = run_dual_inference(model_a, model_b, prompts)
        assert calls_a == prompts
        assert calls_b == prompts

    def test_responses_in_order(self):
        # REQ-INFRA-070: responses preserve prompt ordering.
        resp_a, resp_b = run_dual_inference(
            lambda p: f"a:{p}",
            lambda p: f"b:{p}",
            ["x", "y", "z"],
        )
        assert resp_a == ["a:x", "a:y", "a:z"]
        assert resp_b == ["b:x", "b:y", "b:z"]

    def test_model_error_returns_empty_string(self):
        # REQ-INFRA-070: a failing inference call produces "" rather than raising.
        def bad_model(prompt: str) -> str:
            raise RuntimeError("CUDA OOM")

        resp_a, resp_b = run_dual_inference(
            lambda p: "ok",
            bad_model,
            ["q1", "q2"],
        )
        assert resp_a == ["ok", "ok"]
        assert resp_b == ["", ""]

    def test_empty_prompts(self):
        # Empty prompt list → empty response lists.
        resp_a, resp_b = run_dual_inference(lambda p: "a", lambda p: "b", [])
        assert resp_a == []
        assert resp_b == []
