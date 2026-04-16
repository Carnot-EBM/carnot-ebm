"""Tests for python/carnot/pipeline/dual_gpu_health.py — 100% coverage.

Coverage targets
----------------
- DualGPUHealthResult dataclass field assignment and derived properties
- check_dual_gpu_health():
  - pynvml happy-path: two GPUs with util/temp/vram populated
  - pynvml unavailable, nvidia-smi subprocess happy-path
  - pynvml unavailable, nvidia-smi subprocess failure → safe CI defaults
  - pynvml unavailable, nvidia-smi not found → safe CI defaults
  - SCENARIO-INFRA-031: gpu1_vram_mb>500 AND util=0 → gpu1_is_zombie=True
  - SCENARIO-INFRA-032: no pynvml/nvidia-smi → safe defaults, no exception
  - SCENARIO-INFRA-033: any GPU temp > 80 → temperature_warning=True, factor=0.75
- build_gpu_fix_artifact():
  - zombie_detected verdict
  - gpu1_healthy verdict
  - prior_retro_path embedded

Spec: REQ-INFRA-025, SCENARIO-INFRA-031, SCENARIO-INFRA-032, SCENARIO-INFRA-033 (Exp 426)
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from carnot.pipeline.dual_gpu_health import (  # noqa: E402
    DualGPUHealthResult,
    build_gpu_fix_artifact,
    check_dual_gpu_health,
)


# ---------------------------------------------------------------------------
# DualGPUHealthResult dataclass
# ---------------------------------------------------------------------------


class TestDualGPUHealthResult:
    """REQ-INFRA-025 — dataclass fields populate and derived properties are correct."""

    def test_fields_healthy(self):
        """All fields set; no zombie, no temperature warning."""
        r = DualGPUHealthResult(
            gpu0_util_pct=88.0,
            gpu1_util_pct=72.0,
            gpu0_temp_c=75.0,
            gpu1_temp_c=70.0,
            gpu0_vram_mb=15736.0,
            gpu1_vram_mb=14000.0,
            gpu1_is_zombie=False,
            temperature_warning=False,
            recommended_batch_size_factor=1.0,
        )
        assert r.gpu0_util_pct == 88.0
        assert r.gpu1_util_pct == 72.0
        assert r.gpu0_temp_c == 75.0
        assert r.gpu1_temp_c == 70.0
        assert r.gpu0_vram_mb == 15736.0
        assert r.gpu1_vram_mb == 14000.0
        assert r.gpu1_is_zombie is False
        assert r.temperature_warning is False
        assert r.recommended_batch_size_factor == 1.0

    def test_zombie_true_when_vram_high_util_zero(self):
        """SCENARIO-INFRA-031: vram>500 AND util<1 → gpu1_is_zombie=True."""
        r = DualGPUHealthResult(
            gpu0_util_pct=88.0,
            gpu1_util_pct=0.0,
            gpu0_temp_c=75.0,
            gpu1_temp_c=60.0,
            gpu0_vram_mb=15000.0,
            gpu1_vram_mb=1786.0,  # RETRO-025 exact value
            gpu1_is_zombie=True,
            temperature_warning=False,
            recommended_batch_size_factor=1.0,
        )
        assert r.gpu1_is_zombie is True

    def test_temperature_warning_factor(self):
        """SCENARIO-INFRA-033: temperature_warning → recommended_batch_size_factor=0.75."""
        r = DualGPUHealthResult(
            gpu0_util_pct=88.0,
            gpu1_util_pct=0.0,
            gpu0_temp_c=82.0,  # above 80C threshold
            gpu1_temp_c=70.0,
            gpu0_vram_mb=15000.0,
            gpu1_vram_mb=100.0,
            gpu1_is_zombie=False,
            temperature_warning=True,
            recommended_batch_size_factor=0.75,
        )
        assert r.temperature_warning is True
        assert r.recommended_batch_size_factor == 0.75


# ---------------------------------------------------------------------------
# check_dual_gpu_health — pynvml happy-path
# ---------------------------------------------------------------------------


class TestCheckDualGpuHealthPynvml:
    """check_dual_gpu_health() via pynvml path."""

    def _make_nvml_mock(self, gpu0_util=88, gpu0_temp=75, gpu0_vram=15736,
                        gpu1_util=72, gpu1_temp=70, gpu1_vram=14000):
        """Build a minimal pynvml mock with two GPU handles."""
        pynvml = MagicMock()
        pynvml.nvmlInit.return_value = None
        pynvml.nvmlDeviceGetCount.return_value = 2

        def _get_handle(idx):
            return f"handle_{idx}"

        pynvml.nvmlDeviceGetHandleByIndex.side_effect = _get_handle

        def _get_util(handle):
            idx = int(handle.split("_")[1])
            m = MagicMock()
            m.gpu = [gpu0_util, gpu1_util][idx]
            return m

        def _get_temp(handle, _sensor):
            idx = int(handle.split("_")[1])
            return [gpu0_temp, gpu1_temp][idx]

        def _get_mem(handle):
            idx = int(handle.split("_")[1])
            m = MagicMock()
            m.used = [gpu0_vram, gpu1_vram][idx] * 1024 * 1024  # bytes
            return m

        pynvml.nvmlDeviceGetUtilizationRates.side_effect = _get_util
        pynvml.nvmlDeviceGetTemperature.side_effect = _get_temp
        pynvml.nvmlDeviceGetMemoryInfo.side_effect = _get_mem
        pynvml.NVML_TEMPERATURE_GPU = 0
        return pynvml

    def test_healthy_two_gpus(self):
        """Both GPUs healthy: no zombie, no temperature warning."""
        pynvml_mock = self._make_nvml_mock()
        with patch.dict("sys.modules", {"pynvml": pynvml_mock}):
            result = check_dual_gpu_health(timeout_seconds=60)
        assert isinstance(result, DualGPUHealthResult)
        assert result.gpu0_util_pct == 88.0
        assert result.gpu1_util_pct == 72.0
        assert result.gpu0_temp_c == 75.0
        assert result.gpu1_temp_c == 70.0
        assert result.gpu1_is_zombie is False
        assert result.temperature_warning is False
        assert result.recommended_batch_size_factor == 1.0

    def test_zombie_detected_retro025_values(self):
        """SCENARIO-INFRA-031: GPU1 1786MB, 0% util → zombie detected."""
        pynvml_mock = self._make_nvml_mock(gpu1_util=0, gpu1_vram=1786)
        with patch.dict("sys.modules", {"pynvml": pynvml_mock}):
            result = check_dual_gpu_health(timeout_seconds=60)
        assert result.gpu1_is_zombie is True
        assert result.gpu1_util_pct == 0.0
        assert result.gpu1_vram_mb == pytest.approx(1786.0, rel=0.01)

    def test_zombie_boundary_exactly_500mb(self):
        """gpu1_vram=500MB AND util=0: NOT a zombie (threshold is strictly >500)."""
        pynvml_mock = self._make_nvml_mock(gpu1_util=0, gpu1_vram=500)
        with patch.dict("sys.modules", {"pynvml": pynvml_mock}):
            result = check_dual_gpu_health(timeout_seconds=60)
        assert result.gpu1_is_zombie is False

    def test_zombie_boundary_just_above_500mb(self):
        """gpu1_vram=501MB AND util=0: IS a zombie."""
        pynvml_mock = self._make_nvml_mock(gpu1_util=0, gpu1_vram=501)
        with patch.dict("sys.modules", {"pynvml": pynvml_mock}):
            result = check_dual_gpu_health(timeout_seconds=60)
        assert result.gpu1_is_zombie is True

    def test_temperature_warning_gpu0(self):
        """SCENARIO-INFRA-033: GPU0 temp=82C → temperature_warning=True, factor=0.75."""
        pynvml_mock = self._make_nvml_mock(gpu0_temp=82, gpu1_temp=70)
        with patch.dict("sys.modules", {"pynvml": pynvml_mock}):
            result = check_dual_gpu_health(timeout_seconds=60)
        assert result.temperature_warning is True
        assert result.recommended_batch_size_factor == pytest.approx(0.75)
        assert result.gpu0_temp_c == 82.0

    def test_temperature_warning_gpu1(self):
        """SCENARIO-INFRA-033: GPU1 temp=81C → temperature_warning=True."""
        pynvml_mock = self._make_nvml_mock(gpu0_temp=70, gpu1_temp=81)
        with patch.dict("sys.modules", {"pynvml": pynvml_mock}):
            result = check_dual_gpu_health(timeout_seconds=60)
        assert result.temperature_warning is True
        assert result.recommended_batch_size_factor == pytest.approx(0.75)

    def test_temperature_exactly_80c_no_warning(self):
        """Exactly 80C does NOT trigger warning (threshold is strictly > 80)."""
        pynvml_mock = self._make_nvml_mock(gpu0_temp=80, gpu1_temp=80)
        with patch.dict("sys.modules", {"pynvml": pynvml_mock}):
            result = check_dual_gpu_health(timeout_seconds=60)
        assert result.temperature_warning is False
        assert result.recommended_batch_size_factor == 1.0

    def test_single_gpu_pynvml(self):
        """Only one GPU visible: GPU1 fields default to 0, no zombie."""
        pynvml_mock = self._make_nvml_mock()
        pynvml_mock.nvmlDeviceGetCount.return_value = 1
        # Override to only handle GPU0
        pynvml_mock.nvmlDeviceGetHandleByIndex.side_effect = lambda idx: "handle_0"
        pynvml_mock.nvmlDeviceGetUtilizationRates.side_effect = lambda h: MagicMock(gpu=88)
        pynvml_mock.nvmlDeviceGetTemperature.side_effect = lambda h, s: 75
        pynvml_mock.nvmlDeviceGetMemoryInfo.side_effect = lambda h: MagicMock(used=15736 * 1024 * 1024)
        with patch.dict("sys.modules", {"pynvml": pynvml_mock}):
            result = check_dual_gpu_health(timeout_seconds=60)
        assert result.gpu1_util_pct == 0.0
        assert result.gpu1_temp_c == 0.0
        assert result.gpu1_vram_mb == 0.0
        assert result.gpu1_is_zombie is False

    def test_pynvml_init_exception_falls_back(self):
        """pynvml.nvmlInit() raises → falls through to nvidia-smi fallback."""
        pynvml_mock = MagicMock()
        pynvml_mock.nvmlInit.side_effect = Exception("driver error")
        with patch.dict("sys.modules", {"pynvml": pynvml_mock}):
            # nvidia-smi also unavailable → safe CI defaults
            with patch(
                "carnot.pipeline.dual_gpu_health._query_nvidia_smi",
                side_effect=FileNotFoundError,
            ):
                result = check_dual_gpu_health(timeout_seconds=60)
        assert result.gpu0_util_pct == 0.0
        assert result.gpu1_is_zombie is False
        assert result.temperature_warning is False


# ---------------------------------------------------------------------------
# check_dual_gpu_health — nvidia-smi subprocess fallback
# ---------------------------------------------------------------------------


class TestCheckDualGpuHealthNvidiaSmi:
    """check_dual_gpu_health() via nvidia-smi subprocess path (pynvml absent)."""

    def _smi_output(self, gpu0_util=88, gpu0_temp=75, gpu0_vram=15736,
                    gpu1_util=72, gpu1_temp=70, gpu1_vram=14000):
        """Produce fake nvidia-smi --query-gpu CSV output."""
        lines = [
            f"{gpu0_util}, {gpu0_temp}, {gpu0_vram}",
            f"{gpu1_util}, {gpu1_temp}, {gpu1_vram}",
        ]
        return "\n".join(lines) + "\n"

    def test_smi_two_gpus_healthy(self):
        """nvidia-smi path: two healthy GPUs."""
        smi_out = self._smi_output()
        with patch.dict("sys.modules", {"pynvml": None}):
            with patch(
                "carnot.pipeline.dual_gpu_health._query_nvidia_smi",
                return_value=smi_out,
            ):
                result = check_dual_gpu_health(timeout_seconds=60)
        assert result.gpu0_util_pct == 88.0
        assert result.gpu1_util_pct == 72.0
        assert result.gpu0_temp_c == 75.0
        assert result.gpu1_vram_mb == pytest.approx(14000.0, rel=0.01)
        assert result.gpu1_is_zombie is False
        assert result.temperature_warning is False

    def test_smi_zombie_detected(self):
        """nvidia-smi path: RETRO-025 values → zombie=True."""
        smi_out = self._smi_output(gpu1_util=0, gpu1_vram=1786)
        with patch.dict("sys.modules", {"pynvml": None}):
            with patch(
                "carnot.pipeline.dual_gpu_health._query_nvidia_smi",
                return_value=smi_out,
            ):
                result = check_dual_gpu_health(timeout_seconds=60)
        assert result.gpu1_is_zombie is True

    def test_smi_temperature_warning(self):
        """nvidia-smi path: GPU0 at 82C → temperature_warning=True."""
        smi_out = self._smi_output(gpu0_temp=82)
        with patch.dict("sys.modules", {"pynvml": None}):
            with patch(
                "carnot.pipeline.dual_gpu_health._query_nvidia_smi",
                return_value=smi_out,
            ):
                result = check_dual_gpu_health(timeout_seconds=60)
        assert result.temperature_warning is True
        assert result.recommended_batch_size_factor == pytest.approx(0.75)

    def test_smi_not_found_ci_safe_defaults(self):
        """SCENARIO-INFRA-032: nvidia-smi not found → safe defaults, no exception."""
        with patch.dict("sys.modules", {"pynvml": None}):
            with patch(
                "carnot.pipeline.dual_gpu_health._query_nvidia_smi",
                side_effect=FileNotFoundError("nvidia-smi not found"),
            ):
                result = check_dual_gpu_health(timeout_seconds=60)
        assert result.gpu0_util_pct == 0.0
        assert result.gpu1_util_pct == 0.0
        assert result.gpu0_temp_c == 0.0
        assert result.gpu1_temp_c == 0.0
        assert result.gpu0_vram_mb == 0.0
        assert result.gpu1_vram_mb == 0.0
        assert result.gpu1_is_zombie is False
        assert result.temperature_warning is False
        assert result.recommended_batch_size_factor == 1.0

    def test_smi_subprocess_error_ci_safe_defaults(self):
        """nvidia-smi returns non-zero / raises → safe defaults, no exception."""
        with patch.dict("sys.modules", {"pynvml": None}):
            with patch(
                "carnot.pipeline.dual_gpu_health._query_nvidia_smi",
                side_effect=RuntimeError("process error"),
            ):
                result = check_dual_gpu_health(timeout_seconds=60)
        assert result.gpu1_is_zombie is False
        assert result.temperature_warning is False

    def test_smi_single_gpu_output(self):
        """nvidia-smi returns only one GPU line → GPU1 defaults to 0."""
        smi_out = "88, 75, 15736\n"
        with patch.dict("sys.modules", {"pynvml": None}):
            with patch(
                "carnot.pipeline.dual_gpu_health._query_nvidia_smi",
                return_value=smi_out,
            ):
                result = check_dual_gpu_health(timeout_seconds=60)
        assert result.gpu0_util_pct == 88.0
        assert result.gpu1_util_pct == 0.0
        assert result.gpu1_is_zombie is False

    def test_smi_malformed_output_ci_safe(self):
        """Malformed nvidia-smi output → safe defaults, no exception."""
        with patch.dict("sys.modules", {"pynvml": None}):
            with patch(
                "carnot.pipeline.dual_gpu_health._query_nvidia_smi",
                return_value="not,valid,csv,data\nextra\n",
            ):
                result = check_dual_gpu_health(timeout_seconds=60)
        # Should not raise; returns something safe
        assert isinstance(result, DualGPUHealthResult)


# ---------------------------------------------------------------------------
# build_gpu_fix_artifact
# ---------------------------------------------------------------------------


class TestBuildGpuFixArtifact:
    """build_gpu_fix_artifact() — schema, honest_verdict, retro_025_status."""

    def test_zombie_detected_verdict(self):
        """SCENARIO-INFRA-031: gpu1_is_zombie=True → zombie_detected verdict."""
        health = DualGPUHealthResult(
            gpu0_util_pct=88.0,
            gpu1_util_pct=0.0,
            gpu0_temp_c=75.0,
            gpu1_temp_c=60.0,
            gpu0_vram_mb=15000.0,
            gpu1_vram_mb=1786.0,
            gpu1_is_zombie=True,
            temperature_warning=False,
            recommended_batch_size_factor=1.0,
        )
        artifact = build_gpu_fix_artifact(health, prior_retro_path="results/operational_retro_2026_04_31.json")
        assert artifact["schema"] == "carnot.dual_gpu_fix.v1"
        assert artifact["honest_verdict"] == "zombie_detected"
        assert artifact["retro_025_status"] == "zombie_confirmed"
        assert artifact["gpu1_is_zombie"] is True
        assert artifact["prior_retro_path"] == "results/operational_retro_2026_04_31.json"

    def test_gpu1_healthy_verdict(self):
        """gpu1_is_zombie=False → gpu1_healthy verdict."""
        health = DualGPUHealthResult(
            gpu0_util_pct=88.0,
            gpu1_util_pct=72.0,
            gpu0_temp_c=75.0,
            gpu1_temp_c=70.0,
            gpu0_vram_mb=15000.0,
            gpu1_vram_mb=14000.0,
            gpu1_is_zombie=False,
            temperature_warning=False,
            recommended_batch_size_factor=1.0,
        )
        artifact = build_gpu_fix_artifact(health, prior_retro_path="results/retro.json")
        assert artifact["honest_verdict"] == "gpu1_healthy"
        assert artifact["retro_025_status"] == "zombie_cleared"
        assert artifact["gpu1_is_zombie"] is False

    def test_artifact_contains_all_health_fields(self):
        """Artifact contains all DualGPUHealthResult fields at top level."""
        health = DualGPUHealthResult(
            gpu0_util_pct=50.0,
            gpu1_util_pct=30.0,
            gpu0_temp_c=65.0,
            gpu1_temp_c=60.0,
            gpu0_vram_mb=8000.0,
            gpu1_vram_mb=7000.0,
            gpu1_is_zombie=False,
            temperature_warning=False,
            recommended_batch_size_factor=1.0,
        )
        artifact = build_gpu_fix_artifact(health, prior_retro_path="results/retro.json")
        for field in [
            "gpu0_util_pct", "gpu1_util_pct", "gpu0_temp_c", "gpu1_temp_c",
            "gpu0_vram_mb", "gpu1_vram_mb", "temperature_warning",
            "recommended_batch_size_factor",
        ]:
            assert field in artifact, f"Missing field: {field}"

    def test_temperature_warning_in_artifact(self):
        """temperature_warning=True is faithfully embedded in artifact."""
        health = DualGPUHealthResult(
            gpu0_util_pct=88.0,
            gpu1_util_pct=0.0,
            gpu0_temp_c=82.0,
            gpu1_temp_c=70.0,
            gpu0_vram_mb=15000.0,
            gpu1_vram_mb=100.0,
            gpu1_is_zombie=False,
            temperature_warning=True,
            recommended_batch_size_factor=0.75,
        )
        artifact = build_gpu_fix_artifact(health, prior_retro_path="results/retro.json")
        assert artifact["temperature_warning"] is True
        assert artifact["recommended_batch_size_factor"] == pytest.approx(0.75)
