"""Tests for Exp 614: ExclusionManifest DualGPU Validation.

100% targeted coverage on functions added in
scripts/experiment_614_exclusion_manifest_dualgpu.py:
  - run_precheck_and_time()
  - _forward_pass_on_device()
  - run_dualgpu_test()
  - run_experiment()

Tests run without GPU hardware by mocking torch and subprocess.

Spec: REQ-INFRA-087, REQ-INFRA-088, SCENARIO-INFRA-095, SCENARIO-INFRA-096
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "scripts"))

# Prevent GPU assertion from firing in CI (no live GPU in test environment).
os.environ["CARNOT_IS_CI"] = "1"
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import scripts.experiment_614_exclusion_manifest_dualgpu as exp614  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_completed_proc(returncode: int, stdout: str) -> MagicMock:
    """Return a mock subprocess.CompletedProcess with given returncode and stdout."""
    proc = MagicMock()
    proc.returncode = returncode
    proc.stdout = stdout
    return proc


# ---------------------------------------------------------------------------
# run_precheck_and_time
# ---------------------------------------------------------------------------


class TestRunPrecheckAndTime:
    """REQ-INFRA-087: sentinel age must be recorded immediately after precheck runs."""

    def test_precheck_ok_sentinel_fresh(self, tmp_path: Path) -> None:
        # SCENARIO-INFRA-095: precheck exits 0, sentinel written just-in-time.
        sentinel = tmp_path / "conductor_consulted_at.txt"
        sentinel.write_text("2026-04-21T01:00:00Z\n")

        proc_mock = _make_completed_proc(0, "[PRECHECK OK] conductor_consulted=True")

        with (
            patch.object(exp614, "_PRECHECK_PATH", Path(sys.executable)),
            patch.object(exp614, "_SENTINEL_PATH", sentinel),
            patch("subprocess.run", return_value=proc_mock),
            patch("time.time", return_value=os.path.getmtime(str(sentinel)) + 5),
        ):
            precheck_ok, age = exp614.run_precheck_and_time(614)

        assert precheck_ok is True
        # Age should be approximately 5 seconds (mocked).
        assert 0 < age < 60

    def test_precheck_excluded_exits_1(self, tmp_path: Path) -> None:
        # Precheck exits 1 for an excluded experiment — sentinel is NOT written.
        sentinel = tmp_path / "conductor_consulted_at.txt"
        # sentinel does NOT exist

        proc_mock = _make_completed_proc(1, "[EXCLUDED] Exp 308 ...")

        with (
            patch.object(exp614, "_PRECHECK_PATH", Path(sys.executable)),
            patch.object(exp614, "_SENTINEL_PATH", sentinel),
            patch("subprocess.run", return_value=proc_mock),
        ):
            precheck_ok, age = exp614.run_precheck_and_time(308)

        assert precheck_ok is False
        assert age == float("inf")

    def test_sentinel_missing_returns_inf(self, tmp_path: Path) -> None:
        # Precheck exits 0 but sentinel was not written (unexpected path).
        sentinel = tmp_path / "missing_sentinel.txt"

        proc_mock = _make_completed_proc(0, "[PRECHECK OK]")

        with (
            patch.object(exp614, "_PRECHECK_PATH", Path(sys.executable)),
            patch.object(exp614, "_SENTINEL_PATH", sentinel),
            patch("subprocess.run", return_value=proc_mock),
        ):
            precheck_ok, age = exp614.run_precheck_and_time(614)

        assert precheck_ok is True
        assert age == float("inf")

    def test_old_sentinel_returns_high_age(self, tmp_path: Path) -> None:
        # Sentinel exists but is hours old — age > 60.
        sentinel = tmp_path / "conductor_consulted_at.txt"
        sentinel.write_text("old\n")
        # Backdate the mtime by 3600 seconds.
        old_mtime = time.time() - 3600
        os.utime(str(sentinel), (old_mtime, old_mtime))

        proc_mock = _make_completed_proc(0, "[PRECHECK OK]")

        with (
            patch.object(exp614, "_PRECHECK_PATH", Path(sys.executable)),
            patch.object(exp614, "_SENTINEL_PATH", sentinel),
            patch("subprocess.run", return_value=proc_mock),
        ):
            precheck_ok, age = exp614.run_precheck_and_time(614)

        assert precheck_ok is True
        # Age should reflect the backdated mtime — well over 60.
        assert age > 60


# ---------------------------------------------------------------------------
# _forward_pass_on_device
# ---------------------------------------------------------------------------


class TestForwardPassOnDevice:
    """Unit tests for the thread worker function."""

    def test_success_stores_ok(self) -> None:
        # SCENARIO-INFRA-096 (unit): thread worker records 'ok' when forward pass completes.
        torch_mock = MagicMock()
        nn_mock = MagicMock()
        linear_instance = MagicMock()
        nn_mock.Linear.return_value = linear_instance
        linear_instance.to.return_value = linear_instance
        torch_mock.device.return_value = MagicMock()
        torch_mock.randn.return_value = MagicMock()

        results: dict[str, str] = {}
        with patch.dict(sys.modules, {"torch": torch_mock, "torch.nn": nn_mock}):
            exp614._forward_pass_on_device("cuda:0", results, "gpu0")

        assert results.get("gpu0") == "ok"

    def test_exception_stores_error_string(self) -> None:
        # _forward_pass_on_device must catch ALL exceptions and store them — never raise.
        results: dict[str, str] = {}
        # An invalid device string causes torch to raise RuntimeError.
        exp614._forward_pass_on_device("invalid::device::string::::", results, "key1")
        assert "key1" in results
        assert isinstance(results["key1"], str)
        assert results["key1"] != "ok"


# ---------------------------------------------------------------------------
# run_dualgpu_test
# ---------------------------------------------------------------------------


class TestRunDualGPUTest:
    """REQ-INFRA-088: DualGPU test must handle all hardware configurations."""

    def test_torch_unavailable(self) -> None:
        # REQ-INFRA-088-6: torch import fails -> cuda_unavailable.
        with patch.dict(sys.modules, {"torch": None}):
            n, confirmed, reason = exp614.run_dualgpu_test()
        assert n == 0
        assert confirmed is False
        assert reason == "cuda_unavailable"

    def test_cuda_not_available(self) -> None:
        # torch imports but cuda is not available.
        torch_mock = MagicMock()
        torch_mock.cuda.is_available.return_value = False
        with patch.dict(sys.modules, {"torch": torch_mock}):
            n, confirmed, reason = exp614.run_dualgpu_test()
        assert confirmed is False
        assert reason == "cuda_unavailable"

    def test_only_one_gpu(self) -> None:
        # REQ-INFRA-088-6: only one GPU -> only_one_gpu.
        torch_mock = MagicMock()
        torch_mock.cuda.is_available.return_value = True
        torch_mock.cuda.device_count.return_value = 1
        with patch.dict(sys.modules, {"torch": torch_mock}):
            n, confirmed, reason = exp614.run_dualgpu_test()
        assert n == 1
        assert confirmed is False
        assert reason == "only_one_gpu"

    def test_two_gpus_util_confirmed(self) -> None:
        # REQ-INFRA-088-5: two GPUs, forward passes succeed, utilization > 0.
        torch_mock = MagicMock()
        torch_mock.cuda.is_available.return_value = True
        torch_mock.cuda.device_count.return_value = 2
        torch_mock.cuda.utilization.return_value = 42  # > 0

        nn_mock = MagicMock()
        linear_instance = MagicMock()
        nn_mock.Linear.return_value = linear_instance
        linear_instance.to.return_value = linear_instance

        def fake_forward_pass(device_str: str, results_dict: dict, key: str) -> None:
            results_dict[key] = "ok"

        with (
            patch.dict(sys.modules, {"torch": torch_mock, "torch.nn": nn_mock}),
            patch.object(exp614, "_forward_pass_on_device", side_effect=fake_forward_pass),
        ):
            n, confirmed, reason = exp614.run_dualgpu_test()

        assert n == 2
        assert confirmed is True
        assert reason is None

    def test_two_gpus_util_zero(self) -> None:
        # Two GPUs but utilization reads 0 — not confirmed.
        torch_mock = MagicMock()
        torch_mock.cuda.is_available.return_value = True
        torch_mock.cuda.device_count.return_value = 2
        torch_mock.cuda.utilization.return_value = 0

        def fake_forward_pass(device_str: str, results_dict: dict, key: str) -> None:
            results_dict[key] = "ok"

        with (
            patch.dict(sys.modules, {"torch": torch_mock}),
            patch.object(exp614, "_forward_pass_on_device", side_effect=fake_forward_pass),
        ):
            n, confirmed, reason = exp614.run_dualgpu_test()

        assert n == 2
        assert confirmed is False
        assert reason is None

    def test_gpu1_forward_failed(self) -> None:
        # GPU1 forward pass raises — blocked reason includes the error.
        torch_mock = MagicMock()
        torch_mock.cuda.is_available.return_value = True
        torch_mock.cuda.device_count.return_value = 2

        def fake_forward_pass(device_str: str, results_dict: dict, key: str) -> None:
            if key == "gpu1":
                results_dict[key] = "CUDA error: device-side assert triggered"
            else:
                results_dict[key] = "ok"

        with (
            patch.dict(sys.modules, {"torch": torch_mock}),
            patch.object(exp614, "_forward_pass_on_device", side_effect=fake_forward_pass),
        ):
            n, confirmed, reason = exp614.run_dualgpu_test()

        assert n == 2
        assert confirmed is False
        assert reason is not None
        assert "gpu1_forward_failed" in reason

    def test_utilization_query_exception(self) -> None:
        # torch.cuda.utilization() raises unexpectedly.
        torch_mock = MagicMock()
        torch_mock.cuda.is_available.return_value = True
        torch_mock.cuda.device_count.return_value = 2
        torch_mock.cuda.utilization.side_effect = RuntimeError("driver error")

        def fake_forward_pass(device_str: str, results_dict: dict, key: str) -> None:
            results_dict[key] = "ok"

        with (
            patch.dict(sys.modules, {"torch": torch_mock}),
            patch.object(exp614, "_forward_pass_on_device", side_effect=fake_forward_pass),
        ):
            n, confirmed, reason = exp614.run_dualgpu_test()

        assert n == 2
        assert confirmed is False
        assert reason is not None
        assert "utilization_query_failed" in reason


# ---------------------------------------------------------------------------
# run_experiment (integration of both checks)
# ---------------------------------------------------------------------------


class TestRunExperiment:
    """REQ-INFRA-087+088: run_experiment must produce all required artifact fields."""

    def _patch_precheck(self, sentinel_age: float, precheck_ok: bool = True) -> Any:
        return patch.object(
            exp614,
            "run_precheck_and_time",
            return_value=(precheck_ok, sentinel_age),
        )

    def _patch_dualgpu(self, n: int, confirmed: bool, reason: str | None) -> Any:
        return patch.object(
            exp614,
            "run_dualgpu_test",
            return_value=(n, confirmed, reason),
        )

    def test_full_success_verdict(self) -> None:
        # Both timing and DualGPU confirmed -> precheck_timed_dualgpu_confirmed.
        with self._patch_precheck(5.0), self._patch_dualgpu(2, True, None):
            result = exp614.run_experiment()

        assert result["sentinel_within_60s"] is True
        assert result["retro_067_timing_confirmed"] is True
        assert result["gpu1_utilization_confirmed"] is True
        assert result["honest_verdict"] == "precheck_timed_dualgpu_confirmed"

    def test_timed_dualgpu_blocked_verdict(self) -> None:
        # Timing ok, but DualGPU blocked (only one GPU).
        with self._patch_precheck(10.0), self._patch_dualgpu(1, False, "only_one_gpu"):
            result = exp614.run_experiment()

        assert result["sentinel_within_60s"] is True
        assert result["gpu1_utilization_confirmed"] is False
        assert result["honest_verdict"] == "precheck_timed_dualgpu_blocked"
        assert result["dualgpu_blocked_reason"] == "only_one_gpu"

    def test_precheck_not_timed_verdict(self) -> None:
        # Sentinel is stale (age > 60).
        with self._patch_precheck(3600.0), self._patch_dualgpu(2, True, None):
            result = exp614.run_experiment()

        assert result["sentinel_within_60s"] is False
        assert result["retro_067_timing_confirmed"] is False
        assert result["honest_verdict"] == "precheck_not_timed_dualgpu_checked"

    def test_all_required_fields_present(self) -> None:
        # REQ-INFRA-087-4 + REQ-INFRA-088-7: all required fields in artifact.
        with self._patch_precheck(5.0), self._patch_dualgpu(2, True, None):
            result = exp614.run_experiment()

        required = [
            "sentinel_within_60s",
            "sentinel_age_seconds",
            "retro_067_timing_confirmed",
            "n_gpus_detected",
            "gpu1_utilization_confirmed",
            "dualgpu_blocked_reason",
            "honest_verdict",
        ]
        for field in required:
            assert field in result, f"Missing required field: {field}"

    def test_sentinel_age_recorded(self) -> None:
        # REQ-INFRA-087-1: sentinel_age_seconds must equal the value returned by precheck.
        with self._patch_precheck(42.7), self._patch_dualgpu(0, False, "cuda_unavailable"):
            result = exp614.run_experiment()

        assert result["sentinel_age_seconds"] == pytest.approx(42.7)

    def test_no_gpus_blocked(self) -> None:
        # cuda_unavailable path records n_gpus=0 and no reason for timing.
        with self._patch_precheck(5.0), self._patch_dualgpu(0, False, "cuda_unavailable"):
            result = exp614.run_experiment()

        assert result["n_gpus_detected"] == 0
        assert result["dualgpu_blocked_reason"] == "cuda_unavailable"
        assert result["honest_verdict"] == "precheck_timed_dualgpu_blocked"
