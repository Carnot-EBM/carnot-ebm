"""Tests for carnot.hardware.iron_runner — NPU environment probes and IRON inference.

REQ-HARDWARE-010: IRON installs via pip install mlir-aie
REQ-HARDWARE-011: IRONRunner runs JEPA ONNX on NPU, falls back to CPU
REQ-HARDWARE-012: honest_verdict distinguishes npu_executed from cpu_fallback

SCENARIO-HARDWARE-010: iron_available() returns False when mlir_aie not installed
SCENARIO-HARDWARE-011: IRONRunner falls back to CPU when NPU device absent
SCENARIO-HARDWARE-012: honest_verdict is cpu_baseline_only on no-NPU machines
"""

from __future__ import annotations

import importlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from carnot.hardware.iron_runner import IRONRunner, NPUEnvironment


# ---------------------------------------------------------------------------
# NPUEnvironment tests
# ---------------------------------------------------------------------------


class TestNPUEnvironmentIronAvailable:
    """SCENARIO-HARDWARE-010: iron_available() returns False when mlir_aie absent."""

    def test_iron_available_false_when_import_fails(self):
        # Simulate mlir_aie not installed by patching builtins.__import__
        import builtins
        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "mlir_aie":
                raise ImportError("No module named 'mlir_aie'")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=fake_import):
            # Re-execute the probe inside the patched context
            result = NPUEnvironment.iron_available()
        # In the real test environment mlir_aie is almost certainly absent, but
        # the patch guarantees the False path regardless.
        # We only assert False when import actually fails:
        # The patch above forces ImportError so the return must be False.
        assert result is False

    def test_iron_available_returns_bool(self):
        # Always returns a bool regardless of environment
        result = NPUEnvironment.iron_available()
        assert isinstance(result, bool)

    def test_iron_available_true_when_mlir_aie_importable(self):
        mock_module = MagicMock()
        with patch.dict(sys.modules, {"mlir_aie": mock_module}):
            # Force re-evaluation by calling directly
            import builtins
            real_import = builtins.__import__

            def fake_import(name, *args, **kwargs):
                if name == "mlir_aie":
                    return mock_module
                return real_import(name, *args, **kwargs)

            with patch("builtins.__import__", side_effect=fake_import):
                result = NPUEnvironment.iron_available()
        assert result is True


class TestNPUEnvironmentDevicePresent:
    """npu_device_present() checks /dev/accel0."""

    def test_npu_absent_on_most_machines(self, tmp_path):
        # On machines without NPU, /dev/accel0 does not exist.
        # We test the logic by patching the device path.
        with patch("carnot.hardware.iron_runner._NPU_DEVICE_PATH", str(tmp_path / "nonexistent")):
            assert NPUEnvironment.npu_device_present() is False

    def test_npu_present_when_device_exists(self, tmp_path):
        # Simulate device presence by creating a temp file at the path
        device_file = tmp_path / "accel0"
        device_file.touch()
        with patch("carnot.hardware.iron_runner._NPU_DEVICE_PATH", str(device_file)):
            assert NPUEnvironment.npu_device_present() is True

    def test_npu_device_present_returns_bool(self):
        result = NPUEnvironment.npu_device_present()
        assert isinstance(result, bool)


class TestNPUEnvironmentInstallIron:
    """install_iron() runs pip install mlir-aie and returns success bool."""

    def test_install_iron_returns_true_on_success(self):
        mock_result = MagicMock()
        mock_result.returncode = 0
        with patch("subprocess.run", return_value=mock_result) as mock_run:
            result = NPUEnvironment.install_iron()
        assert result is True
        # Verify it ran pip install mlir-aie
        args = mock_run.call_args[0][0]
        assert "mlir-aie" in args

    def test_install_iron_returns_false_on_failure(self):
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stderr = "ERROR: Could not find a version"
        with patch("subprocess.run", return_value=mock_result):
            result = NPUEnvironment.install_iron()
        assert result is False

    def test_install_iron_returns_false_on_exception(self):
        with patch("subprocess.run", side_effect=OSError("no pip")):
            result = NPUEnvironment.install_iron()
        assert result is False

    def test_install_iron_uses_current_python(self):
        mock_result = MagicMock()
        mock_result.returncode = 0
        with patch("subprocess.run", return_value=mock_result) as mock_run:
            NPUEnvironment.install_iron()
        args = mock_run.call_args[0][0]
        assert args[0] == sys.executable


# ---------------------------------------------------------------------------
# IRONRunner tests — CPU fallback path (no NPU hardware)
# ---------------------------------------------------------------------------


def _make_tiny_onnx(tmp_path: Path) -> str:
    """Create a minimal valid ONNX model (identity: output = input) for testing.

    Why a real ONNX model: onnxruntime.InferenceSession requires a parseable
    ONNX file.  A synthetic model lets tests run without needing the actual
    JEPA predictor file.
    """
    import onnx
    from onnx import TensorProto, helper

    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 4])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 4])
    node = helper.make_node("Identity", inputs=["X"], outputs=["Y"])
    graph = helper.make_graph([node], "tiny", [X], [Y])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 11)])
    path = str(tmp_path / "tiny.onnx")
    onnx.save(model, path)
    return path


class TestIRONRunnerCPUFallback:
    """SCENARIO-HARDWARE-011/012: IRONRunner falls back to CPU when NPU absent."""

    @pytest.fixture
    def tiny_model(self, tmp_path):
        return _make_tiny_onnx(tmp_path)

    @pytest.fixture
    def runner(self, tiny_model):
        return IRONRunner(tiny_model)

    def test_run_returns_outputs_on_cpu_fallback(self, runner):
        # NPU absent → CPU onnxruntime path
        with patch.object(NPUEnvironment, "iron_available", return_value=False):
            with patch.object(NPUEnvironment, "npu_device_present", return_value=False):
                inputs = {"X": np.zeros((1, 4), dtype=np.float32)}
                result = runner.run(inputs)
        assert "outputs" in result
        assert result["npu_executed"] is False
        assert result["fallback_reason"] == "iron_not_installed"

    def test_run_npu_device_absent_reason(self, runner):
        # IRON installed but no device
        with patch.object(NPUEnvironment, "iron_available", return_value=True):
            with patch.object(NPUEnvironment, "npu_device_present", return_value=False):
                inputs = {"X": np.zeros((1, 4), dtype=np.float32)}
                result = runner.run(inputs)
        assert result["npu_executed"] is False
        assert result["fallback_reason"] == "npu_device_absent"

    def test_benchmark_returns_float(self, runner):
        with patch.object(NPUEnvironment, "iron_available", return_value=False):
            with patch.object(NPUEnvironment, "npu_device_present", return_value=False):
                inputs = {"X": np.zeros((1, 4), dtype=np.float32)}
                latency = runner.benchmark(inputs, n_runs=3)
        assert isinstance(latency, float)
        assert latency > 0.0

    def test_benchmark_uses_n_runs_samples(self, runner):
        """Benchmark runs exactly n_runs inference calls."""
        call_count = []

        original_run = runner.run

        def counting_run(inputs):
            call_count.append(1)
            return original_run(inputs)

        with patch.object(NPUEnvironment, "iron_available", return_value=False):
            with patch.object(NPUEnvironment, "npu_device_present", return_value=False):
                with patch.object(runner, "run", side_effect=counting_run):
                    inputs = {"X": np.zeros((1, 4), dtype=np.float32)}
                    runner.benchmark(inputs, n_runs=5)

        assert len(call_count) == 5

    def test_run_npu_error_falls_back_to_cpu(self, runner):
        """When NPU compile/run raises, fallback_reason starts with 'npu_error'."""
        with patch.object(NPUEnvironment, "iron_available", return_value=True):
            with patch.object(NPUEnvironment, "npu_device_present", return_value=True):
                with patch.object(runner, "compile_onnx", side_effect=RuntimeError("compile fail")):
                    inputs = {"X": np.zeros((1, 4), dtype=np.float32)}
                    result = runner.run(inputs)
        assert result["npu_executed"] is False
        assert result["fallback_reason"].startswith("npu_error")

    def test_compile_onnx_raises_when_iron_absent(self, runner):
        """compile_onnx raises ImportError when mlir_aie is not installed."""
        import builtins
        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "mlir_aie":
                raise ImportError("No module named 'mlir_aie'")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=fake_import):
            with pytest.raises(ImportError):
                runner.compile_onnx(runner.model_path)

    def test_session_is_cached(self, runner):
        """_get_ort_session() returns the same object on repeated calls (lazy cache)."""
        with patch.object(NPUEnvironment, "iron_available", return_value=False):
            with patch.object(NPUEnvironment, "npu_device_present", return_value=False):
                inputs = {"X": np.zeros((1, 4), dtype=np.float32)}
                runner.run(inputs)
                s1 = runner._session
                runner.run(inputs)
                s2 = runner._session
        assert s1 is s2


class TestIRONRunnerNPUPath:
    """Test the NPU execution path when IRON is available."""

    @pytest.fixture
    def tiny_model(self, tmp_path):
        return _make_tiny_onnx(tmp_path)

    def test_run_uses_npu_when_available(self, tiny_model):
        mock_kernel = b"fake_kernel"
        mock_outputs = [np.zeros((1, 4), dtype=np.float32)]
        mock_mlir = MagicMock()
        mock_mlir.compile_onnx.return_value = mock_kernel
        mock_mlir.run_kernel.return_value = mock_outputs

        runner = IRONRunner(tiny_model)

        import builtins
        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "mlir_aie":
                return mock_mlir
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=fake_import):
            with patch.object(NPUEnvironment, "iron_available", return_value=True):
                with patch.object(NPUEnvironment, "npu_device_present", return_value=True):
                    with patch.object(runner, "compile_onnx", return_value=mock_kernel):
                        with patch.dict(sys.modules, {"mlir_aie": mock_mlir}):
                            inputs = {"X": np.zeros((1, 4), dtype=np.float32)}
                            result = runner.run(inputs)

        assert result["npu_executed"] is True
        assert result["fallback_reason"] is None
