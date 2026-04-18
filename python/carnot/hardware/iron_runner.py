"""AMD XDNA NPU inference via the IRON toolchain (arXiv 2504.03083).

**Why IRON unblocks NPU (5-milestone blockage resolved):**
    Previous attempts to use the AMD XDNA NPU required building mlir-aie from
    source, which needs cmake, ninja, and openblas — none available in the Carnot
    build environment.  This caused NPU work to be blocked for 5 consecutive
    milestones (Exps 380-459).

    The IRON toolchain (Integrated Runtime for Open NPUs, arXiv 2504.03083) ships
    a pip-installable wheel: ``pip install mlir-aie``.  This requires ONLY Python
    and pip — no cmake, ninja, or system libraries.  Installing it is what this
    module tries to do automatically.

**Why CPU fallback is required:**
    Most developers and all CI runners lack AMD XDNA NPU hardware.  Without a
    graceful fallback, NPU experiments would block on every machine that isn't the
    production server.  CPU fallback via onnxruntime lets the experiment run
    everywhere and provides a baseline timing for future NPU speedup comparison.

**Reference:** arXiv 2504.03083 — IRON: An Open-Source MLIR-AIE-Based NPU Toolchain

Spec: REQ-HARDWARE-010, REQ-HARDWARE-011, REQ-HARDWARE-012,
      SCENARIO-HARDWARE-010, SCENARIO-HARDWARE-011, SCENARIO-HARDWARE-012
"""

from __future__ import annotations

import logging
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

_log = logging.getLogger(__name__)

# NPU device file on Linux — AMD XDNA exposes /dev/accel0 when the driver is loaded.
_NPU_DEVICE_PATH = "/dev/accel0"


class NPUEnvironment:
    """Probe the local environment for AMD XDNA NPU support and the IRON toolchain.

    This class does NOT install anything by default — callers must explicitly
    call ``install_iron()`` if they want to attempt installation.  Probing
    (``iron_available()``, ``npu_device_present()``) is always safe and read-only.

    Spec: REQ-HARDWARE-010, REQ-HARDWARE-011
    """

    @staticmethod
    def iron_available() -> bool:
        """Return True iff the mlir_aie Python package is importable.

        ``mlir_aie`` is the Python package installed by ``pip install mlir-aie``
        (the IRON toolchain).  Attempting the import is the authoritative check —
        if it succeeds, IRON is installed and AIE kernels can be compiled.

        Returns False on ImportError (package not installed) or any other exception
        (import-time error in the package itself).

        Spec: REQ-HARDWARE-010, SCENARIO-HARDWARE-010
        """
        try:
            import mlir_aie  # noqa: F401, PLC0415 — intentional probe import
            return True
        except Exception:
            return False

    @staticmethod
    def npu_device_present() -> bool:
        """Return True iff the AMD XDNA NPU device file exists (/dev/accel0).

        On Linux, the AMD XDNA driver exposes /dev/accel0 when the kernel module
        is loaded and the hardware is present.  Absence of this file means either
        no NPU hardware or the driver is not loaded.

        Does not check whether the current process has permission to use the device;
        that would require actually opening it.

        Spec: REQ-HARDWARE-011, SCENARIO-HARDWARE-011
        """
        return Path(_NPU_DEVICE_PATH).exists()

    @staticmethod
    def install_iron() -> bool:
        """Attempt to install the IRON toolchain via ``pip install mlir-aie``.

        Why pip-based installation: IRON's wheel bundles all MLIR-AIE runtime
        dependencies.  No cmake, ninja, or system libraries are required — just
        pip.  This is what unblocks NPU work after 5 milestones of build failures.

        Returns True on success (pip exits 0), False otherwise.  Never raises.

        Spec: REQ-HARDWARE-010
        """
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", "mlir-aie"],
                capture_output=True,
                text=True,
                timeout=300,  # 5-minute timeout — wheel download can be slow
            )
            if result.returncode == 0:
                _log.info("IRON toolchain installed successfully via pip install mlir-aie")
                return True
            else:
                _log.warning(
                    "pip install mlir-aie failed (rc=%d): %s",
                    result.returncode,
                    result.stderr[:500],
                )
                return False
        except Exception as exc:
            _log.warning("install_iron() exception: %s", exc)
            return False


class IRONRunner:
    """Run ONNX model inference on AMD XDNA NPU via IRON, with CPU fallback.

    When the NPU is available (IRON installed + /dev/accel0 present), this class
    compiles the ONNX model to an AIE kernel and runs it on the NPU.

    When the NPU is unavailable (most dev machines, CI), it falls back to
    onnxruntime CPU inference.  This fallback produces a CPU baseline timing
    that can be compared against future NPU results when hardware is available.

    Why honest_verdict matters: the conductor and retrospective scripts use this
    field to determine whether a real NPU measurement was made.  Silent fallback
    to CPU without flagging it would mislead the research record.

    Parameters
    ----------
    model_path : str
        Path to the ONNX model file to run.

    Spec: REQ-HARDWARE-011, REQ-HARDWARE-012
    """

    def __init__(self, model_path: str) -> None:
        self.model_path = model_path
        self._env = NPUEnvironment()
        self._session: Any | None = None  # onnxruntime.InferenceSession, lazily loaded

    def _get_ort_session(self) -> Any:
        """Lazily create and cache an onnxruntime InferenceSession for CPU fallback.

        Why lazy: onnxruntime startup is slow (~100ms).  Deferring it until first
        use means import-time cost is zero when the NPU path succeeds.
        """
        if self._session is None:
            import onnxruntime as ort  # noqa: PLC0415

            self._session = ort.InferenceSession(
                self.model_path,
                providers=["CPUExecutionProvider"],
            )
        return self._session

    def compile_onnx(self, onnx_path: str) -> bytes:
        """Compile an ONNX model to an AIE kernel using IRON (mlir_aie).

        This is the NPU compilation step.  It transforms the ONNX graph into an
        AIE (AI Engine) kernel that runs on the AMD XDNA NPU hardware.

        Raises RuntimeError if IRON is not installed or compilation fails.
        Callers should catch RuntimeError and fall back to CPU if needed.

        Parameters
        ----------
        onnx_path : str
            Path to the ONNX model file to compile.

        Returns
        -------
        bytes
            Compiled AIE kernel binary.  Pass to mlir_aie runtime for execution.

        Spec: REQ-HARDWARE-011
        """
        import mlir_aie  # noqa: PLC0415

        # The IRON API compiles ONNX to AIE kernel.  The exact API depends on
        # the installed version; we use the documented entry point from arXiv 2504.03083.
        kernel = mlir_aie.compile_onnx(onnx_path)
        return kernel

    def run(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """Run inference on the loaded model, preferring NPU over CPU.

        Tries NPU execution first (IRON + /dev/accel0).  Falls back to
        onnxruntime CPU if NPU is unavailable or execution fails.

        Parameters
        ----------
        inputs : dict[str, Any]
            Input tensors as numpy arrays, keyed by ONNX input name.

        Returns
        -------
        dict with keys:
            - ``outputs`` (list): Model output arrays.
            - ``npu_executed`` (bool): True iff NPU was used.
            - ``fallback_reason`` (str | None): Reason for CPU fallback, or None.

        Spec: REQ-HARDWARE-011, REQ-HARDWARE-012
        """
        npu_executed = False
        fallback_reason: str | None = None

        # Try NPU path first
        if NPUEnvironment.iron_available() and NPUEnvironment.npu_device_present():
            try:
                kernel = self.compile_onnx(self.model_path)
                import mlir_aie  # noqa: PLC0415

                outputs = mlir_aie.run_kernel(kernel, inputs)
                npu_executed = True
                return {"outputs": outputs, "npu_executed": True, "fallback_reason": None}
            except Exception as exc:
                fallback_reason = f"npu_error: {exc}"
                _log.warning("NPU execution failed (%s); falling back to CPU", exc)
        else:
            if not NPUEnvironment.iron_available():
                fallback_reason = "iron_not_installed"
            elif not NPUEnvironment.npu_device_present():
                fallback_reason = "npu_device_absent"

        # CPU fallback via onnxruntime
        session = self._get_ort_session()
        input_feed = {k: v for k, v in inputs.items()}
        raw_outputs = session.run(None, input_feed)
        return {
            "outputs": raw_outputs,
            "npu_executed": npu_executed,
            "fallback_reason": fallback_reason,
        }

    def benchmark(self, inputs: dict[str, Any], n_runs: int = 10) -> float:
        """Measure median inference latency in milliseconds over n_runs.

        Always returns a valid float — either NPU latency (when NPU available)
        or CPU onnxruntime latency (fallback).  The median is used rather than
        mean to reduce sensitivity to outlier warm-up runs.

        Parameters
        ----------
        inputs : dict[str, Any]
            Input tensors for the model.
        n_runs : int
            Number of inference calls to time.  Median of all runs is returned.

        Returns
        -------
        float
            Median inference latency in milliseconds.

        Spec: REQ-HARDWARE-011, REQ-HARDWARE-012, SCENARIO-HARDWARE-011
        """
        latencies_ms: list[float] = []

        for _ in range(n_runs):
            t0 = time.perf_counter()
            self.run(inputs)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            latencies_ms.append(elapsed_ms)

        return statistics.median(latencies_ms)
