#!/usr/bin/env python3
"""Exp 460: AMD XDNA IRON NPU Unblock — pip-only mlir-aie install + JEPA ONNX benchmark.

**Background (5-milestone NPU blockage):**
    Exps 380-459 all blocked on NPU work because building mlir-aie from source
    requires cmake, ninja, and openblas — none available in the Carnot env.

    The IRON toolchain (arXiv 2504.03083) ships as a pip wheel requiring ONLY
    Python and pip.  This experiment is the first attempt to install it and
    run the JEPA predictor ONNX model on the AMD XDNA NPU.

**Expected outcomes (machine-dependent):**
    1. NPU machine (AMD XDNA + driver + IRON wheel available):
       honest_verdict='npu_executed', npu_ms=<float>, speedup=npu_ms/cpu_ms
    2. Dev machine (no /dev/accel0 or IRON not on PyPI):
       honest_verdict='cpu_baseline_only', cpu_ms=<float>, npu_ms=null
    3. pip install fails entirely:
       honest_verdict='install_failed', both ms fields null

    All three outcomes produce a valid, fully-populated artifact so the conductor
    can proceed regardless of hardware.

Deliverable: results/experiment_460_npu_iron.json

Spec: REQ-HARDWARE-010, REQ-HARDWARE-011, REQ-HARDWARE-012
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

# Ensure repo root is on sys.path so scripts/ and python/ are importable.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_REPO_ROOT / "python") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "python"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
_log = logging.getLogger("exp460")

# ---------------------------------------------------------------------------
# Step 1: apply_env_autofix FIRST (REQ-INFRA-021)
# ---------------------------------------------------------------------------
from carnot.pipeline.env_autofix import apply_env_autofix  # noqa: E402

_autofix = apply_env_autofix()
_log.info("env_autofix: gpu_detected=%s, auto_fix_applied=%s", _autofix.gpu_detected, _autofix.auto_fix_applied)

# ---------------------------------------------------------------------------
# Step 2: ExperimentTemplate setup
# ---------------------------------------------------------------------------
from scripts.experiment_template import ExperimentTemplate  # noqa: E402

DELIVERABLE = "results/experiment_460_npu_iron.json"
tmpl = ExperimentTemplate(
    460,
    "AMD XDNA IRON NPU Unblock",
    DELIVERABLE,
    requires_gpu=False,  # NPU is separate from GPU; no CUDA required
)
tmpl.setup()

from carnot.hardware.iron_runner import IRONRunner, NPUEnvironment  # noqa: E402

# ---------------------------------------------------------------------------
# Step 3: Attempt pip install mlir-aie
# ---------------------------------------------------------------------------
_log.info("Attempting pip install mlir-aie (IRON toolchain, arXiv 2504.03083)...")
iron_installed = NPUEnvironment.install_iron()
_log.info("iron_installed=%s", iron_installed)

# After install attempt, re-probe whether the package is now importable.
iron_available = NPUEnvironment.iron_available()
npu_present = NPUEnvironment.npu_device_present()

_log.info("iron_available=%s, npu_present=%s", iron_available, npu_present)

# ---------------------------------------------------------------------------
# Step 4: Locate or create an ONNX model for benchmarking
# ---------------------------------------------------------------------------
JEPA_ONNX = _REPO_ROOT / "results" / "jepa_predictor_291.onnx"
using_synthetic = False

if JEPA_ONNX.exists():
    onnx_path = str(JEPA_ONNX)
    _log.info("Using JEPA ONNX model: %s", onnx_path)
else:
    _log.warning("JEPA ONNX model not found at %s; creating synthetic model for baseline", JEPA_ONNX)
    using_synthetic = True
    import tempfile
    import onnx
    from onnx import TensorProto, helper

    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 64])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 64])
    node = helper.make_node("Identity", inputs=["X"], outputs=["Y"])
    graph = helper.make_graph([node], "synthetic_jepa", [X], [Y])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 11)])
    _syn_dir = _REPO_ROOT / "results"
    _syn_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = str(_syn_dir / "synthetic_460.onnx")
    onnx.save(model, onnx_path)
    _log.info("Synthetic ONNX written: %s", onnx_path)

# ---------------------------------------------------------------------------
# Step 5: Benchmark — NPU if available, CPU fallback otherwise
# ---------------------------------------------------------------------------
import numpy as np  # noqa: E402

runner = IRONRunner(onnx_path)

# Determine input shape from the ONNX model
import onnxruntime as ort  # noqa: E402
_sess_probe = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
_input_meta = _sess_probe.get_inputs()[0]
_shape = [d if isinstance(d, int) and d > 0 else 1 for d in _input_meta.shape]
_dtype = np.float32
inputs = {_input_meta.name: np.random.rand(*_shape).astype(_dtype)}
_log.info("Benchmark input: name=%s shape=%s", _input_meta.name, _shape)

N_RUNS = 10

npu_ms: float | None = None
cpu_ms: float | None = None
npu_executed = False

if not iron_installed:
    _log.warning("IRON install failed — running CPU baseline only")
    honest_verdict = "install_failed"
    # Still measure CPU baseline for future comparison
    try:
        cpu_runner = IRONRunner(onnx_path)
        cpu_ms = cpu_runner.benchmark(inputs, n_runs=N_RUNS)
        _log.info("CPU baseline: %.3f ms (median over %d runs)", cpu_ms, N_RUNS)
    except Exception as exc:
        _log.warning("CPU baseline failed: %s", exc)
        cpu_ms = None
else:
    # Run benchmark (will use NPU if available, CPU otherwise)
    _log.info("Running benchmark (NPU path attempted if iron+device available)...")
    result_ms = runner.benchmark(inputs, n_runs=N_RUNS)

    # Determine which path was used
    probe_result = runner.run(inputs)
    npu_executed = probe_result["npu_executed"]

    if npu_executed:
        npu_ms = result_ms
        # Also measure CPU for speedup denominator
        _log.info("NPU executed! Measuring CPU baseline for speedup ratio...")
        cpu_runner = IRONRunner(onnx_path)
        from unittest.mock import patch
        with patch.object(NPUEnvironment, "iron_available", return_value=False):
            with patch.object(NPUEnvironment, "npu_device_present", return_value=False):
                cpu_ms = cpu_runner.benchmark(inputs, n_runs=N_RUNS)
        _log.info("NPU: %.3f ms  CPU: %.3f ms  speedup=%.2fx", npu_ms, cpu_ms, cpu_ms / npu_ms if npu_ms else 0)
        honest_verdict = "npu_executed"
    else:
        cpu_ms = result_ms
        _log.info("CPU fallback: %.3f ms (median over %d runs)", cpu_ms, N_RUNS)
        honest_verdict = "cpu_baseline_only"

speedup: float | None = None
if npu_ms is not None and cpu_ms is not None and npu_ms > 0:
    speedup = cpu_ms / npu_ms

_log.info(
    "honest_verdict=%s iron_installed=%s npu_present=%s npu_executed=%s "
    "npu_ms=%s cpu_ms=%s speedup=%s",
    honest_verdict, iron_installed, npu_present, npu_executed,
    npu_ms, cpu_ms, speedup,
)

# ---------------------------------------------------------------------------
# Step 6: Build and write artifact
# ---------------------------------------------------------------------------
artifact = tmpl.build_result(
    {
        "schema": "carnot.npu_iron.v1",
        "iron_installed": iron_installed,
        "iron_available": iron_available,
        "npu_present": npu_present,
        "npu_executed": npu_executed,
        "npu_ms": npu_ms,
        "cpu_ms": cpu_ms,
        "speedup": speedup,
        "honest_verdict": honest_verdict,
        "onnx_model": onnx_path,
        "using_synthetic_model": using_synthetic,
        "n_runs": N_RUNS,
        "iron_paper": "arXiv 2504.03083",
        "blockage_resolved": iron_installed,
    },
    status="success" if honest_verdict in ("npu_executed", "cpu_baseline_only") else "blocked",
)

output_path = _REPO_ROOT / DELIVERABLE
output_path.parent.mkdir(parents=True, exist_ok=True)
output_path.write_text(json.dumps(artifact, indent=2))
_log.info("Artifact written: %s", output_path)

print(json.dumps(artifact, indent=2))
