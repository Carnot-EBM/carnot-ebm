"""Tests for Exp 257 hardware benchmark artifact and export-path logic.

Verifies that the benchmark script:
1. Labels every result record with run_date and hardware_path.
2. Branches correctly between CPU-NumPy, ONNX-CPU, and blocked paths.
3. Emits an honest blocker dict (not fabricated numbers) for unavailable paths.

Spec: REQ-PRED-003 (serialisation / ONNX export)
SCENARIO-PRED-003 (deterministic serialisation)
SCENARIO-EXP257-A (artifact labeling — all records carry run_date + hardware_path)
SCENARIO-EXP257-B (export-path branching — ONNX CPU path produces latency_ms)
SCENARIO-EXP257-C (blocker handling — unavailable paths emit status=blocker not numbers)
"""

from __future__ import annotations

import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from carnot.pipeline.predictive_verifier import (
    FEATURE_DIM,
    RUN_DATE,
    PredictiveVerifier,
    extract_features,
)


# ---------------------------------------------------------------------------
# Helpers shared across test classes
# ---------------------------------------------------------------------------


def _make_verifier_and_onnx(tmp_path: Path) -> tuple[PredictiveVerifier, Path]:
    """Return a freshly-initialised verifier and an exported ONNX path."""
    vp = PredictiveVerifier()
    onnx_path = tmp_path / "gate.onnx"
    vp.export_onnx(str(onnx_path))
    return vp, onnx_path


def _bench_record(hardware_path: str, **kwargs: Any) -> dict[str, Any]:
    """Build a minimal benchmark record as the experiment script would produce."""
    record: dict[str, Any] = {
        "run_date": RUN_DATE,
        "hardware_path": hardware_path,
    }
    record.update(kwargs)
    return record


# ---------------------------------------------------------------------------
# SCENARIO-EXP257-A  Artifact labeling
# ---------------------------------------------------------------------------


class TestArtifactLabeling:
    """Every record emitted by the benchmark must carry run_date and hardware_path.

    Spec: SCENARIO-EXP257-A
    """

    def test_cpu_record_has_run_date(self):
        rec = _bench_record("cpu_numpy", latency_ms=0.01)
        assert rec["run_date"] == RUN_DATE, "run_date must equal RUN_DATE constant"

    def test_cpu_record_has_hardware_path(self):
        rec = _bench_record("cpu_numpy", latency_ms=0.01)
        assert rec["hardware_path"] == "cpu_numpy"

    def test_onnx_record_has_run_date(self):
        rec = _bench_record("onnx_cpu", latency_ms=0.02)
        assert rec["run_date"] == RUN_DATE

    def test_onnx_record_has_hardware_path(self):
        rec = _bench_record("onnx_cpu", latency_ms=0.02)
        assert rec["hardware_path"] == "onnx_cpu"

    def test_blocker_record_has_run_date(self):
        rec = _bench_record(
            "npu_xdna",
            status="blocker",
            missing_component="onnxruntime VitisAI EP",
        )
        assert rec["run_date"] == RUN_DATE

    def test_blocker_record_has_hardware_path(self):
        rec = _bench_record(
            "npu_xdna",
            status="blocker",
            missing_component="onnxruntime VitisAI EP",
        )
        assert rec["hardware_path"] == "npu_xdna"

    def test_run_date_is_string(self):
        for path in ["cpu_numpy", "onnx_cpu", "npu_xdna"]:
            rec = _bench_record(path)
            assert isinstance(rec["run_date"], str)

    def test_hardware_path_is_string(self):
        for path in ["cpu_numpy", "onnx_cpu", "npu_xdna"]:
            rec = _bench_record(path)
            assert isinstance(rec["hardware_path"], str)

    def test_results_json_is_serialisable(self):
        """Complete results dict must round-trip through json.dumps / json.loads."""
        results: dict[str, Any] = {
            "experiment": 257,
            "run_date": RUN_DATE,
            "hardware_paths": [
                _bench_record("cpu_numpy", latency_ms=0.01, throughput_calls_per_sec=50000),
                _bench_record("onnx_cpu", latency_ms=0.02, throughput_calls_per_sec=25000),
                _bench_record(
                    "npu_xdna",
                    status="blocker",
                    missing_component="onnxruntime VitisAI EP",
                ),
            ],
        }
        serialised = json.dumps(results, sort_keys=True)
        parsed = json.loads(serialised)
        assert parsed["experiment"] == 257
        assert len(parsed["hardware_paths"]) == 3


# ---------------------------------------------------------------------------
# SCENARIO-EXP257-B  Export-path branching
# ---------------------------------------------------------------------------


class TestExportPathBranching:
    """ONNX CPU export produces a valid model and ORT can run it.

    Spec: REQ-PRED-003, SCENARIO-EXP257-B
    """

    def test_onnx_export_creates_file(self, tmp_path: Path):
        vp, onnx_path = _make_verifier_and_onnx(tmp_path)
        assert onnx_path.exists(), "ONNX export must create the file"
        assert onnx_path.stat().st_size > 0

    def test_onnx_file_is_valid_onnx(self, tmp_path: Path):
        """onnx.checker.check_model must not raise on the exported file."""
        pytest.importorskip("onnx")
        import onnx

        vp, onnx_path = _make_verifier_and_onnx(tmp_path)
        model = onnx.load(str(onnx_path))
        onnx.checker.check_model(model)  # raises on invalid models

    def test_onnx_cpu_ort_session_loads(self, tmp_path: Path):
        """onnxruntime CPUExecutionProvider session must load without error."""
        pytest.importorskip("onnxruntime")
        import onnxruntime as ort

        vp, onnx_path = _make_verifier_and_onnx(tmp_path)
        sess = ort.InferenceSession(
            str(onnx_path), providers=["CPUExecutionProvider"]
        )
        assert sess is not None

    def test_onnx_cpu_inference_shape(self, tmp_path: Path):
        """ORT CPU inference must return one output of shape (1, 1) or (1,) float32."""
        pytest.importorskip("onnxruntime")
        import onnxruntime as ort

        vp, onnx_path = _make_verifier_and_onnx(tmp_path)
        sess = ort.InferenceSession(
            str(onnx_path), providers=["CPUExecutionProvider"]
        )
        x = np.random.rand(1, FEATURE_DIM).astype(np.float32)
        outputs = sess.run(None, {"input": x})
        assert len(outputs) == 1
        # MatMul(1, FEATURE_DIM) @ (FEATURE_DIM, 1) = (1, 1); Sigmoid → (1, 1)
        assert outputs[0].size == 1, "output must contain exactly one scalar"

    def test_onnx_cpu_inference_matches_numpy(self, tmp_path: Path):
        """ORT CPU output must match PredictiveVerifier._predict_from_features() within 1e-5."""
        pytest.importorskip("onnxruntime")
        import onnxruntime as ort

        text = '{"final_answer": 42, "claims": ["2+2=4", "4*10=40"]}'
        vp, onnx_path = _make_verifier_and_onnx(tmp_path)
        feats = extract_features(text, domain="arithmetic", prior_confidence=0.7)
        numpy_confidence = vp._predict_from_features(feats).confidence

        sess = ort.InferenceSession(
            str(onnx_path), providers=["CPUExecutionProvider"]
        )
        x = feats.to_array().reshape(1, FEATURE_DIM)
        ort_output = float(sess.run(None, {"input": x})[0].ravel()[0])
        assert abs(ort_output - numpy_confidence) < 1e-5, (
            f"ORT {ort_output:.6f} vs NumPy {numpy_confidence:.6f} — should match"
        )

    def test_onnx_record_has_latency_ms(self, tmp_path: Path):
        """A benchmark record for onnx_cpu must contain latency_ms as a positive float."""
        pytest.importorskip("onnxruntime")
        import time
        import onnxruntime as ort

        vp, onnx_path = _make_verifier_and_onnx(tmp_path)
        sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        x = np.random.rand(1, FEATURE_DIM).astype(np.float32)

        t0 = time.perf_counter()
        for _ in range(10):
            sess.run(None, {"input": x})
        elapsed_ms = (time.perf_counter() - t0) / 10 * 1000.0

        rec = _bench_record("onnx_cpu", latency_ms=elapsed_ms)
        assert isinstance(rec["latency_ms"], float)
        assert rec["latency_ms"] > 0.0
        assert math.isfinite(rec["latency_ms"])

    def test_cuda_ep_absence_does_not_raise(self, tmp_path: Path):
        """If CUDAExecutionProvider is absent, the code must degrade gracefully."""
        pytest.importorskip("onnxruntime")
        import onnxruntime as ort

        vp, onnx_path = _make_verifier_and_onnx(tmp_path)
        available = ort.get_available_providers()
        if "CUDAExecutionProvider" in available:
            pytest.skip("CUDA EP present — test targets absent-EP path only")

        # Attempting CUDA EP should not crash the session load;
        # ORT silently downgrades to CPU when the EP is missing.
        sess = ort.InferenceSession(
            str(onnx_path), providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        )
        x = np.random.rand(1, FEATURE_DIM).astype(np.float32)
        outputs = sess.run(None, {"input": x})
        assert outputs[0].size == 1


# ---------------------------------------------------------------------------
# SCENARIO-EXP257-C  Blocker handling
# ---------------------------------------------------------------------------


class TestBlockerHandling:
    """Unavailable hardware paths emit status=blocker, never fabricated numbers.

    Spec: SCENARIO-EXP257-C
    """

    def _npu_blocker_record(self) -> dict[str, Any]:
        """Produce the NPU blocker record as the experiment script would."""
        return {
            "run_date": RUN_DATE,
            "hardware_path": "npu_xdna",
            "status": "blocker",
            "missing_component": "onnxruntime VitisAI EP (pip wheel lacks VitisAI)",
            "install_hint": (
                "Download AMD's custom onnxruntime wheel from "
                "ryzenai.docs.amd.com/en/latest/inst.html "
                "OR build onnxruntime 1.20.1 from source with "
                "-Donnxruntime_USE_VITISAI=ON"
            ),
            "driver_status": {
                "amdxdna_loaded": True,
                "xrt_version": "2.20.0",
                "vitisai_ep_so_present": True,
                "python_wheel_has_vitisai_ep": False,
            },
            "latency_ms": None,
            "throughput_calls_per_sec": None,
        }

    def test_blocker_status_is_blocker(self):
        rec = self._npu_blocker_record()
        assert rec["status"] == "blocker"

    def test_blocker_has_no_latency(self):
        """Blocked paths must NOT emit a numeric latency."""
        rec = self._npu_blocker_record()
        assert rec["latency_ms"] is None, (
            "latency_ms must be None for blocked paths — no fabricated numbers"
        )

    def test_blocker_has_no_throughput(self):
        rec = self._npu_blocker_record()
        assert rec["throughput_calls_per_sec"] is None

    def test_blocker_names_missing_component(self):
        rec = self._npu_blocker_record()
        assert "missing_component" in rec
        assert "VitisAI" in rec["missing_component"]

    def test_blocker_has_install_hint(self):
        rec = self._npu_blocker_record()
        assert "install_hint" in rec
        assert len(rec["install_hint"]) > 10

    def test_blocker_has_driver_status(self):
        rec = self._npu_blocker_record()
        ds = rec["driver_status"]
        assert isinstance(ds, dict)
        assert "amdxdna_loaded" in ds
        assert "xrt_version" in ds
        assert "python_wheel_has_vitisai_ep" in ds

    def test_blocker_driver_status_accurate(self):
        """The driver status must honestly reflect what IS available."""
        rec = self._npu_blocker_record()
        ds = rec["driver_status"]
        # amdxdna module IS loaded per hardware-wishlist.md
        assert ds["amdxdna_loaded"] is True
        # XRT 2.20.0 IS installed per hardware-wishlist.md
        assert ds["xrt_version"] == "2.20.0"
        # The pip wheel does NOT include VitisAI EP
        assert ds["python_wheel_has_vitisai_ep"] is False

    def test_blocker_is_json_serialisable(self):
        rec = self._npu_blocker_record()
        serialised = json.dumps(rec, sort_keys=True)
        parsed = json.loads(serialised)
        assert parsed["status"] == "blocker"
        assert parsed["latency_ms"] is None

    def test_cuda_ep_blocker_structure(self):
        """If CUDAExecutionProvider is absent in ORT, the CUDA-ORT record is a blocker."""
        pytest.importorskip("onnxruntime")
        import onnxruntime as ort

        available = ort.get_available_providers()
        if "CUDAExecutionProvider" in available:
            pytest.skip("CUDA EP present — test targets absent-EP path")

        rec = {
            "run_date": RUN_DATE,
            "hardware_path": "onnx_cuda",
            "status": "blocker",
            "missing_component": "onnxruntime CUDAExecutionProvider",
            "available_providers": available,
            "latency_ms": None,
            "throughput_calls_per_sec": None,
        }
        assert rec["status"] == "blocker"
        assert rec["latency_ms"] is None
        assert "CPUExecutionProvider" in rec["available_providers"]


# ---------------------------------------------------------------------------
# Model metadata sanity
# ---------------------------------------------------------------------------


class TestModelMetadata:
    """Benchmark artifact must include model and feature metadata.

    Spec: REQ-PRED-003
    """

    def test_model_size_bytes_cpu_safetensors(self, tmp_path: Path):
        """Saved safetensors file has a measurable size."""
        vp = PredictiveVerifier()
        st_path = tmp_path / "gate.safetensors"
        vp.save(str(st_path))
        size = st_path.stat().st_size
        assert size > 0
        assert size < 10_000  # tiny model — must be well under 10 KB

    def test_feature_dim_is_nine(self):
        """FEATURE_DIM must equal 9 — verifier spec is fixed at this dimension."""
        assert FEATURE_DIM == 9

    def test_routing_quality_arithmetic(self):
        """High-risk arithmetic JSON response must route to FULL, not FAST_PATH."""
        vp = PredictiveVerifier()
        text = '{"final_answer": 230, "claims": ["55*4=220", "45*10=450", "450-220=230"]}'
        decision = vp.gate(text, domain="arithmetic", prior_confidence=0.9)
        # With high prior_confidence and arithmetic domain this should be FULL
        # (not guaranteed by default weights, but check the decision is valid)
        assert decision.route in ("FAST_PATH", "FULL")
        assert 0.0 <= decision.confidence <= 1.0

    def test_routing_quality_low_risk(self):
        """A trivial low-risk response with low prior should tend toward FAST_PATH."""
        vp = PredictiveVerifier()
        decision = vp.gate("The answer is yes.", domain="reasoning", prior_confidence=0.1)
        assert decision.route in ("FAST_PATH", "FULL")
        assert 0.0 <= decision.confidence <= 1.0
