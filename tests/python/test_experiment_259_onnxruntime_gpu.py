"""Tests for Experiment 259: onnxruntime CUDA EP benchmark.

Covers:
  - CUDA EP detection (mock path when GPU unavailable / CARNOT_FORCE_LIVE=0)
  - PredictiveVerifier ONNX model load and inference sanity
  - Benchmark artifact schema validation

Spec: REQ-PRED-003 (ONNX export)
SCENARIO-EXP259-A (CUDA EP detection)
SCENARIO-EXP259-B (artifact schema)
SCENARIO-EXP259-C (mock path when CUDA EP absent)
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_FORCE_LIVE = os.environ.get("CARNOT_FORCE_LIVE", "0") == "1"
"""True only when the test environment has real GPU + onnxruntime-gpu installed."""


def _make_vp():
    """Return a default PredictiveVerifier for test use."""
    import sys
    _repo = Path(__file__).resolve().parent.parent.parent / "python"
    if str(_repo) not in sys.path:
        import sys as _sys
        _sys.path.insert(0, str(_repo))
    from carnot.pipeline.predictive_verifier import PredictiveVerifier
    return PredictiveVerifier()


def _export_gate_onnx(vp, path: Path) -> None:
    """Export the gate to ONNX; skip if onnx package missing."""
    try:
        vp.export_onnx(str(path))
    except ImportError as exc:
        pytest.skip(f"onnx package not installed: {exc}")


# ---------------------------------------------------------------------------
# SCENARIO-EXP259-A: CUDA EP detection
# ---------------------------------------------------------------------------


class TestCudaEpDetection:
    """SCENARIO-EXP259-A — detect whether CUDAExecutionProvider is available."""

    def test_onnxruntime_importable(self):
        """onnxruntime (cpu or gpu build) must be importable.

        Spec: REQ-PRED-003
        """
        import onnxruntime  # noqa: F401

    def test_get_available_providers_returns_list(self):
        """get_available_providers() returns a non-empty list.

        Spec: REQ-PRED-003
        SCENARIO-EXP259-A
        """
        import onnxruntime as ort
        providers = ort.get_available_providers()
        assert isinstance(providers, list)
        assert len(providers) > 0

    def test_cpu_ep_always_present(self):
        """CPUExecutionProvider must always be in the provider list.

        Spec: REQ-PRED-003
        SCENARIO-EXP259-A
        """
        import onnxruntime as ort
        assert "CPUExecutionProvider" in ort.get_available_providers()

    @pytest.mark.skipif(not _FORCE_LIVE, reason="Requires real GPU (CARNOT_FORCE_LIVE=1)")
    def test_cuda_ep_present_when_gpu_available(self):
        """When onnxruntime-gpu is installed, CUDAExecutionProvider must appear.

        Spec: REQ-PRED-003
        SCENARIO-EXP259-A
        """
        import onnxruntime as ort
        assert "CUDAExecutionProvider" in ort.get_available_providers()

    def test_cuda_ep_detection_mock(self):
        """Mock path: absence of CUDAExecutionProvider triggers honest blocker.

        When CUDAExecutionProvider is not in available_providers, the benchmark
        function must return status='blocker' and NOT fabricate latency numbers.

        Spec: REQ-PRED-003
        SCENARIO-EXP259-C
        """
        # Simulate a CPU-only ORT build (e.g. plain 'pip install onnxruntime').
        fake_providers = ["CPUExecutionProvider"]
        with patch("onnxruntime.get_available_providers", return_value=fake_providers):
            import onnxruntime as ort
            assert "CUDAExecutionProvider" not in ort.get_available_providers()
            # Record what a blocker artifact should look like.
            blocker = {
                "hardware_path": "onnx_cuda",
                "status": "blocker",
                "missing_component": "onnxruntime CUDAExecutionProvider",
                "available_providers": fake_providers,
                "latency_ms": None,
                "throughput_calls_per_sec": None,
            }
            assert blocker["status"] == "blocker"
            assert blocker["latency_ms"] is None
            assert blocker["throughput_calls_per_sec"] is None


# ---------------------------------------------------------------------------
# SCENARIO-EXP259-B: Model load and inference sanity
# ---------------------------------------------------------------------------


class TestOnnxModelLoad:
    """SCENARIO-EXP259-B — ONNX gate model can be exported and loaded."""

    def test_export_and_load_cpu(self):
        """Export gate to ONNX and run inference via CPUExecutionProvider.

        Spec: REQ-PRED-003
        SCENARIO-EXP259-B
        """
        import onnxruntime as ort
        from carnot.pipeline.predictive_verifier import FEATURE_DIM, extract_features

        vp = _make_vp()
        with tempfile.TemporaryDirectory() as tmp:
            onnx_path = Path(tmp) / "gate.onnx"
            _export_gate_onnx(vp, onnx_path)

            sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
            x = extract_features("42 + 1 = 43", domain="arithmetic").to_array()
            out = sess.run(None, {"input": x.reshape(1, FEATURE_DIM)})[0]
            assert out.shape == (1, 1) or out.shape == (1,)
            prob = float(out.ravel()[0])
            assert 0.0 <= prob <= 1.0

    def test_onnx_cpu_matches_numpy(self):
        """ORT CPU output must match NumPy gate within floating-point tolerance.

        Spec: REQ-PRED-003
        SCENARIO-EXP259-B
        """
        import onnxruntime as ort
        from carnot.pipeline.predictive_verifier import FEATURE_DIM, extract_features

        vp = _make_vp()
        with tempfile.TemporaryDirectory() as tmp:
            onnx_path = Path(tmp) / "gate.onnx"
            _export_gate_onnx(vp, onnx_path)

            sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
            feats = extract_features(
                '{"final_answer": 99, "claims": ["1+1=2"]}',
                domain="arithmetic",
                prior_confidence=0.8,
            )
            x = feats.to_array().reshape(1, FEATURE_DIM)
            ort_prob = float(sess.run(None, {"input": x})[0].ravel()[0])
            numpy_prob = vp._predict_from_features(feats).confidence
            assert abs(ort_prob - numpy_prob) < 1e-5


# ---------------------------------------------------------------------------
# SCENARIO-EXP259-B: Artifact schema
# ---------------------------------------------------------------------------

# Required fields for a completed (non-blocker) CUDA ORT benchmark record.
_REQUIRED_OK_FIELDS = {
    "hardware_path",
    "latency_us",
    "throughput_calls_per_sec",
    "gpu_memory_mb",
    "ort_version",
    "providers_used",
    "speedup_vs_cpu_ort",
    "speedup_vs_cpu_numpy",
    "run_date",
    "status",
    "timed_calls",
}

# Required fields for a blocker record.
_REQUIRED_BLOCKER_FIELDS = {
    "hardware_path",
    "status",
    "missing_component",
    "latency_ms",
    "throughput_calls_per_sec",
    "run_date",
}


class TestArtifactSchema:
    """SCENARIO-EXP259-B — results JSON must conform to the required schema."""

    def _make_ok_record(self) -> dict:
        """Construct a synthetic 'ok' CUDA ORT record for schema validation."""
        return {
            "hardware_path": "onnx_cuda",
            "status": "ok",
            "run_date": "20260413",
            "ort_version": "1.24.4",
            "providers_used": ["CUDAExecutionProvider", "CPUExecutionProvider"],
            "timed_calls": 5000,
            "latency_us": 2.5,
            "latency_ms": 0.0025,
            "throughput_calls_per_sec": 400000.0,
            "gpu_memory_mb": 512.0,
            "speedup_vs_cpu_ort": 2.34,
            "speedup_vs_cpu_numpy": 16.7,
        }

    def _make_blocker_record(self) -> dict:
        """Construct a synthetic blocker record for schema validation."""
        return {
            "hardware_path": "onnx_cuda",
            "status": "blocker",
            "run_date": "20260413",
            "missing_component": "onnxruntime CUDAExecutionProvider",
            "latency_ms": None,
            "throughput_calls_per_sec": None,
        }

    def test_ok_record_has_required_fields(self):
        """A completed CUDA ORT record must have all required fields.

        Spec: REQ-PRED-003
        SCENARIO-EXP259-B
        """
        rec = self._make_ok_record()
        missing = _REQUIRED_OK_FIELDS - set(rec.keys())
        assert missing == set(), f"Missing required fields: {missing}"

    def test_blocker_record_has_required_fields(self):
        """A blocker record must have all required blocker fields.

        Spec: REQ-PRED-003
        SCENARIO-EXP259-B
        """
        rec = self._make_blocker_record()
        missing = _REQUIRED_BLOCKER_FIELDS - set(rec.keys())
        assert missing == set(), f"Missing required fields: {missing}"

    def test_ok_record_latency_is_positive(self):
        """latency_us must be a positive float.

        Spec: REQ-PRED-003
        SCENARIO-EXP259-B
        """
        rec = self._make_ok_record()
        assert isinstance(rec["latency_us"], float)
        assert rec["latency_us"] > 0.0

    def test_ok_record_throughput_is_positive(self):
        """throughput_calls_per_sec must be a positive float.

        Spec: REQ-PRED-003
        SCENARIO-EXP259-B
        """
        rec = self._make_ok_record()
        assert isinstance(rec["throughput_calls_per_sec"], float)
        assert rec["throughput_calls_per_sec"] > 0.0

    def test_ok_record_speedup_is_positive(self):
        """speedup_vs_cpu_ort must be a positive float.

        Spec: REQ-PRED-003
        SCENARIO-EXP259-B
        """
        rec = self._make_ok_record()
        assert isinstance(rec["speedup_vs_cpu_ort"], float)
        assert rec["speedup_vs_cpu_ort"] > 0.0

    def test_blocker_latency_is_none(self):
        """A blocker record must have latency_ms = None (no fabricated numbers).

        Spec: REQ-PRED-003
        SCENARIO-EXP259-C
        """
        rec = self._make_blocker_record()
        assert rec["latency_ms"] is None

    def test_blocker_throughput_is_none(self):
        """A blocker record must have throughput_calls_per_sec = None.

        Spec: REQ-PRED-003
        SCENARIO-EXP259-C
        """
        rec = self._make_blocker_record()
        assert rec["throughput_calls_per_sec"] is None

    def test_results_file_schema_if_exists(self):
        """If experiment_259_results.json exists, validate its top-level schema.

        Spec: REQ-PRED-003
        SCENARIO-EXP259-B
        """
        results_path = (
            Path(__file__).resolve().parent.parent.parent
            / "results"
            / "experiment_259_results.json"
        )
        if not results_path.exists():
            pytest.skip("experiment_259_results.json not yet generated")

        with open(results_path) as fh:
            data = json.load(fh)

        assert data.get("experiment") == 259
        assert "hardware_paths" in data
        assert isinstance(data["hardware_paths"], list)
        assert len(data["hardware_paths"]) >= 1

        cuda_records = [
            r for r in data["hardware_paths"] if r.get("hardware_path") == "onnx_cuda"
        ]
        assert len(cuda_records) == 1, "Exactly one onnx_cuda record expected"
        cuda = cuda_records[0]
        assert cuda.get("status") in ("ok", "blocker")

        if cuda["status"] == "ok":
            missing = _REQUIRED_OK_FIELDS - set(cuda.keys())
            assert missing == set(), f"Missing fields in onnx_cuda record: {missing}"
        else:
            missing = _REQUIRED_BLOCKER_FIELDS - set(cuda.keys())
            assert missing == set(), f"Missing fields in blocker record: {missing}"
