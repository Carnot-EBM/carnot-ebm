"""Tests for NPUEntropyProbe and NPUBenchmarkResult.

Spec: REQ-INFRA-061, REQ-INFRA-062, REQ-INFRA-063,
      SCENARIO-INFRA-070, SCENARIO-INFRA-071, SCENARIO-INFRA-072
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

from carnot.pipeline.npu_entropy_probe import NPUBenchmarkResult, NPUEntropyProbe


# ---------------------------------------------------------------------------
# NPUBenchmarkResult tests
# ---------------------------------------------------------------------------


class TestNPUBenchmarkResult:
    # SCENARIO-INFRA-070
    def test_npu_viable_false_when_not_available(self):
        """npu_viable must be False when npu_available=False (REQ-INFRA-063)."""
        result = NPUBenchmarkResult(
            npu_latency_ms=None,
            cpu_latency_ms=1.5,
            npu_available=False,
            speedup_ratio=None,
        )
        assert result.npu_viable is False

    # SCENARIO-INFRA-071
    def test_speedup_ratio_none_when_not_available(self):
        """speedup_ratio must be None when npu_available=False (REQ-INFRA-063)."""
        result = NPUBenchmarkResult(
            npu_latency_ms=None,
            cpu_latency_ms=1.5,
            npu_available=False,
            speedup_ratio=None,
        )
        assert result.speedup_ratio is None

    def test_npu_viable_false_when_speedup_below_threshold(self):
        """npu_viable is False when speedup_ratio < 2.0 even if NPU is available."""
        result = NPUBenchmarkResult(
            npu_latency_ms=1.0,
            cpu_latency_ms=1.5,
            npu_available=True,
            speedup_ratio=1.5,
        )
        assert result.npu_viable is False

    def test_npu_viable_true_when_fast_enough(self):
        """npu_viable is True when npu_available=True and speedup_ratio >= 2.0."""
        result = NPUBenchmarkResult(
            npu_latency_ms=0.5,
            cpu_latency_ms=1.5,
            npu_available=True,
            speedup_ratio=3.0,
        )
        assert result.npu_viable is True

    def test_npu_viable_exactly_at_threshold(self):
        """npu_viable is True when speedup_ratio == 2.0 exactly."""
        result = NPUBenchmarkResult(
            npu_latency_ms=0.75,
            cpu_latency_ms=1.5,
            npu_available=True,
            speedup_ratio=2.0,
        )
        assert result.npu_viable is True

    def test_to_dict_keys(self):
        """to_dict() returns all expected keys."""
        result = NPUBenchmarkResult(
            npu_latency_ms=None,
            cpu_latency_ms=1.5,
            npu_available=False,
            speedup_ratio=None,
        )
        d = result.to_dict()
        assert set(d.keys()) == {
            "npu_latency_ms",
            "cpu_latency_ms",
            "npu_available",
            "speedup_ratio",
            "npu_viable",
        }

    def test_to_dict_values_match(self):
        """to_dict() values match field values."""
        result = NPUBenchmarkResult(
            npu_latency_ms=2.0,
            cpu_latency_ms=5.0,
            npu_available=True,
            speedup_ratio=2.5,
        )
        d = result.to_dict()
        assert d["npu_latency_ms"] == 2.0
        assert d["cpu_latency_ms"] == 5.0
        assert d["npu_available"] is True
        assert d["speedup_ratio"] == 2.5
        assert d["npu_viable"] is True


# ---------------------------------------------------------------------------
# NPUEntropyProbe tests
# ---------------------------------------------------------------------------


class TestNPUEntropyProbe:
    def test_init_defaults(self):
        """Default seq_len=64 and vocab_size=50000."""
        probe = NPUEntropyProbe()
        assert probe.seq_len == 64
        assert probe.vocab_size == 50000

    def test_init_custom(self):
        """Custom seq_len and vocab_size are stored."""
        probe = NPUEntropyProbe(seq_len=16, vocab_size=1000)
        assert probe.seq_len == 16
        assert probe.vocab_size == 1000

    # SCENARIO-INFRA-072
    def test_export_onnx_creates_file(self, tmp_path):
        """export_onnx creates a file at the given path (REQ-INFRA-061)."""
        probe = NPUEntropyProbe(seq_len=4, vocab_size=100)
        out = str(tmp_path / "test_entropy.onnx")
        probe.export_onnx(out)
        assert Path(out).exists()
        assert Path(out).stat().st_size > 0

    def test_export_onnx_returns_bool(self, tmp_path):
        """export_onnx returns a bool."""
        probe = NPUEntropyProbe(seq_len=4, vocab_size=100)
        out = str(tmp_path / "test_entropy.onnx")
        result = probe.export_onnx(out)
        assert isinstance(result, bool)

    def test_load_vitisai_returns_false_without_ep(self, tmp_path):
        """load_vitisai returns False when VitisAI EP is not available (REQ-INFRA-063)."""
        probe = NPUEntropyProbe(seq_len=4, vocab_size=100)
        out = str(tmp_path / "entropy.onnx")
        probe.export_onnx(out)
        result = probe.load_vitisai(out)
        # On CI without VitisAI EP installed, this must return False
        assert isinstance(result, bool)

    def test_compute_entropy_shape(self):
        """compute_entropy returns array of shape (seq_len,)."""
        probe = NPUEntropyProbe(seq_len=8, vocab_size=200)
        rng = np.random.default_rng(0)
        activations = rng.standard_normal((8, 200)).astype(np.float32)
        entropy = probe.compute_entropy(activations)
        assert entropy.shape == (8,)

    def test_compute_entropy_nonnegative(self):
        """Shannon entropy is always >= 0."""
        probe = NPUEntropyProbe(seq_len=8, vocab_size=200)
        rng = np.random.default_rng(1)
        activations = rng.standard_normal((8, 200)).astype(np.float32)
        entropy = probe.compute_entropy(activations)
        assert np.all(entropy >= -1e-6)

    def test_compute_entropy_uniform_maximises(self):
        """Uniform distribution maximises entropy over non-uniform distribution."""
        probe = NPUEntropyProbe(seq_len=2, vocab_size=100)
        # Uniform logits → maximum entropy
        uniform = np.zeros((2, 100), dtype=np.float32)
        # Peaked logits → low entropy
        peaked = np.zeros((2, 100), dtype=np.float32)
        peaked[:, 0] = 10.0  # one very dominant token
        h_uniform = probe.compute_entropy(uniform)
        h_peaked = probe.compute_entropy(peaked)
        assert np.all(h_uniform > h_peaked)

    def test_benchmark_returns_result(self):
        """benchmark() returns an NPUBenchmarkResult."""
        probe = NPUEntropyProbe(seq_len=4, vocab_size=100)
        result = probe.benchmark(n_trials=5)
        assert isinstance(result, NPUBenchmarkResult)

    def test_benchmark_cpu_latency_positive(self):
        """cpu_latency_ms is always a positive float."""
        probe = NPUEntropyProbe(seq_len=4, vocab_size=100)
        result = probe.benchmark(n_trials=5)
        assert result.cpu_latency_ms > 0.0

    def test_benchmark_npu_not_available_without_ep(self):
        """Without VitisAI EP, benchmark returns npu_available=False."""
        probe = NPUEntropyProbe(seq_len=4, vocab_size=100)
        # No load_vitisai() called — _using_npu is False
        result = probe.benchmark(n_trials=5)
        assert result.npu_available is False
        assert result.npu_latency_ms is None
        assert result.speedup_ratio is None
        assert result.npu_viable is False

    def test_export_onnx_creates_parent_dirs(self, tmp_path):
        """export_onnx creates parent directories if they don't exist."""
        probe = NPUEntropyProbe(seq_len=4, vocab_size=100)
        nested = str(tmp_path / "deep" / "nested" / "entropy.onnx")
        probe.export_onnx(nested)
        assert Path(nested).exists()

    def test_compute_entropy_no_nan(self):
        """compute_entropy produces no NaN values for typical logit inputs."""
        probe = NPUEntropyProbe(seq_len=16, vocab_size=500)
        rng = np.random.default_rng(99)
        activations = rng.standard_normal((16, 500)).astype(np.float32) * 5
        entropy = probe.compute_entropy(activations)
        assert not np.any(np.isnan(entropy))

    def test_compute_entropy_with_session(self, tmp_path):
        """compute_entropy uses onnxruntime session when available."""
        import unittest.mock as mock

        probe = NPUEntropyProbe(seq_len=4, vocab_size=100)
        # Inject a mock session to hit the session.run() branch
        mock_session = mock.MagicMock()
        expected = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        mock_session.run.return_value = [expected]
        probe._session = mock_session
        activations = np.zeros((4, 100), dtype=np.float32)
        result = probe.compute_entropy(activations)
        assert mock_session.run.called
        np.testing.assert_array_equal(result, expected)

    def test_load_vitisai_ort_import_error(self, tmp_path, monkeypatch):
        """load_vitisai returns False when onnxruntime is not importable."""
        import builtins
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "onnxruntime":
                raise ImportError("mocked missing onnxruntime")
            return real_import(name, *args, **kwargs)

        probe = NPUEntropyProbe(seq_len=4, vocab_size=100)
        out = str(tmp_path / "entropy.onnx")
        probe.export_onnx(out)

        monkeypatch.setattr(builtins, "__import__", mock_import)
        result = probe.load_vitisai(out)
        assert result is False

    def test_load_vitisai_cpu_session_exception(self, tmp_path):
        """load_vitisai returns False and sets session=None when CPU EP also fails."""
        import unittest.mock as mock

        probe = NPUEntropyProbe(seq_len=4, vocab_size=100)
        out = str(tmp_path / "entropy.onnx")
        probe.export_onnx(out)

        with mock.patch("carnot.pipeline.npu_entropy_probe._VITISAI_EP", "VitisAIExecutionProvider"):
            with mock.patch("onnxruntime.get_available_providers", return_value=[]):
                with mock.patch("onnxruntime.InferenceSession", side_effect=RuntimeError("no EP")):
                    result = probe.load_vitisai(out)
        assert result is False
        assert probe._session is None

    def test_load_vitisai_vitisai_ep_success(self, tmp_path):
        """load_vitisai returns True and sets _using_npu when VitisAI EP is present."""
        import unittest.mock as mock

        probe = NPUEntropyProbe(seq_len=4, vocab_size=100)
        out = str(tmp_path / "entropy.onnx")
        probe.export_onnx(out)

        mock_session = mock.MagicMock()
        with mock.patch(
            "onnxruntime.get_available_providers",
            return_value=["VitisAIExecutionProvider", "CPUExecutionProvider"],
        ):
            with mock.patch("onnxruntime.InferenceSession", return_value=mock_session):
                result = probe.load_vitisai(out)

        assert result is True
        assert probe._using_npu is True
        assert probe._session is mock_session

    def test_load_vitisai_vitisai_ep_load_fails(self, tmp_path):
        """load_vitisai returns False when VitisAI EP raises during session creation."""
        import unittest.mock as mock

        probe = NPUEntropyProbe(seq_len=4, vocab_size=100)
        out = str(tmp_path / "entropy.onnx")
        probe.export_onnx(out)

        with mock.patch(
            "onnxruntime.get_available_providers",
            return_value=["VitisAIExecutionProvider", "CPUExecutionProvider"],
        ):
            with mock.patch("onnxruntime.InferenceSession", side_effect=RuntimeError("npu fail")):
                result = probe.load_vitisai(out)

        assert result is False
        assert probe._using_npu is False

    def test_benchmark_with_npu_session(self):
        """benchmark() measures NPU latency when _using_npu=True."""
        import unittest.mock as mock
        import numpy as np

        probe = NPUEntropyProbe(seq_len=4, vocab_size=100)
        mock_session = mock.MagicMock()
        mock_session.run.return_value = [np.zeros(4, dtype=np.float32)]
        probe._session = mock_session
        probe._using_npu = True

        result = probe.benchmark(n_trials=5)
        assert result.npu_available is True
        assert result.npu_latency_ms is not None
        assert result.npu_latency_ms > 0.0
        assert result.speedup_ratio is not None

    def test_export_onnx_fallback_when_onnx_missing(self, tmp_path, monkeypatch):
        """export_onnx writes a stub file when onnx package is not installed."""
        import builtins
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name in ("onnx",):
                raise ImportError("mocked missing onnx")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)
        probe = NPUEntropyProbe(seq_len=4, vocab_size=100)
        out = str(tmp_path / "stub.onnx")
        result = probe.export_onnx(out)
        assert result is False
        assert Path(out).exists()
