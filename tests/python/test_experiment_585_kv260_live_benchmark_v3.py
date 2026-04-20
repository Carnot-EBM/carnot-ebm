"""Tests for experiment_585_kv260_live_benchmark_v3.py — targeted 100% coverage.

Every test references the spec requirement or scenario it covers.

Spec: REQ-SAMPLE-033, SCENARIO-SAMPLE-055, SCENARIO-SAMPLE-056, SCENARIO-SAMPLE-057
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import numpy as np
import pytest

from scripts.experiment_585_kv260_live_benchmark_v3 import (
    CPU_BASELINE_LATENCY_US,
    DELIVERABLE,
    EXPERIMENT_ID,
    FPGA_LATENCY_TARGET_US,
    N_SPINS,
    N_TRIALS,
    SCHEMA,
    check_bitfile_env_match,
    choose_verdict,
    compute_benchmark_stats,
    load_gate_result,
    run_hardware_benchmark,
)


# ---------------------------------------------------------------------------
# load_gate_result
# ---------------------------------------------------------------------------


def test_load_gate_result_file_present_bitfile_built(tmp_path):
    """SCENARIO-SAMPLE-056: Gate file present with bitfile_built=True returns True."""
    gate = {"bitfile_built": True, "bitfile_path": "/some/path.bit"}
    (tmp_path / "results").mkdir()
    (tmp_path / "results" / "experiment_584_kv260_synthesis.json").write_text(
        json.dumps(gate)
    )
    result = load_gate_result(tmp_path)
    assert result["bitfile_built"] is True
    assert result["bitfile_path"] == "/some/path.bit"


def test_load_gate_result_file_present_bitfile_not_built(tmp_path):
    """SCENARIO-SAMPLE-055: Gate file present with bitfile_built=False returns False."""
    gate = {"bitfile_built": False, "bitfile_path": None, "honest_verdict": "vivado_not_installed"}
    (tmp_path / "results").mkdir()
    (tmp_path / "results" / "experiment_584_kv260_synthesis.json").write_text(
        json.dumps(gate)
    )
    result = load_gate_result(tmp_path)
    assert result["bitfile_built"] is False


def test_load_gate_result_file_missing(tmp_path):
    """SCENARIO-SAMPLE-055: Missing gate file returns bitfile_built=False sentinel."""
    result = load_gate_result(tmp_path)
    assert result["bitfile_built"] is False
    assert result["bitfile_path"] is None
    assert result["honest_verdict"] == "missing"


# ---------------------------------------------------------------------------
# check_bitfile_env_match
# ---------------------------------------------------------------------------


def test_check_bitfile_env_match_env_unset():
    """Env var not set returns True (no mismatch to warn about)."""
    with patch.dict(os.environ, {}, clear=False):
        os.environ.pop("CARNOT_KV260_BITFILE", None)
        assert check_bitfile_env_match("/some/path.bit") is True


def test_check_bitfile_env_match_matches():
    """Env var matches bitfile_path — no warning, returns True."""
    with patch.dict(os.environ, {"CARNOT_KV260_BITFILE": "/some/path.bit"}):
        assert check_bitfile_env_match("/some/path.bit") is True


def test_check_bitfile_env_match_mismatch(caplog):
    """Env var differs from bitfile_path — warning logged, returns False."""
    import logging
    with patch.dict(os.environ, {"CARNOT_KV260_BITFILE": "/other/path.bit"}):
        with caplog.at_level(logging.WARNING):
            result = check_bitfile_env_match("/some/path.bit")
    assert result is False
    assert "does not match" in caplog.text


# ---------------------------------------------------------------------------
# compute_benchmark_stats
# ---------------------------------------------------------------------------


def test_compute_benchmark_stats_empty():
    """Empty latency list returns None values and fpga_target_met=False."""
    stats = compute_benchmark_stats([])
    assert stats["mean_hardware_latency_us"] is None
    assert stats["std_hardware_latency_us"] is None
    assert stats["speedup_ratio"] is None
    assert stats["fpga_target_met"] is False


def test_compute_benchmark_stats_fast():
    """SCENARIO-SAMPLE-056: Latencies below 100 µs → fpga_target_met=True and speedup > 0."""
    latencies = [50.0] * 100  # 50 µs each
    stats = compute_benchmark_stats(latencies)
    assert stats["mean_hardware_latency_us"] == pytest.approx(50.0)
    assert stats["std_hardware_latency_us"] == pytest.approx(0.0)
    assert stats["speedup_ratio"] == pytest.approx(CPU_BASELINE_LATENCY_US / 50.0)
    assert stats["fpga_target_met"] is True


def test_compute_benchmark_stats_slow():
    """SCENARIO-SAMPLE-057: Latencies above 100 µs → fpga_target_met=False."""
    latencies = [500.0] * 10  # 500 µs each — slower than target
    stats = compute_benchmark_stats(latencies)
    assert stats["fpga_target_met"] is False
    assert stats["mean_hardware_latency_us"] == pytest.approx(500.0)


def test_compute_benchmark_stats_speedup_ratio():
    """REQ-SAMPLE-033-3: speedup_ratio = cpu_baseline / mean_latency."""
    latencies = [100.0] * 10
    stats = compute_benchmark_stats(latencies, cpu_baseline_us=1000.0)
    assert stats["speedup_ratio"] == pytest.approx(10.0)


def test_compute_benchmark_stats_target_boundary():
    """Latency exactly at target (100 µs) is NOT 'fast enough' (< is strict)."""
    latencies = [FPGA_LATENCY_TARGET_US] * 10
    stats = compute_benchmark_stats(latencies)
    assert stats["fpga_target_met"] is False


# ---------------------------------------------------------------------------
# choose_verdict
# ---------------------------------------------------------------------------


def test_choose_verdict_hardware_failed_flag():
    """REQ-SAMPLE-033-5: hardware_failed=True → 'hardware_failed'."""
    assert choose_verdict(hardware_failed=True, mean_latency_us=50.0) == "hardware_failed"


def test_choose_verdict_none_latency():
    """REQ-SAMPLE-033-5: mean_latency_us=None → 'hardware_failed'."""
    assert choose_verdict(hardware_failed=False, mean_latency_us=None) == "hardware_failed"


def test_choose_verdict_fast():
    """SCENARIO-SAMPLE-056: latency < target → 'hardware_working'."""
    assert choose_verdict(hardware_failed=False, mean_latency_us=50.0) == "hardware_working"


def test_choose_verdict_slow():
    """SCENARIO-SAMPLE-057: latency >= target → 'hardware_too_slow'."""
    assert choose_verdict(hardware_failed=False, mean_latency_us=200.0) == "hardware_too_slow"


# ---------------------------------------------------------------------------
# run_hardware_benchmark (unit — mock FpgaBackend)
# ---------------------------------------------------------------------------


def _make_fake_fpga_backend(n_spins: int, fail_on_trial: int | None = None) -> MagicMock:
    """Build a MagicMock that mimics FpgaBackend.sample() with optional failure."""
    mock_backend = MagicMock()
    call_count = [0]

    def fake_sample(biases, couplings, n_samples, config):
        call_count[0] += 1
        if fail_on_trial is not None and call_count[0] == fail_on_trial:
            raise RuntimeError("Simulated FPGA error on trial %d" % fail_on_trial)
        return np.zeros((n_samples, n_spins), dtype=bool)

    mock_backend.sample.side_effect = fake_sample
    return mock_backend


def test_run_hardware_benchmark_success(tmp_path):
    """REQ-SAMPLE-033-2: Successful benchmark returns n_trials latencies, hardware_failed=False."""
    mock_backend = _make_fake_fpga_backend(n_spins=10)
    mock_fpga_module = MagicMock()
    mock_fpga_module.FpgaBackend = lambda **kwargs: mock_backend
    with patch.dict("sys.modules", {"carnot.samplers.fpga_backend": mock_fpga_module}):
        result = run_hardware_benchmark("/fake/path.bit", n_spins=10, n_trials=5)

    assert result["hardware_failed"] is False
    assert result["n_completed"] == 5
    assert len(result["latencies_us"]) == 5
    assert result["error_message"] is None


def test_run_hardware_benchmark_exception(tmp_path):
    """REQ-SAMPLE-033-2: FpgaBackend raises → hardware_failed=True, partial results kept."""
    mock_backend = _make_fake_fpga_backend(n_spins=10, fail_on_trial=3)
    mock_fpga_module = MagicMock()
    mock_fpga_module.FpgaBackend = lambda **kwargs: mock_backend
    with patch.dict("sys.modules", {"carnot.samplers.fpga_backend": mock_fpga_module}):
        result = run_hardware_benchmark("/fake/path.bit", n_spins=10, n_trials=5)

    assert result["hardware_failed"] is True
    assert result["n_completed"] == 2  # 2 succeeded before failure on trial 3
    assert "Simulated FPGA error" in result["error_message"]


# ---------------------------------------------------------------------------
# main() integration — blocked path (Exp 584 bitfile_built=False)
# ---------------------------------------------------------------------------


def test_main_blocked_no_bitfile(tmp_path, monkeypatch):
    """SCENARIO-SAMPLE-055: bitfile_built=False → blocked artifact written, sys.exit not raised."""
    # Write a gate file with bitfile_built=False
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    gate_file = results_dir / "experiment_584_kv260_synthesis.json"
    gate_file.write_text(json.dumps({"bitfile_built": False, "bitfile_path": None}))

    deliverable = results_dir / "experiment_585_kv260_live_benchmark_v3.json"

    # Patch repo root and ExperimentTemplate/ExperimentTimeoutWatchdog
    import scripts.experiment_585_kv260_live_benchmark_v3 as exp585

    monkeypatch.setattr(exp585, "_REPO_ROOT", tmp_path)

    # Use a plain object so assert_deliverable_written isn't confused for a pytest assertion.
    deliverable_written_calls = []

    class FakeTmpl:
        def setup(self):
            pass
        def assert_deliverable_written(self):
            deliverable_written_calls.append(True)

    mock_watchdog = MagicMock()
    mock_watchdog.__enter__ = MagicMock(return_value=mock_watchdog)
    mock_watchdog.__exit__ = MagicMock(return_value=False)

    with patch.object(exp585, "ExperimentTemplate", return_value=FakeTmpl()):
        with patch.object(exp585, "ExperimentTimeoutWatchdog", return_value=mock_watchdog):
            exp585.main()

    assert deliverable.exists()
    artifact = json.loads(deliverable.read_text())
    assert artifact["honest_verdict"] == "blocked_no_bitfile"
    assert artifact["bitfile_built"] is False
    assert artifact["upstream_exp"] == 584
    assert artifact["schema"] == SCHEMA
    assert len(deliverable_written_calls) == 1


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


def test_constants():
    """Verify critical constants match the spec."""
    assert CPU_BASELINE_LATENCY_US == 289608.0
    assert FPGA_LATENCY_TARGET_US == 100.0
    assert N_SPINS == 100
    assert N_TRIALS == 1000
    assert EXPERIMENT_ID == 585
    assert SCHEMA == "carnot.kv260_benchmark.v3"
    assert DELIVERABLE == "results/experiment_585_kv260_live_benchmark_v3.json"
