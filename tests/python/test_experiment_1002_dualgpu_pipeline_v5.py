"""Tests for Experiment 1002 — DualGPU Pipeline v5 throughput benchmark logic.

Spec: REQ-GPU-010, REQ-INFRA-007
"""

from __future__ import annotations

import time
from pathlib import Path


# ---------------------------------------------------------------------------
# Import the functions we want to test directly from the script.
# We add the scripts/ directory to sys.path inside fixtures rather than
# relying on a package install so these tests work in plain CI.
# ---------------------------------------------------------------------------

import sys

_SCRIPTS_DIR = Path(__file__).parent.parent.parent / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import experiment_1002_dualgpu_pipeline_v5 as exp1002  # noqa: E402


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_check_dualgpu_importable_finds_class():
    """DualGPURunner source file exists and contains the class definition."""
    ok, reason = exp1002._check_dualgpu_importable()
    assert ok, f"Expected DualGPURunner to be found, got: {reason}"
    assert "DualGPURunner" in reason


def test_check_cuda_returns_tuple():
    """_check_cuda always returns a (bool, int) even without torch installed."""
    cuda_ok, n_gpus = exp1002._check_cuda()
    assert isinstance(cuda_ok, bool)
    assert isinstance(n_gpus, int)
    assert n_gpus >= 0


def test_probe_verify_repair_wiring_finds_flag():
    """DUAL_GPU_ENABLED is present in verify_repair.py source."""
    wiring = exp1002._probe_verify_repair_wiring()
    assert wiring["error"] is None, f"Probe error: {wiring['error']}"
    assert wiring["dual_gpu_flag_present"] is True
    assert wiring["dual_gpu_env_var"] == "CARNOT_DUAL_GPU"


def test_sequential_throughput_positive():
    """Sequential benchmark produces a non-zero throughput."""
    questions = [f"Q{i}" for i in range(4)]
    gen = exp1002._make_mock_generate(gpu_id=0, latency_s=0.001)
    elapsed, tput = exp1002._run_sequential(questions, gen)
    assert elapsed > 0
    assert tput > 0


def test_dualgpu_throughput_higher_than_sequential():
    """DualGPU benchmark achieves >1.5x the sequential throughput."""
    questions = [f"Q{i}" for i in range(10)]
    gen0 = exp1002._make_mock_generate(gpu_id=0, latency_s=0.01)
    gen1 = exp1002._make_mock_generate(gpu_id=1, latency_s=0.01)
    gen_seq = exp1002._make_mock_generate(gpu_id=0, latency_s=0.01)

    _, seq_tput = exp1002._run_sequential(questions, gen_seq)
    _, dual_tput = exp1002._run_dualgpu(questions, [gen0, gen1])

    ratio = dual_tput / seq_tput
    assert ratio >= 1.5, f"Expected ratio >= 1.5, got {ratio:.3f}"


def test_result_file_has_required_fields():
    """The written JSON result file contains all required schema fields."""
    result_path = (
        Path(__file__).parent.parent.parent / "results" / "experiment_1002_dualgpu_pipeline_v5.json"
    )
    assert result_path.exists(), "Result file must exist after experiment run"
    import json

    data = json.loads(result_path.read_text())

    required = [
        "experiment",
        "run_date",
        "dualgpu_wired",
        "throughput_ratio",
        "inference_mode",
        "honest_verdict",
        "status",
    ]
    for field in required:
        assert field in data, f"Missing required field: {field}"

    assert data["run_date"] == "20260428"
    assert data["experiment"] == 1002
    assert isinstance(data["dualgpu_wired"], bool)
    assert isinstance(data["throughput_ratio"], float)
    assert data["honest_verdict"] in {
        "dualgpu_production_wired",
        "wired_synthetic_only",
        "wiring_failed",
    }
    assert data["inference_mode"] in {"live_gpu", "synthetic_validation"}
