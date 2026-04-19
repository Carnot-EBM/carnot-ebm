"""Tests for Experiment 539: Live 100q VeriCoT+VPRM v8.

Covers:
  - GPU-required fast-path (CARNOT_FORCE_LIVE not set)
  - n_questions dynamic calculation from Exp 538 latency
  - Wilson CI computation and retro_038_closed gate

Spec: REQ-BENCH-016 (v2), SCENARIO-BENCH-036 (v2), SCENARIO-BENCH-037 (v2)
"""

from __future__ import annotations

import json
import math
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Import the module under test
# ---------------------------------------------------------------------------

import sys

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from scripts.experiment_539_live_100q_vericot_v8 import (
    compute_n_questions_from_latency,
    compute_wilson_ci,
    compute_wilson_ci_on_improvement,
    _build_v8_artifact,
    MAX_QUESTIONS,
    INFERENCE_BUDGET_MINUTES,
)


# ---------------------------------------------------------------------------
# SCENARIO-BENCH-036 (v2): n_questions dynamic calculation
# ---------------------------------------------------------------------------


def test_compute_n_questions_from_latency_normal(tmp_path: Path) -> None:
    """SCENARIO-BENCH-036 (v2): 24.1s latency → n_questions = 100."""
    result_file = tmp_path / "exp538.json"
    result_file.write_text(json.dumps({"mean_latency_s": 24.1}))

    n = compute_n_questions_from_latency(result_file)

    expected = min(MAX_QUESTIONS, int(INFERENCE_BUDGET_MINUTES * 60 / 24.1))
    assert n == expected
    assert n == 100  # 80*60/24.1 = 199, capped at 100


def test_compute_n_questions_from_latency_high_latency(tmp_path: Path) -> None:
    """SCENARIO-BENCH-036 (v2): >40s latency → n_questions ≤ 50."""
    result_file = tmp_path / "exp538.json"
    result_file.write_text(json.dumps({"mean_latency_s": 100.0}))

    n = compute_n_questions_from_latency(result_file)

    # 80*60/100 = 48 < 50
    assert n <= 50
    assert n == 48


def test_compute_n_questions_from_latency_missing_file(tmp_path: Path) -> None:
    """Missing Exp 538 result falls back to 50q."""
    n = compute_n_questions_from_latency(tmp_path / "nonexistent.json")
    assert n == 50


def test_compute_n_questions_from_latency_zero_latency(tmp_path: Path) -> None:
    """Zero latency falls back to 50q (avoids divide-by-zero)."""
    result_file = tmp_path / "exp538.json"
    result_file.write_text(json.dumps({"mean_latency_s": 0.0}))

    n = compute_n_questions_from_latency(result_file)
    assert n == 50


# ---------------------------------------------------------------------------
# SCENARIO-BENCH-037 (v2): Wilson CI computation
# ---------------------------------------------------------------------------


def test_compute_wilson_ci_bounds_valid() -> None:
    """Wilson CI lower <= p_hat <= upper and both in [0, 1]."""
    lo, hi = compute_wilson_ci(60, 100)
    p_hat = 0.60
    assert 0.0 <= lo <= p_hat <= hi <= 1.0


def test_compute_wilson_ci_zero_total() -> None:
    """n=0 returns (0.0, 0.0) without error."""
    assert compute_wilson_ci(0, 0) == (0.0, 0.0)


def test_compute_wilson_ci_extreme_proportions() -> None:
    """Wilson CI handles p=0 and p=1 without negative bounds."""
    lo_zero, hi_zero = compute_wilson_ci(0, 100)
    lo_one, hi_one = compute_wilson_ci(100, 100)
    assert lo_zero >= 0.0
    assert hi_one <= 1.0


def test_compute_wilson_ci_on_improvement_excludes_zero() -> None:
    """SCENARIO-BENCH-037 (v2): large improvement → CI excludes zero."""
    # 80% pipeline vs 20% baseline over 100q — CI should clearly exclude 0
    lo, hi, excludes = compute_wilson_ci_on_improvement(
        baseline_correct=20, pipeline_correct=80, n_total=100
    )
    assert excludes is True
    assert lo > 0.0
    assert hi > lo


def test_compute_wilson_ci_on_improvement_no_difference() -> None:
    """Equal baseline and pipeline → CI does not exclude zero."""
    lo, hi, excludes = compute_wilson_ci_on_improvement(
        baseline_correct=50, pipeline_correct=50, n_total=100
    )
    assert excludes is False


def test_compute_wilson_ci_on_improvement_small_gain() -> None:
    """Small gain (1pp) on small n → CI straddles zero → not publishable."""
    lo, hi, excludes = compute_wilson_ci_on_improvement(
        baseline_correct=5, pipeline_correct=6, n_total=10
    )
    assert excludes is False


# ---------------------------------------------------------------------------
# _build_v8_artifact
# ---------------------------------------------------------------------------


def test_build_v8_artifact_fields_complete() -> None:
    """Artifact contains all required schema fields."""
    art = _build_v8_artifact(
        n_questions=100,
        baseline_correct=50,
        pipeline_correct=60,
        n_scored=100,
        per_question_latencies=[1.0] * 100,
        inference_mode="live_gpu",
        wilson_ci_lower=0.01,
        wilson_ci_upper=0.19,
        retro_038_closed=True,
        env_autofix_dict={},
    )
    required = [
        "schema", "inference_mode", "n_questions", "n_scored",
        "baseline_accuracy", "pipeline_accuracy", "signed_improvement",
        "wilson_ci_lower", "wilson_ci_upper", "retro_038_closed",
        "mean_latency_s", "per_question_latencies", "honest_verdict",
    ]
    for field in required:
        assert field in art, f"Missing required field: {field}"
    assert art["schema"] == "carnot.vericot_benchmark.v8"


def test_build_v8_artifact_gpu_required_verdict() -> None:
    """inference_mode='gpu_required' → honest_verdict='gpu_required'."""
    art = _build_v8_artifact(
        n_questions=100,
        baseline_correct=0, pipeline_correct=0, n_scored=0,
        per_question_latencies=[],
        inference_mode="gpu_required",
        wilson_ci_lower=0.0, wilson_ci_upper=0.0,
        retro_038_closed=False,
        env_autofix_dict={},
    )
    assert art["honest_verdict"] == "gpu_required"
    assert art["retro_038_closed"] is False


def test_build_v8_artifact_publishable_verdict() -> None:
    """retro_038_closed=True → honest_verdict='wilson_ci_publishable'."""
    art = _build_v8_artifact(
        n_questions=100,
        baseline_correct=50, pipeline_correct=60, n_scored=100,
        per_question_latencies=[1.0] * 100,
        inference_mode="live_gpu",
        wilson_ci_lower=0.02, wilson_ci_upper=0.18,
        retro_038_closed=True,
        env_autofix_dict={},
    )
    assert art["honest_verdict"] == "wilson_ci_publishable"
    assert art["retro_038_closed"] is True
    assert abs(art["signed_improvement"] - 0.10) < 1e-9


def test_build_v8_artifact_no_improvement_verdict() -> None:
    """No improvement → honest_verdict='no_improvement'."""
    art = _build_v8_artifact(
        n_questions=100,
        baseline_correct=60, pipeline_correct=60, n_scored=100,
        per_question_latencies=[1.0] * 100,
        inference_mode="live_gpu",
        wilson_ci_lower=-0.05, wilson_ci_upper=0.05,
        retro_038_closed=False,
        env_autofix_dict={},
    )
    assert art["honest_verdict"] == "no_improvement"


# ---------------------------------------------------------------------------
# GPU-required fast-path (integration-level, patching LiveGPUGate)
# ---------------------------------------------------------------------------


def test_run_experiment_gpu_required_fast_path(tmp_path: Path) -> None:
    """CARNOT_FORCE_LIVE not set → gpu_required artifact written immediately."""
    # Write a fake Exp 538 result
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    (results_dir / "experiment_538_live_25q_precision_v9.json").write_text(
        json.dumps({"mean_latency_s": 24.1})
    )

    from scripts.experiment_539_live_100q_vericot_v8 import run_experiment

    # Patch LiveGPUGate to simulate "not live" (returns a non-None gate result)
    mock_gate_result = MagicMock()
    mock_gate_result.__str__ = lambda self: "gpu_required"

    with (
        patch(
            "scripts.experiment_539_live_100q_vericot_v8.LiveGPUGate.require_live_or_blocked",
            return_value=mock_gate_result,
        ),
        patch(
            "scripts.experiment_539_live_100q_vericot_v8.ExperimentTemplate.kill_gpu_zombies",
        ),
        patch(
            "scripts.experiment_539_live_100q_vericot_v8.ExperimentTemplate.setup",
        ),
        patch(
            "scripts.experiment_539_live_100q_vericot_v8.ExperimentTemplate.build_result",
            side_effect=lambda fields, status: {**fields, "status": status},
        ),
        patch(
            "scripts.experiment_539_live_100q_vericot_v8.ExperimentTemplate.assert_deliverable_written",
        ),
        patch(
            "scripts.experiment_539_live_100q_vericot_v8.DeliverableGuard",
        ),
    ):
        artifact = run_experiment(repo_root=tmp_path)

    assert artifact["honest_verdict"] == "gpu_required"
    deliverable = tmp_path / "results" / "experiment_539_live_100q_vericot_v8.json"
    assert deliverable.exists()
    written = json.loads(deliverable.read_text())
    assert written["honest_verdict"] == "gpu_required"
