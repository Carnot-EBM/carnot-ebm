"""Tests for Exp 932: DualGPU Throughput Benchmark — 50 GSM8K Questions.

Covers the logic added by this experiment:
  (a) build_corpus() shape and content
  (b) compute_verdict() threshold boundaries
  (c) build_pipeline() single-GPU vs dual-GPU configuration
  (d) main() integration: produces a valid result file

All GPU/sleep paths are mocked so the suite runs on any CI host.

Spec: REQ-PERF-004, SCENARIO-PERF-004
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT))

import scripts.experiment_932_dualgpu_throughput_benchmark as _mod  # noqa: E402


# ---------------------------------------------------------------------------
# (a) build_corpus()
# ---------------------------------------------------------------------------


class TestBuildCorpus:
    """Verify corpus shape and content invariants."""

    def test_returns_50_items(self):
        """build_corpus must return exactly 50 response dicts and 50 labels."""
        dicts, gt = _mod.build_corpus()
        assert len(dicts) == 50
        assert len(gt) == 50

    def test_all_ground_truth_true(self):
        """All ground-truth labels must be True (throughput benchmark, not accuracy)."""
        _, gt = _mod.build_corpus()
        assert all(gt)

    def test_response_dict_keys(self):
        """Each response dict must have 'question', 'response', 'attention_matrix'."""
        dicts, _ = _mod.build_corpus()
        for d in dicts:
            assert "question" in d
            assert "response" in d
            assert "attention_matrix" in d

    def test_attention_matrix_is_none(self):
        """attention_matrix must be None so Tier 1 (SinkProbe) is skipped in CI."""
        dicts, _ = _mod.build_corpus()
        assert all(d["attention_matrix"] is None for d in dicts)


# ---------------------------------------------------------------------------
# (b) compute_verdict()
# ---------------------------------------------------------------------------


class TestComputeVerdict:
    """Verify the three verdict branches."""

    def test_confirmed_at_threshold(self):
        assert _mod.compute_verdict(1.4) == "dualgpu_speedup_confirmed"

    def test_confirmed_above_threshold(self):
        assert _mod.compute_verdict(2.0) == "dualgpu_speedup_confirmed"

    def test_partial_just_above_one(self):
        assert _mod.compute_verdict(1.01) == "dualgpu_speedup_partial"

    def test_partial_just_below_threshold(self):
        assert _mod.compute_verdict(1.399) == "dualgpu_speedup_partial"

    def test_no_speedup_at_one(self):
        assert _mod.compute_verdict(1.0) == "dualgpu_no_speedup"

    def test_no_speedup_below_one(self):
        assert _mod.compute_verdict(0.5) == "dualgpu_no_speedup"


# ---------------------------------------------------------------------------
# (c) build_pipeline()
# ---------------------------------------------------------------------------


class TestBuildPipeline:
    """Verify pipeline construction for both modes."""

    def test_single_gpu_no_runner(self):
        """Single-GPU pipeline must have no runner wired."""
        pipeline = _mod.build_pipeline(dual_gpu_enabled=False, latency_s=0.0)
        assert pipeline._dual_gpu_runner is None

    def test_dual_gpu_runner_wired(self):
        """Dual-GPU pipeline must have a runner wired."""
        pipeline = _mod.build_pipeline(dual_gpu_enabled=True, latency_s=0.0)
        assert pipeline._dual_gpu_runner is not None

    def test_dual_gpu_flag_propagated(self):
        """DUAL_GPU_ENABLED class attribute must match the requested mode."""
        from carnot.pipeline.three_tier_pipeline import ThreeTierPipeline

        _mod.build_pipeline(dual_gpu_enabled=True, latency_s=0.0)
        assert ThreeTierPipeline.DUAL_GPU_ENABLED is True

        _mod.build_pipeline(dual_gpu_enabled=False, latency_s=0.0)
        assert ThreeTierPipeline.DUAL_GPU_ENABLED is False


# ---------------------------------------------------------------------------
# (d) main() integration
# ---------------------------------------------------------------------------


class TestMainIntegration:
    """Verify that main() produces a valid result file without real GPU."""

    def test_main_writes_deliverable(self, tmp_path, monkeypatch):
        """main() must write experiment_932_*.json with all required schema fields."""
        out_file = tmp_path / "results" / "experiment_932_dualgpu_throughput_benchmark.json"

        # Redirect the output path inside the module.
        monkeypatch.setattr(_mod, "_REPO_ROOT", tmp_path)
        (tmp_path / "results").mkdir()
        (tmp_path / "checkpoints").mkdir(exist_ok=True)

        # Use zero latency so the test completes instantly.
        monkeypatch.setattr(_mod, "_PER_CALL_LATENCY_S", 0.0)

        # Mock ExperimentTemplate so we don't write to the real results dir.
        mock_tmpl = MagicMock()
        # Explicitly set assert_deliverable_written so MagicMock doesn't intercept it
        # as a pytest-style assertion (MagicMock raises AttributeError for assert_* names).
        mock_tmpl.assert_deliverable_written = MagicMock(return_value=None)
        mock_tmpl.build_result.side_effect = lambda payload, **kw: {
            "experiment": 932,
            "title": "DualGPU Throughput Benchmark — 50 GSM8K Questions",
            "run_date": "20260426",
            "started_at": "2026-04-26T00:00:00Z",
            "finished_at": "2026-04-26T00:00:00Z",
            "duration_s": 0.01,
            "status": kw.get("status", "success"),
            "schema": sorted(list(payload.keys()) + [
                "experiment", "title", "run_date", "started_at",
                "finished_at", "duration_s", "status",
            ]),
            **payload,
        }

        with patch(
            "scripts.experiment_932_dualgpu_throughput_benchmark.ExperimentTemplate",
            return_value=mock_tmpl,
        ):
            _mod.main()

        assert out_file.exists(), "Deliverable JSON was not written"
        data = json.loads(out_file.read_text())

        # Must contain all required fields from experiment template contract.
        for field in ("experiment", "run_date", "started_at", "finished_at", "duration_s", "status"):
            assert field in data, f"Missing required field: {field}"

        # Must contain throughput and verdict fields.
        assert "honest_verdict" in data
        assert data["n_questions"] == 50
        assert data["honest_verdict"] in (
            "dualgpu_speedup_confirmed",
            "dualgpu_speedup_partial",
            "dualgpu_no_speedup",
        )
