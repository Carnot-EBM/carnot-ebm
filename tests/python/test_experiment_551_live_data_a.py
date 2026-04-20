"""Tests for Exp 551: live 50q data collection A.

100% coverage on functions added in scripts/experiment_551_live_data_a.py.

Spec: REQ-DATA-001, REQ-DATA-002,
      SCENARIO-DATA-001, SCENARIO-DATA-002, SCENARIO-DATA-003
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import scripts.experiment_551_live_data_a as exp551


# ---------------------------------------------------------------------------
# _build_live_data_artifact
# ---------------------------------------------------------------------------


class TestBuildLiveDataArtifact:
    """REQ-DATA-001: artifact must have all required fields."""

    def test_schema_field(self):
        art = exp551._build_live_data_artifact("live_gpu", 50, 100, "results/live_pairs_551.json", [1.0] * 50)
        assert art["schema"] == "carnot.live_data_collection.v1"

    def test_honest_verdict_live_data_collected(self):
        # SCENARIO-DATA-001: >=50 pairs -> live_data_collected
        art = exp551._build_live_data_artifact("live_gpu", 50, 50, "results/live_pairs_551.json", [])
        assert art["honest_verdict"] == "live_data_collected"

    def test_honest_verdict_partial_collection(self):
        art = exp551._build_live_data_artifact("live_gpu", 50, 30, "results/live_pairs_551.json", [])
        assert art["honest_verdict"] == "partial_collection"

    def test_honest_verdict_gpu_required(self):
        # SCENARIO-DATA-003: blocked when GPU not live
        art = exp551._build_live_data_artifact("gpu_required", 0, 0, None, [])
        assert art["honest_verdict"] == "gpu_required"

    def test_n_pairs_collected(self):
        art = exp551._build_live_data_artifact("live_gpu", 50, 77, "f.json", [1.0] * 50)
        assert art["n_pairs_collected"] == 77

    def test_models_field(self):
        art = exp551._build_live_data_artifact("live_gpu", 50, 50, "f.json", [])
        assert "google/gemma-4-E4B-it" in art["models"]
        assert "Qwen/Qwen3.5-0.8B" in art["models"]

    def test_question_indices_field(self):
        art = exp551._build_live_data_artifact("live_gpu", 50, 50, "f.json", [])
        assert art["question_indices"] == "0-49"

    def test_mean_latency_calculated(self):
        art = exp551._build_live_data_artifact("live_gpu", 50, 50, "f.json", [2.0, 4.0])
        assert abs(art["mean_latency_s"] - 3.0) < 1e-9

    def test_mean_latency_empty(self):
        art = exp551._build_live_data_artifact("live_gpu", 50, 50, "f.json", [])
        assert art["mean_latency_s"] == 0.0

    def test_live_pairs_file_field(self):
        art = exp551._build_live_data_artifact("live_gpu", 50, 50, "results/live_pairs_551.json", [])
        assert art["live_pairs_file"] == "results/live_pairs_551.json"


# ---------------------------------------------------------------------------
# _write_json_atomic
# ---------------------------------------------------------------------------


class TestWriteJsonAtomic:
    """REQ-DATA-002: atomic write via .tmp then rename."""

    def test_writes_valid_json(self, tmp_path):
        p = tmp_path / "out.json"
        exp551._write_json_atomic(p, {"key": "value"})
        assert json.loads(p.read_text()) == {"key": "value"}

    def test_no_tmp_file_left(self, tmp_path):
        p = tmp_path / "out.json"
        exp551._write_json_atomic(p, {"x": 1})
        assert not (tmp_path / "out.tmp").exists()

    def test_creates_parent_dirs(self, tmp_path):
        p = tmp_path / "sub" / "dir" / "out.json"
        exp551._write_json_atomic(p, {})
        assert p.exists()

    def test_overwrites_existing(self, tmp_path):
        p = tmp_path / "out.json"
        p.write_text('{"old": true}')
        exp551._write_json_atomic(p, {"new": True})
        assert json.loads(p.read_text()) == {"new": True}


# ---------------------------------------------------------------------------
# _annotate_response
# ---------------------------------------------------------------------------


class TestAnnotateResponse:
    """SCENARIO-DATA-002: each pair records cot_steps and fover_labels."""

    def test_returns_cot_steps_and_labels(self):
        from carnot.pipeline.fover_annotator import FOVERAnnotator
        annotator = FOVERAnnotator()
        result = exp551._annotate_response(annotator, "1. 2 + 2 = 4", "q0")
        assert "cot_steps" in result
        assert "fover_labels" in result

    def test_cot_steps_have_required_keys(self):
        from carnot.pipeline.fover_annotator import FOVERAnnotator
        annotator = FOVERAnnotator()
        result = exp551._annotate_response(annotator, "1. 2 + 2 = 4", "q0")
        for step in result["cot_steps"]:
            assert "step_idx" in step
            assert "z3_label" in step

    def test_fover_labels_parallel_to_cot_steps(self):
        from carnot.pipeline.fover_annotator import FOVERAnnotator
        annotator = FOVERAnnotator()
        result = exp551._annotate_response(annotator, "1. Step A\n2. Step B", "q1")
        assert len(result["cot_steps"]) == len(result["fover_labels"])

    def test_empty_response_returns_single_step(self):
        from carnot.pipeline.fover_annotator import FOVERAnnotator
        annotator = FOVERAnnotator()
        result = exp551._annotate_response(annotator, "No steps here", "q2")
        # Single-step response: parse_cot_into_steps returns one step
        assert len(result["cot_steps"]) >= 1


# ---------------------------------------------------------------------------
# _load_gsm8k_questions
# ---------------------------------------------------------------------------


class TestLoadGsm8kQuestions:
    """Verify question loading falls back to synthetic when datasets unavailable."""

    def test_synthetic_fallback_returns_n_questions(self):
        with patch.dict("sys.modules", {"datasets": None}):
            qs = exp551._load_gsm8k_questions(5, 42)
        assert len(qs) == 5

    def test_synthetic_fallback_has_index(self):
        with patch.dict("sys.modules", {"datasets": None}):
            qs = exp551._load_gsm8k_questions(3, 42)
        assert all("index" in q for q in qs)

    def test_synthetic_fallback_index_sequential(self):
        with patch.dict("sys.modules", {"datasets": None}):
            qs = exp551._load_gsm8k_questions(5, 42)
        assert [q["index"] for q in qs] == list(range(5))


# ---------------------------------------------------------------------------
# _qwen_generate
# ---------------------------------------------------------------------------


class TestQwenGenerate:
    """Verify _qwen_generate normalises pipeline output."""

    def test_list_dict_output(self):
        pipeline = MagicMock(return_value=[{"generated_text": "hello"}])
        assert exp551._qwen_generate(pipeline, "prompt") == "hello"

    def test_exception_returns_error_string(self):
        pipeline = MagicMock(side_effect=RuntimeError("boom"))
        result = exp551._qwen_generate(pipeline, "prompt")
        assert "qwen_error" in result

    def test_non_list_output(self):
        pipeline = MagicMock(return_value="direct_string")
        result = exp551._qwen_generate(pipeline, "prompt")
        assert "direct_string" in result


# ---------------------------------------------------------------------------
# run_experiment — blocked path (SCENARIO-DATA-003)
# ---------------------------------------------------------------------------


class TestRunExperimentBlocked:
    """SCENARIO-DATA-003: blocked artifact written when GPU not live."""

    def test_blocked_when_force_live_not_set(self, tmp_path):
        with patch.dict(os.environ, {"CARNOT_FORCE_LIVE": "0"}, clear=False):
            with patch.object(exp551.ExperimentTemplate, "kill_gpu_zombies", return_value={}):
                result = exp551.run_experiment(repo_root=tmp_path)
        assert result["status"] == "blocked"
        assert (tmp_path / exp551.DELIVERABLE).exists()
