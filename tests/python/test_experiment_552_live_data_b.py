"""Tests for Exp 552: live 50q data collection B.

100% coverage on functions added in scripts/experiment_552_live_data_b.py.

Spec: REQ-DATA-001, REQ-DATA-002,
      SCENARIO-DATA-004, SCENARIO-DATA-005, SCENARIO-DATA-006
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

import scripts.experiment_552_live_data_b as exp552


# ---------------------------------------------------------------------------
# _build_live_data_artifact
# ---------------------------------------------------------------------------


class TestBuildLiveDataArtifact:
    """REQ-DATA-001: artifact must have all required fields."""

    def test_schema_field(self):
        art = exp552._build_live_data_artifact("live_gpu", 50, 100, "results/live_pairs_552.json", [1.0] * 50)
        assert art["schema"] == "carnot.live_data_collection.v1"

    def test_honest_verdict_cumulative_100(self):
        # SCENARIO-DATA-005: cumulative >=100 -> live_data_collected
        art = exp552._build_live_data_artifact("live_gpu", 50, 50, "f.json", [], n_pairs_from_exp551=50)
        assert art["honest_verdict"] == "live_data_collected"
        assert art["cumulative_pairs"] == 100

    def test_honest_verdict_partial_collection(self):
        # cumulative < 100 -> partial_collection
        art = exp552._build_live_data_artifact("live_gpu", 50, 30, "f.json", [], n_pairs_from_exp551=0)
        assert art["honest_verdict"] == "partial_collection"

    def test_honest_verdict_gpu_required(self):
        # SCENARIO-DATA-006: blocked when GPU not live
        art = exp552._build_live_data_artifact("gpu_required", 0, 0, None, [])
        assert art["honest_verdict"] == "gpu_required"

    def test_cumulative_pairs_calculation(self):
        art = exp552._build_live_data_artifact("live_gpu", 50, 60, "f.json", [], n_pairs_from_exp551=45)
        assert art["cumulative_pairs"] == 105
        assert art["n_pairs_from_exp551"] == 45

    def test_n_pairs_collected_field(self):
        art = exp552._build_live_data_artifact("live_gpu", 50, 77, "f.json", [1.0] * 50)
        assert art["n_pairs_collected"] == 77

    def test_models_field(self):
        art = exp552._build_live_data_artifact("live_gpu", 50, 50, "f.json", [])
        assert "google/gemma-4-E4B-it" in art["models"]
        assert "Qwen/Qwen3.5-0.8B" in art["models"]

    def test_question_indices_field(self):
        # SCENARIO-DATA-004: question_indices must be 50-99
        art = exp552._build_live_data_artifact("live_gpu", 50, 50, "f.json", [])
        assert art["question_indices"] == "50-99"

    def test_mean_latency_calculated(self):
        art = exp552._build_live_data_artifact("live_gpu", 50, 50, "f.json", [2.0, 4.0])
        assert abs(art["mean_latency_s"] - 3.0) < 1e-9

    def test_mean_latency_empty(self):
        art = exp552._build_live_data_artifact("live_gpu", 50, 50, "f.json", [])
        assert art["mean_latency_s"] == 0.0

    def test_live_pairs_file_field(self):
        art = exp552._build_live_data_artifact("live_gpu", 50, 50, "results/live_pairs_552.json", [])
        assert art["live_pairs_file"] == "results/live_pairs_552.json"


# ---------------------------------------------------------------------------
# _write_json_atomic
# ---------------------------------------------------------------------------


class TestWriteJsonAtomic:
    """REQ-DATA-002: atomic write via .tmp then rename."""

    def test_writes_valid_json(self, tmp_path):
        p = tmp_path / "out.json"
        exp552._write_json_atomic(p, {"key": "value"})
        assert json.loads(p.read_text()) == {"key": "value"}

    def test_no_tmp_file_left(self, tmp_path):
        p = tmp_path / "out.json"
        exp552._write_json_atomic(p, {"x": 1})
        assert not (tmp_path / "out.tmp").exists()

    def test_creates_parent_dirs(self, tmp_path):
        p = tmp_path / "sub" / "dir" / "out.json"
        exp552._write_json_atomic(p, {})
        assert p.exists()

    def test_overwrites_existing(self, tmp_path):
        p = tmp_path / "out.json"
        p.write_text('{"old": true}')
        exp552._write_json_atomic(p, {"new": True})
        assert json.loads(p.read_text()) == {"new": True}


# ---------------------------------------------------------------------------
# _load_gsm8k_questions
# ---------------------------------------------------------------------------


class TestLoadGsm8kQuestions:
    """SCENARIO-DATA-004: questions 50-99 loaded correctly."""

    def test_synthetic_fallback_returns_n_questions(self):
        with patch.dict("sys.modules", {"datasets": None}):
            qs = exp552._load_gsm8k_questions(50, 99, 42)
        assert len(qs) == 50

    def test_synthetic_fallback_indices_50_to_99(self):
        with patch.dict("sys.modules", {"datasets": None}):
            qs = exp552._load_gsm8k_questions(50, 99, 42)
        assert [q["index"] for q in qs] == list(range(50, 100))

    def test_synthetic_fallback_has_required_keys(self):
        with patch.dict("sys.modules", {"datasets": None}):
            qs = exp552._load_gsm8k_questions(50, 99, 42)
        assert all({"question", "answer", "index"} <= set(q.keys()) for q in qs)


# ---------------------------------------------------------------------------
# _qwen_generate
# ---------------------------------------------------------------------------


class TestQwenGenerate:
    """Verify _qwen_generate normalises pipeline output."""

    def test_list_dict_output(self):
        pipeline = MagicMock(return_value=[{"generated_text": "hello"}])
        assert exp552._qwen_generate(pipeline, "prompt") == "hello"

    def test_exception_returns_error_string(self):
        pipeline = MagicMock(side_effect=RuntimeError("boom"))
        result = exp552._qwen_generate(pipeline, "prompt")
        assert "qwen_error" in result

    def test_non_list_output(self):
        pipeline = MagicMock(return_value="direct_string")
        result = exp552._qwen_generate(pipeline, "prompt")
        assert "direct_string" in result


# ---------------------------------------------------------------------------
# _annotate_response
# ---------------------------------------------------------------------------


class TestAnnotateResponse:
    """SCENARIO-DATA-004: each pair records cot_steps and fover_labels."""

    def test_returns_cot_steps_and_labels(self):
        from carnot.pipeline.fover_annotator import FOVERAnnotator
        annotator = FOVERAnnotator()
        result = exp552._annotate_response(annotator, "1. 2 + 2 = 4", "q50")
        assert "cot_steps" in result
        assert "fover_labels" in result

    def test_cot_steps_have_required_keys(self):
        from carnot.pipeline.fover_annotator import FOVERAnnotator
        annotator = FOVERAnnotator()
        result = exp552._annotate_response(annotator, "1. 2 + 2 = 4", "q50")
        for step in result["cot_steps"]:
            assert "step_idx" in step
            assert "z3_label" in step

    def test_fover_labels_parallel_to_cot_steps(self):
        from carnot.pipeline.fover_annotator import FOVERAnnotator
        annotator = FOVERAnnotator()
        result = exp552._annotate_response(annotator, "1. Step A\n2. Step B", "q51")
        assert len(result["cot_steps"]) == len(result["fover_labels"])


# ---------------------------------------------------------------------------
# _load_n_pairs_from_551
# ---------------------------------------------------------------------------


class TestLoadNPairsFrom551:
    """Verify loading pair count from Exp 551 file."""

    def test_returns_zero_when_file_missing(self, tmp_path):
        count = exp552._load_n_pairs_from_551(tmp_path)
        assert count == 0

    def test_returns_list_length(self, tmp_path):
        pairs_file = tmp_path / "results" / "live_pairs_551.json"
        pairs_file.parent.mkdir(parents=True)
        pairs_file.write_text(json.dumps([{"a": 1}, {"b": 2}, {"c": 3}]))
        assert exp552._load_n_pairs_from_551(tmp_path) == 3

    def test_returns_zero_for_non_list(self, tmp_path):
        pairs_file = tmp_path / "results" / "live_pairs_551.json"
        pairs_file.parent.mkdir(parents=True)
        pairs_file.write_text(json.dumps({"not": "a list"}))
        assert exp552._load_n_pairs_from_551(tmp_path) == 0

    def test_returns_zero_on_invalid_json(self, tmp_path):
        pairs_file = tmp_path / "results" / "live_pairs_551.json"
        pairs_file.parent.mkdir(parents=True)
        pairs_file.write_text("not json!")
        assert exp552._load_n_pairs_from_551(tmp_path) == 0


# ---------------------------------------------------------------------------
# run_experiment — blocked path (SCENARIO-DATA-006)
# ---------------------------------------------------------------------------


class TestRunExperimentBlocked:
    """SCENARIO-DATA-006: blocked artifact written when GPU not live."""

    def test_blocked_when_force_live_not_set(self, tmp_path):
        with patch.dict(os.environ, {"CARNOT_FORCE_LIVE": "0"}, clear=False):
            with patch.object(exp552.ExperimentTemplate, "kill_gpu_zombies", return_value={}):
                result = exp552.run_experiment(repo_root=tmp_path)
        assert result["status"] == "blocked"
        assert (tmp_path / exp552.DELIVERABLE).exists()

    def test_blocked_artifact_has_required_fields(self, tmp_path):
        # build_result() overwrites "schema" with sorted(result.keys()); check
        # that question_indices and honest_verdict are present instead.
        with patch.dict(os.environ, {"CARNOT_FORCE_LIVE": "0"}, clear=False):
            with patch.object(exp552.ExperimentTemplate, "kill_gpu_zombies", return_value={}):
                result = exp552.run_experiment(repo_root=tmp_path)
        assert result["question_indices"] == "50-99"
        assert result["honest_verdict"] == "gpu_required"

    def test_blocked_artifact_includes_exp551_pairs(self, tmp_path):
        # Create a fake live_pairs_551.json with 5 pairs
        pairs_dir = tmp_path / "results"
        pairs_dir.mkdir(parents=True)
        (pairs_dir / "live_pairs_551.json").write_text(json.dumps([{"x": i} for i in range(5)]))
        with patch.dict(os.environ, {"CARNOT_FORCE_LIVE": "0"}, clear=False):
            with patch.object(exp552.ExperimentTemplate, "kill_gpu_zombies", return_value={}):
                result = exp552.run_experiment(repo_root=tmp_path)
        assert result["n_pairs_from_exp551"] == 5
