"""Tests for Exp 563: live 50q data collection A v2 (RETRO-062 fix).

100% targeted coverage on functions added in scripts/experiment_563_live_data_a_v2.py.

Spec: REQ-DATA-001, REQ-DATA-002,
      SCENARIO-DATA-010, SCENARIO-DATA-011, SCENARIO-DATA-012
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

import scripts.experiment_563_live_data_a_v2 as exp563


# ---------------------------------------------------------------------------
# _build_live_data_artifact
# ---------------------------------------------------------------------------


class TestBuildLiveDataArtifact:
    """REQ-DATA-001: artifact must have all required fields."""

    def test_schema_field(self):
        art = exp563._build_live_data_artifact("live_gpu", 50, 50, "results/live_pairs_563.json", [1.0] * 50)
        assert art["schema"] == "carnot.live_data_collection.v1"

    def test_honest_verdict_live_data_collected_at_40(self):
        # SCENARIO-DATA-010: n_pairs_collected >= 40 -> live_data_collected
        art = exp563._build_live_data_artifact("live_gpu", 50, 40, "f.json", [])
        assert art["honest_verdict"] == "live_data_collected"

    def test_honest_verdict_partial_below_40(self):
        # partial_collection when < 40 pairs
        art = exp563._build_live_data_artifact("live_gpu", 50, 30, "f.json", [])
        assert art["honest_verdict"] == "partial_collection"

    def test_honest_verdict_gpu_required(self):
        # SCENARIO-DATA-011: blocked when GPU not live
        art = exp563._build_live_data_artifact("gpu_required", 0, 0, None, [])
        assert art["honest_verdict"] == "gpu_required"

    def test_retro_062_resolved_true_when_40_pairs(self):
        # SCENARIO-DATA-010: retro_062_resolved=True when n_pairs_collected>=40
        art = exp563._build_live_data_artifact("live_gpu", 50, 40, "f.json", [])
        assert art["retro_062_resolved"] is True

    def test_retro_062_resolved_false_when_below_40(self):
        art = exp563._build_live_data_artifact("live_gpu", 50, 39, "f.json", [])
        assert art["retro_062_resolved"] is False

    def test_n_pairs_collected_field(self):
        art = exp563._build_live_data_artifact("live_gpu", 50, 77, "f.json", [1.0] * 50)
        assert art["n_pairs_collected"] == 77

    def test_models_field(self):
        art = exp563._build_live_data_artifact("live_gpu", 50, 50, "f.json", [])
        assert "google/gemma-4-E4B-it" in art["models"]
        assert "Qwen/Qwen3.5-0.8B" in art["models"]

    def test_question_indices_field(self):
        # SCENARIO-DATA-010: question_indices must be 0-49
        art = exp563._build_live_data_artifact("live_gpu", 50, 50, "f.json", [])
        assert art["question_indices"] == "0-49"

    def test_mean_latency_calculated(self):
        art = exp563._build_live_data_artifact("live_gpu", 50, 50, "f.json", [2.0, 4.0])
        assert abs(art["mean_latency_s"] - 3.0) < 1e-9

    def test_mean_latency_empty(self):
        art = exp563._build_live_data_artifact("live_gpu", 50, 50, "f.json", [])
        assert art["mean_latency_s"] == 0.0

    def test_live_pairs_file_field(self):
        art = exp563._build_live_data_artifact("live_gpu", 50, 50, "results/live_pairs_563.json", [])
        assert art["live_pairs_file"] == "results/live_pairs_563.json"

    def test_n_questions_field(self):
        art = exp563._build_live_data_artifact("live_gpu", 50, 50, "f.json", [])
        assert art["n_questions"] == 50

    def test_inference_mode_field(self):
        art = exp563._build_live_data_artifact("live_gpu", 50, 50, "f.json", [])
        assert art["inference_mode"] == "live_gpu"


# ---------------------------------------------------------------------------
# _write_json_atomic
# ---------------------------------------------------------------------------


class TestWriteJsonAtomic:
    """REQ-DATA-002: atomic write via .tmp then rename."""

    def test_writes_valid_json(self, tmp_path):
        p = tmp_path / "out.json"
        exp563._write_json_atomic(p, {"key": "value"})
        assert json.loads(p.read_text()) == {"key": "value"}

    def test_no_tmp_file_left(self, tmp_path):
        p = tmp_path / "out.json"
        exp563._write_json_atomic(p, {"x": 1})
        assert not (tmp_path / "out.tmp").exists()

    def test_creates_parent_dirs(self, tmp_path):
        p = tmp_path / "sub" / "dir" / "out.json"
        exp563._write_json_atomic(p, {})
        assert p.exists()

    def test_overwrites_existing(self, tmp_path):
        p = tmp_path / "out.json"
        p.write_text('{"old": true}')
        exp563._write_json_atomic(p, {"new": True})
        assert json.loads(p.read_text()) == {"new": True}


# ---------------------------------------------------------------------------
# _load_gsm8k_questions
# ---------------------------------------------------------------------------


class TestLoadGsm8kQuestions:
    """SCENARIO-DATA-010: questions 0-49 loaded correctly."""

    def test_synthetic_fallback_returns_n_questions(self):
        with patch.dict("sys.modules", {"datasets": None}):
            qs = exp563._load_gsm8k_questions(0, 49, 42)
        assert len(qs) == 50

    def test_synthetic_fallback_indices_0_to_49(self):
        with patch.dict("sys.modules", {"datasets": None}):
            qs = exp563._load_gsm8k_questions(0, 49, 42)
        assert [q["index"] for q in qs] == list(range(0, 50))

    def test_synthetic_fallback_has_required_keys(self):
        with patch.dict("sys.modules", {"datasets": None}):
            qs = exp563._load_gsm8k_questions(0, 49, 42)
        assert all({"question", "answer", "index"} <= set(q.keys()) for q in qs)

    def test_different_range(self):
        with patch.dict("sys.modules", {"datasets": None}):
            qs = exp563._load_gsm8k_questions(5, 9, 42)
        assert len(qs) == 5
        assert qs[0]["index"] == 5


# ---------------------------------------------------------------------------
# _qwen_generate
# ---------------------------------------------------------------------------


class TestQwenGenerate:
    """Verify _qwen_generate normalises pipeline output."""

    def test_list_dict_output(self):
        pipeline = MagicMock(return_value=[{"generated_text": "hello"}])
        assert exp563._qwen_generate(pipeline, "prompt") == "hello"

    def test_exception_returns_error_string(self):
        pipeline = MagicMock(side_effect=RuntimeError("boom"))
        result = exp563._qwen_generate(pipeline, "prompt")
        assert "qwen_error" in result

    def test_non_list_output(self):
        pipeline = MagicMock(return_value="direct_string")
        result = exp563._qwen_generate(pipeline, "prompt")
        assert "direct_string" in result

    def test_empty_list_returns_str(self):
        pipeline = MagicMock(return_value=[])
        result = exp563._qwen_generate(pipeline, "prompt")
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# _annotate_response
# ---------------------------------------------------------------------------


class TestAnnotateResponse:
    """SCENARIO-DATA-012: each pair records cot_steps and fover_labels."""

    def test_returns_cot_steps_and_labels(self):
        from carnot.pipeline.fover_annotator import FOVERAnnotator
        annotator = FOVERAnnotator()
        result = exp563._annotate_response(annotator, "1. 2 + 2 = 4", "q0")
        assert "cot_steps" in result
        assert "fover_labels" in result

    def test_cot_steps_have_required_keys(self):
        from carnot.pipeline.fover_annotator import FOVERAnnotator
        annotator = FOVERAnnotator()
        result = exp563._annotate_response(annotator, "1. 2 + 2 = 4", "q0")
        for step in result["cot_steps"]:
            assert "step_idx" in step
            assert "z3_label" in step

    def test_fover_labels_parallel_to_cot_steps(self):
        from carnot.pipeline.fover_annotator import FOVERAnnotator
        annotator = FOVERAnnotator()
        result = exp563._annotate_response(annotator, "1. Step A\n2. Step B", "q1")
        assert len(result["cot_steps"]) == len(result["fover_labels"])


# ---------------------------------------------------------------------------
# _write_blocked_preflight
# ---------------------------------------------------------------------------


class TestWriteBlockedPreflight:
    """SCENARIO-DATA-011: blocked artifact written when CARNOT_FORCE_LIVE not set."""

    def test_writes_blocked_artifact(self, tmp_path):
        output_path = tmp_path / "results" / "experiment_563_live_data_a_v2.json"
        artifact = exp563._write_blocked_preflight(output_path, "test reason")
        assert output_path.exists()
        on_disk = json.loads(output_path.read_text())
        assert on_disk["status"] == "blocked"

    def test_blocked_reason_preserved(self, tmp_path):
        output_path = tmp_path / "out.json"
        artifact = exp563._write_blocked_preflight(output_path, "CARNOT_FORCE_LIVE not set")
        assert "CARNOT_FORCE_LIVE not set" in artifact["blocked_reason"]

    def test_has_required_schema_fields(self, tmp_path):
        output_path = tmp_path / "out.json"
        artifact = exp563._write_blocked_preflight(output_path, "blocked")
        assert artifact["schema"] == "carnot.live_data_collection.v1"
        assert artifact["experiment"] == 563

    def test_honest_verdict_is_gpu_required(self, tmp_path):
        output_path = tmp_path / "out.json"
        artifact = exp563._write_blocked_preflight(output_path, "blocked")
        assert artifact["honest_verdict"] == "gpu_required"

    def test_creates_parent_dirs(self, tmp_path):
        output_path = tmp_path / "deep" / "nested" / "out.json"
        exp563._write_blocked_preflight(output_path, "test")
        assert output_path.exists()


# ---------------------------------------------------------------------------
# run_experiment — hard preflight branch
# ---------------------------------------------------------------------------


class TestRunExperimentPreflight:
    """SCENARIO-DATA-011: hard preflight exits when CARNOT_FORCE_LIVE not set."""

    def test_exits_when_force_live_not_set(self, tmp_path):
        """run_experiment() must call sys.exit(1) when CARNOT_FORCE_LIVE is absent."""
        with patch.dict(os.environ, {}, clear=False) as env:
            env.pop("CARNOT_FORCE_LIVE", None)
            # kill_gpu_zombies and watchdog must not raise
            with patch.object(exp563.ExperimentTemplate, "kill_gpu_zombies"):
                with patch("scripts.experiment_563_live_data_a_v2.ExperimentTimeoutWatchdog") as mock_wd:
                    mock_wd.return_value.start = MagicMock()
                    mock_wd.return_value.stop = MagicMock()
                    with pytest.raises(SystemExit) as exc_info:
                        exp563.run_experiment(repo_root=tmp_path)
                    assert exc_info.value.code == 1

    def test_blocked_artifact_written_on_preflight_failure(self, tmp_path):
        """A blocked artifact must be written before exit(1)."""
        # Pre-create results dir structure
        (tmp_path / "results").mkdir(parents=True, exist_ok=True)
        with patch.dict(os.environ, {}, clear=False) as env:
            env.pop("CARNOT_FORCE_LIVE", None)
            with patch.object(exp563.ExperimentTemplate, "kill_gpu_zombies"):
                with patch("scripts.experiment_563_live_data_a_v2.ExperimentTimeoutWatchdog") as mock_wd:
                    mock_wd.return_value.start = MagicMock()
                    mock_wd.return_value.stop = MagicMock()
                    with pytest.raises(SystemExit):
                        exp563.run_experiment(repo_root=tmp_path)
        deliverable = tmp_path / exp563.DELIVERABLE
        assert deliverable.exists(), "blocked artifact must be written before exit"
        art = json.loads(deliverable.read_text())
        assert art["status"] == "blocked"
