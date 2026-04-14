"""Tests for scripts/experiment_template.py.

All tests run in mock mode (no GPU hardware required).

Spec coverage:
  REQ-VERIFY-083  — Experiment scaffolding template eliminates cold-start boilerplate
  REQ-VERIFY-084  — BatchedInferenceRunner groups questions and enforces batch timeout
  SCENARIO-VERIFY-109 — ExperimentTemplate instantiates with required fields
  SCENARIO-VERIFY-110 — setup_gpu() returns health_status dict from pre-warm
  SCENARIO-VERIFY-111 — checkpoint_save/resume round-trips correctly
  SCENARIO-VERIFY-112 — build_result() includes all required artifact fields
  SCENARIO-VERIFY-113 — BatchedInferenceRunner groups N questions into ceil(N/batch_size) batches
  SCENARIO-VERIFY-114 — batch timeout is batch_size * 60s, not per-question
  SCENARIO-VERIFY-115 — batch logging records batch_id, batch_size, batch_time_s per batch
  SCENARIO-VERIFY-116 — run_with_timeout returns partial result dict on timeout
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Module import
# ---------------------------------------------------------------------------

_SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "experiment_template.py"
sys.path.insert(0, str(_SCRIPT_PATH.parent))

import importlib.util

_spec = importlib.util.spec_from_file_location("experiment_template", _SCRIPT_PATH)
assert _spec is not None and _spec.loader is not None
_mod = importlib.util.module_from_spec(_spec)
sys.modules["experiment_template"] = _mod
_spec.loader.exec_module(_mod)  # type: ignore[union-attr]

ExperimentTemplate = _mod.ExperimentTemplate
BatchedInferenceRunner = _mod.BatchedInferenceRunner
InferenceResult = _mod.InferenceResult
REQUIRED_RESULT_FIELDS = _mod.REQUIRED_RESULT_FIELDS


# ---------------------------------------------------------------------------
# TestExperimentTemplateInit
# SCENARIO-VERIFY-109
# ---------------------------------------------------------------------------


class TestExperimentTemplateInit:
    """ExperimentTemplate instantiates correctly with required parameters."""

    def test_init_stores_exp_id(self, tmp_path: Path) -> None:
        """exp_id is stored on the instance.  REQ-VERIFY-083"""
        t = ExperimentTemplate(306, "Test Exp", "results/test.json", repo_root=tmp_path)
        assert t.exp_id == 306

    def test_init_stores_title(self, tmp_path: Path) -> None:
        """title is stored on the instance.  REQ-VERIFY-083"""
        t = ExperimentTemplate(306, "Test Exp Title", "results/test.json", repo_root=tmp_path)
        assert t.title == "Test Exp Title"

    def test_init_stores_deliverable(self, tmp_path: Path) -> None:
        """deliverable path is stored.  REQ-VERIFY-083"""
        t = ExperimentTemplate(306, "T", "results/out.json", repo_root=tmp_path)
        assert t.deliverable == "results/out.json"

    def test_init_requires_gpu_default_false(self, tmp_path: Path) -> None:
        """requires_gpu defaults to False.  REQ-VERIFY-083"""
        t = ExperimentTemplate(306, "T", "results/out.json", repo_root=tmp_path)
        assert t.requires_gpu is False

    def test_init_requires_gpu_true(self, tmp_path: Path) -> None:
        """requires_gpu=True is stored.  REQ-VERIFY-083"""
        t = ExperimentTemplate(306, "T", "results/out.json", requires_gpu=True, repo_root=tmp_path)
        assert t.requires_gpu is True

    def test_checkpoint_is_none_before_setup(self, tmp_path: Path) -> None:
        """No checkpoint loaded before setup() is called.  REQ-VERIFY-083"""
        t = ExperimentTemplate(306, "T", "results/out.json", repo_root=tmp_path)
        assert t.checkpoint is None


# ---------------------------------------------------------------------------
# TestExperimentTemplateSetup
# SCENARIO-VERIFY-109
# ---------------------------------------------------------------------------


class TestExperimentTemplateSetup:
    """setup() creates directories and loads checkpoint if present."""

    def test_setup_creates_results_dir(self, tmp_path: Path) -> None:
        """setup() creates the results directory.  REQ-VERIFY-083"""
        t = ExperimentTemplate(306, "T", "results/exp_306.json", repo_root=tmp_path)
        t.setup()
        assert (tmp_path / "results").is_dir()

    def test_setup_creates_checkpoint_dir(self, tmp_path: Path) -> None:
        """setup() creates the checkpoint directory.  REQ-VERIFY-083"""
        t = ExperimentTemplate(306, "T", "results/exp_306.json", repo_root=tmp_path)
        t.setup()
        assert (tmp_path / "results" / "checkpoints" / "experiment_306").is_dir()

    def test_setup_loads_existing_checkpoint(self, tmp_path: Path) -> None:
        """setup() loads an existing checkpoint if present.  REQ-VERIFY-083"""
        ckpt_dir = tmp_path / "results" / "checkpoints" / "experiment_306"
        ckpt_dir.mkdir(parents=True)
        ckpt_file = ckpt_dir / "checkpoint.json"
        payload = {"step": 5, "results": [1, 2, 3]}
        ckpt_file.write_text(json.dumps(payload))

        t = ExperimentTemplate(306, "T", "results/exp_306.json", repo_root=tmp_path)
        t.setup()
        assert t.checkpoint is not None
        assert t.checkpoint["step"] == 5

    def test_setup_checkpoint_none_when_absent(self, tmp_path: Path) -> None:
        """setup() leaves checkpoint=None when no checkpoint file exists.  REQ-VERIFY-083"""
        t = ExperimentTemplate(306, "T", "results/exp_306.json", repo_root=tmp_path)
        t.setup()
        assert t.checkpoint is None


# ---------------------------------------------------------------------------
# TestSetupGpu
# SCENARIO-VERIFY-110
# ---------------------------------------------------------------------------


class TestSetupGpu:
    """setup_gpu() calls pre-warm and returns health_status dict."""

    def test_returns_dict(self, tmp_path: Path) -> None:
        """setup_gpu() returns a dict.  REQ-VERIFY-083"""
        model_specs = [{"name": "MockModel", "hf_id": "mock/model", "gpu": 0}]

        def _mock_prewarm(model_name: str, hf_id: str, gpu_id: int, **kwargs: Any) -> Any:
            result = MagicMock()
            result.health_ok = True
            result.load_time_s = 0.1
            result.stall_root_cause = None
            return result

        t = ExperimentTemplate(306, "T", "results/out.json", requires_gpu=True, repo_root=tmp_path)
        status = t.setup_gpu(model_specs, prewarm_fn=_mock_prewarm)
        assert isinstance(status, dict)

    def test_health_status_keys(self, tmp_path: Path) -> None:
        """health_status dict contains all_healthy, models, prewarm_time_s.  SCENARIO-VERIFY-110"""
        model_specs = [{"name": "MockModel", "hf_id": "mock/model", "gpu": 0}]

        def _mock_prewarm(model_name: str, hf_id: str, gpu_id: int, **kwargs: Any) -> Any:
            result = MagicMock()
            result.health_ok = True
            result.load_time_s = 0.05
            result.stall_root_cause = None
            return result

        t = ExperimentTemplate(306, "T", "results/out.json", requires_gpu=True, repo_root=tmp_path)
        status = t.setup_gpu(model_specs, prewarm_fn=_mock_prewarm)
        assert "all_healthy" in status
        assert "models" in status
        assert "prewarm_time_s" in status

    def test_all_healthy_true_when_all_pass(self, tmp_path: Path) -> None:
        """all_healthy=True when all models report health_ok=True.  SCENARIO-VERIFY-110"""
        model_specs = [
            {"name": "ModelA", "hf_id": "mock/a", "gpu": 0},
            {"name": "ModelB", "hf_id": "mock/b", "gpu": 1},
        ]

        def _mock_prewarm(model_name: str, hf_id: str, gpu_id: int, **kwargs: Any) -> Any:
            r = MagicMock()
            r.health_ok = True
            r.load_time_s = 0.1
            r.stall_root_cause = None
            return r

        t = ExperimentTemplate(306, "T", "results/out.json", requires_gpu=True, repo_root=tmp_path)
        status = t.setup_gpu(model_specs, prewarm_fn=_mock_prewarm)
        assert status["all_healthy"] is True

    def test_all_healthy_false_when_one_fails(self, tmp_path: Path) -> None:
        """all_healthy=False when any model reports health_ok=False.  SCENARIO-VERIFY-110"""
        model_specs = [
            {"name": "ModelA", "hf_id": "mock/a", "gpu": 0},
            {"name": "ModelB", "hf_id": "mock/b", "gpu": 1},
        ]
        call_count = [0]

        def _mock_prewarm(model_name: str, hf_id: str, gpu_id: int, **kwargs: Any) -> Any:
            r = MagicMock()
            r.health_ok = call_count[0] == 0  # first healthy, second fails
            r.load_time_s = 0.1
            r.stall_root_cause = "lazy_load_stall" if call_count[0] > 0 else None
            call_count[0] += 1
            return r

        t = ExperimentTemplate(306, "T", "results/out.json", requires_gpu=True, repo_root=tmp_path)
        status = t.setup_gpu(model_specs, prewarm_fn=_mock_prewarm)
        assert status["all_healthy"] is False

    def test_models_list_has_entry_per_spec(self, tmp_path: Path) -> None:
        """models list has one entry per model_spec.  SCENARIO-VERIFY-110"""
        model_specs = [
            {"name": "M1", "hf_id": "x/m1", "gpu": 0},
            {"name": "M2", "hf_id": "x/m2", "gpu": 1},
        ]

        def _mock_prewarm(model_name: str, hf_id: str, gpu_id: int, **kwargs: Any) -> Any:
            r = MagicMock()
            r.health_ok = True
            r.load_time_s = 0.1
            r.stall_root_cause = None
            return r

        t = ExperimentTemplate(306, "T", "results/out.json", requires_gpu=True, repo_root=tmp_path)
        status = t.setup_gpu(model_specs, prewarm_fn=_mock_prewarm)
        assert len(status["models"]) == 2


# ---------------------------------------------------------------------------
# TestCheckpointSaveResume
# SCENARIO-VERIFY-111
# ---------------------------------------------------------------------------


class TestCheckpointSaveResume:
    """checkpoint_save / checkpoint_resume round-trip."""

    def test_checkpoint_save_creates_file(self, tmp_path: Path) -> None:
        """checkpoint_save() writes a JSON file.  REQ-VERIFY-083"""
        t = ExperimentTemplate(306, "T", "results/out.json", repo_root=tmp_path)
        t.setup()
        t.checkpoint_save({"items": [1, 2, 3]}, step=3)
        ckpt_file = tmp_path / "results" / "checkpoints" / "experiment_306" / "checkpoint.json"
        assert ckpt_file.exists()

    def test_checkpoint_save_stores_step(self, tmp_path: Path) -> None:
        """checkpoint_save() stores the step number.  REQ-VERIFY-083"""
        t = ExperimentTemplate(306, "T", "results/out.json", repo_root=tmp_path)
        t.setup()
        t.checkpoint_save({"x": 99}, step=7)
        ckpt_file = tmp_path / "results" / "checkpoints" / "experiment_306" / "checkpoint.json"
        data = json.loads(ckpt_file.read_text())
        assert data["step"] == 7

    def test_checkpoint_save_stores_results(self, tmp_path: Path) -> None:
        """checkpoint_save() stores the partial results.  REQ-VERIFY-083"""
        t = ExperimentTemplate(306, "T", "results/out.json", repo_root=tmp_path)
        t.setup()
        t.checkpoint_save({"items": ["a", "b"]}, step=2)
        ckpt_file = tmp_path / "results" / "checkpoints" / "experiment_306" / "checkpoint.json"
        data = json.loads(ckpt_file.read_text())
        assert data["results"]["items"] == ["a", "b"]

    def test_checkpoint_resume_returns_none_when_absent(self, tmp_path: Path) -> None:
        """checkpoint_resume() returns None when no checkpoint file.  SCENARIO-VERIFY-111"""
        t = ExperimentTemplate(306, "T", "results/out.json", repo_root=tmp_path)
        t.setup()
        result = t.checkpoint_resume()
        assert result is None

    def test_checkpoint_resume_returns_data_when_present(self, tmp_path: Path) -> None:
        """checkpoint_resume() returns saved data when checkpoint exists.  SCENARIO-VERIFY-111"""
        t = ExperimentTemplate(306, "T", "results/out.json", repo_root=tmp_path)
        t.setup()
        t.checkpoint_save({"items": [10, 20]}, step=5)
        resumed = t.checkpoint_resume()
        assert resumed is not None
        assert resumed["step"] == 5
        assert resumed["results"]["items"] == [10, 20]

    def test_checkpoint_save_is_atomic(self, tmp_path: Path) -> None:
        """checkpoint_save() writes atomically (no .tmp file left behind).  REQ-VERIFY-083"""
        t = ExperimentTemplate(306, "T", "results/out.json", repo_root=tmp_path)
        t.setup()
        t.checkpoint_save({"x": 1}, step=1)
        ckpt_dir = tmp_path / "results" / "checkpoints" / "experiment_306"
        tmp_files = list(ckpt_dir.glob("*.tmp"))
        assert tmp_files == []


# ---------------------------------------------------------------------------
# TestBuildResult
# SCENARIO-VERIFY-112
# ---------------------------------------------------------------------------


class TestBuildResult:
    """build_result() produces artifact with all required fields."""

    def test_required_fields_present(self, tmp_path: Path) -> None:
        """build_result() includes all REQUIRED_RESULT_FIELDS.  SCENARIO-VERIFY-112"""
        t = ExperimentTemplate(306, "T", "results/out.json", repo_root=tmp_path)
        result = t.build_result({"accuracy": 0.9}, status="success")
        for field in REQUIRED_RESULT_FIELDS:
            assert field in result, f"Missing required field: {field}"

    def test_experiment_field(self, tmp_path: Path) -> None:
        """build_result() sets experiment=exp_id.  SCENARIO-VERIFY-112"""
        t = ExperimentTemplate(306, "T", "results/out.json", repo_root=tmp_path)
        result = t.build_result({}, status="success")
        assert result["experiment"] == 306

    def test_run_date_format(self, tmp_path: Path) -> None:
        """build_result() sets run_date in YYYYMMDD format.  SCENARIO-VERIFY-112"""
        t = ExperimentTemplate(306, "T", "results/out.json", repo_root=tmp_path)
        result = t.build_result({}, status="success")
        assert len(result["run_date"]) == 8
        assert result["run_date"].isdigit()

    def test_schema_field_is_list(self, tmp_path: Path) -> None:
        """build_result() sets schema to a list of field names.  SCENARIO-VERIFY-112"""
        t = ExperimentTemplate(306, "T", "results/out.json", repo_root=tmp_path)
        result = t.build_result({}, status="success")
        assert isinstance(result["schema"], list)

    def test_duration_s_is_nonnegative(self, tmp_path: Path) -> None:
        """build_result() sets duration_s >= 0.  SCENARIO-VERIFY-112"""
        t = ExperimentTemplate(306, "T", "results/out.json", repo_root=tmp_path)
        result = t.build_result({}, status="success")
        assert result["duration_s"] >= 0.0

    def test_data_merged_into_result(self, tmp_path: Path) -> None:
        """build_result() merges data dict into the result.  SCENARIO-VERIFY-112"""
        t = ExperimentTemplate(306, "T", "results/out.json", repo_root=tmp_path)
        result = t.build_result({"accuracy": 0.95, "n": 100}, status="success")
        assert result["accuracy"] == 0.95
        assert result["n"] == 100

    def test_extra_fields_merged(self, tmp_path: Path) -> None:
        """build_result() includes **extra_fields in the result.  SCENARIO-VERIFY-112"""
        t = ExperimentTemplate(306, "T", "results/out.json", repo_root=tmp_path)
        result = t.build_result({}, status="success", custom_tag="hello")
        assert result["custom_tag"] == "hello"

    def test_status_field_stored(self, tmp_path: Path) -> None:
        """build_result() stores the status field.  SCENARIO-VERIFY-112"""
        t = ExperimentTemplate(306, "T", "results/out.json", repo_root=tmp_path)
        result = t.build_result({}, status="blocked")
        assert result["status"] == "blocked"


# ---------------------------------------------------------------------------
# TestRunWithTimeout
# REQ-VERIFY-083
# ---------------------------------------------------------------------------


class TestRunWithTimeout:
    """run_with_timeout() returns result or partial on timeout."""

    def test_returns_result_when_fast(self, tmp_path: Path) -> None:
        """run_with_timeout returns function result when it completes in time.  REQ-VERIFY-083"""
        t = ExperimentTemplate(306, "T", "results/out.json", repo_root=tmp_path)

        def fast_fn() -> dict[str, Any]:
            return {"answer": 42}

        result = t.run_with_timeout(fast_fn, timeout_s=5.0)
        assert result == {"answer": 42}

    def test_returns_partial_on_timeout(self, tmp_path: Path) -> None:
        """run_with_timeout returns partial dict with timed_out=True on timeout.  REQ-VERIFY-083"""
        t = ExperimentTemplate(306, "T", "results/out.json", repo_root=tmp_path)

        def slow_fn() -> dict[str, Any]:
            time.sleep(10)
            return {"answer": 42}

        result = t.run_with_timeout(slow_fn, timeout_s=0.1)
        assert isinstance(result, dict)
        assert result.get("timed_out") is True

    def test_timeout_result_has_partial_flag(self, tmp_path: Path) -> None:
        """Timeout result has partial=True.  REQ-VERIFY-083"""
        t = ExperimentTemplate(306, "T", "results/out.json", repo_root=tmp_path)

        def slow_fn() -> dict[str, Any]:
            time.sleep(10)
            return {}

        result = t.run_with_timeout(slow_fn, timeout_s=0.1)
        assert result.get("partial") is True


# ---------------------------------------------------------------------------
# TestBatchedInferenceRunnerGrouping
# SCENARIO-VERIFY-113
# ---------------------------------------------------------------------------


class TestBatchedInferenceRunnerGrouping:
    """BatchedInferenceRunner groups questions into correct batch counts."""

    def _make_runner(self, responses: list[str]) -> Any:
        """Create a mock runner that returns pre-set responses in order."""
        idx = [0]
        responses_list = list(responses)

        def _run_fn(prompt: str) -> str:
            r = responses_list[idx[0] % len(responses_list)]
            idx[0] += 1
            return r

        return _run_fn

    def test_8_questions_1_batch(self, tmp_path: Path) -> None:
        """8 questions with batch_size=8 produces exactly 1 batch.  SCENARIO-VERIFY-113"""
        runner_fn = self._make_runner(["ok"] * 8)
        bir = BatchedInferenceRunner(runner_fn, batch_size=8)
        questions = [f"q{i}" for i in range(8)]
        results = bir.run_batch(questions)
        assert len(results) == 8
        assert bir.batch_log[0]["batch_id"] == 0
        assert len(bir.batch_log) == 1

    def test_9_questions_2_batches(self, tmp_path: Path) -> None:
        """9 questions with batch_size=8 produces 2 batches.  SCENARIO-VERIFY-113"""
        runner_fn = self._make_runner(["ok"] * 9)
        bir = BatchedInferenceRunner(runner_fn, batch_size=8)
        questions = [f"q{i}" for i in range(9)]
        results = bir.run_batch(questions)
        assert len(results) == 9
        assert len(bir.batch_log) == 2

    def test_16_questions_2_batches_of_8(self, tmp_path: Path) -> None:
        """16 questions with batch_size=8 produces exactly 2 batches.  SCENARIO-VERIFY-113"""
        runner_fn = self._make_runner(["ok"] * 16)
        bir = BatchedInferenceRunner(runner_fn, batch_size=8)
        questions = [f"q{i}" for i in range(16)]
        results = bir.run_batch(questions)
        assert len(results) == 16
        assert len(bir.batch_log) == 2

    def test_results_in_original_order(self) -> None:
        """Results are returned in original question order.  SCENARIO-VERIFY-113"""
        responses = [f"answer_{i}" for i in range(10)]
        idx = [0]

        def _run_fn(prompt: str) -> str:
            r = responses[idx[0]]
            idx[0] += 1
            return r

        bir = BatchedInferenceRunner(_run_fn, batch_size=3)
        questions = [f"q{i}" for i in range(10)]
        results = bir.run_batch(questions)
        assert len(results) == 10
        for i, res in enumerate(results):
            assert res.response == f"answer_{i}"

    def test_batch_size_16(self) -> None:
        """batch_size=16 groups 20 questions into 2 batches.  SCENARIO-VERIFY-113"""
        runner_fn = self._make_runner(["ok"] * 20)
        bir = BatchedInferenceRunner(runner_fn, batch_size=16)
        questions = [f"q{i}" for i in range(20)]
        results = bir.run_batch(questions)
        assert len(results) == 20
        assert len(bir.batch_log) == 2


# ---------------------------------------------------------------------------
# TestBatchedInferenceRunnerTimeout
# SCENARIO-VERIFY-114
# ---------------------------------------------------------------------------


class TestBatchedInferenceRunnerTimeout:
    """Batch timeout is batch_size * 60s, not per-question."""

    def test_timeout_is_batch_size_times_60(self) -> None:
        """BatchedInferenceRunner uses batch_size*60 as timeout per batch.  SCENARIO-VERIFY-114"""
        runner_fn = lambda p: "ok"
        bir = BatchedInferenceRunner(runner_fn, batch_size=8)
        assert bir.batch_timeout_s == 8 * 60

    def test_timeout_scales_with_batch_size(self) -> None:
        """batch_timeout_s scales linearly with batch_size.  SCENARIO-VERIFY-114"""
        bir_small = BatchedInferenceRunner(lambda p: "ok", batch_size=4)
        bir_large = BatchedInferenceRunner(lambda p: "ok", batch_size=16)
        assert bir_small.batch_timeout_s == 4 * 60
        assert bir_large.batch_timeout_s == 16 * 60


# ---------------------------------------------------------------------------
# TestBatchedInferenceRunnerLogging
# SCENARIO-VERIFY-115
# ---------------------------------------------------------------------------


class TestBatchedInferenceRunnerLogging:
    """Batch logging records batch_id, batch_size, batch_time_s per batch."""

    def test_batch_log_has_batch_id(self) -> None:
        """batch_log entries have batch_id.  SCENARIO-VERIFY-115"""
        bir = BatchedInferenceRunner(lambda p: "ok", batch_size=4)
        bir.run_batch(["q1", "q2", "q3", "q4"])
        assert "batch_id" in bir.batch_log[0]

    def test_batch_log_has_batch_size(self) -> None:
        """batch_log entries have batch_size.  SCENARIO-VERIFY-115"""
        bir = BatchedInferenceRunner(lambda p: "ok", batch_size=4)
        bir.run_batch(["q1", "q2", "q3", "q4"])
        assert "batch_size" in bir.batch_log[0]
        assert bir.batch_log[0]["batch_size"] == 4

    def test_batch_log_has_batch_time_s(self) -> None:
        """batch_log entries have batch_time_s.  SCENARIO-VERIFY-115"""
        bir = BatchedInferenceRunner(lambda p: "ok", batch_size=4)
        bir.run_batch(["q1", "q2", "q3", "q4"])
        assert "batch_time_s" in bir.batch_log[0]
        assert bir.batch_log[0]["batch_time_s"] >= 0.0

    def test_batch_log_increments_batch_id(self) -> None:
        """batch_log batch_ids are sequential starting at 0.  SCENARIO-VERIFY-115"""
        bir = BatchedInferenceRunner(lambda p: "ok", batch_size=2)
        bir.run_batch(["q1", "q2", "q3", "q4"])
        assert bir.batch_log[0]["batch_id"] == 0
        assert bir.batch_log[1]["batch_id"] == 1

    def test_batch_log_last_batch_smaller_size(self) -> None:
        """Last batch log entry shows actual (smaller) batch size.  SCENARIO-VERIFY-115"""
        bir = BatchedInferenceRunner(lambda p: "ok", batch_size=4)
        bir.run_batch(["q1", "q2", "q3", "q4", "q5"])  # 5 questions → [4, 1]
        assert bir.batch_log[1]["batch_size"] == 1

    def test_batch_log_cleared_between_run_batch_calls(self) -> None:
        """Calling run_batch() again clears the previous batch_log.  SCENARIO-VERIFY-115"""
        bir = BatchedInferenceRunner(lambda p: "ok", batch_size=2)
        bir.run_batch(["q1", "q2"])
        assert len(bir.batch_log) == 1
        bir.run_batch(["q3", "q4", "q5", "q6"])
        assert len(bir.batch_log) == 2  # fresh log for 2nd call


# ---------------------------------------------------------------------------
# TestInferenceResult
# REQ-VERIFY-084
# ---------------------------------------------------------------------------


class TestInferenceResult:
    """InferenceResult dataclass has expected fields."""

    def test_inference_result_fields(self) -> None:
        """InferenceResult has prompt, response, batch_id, timed_out.  REQ-VERIFY-084"""
        r = InferenceResult(prompt="q", response="a", batch_id=0, timed_out=False)
        assert r.prompt == "q"
        assert r.response == "a"
        assert r.batch_id == 0
        assert r.timed_out is False

    def test_inference_result_timed_out_default_false(self) -> None:
        """InferenceResult.timed_out defaults to False.  REQ-VERIFY-084"""
        r = InferenceResult(prompt="q", response="a", batch_id=0)
        assert r.timed_out is False

    def test_inference_result_timed_out_true(self) -> None:
        """InferenceResult.timed_out=True when batch times out.  REQ-VERIFY-084"""
        r = InferenceResult(prompt="q", response="", batch_id=0, timed_out=True)
        assert r.timed_out is True


# ---------------------------------------------------------------------------
# TestRequiredResultFields
# REQ-VERIFY-083
# ---------------------------------------------------------------------------


class TestBatchTimeoutIntegration:
    """Integration: run_batch() with a slow runner returns timed_out=True results."""

    def test_batch_timeout_returns_timed_out_results(self) -> None:
        """When batch times out, all results in that batch have timed_out=True.  SCENARIO-VERIFY-114"""

        def _very_slow(prompt: str) -> str:
            time.sleep(999)
            return "never"

        # Use a tiny batch_size to keep timeout small (2 * 60 = 120 s is too long for tests;
        # override batch_timeout_s directly after construction for test speed)
        bir = BatchedInferenceRunner(_very_slow, batch_size=1)
        bir.batch_timeout_s = 0.05  # 50 ms — triggers timeout without waiting
        results = bir.run_batch(["q1"])
        assert len(results) == 1
        assert results[0].timed_out is True
        assert results[0].response == ""

    def test_non_timed_out_batch_has_timed_out_false(self) -> None:
        """Fast runner produces timed_out=False results.  SCENARIO-VERIFY-114"""
        bir = BatchedInferenceRunner(lambda p: "ok", batch_size=4)
        results = bir.run_batch(["q1", "q2"])
        assert all(r.timed_out is False for r in results)


class TestCheckpointCorruptRecovery:
    """checkpoint_resume() handles corrupt JSON gracefully."""

    def test_corrupt_json_returns_none(self, tmp_path: Path) -> None:
        """checkpoint_resume() returns None when checkpoint file is corrupt.  REQ-VERIFY-083"""
        t = ExperimentTemplate(306, "T", "results/out.json", repo_root=tmp_path)
        t.setup()
        # Write corrupt JSON to the checkpoint file
        ckpt_path = tmp_path / "results" / "checkpoints" / "experiment_306" / "checkpoint.json"
        ckpt_path.write_text("{not valid json!!!}")
        result = t.checkpoint_resume()
        assert result is None

    def test_empty_file_returns_none(self, tmp_path: Path) -> None:
        """checkpoint_resume() returns None when checkpoint file is empty.  REQ-VERIFY-083"""
        t = ExperimentTemplate(306, "T", "results/out.json", repo_root=tmp_path)
        t.setup()
        ckpt_path = tmp_path / "results" / "checkpoints" / "experiment_306" / "checkpoint.json"
        ckpt_path.write_text("")
        result = t.checkpoint_resume()
        assert result is None


class TestRequiredResultFields:
    """REQUIRED_RESULT_FIELDS constant lists all mandatory artifact fields."""

    def test_required_fields_is_list(self) -> None:
        """REQUIRED_RESULT_FIELDS is a list.  REQ-VERIFY-083"""
        assert isinstance(REQUIRED_RESULT_FIELDS, list)

    def test_required_fields_includes_experiment(self) -> None:
        """REQUIRED_RESULT_FIELDS includes 'experiment'.  REQ-VERIFY-083"""
        assert "experiment" in REQUIRED_RESULT_FIELDS

    def test_required_fields_includes_run_date(self) -> None:
        """REQUIRED_RESULT_FIELDS includes 'run_date'.  REQ-VERIFY-083"""
        assert "run_date" in REQUIRED_RESULT_FIELDS

    def test_required_fields_includes_schema(self) -> None:
        """REQUIRED_RESULT_FIELDS includes 'schema'.  REQ-VERIFY-083"""
        assert "schema" in REQUIRED_RESULT_FIELDS

    def test_required_fields_includes_duration_s(self) -> None:
        """REQUIRED_RESULT_FIELDS includes 'duration_s'.  REQ-VERIFY-083"""
        assert "duration_s" in REQUIRED_RESULT_FIELDS

    def test_required_fields_includes_status(self) -> None:
        """REQUIRED_RESULT_FIELDS includes 'status'.  REQ-VERIFY-083"""
        assert "status" in REQUIRED_RESULT_FIELDS
