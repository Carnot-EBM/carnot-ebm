"""Tests for Exp 569: Live Verify-Repair with CoACEExtractor — 50q GSM8K benchmark.

100% targeted coverage on functions added in scripts/experiment_569_live_vr_coace.py.

Spec: REQ-BENCH-014 (v3),
      SCENARIO-BENCH-033, SCENARIO-BENCH-034, SCENARIO-BENCH-035
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import scripts.experiment_569_live_vr_coace as exp569


# ---------------------------------------------------------------------------
# _load_gate
# ---------------------------------------------------------------------------


class TestLoadGate:
    """Gate file loading must handle missing, corrupt, and valid files."""

    def test_returns_none_when_file_missing(self, tmp_path):
        # SCENARIO-BENCH-033: missing gate file -> None -> blocked artifact
        result = exp569._load_gate(tmp_path)
        assert result is None

    def test_returns_dict_when_file_valid(self, tmp_path):
        gate_data = {"gate_open": True, "experiment": 565}
        (tmp_path / "results").mkdir()
        (tmp_path / exp569.GATE_FILE).write_text(json.dumps(gate_data))
        result = exp569._load_gate(tmp_path)
        assert result is not None
        assert result["gate_open"] is True

    def test_returns_none_when_file_corrupt(self, tmp_path):
        (tmp_path / "results").mkdir()
        (tmp_path / exp569.GATE_FILE).write_text("{not valid json")
        result = exp569._load_gate(tmp_path)
        assert result is None


# ---------------------------------------------------------------------------
# _load_gsm8k_questions (fallback path)
# ---------------------------------------------------------------------------


class TestLoadGsm8kQuestions:
    """When datasets is unavailable, synthetic fallback should return correct count."""

    def test_returns_correct_count(self):
        # Patch load_dataset to fail -> triggers synthetic fallback
        with patch.dict("sys.modules", {"datasets": None}):
            questions = exp569._load_gsm8k_questions(100, 149)
        assert len(questions) == 50

    def test_questions_have_required_keys(self):
        with patch.dict("sys.modules", {"datasets": None}):
            questions = exp569._load_gsm8k_questions(100, 110)
        for q in questions:
            assert "question" in q
            assert "answer" in q

    def test_start_end_inclusive(self):
        with patch.dict("sys.modules", {"datasets": None}):
            questions = exp569._load_gsm8k_questions(5, 5)
        assert len(questions) == 1


# ---------------------------------------------------------------------------
# _qwen_generate
# ---------------------------------------------------------------------------


class TestQwenGenerate:
    """_qwen_generate must normalise pipeline output and handle errors."""

    def test_returns_string_from_list_output(self):
        mock_pipeline = MagicMock(return_value=[{"generated_text": "hello world"}])
        result = exp569._qwen_generate(mock_pipeline, "test prompt")
        assert result == "hello world"

    def test_returns_error_string_on_exception(self):
        mock_pipeline = MagicMock(side_effect=RuntimeError("boom"))
        result = exp569._qwen_generate(mock_pipeline, "test prompt")
        assert result.startswith("[qwen_error:")


# ---------------------------------------------------------------------------
# _build_repair_prompt
# ---------------------------------------------------------------------------


class TestBuildRepairPrompt:
    """Repair prompt must include the original question."""

    def test_includes_question(self):
        question = "How many apples are there?"
        prompt = exp569._build_repair_prompt(question)
        assert question in prompt

    def test_mentions_arithmetic_errors(self):
        prompt = exp569._build_repair_prompt("q")
        # The repair instruction should mention errors so model knows to fix them
        assert "error" in prompt.lower() or "arithmetic" in prompt.lower()


# ---------------------------------------------------------------------------
# _run_per_question
# ---------------------------------------------------------------------------


class TestRunPerQuestion:
    """Core loop: verify baseline, extract violations, repair if needed."""

    def _make_extractor(self):
        from carnot.extraction.coace_extractor import CoACEExtractor
        return CoACEExtractor()

    def test_no_violations_baseline_equals_pipeline(self):
        # SCENARIO-BENCH-034: when CoACE finds no violations, pipeline == baseline
        extractor = self._make_extractor()
        questions = [{"question": "What is 2+2?", "answer": "#### 4"}]

        responses = ["The answer is 4."]
        call_count = {"n": 0}

        def generate_fn(prompt):
            r = responses[min(call_count["n"], len(responses) - 1)]
            call_count["n"] += 1
            return r

        stats = exp569._run_per_question(extractor, generate_fn, questions)
        assert stats["n_violations_found"] == 0
        assert stats["n_repairs_applied"] == 0

    def test_violation_triggers_repair_call(self):
        # When CoACE finds a violation, a second generate call (repair) is made
        extractor = self._make_extractor()
        questions = [{"question": "What is 3+4?", "answer": "#### 7"}]

        # First call: response with wrong arithmetic that CoACE can flag
        # Second call: repaired response
        responses = ["3 + 4 = 99, so the answer is 7.", "The answer is 7."]
        call_count = {"n": 0}

        def generate_fn(prompt):
            r = responses[min(call_count["n"], len(responses) - 1)]
            call_count["n"] += 1
            return r

        stats = exp569._run_per_question(extractor, generate_fn, questions)
        # CoACE should flag '3 + 4 = 99'
        assert stats["n_violations_found"] >= 1
        assert stats["n_repairs_applied"] >= 1

    def test_inference_error_handled_gracefully(self):
        extractor = self._make_extractor()
        questions = [{"question": "q", "answer": "#### 1"}]

        call_count = {"n": 0}

        def generate_fn(prompt):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise RuntimeError("inference failed")
            return "The answer is 1."

        # Should not raise; baseline_resp becomes ""
        stats = exp569._run_per_question(extractor, generate_fn, questions)
        assert "baseline_accuracy" in stats

    def test_returns_required_fields(self):
        extractor = self._make_extractor()
        questions = [{"question": "q", "answer": "#### 0"}]
        stats = exp569._run_per_question(extractor, lambda p: "answer is 0", questions)
        for field in ("baseline_accuracy", "pipeline_accuracy", "n_violations_found",
                      "n_repairs_applied", "n_repairs_improved", "per_question"):
            assert field in stats, f"Missing field: {field}"


# ---------------------------------------------------------------------------
# _build_artifact
# ---------------------------------------------------------------------------


class TestBuildArtifact:
    """Artifact builder must produce all required schema fields."""

    def _make_tmpl(self):
        mock_tmpl = MagicMock()
        captured = {}

        def build_result(data, **kwargs):
            captured.update(data)
            captured["status"] = kwargs.get("status", "success")
            return captured

        mock_tmpl.build_result.side_effect = build_result
        return mock_tmpl, captured

    def test_schema_field_present(self):
        tmpl, captured = self._make_tmpl()
        exp569._build_artifact(tmpl, {}, inference_mode="live_gpu")
        assert captured.get("schema") == "carnot.live_vr_coace.v1"

    def test_retro_033_resolved_true_when_positive_live(self):
        # SCENARIO-BENCH-035: signed_improvement > 0 + live_gpu -> resolved=True
        tmpl, captured = self._make_tmpl()
        exp569._build_artifact(
            tmpl,
            {"baseline_accuracy": 0.4, "pipeline_accuracy": 0.6},
            inference_mode="live_gpu",
        )
        assert captured["retro_033_resolved"] is True
        assert captured["honest_verdict"] == "first_positive"

    def test_retro_033_resolved_false_when_no_improvement(self):
        tmpl, captured = self._make_tmpl()
        exp569._build_artifact(
            tmpl,
            {"baseline_accuracy": 0.5, "pipeline_accuracy": 0.5},
            inference_mode="live_gpu",
        )
        assert captured["retro_033_resolved"] is False
        assert captured["honest_verdict"] == "live_no_improvement_11q"

    def test_blocked_no_extraction_verdict(self):
        tmpl, captured = self._make_tmpl()
        exp569._build_artifact(tmpl, {}, inference_mode="blocked_no_extraction", status="blocked")
        assert captured["honest_verdict"] == "blocked_no_extraction"

    def test_question_indices_field(self):
        tmpl, captured = self._make_tmpl()
        exp569._build_artifact(tmpl, {}, inference_mode="live_gpu")
        assert captured.get("question_indices") == "100-149"

    def test_extractor_field(self):
        tmpl, captured = self._make_tmpl()
        exp569._build_artifact(tmpl, {}, inference_mode="live_gpu")
        assert captured.get("extractor") == "coace"


# ---------------------------------------------------------------------------
# run_experiment — gate check path (SCENARIO-BENCH-033)
# ---------------------------------------------------------------------------


class TestRunExperimentGateBlocked:
    """SCENARIO-BENCH-033: when gate_open=False, write blocked artifact and exit."""

    def test_writes_blocked_artifact_when_gate_false(self, tmp_path):
        # Write a gate file with gate_open=False
        results_dir = tmp_path / "results"
        results_dir.mkdir()
        gate_data = {"gate_open": False, "experiment": 565}
        (tmp_path / exp569.GATE_FILE).write_text(json.dumps(gate_data))

        mock_tmpl = MagicMock()
        deliverable_called = []
        mock_tmpl.assert_deliverable_written = lambda: deliverable_called.append(True)

        captured = {}

        def build_result(data, **kwargs):
            captured.update(data)
            captured["status"] = kwargs.get("status", "success")
            return captured

        mock_tmpl.build_result.side_effect = build_result

        with (
            patch.object(exp569, "ExperimentTemplate", return_value=mock_tmpl),
            patch("scripts.experiment_569_live_vr_coace.ExperimentTemplate.kill_gpu_zombies"),
        ):
            result = exp569.run_experiment(repo_root=tmp_path)

        assert result.get("honest_verdict") == "blocked_no_extraction"
        assert result.get("upstream_exp") == 565
        assert deliverable_called  # assert_deliverable_written was called

    def test_writes_blocked_artifact_when_gate_file_missing(self, tmp_path):
        # No gate file at all -> same blocked path
        mock_tmpl = MagicMock()
        deliverable_called = []
        mock_tmpl.assert_deliverable_written = lambda: deliverable_called.append(True)

        captured = {}

        def build_result(data, **kwargs):
            captured.update(data)
            captured["status"] = kwargs.get("status", "success")
            return captured

        mock_tmpl.build_result.side_effect = build_result

        with (
            patch.object(exp569, "ExperimentTemplate", return_value=mock_tmpl),
            patch("scripts.experiment_569_live_vr_coace.ExperimentTemplate.kill_gpu_zombies"),
        ):
            result = exp569.run_experiment(repo_root=tmp_path)

        assert result.get("honest_verdict") == "blocked_no_extraction"
        assert deliverable_called


# ---------------------------------------------------------------------------
# run_experiment — gpu_required path
# ---------------------------------------------------------------------------


class TestRunExperimentGpuRequired:
    """When LiveGPUGate blocks, write gpu_required artifact."""

    def test_writes_gpu_required_artifact(self, tmp_path):
        results_dir = tmp_path / "results"
        results_dir.mkdir()
        gate_data = {"gate_open": True, "experiment": 565}
        (tmp_path / exp569.GATE_FILE).write_text(json.dumps(gate_data))

        mock_tmpl = MagicMock()
        mock_tmpl.build_result.side_effect = lambda d, **kw: {**d, "status": kw.get("status")}
        mock_tmpl.assert_deliverable_written = lambda: None

        with (
            patch.object(exp569, "ExperimentTemplate", return_value=mock_tmpl),
            patch("scripts.experiment_569_live_vr_coace.ExperimentTemplate.kill_gpu_zombies"),
            patch.object(exp569.LiveGPUGate, "require_live_or_blocked", return_value="not_live"),
        ):
            result = exp569.run_experiment(repo_root=tmp_path)

        assert result.get("inference_mode") == "gpu_required"
