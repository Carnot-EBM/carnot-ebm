"""Tests for Exp 582: Live Verify-Repair with CoACEExtractorV2 -- 50q GSM8K benchmark.

100% targeted coverage on functions added in scripts/experiment_582_live_vr_coace_v2.py.

Spec: REQ-BENCH-015 (v4),
      SCENARIO-BENCH-036, SCENARIO-BENCH-037, SCENARIO-BENCH-038
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

# The module-level gate fires on import; set CARNOT_FORCE_LIVE=1 first.
os.environ["CARNOT_FORCE_LIVE"] = "1"

import scripts.experiment_582_live_vr_coace_v2 as exp582  # noqa: E402


# ---------------------------------------------------------------------------
# _load_gate
# ---------------------------------------------------------------------------


class TestLoadGate:
    """Gate file loading must handle missing, corrupt, and valid files."""

    def test_returns_none_when_file_missing(self, tmp_path):
        # SCENARIO-BENCH-036: missing gate file -> None -> blocked artifact
        result = exp582._load_gate(tmp_path)
        assert result is None

    def test_returns_dict_when_file_valid(self, tmp_path):
        gate_data = {"gate_open": True, "v2_recall": 0.25, "experiment": 581}
        (tmp_path / "results").mkdir()
        (tmp_path / exp582.GATE_FILE).write_text(json.dumps(gate_data))
        result = exp582._load_gate(tmp_path)
        assert result is not None
        assert result["gate_open"] is True
        assert result["v2_recall"] == 0.25

    def test_returns_none_when_file_corrupt(self, tmp_path):
        (tmp_path / "results").mkdir()
        (tmp_path / exp582.GATE_FILE).write_text("{not valid json")
        result = exp582._load_gate(tmp_path)
        assert result is None

    def test_returns_none_when_value_not_dict(self, tmp_path):
        (tmp_path / "results").mkdir()
        (tmp_path / exp582.GATE_FILE).write_text("[1, 2, 3]")
        result = exp582._load_gate(tmp_path)
        assert result is None


# ---------------------------------------------------------------------------
# _load_gsm8k_questions (fallback path)
# ---------------------------------------------------------------------------


class TestLoadGsm8kQuestions:
    """When datasets is unavailable, synthetic fallback must return correct count."""

    def test_returns_correct_count(self):
        with patch.dict("sys.modules", {"datasets": None}):
            questions = exp582._load_gsm8k_questions(250, 299)
        assert len(questions) == 50

    def test_questions_have_required_keys(self):
        with patch.dict("sys.modules", {"datasets": None}):
            questions = exp582._load_gsm8k_questions(250, 260)
        for q in questions:
            assert "question" in q
            assert "answer" in q

    def test_start_end_inclusive(self):
        with patch.dict("sys.modules", {"datasets": None}):
            questions = exp582._load_gsm8k_questions(5, 5)
        assert len(questions) == 1

    def test_question_indices_match_range(self):
        with patch.dict("sys.modules", {"datasets": None}):
            questions = exp582._load_gsm8k_questions(250, 254)
        assert len(questions) == 5


# ---------------------------------------------------------------------------
# _qwen_generate
# ---------------------------------------------------------------------------


class TestQwenGenerate:
    """_qwen_generate must normalise pipeline output and handle errors."""

    def test_returns_string_from_list_output(self):
        mock_pipeline = MagicMock(return_value=[{"generated_text": "hello world"}])
        result = exp582._qwen_generate(mock_pipeline, "test prompt")
        assert result == "hello world"

    def test_returns_error_string_on_exception(self):
        mock_pipeline = MagicMock(side_effect=RuntimeError("boom"))
        result = exp582._qwen_generate(mock_pipeline, "test prompt")
        assert result.startswith("[qwen_error:")

    def test_returns_str_for_non_list_output(self):
        mock_pipeline = MagicMock(return_value="raw string")
        result = exp582._qwen_generate(mock_pipeline, "test prompt")
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# _build_repair_prompt
# ---------------------------------------------------------------------------


class TestBuildRepairPrompt:
    """Repair prompt must include the original question."""

    def test_includes_question(self):
        question = "How many apples are there?"
        prompt = exp582._build_repair_prompt(question)
        assert question in prompt

    def test_mentions_arithmetic_errors(self):
        prompt = exp582._build_repair_prompt("q")
        assert "error" in prompt.lower() or "arithmetic" in prompt.lower()


# ---------------------------------------------------------------------------
# _run_per_question
# ---------------------------------------------------------------------------


class TestRunPerQuestion:
    """Core loop: verify baseline, extract violations, repair if needed."""

    def _make_extractor(self):
        from carnot.extraction.coace_extractor_v2 import CoACEExtractorV2
        return CoACEExtractorV2()

    def test_no_violations_baseline_equals_pipeline(self):
        # SCENARIO-BENCH-037: when CoACEV2 finds no violations, pipeline == baseline
        extractor = self._make_extractor()
        questions = [{"question": "What is 2+2?", "answer": "#### 4"}]

        responses = ["The answer is 4."]
        call_count = {"n": 0}

        def generate_fn(prompt):
            r = responses[min(call_count["n"], len(responses) - 1)]
            call_count["n"] += 1
            return r

        stats = exp582._run_per_question(extractor, generate_fn, questions)
        assert stats["n_violations_found"] == 0
        assert stats["n_repairs_applied"] == 0

    def test_violation_triggers_repair_call(self):
        # When CoACEV2 finds a violation, a second generate call (repair) is made
        extractor = self._make_extractor()
        questions = [{"question": "What is 3+4?", "answer": "#### 7"}]

        responses = ["3 + 4 = 99, so the answer is 7.", "The answer is 7."]
        call_count = {"n": 0}

        def generate_fn(prompt):
            r = responses[min(call_count["n"], len(responses) - 1)]
            call_count["n"] += 1
            return r

        stats = exp582._run_per_question(extractor, generate_fn, questions)
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

        # Must not raise; baseline_resp falls back to ""
        stats = exp582._run_per_question(extractor, generate_fn, questions)
        assert "baseline_accuracy" in stats

    def test_returns_required_fields(self):
        extractor = self._make_extractor()
        questions = [{"question": "q", "answer": "#### 0"}]
        stats = exp582._run_per_question(extractor, lambda p: "answer is 0", questions)
        for field in (
            "baseline_accuracy", "pipeline_accuracy", "n_violations_found",
            "n_repairs_applied", "n_repairs_improved", "per_question",
        ):
            assert field in stats, f"Missing field: {field}"

    def test_empty_questions_returns_zero_accuracies(self):
        extractor = self._make_extractor()
        stats = exp582._run_per_question(extractor, lambda p: "answer", [])
        assert stats["baseline_accuracy"] == 0.0
        assert stats["pipeline_accuracy"] == 0.0


# ---------------------------------------------------------------------------
# _build_artifact
# ---------------------------------------------------------------------------


class TestBuildArtifact:
    """Artifact builder must produce all required schema fields on every exit path."""

    def _make_tmpl(self):
        mock_tmpl = MagicMock()
        captured = {}

        def build_result(data, **kwargs):
            captured.update(data)
            captured["status"] = kwargs.get("status", "success")
            return captured

        mock_tmpl.build_result.side_effect = build_result
        return mock_tmpl, captured

    def test_schema_field_is_v2(self):
        tmpl, captured = self._make_tmpl()
        exp582._build_artifact(tmpl, {}, inference_mode="live_gpu")
        assert captured.get("schema") == "carnot.live_vr_coace.v2"

    def test_extractor_field_is_coace_v2(self):
        tmpl, captured = self._make_tmpl()
        exp582._build_artifact(tmpl, {}, inference_mode="live_gpu")
        assert captured.get("extractor") == "coace_v2"

    def test_question_indices_field(self):
        tmpl, captured = self._make_tmpl()
        exp582._build_artifact(tmpl, {}, inference_mode="live_gpu")
        assert captured.get("question_indices") == "250-299"

    def test_retro_033_resolved_true_when_positive_live(self):
        # SCENARIO-BENCH-038: signed_improvement > 0 + live_gpu -> resolved=True
        tmpl, captured = self._make_tmpl()
        exp582._build_artifact(
            tmpl,
            {"baseline_accuracy": 0.4, "pipeline_accuracy": 0.6},
            inference_mode="live_gpu",
        )
        assert captured["retro_033_resolved"] is True
        assert captured["honest_verdict"] == "first_positive"

    def test_retro_033_resolved_false_when_no_improvement_live(self):
        tmpl, captured = self._make_tmpl()
        exp582._build_artifact(
            tmpl,
            {"baseline_accuracy": 0.5, "pipeline_accuracy": 0.5},
            inference_mode="live_gpu",
        )
        assert captured["retro_033_resolved"] is False
        assert captured["honest_verdict"] == "live_no_improvement_v2"

    def test_blocked_verdict_when_gate_closed(self):
        tmpl, captured = self._make_tmpl()
        exp582._build_artifact(
            tmpl,
            {},
            inference_mode="blocked_gate_closed_recall_too_low",
            status="blocked",
        )
        assert captured["honest_verdict"] == "blocked_gate_closed_recall_too_low"

    def test_v2_recall_at_gate_propagated(self):
        tmpl, captured = self._make_tmpl()
        exp582._build_artifact(tmpl, {}, inference_mode="live_gpu", v2_recall_at_gate=0.25)
        assert captured.get("v2_recall_at_gate") == 0.25

    def test_v2_recall_at_gate_none_by_default(self):
        tmpl, captured = self._make_tmpl()
        exp582._build_artifact(tmpl, {}, inference_mode="live_gpu")
        assert captured.get("v2_recall_at_gate") is None


# ---------------------------------------------------------------------------
# run_experiment -- gate check path (SCENARIO-BENCH-036)
# ---------------------------------------------------------------------------


class TestRunExperimentGateBlocked:
    """SCENARIO-BENCH-036: when gate_open=False, write blocked artifact and exit."""

    def test_writes_blocked_artifact_when_gate_false(self, tmp_path):
        results_dir = tmp_path / "results"
        results_dir.mkdir()
        gate_data = {"gate_open": False, "v2_recall": 0.059, "experiment": 581}
        (tmp_path / exp582.GATE_FILE).write_text(json.dumps(gate_data))

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
            patch.object(exp582, "ExperimentTemplate", return_value=mock_tmpl),
            patch("scripts.experiment_582_live_vr_coace_v2.ExperimentTemplate.kill_gpu_zombies"),
        ):
            result = exp582.run_experiment(repo_root=tmp_path)

        assert result.get("honest_verdict") == "blocked_gate_closed_recall_too_low"
        assert result.get("upstream_exp") == 581
        assert deliverable_called

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
            patch.object(exp582, "ExperimentTemplate", return_value=mock_tmpl),
            patch("scripts.experiment_582_live_vr_coace_v2.ExperimentTemplate.kill_gpu_zombies"),
        ):
            result = exp582.run_experiment(repo_root=tmp_path)

        assert result.get("honest_verdict") == "blocked_gate_closed_recall_too_low"
        assert deliverable_called

    def test_blocked_artifact_written_to_disk(self, tmp_path):
        results_dir = tmp_path / "results"
        results_dir.mkdir()
        gate_data = {"gate_open": False, "v2_recall": 0.059, "experiment": 581}
        (tmp_path / exp582.GATE_FILE).write_text(json.dumps(gate_data))

        mock_tmpl = MagicMock()
        mock_tmpl.assert_deliverable_written = lambda: None
        mock_tmpl.build_result.side_effect = lambda d, **kw: {**d, "status": kw.get("status")}

        with (
            patch.object(exp582, "ExperimentTemplate", return_value=mock_tmpl),
            patch("scripts.experiment_582_live_vr_coace_v2.ExperimentTemplate.kill_gpu_zombies"),
        ):
            exp582.run_experiment(repo_root=tmp_path)

        out_path = tmp_path / exp582._DELIVERABLE
        assert out_path.exists()
        data = json.loads(out_path.read_text())
        assert data.get("honest_verdict") == "blocked_gate_closed_recall_too_low"
        assert data.get("upstream_exp") == 581


# ---------------------------------------------------------------------------
# run_experiment -- gpu_required path
# ---------------------------------------------------------------------------


class TestRunExperimentGpuRequired:
    """When LiveGPUGate blocks, write gpu_required artifact."""

    def test_writes_gpu_required_artifact(self, tmp_path):
        results_dir = tmp_path / "results"
        results_dir.mkdir()
        gate_data = {"gate_open": True, "v2_recall": 0.25, "experiment": 581}
        (tmp_path / exp582.GATE_FILE).write_text(json.dumps(gate_data))

        mock_tmpl = MagicMock()
        mock_tmpl.build_result.side_effect = lambda d, **kw: {**d, "status": kw.get("status")}
        mock_tmpl.assert_deliverable_written = lambda: None

        with (
            patch.object(exp582, "ExperimentTemplate", return_value=mock_tmpl),
            patch("scripts.experiment_582_live_vr_coace_v2.ExperimentTemplate.kill_gpu_zombies"),
            patch.object(exp582.LiveGPUGate, "require_live_or_blocked", return_value="not_live"),
        ):
            result = exp582.run_experiment(repo_root=tmp_path)

        assert result.get("inference_mode") == "gpu_required"
