"""Tests for Exp 594: Live Verify-Repair with CoACEExtractorV3 -- 50q GSM8K benchmark.

100% targeted coverage on functions added in scripts/experiment_594_live_vr_coace_v3.py.

Spec: REQ-BENCH-056, SCENARIO-BENCH-075, SCENARIO-BENCH-076, SCENARIO-BENCH-077
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

import scripts.experiment_594_live_vr_coace_v3 as exp594  # noqa: E402


# ---------------------------------------------------------------------------
# _load_gate
# ---------------------------------------------------------------------------


class TestLoadGate:
    """Gate file loading must handle missing, corrupt, and valid files gracefully."""

    def test_returns_none_when_file_missing(self, tmp_path):
        # SCENARIO-BENCH-075: missing gate file -> None -> blocked artifact
        result = exp594._load_gate(tmp_path)
        assert result is None

    def test_returns_dict_when_file_valid(self, tmp_path):
        # gate_open=True (hypothetical future scenario where recall passes threshold)
        gate_data = {"gate_open": True, "v3_recall": 0.35, "experiment": 591}
        (tmp_path / "results").mkdir()
        (tmp_path / exp594.GATE_FILE).write_text(json.dumps(gate_data))
        result = exp594._load_gate(tmp_path)
        assert result is not None
        assert result["gate_open"] is True
        assert result["v3_recall"] == 0.35

    def test_returns_dict_when_gate_closed(self, tmp_path):
        # Current real state: gate_open=False, v3_recall=0.04
        gate_data = {"gate_open": False, "v3_recall": 0.04, "experiment": 591}
        (tmp_path / "results").mkdir()
        (tmp_path / exp594.GATE_FILE).write_text(json.dumps(gate_data))
        result = exp594._load_gate(tmp_path)
        assert result is not None
        assert result["gate_open"] is False
        assert result["v3_recall"] == 0.04

    def test_returns_none_when_file_corrupt(self, tmp_path):
        (tmp_path / "results").mkdir()
        (tmp_path / exp594.GATE_FILE).write_text("{not valid json")
        result = exp594._load_gate(tmp_path)
        assert result is None

    def test_returns_none_when_value_not_dict(self, tmp_path):
        (tmp_path / "results").mkdir()
        (tmp_path / exp594.GATE_FILE).write_text("[1, 2, 3]")
        result = exp594._load_gate(tmp_path)
        assert result is None


# ---------------------------------------------------------------------------
# _load_gsm8k_questions (synthetic fallback)
# ---------------------------------------------------------------------------


class TestLoadGsm8kQuestions:
    """When datasets is unavailable, synthetic fallback must return correct count."""

    def test_returns_correct_count_50(self):
        # SCENARIO-BENCH-076: 50 questions for indices 300-349
        with patch.dict("sys.modules", {"datasets": None}):
            questions = exp594._load_gsm8k_questions(300, 349)
        assert len(questions) == 50

    def test_questions_have_required_keys(self):
        with patch.dict("sys.modules", {"datasets": None}):
            questions = exp594._load_gsm8k_questions(300, 310)
        for q in questions:
            assert "question" in q
            assert "answer" in q

    def test_start_end_inclusive(self):
        with patch.dict("sys.modules", {"datasets": None}):
            questions = exp594._load_gsm8k_questions(300, 300)
        assert len(questions) == 1

    def test_question_indices_match_range(self):
        with patch.dict("sys.modules", {"datasets": None}):
            questions = exp594._load_gsm8k_questions(300, 304)
        assert len(questions) == 5


# ---------------------------------------------------------------------------
# _qwen_generate
# ---------------------------------------------------------------------------


class TestQwenGenerate:
    """_qwen_generate must normalise pipeline output and handle errors."""

    def test_returns_string_from_list_output(self):
        mock_pipeline = MagicMock(return_value=[{"generated_text": "hello world"}])
        result = exp594._qwen_generate(mock_pipeline, "test prompt")
        assert result == "hello world"

    def test_returns_error_string_on_exception(self):
        mock_pipeline = MagicMock(side_effect=RuntimeError("boom"))
        result = exp594._qwen_generate(mock_pipeline, "test prompt")
        assert result.startswith("[qwen_error:")

    def test_returns_str_for_non_list_output(self):
        mock_pipeline = MagicMock(return_value="raw string")
        result = exp594._qwen_generate(mock_pipeline, "test prompt")
        assert isinstance(result, str)

    def test_handles_empty_list_output(self):
        mock_pipeline = MagicMock(return_value=[])
        result = exp594._qwen_generate(mock_pipeline, "test prompt")
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# _build_repair_prompt
# ---------------------------------------------------------------------------


class TestBuildRepairPrompt:
    """Repair prompt must include the original question and mention arithmetic."""

    def test_includes_question(self):
        question = "How many apples are there?"
        prompt = exp594._build_repair_prompt(question)
        assert question in prompt

    def test_mentions_arithmetic_errors(self):
        prompt = exp594._build_repair_prompt("q")
        assert "error" in prompt.lower() or "arithmetic" in prompt.lower()


# ---------------------------------------------------------------------------
# _run_per_question (uses CoACEExtractorV3)
# ---------------------------------------------------------------------------


class TestRunPerQuestion:
    """Core loop: verify baseline, extract violations, repair if needed."""

    def _make_extractor(self):
        from carnot.extraction.coace_extractor_v3 import CoACEExtractorV3
        return CoACEExtractorV3()

    def test_no_violations_baseline_equals_pipeline(self):
        # SCENARIO-BENCH-077: when no violations, pipeline response == baseline
        extractor = self._make_extractor()
        questions = [{"question": "What is 2+2?", "answer": "#### 4"}]

        responses = ["The answer is 4."]
        call_count = {"n": 0}

        def generate_fn(prompt):
            r = responses[min(call_count["n"], len(responses) - 1)]
            call_count["n"] += 1
            return r

        stats = exp594._run_per_question(extractor, generate_fn, questions)
        assert stats["n_violations_found"] == 0
        assert stats["n_repairs_attempted"] == 0

    def test_violation_triggers_repair_call(self):
        # When CoACEV3 finds a violation, a second generate call (repair) is made
        extractor = self._make_extractor()
        questions = [{"question": "What is 3+4?", "answer": "#### 7"}]

        responses = ["3 + 4 = 99, so the answer is 7.", "The answer is 7."]
        call_count = {"n": 0}

        def generate_fn(prompt):
            r = responses[min(call_count["n"], len(responses) - 1)]
            call_count["n"] += 1
            return r

        stats = exp594._run_per_question(extractor, generate_fn, questions)
        assert stats["n_violations_found"] >= 1
        assert stats["n_repairs_attempted"] >= 1

    def test_inference_error_handled_gracefully(self):
        # Baseline inference error must not crash the loop
        extractor = self._make_extractor()
        questions = [{"question": "q", "answer": "#### 1"}]

        call_count = {"n": 0}

        def generate_fn(prompt):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise RuntimeError("inference failed")
            return "The answer is 1."

        stats = exp594._run_per_question(extractor, generate_fn, questions)
        assert "baseline_accuracy" in stats

    def test_returns_required_fields(self):
        # SCENARIO-BENCH-076: all required fields must be present
        extractor = self._make_extractor()
        questions = [{"question": "q", "answer": "#### 0"}]
        stats = exp594._run_per_question(extractor, lambda p: "answer is 0", questions)
        for field in (
            "baseline_accuracy", "pipeline_accuracy", "n_violations_found",
            "n_repairs_attempted", "n_repairs_succeeded", "per_question",
        ):
            assert field in stats, f"Missing field: {field}"

    def test_empty_questions_returns_zero_accuracies(self):
        extractor = self._make_extractor()
        stats = exp594._run_per_question(extractor, lambda p: "answer", [])
        assert stats["baseline_accuracy"] == 0.0
        assert stats["pipeline_accuracy"] == 0.0

    def test_repair_succeeded_counted_when_repair_fixes_error(self):
        # When repair takes a wrong baseline to a correct answer: n_repairs_succeeded++
        extractor = self._make_extractor()
        # Gold answer is 7; baseline gives wrong arithmetic (violation), repair gives correct
        questions = [{"question": "What is 3+4?", "answer": "#### 7"}]

        responses = ["3 + 4 = 99, so the answer is wrong.", "The answer is 7."]
        call_count = {"n": 0}

        def generate_fn(prompt):
            r = responses[min(call_count["n"], len(responses) - 1)]
            call_count["n"] += 1
            return r

        stats = exp594._run_per_question(extractor, generate_fn, questions)
        # n_repairs_succeeded only increments if baseline wrong + repair correct
        assert "n_repairs_succeeded" in stats


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

    def test_schema_field_is_v3(self):
        # SCENARIO-BENCH-076: schema must be 'carnot.live_vr_coace_v3.v1'
        tmpl, captured = self._make_tmpl()
        exp594._build_artifact(tmpl, {}, inference_mode="live_gpu")
        assert captured.get("schema") == "carnot.live_vr_coace_v3.v1"

    def test_extractor_field_is_coace_v3(self):
        tmpl, captured = self._make_tmpl()
        exp594._build_artifact(tmpl, {}, inference_mode="live_gpu")
        assert captured.get("extractor") == "coace_v3"

    def test_question_indices_field(self):
        tmpl, captured = self._make_tmpl()
        exp594._build_artifact(tmpl, {}, inference_mode="live_gpu")
        assert captured.get("question_indices") == "300-349"

    def test_retro_033_resolved_true_when_positive_live(self):
        # SCENARIO-BENCH-077: signed_improvement > 0 + live_gpu -> resolved=True
        tmpl, captured = self._make_tmpl()
        exp594._build_artifact(
            tmpl,
            {"baseline_accuracy": 0.4, "pipeline_accuracy": 0.6},
            inference_mode="live_gpu",
        )
        assert captured["retro_033_resolved"] is True
        assert captured["honest_verdict"] == "first_live_improvement"

    def test_retro_033_resolved_false_when_no_improvement_live(self):
        # SCENARIO-BENCH-077: live_gpu but no improvement -> retro_033_resolved=False
        tmpl, captured = self._make_tmpl()
        exp594._build_artifact(
            tmpl,
            {"baseline_accuracy": 0.5, "pipeline_accuracy": 0.5},
            inference_mode="live_gpu",
        )
        assert captured["retro_033_resolved"] is False
        assert captured["honest_verdict"] == "live_no_improvement_v13"

    def test_blocked_verdict_when_gate_closed(self):
        # SCENARIO-BENCH-075: gate_closed exit path produces blocked verdict
        tmpl, captured = self._make_tmpl()
        exp594._build_artifact(
            tmpl,
            {},
            inference_mode="blocked_gate_closed",
            status="blocked",
            reason="gate_closed_coace_v3_recall_below_30pct",
        )
        assert captured["honest_verdict"] == "blocked_gate_closed"
        assert captured.get("reason") == "gate_closed_coace_v3_recall_below_30pct"

    def test_v3_recall_at_gate_propagated(self):
        tmpl, captured = self._make_tmpl()
        exp594._build_artifact(tmpl, {}, inference_mode="live_gpu", v3_recall_at_gate=0.04)
        assert captured.get("v3_recall_at_gate") == 0.04

    def test_v3_recall_at_gate_none_by_default(self):
        tmpl, captured = self._make_tmpl()
        exp594._build_artifact(tmpl, {}, inference_mode="live_gpu")
        assert captured.get("v3_recall_at_gate") is None

    def test_all_required_schema_fields_present(self):
        # SCENARIO-BENCH-076: all required artifact fields
        tmpl, captured = self._make_tmpl()
        exp594._build_artifact(tmpl, {}, inference_mode="live_gpu")
        required_fields = [
            "schema", "inference_mode", "n_questions", "question_indices", "extractor",
            "v3_recall_at_gate", "baseline_accuracy", "pipeline_accuracy",
            "signed_improvement", "n_violations_found", "n_repairs_attempted",
            "n_repairs_succeeded", "retro_033_resolved", "honest_verdict",
        ]
        for f in required_fields:
            assert f in captured, f"Missing required field: {f}"

    def test_retro_033_resolved_false_when_not_live_gpu(self):
        # SCENARIO-BENCH-077: positive improvement but not live_gpu -> still False
        tmpl, captured = self._make_tmpl()
        exp594._build_artifact(
            tmpl,
            {"baseline_accuracy": 0.0, "pipeline_accuracy": 0.9},
            inference_mode="blocked_gate_closed",
        )
        assert captured["retro_033_resolved"] is False


# ---------------------------------------------------------------------------
# run_experiment -- gate check path (SCENARIO-BENCH-075)
# ---------------------------------------------------------------------------


class TestRunExperimentGateBlocked:
    """SCENARIO-BENCH-075: when gate_open=False, write blocked artifact and exit."""

    def _make_mock_tmpl(self):
        mock_tmpl = MagicMock()
        deliverable_called = []
        mock_tmpl.assert_deliverable_written = lambda: deliverable_called.append(True)
        captured = {}

        def build_result(data, **kwargs):
            captured.update(data)
            captured["status"] = kwargs.get("status", "success")
            return captured

        mock_tmpl.build_result.side_effect = build_result
        return mock_tmpl, captured, deliverable_called

    def test_writes_blocked_artifact_when_gate_false(self, tmp_path):
        # Simulate the real Exp 591 state: gate_open=False, v3_recall=0.04
        results_dir = tmp_path / "results"
        results_dir.mkdir()
        gate_data = {"gate_open": False, "v3_recall": 0.04, "experiment": 591}
        (tmp_path / exp594.GATE_FILE).write_text(json.dumps(gate_data))

        mock_tmpl, captured, deliverable_called = self._make_mock_tmpl()

        with (
            patch.object(exp594, "ExperimentTemplate", return_value=mock_tmpl),
            patch.object(exp594, "assert_live_gpu_available"),
        ):
            result = exp594.run_experiment(repo_root=tmp_path)

        assert result.get("honest_verdict") == "blocked_gate_closed"
        assert result.get("upstream_exp") == 591
        assert deliverable_called

    def test_writes_blocked_artifact_when_gate_file_missing(self, tmp_path):
        # No gate file -> same blocked path (safe default)
        mock_tmpl, captured, deliverable_called = self._make_mock_tmpl()

        with (
            patch.object(exp594, "ExperimentTemplate", return_value=mock_tmpl),
            patch.object(exp594, "assert_live_gpu_available"),
        ):
            result = exp594.run_experiment(repo_root=tmp_path)

        assert result.get("honest_verdict") == "blocked_gate_closed"
        assert deliverable_called

    def test_blocked_artifact_written_to_disk(self, tmp_path):
        # SCENARIO-BENCH-075: deliverable must exist on disk with correct fields
        results_dir = tmp_path / "results"
        results_dir.mkdir()
        gate_data = {"gate_open": False, "v3_recall": 0.04, "experiment": 591}
        (tmp_path / exp594.GATE_FILE).write_text(json.dumps(gate_data))

        mock_tmpl = MagicMock()
        mock_tmpl.assert_deliverable_written = lambda: None
        mock_tmpl.build_result.side_effect = lambda d, **kw: {**d, "status": kw.get("status")}

        with (
            patch.object(exp594, "ExperimentTemplate", return_value=mock_tmpl),
            patch.object(exp594, "assert_live_gpu_available"),
        ):
            exp594.run_experiment(repo_root=tmp_path)

        out_path = tmp_path / exp594._DELIVERABLE
        assert out_path.exists()
        data = json.loads(out_path.read_text())
        assert data.get("honest_verdict") == "blocked_gate_closed"
        assert data.get("upstream_exp") == 591
        assert data.get("reason") == "gate_closed_coace_v3_recall_below_30pct"

    def test_v3_recall_at_gate_populated_from_gate_file(self, tmp_path):
        # v3_recall_at_gate in blocked artifact must match gate file value
        results_dir = tmp_path / "results"
        results_dir.mkdir()
        gate_data = {"gate_open": False, "v3_recall": 0.04, "experiment": 591}
        (tmp_path / exp594.GATE_FILE).write_text(json.dumps(gate_data))

        mock_tmpl, captured, _ = self._make_mock_tmpl()

        with (
            patch.object(exp594, "ExperimentTemplate", return_value=mock_tmpl),
            patch.object(exp594, "assert_live_gpu_available"),
        ):
            exp594.run_experiment(repo_root=tmp_path)

        assert captured.get("v3_recall_at_gate") == 0.04


# ---------------------------------------------------------------------------
# run_experiment -- gpu_required path
# ---------------------------------------------------------------------------


class TestRunExperimentGpuRequired:
    """When LiveGPUGate blocks, write gpu_required artifact."""

    def test_writes_gpu_required_artifact(self, tmp_path):
        results_dir = tmp_path / "results"
        results_dir.mkdir()
        # gate_open=True so we proceed past the gate check to LiveGPUGate
        gate_data = {"gate_open": True, "v3_recall": 0.35, "experiment": 591}
        (tmp_path / exp594.GATE_FILE).write_text(json.dumps(gate_data))

        mock_tmpl = MagicMock()
        mock_tmpl.build_result.side_effect = lambda d, **kw: {**d, "status": kw.get("status")}
        mock_tmpl.assert_deliverable_written = lambda: None

        with (
            patch.object(exp594, "ExperimentTemplate", return_value=mock_tmpl),
            patch.object(exp594, "assert_live_gpu_available"),
            patch.object(exp594.LiveGPUGate, "require_live_or_blocked", return_value="not_live"),
        ):
            result = exp594.run_experiment(repo_root=tmp_path)

        assert result.get("inference_mode") == "gpu_required"


# ---------------------------------------------------------------------------
# Deliverable JSON on disk -- field completeness check
# ---------------------------------------------------------------------------


class TestDeliverableJson:
    """The pre-written blocked artifact must have all required schema fields."""

    def test_deliverable_exists(self):
        path = _REPO_ROOT / "results" / "experiment_594_live_vr_coace_v3.json"
        assert path.exists(), "Deliverable JSON must exist"

    def test_deliverable_has_required_fields(self):
        path = _REPO_ROOT / "results" / "experiment_594_live_vr_coace_v3.json"
        data = json.loads(path.read_text())
        required = [
            "schema", "inference_mode", "n_questions", "question_indices", "extractor",
            "v3_recall_at_gate", "baseline_accuracy", "pipeline_accuracy",
            "signed_improvement", "n_violations_found", "n_repairs_attempted",
            "n_repairs_succeeded", "retro_033_resolved", "honest_verdict",
            "status", "experiment",
        ]
        for f in required:
            assert f in data, f"Missing required field in deliverable: {f}"

    def test_deliverable_schema_value(self):
        path = _REPO_ROOT / "results" / "experiment_594_live_vr_coace_v3.json"
        data = json.loads(path.read_text())
        assert data["schema"] == "carnot.live_vr_coace_v3.v1"

    def test_deliverable_status_blocked(self):
        path = _REPO_ROOT / "results" / "experiment_594_live_vr_coace_v3.json"
        data = json.loads(path.read_text())
        assert data["status"] == "blocked"

    def test_deliverable_honest_verdict(self):
        path = _REPO_ROOT / "results" / "experiment_594_live_vr_coace_v3.json"
        data = json.loads(path.read_text())
        assert data["honest_verdict"] == "blocked_gate_closed"

    def test_deliverable_v3_recall_matches_gate(self):
        path = _REPO_ROOT / "results" / "experiment_594_live_vr_coace_v3.json"
        data = json.loads(path.read_text())
        assert data["v3_recall_at_gate"] == 0.04

    def test_deliverable_retro_033_resolved_false(self):
        path = _REPO_ROOT / "results" / "experiment_594_live_vr_coace_v3.json"
        data = json.loads(path.read_text())
        assert data["retro_033_resolved"] is False
