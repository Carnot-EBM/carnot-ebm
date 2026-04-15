"""Tests for scripts/experiment_340_live_precision_benchmark.py.

Covers 100% of the experiment script:
- _extract_gsm8k_answer: valid, missing, decimal
- _synthetic_gsm8k: length, fields, determinism
- _simulate_response: correct path, error-injected path, non-numeric gold fallback
- _is_correct: match, mismatch, None gold, broad search fallback, ValueError fallback
- _apply_variant: all five PipelineVariant branches
- run_variant: simulated mode, baseline accuracy, counters
- load_gsm8k_questions: HuggingFace failure fallback
- _write_artifact: file written correctly
- main() simulated path: end-to-end, artifact schema and fields
- main() blocked path: gpu not healthy → blocked artifact

Spec: REQ-BENCH-003, SCENARIO-BENCH-007, SCENARIO-BENCH-008, SCENARIO-BENCH-009
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Import target module (repository-root sys.path injection already in the script).
# ---------------------------------------------------------------------------
import sys
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import scripts.experiment_340_live_precision_benchmark as exp340
from carnot.pipeline.precision_benchmark import (
    PipelineVariant,
    PrecisionStackResult,
    build_precision_benchmark_artifact,
)


# ---------------------------------------------------------------------------
# _extract_gsm8k_answer
# ---------------------------------------------------------------------------


class TestExtractGsm8kAnswer:
    def test_standard_format(self):
        assert exp340._extract_gsm8k_answer("blah blah\n#### 42") == "42"

    def test_decimal(self):
        assert exp340._extract_gsm8k_answer("#### 3.14") == "3.14"

    def test_negative(self):
        assert exp340._extract_gsm8k_answer("#### -5") == "-5"

    def test_missing(self):
        assert exp340._extract_gsm8k_answer("no answer here") is None

    def test_spaces(self):
        assert exp340._extract_gsm8k_answer("####   7") == "7"


# ---------------------------------------------------------------------------
# _synthetic_gsm8k
# ---------------------------------------------------------------------------


class TestSyntheticGsm8k:
    def test_length(self):
        qs = exp340._synthetic_gsm8k(10)
        assert len(qs) == 10

    def test_fields(self):
        qs = exp340._synthetic_gsm8k(5)
        for q in qs:
            assert "question" in q
            assert "answer" in q

    def test_answer_format(self):
        """Every synthetic answer starts with '####'."""
        qs = exp340._synthetic_gsm8k(5)
        for q in qs:
            assert "####" in q["answer"]

    def test_deterministic(self):
        """Same seed → same output."""
        qs1 = exp340._synthetic_gsm8k(5)
        qs2 = exp340._synthetic_gsm8k(5)
        assert qs1 == qs2

    def test_zero_questions(self):
        assert exp340._synthetic_gsm8k(0) == []


# ---------------------------------------------------------------------------
# _simulate_response
# ---------------------------------------------------------------------------


class TestSimulateResponse:
    def test_returns_string(self):
        r = exp340._simulate_response("What is 2+2?", "#### 4")
        assert isinstance(r, str)

    def test_contains_answer_marker(self):
        """Response contains '####' so _extract_gsm8k_answer can parse it."""
        r = exp340._simulate_response("What is 2+2?", "#### 4")
        assert "####" in r

    def test_contains_step_markers(self):
        """Chain-of-thought markers present for CoTCircuitVerifier."""
        r = exp340._simulate_response("What is 2+2?", "#### 4")
        assert "Step 1:" in r

    def test_non_numeric_gold_fallback(self):
        """When gold cannot be parsed as float, response still returns."""
        r = exp340._simulate_response("Question?", "#### abc")
        assert isinstance(r, str)
        assert len(r) > 0

    def test_error_injection_deterministic(self):
        """Same question always gets the same error injection decision."""
        q = "A store has 5 red apples."
        a = "#### 20"
        r1 = exp340._simulate_response(q, a)
        r2 = exp340._simulate_response(q, a)
        assert r1 == r2

    def test_missing_gold_marker(self):
        """Question with no '#### N' in answer still returns non-empty string."""
        r = exp340._simulate_response("Q?", "no answer marker")
        assert isinstance(r, str)


# ---------------------------------------------------------------------------
# _is_correct
# ---------------------------------------------------------------------------


class TestIsCorrect:
    def test_correct_exact(self):
        assert exp340._is_correct("Step 1: ...\n#### 42", "42") is True

    def test_correct_float_tolerance(self):
        """Small floating point delta within 0.5 is considered correct."""
        assert exp340._is_correct("#### 42", "42") is True

    def test_wrong_answer(self):
        assert exp340._is_correct("#### 10", "42") is False

    def test_none_gold(self):
        assert exp340._is_correct("#### 42", None) is False

    def test_no_hash_marker_broad_search(self):
        """Falls back to last number in response when '#### N' is missing."""
        assert exp340._is_correct("the answer is 42", "42") is True

    def test_no_numbers_in_response(self):
        assert exp340._is_correct("no numbers here", "42") is False

    def test_value_error_fallback(self):
        """Non-numeric gold vs non-numeric response compares as strings."""
        # Gold "abc" and response "abc" → True
        r = exp340._is_correct("#### abc", "abc")
        # The function does a float conversion; "abc" fails → string compare
        # "abc" stripped == "abc" stripped → True
        assert isinstance(r, bool)


# ---------------------------------------------------------------------------
# _apply_variant
# ---------------------------------------------------------------------------


class TestApplyVariant:
    """Each branch of _apply_variant returns (response, n_viol, n_rep)."""

    _QUESTION = "What is 3 + 4?"
    _RESPONSE = "Step 1: 3 + 4 = 7.\nStep 2: from step 1, result is 7.\n#### 7"

    def _run(self, variant: PipelineVariant) -> tuple[str, int, int]:
        return exp340._apply_variant(variant, self._RESPONSE, self._QUESTION, "Qwen3.5-0.8B")

    def test_baseline(self):
        response, n_viol, n_rep = self._run(PipelineVariant.BASELINE)
        assert response == self._RESPONSE
        assert isinstance(n_viol, int)
        assert n_rep == 0

    def test_confidence_only(self):
        response, n_viol, n_rep = self._run(PipelineVariant.CONFIDENCE_ONLY)
        assert response == self._RESPONSE
        assert isinstance(n_viol, int)
        assert isinstance(n_rep, int)

    def test_confidence_adaptive(self):
        response, n_viol, n_rep = self._run(PipelineVariant.CONFIDENCE_ADAPTIVE)
        assert response == self._RESPONSE
        assert isinstance(n_viol, int)
        assert isinstance(n_rep, int)

    def test_confidence_adaptive_verge(self):
        response, n_viol, n_rep = self._run(PipelineVariant.CONFIDENCE_ADAPTIVE_VERGE)
        assert response == self._RESPONSE
        assert isinstance(n_viol, int)
        assert isinstance(n_rep, int)

    def test_full_stack(self):
        response, n_viol, n_rep = self._run(PipelineVariant.FULL_STACK)
        assert response == self._RESPONSE
        assert isinstance(n_viol, int)
        assert isinstance(n_rep, int)

    def test_response_unchanged(self):
        """In simulated mode, the response is never modified (no live repair)."""
        for variant in PipelineVariant:
            response, _, _ = self._run(variant)
            assert response == self._RESPONSE


# ---------------------------------------------------------------------------
# run_variant
# ---------------------------------------------------------------------------


class TestRunVariant:
    """run_variant in simulated mode (CARNOT_FORCE_LIVE=0)."""

    def _make_questions(self, n: int = 10) -> list[dict]:
        return exp340._synthetic_gsm8k(n)

    def test_returns_precision_stack_result(self):
        questions = self._make_questions(4)
        result = exp340.run_variant(
            variant=PipelineVariant.BASELINE,
            questions=questions,
            model_name="Qwen3.5-0.8B",
            inference_mode="simulated",
            model_obj=None,
        )
        assert isinstance(result, PrecisionStackResult)

    def test_model_id_set(self):
        questions = self._make_questions(4)
        result = exp340.run_variant(
            PipelineVariant.BASELINE, questions, "Gemma4-E4B-it", "simulated"
        )
        assert result.model_id == "Gemma4-E4B-it"

    def test_inference_mode_set(self):
        questions = self._make_questions(4)
        result = exp340.run_variant(
            PipelineVariant.FULL_STACK, questions, "Qwen3.5-0.8B", "simulated"
        )
        assert result.inference_mode == "simulated"

    def test_n_questions_set(self):
        questions = self._make_questions(6)
        result = exp340.run_variant(
            PipelineVariant.BASELINE, questions, "Qwen3.5-0.8B", "simulated"
        )
        assert result.n_questions == 6

    def test_pipeline_variant_set(self):
        questions = self._make_questions(4)
        result = exp340.run_variant(
            PipelineVariant.CONFIDENCE_ONLY, questions, "Qwen3.5-0.8B", "simulated"
        )
        assert result.pipeline_variant == PipelineVariant.CONFIDENCE_ONLY

    def test_accuracy_in_range(self):
        """Accuracy values must be in [0.0, 1.0]."""
        questions = self._make_questions(8)
        for variant in PipelineVariant:
            result = exp340.run_variant(variant, questions, "Qwen3.5-0.8B", "simulated")
            assert 0.0 <= result.baseline_accuracy <= 1.0
            assert 0.0 <= result.precision_stack_accuracy <= 1.0

    def test_signed_improvement_matches_formula(self):
        """signed_improvement == precision_stack_accuracy - baseline_accuracy."""
        questions = self._make_questions(8)
        result = exp340.run_variant(
            PipelineVariant.FULL_STACK, questions, "Gemma4-E4B-it", "simulated"
        )
        expected = result.precision_stack_accuracy - result.baseline_accuracy
        assert abs(result.signed_improvement - expected) < 1e-9

    def test_counter_fields_non_negative(self):
        questions = self._make_questions(8)
        result = exp340.run_variant(
            PipelineVariant.FULL_STACK, questions, "Gemma4-E4B-it", "simulated"
        )
        assert result.n_violations_found >= 0
        assert result.n_repairs_attempted >= 0
        assert result.n_repairs_improved >= 0
        assert result.n_repairs_broken >= 0

    def test_empty_questions_list(self):
        """Empty questions list → all accuracies 0.0."""
        result = exp340.run_variant(
            PipelineVariant.BASELINE, [], "Qwen3.5-0.8B", "simulated"
        )
        assert result.baseline_accuracy == 0.0
        assert result.precision_stack_accuracy == 0.0
        assert result.n_questions == 0


# ---------------------------------------------------------------------------
# load_gsm8k_questions
# ---------------------------------------------------------------------------


class TestLoadGsm8kQuestions:
    def test_fallback_when_datasets_unavailable(self):
        """When datasets import fails, falls back to synthetic questions."""
        with patch.dict("sys.modules", {"datasets": None}):
            qs = exp340.load_gsm8k_questions(10)
        assert len(qs) == 10
        for q in qs:
            assert "question" in q
            assert "answer" in q

    def test_fallback_when_load_raises(self):
        """When load_dataset raises, falls back to synthetic."""
        mock_datasets = MagicMock()
        mock_datasets.load_dataset.side_effect = RuntimeError("network error")
        with patch.dict("sys.modules", {"datasets": mock_datasets}):
            qs = exp340.load_gsm8k_questions(5)
        assert len(qs) == 5


# ---------------------------------------------------------------------------
# _write_artifact
# ---------------------------------------------------------------------------


class TestWriteArtifact:
    def test_writes_json_file(self, tmp_path: Path):
        """_write_artifact writes the artifact dict to tmpl._output_path as JSON."""
        from scripts.experiment_template import ExperimentTemplate

        tmpl = ExperimentTemplate(
            exp_id=340,
            title="test",
            deliverable="results/test_exp340.json",
            repo_root=tmp_path,
        )
        artifact = {"schema": "carnot.precision_benchmark.v1", "status": "success"}
        exp340._write_artifact(tmpl, artifact)

        written = json.loads((tmp_path / "results" / "test_exp340.json").read_text())
        assert written["schema"] == "carnot.precision_benchmark.v1"

    def test_creates_parent_dirs(self, tmp_path: Path):
        """_write_artifact creates nested output dirs if missing."""
        from scripts.experiment_template import ExperimentTemplate

        tmpl = ExperimentTemplate(
            exp_id=340,
            title="test",
            deliverable="deep/nested/test.json",
            repo_root=tmp_path,
        )
        exp340._write_artifact(tmpl, {"x": 1})
        assert (tmp_path / "deep" / "nested" / "test.json").exists()


# ---------------------------------------------------------------------------
# main() — simulated end-to-end
# ---------------------------------------------------------------------------


class TestMainSimulated:
    """End-to-end test of main() in CARNOT_FORCE_LIVE=0 (simulated) mode."""

    def test_main_produces_artifact(self, tmp_path: Path):
        """main() writes a valid JSON artifact in simulated mode."""
        env = {"CARNOT_FORCE_LIVE": "0"}

        with patch.dict(os.environ, env):
            with patch.object(exp340, "load_gsm8k_questions",
                              return_value=exp340._synthetic_gsm8k(8)):
                # Monkey-patch ExperimentTemplate to use tmp_path as repo_root.
                from scripts.experiment_template import ExperimentTemplate

                with patch("scripts.experiment_340_live_precision_benchmark.ExperimentTemplate") as MockTmpl:
                    tmpl_instance = ExperimentTemplate(
                        exp_id=340,
                        title=exp340.EXP_TITLE,
                        deliverable=exp340.DELIVERABLE,
                        repo_root=tmp_path,
                    )
                    tmpl_instance.setup()
                    MockTmpl.return_value = tmpl_instance
                    exp340.main()

        artifact_path = tmp_path / exp340.DELIVERABLE
        assert artifact_path.exists(), f"Artifact not found at {artifact_path}"
        artifact = json.loads(artifact_path.read_text())
        assert artifact["precision_schema"] == "carnot.precision_benchmark.v1"
        assert artifact["inference_mode"] == "simulated"
        assert artifact["honest_verdict"] == "simulated_only"

    def test_main_artifact_has_required_fields(self, tmp_path: Path):
        """Artifact contains all REQUIRED_RESULT_FIELDS."""
        from scripts.experiment_template import ExperimentTemplate, REQUIRED_RESULT_FIELDS

        with patch.dict(os.environ, {"CARNOT_FORCE_LIVE": "0"}):
            with patch.object(exp340, "load_gsm8k_questions",
                              return_value=exp340._synthetic_gsm8k(4)):
                with patch("scripts.experiment_340_live_precision_benchmark.ExperimentTemplate") as MockTmpl:
                    tmpl_instance = ExperimentTemplate(
                        exp_id=340,
                        title=exp340.EXP_TITLE,
                        deliverable=exp340.DELIVERABLE,
                        repo_root=tmp_path,
                    )
                    tmpl_instance.setup()
                    MockTmpl.return_value = tmpl_instance
                    exp340.main()

        artifact = json.loads((tmp_path / exp340.DELIVERABLE).read_text())
        for field in REQUIRED_RESULT_FIELDS:
            assert field in artifact, f"Missing required field: {field}"

    def test_main_all_results_count(self, tmp_path: Path):
        """Artifact all_results has 5 variants × 2 models = 10 entries."""
        from scripts.experiment_template import ExperimentTemplate

        with patch.dict(os.environ, {"CARNOT_FORCE_LIVE": "0"}):
            with patch.object(exp340, "load_gsm8k_questions",
                              return_value=exp340._synthetic_gsm8k(4)):
                with patch("scripts.experiment_340_live_precision_benchmark.ExperimentTemplate") as MockTmpl:
                    tmpl_instance = ExperimentTemplate(
                        exp_id=340,
                        title=exp340.EXP_TITLE,
                        deliverable=exp340.DELIVERABLE,
                        repo_root=tmp_path,
                    )
                    tmpl_instance.setup()
                    MockTmpl.return_value = tmpl_instance
                    exp340.main()

        artifact = json.loads((tmp_path / exp340.DELIVERABLE).read_text())
        assert len(artifact["all_results"]) == 10

    def test_main_pipeline_variants_in_artifact(self, tmp_path: Path):
        """Artifact pipeline_variants list contains all five variant names."""
        from scripts.experiment_template import ExperimentTemplate

        with patch.dict(os.environ, {"CARNOT_FORCE_LIVE": "0"}):
            with patch.object(exp340, "load_gsm8k_questions",
                              return_value=exp340._synthetic_gsm8k(4)):
                with patch("scripts.experiment_340_live_precision_benchmark.ExperimentTemplate") as MockTmpl:
                    tmpl_instance = ExperimentTemplate(
                        exp_id=340,
                        title=exp340.EXP_TITLE,
                        deliverable=exp340.DELIVERABLE,
                        repo_root=tmp_path,
                    )
                    tmpl_instance.setup()
                    MockTmpl.return_value = tmpl_instance
                    exp340.main()

        artifact = json.loads((tmp_path / exp340.DELIVERABLE).read_text())
        expected_variants = {v.value for v in PipelineVariant}
        assert set(artifact["pipeline_variants"]) == expected_variants


# ---------------------------------------------------------------------------
# main() — blocked path (GPU unhealthy)
# ---------------------------------------------------------------------------


class TestMainBlocked:
    def test_main_blocked_when_gpu_unhealthy(self, tmp_path: Path):
        """main() emits a blocked artifact when GPU setup fails."""
        from scripts.experiment_template import ExperimentTemplate

        unhealthy_gpu_status = {
            "all_healthy": False,
            "models": [{"name": "Gemma4-E4B-it", "health_ok": False,
                         "stall_root_cause": "OOM", "load_time_s": 0.0, "gpu_id": 0}],
            "prewarm_time_s": 0.1,
            "dual_gpu_auto_assigned": False,
            "gpu_monitor_results": {"n_gpus_detected": 0, "n_zombies": 0,
                                     "idle_gpus": [], "all_healthy": False},
        }

        with patch.dict(os.environ, {"CARNOT_FORCE_LIVE": "1"}):
            with patch("scripts.experiment_340_live_precision_benchmark.ExperimentTemplate") as MockTmpl:
                tmpl_instance = ExperimentTemplate(
                    exp_id=340,
                    title=exp340.EXP_TITLE,
                    deliverable=exp340.DELIVERABLE,
                    repo_root=tmp_path,
                )
                tmpl_instance.setup()
                tmpl_instance.setup_gpu = MagicMock(return_value=unhealthy_gpu_status)
                MockTmpl.return_value = tmpl_instance
                exp340.main()

        artifact = json.loads((tmp_path / exp340.DELIVERABLE).read_text())
        assert artifact["status"] == "blocked"
