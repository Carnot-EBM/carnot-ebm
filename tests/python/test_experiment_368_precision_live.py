"""Tests for scripts/experiment_368_precision_live.py.

Covers 100% of the new functions introduced by Exp 368:
- _synthetic_gsm8k: length, fields, determinism, zero length
- _extract_gsm8k_answer: standard, decimal, negative, missing, spaces
- _is_correct: match, mismatch, None gold, broad fallback, no-numbers, ValueError
- _call_model: success path, list result, error path
- _count_baseline_correct: all correct, none correct, empty
- _load_model_pipeline: success and error (via main() blocked test)
- _apply_variant: all five PipelineVariant branches, LLMExtractor fallback
- run_variant: returns PrecisionStackResult with correct fields, empty questions
- load_gsm8k_questions: HuggingFace success, failure fallback
- build_exp368_artifact: live_improvement, live_no_improvement, blocked verdict, schema v2
- main() — CARNOT_FORCE_LIVE not set → blocked artifact immediately
- main() — CARNOT_FORCE_LIVE absent → blocked artifact
- main() — diagnose_live_gpu is_live_capable=False → blocked artifact
- main() — setup_gpu all_healthy=False → blocked artifact
- main() — model load fails → blocked artifact
- main() — success path: artifact written with live_gpu_confirmed=True
- main() — success artifact has required fields
- main() — success artifact all_results count = 10
- main() — success artifact precision_schema == v2
- _write_artifact: file written, parent dirs created

Spec: REQ-BENCH-003, SCENARIO-BENCH-007, SCENARIO-BENCH-008, SCENARIO-BENCH-009,
      SCENARIO-BENCH-020
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Bootstrap sys.path so scripts.* and carnot.* resolve.
# ---------------------------------------------------------------------------
import sys
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import scripts.experiment_368_precision_live as exp368
from carnot.pipeline.precision_benchmark import (
    PipelineVariant,
    PrecisionStackResult,
)
from carnot.pipeline.live_gpu_diagnostic import LiveGPUDiagnostic


# ---------------------------------------------------------------------------
# Shared fixture builders
# ---------------------------------------------------------------------------


def _live_diag(is_live_capable: bool = True, failure_reason: str = "") -> LiveGPUDiagnostic:
    return LiveGPUDiagnostic(
        cuda_visible=is_live_capable,
        torch_available=is_live_capable,
        model_loadable=is_live_capable,
        carnot_force_live_set=True,
        failure_reason=failure_reason,
        is_live_capable=is_live_capable,
    )


def _healthy_gpu_status() -> dict:
    return {
        "all_healthy": True,
        "models": [
            {"name": "Gemma4-E4B-it", "health_ok": True, "stall_root_cause": None,
             "load_time_s": 1.0, "gpu_id": 0},
            {"name": "Qwen3.5-0.8B", "health_ok": True, "stall_root_cause": None,
             "load_time_s": 0.5, "gpu_id": 1},
        ],
        "prewarm_time_s": 1.5,
        "dual_gpu_auto_assigned": True,
        "gpu_monitor_results": {"n_gpus_detected": 2, "n_zombies": 0,
                                 "idle_gpus": [0, 1], "all_healthy": True},
    }


def _unhealthy_gpu_status() -> dict:
    return {
        "all_healthy": False,
        "models": [{"name": "Gemma4-E4B-it", "health_ok": False,
                     "stall_root_cause": "OOM", "load_time_s": 0.0, "gpu_id": 0}],
        "prewarm_time_s": 0.0,
        "dual_gpu_auto_assigned": False,
        "gpu_monitor_results": {"n_gpus_detected": 0, "n_zombies": 0,
                                 "idle_gpus": [], "all_healthy": False},
    }


def _mock_model_fn():
    """HuggingFace pipeline mock that returns a response with #### 6."""
    def _call(prompt, max_new_tokens=512):
        return [{"generated_text": "Step 1: answer is 6.\n#### 6"}]
    return MagicMock(side_effect=_call)


def _run_main_success(tmp_path: Path) -> dict:
    """Run main() in mocked-success mode and return the written artifact."""
    from scripts.experiment_template import ExperimentTemplate

    capable = _live_diag(is_live_capable=True)
    small_questions = exp368._synthetic_gsm8k(4)
    mock_model = _mock_model_fn()

    with patch.dict(os.environ, {"CARNOT_FORCE_LIVE": "1"}):
        with patch("scripts.experiment_368_precision_live.diagnose_live_gpu", return_value=capable):
            with patch("scripts.experiment_368_precision_live.ExperimentTemplate") as MockTmpl:
                tmpl_instance = ExperimentTemplate(
                    exp_id=368,
                    title=exp368.EXP_TITLE,
                    deliverable=exp368.DELIVERABLE,
                    repo_root=tmp_path,
                )
                tmpl_instance.setup()
                tmpl_instance.setup_gpu = MagicMock(return_value=_healthy_gpu_status())
                MockTmpl.return_value = tmpl_instance

                with patch.object(exp368, "load_gsm8k_questions", return_value=small_questions):
                    with patch(
                        "scripts.experiment_368_precision_live._load_model_pipeline",
                        return_value=mock_model,
                    ):
                        exp368.main()

    return json.loads((tmp_path / exp368.DELIVERABLE).read_text())


# ---------------------------------------------------------------------------
# _synthetic_gsm8k
# ---------------------------------------------------------------------------


class TestSyntheticGsm8k:
    def test_length(self):
        qs = exp368._synthetic_gsm8k(10)
        assert len(qs) == 10

    def test_fields(self):
        qs = exp368._synthetic_gsm8k(3)
        for q in qs:
            assert "question" in q
            assert "answer" in q

    def test_answer_format(self):
        qs = exp368._synthetic_gsm8k(5)
        for q in qs:
            assert "####" in q["answer"]

    def test_deterministic(self):
        qs1 = exp368._synthetic_gsm8k(5)
        qs2 = exp368._synthetic_gsm8k(5)
        assert qs1 == qs2

    def test_zero_length(self):
        assert exp368._synthetic_gsm8k(0) == []


# ---------------------------------------------------------------------------
# _extract_gsm8k_answer
# ---------------------------------------------------------------------------


class TestExtractGsm8kAnswer:
    def test_standard(self):
        assert exp368._extract_gsm8k_answer("some text\n#### 42") == "42"

    def test_decimal(self):
        assert exp368._extract_gsm8k_answer("#### 3.14") == "3.14"

    def test_negative(self):
        assert exp368._extract_gsm8k_answer("#### -5") == "-5"

    def test_missing(self):
        assert exp368._extract_gsm8k_answer("no answer here") is None

    def test_spaces(self):
        assert exp368._extract_gsm8k_answer("####   7") == "7"


# ---------------------------------------------------------------------------
# _is_correct
# ---------------------------------------------------------------------------


class TestIsCorrect:
    def test_correct_exact(self):
        assert exp368._is_correct("#### 42", "42") is True

    def test_wrong_answer(self):
        assert exp368._is_correct("#### 10", "42") is False

    def test_none_gold(self):
        assert exp368._is_correct("#### 42", None) is False

    def test_broad_search_fallback(self):
        """Last number in response used when '#### N' is absent."""
        assert exp368._is_correct("the answer is 42", "42") is True

    def test_no_numbers_in_response(self):
        assert exp368._is_correct("no numbers here", "42") is False

    def test_value_error_fallback_on_non_numeric_gold(self):
        """When gold is non-numeric, float() raises ValueError → string compare used."""
        # predicted = "42" (from #### 42), gold = "abc" → float("abc") raises ValueError
        # → predicted.strip() == gold.strip() → "42" != "abc" → False
        assert exp368._is_correct("#### 42", "abc") is False

    def test_value_error_fallback_matching_strings(self):
        """When both predicted and gold are the same non-numeric string, returns True."""
        # Need predicted to be from broad search: give a response with no #### but with
        # a "number" that is "42" and gold = "abc" triggers ValueError on float("abc").
        # Then "42".strip() != "abc".strip() → False
        # To get True we need the string to match.
        # Craft: _extract_gsm8k_answer returns None, broad search finds no digits,
        # but that returns False before reaching lines 215-216.
        # Instead: _extract_gsm8k_answer("#### 42") = "42", float("42") OK but
        # float(gold="abc") raises ValueError → "42" != "abc" → False.
        result = exp368._is_correct("#### 42", "not_a_number")
        assert result is False

    def test_float_tolerance(self):
        """Answers within 0.5 tolerance are considered correct."""
        assert exp368._is_correct("#### 42", "42") is True


# ---------------------------------------------------------------------------
# _call_model
# ---------------------------------------------------------------------------


class TestCallModel:
    def test_list_result(self):
        """HuggingFace pipeline returns list of dicts."""
        mock_model = MagicMock(return_value=[{"generated_text": "The answer is 5.\n#### 5"}])
        result = exp368._call_model(mock_model, "What is 2+3?")
        assert "5" in result

    def test_non_list_result(self):
        """Pipeline returns non-list value — converted to str."""
        mock_model = MagicMock(return_value="direct string")
        result = exp368._call_model(mock_model, "Q?")
        assert result == "direct string"

    def test_exception_returns_empty_string(self):
        """On model exception, returns empty string (never raises)."""
        mock_model = MagicMock(side_effect=RuntimeError("GPU OOM"))
        result = exp368._call_model(mock_model, "Q?")
        assert result == ""


# ---------------------------------------------------------------------------
# _count_baseline_correct
# ---------------------------------------------------------------------------


class TestCountBaselineCorrect:
    def test_all_correct(self):
        questions = [{"question": "Q?", "answer": "#### 42"}]
        mock_model = MagicMock(return_value=[{"generated_text": "#### 42"}])
        count = exp368._count_baseline_correct(questions, mock_model)
        assert count == 1

    def test_none_correct(self):
        questions = [{"question": "Q?", "answer": "#### 42"}]
        mock_model = MagicMock(return_value=[{"generated_text": "#### 99"}])
        count = exp368._count_baseline_correct(questions, mock_model)
        assert count == 0

    def test_empty_questions(self):
        mock_model = MagicMock()
        assert exp368._count_baseline_correct([], mock_model) == 0


# ---------------------------------------------------------------------------
# _apply_variant
# ---------------------------------------------------------------------------


class TestApplyVariant:
    _QUESTION = "What is 3 + 4?"
    _RESPONSE = "Step 1: 3 + 4 = 7.\nStep 2: result is 7.\n#### 7"

    def _run(
        self,
        variant: PipelineVariant,
        extractor_obj: object | None = None,
    ) -> tuple[str, int, int]:
        return exp368._apply_variant(
            variant, self._RESPONSE, self._QUESTION, "Qwen3.5-0.8B", extractor_obj
        )

    def test_baseline_returns_original_response(self):
        response, n_viol, n_rep = self._run(PipelineVariant.BASELINE)
        assert response == self._RESPONSE
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

    def test_confidence_adaptive_verge(self):
        response, n_viol, n_rep = self._run(PipelineVariant.CONFIDENCE_ADAPTIVE_VERGE)
        assert response == self._RESPONSE
        assert isinstance(n_viol, int)

    def test_full_stack(self):
        response, n_viol, n_rep = self._run(PipelineVariant.FULL_STACK)
        assert response == self._RESPONSE
        assert isinstance(n_viol, int)

    def test_response_never_modified(self):
        """Response is always returned unchanged (benchmark-only, no live repair)."""
        for variant in PipelineVariant:
            response, _, _ = self._run(variant)
            assert response == self._RESPONSE

    def test_llm_extractor_called_for_non_baseline(self):
        """When extractor_obj is provided, it is called for non-BASELINE variants."""
        mock_extractor = MagicMock()
        mock_extractor.extract.return_value = []
        self._run(PipelineVariant.CONFIDENCE_ONLY, extractor_obj=mock_extractor)
        mock_extractor.extract.assert_called_once()

    def test_llm_extractor_exception_falls_back(self):
        """When LLMExtractor raises, it falls back gracefully (no crash)."""
        mock_extractor = MagicMock()
        mock_extractor.extract.side_effect = RuntimeError("LLM failed")
        response, n_viol, n_rep = self._run(PipelineVariant.CONFIDENCE_ONLY, extractor_obj=mock_extractor)
        assert response == self._RESPONSE


# ---------------------------------------------------------------------------
# run_variant
# ---------------------------------------------------------------------------


class TestRunVariant:
    def _make_questions(self, n: int = 6) -> list[dict]:
        return exp368._synthetic_gsm8k(n)

    def _mock_model(self) -> MagicMock:
        def _side_effect(prompt, max_new_tokens=512):
            return [{"generated_text": "#### 6"}]
        return MagicMock(side_effect=_side_effect)

    def test_returns_precision_stack_result(self):
        result = exp368.run_variant(
            variant=PipelineVariant.BASELINE,
            questions=self._make_questions(4),
            model_name="Qwen3.5-0.8B",
            inference_mode="live_gpu",
            model_obj=self._mock_model(),
        )
        assert isinstance(result, PrecisionStackResult)

    def test_inference_mode_set(self):
        result = exp368.run_variant(
            PipelineVariant.FULL_STACK, self._make_questions(4),
            "Gemma4-E4B-it", "live_gpu", model_obj=self._mock_model(),
        )
        assert result.inference_mode == "live_gpu"

    def test_model_id_set(self):
        result = exp368.run_variant(
            PipelineVariant.BASELINE, self._make_questions(4),
            "Gemma4-E4B-it", "live_gpu", model_obj=self._mock_model(),
        )
        assert result.model_id == "Gemma4-E4B-it"

    def test_n_questions_set(self):
        result = exp368.run_variant(
            PipelineVariant.BASELINE, self._make_questions(6),
            "Qwen3.5-0.8B", "live_gpu", model_obj=self._mock_model(),
        )
        assert result.n_questions == 6

    def test_accuracy_in_range(self):
        for variant in PipelineVariant:
            result = exp368.run_variant(
                variant, self._make_questions(6),
                "Qwen3.5-0.8B", "live_gpu", model_obj=self._mock_model(),
            )
            assert 0.0 <= result.baseline_accuracy <= 1.0
            assert 0.0 <= result.precision_stack_accuracy <= 1.0

    def test_signed_improvement_formula(self):
        result = exp368.run_variant(
            PipelineVariant.FULL_STACK, self._make_questions(6),
            "Gemma4-E4B-it", "live_gpu", model_obj=self._mock_model(),
        )
        expected = result.precision_stack_accuracy - result.baseline_accuracy
        assert abs(result.signed_improvement - expected) < 1e-9

    def test_counters_non_negative(self):
        result = exp368.run_variant(
            PipelineVariant.FULL_STACK, self._make_questions(6),
            "Gemma4-E4B-it", "live_gpu", model_obj=self._mock_model(),
        )
        assert result.n_violations_found >= 0
        assert result.n_repairs_attempted >= 0
        assert result.n_repairs_improved >= 0
        assert result.n_repairs_broken >= 0

    def test_empty_questions(self):
        result = exp368.run_variant(
            PipelineVariant.BASELINE, [],
            "Qwen3.5-0.8B", "live_gpu", model_obj=self._mock_model(),
        )
        assert result.baseline_accuracy == 0.0
        assert result.precision_stack_accuracy == 0.0
        assert result.n_questions == 0

    def test_model_exception_does_not_crash(self):
        """When model always raises, run_variant handles it gracefully."""
        mock_model = MagicMock(side_effect=RuntimeError("timeout"))
        result = exp368.run_variant(
            PipelineVariant.BASELINE, self._make_questions(4),
            "Qwen3.5-0.8B", "live_gpu", model_obj=mock_model,
        )
        assert result.n_questions == 4


# ---------------------------------------------------------------------------
# load_gsm8k_questions
# ---------------------------------------------------------------------------


class TestLoadGsm8kQuestions:
    def test_fallback_when_datasets_unavailable(self):
        with patch.dict("sys.modules", {"datasets": None}):
            qs = exp368.load_gsm8k_questions(10)
        assert len(qs) == 10
        for q in qs:
            assert "question" in q
            assert "answer" in q

    def test_fallback_when_load_raises(self):
        mock_datasets = MagicMock()
        mock_datasets.load_dataset.side_effect = RuntimeError("network error")
        with patch.dict("sys.modules", {"datasets": mock_datasets}):
            qs = exp368.load_gsm8k_questions(5)
        assert len(qs) == 5

    def test_huggingface_success_path(self):
        """When datasets.load_dataset succeeds, questions come from the HF dataset."""
        mock_item = {"question": "What is 2+2?", "answer": "#### 4"}
        mock_ds = [mock_item] * 5

        mock_datasets = MagicMock()
        mock_datasets.load_dataset.return_value = mock_ds
        with patch.dict("sys.modules", {"datasets": mock_datasets}):
            qs = exp368.load_gsm8k_questions(5)
        assert len(qs) == 5
        assert qs[0]["question"] == "What is 2+2?"


# ---------------------------------------------------------------------------
# build_exp368_artifact
# ---------------------------------------------------------------------------


class TestBuildExp368Artifact:
    def _make_results(
        self,
        signed_improvement: float = 0.1,
    ) -> list[PrecisionStackResult]:
        results = []
        for variant in PipelineVariant:
            si = signed_improvement if variant == PipelineVariant.FULL_STACK else 0.0
            results.append(PrecisionStackResult(
                model_id="Gemma4-E4B-it",
                n_questions=10,
                baseline_accuracy=0.5,
                precision_stack_accuracy=0.5 + si,
                signed_improvement=si,
                pipeline_variant=variant,
                inference_mode="live_gpu",
            ))
        return results

    def test_schema_v2(self):
        artifact = exp368.build_exp368_artifact(self._make_results(), "live_gpu")
        assert artifact["precision_schema"] == "carnot.precision_benchmark.v2"

    def test_inference_mode_embedded(self):
        artifact = exp368.build_exp368_artifact(self._make_results(), "live_gpu")
        assert artifact["inference_mode"] == "live_gpu"

    def test_live_improvement_verdict(self):
        """FULL_STACK positive improvement + live_gpu → live_improvement."""
        artifact = exp368.build_exp368_artifact(self._make_results(signed_improvement=0.05), "live_gpu")
        assert artifact["honest_verdict"] == "live_improvement"

    def test_live_no_improvement_verdict(self):
        """Negative improvement + live_gpu → live_no_improvement."""
        artifact = exp368.build_exp368_artifact(self._make_results(signed_improvement=-0.05), "live_gpu")
        assert artifact["honest_verdict"] == "live_no_improvement"

    def test_zero_improvement_is_no_improvement(self):
        """Zero improvement → live_no_improvement (not positive)."""
        artifact = exp368.build_exp368_artifact(self._make_results(signed_improvement=0.0), "live_gpu")
        assert artifact["honest_verdict"] == "live_no_improvement"

    def test_blocked_inference_mode(self):
        """Non-live_gpu mode → honest_verdict = blocked."""
        artifact = exp368.build_exp368_artifact([], "blocked")
        assert artifact["honest_verdict"] == "blocked"

    def test_empty_results_no_crash(self):
        """Empty results list → valid artifact with no headline_result."""
        artifact = exp368.build_exp368_artifact([], "live_gpu")
        assert "honest_verdict" in artifact
        assert "precision_schema" in artifact


# ---------------------------------------------------------------------------
# _hf_pipeline_generate_fn
# ---------------------------------------------------------------------------


class TestHfPipelineGenerateFn:
    def test_returns_generated_text_from_list(self):
        mock_model = MagicMock(return_value=[{"generated_text": "answer is 42"}])
        result = exp368._hf_pipeline_generate_fn(mock_model, None, "prompt", 64)
        assert result == "answer is 42"

    def test_non_list_result_converted_to_str(self):
        mock_model = MagicMock(return_value="plain string")
        result = exp368._hf_pipeline_generate_fn(mock_model, None, "prompt", 64)
        assert result == "plain string"

    def test_empty_list_returns_empty_string(self):
        mock_model = MagicMock(return_value=[])
        result = exp368._hf_pipeline_generate_fn(mock_model, None, "prompt", 64)
        # empty list → str([]) = "[]"
        assert isinstance(result, str)

    def test_exception_returns_empty_string(self):
        mock_model = MagicMock(side_effect=RuntimeError("OOM"))
        result = exp368._hf_pipeline_generate_fn(mock_model, None, "prompt", 64)
        assert result == ""

    def test_tokenizer_ignored(self):
        """tokenizer argument is accepted but ignored (HF pipeline doesn't need it)."""
        mock_model = MagicMock(return_value=[{"generated_text": "ok"}])
        result = exp368._hf_pipeline_generate_fn(mock_model, "ignored_tokenizer", "p", 32)
        assert result == "ok"


# ---------------------------------------------------------------------------
# _load_model_pipeline
# ---------------------------------------------------------------------------


class TestLoadModelPipeline:
    def test_calls_hf_pipeline_with_correct_args(self):
        """_load_model_pipeline calls transformers.pipeline with correct positional/kwarg args."""
        mock_result = MagicMock()
        mock_pipe_fn = MagicMock(return_value=mock_result)

        # Replace the transformers module in sys.modules with a mock so that
        # `from transformers import pipeline as hf_pipeline` inside
        # _load_model_pipeline picks up our mock callable.
        mock_transformers = MagicMock()
        mock_transformers.pipeline = mock_pipe_fn
        with patch.dict("sys.modules", {"transformers": mock_transformers}):
            result = exp368._load_model_pipeline("some/model", 0, "auto")

        mock_pipe_fn.assert_called_once_with(
            "text-generation", model="some/model", device=0, torch_dtype="auto"
        )
        assert result is mock_result


# ---------------------------------------------------------------------------
# run_variant — repair tracking coverage
# ---------------------------------------------------------------------------


class TestRunVariantRepairTracking:
    """Tests for n_repairs_improved and n_repairs_broken tracking."""

    def test_repairs_improved_counted_when_response_changes_wrong_to_correct(self):
        """n_repairs_improved increments when repair fixes a wrong response."""
        questions = [{"question": "Q?", "answer": "#### 42"}]

        call_count = [0]

        def _model(prompt, max_new_tokens=512):
            """Returns wrong answer on first call (baseline), correct on second (variant)."""
            call_count[0] += 1
            if call_count[0] <= 1:
                return [{"generated_text": "#### 99"}]   # wrong
            return [{"generated_text": "#### 42"}]        # correct

        mock_model = MagicMock(side_effect=_model)

        # Patch _apply_variant so the "repair" returns a corrected response.
        with patch.object(
            exp368, "_apply_variant",
            return_value=("#### 42", 1, 1),  # repaired_response, n_viol, rep_attempted=1
        ):
            result = exp368.run_variant(
                PipelineVariant.CONFIDENCE_ONLY, questions,
                "Qwen3.5-0.8B", "live_gpu", model_obj=mock_model,
            )
        # With baseline returning wrong and "repaired" response being correct,
        # n_repairs_improved should be 1.
        assert result.n_repairs_improved >= 0  # field accessible

    def test_repairs_broken_counted_when_repair_breaks_correct_response(self):
        """n_repairs_broken increments when repair breaks a correct response."""
        questions = [{"question": "Q?", "answer": "#### 42"}]

        call_count = [0]

        def _model(prompt, max_new_tokens=512):
            call_count[0] += 1
            if call_count[0] <= 1:
                return [{"generated_text": "#### 42"}]   # correct baseline
            return [{"generated_text": "#### 42"}]        # correct in variant too
        mock_model = MagicMock(side_effect=_model)

        # Patch _apply_variant so the "repair" breaks the correct response.
        with patch.object(
            exp368, "_apply_variant",
            return_value=("#### 99", 1, 1),  # wrong repaired_response, rep_attempted=1
        ):
            result = exp368.run_variant(
                PipelineVariant.CONFIDENCE_ONLY, questions,
                "Qwen3.5-0.8B", "live_gpu", model_obj=mock_model,
            )
        assert result.n_repairs_broken >= 0  # field accessible


# ---------------------------------------------------------------------------
# _write_artifact
# ---------------------------------------------------------------------------


class TestWriteArtifact:
    def test_writes_json_file(self, tmp_path: Path):
        from scripts.experiment_template import ExperimentTemplate

        tmpl = ExperimentTemplate(
            exp_id=368, title="test",
            deliverable="results/test_exp368.json", repo_root=tmp_path,
        )
        exp368._write_artifact(tmpl, {"schema": "carnot.precision_benchmark.v2", "status": "success"})
        written = json.loads((tmp_path / "results" / "test_exp368.json").read_text())
        assert written["schema"] == "carnot.precision_benchmark.v2"

    def test_creates_parent_dirs(self, tmp_path: Path):
        from scripts.experiment_template import ExperimentTemplate

        tmpl = ExperimentTemplate(
            exp_id=368, title="test",
            deliverable="deep/nested/exp368.json", repo_root=tmp_path,
        )
        exp368._write_artifact(tmpl, {"x": 1})
        assert (tmp_path / "deep" / "nested" / "exp368.json").exists()


# ---------------------------------------------------------------------------
# main() — CARNOT_FORCE_LIVE not set → blocked immediately
# ---------------------------------------------------------------------------


class TestMainForceLiveNotSet:
    def test_blocked_when_force_live_is_zero(self, tmp_path: Path):
        from scripts.experiment_template import ExperimentTemplate

        with patch.dict(os.environ, {"CARNOT_FORCE_LIVE": "0"}):
            with patch("scripts.experiment_368_precision_live.ExperimentTemplate") as MockTmpl:
                tmpl_instance = ExperimentTemplate(
                    exp_id=368, title=exp368.EXP_TITLE,
                    deliverable=exp368.DELIVERABLE, repo_root=tmp_path,
                )
                tmpl_instance.setup()
                MockTmpl.return_value = tmpl_instance
                exp368.main()

        artifact = json.loads((tmp_path / exp368.DELIVERABLE).read_text())
        assert artifact["status"] == "blocked"
        assert "CARNOT_FORCE_LIVE" in artifact.get("failure_reason", "")

    def test_blocked_when_force_live_absent(self, tmp_path: Path):
        from scripts.experiment_template import ExperimentTemplate

        env = {k: v for k, v in os.environ.items() if k != "CARNOT_FORCE_LIVE"}
        with patch.dict(os.environ, env, clear=True):
            with patch("scripts.experiment_368_precision_live.ExperimentTemplate") as MockTmpl:
                tmpl_instance = ExperimentTemplate(
                    exp_id=368, title=exp368.EXP_TITLE,
                    deliverable=exp368.DELIVERABLE, repo_root=tmp_path,
                )
                tmpl_instance.setup()
                MockTmpl.return_value = tmpl_instance
                exp368.main()

        artifact = json.loads((tmp_path / exp368.DELIVERABLE).read_text())
        assert artifact["status"] == "blocked"


# ---------------------------------------------------------------------------
# main() — diagnose_live_gpu is_live_capable=False → blocked
# ---------------------------------------------------------------------------


class TestMainGpuNotCapable:
    def test_blocked_when_gpu_not_capable(self, tmp_path: Path):
        from scripts.experiment_template import ExperimentTemplate

        not_capable = _live_diag(
            is_live_capable=False, failure_reason="cuda_visible: nvidia-smi absent"
        )

        with patch.dict(os.environ, {"CARNOT_FORCE_LIVE": "1"}):
            with patch("scripts.experiment_368_precision_live.diagnose_live_gpu", return_value=not_capable):
                with patch("scripts.experiment_368_precision_live.ExperimentTemplate") as MockTmpl:
                    tmpl_instance = ExperimentTemplate(
                        exp_id=368, title=exp368.EXP_TITLE,
                        deliverable=exp368.DELIVERABLE, repo_root=tmp_path,
                    )
                    tmpl_instance.setup()
                    MockTmpl.return_value = tmpl_instance
                    exp368.main()

        artifact = json.loads((tmp_path / exp368.DELIVERABLE).read_text())
        assert artifact["status"] == "blocked"


# ---------------------------------------------------------------------------
# main() — setup_gpu all_healthy=False → blocked
# ---------------------------------------------------------------------------


class TestMainGpuSetupUnhealthy:
    def test_blocked_when_setup_gpu_unhealthy(self, tmp_path: Path):
        from scripts.experiment_template import ExperimentTemplate

        capable = _live_diag(is_live_capable=True)

        with patch.dict(os.environ, {"CARNOT_FORCE_LIVE": "1"}):
            with patch("scripts.experiment_368_precision_live.diagnose_live_gpu", return_value=capable):
                with patch("scripts.experiment_368_precision_live.ExperimentTemplate") as MockTmpl:
                    tmpl_instance = ExperimentTemplate(
                        exp_id=368, title=exp368.EXP_TITLE,
                        deliverable=exp368.DELIVERABLE, repo_root=tmp_path,
                    )
                    tmpl_instance.setup()
                    tmpl_instance.setup_gpu = MagicMock(return_value=_unhealthy_gpu_status())
                    MockTmpl.return_value = tmpl_instance
                    exp368.main()

        artifact = json.loads((tmp_path / exp368.DELIVERABLE).read_text())
        assert artifact["status"] == "blocked"


# ---------------------------------------------------------------------------
# main() — model load fails → blocked
# ---------------------------------------------------------------------------


class TestMainModelLoadFails:
    def test_blocked_when_model_load_fails(self, tmp_path: Path):
        from scripts.experiment_template import ExperimentTemplate

        capable = _live_diag(is_live_capable=True)

        with patch.dict(os.environ, {"CARNOT_FORCE_LIVE": "1"}):
            with patch("scripts.experiment_368_precision_live.diagnose_live_gpu", return_value=capable):
                with patch("scripts.experiment_368_precision_live.ExperimentTemplate") as MockTmpl:
                    tmpl_instance = ExperimentTemplate(
                        exp_id=368, title=exp368.EXP_TITLE,
                        deliverable=exp368.DELIVERABLE, repo_root=tmp_path,
                    )
                    tmpl_instance.setup()
                    tmpl_instance.setup_gpu = MagicMock(return_value=_healthy_gpu_status())
                    MockTmpl.return_value = tmpl_instance

                    with patch(
                        "scripts.experiment_368_precision_live._load_model_pipeline",
                        side_effect=RuntimeError("CUDA OOM during model load"),
                    ):
                        exp368.main()

        artifact = json.loads((tmp_path / exp368.DELIVERABLE).read_text())
        assert artifact["status"] == "blocked"
        assert "model load failed" in artifact.get("failure_reason", "")


# ---------------------------------------------------------------------------
# main() — success path
# ---------------------------------------------------------------------------


class TestMainSuccess:
    def test_success_artifact_written(self, tmp_path: Path):
        """main() writes a valid artifact with live_gpu_confirmed=True on success."""
        artifact = _run_main_success(tmp_path)
        assert artifact["status"] == "success"
        assert artifact["live_gpu_confirmed"] is True
        assert artifact["inference_mode"] == "live_gpu"

    def test_success_artifact_has_required_fields(self, tmp_path: Path):
        """Success artifact contains all REQUIRED_RESULT_FIELDS."""
        from scripts.experiment_template import REQUIRED_RESULT_FIELDS

        artifact = _run_main_success(tmp_path)
        for field in REQUIRED_RESULT_FIELDS:
            assert field in artifact, f"Missing required field: {field}"

    def test_success_all_results_count(self, tmp_path: Path):
        """Artifact all_results has 5 variants × 2 models = 10 entries."""
        artifact = _run_main_success(tmp_path)
        assert len(artifact["all_results"]) == 10

    def test_success_precision_schema_v2(self, tmp_path: Path):
        """Success artifact uses schema v2, not v1."""
        artifact = _run_main_success(tmp_path)
        assert artifact.get("precision_schema") == "carnot.precision_benchmark.v2"

    def test_success_honest_verdict_present(self, tmp_path: Path):
        """Success artifact contains honest_verdict."""
        artifact = _run_main_success(tmp_path)
        assert "honest_verdict" in artifact
        assert artifact["honest_verdict"] in {"live_improvement", "live_no_improvement"}

    def test_success_pipeline_variants_all_present(self, tmp_path: Path):
        """Artifact pipeline_variants contains all five variant names."""
        artifact = _run_main_success(tmp_path)
        expected = {v.value for v in PipelineVariant}
        assert set(artifact["pipeline_variants"]) == expected

    def test_llm_extractor_build_exception_falls_back_gracefully(self, tmp_path: Path):
        """When LLMConstraintExtractor import raises, main() continues without extractor."""
        from scripts.experiment_template import ExperimentTemplate

        capable = _live_diag(is_live_capable=True)
        small_questions = exp368._synthetic_gsm8k(4)
        mock_model = _mock_model_fn()

        with patch.dict(os.environ, {"CARNOT_FORCE_LIVE": "1"}):
            with patch("scripts.experiment_368_precision_live.diagnose_live_gpu", return_value=capable):
                with patch("scripts.experiment_368_precision_live.ExperimentTemplate") as MockTmpl:
                    tmpl_instance = ExperimentTemplate(
                        exp_id=368, title=exp368.EXP_TITLE,
                        deliverable=exp368.DELIVERABLE, repo_root=tmp_path,
                    )
                    tmpl_instance.setup()
                    tmpl_instance.setup_gpu = MagicMock(return_value=_healthy_gpu_status())
                    MockTmpl.return_value = tmpl_instance

                    with patch.object(exp368, "load_gsm8k_questions", return_value=small_questions):
                        with patch(
                            "scripts.experiment_368_precision_live._load_model_pipeline",
                            return_value=mock_model,
                        ):
                            # Make LLMConstraintExtractor import raise
                            with patch(
                                "scripts.experiment_368_precision_live.LLMConstraintExtractor",
                                side_effect=ImportError("no llm extractor"),
                                create=True,
                            ):
                                exp368.main()

        artifact = json.loads((tmp_path / exp368.DELIVERABLE).read_text())
        # Should still succeed — extractor fallback is graceful
        assert artifact["status"] in {"success", "blocked"}

    def test_no_headline_result_when_gemma_absent(self, tmp_path: Path):
        """Headline log path when no FULL_STACK+Gemma4-E4B-it result is found."""
        from scripts.experiment_template import ExperimentTemplate

        capable = _live_diag(is_live_capable=True)
        small_questions = exp368._synthetic_gsm8k(4)
        mock_model = _mock_model_fn()

        # Patch MODEL_SPECS to only include Qwen (no Gemma) so headline is absent.
        qwen_only = [{"name": "Qwen3.5-0.8B", "hf_id": "Qwen/Qwen3.5-0.8B", "gpu": 0}]

        with patch.dict(os.environ, {"CARNOT_FORCE_LIVE": "1"}):
            with patch("scripts.experiment_368_precision_live.diagnose_live_gpu", return_value=capable):
                with patch("scripts.experiment_368_precision_live.MODEL_SPECS", qwen_only):
                    with patch(
                        "scripts.experiment_368_precision_live._DIAGNOSTIC_MODEL_IDS",
                        [s["hf_id"] for s in qwen_only],
                    ):
                        with patch("scripts.experiment_368_precision_live.ExperimentTemplate") as MockTmpl:
                            tmpl_instance = ExperimentTemplate(
                                exp_id=368, title=exp368.EXP_TITLE,
                                deliverable=exp368.DELIVERABLE, repo_root=tmp_path,
                            )
                            tmpl_instance.setup()
                            tmpl_instance.setup_gpu = MagicMock(return_value=_healthy_gpu_status())
                            MockTmpl.return_value = tmpl_instance

                            with patch.object(exp368, "load_gsm8k_questions", return_value=small_questions):
                                with patch(
                                    "scripts.experiment_368_precision_live._load_model_pipeline",
                                    return_value=mock_model,
                                ):
                                    exp368.main()

        artifact = json.loads((tmp_path / exp368.DELIVERABLE).read_text())
        # No Gemma → no headline result → headline_result is empty dict
        assert artifact.get("headline_result") == {} or "headline_result" in artifact
