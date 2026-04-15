"""Tests for scripts/experiment_355_adversarial_gsm8k_benchmark.py.

100% targeted coverage of all functions in Exp 355:
    - _synthetic_gsm8k: length, fields, determinism
    - load_gsm8k_questions: success path (mocked datasets), fallback path
    - _extract_answer: GSM8K format, last-number fallback, no-number case
    - _is_correct: match, mismatch, None pred, non-numeric gold
    - _simulate_response: correct path, error-injected paths
    - _call_model: callable, generate-method, fallback
    - run_adversarial_benchmark: simulated mode (no CARNOT_FORCE_LIVE),
        honest_verdict blocked_simulated in simulated mode
    - _build_per_model_result: all fields present
    - _compute_top_level_verdict: all four verdicts
    - main(): simulated end-to-end, artifact schema/fields

Spec: REQ-BENCH-006, REQ-BENCH-007,
      SCENARIO-BENCH-017, SCENARIO-BENCH-018, SCENARIO-BENCH-019
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Repo-root sys.path injection
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import scripts.experiment_355_adversarial_gsm8k_benchmark as exp355
from carnot.pipeline.adversarial_gsm8k import (
    AdversarialBenchmarkResult,
    SYNTHETIC_CI_RESULTS,
    build_adversarial_artifact,
    build_adversarial_questions,
    compute_adversarial_results,
)


# ---------------------------------------------------------------------------
# _synthetic_gsm8k
# ---------------------------------------------------------------------------


class TestSyntheticGsm8k:
    """SCENARIO-BENCH-017: CI-safe data generation."""

    def test_length(self):
        """Returns exactly n questions."""
        qs = exp355._synthetic_gsm8k(10)
        assert len(qs) == 10

    def test_fields_present(self):
        """Every question has question_id, question, and answer fields."""
        qs = exp355._synthetic_gsm8k(5)
        for q in qs:
            assert "question_id" in q
            assert "question" in q
            assert "answer" in q

    def test_question_id_format(self):
        """question_id is zero-padded 4-digit synthetic ID."""
        qs = exp355._synthetic_gsm8k(3)
        assert qs[0]["question_id"] == "synth_0000"
        assert qs[2]["question_id"] == "synth_0002"

    def test_answer_is_numeric(self):
        """answer field is a numeric string."""
        qs = exp355._synthetic_gsm8k(5)
        for q in qs:
            assert q["answer"].isdigit()

    def test_determinism(self):
        """Two calls with same n return identical questions."""
        a = exp355._synthetic_gsm8k(10)
        b = exp355._synthetic_gsm8k(10)
        assert a == b

    def test_zero_questions(self):
        """Zero-length call returns empty list without error."""
        assert exp355._synthetic_gsm8k(0) == []


# ---------------------------------------------------------------------------
# load_gsm8k_questions
# ---------------------------------------------------------------------------


class TestLoadGsm8kQuestions:
    """load_gsm8k_questions falls back on import errors (SCENARIO-BENCH-017)."""

    def test_fallback_on_import_error(self):
        """Falls back to _synthetic_gsm8k when datasets is not importable."""
        with patch.dict("sys.modules", {"datasets": None}):
            qs = exp355.load_gsm8k_questions(5)
        assert len(qs) == 5
        assert "question" in qs[0]

    def test_fallback_on_dataset_exception(self):
        """Falls back to _synthetic_gsm8k when load_dataset raises."""
        mock_datasets = MagicMock()
        mock_datasets.load_dataset.side_effect = RuntimeError("no network")
        with patch.dict("sys.modules", {"datasets": mock_datasets}):
            qs = exp355.load_gsm8k_questions(3)
        assert len(qs) == 3

    def test_success_path_with_mock(self):
        """Parses question + answer from mocked HuggingFace dataset row."""
        mock_row = {"question": "What is 2+2?", "answer": "Something\n#### 4"}
        mock_ds = MagicMock()
        mock_ds.__len__ = MagicMock(return_value=1)
        mock_ds.select.return_value = [mock_row]
        mock_datasets = MagicMock()
        mock_datasets.load_dataset.return_value = mock_ds
        with patch.dict("sys.modules", {"datasets": mock_datasets}):
            qs = exp355.load_gsm8k_questions(1)
        assert len(qs) == 1
        assert qs[0]["answer"] == "4"
        assert qs[0]["question"] == "What is 2+2?"

    def test_success_no_hash_format(self):
        """Falls back to raw answer string when no #### marker present."""
        mock_row = {"question": "Q?", "answer": "Just the answer: 42"}
        mock_ds = MagicMock()
        mock_ds.__len__ = MagicMock(return_value=1)
        mock_ds.select.return_value = [mock_row]
        mock_datasets = MagicMock()
        mock_datasets.load_dataset.return_value = mock_ds
        with patch.dict("sys.modules", {"datasets": mock_datasets}):
            qs = exp355.load_gsm8k_questions(1)
        assert qs[0]["answer"] == "Just the answer: 42"


# ---------------------------------------------------------------------------
# _extract_answer
# ---------------------------------------------------------------------------


class TestExtractAnswer:
    """All answer-extraction paths."""

    def test_gsm8k_format(self):
        assert exp355._extract_answer("blah\n#### 42") == "42"

    def test_gsm8k_with_spaces(self):
        assert exp355._extract_answer("####   99") == "99"

    def test_decimal(self):
        assert exp355._extract_answer("#### 3.14") == "3.14"

    def test_negative(self):
        assert exp355._extract_answer("#### -5") == "-5"

    def test_comma_stripped(self):
        assert exp355._extract_answer("#### 1,000") == "1000"

    def test_last_number_fallback(self):
        """Returns last number when no #### marker present."""
        assert exp355._extract_answer("The answer is 42 apples.") == "42"

    def test_no_number_returns_none(self):
        """Returns None when no number found."""
        assert exp355._extract_answer("No numbers here.") is None

    def test_empty_string_returns_none(self):
        assert exp355._extract_answer("") is None


# ---------------------------------------------------------------------------
# _is_correct
# ---------------------------------------------------------------------------


class TestIsCorrect:
    """Correctness checking including edge cases."""

    def test_exact_numeric_match(self):
        assert exp355._is_correct("#### 42", "42") is True

    def test_numeric_mismatch(self):
        assert exp355._is_correct("#### 43", "42") is False

    def test_float_tolerance(self):
        assert exp355._is_correct("#### 42.0", "42") is True

    def test_no_number_in_response(self):
        assert exp355._is_correct("I don't know.", "42") is False

    def test_string_fallback_match(self):
        """Non-numeric gold uses string equality."""
        assert exp355._is_correct("The answer is abc", "abc") is False  # extraction gives "abc" not matched

    def test_with_comma_in_gold(self):
        assert exp355._is_correct("#### 1000", "1,000") is True


# ---------------------------------------------------------------------------
# _simulate_response
# ---------------------------------------------------------------------------


class TestSimulateResponse:
    """Synthetic response generation for CI mode."""

    def test_correct_format(self):
        """Most questions return #### <answer> format."""
        # question length % 10 != 3 and != 7 → correct
        q = "A" * 10  # length 10, 10 % 10 = 0 → correct
        r = exp355._simulate_response(q, "42")
        assert "42" in r

    def test_error_injection_index_3(self):
        """Questions with len%10==3 return wrong answer."""
        q = "A" * 3  # length 3, 3 % 10 = 3 → wrong
        r = exp355._simulate_response(q, "10")
        # Wrong answer: 10 + 1 = 11 (since "10".isdigit() is True)
        assert "11" in r

    def test_error_injection_index_7(self):
        """Questions with len%10==7 return wrong answer."""
        q = "A" * 7  # length 7, 7 % 10 = 7 → wrong
        r = exp355._simulate_response(q, "5")
        assert "6" in r

    def test_non_digit_answer(self):
        """Non-digit answer falls back to 999 for error injection."""
        q = "A" * 3  # triggers error injection
        r = exp355._simulate_response(q, "notanumber")
        assert "999" in r


# ---------------------------------------------------------------------------
# _call_model
# ---------------------------------------------------------------------------


class TestCallModel:
    """Model interface adapters."""

    def test_callable_model(self):
        """Callable model_obj(prompt) -> str."""
        model = lambda p: "answer is 42"  # noqa: E731
        # callable without .generate
        assert exp355._call_model(model, "what?") == "answer is 42"

    def test_generate_method(self):
        """model_obj.generate(prompt) interface."""
        model = MagicMock(spec=["generate"])
        model.generate.return_value = "generated answer"
        assert exp355._call_model(model, "q") == "generated answer"

    def test_fallback_to_str(self):
        """Fallback: str(model_obj) when no callable/generate."""

        class _NotCallable:
            """Non-callable object with no generate method."""

            def __str__(self) -> str:
                return "fallback_str"

        model = _NotCallable()
        result = exp355._call_model(model, "q")
        assert result == "fallback_str"


# ---------------------------------------------------------------------------
# run_adversarial_benchmark — simulated mode
# ---------------------------------------------------------------------------


class TestRunAdversarialBenchmarkSimulated:
    """SCENARIO-BENCH-017: CI-safe path, no live GPU."""

    def _make_questions(self, n: int = 5) -> list:
        raw = exp355._synthetic_gsm8k(n)
        return build_adversarial_questions(raw, seed=42)

    def test_simulated_returns_synthetic_ci_results(self):
        """Without CARNOT_FORCE_LIVE=1, returns SYNTHETIC_CI_RESULTS."""
        env = {k: v for k, v in os.environ.items() if k != "CARNOT_FORCE_LIVE"}
        with patch.dict(os.environ, env, clear=True):
            result = exp355.run_adversarial_benchmark(
                "test-model", self._make_questions(), pipeline=None
            )
        assert result.inference_mode == "simulated"

    def test_simulated_mode_returns_known_values(self):
        """Simulated result matches SYNTHETIC_CI_RESULTS constants."""
        with patch.dict(os.environ, {}, clear=True):
            result = exp355.run_adversarial_benchmark(
                "test-model", self._make_questions(), pipeline=None
            )
        assert result.standard_accuracy == SYNTHETIC_CI_RESULTS.standard_accuracy
        assert result.adversarial_accuracy == SYNTHETIC_CI_RESULTS.adversarial_accuracy

    def test_honest_verdict_blocked_when_simulated(self):
        """SCENARIO-BENCH-017: artifact honest_verdict is 'blocked_simulated'."""
        with patch.dict(os.environ, {}, clear=True):
            result = exp355.run_adversarial_benchmark(
                "test-model", self._make_questions(), pipeline=None
            )
        artifact = build_adversarial_artifact(result)
        assert artifact["honest_verdict"] == "blocked_simulated"

    def test_inference_mode_explicit_simulated(self):
        """Passing inference_mode='simulated' also triggers CI path."""
        result = exp355.run_adversarial_benchmark(
            "test-model",
            self._make_questions(),
            pipeline=None,
            inference_mode="simulated",
        )
        assert result.inference_mode == "simulated"

    def test_honest_verdict_not_improvement_when_simulated(self):
        """SCENARIO-BENCH-018: honest_verdict is never 'improvement_positive' in simulated mode."""
        with patch.dict(os.environ, {}, clear=True):
            result = exp355.run_adversarial_benchmark(
                "test-model", self._make_questions(), pipeline=None
            )
        artifact = build_adversarial_artifact(result)
        assert artifact["honest_verdict"] != "improvement_positive"


# ---------------------------------------------------------------------------
# _build_per_model_result
# ---------------------------------------------------------------------------


class TestBuildPerModelResult:
    """SCENARIO-BENCH-019: per-model result dict fields."""

    def _make_result(self, mode: str = "simulated") -> AdversarialBenchmarkResult:
        return AdversarialBenchmarkResult(
            standard_accuracy=0.8,
            adversarial_accuracy=0.65,
            accuracy_drop=0.15,
            repaired_adversarial_accuracy=0.72,
            repair_improvement=0.07,
            inference_mode=mode,
        )

    def test_all_required_fields(self):
        """All SCENARIO-BENCH-019 fields are present."""
        r = exp355._build_per_model_result("MyModel", self._make_result(), 100)
        assert r["model_id"] == "MyModel"
        assert r["n_questions"] == 100
        assert r["standard_accuracy"] == 0.8
        assert r["adversarial_accuracy"] == 0.65
        assert r["accuracy_drop"] == 0.15
        assert r["repaired_adversarial_accuracy"] == 0.72
        assert r["repair_improvement"] == 0.07
        assert r["inference_mode"] == "simulated"

    def test_live_gpu_mode_preserved(self):
        """inference_mode='live_gpu' is stored faithfully."""
        r = exp355._build_per_model_result("M", self._make_result("live_gpu"), 50)
        assert r["inference_mode"] == "live_gpu"


# ---------------------------------------------------------------------------
# _compute_top_level_verdict
# ---------------------------------------------------------------------------


class TestComputeTopLevelVerdict:
    """SCENARIO-BENCH-018/019: all four verdict branches."""

    def _model(self, repair: float, drop: float) -> dict:
        return {"repair_improvement": repair, "accuracy_drop": drop}

    def test_blocked_simulated_when_not_live(self):
        """inference_mode != 'live_gpu' → 'blocked_simulated'."""
        models = [self._model(0.1, 0.1)]
        assert exp355._compute_top_level_verdict(models, "simulated") == "blocked_simulated"

    def test_improvement_positive_when_any_model_improves(self):
        """SCENARIO-BENCH-018: live_gpu + repair_improvement > 0 → 'improvement_positive'."""
        models = [self._model(0.0, 0.1), self._model(0.05, 0.1)]
        assert exp355._compute_top_level_verdict(models, "live_gpu") == "improvement_positive"

    def test_improvement_positive_requires_live_gpu(self):
        """'improvement_positive' only when inference_mode == 'live_gpu'."""
        models = [self._model(0.1, 0.1)]
        result = exp355._compute_top_level_verdict(models, "simulated")
        assert result == "blocked_simulated"
        # Not "improvement_positive" even though repair_improvement > 0
        assert result != "improvement_positive"

    def test_degradation_positive_all_models_drop_no_repair(self):
        """live_gpu + all models: repair_improvement <= 0 + accuracy_drop > 0 → 'degradation_positive'."""
        models = [self._model(-0.01, 0.1), self._model(0.0, 0.2)]
        assert exp355._compute_top_level_verdict(models, "live_gpu") == "degradation_positive"

    def test_neutral_when_no_drop_no_improvement(self):
        """live_gpu + repair_improvement <= 0 + accuracy_drop <= 0 → 'neutral'."""
        models = [self._model(0.0, 0.0), self._model(-0.01, -0.01)]
        assert exp355._compute_top_level_verdict(models, "live_gpu") == "neutral"

    def test_single_model_improvement_positive(self):
        """Single model with positive repair → 'improvement_positive'."""
        models = [self._model(0.03, 0.1)]
        assert exp355._compute_top_level_verdict(models, "live_gpu") == "improvement_positive"


# ---------------------------------------------------------------------------
# main() — simulated end-to-end
# ---------------------------------------------------------------------------


class TestMain:
    """SCENARIO-BENCH-019: full artifact schema and fields from main()."""

    def test_main_simulated_writes_artifact(self, tmp_path: Path):
        """main() in simulated mode writes a valid artifact to DELIVERABLE."""
        with (
            patch.object(exp355, "_REPO_ROOT", tmp_path),
            patch.dict(os.environ, {}, clear=True),
        ):
            # Ensure CARNOT_FORCE_LIVE is not set
            env = {k: v for k, v in os.environ.items() if k != "CARNOT_FORCE_LIVE"}
            with patch.dict(os.environ, env, clear=True):
                exp355.main()

        artifact_path = tmp_path / exp355.DELIVERABLE
        assert artifact_path.exists(), "Artifact file must be written"

        artifact = json.loads(artifact_path.read_text())

        # Required REQUIRED_RESULT_FIELDS from ExperimentTemplate
        for field in ["experiment", "run_date", "started_at", "finished_at", "duration_s", "status", "title"]:
            assert field in artifact, f"Missing required field: {field}"

        # Exp-355-specific fields
        assert artifact["experiment"] == 355
        assert "per_model_results" in artifact
        assert "headline_result" in artifact
        assert "honest_verdict" in artifact
        assert "inference_mode" in artifact

    def test_main_simulated_honest_verdict(self, tmp_path: Path):
        """main() in simulated mode always produces honest_verdict='blocked_simulated'."""
        with patch.object(exp355, "_REPO_ROOT", tmp_path):
            env = {k: v for k, v in os.environ.items() if k != "CARNOT_FORCE_LIVE"}
            with patch.dict(os.environ, env, clear=True):
                exp355.main()

        artifact = json.loads((tmp_path / exp355.DELIVERABLE).read_text())
        assert artifact["honest_verdict"] == "blocked_simulated"

    def test_main_per_model_results_have_required_fields(self, tmp_path: Path):
        """Each per_model_results entry has all SCENARIO-BENCH-019 required fields."""
        with patch.object(exp355, "_REPO_ROOT", tmp_path):
            env = {k: v for k, v in os.environ.items() if k != "CARNOT_FORCE_LIVE"}
            with patch.dict(os.environ, env, clear=True):
                exp355.main()

        artifact = json.loads((tmp_path / exp355.DELIVERABLE).read_text())
        for entry in artifact["per_model_results"]:
            for key in [
                "model_id",
                "n_questions",
                "standard_accuracy",
                "adversarial_accuracy",
                "accuracy_drop",
                "repaired_adversarial_accuracy",
                "repair_improvement",
                "inference_mode",
            ]:
                assert key in entry, f"per_model_results entry missing: {key}"

    def test_main_headline_result_fields(self, tmp_path: Path):
        """headline_result contains honest_verdict, inference_mode, avg metrics."""
        with patch.object(exp355, "_REPO_ROOT", tmp_path):
            env = {k: v for k, v in os.environ.items() if k != "CARNOT_FORCE_LIVE"}
            with patch.dict(os.environ, env, clear=True):
                exp355.main()

        artifact = json.loads((tmp_path / exp355.DELIVERABLE).read_text())
        hl = artifact["headline_result"]
        assert "honest_verdict" in hl
        assert "inference_mode" in hl
        assert "n_models" in hl
        assert "avg_accuracy_drop" in hl
        assert "avg_repair_improvement" in hl

    def test_main_with_exp353_present(self, tmp_path: Path):
        """main() reads exp353 status when file is present."""
        results_dir = tmp_path / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        exp353_path = results_dir / "experiment_353_live_gpu_smoke_test.json"
        exp353_path.write_text(json.dumps({"finding": "live_confirmed"}))

        with patch.object(exp355, "_REPO_ROOT", tmp_path):
            env = {k: v for k, v in os.environ.items() if k != "CARNOT_FORCE_LIVE"}
            with patch.dict(os.environ, env, clear=True):
                exp355.main()

        artifact = json.loads((tmp_path / exp355.DELIVERABLE).read_text())
        assert artifact["exp353_live_gpu_status"] == "live_confirmed"

    def test_main_with_exp353_missing(self, tmp_path: Path):
        """main() handles missing exp353 file gracefully."""
        with patch.object(exp355, "_REPO_ROOT", tmp_path):
            env = {k: v for k, v in os.environ.items() if k != "CARNOT_FORCE_LIVE"}
            with patch.dict(os.environ, env, clear=True):
                exp355.main()

        artifact = json.loads((tmp_path / exp355.DELIVERABLE).read_text())
        assert "exp353_live_gpu_status" in artifact

    def test_main_n_models_equals_model_specs(self, tmp_path: Path):
        """Artifact n_models matches the number of MODEL_SPECS entries."""
        with patch.object(exp355, "_REPO_ROOT", tmp_path):
            env = {k: v for k, v in os.environ.items() if k != "CARNOT_FORCE_LIVE"}
            with patch.dict(os.environ, env, clear=True):
                exp355.main()

        artifact = json.loads((tmp_path / exp355.DELIVERABLE).read_text())
        assert artifact["n_models"] == len(exp355.MODEL_SPECS)
        assert len(artifact["per_model_results"]) == len(exp355.MODEL_SPECS)
