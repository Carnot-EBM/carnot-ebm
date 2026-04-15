"""Tests for python/carnot/pipeline/extractor_comparison.py (Exp 367 additions) and
scripts/experiment_367_extraction_live.py.

100% targeted coverage:
    - ExtractorComparisonResult: dataclass instantiation and all field access
    - run_extractor_comparison: TP/FP counting, detection_rate, fp_rate,
      zero-denominator safety, mismatched-length error
    - build_extractor_comparison_artifact: winner selection, honest_verdict branches
      (live_gpu_winner, simulated_no_verdict, insufficient_data), tiebreak logic,
      schema field, empty list, n_extractors count
    - Exp 367 main(): CARNOT_FORCE_LIVE=0 → blocked artifact; required fields present;
      load_gsm8k_questions synthetic fallback; _label_responses; detector_fn factories;
      _simulated_response helper

Spec: REQ-EXTRACT-023, SCENARIO-EXTRACT-047, SCENARIO-EXTRACT-048
"""

from __future__ import annotations

import json
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

from carnot.pipeline.extractor_comparison import (
    ExtractorComparisonResult,
    build_extractor_comparison_artifact,
    run_extractor_comparison,
)

# ---------------------------------------------------------------------------
# Helpers shared across test classes
# ---------------------------------------------------------------------------

_FAKE_QUESTIONS = [
    {"question": f"Q{i}", "response": f"R{i}"} for i in range(10)
]
# First 5 are wrong, last 5 are correct
_FAKE_GT = [True] * 5 + [False] * 5


def _always(q: str, r: str) -> bool:
    """Detector that always fires (all TP on wrong, all FP on correct)."""
    return True


def _never(q: str, r: str) -> bool:
    """Detector that never fires (0 TP, 0 FP)."""
    return False


def _first_five(q: str, r: str) -> bool:
    """Perfect detector for _FAKE_QUESTIONS: fires on Q0-Q4 (all wrong), not Q5-Q9."""
    n = int(q[1:])
    return n < 5


# ---------------------------------------------------------------------------
# ExtractorComparisonResult
# ---------------------------------------------------------------------------


class TestExtractorComparisonResult:
    """SCENARIO-EXTRACT-047: dataclass fields and instantiation."""

    def test_all_fields_accessible(self):
        result = ExtractorComparisonResult(
            extractor_name="arithmetic",
            n_questions=10,
            n_correct_questions=5,
            n_wrong_questions=5,
            n_true_positives=3,
            n_false_positives=1,
            detection_rate=0.6,
            fp_rate=0.2,
            inference_mode="simulated",
        )
        assert result.extractor_name == "arithmetic"
        assert result.n_questions == 10
        assert result.n_correct_questions == 5
        assert result.n_wrong_questions == 5
        assert result.n_true_positives == 3
        assert result.n_false_positives == 1
        assert result.detection_rate == 0.6
        assert result.fp_rate == 0.2
        assert result.inference_mode == "simulated"

    def test_live_gpu_mode_field(self):
        result = ExtractorComparisonResult(
            extractor_name="llm",
            n_questions=30,
            n_correct_questions=15,
            n_wrong_questions=15,
            n_true_positives=10,
            n_false_positives=0,
            detection_rate=0.666667,
            fp_rate=0.0,
            inference_mode="live_gpu",
        )
        assert result.inference_mode == "live_gpu"
        assert result.n_correct_questions + result.n_wrong_questions == result.n_questions

    def test_n_correct_plus_wrong_equals_total(self):
        """n_correct_questions + n_wrong_questions must equal n_questions (invariant)."""
        result = ExtractorComparisonResult(
            extractor_name="z3",
            n_questions=7,
            n_correct_questions=3,
            n_wrong_questions=4,
            n_true_positives=2,
            n_false_positives=0,
            detection_rate=0.5,
            fp_rate=0.0,
            inference_mode="live_gpu",
        )
        assert result.n_correct_questions + result.n_wrong_questions == 7


# ---------------------------------------------------------------------------
# run_extractor_comparison
# ---------------------------------------------------------------------------


class TestRunExtractorComparison:
    """SCENARIO-EXTRACT-047: benchmark run metrics."""

    def test_always_fires_all_tp_all_fp(self):
        result = run_extractor_comparison(
            extractor_name="always",
            detector_fn=_always,
            questions=_FAKE_QUESTIONS,
            ground_truth_wrong=_FAKE_GT,
            inference_mode="simulated",
        )
        assert result.n_questions == 10
        assert result.n_wrong_questions == 5
        assert result.n_correct_questions == 5
        assert result.n_true_positives == 5
        assert result.n_false_positives == 5
        assert result.detection_rate == 1.0
        assert result.fp_rate == 1.0
        assert result.extractor_name == "always"
        assert result.inference_mode == "simulated"

    def test_never_fires_zero_rates(self):
        result = run_extractor_comparison(
            extractor_name="never",
            detector_fn=_never,
            questions=_FAKE_QUESTIONS,
            ground_truth_wrong=_FAKE_GT,
            inference_mode="live_gpu",
        )
        assert result.n_true_positives == 0
        assert result.n_false_positives == 0
        assert result.detection_rate == 0.0
        assert result.fp_rate == 0.0
        assert result.inference_mode == "live_gpu"

    def test_perfect_detector(self):
        """Perfect detector: all wrong caught, no correct flagged."""
        result = run_extractor_comparison(
            extractor_name="perfect",
            detector_fn=_first_five,
            questions=_FAKE_QUESTIONS,
            ground_truth_wrong=_FAKE_GT,
            inference_mode="live_gpu",
        )
        assert result.n_true_positives == 5
        assert result.n_false_positives == 0
        assert result.detection_rate == 1.0
        assert result.fp_rate == 0.0

    def test_partial_detection(self):
        """Partial detection: 1 out of 2 wrong answers detected."""
        qs = [{"question": "Q0", "response": "R"}, {"question": "Q1", "response": "R"}]
        gt = [True, True]
        result = run_extractor_comparison("partial", lambda q, r: q == "Q0", qs, gt, "simulated")
        assert result.n_true_positives == 1
        assert result.detection_rate == pytest.approx(0.5)

    def test_zero_wrong_detection_rate_zero(self):
        """All answers correct → detection_rate denominator is 0 → 0.0."""
        qs = [{"question": "Q", "response": "R"}]
        gt = [False]
        result = run_extractor_comparison("x", _always, qs, gt, "simulated")
        assert result.detection_rate == 0.0
        assert result.fp_rate == 1.0

    def test_zero_correct_fp_rate_zero(self):
        """All answers wrong → fp_rate denominator is 0 → 0.0."""
        qs = [{"question": "Q", "response": "R"}]
        gt = [True]
        result = run_extractor_comparison("x", _always, qs, gt, "simulated")
        assert result.fp_rate == 0.0
        assert result.detection_rate == 1.0

    def test_mismatched_lengths_raises_valueerror(self):
        with pytest.raises(ValueError, match="must equal"):
            run_extractor_comparison("x", _never, _FAKE_QUESTIONS, [True], "simulated")

    def test_empty_inputs_zero_counts(self):
        result = run_extractor_comparison("x", _always, [], [], "simulated")
        assert result.n_questions == 0
        assert result.n_true_positives == 0
        assert result.n_false_positives == 0
        assert result.detection_rate == 0.0
        assert result.fp_rate == 0.0

    def test_rates_rounded_to_6_decimals(self):
        """Rates are rounded to 6 decimal places for stable JSON serialisation."""
        qs = [{"question": f"Q{i}", "response": "R"} for i in range(3)]
        gt = [True, True, True]
        result = run_extractor_comparison("x", lambda q, r: q == "Q0", qs, gt, "simulated")
        # 1/3 = 0.333333...  should be rounded
        assert isinstance(result.detection_rate, float)
        assert len(str(result.detection_rate).split(".")[-1]) <= 7  # at most 6 decimal places


# ---------------------------------------------------------------------------
# build_extractor_comparison_artifact
# ---------------------------------------------------------------------------


class TestBuildExtractorComparisonArtifact:
    """SCENARIO-EXTRACT-048: artifact schema, honest_verdict, winner selection."""

    def _make(
        self,
        name: str,
        detection: float,
        fp: float,
        mode: str,
        n_questions: int = 10,
    ) -> ExtractorComparisonResult:
        n_wrong = 5
        n_correct = n_questions - n_wrong
        tp = round(detection * n_wrong)
        fp_count = round(fp * n_correct)
        return ExtractorComparisonResult(
            extractor_name=name,
            n_questions=n_questions,
            n_correct_questions=n_correct,
            n_wrong_questions=n_wrong,
            n_true_positives=tp,
            n_false_positives=fp_count,
            detection_rate=detection,
            fp_rate=fp,
            inference_mode=mode,
        )

    def test_empty_results_insufficient_data(self):
        artifact = build_extractor_comparison_artifact([])
        assert artifact["winner_extractor"] is None
        assert artifact["honest_verdict"] == "insufficient_data"
        assert artifact["n_extractors"] == 0
        assert artifact["per_extractor_results"] == []

    def test_schema_field_correct(self):
        results = [self._make("arithmetic", 0.0, 0.0, "simulated")]
        artifact = build_extractor_comparison_artifact(results)
        assert artifact["schema"] == "carnot.extraction_comparison.v1"

    def test_all_live_gpu_honest_verdict_live_gpu_winner(self):
        """SCENARIO-EXTRACT-048: all live_gpu → honest_verdict = live_gpu_winner."""
        results = [
            self._make("arithmetic", 0.0, 0.0, "live_gpu"),
            self._make("llm", 0.6, 0.1, "live_gpu"),
            self._make("z3", 0.4, 0.05, "live_gpu"),
        ]
        artifact = build_extractor_comparison_artifact(results)
        assert artifact["honest_verdict"] == "live_gpu_winner"
        assert artifact["winner_extractor"] == "llm"

    def test_any_simulated_verdict_simulated_no_verdict(self):
        """SCENARIO-EXTRACT-048: any simulated → honest_verdict = simulated_no_verdict."""
        results = [
            self._make("arithmetic", 0.0, 0.0, "live_gpu"),
            self._make("llm", 0.8, 0.1, "simulated"),  # one simulated
            self._make("z3", 0.6, 0.0, "live_gpu"),
        ]
        artifact = build_extractor_comparison_artifact(results)
        assert artifact["honest_verdict"] == "simulated_no_verdict"

    def test_all_simulated_no_verdict(self):
        """All simulated → simulated_no_verdict regardless of detection rates."""
        results = [
            self._make("arithmetic", 0.0, 0.0, "simulated"),
            self._make("llm", 0.9, 0.1, "simulated"),
        ]
        artifact = build_extractor_comparison_artifact(results)
        assert artifact["honest_verdict"] == "simulated_no_verdict"
        assert artifact["honest_verdict"] != "live_gpu_winner"

    def test_winner_highest_detection_rate(self):
        results = [
            self._make("arithmetic", 0.1, 0.0, "live_gpu"),
            self._make("llm", 0.9, 0.0, "live_gpu"),
            self._make("z3", 0.5, 0.0, "live_gpu"),
        ]
        artifact = build_extractor_comparison_artifact(results)
        assert artifact["winner_extractor"] == "llm"

    def test_winner_tiebreak_lower_fp_rate(self):
        """Equal detection_rate: lower fp_rate wins."""
        results = [
            self._make("arithmetic", 0.5, 0.3, "live_gpu"),
            self._make("llm", 0.5, 0.1, "live_gpu"),
        ]
        artifact = build_extractor_comparison_artifact(results)
        assert artifact["winner_extractor"] == "llm"

    def test_winner_tiebreak_name_sort(self):
        """Equal detection_rate AND fp_rate: lexicographic name wins."""
        results = [
            self._make("z3", 0.5, 0.1, "live_gpu"),
            self._make("arithmetic", 0.5, 0.1, "live_gpu"),
        ]
        artifact = build_extractor_comparison_artifact(results)
        assert artifact["winner_extractor"] == "arithmetic"

    def test_n_extractors_count(self):
        results = [
            self._make("arithmetic", 0.0, 0.0, "simulated"),
            self._make("llm", 0.2, 0.0, "simulated"),
            self._make("z3", 0.1, 0.0, "simulated"),
        ]
        artifact = build_extractor_comparison_artifact(results)
        assert artifact["n_extractors"] == 3

    def test_per_extractor_results_fields(self):
        """Every per_extractor_results entry has all required keys."""
        results = [self._make("arithmetic", 0.0, 0.0, "live_gpu")]
        artifact = build_extractor_comparison_artifact(results)
        entry = artifact["per_extractor_results"][0]
        required_keys = [
            "extractor_name", "n_questions", "n_correct_questions", "n_wrong_questions",
            "n_true_positives", "n_false_positives", "detection_rate", "fp_rate",
            "inference_mode",
        ]
        for key in required_keys:
            assert key in entry, f"Missing key: {key}"

    def test_single_result_live_gpu_winner(self):
        """Single live_gpu result → live_gpu_winner (sufficient data)."""
        results = [self._make("llm", 0.5, 0.0, "live_gpu")]
        artifact = build_extractor_comparison_artifact(results)
        assert artifact["honest_verdict"] == "live_gpu_winner"

    def test_live_gpu_zero_detection_still_winner(self):
        """live_gpu_winner is granted even with 0.0 detection rate (best among live)."""
        results = [
            self._make("arithmetic", 0.0, 0.0, "live_gpu"),
            self._make("llm", 0.0, 0.0, "live_gpu"),
        ]
        artifact = build_extractor_comparison_artifact(results)
        assert artifact["honest_verdict"] == "live_gpu_winner"


# ---------------------------------------------------------------------------
# Exp 367 main() — simulated/blocked mode tests
# ---------------------------------------------------------------------------


class TestExp367Main:
    """Tests for scripts/experiment_367_extraction_live.py."""

    def _import_exp367(self):
        import importlib
        import scripts.experiment_367_extraction_live as mod
        importlib.reload(mod)
        return mod

    def test_main_blocked_when_force_live_not_set(self, tmp_path, monkeypatch):
        """main() writes blocked artifact when CARNOT_FORCE_LIVE is not 1."""
        monkeypatch.setenv("CARNOT_FORCE_LIVE", "0")
        monkeypatch.setenv("JAX_PLATFORMS", "cpu")

        import scripts.experiment_367_extraction_live as exp367
        monkeypatch.setattr(exp367, "_REPO_ROOT", tmp_path)

        exp367.main()

        artifact_path = tmp_path / "results" / "experiment_367_extraction_live.json"
        assert artifact_path.exists(), "Blocked artifact file not written"
        artifact = json.loads(artifact_path.read_text())
        assert artifact["status"] == "blocked"
        assert "blocked" in artifact["honest_verdict"]

    def test_main_blocked_artifact_has_required_fields(self, tmp_path, monkeypatch):
        """Blocked artifact must contain all REQUIRED_RESULT_FIELDS."""
        from scripts.experiment_template import REQUIRED_RESULT_FIELDS

        monkeypatch.setenv("CARNOT_FORCE_LIVE", "0")
        import scripts.experiment_367_extraction_live as exp367
        monkeypatch.setattr(exp367, "_REPO_ROOT", tmp_path)

        exp367.main()

        artifact_path = tmp_path / "results" / "experiment_367_extraction_live.json"
        artifact = json.loads(artifact_path.read_text())
        for field in REQUIRED_RESULT_FIELDS:
            assert field in artifact, f"Missing required field: {field}"

    def test_main_blocked_experiment_id_367(self, tmp_path, monkeypatch):
        """Blocked artifact must have experiment=367."""
        monkeypatch.setenv("CARNOT_FORCE_LIVE", "0")
        import scripts.experiment_367_extraction_live as exp367
        monkeypatch.setattr(exp367, "_REPO_ROOT", tmp_path)

        exp367.main()

        artifact_path = tmp_path / "results" / "experiment_367_extraction_live.json"
        artifact = json.loads(artifact_path.read_text())
        assert artifact["experiment"] == 367

    def test_main_gpu_setup_fails_writes_blocked(self, tmp_path, monkeypatch):
        """When setup_gpu() raises RuntimeError, write blocked artifact."""
        monkeypatch.setenv("CARNOT_FORCE_LIVE", "1")
        import scripts.experiment_367_extraction_live as exp367
        monkeypatch.setattr(exp367, "_REPO_ROOT", tmp_path)

        # Patch ExperimentTemplate.setup_gpu to raise
        with patch("scripts.experiment_367_extraction_live.ExperimentTemplate.setup_gpu",
                   side_effect=RuntimeError("GPU unavailable")):
            exp367.main()

        artifact_path = tmp_path / "results" / "experiment_367_extraction_live.json"
        artifact = json.loads(artifact_path.read_text())
        assert artifact["status"] == "blocked"
        assert "blocked" in artifact.get("honest_verdict", "")

    def test_main_gpu_unhealthy_writes_blocked(self, tmp_path, monkeypatch):
        """When setup_gpu() returns all_healthy=False, write blocked artifact."""
        monkeypatch.setenv("CARNOT_FORCE_LIVE", "1")
        import scripts.experiment_367_extraction_live as exp367
        monkeypatch.setattr(exp367, "_REPO_ROOT", tmp_path)

        with patch("scripts.experiment_367_extraction_live.ExperimentTemplate.setup_gpu",
                   return_value={"all_healthy": False, "models": []}):
            exp367.main()

        artifact_path = tmp_path / "results" / "experiment_367_extraction_live.json"
        artifact = json.loads(artifact_path.read_text())
        assert artifact["status"] == "blocked"

    def test_main_model_load_failure_writes_blocked(self, tmp_path, monkeypatch):
        """When model load fails, write blocked artifact and stop."""
        monkeypatch.setenv("CARNOT_FORCE_LIVE", "1")
        import scripts.experiment_367_extraction_live as exp367
        monkeypatch.setattr(exp367, "_REPO_ROOT", tmp_path)

        with patch("scripts.experiment_367_extraction_live.ExperimentTemplate.setup_gpu",
                   return_value={"all_healthy": True, "models": []}):
            # Patch carnot.inference.model_loader.load_model to raise
            with patch.dict("sys.modules", {"carnot.inference.model_loader": MagicMock(
                load_model=MagicMock(side_effect=RuntimeError("no model"))
            )}):
                exp367.main()

        artifact_path = tmp_path / "results" / "experiment_367_extraction_live.json"
        artifact = json.loads(artifact_path.read_text())
        assert artifact["status"] == "blocked"
        assert "model_load_failed" in artifact.get("honest_verdict", "") or \
               "blocked" in artifact.get("honest_verdict", "")

    def test_load_gsm8k_synthetic_fallback(self, monkeypatch):
        """load_gsm8k_questions returns synthetic data when datasets unavailable."""
        import scripts.experiment_367_extraction_live as exp367

        with patch.dict("sys.modules", {"datasets": None}):
            qs = exp367.load_gsm8k_questions(5)
        assert len(qs) == 5
        for q in qs:
            assert "question" in q
            assert "answer" in q

    def test_label_responses_correct(self):
        """_label_responses returns False for a correct answer."""
        import scripts.experiment_367_extraction_live as exp367
        questions = [{"question": "Q", "answer": "10", "response": "The answer is 10."}]
        labels = exp367._label_responses(questions)
        assert labels[0] is False

    def test_label_responses_wrong(self):
        """_label_responses returns True for a wrong answer."""
        import scripts.experiment_367_extraction_live as exp367
        questions = [{"question": "Q", "answer": "10", "response": "The answer is 9."}]
        labels = exp367._label_responses(questions)
        assert labels[0] is True

    def test_label_responses_no_number(self):
        """No number in response → conservatively labelled wrong."""
        import scripts.experiment_367_extraction_live as exp367
        questions = [{"question": "Q", "answer": "10", "response": "I don't know"}]
        labels = exp367._label_responses(questions)
        assert labels[0] is True

    def test_label_responses_non_numeric_answer(self):
        """Non-numeric ground truth → conservatively labelled wrong."""
        import scripts.experiment_367_extraction_live as exp367
        questions = [{"question": "Q", "answer": "abc", "response": "The answer is 5."}]
        labels = exp367._label_responses(questions)
        assert labels[0] is True

    def test_make_arithmetic_detector_returns_bool(self):
        """_make_arithmetic_detector returns a callable that returns bool."""
        import scripts.experiment_367_extraction_live as exp367
        fn = exp367._make_arithmetic_detector()
        result = fn("Q", "2 + 2 = 5")
        assert isinstance(result, bool)

    def test_make_llm_detector_stub_returns_false(self):
        """_make_llm_detector with stub extractor returns False for empty extract()."""
        import scripts.experiment_367_extraction_live as exp367
        stub = MagicMock()
        stub.extract.return_value = []
        fn = exp367._make_llm_detector(stub)
        assert fn("Q", "no arithmetic") is False

    def test_make_llm_detector_none_stub(self):
        """_make_llm_detector(None) creates a CI stub that returns False."""
        import scripts.experiment_367_extraction_live as exp367
        fn = exp367._make_llm_detector(None)
        assert isinstance(fn("Q", "response"), bool)

    def test_make_z3_detector_ci_stub_returns_bool(self):
        """_make_z3_detector with CI stub (llm_caller=None) returns bool."""
        import scripts.experiment_367_extraction_live as exp367
        fn = exp367._make_z3_detector(llm_caller=None)
        result = fn("Q", "some response")
        assert isinstance(result, bool)

    def test_simulated_response_contains_number(self):
        """_simulated_response echoes the first number from the question."""
        import scripts.experiment_367_extraction_live as exp367
        q = {"question": "Alice has 7 apples."}
        resp = exp367._simulated_response(q)
        assert "7" in resp

    def test_simulated_response_no_number_fallback(self):
        """_simulated_response returns fallback string when no number in question."""
        import scripts.experiment_367_extraction_live as exp367
        q = {"question": "What is the meaning of life?"}
        resp = exp367._simulated_response(q)
        assert isinstance(resp, str)
        assert len(resp) > 0

    def test_synthetic_gsm8k_generates_n_questions(self):
        """_synthetic_gsm8k generates exactly n deterministic questions."""
        import scripts.experiment_367_extraction_live as exp367
        qs = exp367._synthetic_gsm8k(8)
        assert len(qs) == 8
        for q in qs:
            assert "question" in q
            assert "answer" in q
            # Verify answer is numeric
            assert int(q["answer"]) >= 0
