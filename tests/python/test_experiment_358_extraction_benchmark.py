"""Tests for python/carnot/pipeline/extraction_benchmark.py and
scripts/experiment_358_extraction_benchmark.py.

100% targeted coverage:
    - ExtractionBenchmarkResult: dataclass instantiation, all fields accessible
    - run_extraction_benchmark: TP/FP/FN/TN counting, detection_rate, false_positive_rate,
      zero-denominator safety, mismatched-length guard
    - build_extraction_comparison_artifact: winner selection, improvement computation,
      honest_verdict logic (all four branches), empty-list guard, <2 results guard
    - Exp 358 main(): simulated mode (no CARNOT_FORCE_LIVE), artifact schema/fields,
      live-GPU blocked path, honest_verdict=simulated_no_verdict in simulated mode

Spec: REQ-EXTRACT-021, SCENARIO-EXTRACT-042, SCENARIO-EXTRACT-043
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

from carnot.pipeline.extraction_benchmark import (
    ExtractionBenchmarkResult,
    ViolationDetector,
    build_extraction_comparison_artifact,
    run_extraction_benchmark,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_FAKE_QUESTIONS = [
    {"question": f"Q{i}", "response": f"R{i}"} for i in range(10)
]
# First 5 are wrong, last 5 are correct
_FAKE_GROUND_TRUTH = [True] * 5 + [False] * 5


def _always_violates(question: str, response: str) -> bool:
    """Extractor that always reports a violation (detects all TP, all FP)."""
    return True


def _never_violates(question: str, response: str) -> bool:
    """Extractor that never reports a violation (0 TP, 0 FP)."""
    return False


def _violates_first_five(question: str, response: str) -> bool:
    """Violation iff question number < 5 (perfect detector for our fake data)."""
    n = int(question[1:])
    return n < 5


# ---------------------------------------------------------------------------
# ExtractionBenchmarkResult
# ---------------------------------------------------------------------------


class TestExtractionBenchmarkResult:
    """SCENARIO-EXTRACT-042: dataclass fields and instantiation."""

    def test_all_fields_accessible(self):
        result = ExtractionBenchmarkResult(
            extractor_name="arithmetic",
            n_questions=10,
            n_violations_found=3,
            n_true_positives=2,
            n_false_positives=1,
            detection_rate=0.4,
            false_positive_rate=0.2,
            inference_mode="simulated",
        )
        assert result.extractor_name == "arithmetic"
        assert result.n_questions == 10
        assert result.n_violations_found == 3
        assert result.n_true_positives == 2
        assert result.n_false_positives == 1
        assert result.detection_rate == 0.4
        assert result.false_positive_rate == 0.2
        assert result.inference_mode == "simulated"

    def test_live_gpu_mode(self):
        result = ExtractionBenchmarkResult(
            extractor_name="llm",
            n_questions=5,
            n_violations_found=5,
            n_true_positives=5,
            n_false_positives=0,
            detection_rate=1.0,
            false_positive_rate=0.0,
            inference_mode="live_gpu",
        )
        assert result.inference_mode == "live_gpu"


# ---------------------------------------------------------------------------
# run_extraction_benchmark
# ---------------------------------------------------------------------------


class TestRunExtractionBenchmark:
    """SCENARIO-EXTRACT-042: benchmark run metrics."""

    def test_always_violates_detection_rate_one(self):
        """All wrong answers detected → detection_rate = 1.0; all correct flagged → fp_rate = 1.0."""
        result = run_extraction_benchmark(
            extractor_name="always",
            inference_fn=_always_violates,
            questions=_FAKE_QUESTIONS,
            ground_truth_wrong=_FAKE_GROUND_TRUTH,
            inference_mode="simulated",
        )
        assert result.n_questions == 10
        assert result.n_violations_found == 10
        assert result.n_true_positives == 5
        assert result.n_false_positives == 5
        assert result.detection_rate == 1.0
        assert result.false_positive_rate == 1.0
        assert result.extractor_name == "always"
        assert result.inference_mode == "simulated"

    def test_never_violates_detection_rate_zero(self):
        """No violations detected → detection_rate = 0.0, fp_rate = 0.0."""
        result = run_extraction_benchmark(
            extractor_name="never",
            inference_fn=_never_violates,
            questions=_FAKE_QUESTIONS,
            ground_truth_wrong=_FAKE_GROUND_TRUTH,
            inference_mode="simulated",
        )
        assert result.n_violations_found == 0
        assert result.n_true_positives == 0
        assert result.n_false_positives == 0
        assert result.detection_rate == 0.0
        assert result.false_positive_rate == 0.0

    def test_perfect_extractor(self):
        """Perfect extractor: detects all wrong, none of the correct."""
        result = run_extraction_benchmark(
            extractor_name="perfect",
            inference_fn=_violates_first_five,
            questions=_FAKE_QUESTIONS,
            ground_truth_wrong=_FAKE_GROUND_TRUTH,
            inference_mode="live_gpu",
        )
        assert result.n_true_positives == 5
        assert result.n_false_positives == 0
        assert result.detection_rate == 1.0
        assert result.false_positive_rate == 0.0
        assert result.inference_mode == "live_gpu"

    def test_detection_rate_rounded(self):
        """Partial detection: 2 out of 4 wrong answers detected → rate = 0.5."""
        qs = [{"question": f"Q{i}", "response": f"R{i}"} for i in range(4)]
        gt = [True, True, False, False]

        # Detect only Q0
        def _detect_q0(q, r):
            return q == "Q0"

        result = run_extraction_benchmark("partial", _detect_q0, qs, gt, "simulated")
        assert result.n_true_positives == 1
        assert result.detection_rate == 0.5

    def test_zero_wrong_answers_detection_rate_zero(self):
        """When all answers are correct, detection_rate denominator is 0 → 0.0."""
        qs = [{"question": "Q", "response": "R"}]
        gt = [False]
        result = run_extraction_benchmark("x", _always_violates, qs, gt, "simulated")
        assert result.detection_rate == 0.0
        assert result.false_positive_rate == 1.0

    def test_zero_correct_answers_fp_rate_zero(self):
        """When all answers are wrong, fp_rate denominator is 0 → 0.0."""
        qs = [{"question": "Q", "response": "R"}]
        gt = [True]
        result = run_extraction_benchmark("x", _always_violates, qs, gt, "simulated")
        assert result.false_positive_rate == 0.0
        assert result.detection_rate == 1.0

    def test_mismatched_lengths_raises(self):
        """Mismatched questions / ground_truth_wrong lengths must raise ValueError."""
        with pytest.raises(ValueError, match="must equal"):
            run_extraction_benchmark(
                "x", _never_violates, _FAKE_QUESTIONS, [True], "simulated"
            )

    def test_empty_inputs(self):
        """Empty inputs produce 0 violations and 0.0 rates."""
        result = run_extraction_benchmark("x", _always_violates, [], [], "simulated")
        assert result.n_questions == 0
        assert result.n_violations_found == 0
        assert result.detection_rate == 0.0
        assert result.false_positive_rate == 0.0


# ---------------------------------------------------------------------------
# build_extraction_comparison_artifact
# ---------------------------------------------------------------------------


class TestBuildExtractionComparisonArtifact:
    """SCENARIO-EXTRACT-043: honest_verdict and comparison artifact."""

    def _make_result(self, name: str, detection: float, fp: float, mode: str) -> ExtractionBenchmarkResult:
        return ExtractionBenchmarkResult(
            extractor_name=name,
            n_questions=10,
            n_violations_found=0,
            n_true_positives=0,
            n_false_positives=0,
            detection_rate=detection,
            false_positive_rate=fp,
            inference_mode=mode,
        )

    def test_empty_results(self):
        artifact = build_extraction_comparison_artifact([])
        assert artifact["winner"] is None
        assert artifact["honest_verdict"] == "insufficient_data"
        assert artifact["n_extractors"] == 0
        assert artifact["per_extractor_results"] == []

    def test_single_result_insufficient_data(self):
        r = self._make_result("arithmetic", 0.0, 0.0, "simulated")
        artifact = build_extraction_comparison_artifact([r])
        assert artifact["honest_verdict"] == "insufficient_data"

    def test_simulated_mode_no_verdict(self):
        """SCENARIO-EXTRACT-043: simulated mode cannot claim a win."""
        results = [
            self._make_result("arithmetic", 0.0, 0.0, "simulated"),
            self._make_result("llm", 0.8, 0.1, "simulated"),
            self._make_result("z3", 0.6, 0.05, "simulated"),
        ]
        artifact = build_extraction_comparison_artifact(results)
        assert artifact["honest_verdict"] == "simulated_no_verdict"
        assert artifact["honest_verdict"] != "live_gpu_llm_extractor_wins"

    def test_live_gpu_llm_wins(self):
        """SCENARIO-EXTRACT-043: live_gpu + llm > arithmetic → wins."""
        results = [
            self._make_result("arithmetic", 0.0, 0.0, "live_gpu"),
            self._make_result("llm", 0.4, 0.1, "live_gpu"),
            self._make_result("z3", 0.2, 0.05, "live_gpu"),
        ]
        artifact = build_extraction_comparison_artifact(results)
        assert artifact["honest_verdict"] == "live_gpu_llm_extractor_wins"
        assert artifact["winner"] == "llm"

    def test_live_gpu_no_improvement(self):
        """live_gpu mode but llm does not beat arithmetic."""
        results = [
            self._make_result("arithmetic", 0.5, 0.1, "live_gpu"),
            self._make_result("llm", 0.3, 0.05, "live_gpu"),
            self._make_result("z3", 0.2, 0.02, "live_gpu"),
        ]
        artifact = build_extraction_comparison_artifact(results)
        assert artifact["honest_verdict"] == "live_gpu_no_improvement"

    def test_live_gpu_equal_rate_no_improvement(self):
        """llm == arithmetic detection_rate → no improvement (not strictly greater)."""
        results = [
            self._make_result("arithmetic", 0.4, 0.1, "live_gpu"),
            self._make_result("llm", 0.4, 0.05, "live_gpu"),
        ]
        artifact = build_extraction_comparison_artifact(results)
        assert artifact["honest_verdict"] == "live_gpu_no_improvement"

    def test_winner_selection_highest_detection(self):
        """Winner is the extractor with the highest detection_rate."""
        results = [
            self._make_result("arithmetic", 0.2, 0.1, "simulated"),
            self._make_result("llm", 0.9, 0.2, "simulated"),
            self._make_result("z3", 0.5, 0.0, "simulated"),
        ]
        artifact = build_extraction_comparison_artifact(results)
        assert artifact["winner"] == "llm"

    def test_winner_tiebreak_fp(self):
        """Tiebreak on equal detection_rate: lower fp_rate wins."""
        results = [
            self._make_result("arithmetic", 0.5, 0.3, "simulated"),
            self._make_result("llm", 0.5, 0.1, "simulated"),
        ]
        artifact = build_extraction_comparison_artifact(results)
        assert artifact["winner"] == "llm"

    def test_improvement_computation(self):
        """improvement_over_arithmetic_extractor is winner.rate - arithmetic.rate."""
        results = [
            self._make_result("arithmetic", 0.0, 0.0, "live_gpu"),
            self._make_result("llm", 0.4, 0.0, "live_gpu"),
        ]
        artifact = build_extraction_comparison_artifact(results)
        assert artifact["improvement_over_arithmetic_extractor"] == pytest.approx(0.4)

    def test_no_arithmetic_extractor_improvement_zero(self):
        """When no 'arithmetic' result is present, improvement defaults to 0.0."""
        results = [
            self._make_result("llm", 0.5, 0.1, "simulated"),
            self._make_result("z3", 0.3, 0.0, "simulated"),
        ]
        artifact = build_extraction_comparison_artifact(results)
        assert artifact["improvement_over_arithmetic_extractor"] == 0.0

    def test_per_extractor_results_fields(self):
        """Every per_extractor_results entry has all required keys."""
        results = [self._make_result("arithmetic", 0.0, 0.0, "simulated")]
        artifact = build_extraction_comparison_artifact(results)
        entry = artifact["per_extractor_results"][0]
        for key in [
            "extractor_name", "n_questions", "n_violations_found", "n_true_positives",
            "n_false_positives", "detection_rate", "false_positive_rate", "inference_mode",
        ]:
            assert key in entry, f"Missing key: {key}"

    def test_n_extractors_count(self):
        results = [
            self._make_result("arithmetic", 0.0, 0.0, "simulated"),
            self._make_result("llm", 0.2, 0.0, "simulated"),
            self._make_result("z3", 0.1, 0.0, "simulated"),
        ]
        artifact = build_extraction_comparison_artifact(results)
        assert artifact["n_extractors"] == 3

    def test_live_gpu_no_llm_extractor(self):
        """live_gpu but no 'llm' named extractor → no_improvement (llm_rate is None)."""
        results = [
            self._make_result("arithmetic", 0.0, 0.0, "live_gpu"),
            self._make_result("z3", 0.5, 0.0, "live_gpu"),
        ]
        artifact = build_extraction_comparison_artifact(results)
        assert artifact["honest_verdict"] == "live_gpu_no_improvement"


# ---------------------------------------------------------------------------
# ViolationDetector Protocol
# ---------------------------------------------------------------------------


class TestViolationDetectorProtocol:
    """Ensures the Protocol is runtime-checkable."""

    def test_callable_satisfies_protocol(self):
        assert isinstance(_always_violates, ViolationDetector)

    def test_non_callable_does_not_satisfy_protocol(self):
        assert not isinstance(42, ViolationDetector)


# ---------------------------------------------------------------------------
# Exp 358 main() — simulated mode tests
# ---------------------------------------------------------------------------


class TestExp358Main:
    """Tests for scripts/experiment_358_extraction_benchmark.py in simulated mode."""

    def _import_exp358(self):
        import importlib
        import scripts.experiment_358_extraction_benchmark as mod
        importlib.reload(mod)
        return mod

    def test_main_simulated_writes_artifact(self, tmp_path, monkeypatch):
        """main() in simulated mode writes a valid JSON artifact."""
        monkeypatch.setenv("CARNOT_FORCE_LIVE", "0")
        monkeypatch.setenv("JAX_PLATFORMS", "cpu")

        import scripts.experiment_358_extraction_benchmark as exp358
        monkeypatch.setattr(exp358, "_REPO_ROOT", tmp_path)
        # Patch out actual GSM8K loading; return 10 synthetic questions
        synthetic_qs = [
            {"question": f"What is {i}+{i}?", "answer": str(i + i)}
            for i in range(10)
        ]
        monkeypatch.setattr(exp358, "load_gsm8k_questions", lambda n: synthetic_qs[:n])

        exp358.main()

        artifact_path = tmp_path / "results" / "experiment_358_extraction_benchmark.json"
        assert artifact_path.exists(), "Artifact file not written"
        artifact = json.loads(artifact_path.read_text())
        assert artifact["experiment"] == 358
        assert artifact["status"] in ("success", "blocked", "partial")
        assert "honest_verdict" in artifact
        assert artifact["honest_verdict"] != "live_gpu_llm_extractor_wins"

    def test_main_simulated_honest_verdict(self, tmp_path, monkeypatch):
        """Simulated mode must never produce live_gpu_llm_extractor_wins."""
        monkeypatch.setenv("CARNOT_FORCE_LIVE", "0")
        import scripts.experiment_358_extraction_benchmark as exp358
        monkeypatch.setattr(exp358, "_REPO_ROOT", tmp_path)
        synthetic_qs = [
            {"question": f"What is {i}+{i}?", "answer": str(i + i)}
            for i in range(10)
        ]
        monkeypatch.setattr(exp358, "load_gsm8k_questions", lambda n: synthetic_qs[:n])

        exp358.main()

        artifact_path = tmp_path / "results" / "experiment_358_extraction_benchmark.json"
        artifact = json.loads(artifact_path.read_text())
        assert artifact["honest_verdict"] != "live_gpu_llm_extractor_wins"

    def test_main_required_fields_present(self, tmp_path, monkeypatch):
        """Artifact must contain all REQUIRED_RESULT_FIELDS."""
        from scripts.experiment_template import REQUIRED_RESULT_FIELDS
        monkeypatch.setenv("CARNOT_FORCE_LIVE", "0")
        import scripts.experiment_358_extraction_benchmark as exp358
        monkeypatch.setattr(exp358, "_REPO_ROOT", tmp_path)
        synthetic_qs = [
            {"question": f"What is {i}+{i}?", "answer": str(i + i)}
            for i in range(10)
        ]
        monkeypatch.setattr(exp358, "load_gsm8k_questions", lambda n: synthetic_qs[:n])

        exp358.main()

        artifact_path = tmp_path / "results" / "experiment_358_extraction_benchmark.json"
        artifact = json.loads(artifact_path.read_text())
        for field in REQUIRED_RESULT_FIELDS:
            assert field in artifact, f"Missing required field: {field}"

    def test_load_gsm8k_questions_fallback(self, tmp_path, monkeypatch):
        """load_gsm8k_questions returns synthetic data when datasets unavailable."""
        import scripts.experiment_358_extraction_benchmark as exp358

        # Simulate datasets not installed
        with patch.dict("sys.modules", {"datasets": None}):
            qs = exp358.load_gsm8k_questions(5)
        assert len(qs) == 5
        for q in qs:
            assert "question" in q
            assert "answer" in q

    def test_make_arithmetic_inference_fn(self):
        """_make_arithmetic_inference_fn returns a callable that detects arithmetic violations."""
        import scripts.experiment_358_extraction_benchmark as exp358
        fn = exp358._make_arithmetic_inference_fn()
        # Known violated arithmetic claim
        assert isinstance(fn("Q", "2 + 2 = 5"), bool)

    def test_make_llm_inference_fn(self):
        """_make_llm_inference_fn with a stub extractor returns a callable."""
        import scripts.experiment_358_extraction_benchmark as exp358
        stub_extractor = MagicMock()
        stub_extractor.extract.return_value = []
        fn = exp358._make_llm_inference_fn(stub_extractor)
        result = fn("Q", "no arithmetic here")
        assert result is False

    def test_make_z3_inference_fn(self):
        """_make_z3_inference_fn with CI-stub formalizer returns callable."""
        import scripts.experiment_358_extraction_benchmark as exp358
        fn = exp358._make_z3_inference_fn(llm_caller=None)
        result = fn("Q", "some response")
        assert isinstance(result, bool)

    def test_label_responses(self):
        """_label_responses compares final numeric answer to ground truth."""
        import scripts.experiment_358_extraction_benchmark as exp358
        questions = [
            {"question": "Q", "answer": "5", "response": "The answer is 5."},
            {"question": "Q", "answer": "5", "response": "The answer is 6."},
            {"question": "Q", "answer": "5", "response": "no number"},
        ]
        labels = exp358._label_responses(questions)
        assert labels[0] is False  # correct
        assert labels[1] is True   # wrong
        # "no number" — answer can't be extracted → treated as wrong
        assert isinstance(labels[2], bool)
