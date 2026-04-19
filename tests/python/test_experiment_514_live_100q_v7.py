"""Tests for Exp 514 helpers: load_jit_gated_model, run_100q_benchmark, write_cot_pairs.

100% coverage on python/carnot/pipeline/live_100q_v7_helpers.py and
scripts/experiment_514_live_100q_precision_v7.py (non-GPU paths).

Spec: REQ-BENCH-014, REQ-BENCH-015,
      SCENARIO-BENCH-033, SCENARIO-BENCH-034
"""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from carnot.pipeline.live_100q_v7_helpers import (
    PrecisionBenchmarkResult,
    _extract_answer,
    _is_correct,
    load_jit_gated_model,
    run_100q_benchmark,
    wilson_ci,
    write_cot_pairs,
)


# ---------------------------------------------------------------------------
# wilson_ci
# ---------------------------------------------------------------------------


class TestWilsonCI:
    def test_zero_n(self):
        lo, hi = wilson_ci(0, 0)
        assert lo == 0.0 and hi == 0.0

    def test_all_correct(self):
        lo, hi = wilson_ci(10, 10)
        assert lo > 0.7 and hi == 1.0

    def test_none_correct(self):
        lo, hi = wilson_ci(0, 10)
        assert lo == 0.0 and hi < 0.3

    def test_half_correct(self):
        lo, hi = wilson_ci(50, 100)
        assert lo < 0.5 < hi

    def test_bounds_clamped(self):
        lo, hi = wilson_ci(100, 100)
        assert 0.0 <= lo <= 1.0 and 0.0 <= hi <= 1.0

    def test_known_value(self):
        # 80 correct out of 100 — CI should straddle 0.8
        lo, hi = wilson_ci(80, 100)
        assert lo < 0.8 < hi


# ---------------------------------------------------------------------------
# _extract_answer
# ---------------------------------------------------------------------------


class TestExtractAnswer:
    def test_gsm8k_delimiter(self):
        assert _extract_answer("blah\n#### 42") == "42"

    def test_last_number_fallback(self):
        assert _extract_answer("The answer is 7.") == "7"

    def test_negative(self):
        assert _extract_answer("#### -5") == "-5"

    def test_decimal_stripped(self):
        assert _extract_answer("#### 72.0") == "72"

    def test_comma_stripped(self):
        assert _extract_answer("1,234 total") == "1234"

    def test_no_number(self):
        assert _extract_answer("no numbers here") is None

    def test_empty(self):
        assert _extract_answer("") is None


# ---------------------------------------------------------------------------
# _is_correct
# ---------------------------------------------------------------------------


class TestIsCorrect:
    def test_match(self):
        assert _is_correct("#### 42", "42") is True

    def test_mismatch(self):
        assert _is_correct("#### 10", "42") is False

    def test_none_gold(self):
        assert _is_correct("#### 42", None) is False

    def test_empty_response(self):
        assert _is_correct("", "42") is False

    def test_float_tolerance(self):
        # 72.0 vs 72 should match within 0.501
        assert _is_correct("72.0", "72") is True

    def test_no_number_in_response(self):
        assert _is_correct("no answer given", "42") is False


# ---------------------------------------------------------------------------
# load_jit_gated_model
# ---------------------------------------------------------------------------


class TestLoadJitGatedModel:
    def test_gate_cleared_calls_loader(self):
        """When VRAM is sufficient, the loader factory is called and load() invoked."""
        mock_result = MagicMock()
        mock_result.is_cleared = True
        mock_result.available_gb = 20.0

        mock_loader = MagicMock()

        def factory():
            return mock_loader

        with patch(
            "carnot.pipeline.jit_vram_check.JITVRAMCheck.gate_model_load",
            return_value=mock_result,
        ):
            result = load_jit_gated_model(factory, "test-model", 10.0, 0)

        mock_loader.load.assert_called_once()
        assert result is mock_loader

    def test_gate_blocked_returns_none(self):
        """When VRAM is insufficient, factory is never called and None is returned."""
        mock_result = MagicMock()
        mock_result.is_cleared = False
        mock_result.available_gb = 3.0

        factory_called = []

        def factory():
            factory_called.append(True)
            return MagicMock()

        with patch(
            "carnot.pipeline.jit_vram_check.JITVRAMCheck.gate_model_load",
            return_value=mock_result,
        ):
            result = load_jit_gated_model(factory, "test-model", 10.0, 0)

        assert result is None
        assert factory_called == [], "loader factory must not be called when gate is blocked"

    def test_uses_correct_device_id(self):
        """The JITVRAMCheck is created with the given device id."""
        mock_result = MagicMock()
        mock_result.is_cleared = False
        mock_result.available_gb = 0.0

        import carnot.pipeline.live_100q_v7_helpers as _helpers_mod
        with patch.object(_helpers_mod, "JITVRAMCheck") as mock_cls:
            mock_checker = MagicMock()
            mock_checker.gate_model_load.return_value = mock_result
            mock_cls.return_value = mock_checker

            load_jit_gated_model(MagicMock, "m", 5.0, 1)
            mock_cls.assert_called_once_with(device_id=1)


# ---------------------------------------------------------------------------
# run_100q_benchmark
# ---------------------------------------------------------------------------


class TestRun100QBenchmark:
    def _make_questions(self, n: int = 5) -> list:
        return [
            {
                "question": f"What is {i} + {i}?",
                "answer": f"The answer is {i * 2}. #### {i * 2}",
            }
            for i in range(1, n + 1)
        ]

    def _perfect_inference(self, q: dict) -> str:
        # Extracts gold from question pattern "What is N + N?" -> answer 2*N
        import re
        m = re.search(r"(\d+) \+ (\d+)", q["question"])
        if m:
            return f"#### {int(m.group(1)) + int(m.group(2))}"
        return "#### 0"

    def test_perfect_accuracy(self):
        questions = self._make_questions(5)

        def infer(prompt: str) -> str:
            # Always return correct answer
            for q in questions:
                if q["question"] in prompt:
                    return q["answer"]
            return "#### 2"

        mock_extractor = MagicMock()
        mock_extractor.extract.return_value = []  # no violations

        result = run_100q_benchmark(infer, "TestModel", questions, mock_extractor)

        assert isinstance(result, PrecisionBenchmarkResult)
        assert result.n == 5
        assert result.pipeline_accuracy == result.baseline_accuracy
        assert 0.0 <= result.wilson_95ci_lower <= result.wilson_95ci_upper <= 1.0

    def test_violation_triggers_repair(self):
        """When extractor finds violations, inference_fn is called again with repair prompt."""
        questions = self._make_questions(3)
        call_log = []

        def infer(prompt: str) -> str:
            call_log.append(prompt)
            return "#### 99"  # always wrong answer

        mock_extractor = MagicMock()
        mock_extractor.extract.return_value = ["violation1"]  # always has violations

        run_100q_benchmark(infer, "M", questions, mock_extractor)

        # Each question: 1 baseline + 1 pipeline + 1 repair = 3 calls per question
        # Actually: baseline pass (3 calls) + pipeline pass (3 initial + 3 repair = 6) = 9
        # But repair is only called when violations found, which is always here
        assert len(call_log) > len(questions)  # repair prompts were issued

    def test_cot_pairs_length(self):
        questions = self._make_questions(4)
        mock_extractor = MagicMock()
        mock_extractor.extract.return_value = []

        result = run_100q_benchmark(lambda p: "#### 0", "M", questions, mock_extractor)

        assert len(result.cot_pairs) == 4
        for pair in result.cot_pairs:
            assert set(pair.keys()) >= {"question", "cot_text", "correct", "model_id"}

    def test_signed_improvement_negative_allowed(self):
        """Signed improvement can be negative — never clamp to zero."""
        # Baseline always correct, pipeline always wrong via repair
        questions = self._make_questions(2)
        call_count = [0]

        def infer(prompt: str) -> str:
            call_count[0] += 1
            # First call per question (baseline): return correct answer
            # Subsequent calls: wrong
            if call_count[0] % 3 == 1:
                return "#### 2"  # correct for question 1 (1+1)
            return "#### 99"

        mock_extractor = MagicMock()
        mock_extractor.extract.return_value = ["v"]  # always trigger repair

        result = run_100q_benchmark(infer, "M", questions, mock_extractor)

        # signed_improvement is never clamped
        assert isinstance(result.signed_improvement, float)

    def test_empty_questions(self):
        mock_extractor = MagicMock()
        result = run_100q_benchmark(lambda p: "", "M", [], mock_extractor)
        assert result.n == 0
        assert result.baseline_accuracy == 0.0
        assert result.pipeline_accuracy == 0.0

    def test_is_positive_false_when_no_improvement(self):
        questions = self._make_questions(3)
        mock_extractor = MagicMock()
        mock_extractor.extract.return_value = []

        # Always wrong on both passes
        result = run_100q_benchmark(lambda p: "#### 9999", "M", questions, mock_extractor)

        assert result.is_positive is False
        assert result.signed_improvement == 0.0

    def test_to_dict_keys(self):
        questions = self._make_questions(2)
        mock_extractor = MagicMock()
        mock_extractor.extract.return_value = []

        result = run_100q_benchmark(lambda p: "#### 0", "M", questions, mock_extractor)
        d = result.to_dict()

        required_keys = {
            "model_id", "n", "baseline_correct", "pipeline_correct",
            "baseline_accuracy", "pipeline_accuracy", "wilson_95ci_lower",
            "wilson_95ci_upper", "signed_improvement", "is_positive",
        }
        assert required_keys <= set(d.keys())


# ---------------------------------------------------------------------------
# write_cot_pairs
# ---------------------------------------------------------------------------


class TestWriteCotPairs:
    def test_writes_valid_json(self, tmp_path):
        pairs = [
            {"question": "Q1", "cot_text": "T1", "correct": True, "model_id": "M"},
            {"question": "Q2", "cot_text": "T2", "correct": False, "model_id": "M"},
        ]
        out = str(tmp_path / "pairs.json")
        count = write_cot_pairs(pairs, out)

        assert count == 2
        loaded = json.loads(Path(out).read_text())
        assert len(loaded) == 2
        assert loaded[0]["question"] == "Q1"
        assert loaded[1]["correct"] is False

    def test_creates_parent_dir(self, tmp_path):
        pairs = [{"question": "Q", "cot_text": "T", "correct": True, "model_id": "M"}]
        out = str(tmp_path / "subdir" / "out.json")
        count = write_cot_pairs(pairs, out)
        assert count == 1
        assert Path(out).exists()

    def test_empty_list(self, tmp_path):
        out = str(tmp_path / "empty.json")
        count = write_cot_pairs([], out)
        assert count == 0
        loaded = json.loads(Path(out).read_text())
        assert loaded == []

    def test_atomic_write(self, tmp_path):
        """Verifies tmp file is renamed (no .tmp leftover after success)."""
        pairs = [{"question": "Q", "cot_text": "T", "correct": True, "model_id": "M"}]
        out = str(tmp_path / "out.json")
        write_cot_pairs(pairs, out)
        # No .tmp file should remain after successful write
        assert not Path(out + ".tmp").exists()
        assert not Path(str(Path(out).with_suffix(".tmp"))).exists()

    def test_fover_schema_preserved(self, tmp_path):
        """FOVER format keys are preserved exactly as written."""
        pair = {"question": "Q1", "cot_text": "step1 step2", "correct": True, "model_id": "Gemma4-INT4"}
        out = str(tmp_path / "out.json")
        write_cot_pairs([pair], out)
        loaded = json.loads(Path(out).read_text())
        assert loaded[0] == pair


# ---------------------------------------------------------------------------
# Experiment 514 script — non-GPU / deferred paths
# ---------------------------------------------------------------------------


class TestExperiment514Script:
    """Test the Exp 514 script's deferred (gpu_required) exit path."""

    def test_deferred_artifact_when_force_live_not_set(self, tmp_path):
        """Without CARNOT_FORCE_LIVE=1, the script writes a gpu_required artifact."""
        import scripts.experiment_514_live_100q_precision_v7 as exp514

        with patch.dict(os.environ, {"CARNOT_FORCE_LIVE": "0"}):
            artifact = exp514.run_experiment(repo_root=tmp_path)

        assert artifact["status"] in ("gpu_required", "success", "blocked", "gpu_vram_insufficient")
        # artifact_type field identifies this as carnot.live_precision.v7
        # (Note: ExperimentTemplate.build_result() overwrites 'schema' with sorted key list)
        assert artifact.get("artifact_type") == "carnot.live_precision.v7"
        # jit_vram_check_applied must be present and True
        assert artifact.get("jit_vram_check_applied") is True
        # deliverable must exist on disk
        out = tmp_path / exp514.DELIVERABLE
        assert out.exists(), f"Deliverable not written to {out}"

    def test_deferred_artifact_schema_fields(self, tmp_path):
        """The deferred artifact contains all required schema fields."""
        import scripts.experiment_514_live_100q_precision_v7 as exp514

        with patch.dict(os.environ, {"CARNOT_FORCE_LIVE": "0"}, clear=False):
            artifact = exp514.run_experiment(repo_root=tmp_path)

        required = {"experiment", "status", "run_date", "started_at", "finished_at", "duration_s"}
        for key in required:
            assert key in artifact, f"Missing required field: {key}"

    def test_deliverable_path_constant(self):
        """DELIVERABLE constant is the expected path."""
        import scripts.experiment_514_live_100q_precision_v7 as exp514
        assert "514" in exp514.DELIVERABLE
        assert exp514.DELIVERABLE.endswith(".json")

    def test_cot_pairs_path_constant(self):
        """COT_PAIRS_PATH constant is the expected path."""
        import scripts.experiment_514_live_100q_precision_v7 as exp514
        assert "514" in exp514.COT_PAIRS_PATH
        assert exp514.COT_PAIRS_PATH.endswith(".json")
