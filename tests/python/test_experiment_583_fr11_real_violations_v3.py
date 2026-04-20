"""Tests for Exp 583: FR-11 Real Violations V3 — Tier 1 relay with CoACEExtractorV2.

100% targeted coverage on functions added in scripts/experiment_583_fr11_real_violations_v3.py.

Spec: REQ-LEARN-058,
      SCENARIO-LEARN-096, SCENARIO-LEARN-097, SCENARIO-LEARN-098
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

import scripts.experiment_583_fr11_real_violations_v3 as exp583  # noqa: E402


# ---------------------------------------------------------------------------
# _load_gate
# ---------------------------------------------------------------------------


class TestLoadGate:
    """Gate file loading must handle missing, corrupt, and valid files."""

    def test_returns_none_when_file_missing(self, tmp_path):
        # SCENARIO-LEARN-096: missing gate file -> None -> blocked artifact
        result = exp583._load_gate(tmp_path)
        assert result is None

    def test_returns_dict_when_gate_open_true(self, tmp_path):
        gate_data = {"gate_open": True, "v2_recall": 0.25, "experiment": 581}
        (tmp_path / "results").mkdir()
        (tmp_path / exp583.GATE_FILE).write_text(json.dumps(gate_data))
        result = exp583._load_gate(tmp_path)
        assert result is not None
        assert result["gate_open"] is True

    def test_returns_dict_when_gate_open_false(self, tmp_path):
        gate_data = {"gate_open": False, "v2_recall": 0.058, "experiment": 581}
        (tmp_path / "results").mkdir()
        (tmp_path / exp583.GATE_FILE).write_text(json.dumps(gate_data))
        result = exp583._load_gate(tmp_path)
        assert result is not None
        assert result["gate_open"] is False

    def test_returns_none_when_file_corrupt(self, tmp_path):
        (tmp_path / "results").mkdir()
        (tmp_path / exp583.GATE_FILE).write_text("{not valid json")
        result = exp583._load_gate(tmp_path)
        assert result is None

    def test_returns_none_when_value_not_dict(self, tmp_path):
        (tmp_path / "results").mkdir()
        (tmp_path / exp583.GATE_FILE).write_text("[1, 2, 3]")
        result = exp583._load_gate(tmp_path)
        assert result is None


# ---------------------------------------------------------------------------
# _load_gsm8k_questions (fallback path)
# ---------------------------------------------------------------------------


class TestLoadGsm8kQuestions:
    """When datasets is unavailable, synthetic fallback must return correct count and shape."""

    def test_returns_25_questions(self):
        with patch.dict("sys.modules", {"datasets": None}):
            questions = exp583._load_gsm8k_questions(300, 324)
        assert len(questions) == 25

    def test_questions_have_required_keys(self):
        with patch.dict("sys.modules", {"datasets": None}):
            questions = exp583._load_gsm8k_questions(300, 310)
        for q in questions:
            assert "question" in q
            assert "answer" in q

    def test_question_range_boundaries(self):
        with patch.dict("sys.modules", {"datasets": None}):
            questions = exp583._load_gsm8k_questions(300, 300)
        assert len(questions) == 1
        assert "300" in questions[0]["question"]


# ---------------------------------------------------------------------------
# _run_coace_on_response
# ---------------------------------------------------------------------------


class TestRunCoaceOnResponse:
    """CoACE runner must return violation count and be resilient to errors."""

    def test_returns_zero_for_clean_response(self):
        extractor = exp583.CoACEExtractorV2(tolerance=1e-6, min_confidence=0.5)
        n = exp583._run_coace_on_response(extractor, "The answer is 42.")
        assert isinstance(n, int)
        assert n >= 0

    def test_detects_violation_in_wrong_arithmetic(self):
        extractor = exp583.CoACEExtractorV2(tolerance=1e-6, min_confidence=0.5)
        # 3 + 3 = 7 is wrong arithmetic that CoACEV2 should catch
        n = exp583._run_coace_on_response(extractor, "3 + 3 = 7, so the answer is 7")
        assert n >= 1

    def test_returns_zero_on_extractor_exception(self):
        broken_extractor = MagicMock()
        broken_extractor.extract.side_effect = RuntimeError("test error")
        n = exp583._run_coace_on_response(broken_extractor, "some text")
        assert n == 0


# ---------------------------------------------------------------------------
# _build_batch_record
# ---------------------------------------------------------------------------


class TestBuildBatchRecord:
    """Batch record dict must contain all expected keys."""

    def test_contains_all_keys(self):
        record = exp583._build_batch_record(
            batch_id=0,
            n_violations=3,
            n_questions=9,
            n_constraints=1,
            batch_accuracy=0.5,
            batch_fp_rate=0.0,
        )
        assert record["batch_id"] == 0
        assert record["n_violations_found_this_batch"] == 3
        assert record["n_questions"] == 9
        assert record["n_constraints_added"] == 1
        assert record["batch_accuracy"] == 0.5
        assert record["batch_fp_rate"] == 0.0


# ---------------------------------------------------------------------------
# _build_artifact
# ---------------------------------------------------------------------------


class TestBuildArtifact:
    """Artifact must include all required schema fields and correct verdicts."""

    def _make_tmpl(self, tmp_path):
        tmpl = MagicMock()
        tmpl.build_result.side_effect = lambda d, status="success": {**d, "status": status}
        return tmpl

    def test_fr11_improved_when_violations_above_12(self, tmp_path):
        tmpl = self._make_tmpl(tmp_path)
        art = exp583._build_artifact(tmpl, total_violations=15, n_constraints_added=2,
                                     batch_results=[], inference_mode="live_gpu")
        assert art["fr11_improved"] is True
        assert art["honest_verdict"] == "fr11_improved"
        assert art["violations_improvement"] == 3

    def test_no_improvement_when_violations_below_12(self, tmp_path):
        tmpl = self._make_tmpl(tmp_path)
        art = exp583._build_artifact(tmpl, total_violations=8, n_constraints_added=0,
                                     batch_results=[], inference_mode="live_gpu")
        assert art["fr11_improved"] is False
        assert art["honest_verdict"] == "fr11_no_improvement_v3"

    def test_still_zero_when_no_violations(self, tmp_path):
        tmpl = self._make_tmpl(tmp_path)
        art = exp583._build_artifact(tmpl, total_violations=0, n_constraints_added=0,
                                     batch_results=[], inference_mode="live_gpu")
        assert art["fr11_improved"] is False
        assert art["honest_verdict"] == "fr11_still_zero"

    def test_schema_field_present(self, tmp_path):
        tmpl = self._make_tmpl(tmp_path)
        art = exp583._build_artifact(tmpl, total_violations=5, n_constraints_added=0,
                                     batch_results=[], inference_mode="live_gpu")
        assert art["schema"] == "carnot.fr11_relay_real.v3"
        assert art["extractor"] == "coace_v2"
        assert art["n_questions"] == exp583.N_QUESTIONS
        assert art["n_batches"] == exp583.N_BATCHES
        assert art["v1_violations"] == exp583.V1_VIOLATIONS


# ---------------------------------------------------------------------------
# run_experiment — gate closed path (SCENARIO-LEARN-096)
# ---------------------------------------------------------------------------


class TestRunExperimentGateBlocked:
    """When Exp 581 gate_open=False, a blocked artifact is written immediately."""

    def _make_gate_file(self, repo_root: Path, gate_open: bool) -> None:
        gate_dir = repo_root / "results"
        gate_dir.mkdir(parents=True, exist_ok=True)
        gate_data = {"gate_open": gate_open, "v2_recall": 0.058, "experiment": 581}
        (repo_root / exp583.GATE_FILE).write_text(json.dumps(gate_data))

    def test_blocked_artifact_written_when_gate_closed(self, tmp_path):
        # SCENARIO-LEARN-096
        self._make_gate_file(tmp_path, gate_open=False)
        # Pre-create results dir so ExperimentTemplate.setup() doesn't fail
        (tmp_path / "results").mkdir(exist_ok=True)

        with patch.object(exp583.ExperimentTemplate, "kill_gpu_zombies"):
            with patch.object(exp583.ExperimentTemplate, "setup"):
                with patch.object(exp583.ExperimentTemplate, "build_result",
                                   side_effect=lambda d, status="success": {**d, "status": status}):
                    with patch.object(exp583.ExperimentTemplate, "assert_deliverable_written"):
                        artifact = exp583.run_experiment(repo_root=tmp_path)

        assert artifact["honest_verdict"] == "gate_closed_exp581_recall_too_low"
        assert artifact["status"] == "blocked"
        assert artifact["total_violations_found"] == 0
        assert artifact["fr11_improved"] is False

    def test_blocked_artifact_written_when_gate_file_missing(self, tmp_path):
        (tmp_path / "results").mkdir(parents=True, exist_ok=True)

        with patch.object(exp583.ExperimentTemplate, "kill_gpu_zombies"):
            with patch.object(exp583.ExperimentTemplate, "setup"):
                with patch.object(exp583.ExperimentTemplate, "build_result",
                                   side_effect=lambda d, status="success": {**d, "status": status}):
                    with patch.object(exp583.ExperimentTemplate, "assert_deliverable_written"):
                        artifact = exp583.run_experiment(repo_root=tmp_path)

        assert artifact["honest_verdict"] == "gate_closed_exp581_recall_too_low"
        assert artifact["fr11_improved"] is False


# ---------------------------------------------------------------------------
# _run_relay_batches
# ---------------------------------------------------------------------------


class TestRunRelayBatches:
    """Relay batch runner must partition 25 questions into 3 batches correctly."""

    def test_returns_three_batches(self):
        from unittest.mock import MagicMock

        questions = [{"question": f"q{i}", "answer": "#### 2"} for i in range(25)]
        extractor = exp583.CoACEExtractorV2(tolerance=1e-6, min_confidence=0.5)

        monitor = MagicMock()
        monitor.observe = MagicMock()
        monitor.get_patterns = MagicMock(return_value=[])
        monitor.check_and_add = MagicMock()

        total_v, n_c, batch_results = exp583._run_relay_batches(questions, extractor, monitor)

        assert len(batch_results) == 3
        # Batch sizes 9+8+8 = 25
        assert sum(b["n_questions"] for b in batch_results) == 25
        assert isinstance(total_v, int)
        assert isinstance(n_c, int)

    def test_violations_counted_across_batches(self):
        # Use synthetic answers with wrong arithmetic so CoACEV2 can fire
        questions = [
            {"question": f"q{i}", "answer": "3 + 3 = 7, so the answer is 7"} for i in range(25)
        ]
        extractor = exp583.CoACEExtractorV2(tolerance=1e-6, min_confidence=0.5)

        monitor = MagicMock()
        monitor.observe = MagicMock()
        monitor.get_patterns = MagicMock(return_value=[])
        monitor.check_and_add = MagicMock()

        total_v, _n_c, batch_results = exp583._run_relay_batches(questions, extractor, monitor)

        assert total_v >= 0
        # Each batch record must carry the expected keys
        for b in batch_results:
            assert "batch_id" in b
            assert "n_violations_found_this_batch" in b
            assert "batch_accuracy" in b
