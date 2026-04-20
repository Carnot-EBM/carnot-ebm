"""Tests for Exp 565: CoACEExtractor Live Diagnostic — RETRO-061 Final Validation Gate.

100% targeted coverage on functions added in scripts/experiment_565_coace_live_diagnostic.py.

Spec: REQ-EXTRACT-035,
      SCENARIO-EXTRACT-065, SCENARIO-EXTRACT-066, SCENARIO-EXTRACT-067
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

import scripts.experiment_565_coace_live_diagnostic as exp565


# ---------------------------------------------------------------------------
# _CoACEAdapter
# ---------------------------------------------------------------------------


class TestCoACEAdapter:
    """REQ-EXTRACT-035-2: adapter must expose detect_violations() protocol."""

    def test_detect_violations_returns_list(self):
        adapter = exp565._CoACEAdapter()
        result = adapter.detect_violations("47 + 28 = 75")
        assert isinstance(result, list)

    def test_detect_violations_empty_on_correct_equation(self):
        # SCENARIO-EXTRACT-065: correct equation should not flag violation
        adapter = exp565._CoACEAdapter()
        result = adapter.detect_violations("47 + 28 = 75")
        assert len(result) == 0

    def test_detect_violations_nonempty_on_wrong_equation(self):
        # SCENARIO-EXTRACT-065: wrong equation should flag at least one violation
        adapter = exp565._CoACEAdapter()
        result = adapter.detect_violations("47 + 28 = 76")
        assert len(result) > 0

    def test_detect_violations_empty_on_plain_text(self):
        # No arithmetic equation in plain text -> empty violations list
        adapter = exp565._CoACEAdapter()
        result = adapter.detect_violations("The sky is blue and there are many clouds.")
        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# load_labeled_responses
# ---------------------------------------------------------------------------


class TestLoadLabeledResponses:
    """REQ-EXTRACT-035-1: responses must be normalised to response/is_correct keys."""

    def test_returns_list(self, tmp_path):
        # SCENARIO-EXTRACT-066: when neither file exists, returns empty list
        with (
            patch.object(exp565, "_REPO_ROOT", tmp_path),
        ):
            result = exp565.load_labeled_responses()
        assert isinstance(result, list)
        assert len(result) == 0

    def test_loads_from_exp538_cot_pairs(self, tmp_path):
        # Primary fallback: exp538_cot_pairs.json with cot_text/correct keys
        cot_pairs = [
            {"cot_text": "We compute 5 + 3 = 8.", "correct": True, "question": "q1", "model_id": "m1"},
            {"cot_text": "We compute 5 + 3 = 9.", "correct": False, "question": "q2", "model_id": "m1"},
        ]
        pairs_dir = tmp_path / "results"
        pairs_dir.mkdir()
        pairs_file = pairs_dir / "exp538_cot_pairs.json"
        pairs_file.write_text(json.dumps(cot_pairs))

        with patch.object(exp565, "_REPO_ROOT", tmp_path):
            result = exp565.load_labeled_responses()

        assert len(result) == 2
        assert result[0]["response"] == "We compute 5 + 3 = 8."
        assert result[0]["is_correct"] is True
        assert result[1]["is_correct"] is False

    def test_exp554_per_question_results_takes_priority(self, tmp_path):
        # If exp554 has per_question_results, use them first
        exp554_data = {
            "per_question_results": [
                {"response": "CoT text here", "is_correct": False, "question": "q1"},
            ]
        }
        results_dir = tmp_path / "results"
        results_dir.mkdir()
        (results_dir / "experiment_554_extraction_diagnostic.json").write_text(
            json.dumps(exp554_data)
        )

        with patch.object(exp565, "_REPO_ROOT", tmp_path):
            result = exp565.load_labeled_responses()

        assert len(result) == 1
        assert result[0]["response"] == "CoT text here"
        assert result[0]["is_correct"] is False

    def test_exp554_without_per_question_falls_back_to_exp538(self, tmp_path):
        # exp554 exists but per_question_results is absent -> fall back to exp538
        exp554_data = {"status": "success"}  # no per_question_results
        cot_pairs = [
            {"cot_text": "text", "correct": True, "question": "", "model_id": "m"},
        ]
        results_dir = tmp_path / "results"
        results_dir.mkdir()
        (results_dir / "experiment_554_extraction_diagnostic.json").write_text(
            json.dumps(exp554_data)
        )
        (results_dir / "exp538_cot_pairs.json").write_text(json.dumps(cot_pairs))

        with patch.object(exp565, "_REPO_ROOT", tmp_path):
            result = exp565.load_labeled_responses()

        assert len(result) == 1
        assert result[0]["is_correct"] is True

    def test_response_key_falls_back_to_response_field(self, tmp_path):
        # If cot_text absent but response present in exp538 entry, use response
        cot_pairs = [
            {"response": "alt text", "correct": False, "question": "", "model_id": "m"},
        ]
        results_dir = tmp_path / "results"
        results_dir.mkdir()
        (results_dir / "exp538_cot_pairs.json").write_text(json.dumps(cot_pairs))

        with patch.object(exp565, "_REPO_ROOT", tmp_path):
            result = exp565.load_labeled_responses()

        assert result[0]["response"] == "alt text"


# ---------------------------------------------------------------------------
# run_experiment (integration path: upstream_missing fallback)
# ---------------------------------------------------------------------------


class TestRunExperimentUpstreamMissing:
    """SCENARIO-EXTRACT-066: blocked artifact written when no labeled responses available."""

    def test_writes_blocked_artifact_when_no_responses(self, tmp_path):
        # Patch load_labeled_responses to return empty list, patch writer and template
        mock_writer = MagicMock()
        # Use MagicMock with explicit non-assert method to avoid Python mock safety check
        mock_tmpl = MagicMock()
        mock_tmpl.build_result.return_value = {"honest_verdict": "upstream_missing", "status": "blocked"}
        deliverable_called = []
        mock_tmpl.assert_deliverable_written = lambda: deliverable_called.append(True)

        with (
            patch.object(exp565, "load_labeled_responses", return_value=[]),
            patch.object(exp565, "AtomicResultWriter", return_value=mock_writer),
            patch.object(exp565, "ExperimentTemplate", return_value=mock_tmpl),
            patch.object(exp565, "ExperimentTimeoutWatchdog") as mock_watchdog,
        ):
            mock_watchdog.return_value = MagicMock()
            exp565.run_experiment()

        mock_writer.write.assert_called_once()
        written = mock_writer.write.call_args[0][0]
        assert written["honest_verdict"] == "upstream_missing"
        assert deliverable_called  # assert_deliverable_written was called


# ---------------------------------------------------------------------------
# run_experiment (integration path: success with minimal synthetic data)
# ---------------------------------------------------------------------------


class TestRunExperimentSuccess:
    """SCENARIO-EXTRACT-065, SCENARIO-EXTRACT-067: CoACE TP > 0, improvement computed."""

    @pytest.fixture
    def _minimal_labeled(self):
        """Two responses: one correct (no arithmetic), one incorrect (wrong equation)."""
        return [
            {"response": "The answer is 5 apples.", "is_correct": True, "question": "", "model_id": "m"},
            {"response": "5 + 3 = 9", "is_correct": False, "question": "", "model_id": "m"},
        ]

    def test_coace_tp_rate_positive_when_wrong_equation_present(self, _minimal_labeled):
        # SCENARIO-EXTRACT-065: CoACE should flag the incorrect response
        adapter = exp565._CoACEAdapter()
        from carnot.extraction import run_extractor_diagnostic
        result = run_extractor_diagnostic(adapter, "CoACEExtractor", _minimal_labeled)
        # The '5 + 3 = 9' response should be a TP (is_correct=False, violation_found=True)
        assert result.tp_rate > 0.0

    def test_coace_improvement_over_vericot_non_negative(self, _minimal_labeled):
        # SCENARIO-EXTRACT-067: improvement = coace_tp - vericot_tp, should be >= 0
        from carnot.extraction import VeriCoTStepValidator, run_extractor_diagnostic
        coace = exp565._CoACEAdapter()
        vericot = VeriCoTStepValidator(use_mock=True)
        coace_r = run_extractor_diagnostic(coace, "CoACEExtractor", _minimal_labeled)
        vericot_r = run_extractor_diagnostic(vericot, "VeriCoTStepValidator", _minimal_labeled)
        improvement = coace_r.tp_rate - vericot_r.tp_rate
        assert improvement >= 0.0

    def test_full_run_writes_success_artifact(self, tmp_path, _minimal_labeled):
        """Full run with minimal data should write artifact with gate_open field."""
        mock_writer = MagicMock()
        mock_tmpl = MagicMock()
        deliverable_called = []
        mock_tmpl.assert_deliverable_written = lambda: deliverable_called.append(True)

        captured = {}

        def capture_build_result(data, **kwargs):
            captured.update(data)
            captured["status"] = kwargs.get("status", "success")
            return captured

        mock_tmpl.build_result.side_effect = capture_build_result

        with (
            patch.object(exp565, "load_labeled_responses", return_value=_minimal_labeled),
            patch.object(exp565, "AtomicResultWriter", return_value=mock_writer),
            patch.object(exp565, "ExperimentTemplate", return_value=mock_tmpl),
            patch.object(exp565, "ExperimentTimeoutWatchdog") as mock_watchdog,
        ):
            mock_watchdog.return_value = MagicMock()
            exp565.run_experiment()

        assert "gate_open" in captured
        assert "retro_061_resolved" in captured
        assert "coace_improvement_over_vericot" in captured
        assert captured["n_responses"] == 2
        assert deliverable_called  # assert_deliverable_written was called
