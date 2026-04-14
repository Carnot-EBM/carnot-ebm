"""Tests for Tier 1 self-learning trainer from live experimental traces (Exp 272).

Covers:
- Constants: EXPERIMENT_NUMBER, RUN_DATE, RESULT_OUTPUT, WEIGHTS_OUTPUT
- build_exp272_payload: payload construction with experiment number patching
- run: end-to-end file I/O and trainer serialization

Spec: REQ-LEARN-001, SCENARIO-LEARN-001, REQ-LEARN-002
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from python.carnot.pipeline.live_trace_trainer import (
    EXPERIMENT_NUMBER,
    RESULT_OUTPUT,
    RUN_DATE,
    WEIGHTS_OUTPUT,
    build_exp272_payload,
    run,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


class TestConstants:
    """Constants must have expected values."""

    def test_experiment_number(self) -> None:
        """EXPERIMENT_NUMBER should be 272."""
        assert EXPERIMENT_NUMBER == 272

    def test_run_date(self) -> None:
        """RUN_DATE should match the session date."""
        assert RUN_DATE == "20260413"

    def test_result_output_path(self) -> None:
        """RESULT_OUTPUT should point to results directory."""
        assert "experiment_272" in str(RESULT_OUTPUT)
        assert RESULT_OUTPUT.suffix == ".json"

    def test_weights_output_path(self) -> None:
        """WEIGHTS_OUTPUT should point to results directory."""
        assert "tier1_live_weights" in str(WEIGHTS_OUTPUT)
        assert WEIGHTS_OUTPUT.suffix == ".json"


# ---------------------------------------------------------------------------
# build_exp272_payload
# ---------------------------------------------------------------------------


class TestBuildExp272Payload:
    """build_exp272_payload must construct and patch the Exp 272 result."""

    def _make_mock_exp_data(self) -> dict[str, Any]:
        """Helper to create mock experiment data."""
        return {
            "paired_runs": [
                {
                    "model_name": "qwen",
                    "cases": [
                        {
                            "evaluation": {"constraint_results": []},
                            "constraint_extraction_coverage": 1.0,
                            "partial_satisfaction": 1.0,
                            "semantic_violation_count": 0,
                            "mode": "verify_only",
                            "output_style": "code_only",
                            "exact_satisfaction": True,
                        }
                    ],
                }
            ]
        }

    @patch("python.carnot.pipeline.live_trace_trainer.build_tier1_live_retrain_payload")
    def test_build_exp272_patches_experiment_number(
        self, mock_builder: MagicMock
    ) -> None:
        """build_exp272_payload patches the experiment field."""
        mock_payload = {"experiment": 999, "title": "Old"}
        mock_tracker = MagicMock()
        mock_builder.return_value = (mock_payload, mock_tracker)

        exp219 = self._make_mock_exp_data()
        exp220 = self._make_mock_exp_data()
        exp221 = self._make_mock_exp_data()
        exp223 = self._make_mock_exp_data()

        result, tracker = build_exp272_payload(
            exp219=exp219,
            exp220=exp220,
            exp221=exp221,
            exp223_reference=exp223,
        )

        assert result["experiment"] == 272

    @patch("python.carnot.pipeline.live_trace_trainer.build_tier1_live_retrain_payload")
    def test_build_exp272_patches_run_date(self, mock_builder: MagicMock) -> None:
        """build_exp272_payload patches the run_date field."""
        mock_payload = {"experiment": 999, "title": "Old"}
        mock_tracker = MagicMock()
        mock_builder.return_value = (mock_payload, mock_tracker)

        exp219 = self._make_mock_exp_data()
        exp220 = self._make_mock_exp_data()
        exp221 = self._make_mock_exp_data()
        exp223 = self._make_mock_exp_data()

        result, tracker = build_exp272_payload(
            exp219=exp219,
            exp220=exp220,
            exp221=exp221,
            exp223_reference=exp223,
        )

        assert result["run_date"] == "20260413"

    @patch("python.carnot.pipeline.live_trace_trainer.build_tier1_live_retrain_payload")
    def test_build_exp272_patches_title(self, mock_builder: MagicMock) -> None:
        """build_exp272_payload updates the title field."""
        mock_payload = {"experiment": 999, "title": "Old"}
        mock_tracker = MagicMock()
        mock_builder.return_value = (mock_payload, mock_tracker)

        exp219 = self._make_mock_exp_data()
        exp220 = self._make_mock_exp_data()
        exp221 = self._make_mock_exp_data()
        exp223 = self._make_mock_exp_data()

        result, tracker = build_exp272_payload(
            exp219=exp219,
            exp220=exp220,
            exp221=exp221,
            exp223_reference=exp223,
        )

        assert "Exp 272" in result["title"]
        assert "live-only" in result["title"]

    @patch("python.carnot.pipeline.live_trace_trainer.build_tier1_live_retrain_payload")
    def test_build_exp272_returns_tracker(self, mock_builder: MagicMock) -> None:
        """build_exp272_payload returns the ConstraintTracker."""
        mock_payload = {"experiment": 999}
        mock_tracker = MagicMock()
        mock_builder.return_value = (mock_payload, mock_tracker)

        exp219 = self._make_mock_exp_data()
        exp220 = self._make_mock_exp_data()
        exp221 = self._make_mock_exp_data()
        exp223 = self._make_mock_exp_data()

        result, tracker = build_exp272_payload(
            exp219=exp219,
            exp220=exp220,
            exp221=exp221,
            exp223_reference=exp223,
        )

        assert tracker is mock_tracker

    @patch("python.carnot.pipeline.live_trace_trainer.build_tier1_live_retrain_payload")
    def test_build_exp272_updates_comparison_note(
        self, mock_builder: MagicMock
    ) -> None:
        """build_exp272_payload updates the comparison_to_exp223 note."""
        mock_payload = {
            "experiment": 999,
            "comparison_to_exp223": {"note": "Old note"},
        }
        mock_tracker = MagicMock()
        mock_builder.return_value = (mock_payload, mock_tracker)

        exp219 = self._make_mock_exp_data()
        exp220 = self._make_mock_exp_data()
        exp221 = self._make_mock_exp_data()
        exp223 = self._make_mock_exp_data()

        result, tracker = build_exp272_payload(
            exp219=exp219,
            exp220=exp220,
            exp221=exp221,
            exp223_reference=exp223,
        )

        note = result["comparison_to_exp223"]["note"]
        assert "Exp 272" in note
        assert "Exp 224" in note

    @patch("python.carnot.pipeline.live_trace_trainer.build_tier1_live_retrain_payload")
    def test_build_exp272_respects_holdout_fraction(
        self, mock_builder: MagicMock
    ) -> None:
        """build_exp272_payload forwards holdout_fraction parameter."""
        mock_payload = {"experiment": 999}
        mock_tracker = MagicMock()
        mock_builder.return_value = (mock_payload, mock_tracker)

        exp219 = self._make_mock_exp_data()
        exp220 = self._make_mock_exp_data()
        exp221 = self._make_mock_exp_data()
        exp223 = self._make_mock_exp_data()

        build_exp272_payload(
            exp219=exp219,
            exp220=exp220,
            exp221=exp221,
            exp223_reference=exp223,
            holdout_fraction=0.20,
        )

        # Verify that build_tier1_live_retrain_payload was called with the right fraction
        mock_builder.assert_called_once()
        call_kwargs = mock_builder.call_args[1]
        assert call_kwargs["holdout_fraction"] == 0.20


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------


class TestRun:
    """run must orchestrate file loading, training, and serialization."""

    @patch("python.carnot.pipeline.live_trace_trainer.load_json")
    @patch("python.carnot.pipeline.live_trace_trainer.build_exp272_payload")
    def test_run_loads_four_json_files(
        self, mock_builder: MagicMock, mock_load: MagicMock
    ) -> None:
        """run loads experiment_219/220/221/223 JSON files."""
        mock_load.return_value = {"paired_runs": []}
        mock_payload = {
            "experiment": 272,
            "run_date": "20260413",
            "title": "Test",
        }
        mock_tracker = MagicMock()
        mock_tracker.save = MagicMock()
        mock_builder.return_value = (mock_payload, mock_tracker)

        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "result.json"
            weights = Path(tmpdir) / "weights.json"

            run(
                exp219_path="test_219.json",
                exp220_path="test_220.json",
                exp221_path="test_221.json",
                exp223_path="test_223.json",
                output_path=output,
                weights_path=weights,
            )

            # Verify load_json was called 4 times
            assert mock_load.call_count == 4

    @patch("python.carnot.pipeline.live_trace_trainer.load_json")
    @patch("python.carnot.pipeline.live_trace_trainer.build_exp272_payload")
    def test_run_writes_result_json(
        self, mock_builder: MagicMock, mock_load: MagicMock
    ) -> None:
        """run writes the results to output_path."""
        mock_load.return_value = {"paired_runs": []}
        mock_payload = {
            "experiment": 272,
            "run_date": "20260413",
            "title": "Test Exp 272",
        }
        mock_tracker = MagicMock()
        mock_tracker.save = MagicMock()
        mock_builder.return_value = (mock_payload, mock_tracker)

        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "result.json"
            weights = Path(tmpdir) / "weights.json"

            result = run(
                exp219_path="test_219.json",
                exp220_path="test_220.json",
                exp221_path="test_221.json",
                exp223_path="test_223.json",
                output_path=output,
                weights_path=weights,
            )

            # Verify output file was created
            assert output.exists()
            # Verify content is valid JSON
            written = json.loads(output.read_text())
            assert written["experiment"] == 272

    @patch("python.carnot.pipeline.live_trace_trainer.load_json")
    @patch("python.carnot.pipeline.live_trace_trainer.build_exp272_payload")
    def test_run_calls_tracker_save(
        self, mock_builder: MagicMock, mock_load: MagicMock
    ) -> None:
        """run calls tracker.save() with the weights path."""
        mock_load.return_value = {"paired_runs": []}
        mock_payload = {
            "experiment": 272,
            "run_date": "20260413",
            "title": "Test",
        }
        mock_tracker = MagicMock()
        mock_builder.return_value = (mock_payload, mock_tracker)

        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "result.json"
            weights = Path(tmpdir) / "weights.json"

            run(
                exp219_path="test_219.json",
                exp220_path="test_220.json",
                exp221_path="test_221.json",
                exp223_path="test_223.json",
                output_path=output,
                weights_path=weights,
            )

            # Verify tracker.save was called
            mock_tracker.save.assert_called_once()
            saved_path = mock_tracker.save.call_args[0][0]
            assert str(weights) in saved_path

    @patch("python.carnot.pipeline.live_trace_trainer.load_json")
    @patch("python.carnot.pipeline.live_trace_trainer.build_exp272_payload")
    def test_run_returns_result_payload(
        self, mock_builder: MagicMock, mock_load: MagicMock
    ) -> None:
        """run returns the full results dict."""
        mock_load.return_value = {"paired_runs": []}
        mock_payload = {
            "experiment": 272,
            "run_date": "20260413",
            "title": "Test",
            "some_metric": 0.95,
        }
        mock_tracker = MagicMock()
        mock_tracker.save = MagicMock()
        mock_builder.return_value = (mock_payload, mock_tracker)

        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "result.json"
            weights = Path(tmpdir) / "weights.json"

            result = run(
                exp219_path="test_219.json",
                exp220_path="test_220.json",
                exp221_path="test_221.json",
                exp223_path="test_223.json",
                output_path=output,
                weights_path=weights,
            )

            assert result["experiment"] == 272
            assert result["some_metric"] == 0.95

    @patch("python.carnot.pipeline.live_trace_trainer.load_json")
    @patch("python.carnot.pipeline.live_trace_trainer.build_exp272_payload")
    def test_run_creates_output_directories(
        self, mock_builder: MagicMock, mock_load: MagicMock
    ) -> None:
        """run creates parent directories for output and weights."""
        mock_load.return_value = {"paired_runs": []}
        mock_payload = {
            "experiment": 272,
            "run_date": "20260413",
            "title": "Test",
        }
        mock_tracker = MagicMock()
        mock_tracker.save = MagicMock()
        mock_builder.return_value = (mock_payload, mock_tracker)

        with tempfile.TemporaryDirectory() as tmpdir:
            # Use nested directories that don't exist yet
            output = Path(tmpdir) / "deep" / "nested" / "result.json"
            weights = Path(tmpdir) / "deep" / "nested" / "weights.json"

            run(
                exp219_path="test_219.json",
                exp220_path="test_220.json",
                exp221_path="test_221.json",
                exp223_path="test_223.json",
                output_path=output,
                weights_path=weights,
            )

            # Verify directories were created
            assert output.parent.exists()
            assert weights.parent.exists()
