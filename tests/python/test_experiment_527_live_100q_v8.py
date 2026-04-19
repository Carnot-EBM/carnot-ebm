"""Tests for Exp 527 helpers: build_precision_v8_artifact and non-GPU paths in exp 527 script.

100% coverage on python/carnot/pipeline/live_100q_v8_helpers.py
and the non-GPU exit paths in scripts/experiment_527_live_100q_precision_v8.py.

Spec: REQ-BENCH-054, REQ-BENCH-055,
      SCENARIO-BENCH-071, SCENARIO-BENCH-072, SCENARIO-BENCH-073, SCENARIO-BENCH-074
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

from carnot.pipeline.live_100q_v8_helpers import (
    build_precision_v8_artifact,
    load_jit_gated_model,
    wilson_ci,
    write_cot_pairs,
)


# ---------------------------------------------------------------------------
# build_precision_v8_artifact
# ---------------------------------------------------------------------------


class TestBuildPrecisionV8Artifact:
    """SCENARIO-BENCH-074: build_precision_v8_artifact produces all required schema fields."""

    def _make_results(self, baseline: float = 0.50, pipeline: float = 0.55, n: int = 100) -> dict:
        ci_lo, ci_hi = wilson_ci(int(pipeline * n), n)
        return {
            "n_questions": n,
            "baseline_accuracy": baseline,
            "pipeline_accuracy": pipeline,
            "wilson_95ci_lower": ci_lo,
            "wilson_95ci_upper": ci_hi,
        }

    def test_schema_field(self):
        art = build_precision_v8_artifact(self._make_results(), "live_gpu", None)
        assert art["schema"] == "carnot.live_precision.v8"

    def test_retro_033_closed_when_live_and_positive(self):
        # pipeline_accuracy(0.55) > baseline_accuracy(0.50) AND live_gpu => closed
        art = build_precision_v8_artifact(self._make_results(0.50, 0.55), "live_gpu", None)
        assert art["retro_033_closed"] is True
        assert art["honest_verdict"] == "retro_033_closed"

    def test_retro_033_not_closed_when_no_improvement(self):
        art = build_precision_v8_artifact(self._make_results(0.50, 0.50), "live_gpu", None)
        assert art["retro_033_closed"] is False
        assert art["honest_verdict"] == "live_no_improvement"

    def test_retro_033_not_closed_when_gpu_required(self):
        art = build_precision_v8_artifact(self._make_results(0.50, 0.55), "gpu_required", None)
        assert art["retro_033_closed"] is False
        assert art["honest_verdict"] == "gpu_required"

    def test_env_autofix_applied_always_true(self):
        art = build_precision_v8_artifact(self._make_results(), "live_gpu", None)
        assert art["env_autofix_applied"] is True

    def test_cot_pairs_written_none(self):
        art = build_precision_v8_artifact(self._make_results(), "gpu_required", None)
        assert art["cot_pairs_written"] is None

    def test_cot_pairs_written_path(self):
        path = "results/exp527_cot_pairs.json"
        art = build_precision_v8_artifact(self._make_results(), "live_gpu", path)
        assert art["cot_pairs_written"] == path

    def test_signed_improvement_computed(self):
        art = build_precision_v8_artifact(self._make_results(0.40, 0.60), "live_gpu", None)
        assert abs(art["signed_improvement"] - 0.20) < 1e-9

    def test_signed_improvement_negative_allowed(self):
        # Regression: pipeline is WORSE than baseline — never clamp
        art = build_precision_v8_artifact(self._make_results(0.60, 0.40), "live_gpu", None)
        assert art["signed_improvement"] < 0
        assert art["is_positive"] is False
        assert art["retro_033_closed"] is False
        assert art["honest_verdict"] == "live_no_improvement"

    def test_missing_keys_default_to_zero(self):
        # Deferred path: results dict is empty
        art = build_precision_v8_artifact({}, "gpu_required", None)
        assert art["n_questions"] == 0
        assert art["baseline_accuracy"] == 0.0
        assert art["pipeline_accuracy"] == 0.0
        assert art["signed_improvement"] == 0.0

    def test_wilson_ci_fields_present(self):
        art = build_precision_v8_artifact(self._make_results(), "live_gpu", None)
        assert "wilson_95ci_lower" in art
        assert "wilson_95ci_upper" in art
        assert art["wilson_95ci_lower"] <= art["wilson_95ci_upper"]

    def test_all_required_schema_keys_present(self):
        required = {
            "schema", "inference_mode", "n_questions", "baseline_accuracy",
            "pipeline_accuracy", "signed_improvement", "wilson_95ci_lower",
            "wilson_95ci_upper", "is_positive", "retro_033_closed",
            "cot_pairs_written", "env_autofix_applied", "honest_verdict",
        }
        art = build_precision_v8_artifact(self._make_results(), "live_gpu", "some/path.json")
        assert required <= set(art.keys())


# ---------------------------------------------------------------------------
# load_jit_gated_model (re-exported from v7 — verify the re-export works)
# ---------------------------------------------------------------------------


class TestLoadJitGatedModelV8:
    """Verify load_jit_gated_model is correctly re-exported in the v8 module."""

    def test_gate_cleared_calls_loader(self):
        mock_result = MagicMock()
        mock_result.is_cleared = True
        mock_result.available_gb = 20.0

        mock_loader = MagicMock()

        with patch(
            "carnot.pipeline.jit_vram_check.JITVRAMCheck.gate_model_load",
            return_value=mock_result,
        ):
            result = load_jit_gated_model(lambda: mock_loader, "test-model", 10.0, 0)

        mock_loader.load.assert_called_once()
        assert result is mock_loader

    def test_gate_blocked_returns_none(self):
        mock_result = MagicMock()
        mock_result.is_cleared = False
        mock_result.available_gb = 2.0

        with patch(
            "carnot.pipeline.jit_vram_check.JITVRAMCheck.gate_model_load",
            return_value=mock_result,
        ):
            result = load_jit_gated_model(MagicMock, "model", 10.0, 0)

        assert result is None


# ---------------------------------------------------------------------------
# write_cot_pairs (re-exported — verify re-export and FOVER format)
# ---------------------------------------------------------------------------


class TestWriteCotPairsV8:
    def test_fover_format_preserved(self, tmp_path):
        pairs = [
            {"question": "Q1", "cot_text": "step by step", "correct": True, "model_id": "Gemma4-INT4"},
        ]
        out = str(tmp_path / "cot.json")
        n = write_cot_pairs(pairs, out)
        assert n == 1
        loaded = json.loads(Path(out).read_text())
        assert loaded[0] == pairs[0]

    def test_atomic_write_no_tmp_leftover(self, tmp_path):
        pairs = [{"question": "Q", "cot_text": "T", "correct": False, "model_id": "M"}]
        out = str(tmp_path / "out.json")
        write_cot_pairs(pairs, out)
        assert not Path(str(Path(out).with_suffix(".tmp"))).exists()


# ---------------------------------------------------------------------------
# Experiment 527 script — non-GPU / deferred paths
# ---------------------------------------------------------------------------


class TestExperiment527Script:
    """Test the Exp 527 script's deferred (gpu_required) exit path.

    SCENARIO-BENCH-072: deferred artifact is written when GPU is absent.
    """

    def test_deferred_artifact_when_force_live_not_set(self, tmp_path):
        """Without CARNOT_FORCE_LIVE=1, the script writes a gpu_required artifact."""
        import scripts.experiment_527_live_100q_precision_v8 as exp527

        with patch.dict(os.environ, {"CARNOT_FORCE_LIVE": "0"}):
            artifact = exp527.run_experiment(repo_root=tmp_path)

        assert artifact["status"] in ("gpu_required", "success", "blocked", "gpu_vram_insufficient")
        # Deliverable must exist on disk
        out = tmp_path / exp527.DELIVERABLE
        assert out.exists(), f"Deliverable not written to {out}"

    def test_deferred_artifact_required_schema_fields(self, tmp_path):
        """The deferred artifact contains all REQUIRED_RESULT_FIELDS from ExperimentTemplate."""
        import scripts.experiment_527_live_100q_precision_v8 as exp527

        with patch.dict(os.environ, {"CARNOT_FORCE_LIVE": "0"}, clear=False):
            artifact = exp527.run_experiment(repo_root=tmp_path)

        for key in {"experiment", "status", "run_date", "started_at", "finished_at", "duration_s"}:
            assert key in artifact, f"Missing required field: {key}"

    def test_deferred_artifact_has_v8_schema(self, tmp_path):
        """The deferred artifact uses the v8 schema identifier."""
        import scripts.experiment_527_live_100q_precision_v8 as exp527

        with patch.dict(os.environ, {"CARNOT_FORCE_LIVE": "0"}):
            artifact = exp527.run_experiment(repo_root=tmp_path)

        # The artifact_type field must be present in the deferred path
        assert artifact.get("artifact_type") == "carnot.live_precision.v8"

    def test_deferred_artifact_env_autofix_applied(self, tmp_path):
        """env_autofix_applied must be True in all paths — confirms RETRO-053 fix is active."""
        import scripts.experiment_527_live_100q_precision_v8 as exp527

        with patch.dict(os.environ, {"CARNOT_FORCE_LIVE": "0"}):
            artifact = exp527.run_experiment(repo_root=tmp_path)

        assert artifact.get("env_autofix_applied") is True

    def test_deliverable_path_constant(self):
        """DELIVERABLE constant references experiment 527."""
        import scripts.experiment_527_live_100q_precision_v8 as exp527
        assert "527" in exp527.DELIVERABLE
        assert exp527.DELIVERABLE.endswith(".json")

    def test_cot_pairs_path_constant(self):
        """COT_PAIRS_PATH constant references experiment 527."""
        import scripts.experiment_527_live_100q_precision_v8 as exp527
        assert "527" in exp527.COT_PAIRS_PATH
        assert exp527.COT_PAIRS_PATH.endswith(".json")

    def test_deferred_artifact_honest_verdict_gpu_required(self, tmp_path):
        """The deferred artifact honest_verdict is 'gpu_required'."""
        import scripts.experiment_527_live_100q_precision_v8 as exp527

        with patch.dict(os.environ, {"CARNOT_FORCE_LIVE": "0"}):
            artifact = exp527.run_experiment(repo_root=tmp_path)

        assert artifact.get("honest_verdict") == "gpu_required"
