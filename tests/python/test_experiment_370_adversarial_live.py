"""Tests for scripts/experiment_370_adversarial_live.py.

100% targeted coverage of all NEW functions added in Exp 370:
    - diagnose_live_gpu_or_raise: raises when CARNOT_FORCE_LIVE not set,
        raises when diagnose_live_gpu returns is_live_capable=False,
        returns diag result when live capable
    - _write_artifact: writes artifact JSON to expected path
    - main(): blocked when CARNOT_FORCE_LIVE not set,
        blocked when GPU unhealthy after diagnostic,
        blocked when diagnose_live_gpu not live capable,
        success path artifact schema / fields

Functions reused from Exp 355 (tested there) are NOT re-tested here.

Spec: REQ-BENCH-006, REQ-BENCH-007, SCENARIO-BENCH-022
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

import scripts.experiment_370_adversarial_live as exp370


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_live_diag(is_live: bool = True, reason: str = "") -> MagicMock:
    """Build a mock LiveGPUDiagnosticResult."""
    diag = MagicMock()
    diag.is_live_capable = is_live
    diag.cuda_visible = is_live
    diag.torch_available = is_live
    diag.model_loadable = is_live
    diag.carnot_force_live_set = True
    diag.failure_reason = reason if not is_live else ""
    return diag


# ---------------------------------------------------------------------------
# diagnose_live_gpu_or_raise
# ---------------------------------------------------------------------------


class TestDiagnoseLiveGpuOrRaise:
    """SCENARIO-BENCH-022: hard gate raises on bad conditions."""

    def test_raises_when_force_live_not_set(self):
        """Raises RuntimeError if CARNOT_FORCE_LIVE is not '1'."""
        env = {k: v for k, v in os.environ.items() if k != "CARNOT_FORCE_LIVE"}
        with patch.dict(os.environ, env, clear=True):
            with pytest.raises(RuntimeError, match="CARNOT_FORCE_LIVE"):
                exp370.diagnose_live_gpu_or_raise(["model-id"])

    def test_raises_when_force_live_zero(self):
        """Raises RuntimeError if CARNOT_FORCE_LIVE='0'."""
        with patch.dict(os.environ, {"CARNOT_FORCE_LIVE": "0"}):
            with pytest.raises(RuntimeError, match="CARNOT_FORCE_LIVE"):
                exp370.diagnose_live_gpu_or_raise(["model-id"])

    def test_raises_when_gpu_not_live_capable(self):
        """Raises RuntimeError when diagnose_live_gpu returns is_live_capable=False."""
        dead_diag = _make_live_diag(is_live=False, reason="no CUDA device found")
        with (
            patch.dict(os.environ, {"CARNOT_FORCE_LIVE": "1"}),
            patch.object(exp370, "diagnose_live_gpu", return_value=dead_diag),
        ):
            with pytest.raises(RuntimeError, match="no CUDA device found"):
                exp370.diagnose_live_gpu_or_raise(["model-id"])

    def test_returns_diag_when_live_capable(self):
        """Returns diag result when GPU is confirmed live."""
        live_diag = _make_live_diag(is_live=True)
        with (
            patch.dict(os.environ, {"CARNOT_FORCE_LIVE": "1"}),
            patch.object(exp370, "diagnose_live_gpu", return_value=live_diag),
        ):
            result = exp370.diagnose_live_gpu_or_raise(["model-id"])
        assert result.is_live_capable is True

    def test_error_message_contains_failure_reason(self):
        """RuntimeError message includes the failure_reason from diag."""
        dead_diag = _make_live_diag(is_live=False, reason="torch not installed")
        with (
            patch.dict(os.environ, {"CARNOT_FORCE_LIVE": "1"}),
            patch.object(exp370, "diagnose_live_gpu", return_value=dead_diag),
        ):
            with pytest.raises(RuntimeError, match="torch not installed"):
                exp370.diagnose_live_gpu_or_raise(["model-id"])


# ---------------------------------------------------------------------------
# _write_artifact
# ---------------------------------------------------------------------------


class TestWriteArtifact:
    """_write_artifact writes JSON to expected path."""

    def test_writes_json_file(self, tmp_path: Path):
        """Artifact is written as valid JSON to the correct path."""
        tmpl = MagicMock()
        artifact = {"adversarial_schema": "carnot.adversarial_gsm8k.v2", "test": True}
        with patch.object(exp370, "_REPO_ROOT", tmp_path):
            exp370._write_artifact(tmpl, artifact)

        out = tmp_path / exp370.DELIVERABLE
        assert out.exists()
        loaded = json.loads(out.read_text())
        assert loaded["adversarial_schema"] == "carnot.adversarial_gsm8k.v2"
        assert loaded["test"] is True

    def test_creates_parent_dirs(self, tmp_path: Path):
        """Parent directories are created if they do not exist."""
        tmpl = MagicMock()
        with patch.object(exp370, "_REPO_ROOT", tmp_path):
            exp370._write_artifact(tmpl, {"x": 1})
        assert (tmp_path / exp370.DELIVERABLE).exists()


# ---------------------------------------------------------------------------
# main() — blocked paths
# ---------------------------------------------------------------------------


class TestMainBlocked:
    """SCENARIO-BENCH-022: main() writes blocked artifact when gates fail."""

    def test_blocked_when_force_live_not_set(self, tmp_path: Path):
        """main() writes blocked artifact when CARNOT_FORCE_LIVE is not set."""
        env = {k: v for k, v in os.environ.items() if k != "CARNOT_FORCE_LIVE"}
        with (
            patch.object(exp370, "_REPO_ROOT", tmp_path),
            patch.dict(os.environ, env, clear=True),
        ):
            exp370.main()

        out = tmp_path / exp370.DELIVERABLE
        assert out.exists()
        art = json.loads(out.read_text())
        assert art["status"] == "blocked"
        assert art["honest_verdict"] == "blocked"

    def test_blocked_when_gpu_not_live(self, tmp_path: Path):
        """main() writes blocked artifact when diagnose_live_gpu_or_raise raises."""
        dead_diag = _make_live_diag(is_live=False, reason="CUDA not found")

        with (
            patch.object(exp370, "_REPO_ROOT", tmp_path),
            patch.dict(os.environ, {"CARNOT_FORCE_LIVE": "1"}),
            patch.object(
                exp370,
                "diagnose_live_gpu",
                return_value=dead_diag,
            ),
        ):
            exp370.main()

        art = json.loads((tmp_path / exp370.DELIVERABLE).read_text())
        assert art["status"] == "blocked"
        assert art["honest_verdict"] == "blocked"
        assert "CUDA not found" in art["failure_reason"]

    def test_blocked_when_gpu_setup_unhealthy(self, tmp_path: Path):
        """main() writes blocked artifact when setup_gpu returns all_healthy=False."""
        live_diag = _make_live_diag(is_live=True)
        unhealthy_status = {
            "all_healthy": False,
            "models": [],
            "prewarm_time_s": 0.0,
            "dual_gpu_auto_assigned": False,
            "note": "GPU stalled",
        }

        with (
            patch.object(exp370, "_REPO_ROOT", tmp_path),
            patch.dict(os.environ, {"CARNOT_FORCE_LIVE": "1"}),
            patch.object(exp370, "diagnose_live_gpu", return_value=live_diag),
            patch(
                "scripts.experiment_template.ExperimentTemplate.setup_gpu",
                return_value=unhealthy_status,
            ),
        ):
            exp370.main()

        art = json.loads((tmp_path / exp370.DELIVERABLE).read_text())
        assert art["status"] == "blocked"
        assert "GPU setup unhealthy" in art["failure_reason"]

    def test_blocked_artifact_has_schema(self, tmp_path: Path):
        """Blocked artifact still includes schema field."""
        env = {k: v for k, v in os.environ.items() if k != "CARNOT_FORCE_LIVE"}
        with (
            patch.object(exp370, "_REPO_ROOT", tmp_path),
            patch.dict(os.environ, env, clear=True),
        ):
            exp370.main()

        art = json.loads((tmp_path / exp370.DELIVERABLE).read_text())
        assert art["adversarial_schema"] == "carnot.adversarial_gsm8k.v2"

    def test_no_simulated_fallback_when_force_live_set(self, tmp_path: Path):
        """When CARNOT_FORCE_LIVE=1 but GPU dead, honest_verdict is 'blocked' not 'blocked_simulated'."""
        dead_diag = _make_live_diag(is_live=False, reason="no GPU")
        with (
            patch.object(exp370, "_REPO_ROOT", tmp_path),
            patch.dict(os.environ, {"CARNOT_FORCE_LIVE": "1"}),
            patch.object(exp370, "diagnose_live_gpu", return_value=dead_diag),
        ):
            exp370.main()

        art = json.loads((tmp_path / exp370.DELIVERABLE).read_text())
        assert art["honest_verdict"] != "blocked_simulated"
        assert art["honest_verdict"] == "blocked"


# ---------------------------------------------------------------------------
# main() — success path
# ---------------------------------------------------------------------------


def _make_success_run_result() -> MagicMock:
    """Build a mock AdversarialBenchmarkResult for the success path."""
    from carnot.pipeline.adversarial_gsm8k import AdversarialBenchmarkResult

    return AdversarialBenchmarkResult(
        standard_accuracy=0.80,
        adversarial_accuracy=0.78,
        accuracy_drop=0.02,
        repaired_adversarial_accuracy=0.82,
        repair_improvement=0.04,
        inference_mode="live_gpu",
    )


class TestMainSuccess:
    """SCENARIO-BENCH-022: success path artifact schema and fields."""

    def _run_success(self, tmp_path: Path) -> dict:
        """Run main() with all live gates mocked open, return parsed artifact."""
        live_diag = _make_live_diag(is_live=True)
        healthy_gpu = {
            "all_healthy": True,
            "models": [],
            "prewarm_time_s": 0.5,
            "dual_gpu_auto_assigned": True,
            "note": "",
        }
        live_result = _make_success_run_result()

        with (
            patch.object(exp370, "_REPO_ROOT", tmp_path),
            patch.dict(os.environ, {"CARNOT_FORCE_LIVE": "1"}),
            patch.object(exp370, "diagnose_live_gpu", return_value=live_diag),
            patch(
                "scripts.experiment_template.ExperimentTemplate.setup_gpu",
                return_value=healthy_gpu,
            ),
            patch.object(exp370, "run_adversarial_benchmark", return_value=live_result),
        ):
            exp370.main()

        return json.loads((tmp_path / exp370.DELIVERABLE).read_text())

    def test_artifact_has_v2_schema(self, tmp_path: Path):
        """Success artifact uses adversarial_schema='carnot.adversarial_gsm8k.v2'."""
        art = self._run_success(tmp_path)
        assert art.get("adversarial_schema") == "carnot.adversarial_gsm8k.v2"

    def test_inference_mode_live_gpu(self, tmp_path: Path):
        """Success artifact has inference_mode='live_gpu'."""
        art = self._run_success(tmp_path)
        assert art["inference_mode"] == "live_gpu"

    def test_honest_verdict_not_blocked_simulated(self, tmp_path: Path):
        """SCENARIO-BENCH-022: honest_verdict is never 'blocked_simulated' on live GPU."""
        art = self._run_success(tmp_path)
        assert art["honest_verdict"] != "blocked_simulated"

    def test_honest_verdict_improvement_positive(self, tmp_path: Path):
        """repair_improvement > 0 → honest_verdict='improvement_positive'."""
        art = self._run_success(tmp_path)
        # Mock result has repair_improvement=0.04 > 0
        assert art["honest_verdict"] == "improvement_positive"

    def test_required_top_level_fields(self, tmp_path: Path):
        """All REQUIRED_RESULT_FIELDS and exp370-specific fields are present."""
        art = self._run_success(tmp_path)
        for field in [
            "experiment", "schema", "run_date", "started_at", "finished_at",
            "duration_s", "status", "title",
        ]:
            assert field in art, f"Missing required field: {field}"
        for field in [
            "inference_mode", "honest_verdict", "per_model_results",
            "headline_result", "standard_accuracy", "adversarial_accuracy",
            "accuracy_drop", "repaired_adversarial_accuracy", "repair_improvement",
            "robustness_invariant_holds", "n_questions", "n_models",
        ]:
            assert field in art, f"Missing exp370 field: {field}"

    def test_per_model_results_have_required_fields(self, tmp_path: Path):
        """Each per_model_results entry has all SCENARIO-BENCH-019/022 required fields."""
        art = self._run_success(tmp_path)
        for entry in art["per_model_results"]:
            for key in [
                "model_id", "n_questions", "standard_accuracy",
                "adversarial_accuracy", "accuracy_drop",
                "repaired_adversarial_accuracy", "repair_improvement", "inference_mode",
            ]:
                assert key in entry, f"per_model_results entry missing: {key}"

    def test_headline_result_fields(self, tmp_path: Path):
        """headline_result contains all required fields."""
        art = self._run_success(tmp_path)
        hl = art["headline_result"]
        for key in [
            "honest_verdict", "inference_mode", "n_models", "n_questions_per_model",
            "avg_accuracy_drop", "avg_repair_improvement",
            "robustness_invariant_holds", "improvement_positive",
        ]:
            assert key in hl, f"headline_result missing: {key}"

    def test_experiment_id_is_370(self, tmp_path: Path):
        """Artifact experiment field is 370."""
        art = self._run_success(tmp_path)
        assert art["experiment"] == 370

    def test_n_models_matches_model_specs(self, tmp_path: Path):
        """n_models in artifact equals len(MODEL_SPECS)."""
        art = self._run_success(tmp_path)
        assert art["n_models"] == len(exp370.MODEL_SPECS)
        assert len(art["per_model_results"]) == len(exp370.MODEL_SPECS)

    def test_robustness_invariant_holds_when_drop_small(self, tmp_path: Path):
        """robustness_invariant_holds=True when adversarial drop <= 5 pp."""
        art = self._run_success(tmp_path)
        # Mock adversarial_accuracy=0.78, standard_accuracy=0.80 → drop=0.02 < 0.05
        assert art["robustness_invariant_holds"] is True

    def test_status_success(self, tmp_path: Path):
        """Success artifact has status='success'."""
        art = self._run_success(tmp_path)
        assert art["status"] == "success"
