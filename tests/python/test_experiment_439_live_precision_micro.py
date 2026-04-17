"""Tests for Exp 439: live precision micro-benchmark (50q × 3 variants × 2 models).

Coverage targets
----------------
MicroPrecisionResult:
  - Instantiation with all required fields
  - positive signed_improvement (crane_only/full_stack improved on baseline)
  - negative signed_improvement (regression — must not be clamped)
  - zero signed_improvement (no change)
  - crane_detection_rate=0.0 for BASELINE variant

build_micro_precision_artifact():
  - empty results → honest_verdict='blocked', headline_result=None
  - all simulated (inference_mode!='live_gpu') → 'blocked' even if improvement > 0
  - mixed live/simulated → 'blocked'
  - all live_gpu, best improvement > 0 → 'live_improvement'
  - all live_gpu, best improvement == 0 → 'live_no_improvement'
  - all live_gpu, best improvement < 0 → 'live_no_improvement'
  - headline_result is the best non-baseline result by signed_improvement
  - when only baseline results present, headline is drawn from baselines
  - per_model_results includes all results
  - all required headline_result fields present
  - schema = 'carnot.precision_micro.v1'
  - inference_mode = 'live_gpu' for non-blocked verdict

run_experiment() (scripts/experiment_439_live_precision_micro.py):
  - CI mode (CARNOT_FORCE_LIVE=0): honest_verdict='blocked', all REQUIRED_RESULT_FIELDS
  - artifact written to disk
  - env_autofix block embedded in artifact
  - blocked artifact contains precision_micro schema fields

main():
  - calls run_experiment() inside ExperimentTimeoutWatchdog
  - watchdog called with experiment_id=439

Spec: REQ-BENCH-009, SCENARIO-BENCH-025, SCENARIO-BENCH-026 (Exp 439)
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from carnot.pipeline.precision_micro import (  # noqa: E402
    MicroPrecisionResult,
    build_micro_precision_artifact,
)
from scripts.experiment_template import REQUIRED_RESULT_FIELDS  # noqa: E402


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_result(
    variant: str,
    signed_improvement: float,
    inference_mode: str = "live_gpu",
    model_id: str = "Gemma4-E4B-it",
    n_questions: int = 50,
    baseline_accuracy: float = 0.60,
) -> MicroPrecisionResult:
    """Build a MicroPrecisionResult for testing."""
    var_acc = baseline_accuracy + signed_improvement
    crane_rate = 0.0 if variant == "baseline" else 0.3
    return MicroPrecisionResult(
        model_id=model_id,
        variant=variant,
        n_questions=n_questions,
        baseline_accuracy=baseline_accuracy,
        variant_accuracy=var_acc,
        signed_improvement=signed_improvement,
        crane_detection_rate=crane_rate,
        inference_mode=inference_mode,
    )


def _make_autofix(gpu_detected: bool = False) -> object:
    from carnot.pipeline.env_autofix import EnvironmentAutoFix

    return EnvironmentAutoFix(
        gpu_detected=gpu_detected,
        carnot_force_live_was_set=False,
        auto_fix_applied=False,
        final_env_value=None,
    )


# ---------------------------------------------------------------------------
# MicroPrecisionResult tests
# ---------------------------------------------------------------------------


class TestMicroPrecisionResult:
    """MicroPrecisionResult dataclass — all fields, all improvement directions."""

    def test_all_fields_present(self):
        """Instantiation succeeds and all fields are accessible."""
        r = _make_result("baseline", 0.0)
        assert r.model_id == "Gemma4-E4B-it"
        assert r.variant == "baseline"
        assert r.n_questions == 50
        assert r.baseline_accuracy == pytest.approx(0.60)
        assert r.variant_accuracy == pytest.approx(0.60)
        assert r.signed_improvement == pytest.approx(0.0)
        assert r.crane_detection_rate == pytest.approx(0.0)
        assert r.inference_mode == "live_gpu"

    def test_positive_signed_improvement(self):
        """CRANE_ONLY variant that improved on baseline."""
        r = _make_result("crane_only", 0.08)
        assert r.signed_improvement == pytest.approx(0.08)
        assert r.variant_accuracy > r.baseline_accuracy

    def test_negative_signed_improvement_not_clamped(self):
        """Full_stack regression — signed_improvement may be negative, never clamped."""
        r = _make_result("full_stack", -0.06)
        assert r.signed_improvement == pytest.approx(-0.06)
        assert r.variant_accuracy < r.baseline_accuracy

    def test_zero_signed_improvement(self):
        """No change — signed_improvement exactly 0.0."""
        r = _make_result("crane_only", 0.0)
        assert r.signed_improvement == pytest.approx(0.0)

    def test_baseline_crane_rate_zero(self):
        """BASELINE variant must have crane_detection_rate=0.0 (CRANE not run)."""
        r = _make_result("baseline", 0.0)
        assert r.crane_detection_rate == pytest.approx(0.0)

    def test_non_baseline_crane_rate_nonzero(self):
        """Non-baseline variants may have a non-zero crane_detection_rate."""
        r = _make_result("full_stack", 0.05)
        assert r.crane_detection_rate > 0.0

    def test_inference_mode_live_gpu(self):
        """inference_mode='live_gpu' for real GPU results."""
        r = _make_result("baseline", 0.0, inference_mode="live_gpu")
        assert r.inference_mode == "live_gpu"

    def test_inference_mode_blocked(self):
        """inference_mode='blocked' for gate-prevented results."""
        r = _make_result("baseline", 0.0, inference_mode="blocked")
        assert r.inference_mode == "blocked"


# ---------------------------------------------------------------------------
# build_micro_precision_artifact tests
# ---------------------------------------------------------------------------


class TestBuildMicroPrecisionArtifact:
    """build_micro_precision_artifact() — all verdict paths and field assertions."""

    def test_empty_results_blocked(self):
        """Empty results list → honest_verdict='blocked'."""
        a = build_micro_precision_artifact([])
        assert a["honest_verdict"] == "blocked"
        assert a["headline_result"] is None
        assert a["inference_mode"] == "blocked"
        assert a["per_model_results"] == []

    def test_schema_always_present(self):
        """schema='carnot.precision_micro.v1' in all code paths."""
        assert build_micro_precision_artifact([])["schema"] == "carnot.precision_micro.v1"
        assert build_micro_precision_artifact([_make_result("baseline", 0.0)])["schema"] == "carnot.precision_micro.v1"

    def test_simulated_mode_always_blocked(self):
        """inference_mode='simulated' → blocked verdict even with positive improvement."""
        r = _make_result("crane_only", 0.1, inference_mode="simulated")
        a = build_micro_precision_artifact([r])
        assert a["honest_verdict"] == "blocked"
        assert a["headline_result"] is None

    def test_mixed_live_simulated_blocked(self):
        """One live + one simulated → blocked (all must be live_gpu)."""
        results = [
            _make_result("baseline", 0.0, inference_mode="live_gpu"),
            _make_result("crane_only", 0.1, inference_mode="simulated"),
        ]
        a = build_micro_precision_artifact(results)
        assert a["honest_verdict"] == "blocked"

    def test_simulated_per_model_results_preserved(self):
        """Blocked artifact still includes per_model_results for audit trail."""
        r = _make_result("crane_only", 0.1, inference_mode="simulated")
        a = build_micro_precision_artifact([r])
        assert len(a["per_model_results"]) == 1

    def test_live_improvement_verdict_positive(self):
        """All live_gpu + best signed_improvement > 0 → 'live_improvement'."""
        results = [
            _make_result("baseline", 0.0),
            _make_result("crane_only", 0.07),
        ]
        a = build_micro_precision_artifact(results)
        assert a["honest_verdict"] == "live_improvement"
        assert a["inference_mode"] == "live_gpu"

    def test_live_no_improvement_verdict_zero(self):
        """All live_gpu + best signed_improvement == 0 → 'live_no_improvement'."""
        results = [
            _make_result("baseline", 0.0),
            _make_result("crane_only", 0.0),
        ]
        a = build_micro_precision_artifact(results)
        assert a["honest_verdict"] == "live_no_improvement"

    def test_live_no_improvement_verdict_negative(self):
        """All live_gpu + regression → 'live_no_improvement'."""
        results = [
            _make_result("baseline", 0.0),
            _make_result("crane_only", -0.05),
            _make_result("full_stack", -0.03),
        ]
        a = build_micro_precision_artifact(results)
        assert a["honest_verdict"] == "live_no_improvement"

    def test_headline_is_best_non_baseline(self):
        """Headline is the non-baseline variant with highest signed_improvement."""
        results = [
            _make_result("baseline", 0.0),
            _make_result("crane_only", 0.05),
            _make_result("full_stack", 0.12),
        ]
        a = build_micro_precision_artifact(results)
        assert a["headline_result"]["variant"] == "full_stack"
        assert a["headline_result"]["signed_improvement"] == pytest.approx(0.12)

    def test_headline_is_best_among_non_baseline_even_when_baseline_high(self):
        """Baseline is never selected as headline when non-baseline results exist."""
        results = [
            _make_result("baseline", 0.0),
            _make_result("crane_only", 0.03),
        ]
        a = build_micro_precision_artifact(results)
        assert a["headline_result"]["variant"] != "baseline"

    def test_headline_falls_back_to_baseline_when_no_non_baseline(self):
        """Degenerate: only baseline results → headline drawn from baselines."""
        results = [_make_result("baseline", 0.0)]
        a = build_micro_precision_artifact(results)
        assert a["headline_result"] is not None
        assert a["headline_result"]["variant"] == "baseline"

    def test_headline_result_all_fields_present(self):
        """Headline result dict contains all required keys."""
        results = [_make_result("crane_only", 0.08)]
        a = build_micro_precision_artifact(results)
        hr = a["headline_result"]
        for key in (
            "model_id",
            "variant",
            "n_questions",
            "baseline_accuracy",
            "variant_accuracy",
            "signed_improvement",
            "crane_detection_rate",
            "inference_mode",
        ):
            assert key in hr, f"Missing key in headline_result: {key}"

    def test_per_model_results_count(self):
        """per_model_results contains exactly the number of results passed in."""
        results = [
            _make_result("baseline", 0.0),
            _make_result("crane_only", 0.05),
            _make_result("full_stack", 0.09),
        ]
        a = build_micro_precision_artifact(results)
        assert len(a["per_model_results"]) == 3

    def test_per_model_results_all_variants_preserved(self):
        """per_model_results preserves all variants including baseline."""
        results = [
            _make_result("baseline", 0.0),
            _make_result("crane_only", 0.05, model_id="Qwen3.5-0.8B"),
        ]
        a = build_micro_precision_artifact(results)
        variants = {r["variant"] for r in a["per_model_results"]}
        assert "baseline" in variants
        assert "crane_only" in variants

    def test_two_models_best_selected_across_both(self):
        """With 2 models, headline is the best across ALL models, not per model."""
        results = [
            _make_result("crane_only", 0.05, model_id="Gemma4-E4B-it"),
            _make_result("crane_only", 0.12, model_id="Qwen3.5-0.8B"),
        ]
        a = build_micro_precision_artifact(results)
        assert a["headline_result"]["model_id"] == "Qwen3.5-0.8B"
        assert a["headline_result"]["signed_improvement"] == pytest.approx(0.12)


# ---------------------------------------------------------------------------
# run_experiment() tests (CI mode — no real GPU)
# ---------------------------------------------------------------------------


class TestRunExperiment:
    """run_experiment() — all code paths via mocked dependencies."""

    def _run(self, force_live: str = "0") -> dict:
        """Run experiment in CI mode with mocked dependencies."""
        import scripts.experiment_439_live_precision_micro as exp439  # noqa: PLC0415

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_root = Path(tmpdir)

            with (
                patch.object(exp439, "_REPO_ROOT", tmp_root),
                patch.object(exp439, "_autofix_result", _make_autofix()),
                patch.dict("os.environ", {"CARNOT_FORCE_LIVE": force_live}),
            ):
                artifact = exp439.run_experiment(repo_root=tmp_root)

        return artifact

    def test_ci_mode_honest_verdict_blocked(self):
        """CI mode (CARNOT_FORCE_LIVE=0): honest_verdict='blocked'."""
        artifact = self._run(force_live="0")
        assert artifact.get("honest_verdict") == "blocked"

    def test_ci_mode_required_fields_all_present(self):
        """CI mode: all REQUIRED_RESULT_FIELDS are present in the artifact."""
        artifact = self._run(force_live="0")
        for field in REQUIRED_RESULT_FIELDS:
            assert field in artifact, f"Missing required field: {field}"

    def test_ci_mode_artifact_experiment_id(self):
        """CI mode: artifact['experiment'] == 439."""
        artifact = self._run(force_live="0")
        assert artifact["experiment"] == 439

    def test_ci_mode_env_autofix_embedded(self):
        """CI mode: env_autofix block is embedded in the artifact."""
        artifact = self._run(force_live="0")
        assert "env_autofix" in artifact
        ea = artifact["env_autofix"]
        assert "gpu_detected" in ea
        assert "auto_fix_applied" in ea

    def test_ci_mode_precision_micro_schema_present(self):
        """CI mode: artifact contains carnot.precision_micro.v1 fields."""
        artifact = self._run(force_live="0")
        assert artifact.get("per_model_results") == []

    def test_artifact_written_to_disk(self):
        """run_experiment() writes the result JSON to the deliverable path."""
        import scripts.experiment_439_live_precision_micro as exp439  # noqa: PLC0415

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_root = Path(tmpdir)

            with (
                patch.object(exp439, "_REPO_ROOT", tmp_root),
                patch.object(exp439, "_autofix_result", _make_autofix()),
                patch.dict("os.environ", {"CARNOT_FORCE_LIVE": "0"}),
            ):
                exp439.run_experiment(repo_root=tmp_root)

            output_path = tmp_root / "results" / "experiment_439_live_precision_micro.json"
            assert output_path.exists(), "Output JSON was not written"
            data = json.loads(output_path.read_text())
            assert data["experiment"] == 439

    def test_ci_mode_status_blocked(self):
        """CI mode: artifact status is 'blocked'."""
        artifact = self._run(force_live="0")
        assert artifact.get("status") == "blocked"


# ---------------------------------------------------------------------------
# main() tests
# ---------------------------------------------------------------------------


class TestMain:
    """main() — calls run_experiment() inside ExperimentTimeoutWatchdog."""

    def test_main_runs_without_error(self):
        """main() completes without raising when run_experiment() succeeds."""
        import scripts.experiment_439_live_precision_micro as exp439  # noqa: PLC0415

        mock_artifact = {
            "honest_verdict": "blocked",
            "experiment": 439,
            "status": "blocked",
        }

        with (
            patch(
                "scripts.experiment_439_live_precision_micro.run_experiment",
                return_value=mock_artifact,
            ),
            patch(
                "scripts.experiment_439_live_precision_micro.get_timeout_minutes",
                return_value=1,
            ),
            patch(
                "scripts.experiment_439_live_precision_micro.ExperimentTimeoutWatchdog"
            ) as mock_watchdog,
        ):
            mock_watchdog.return_value.__enter__ = lambda s: s
            mock_watchdog.return_value.__exit__ = MagicMock(return_value=False)
            exp439.main()

    def test_main_watchdog_called_with_exp_id_439(self):
        """main() constructs ExperimentTimeoutWatchdog with experiment_id=439."""
        import scripts.experiment_439_live_precision_micro as exp439  # noqa: PLC0415

        mock_artifact = {"honest_verdict": "blocked", "experiment": 439}

        with (
            patch(
                "scripts.experiment_439_live_precision_micro.run_experiment",
                return_value=mock_artifact,
            ),
            patch(
                "scripts.experiment_439_live_precision_micro.get_timeout_minutes",
                return_value=1,
            ),
            patch(
                "scripts.experiment_439_live_precision_micro.ExperimentTimeoutWatchdog"
            ) as mock_watchdog,
        ):
            mock_watchdog.return_value.__enter__ = lambda s: s
            mock_watchdog.return_value.__exit__ = MagicMock(return_value=False)
            exp439.main()

        assert mock_watchdog.call_args[1]["experiment_id"] == 439

    def test_main_calls_run_experiment_once(self):
        """main() calls run_experiment() exactly once."""
        import scripts.experiment_439_live_precision_micro as exp439  # noqa: PLC0415

        mock_artifact = {"honest_verdict": "blocked"}

        with (
            patch(
                "scripts.experiment_439_live_precision_micro.run_experiment",
                return_value=mock_artifact,
            ) as mock_run,
            patch(
                "scripts.experiment_439_live_precision_micro.get_timeout_minutes",
                return_value=1,
            ),
            patch(
                "scripts.experiment_439_live_precision_micro.ExperimentTimeoutWatchdog"
            ) as mock_watchdog,
        ):
            mock_watchdog.return_value.__enter__ = lambda s: s
            mock_watchdog.return_value.__exit__ = MagicMock(return_value=False)
            exp439.main()

        mock_run.assert_called_once()
