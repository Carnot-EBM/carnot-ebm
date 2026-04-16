"""Tests for scripts/experiment_427_precision_live_confirmed.py.

Coverage targets (100% for new functions):

experiment_427_precision_live_confirmed.py:
  - compute_crane_detection_rate: empty list, all True, all False, mixed
  - build_exp427_artifact: experiment=427, confirmed_from, rerun, crane_detection_rate
    fields; honest_verdict and schema inherited from exp419 builder
  - main(): Exp 419 status='success' live_gpu → confirm path (copy with 427 metadata)
  - main(): Exp 419 status='partial' → re-run gate chain
  - main(): Exp 413 verdict bad → blocked (rerun path)
  - main(): Exp 413 file missing → blocked (rerun path)
  - main(): LiveGPUGate blocked (rerun path)
  - main(): check_dual_gpu_health gpu1_is_zombie → logs WARNING but continues
  - main(): check_dual_gpu_health temperature_warning → logs WARNING but continues
  - main(): setup_gpu not all_healthy → blocked
  - main(): model load fails → blocked
  - main(): success path → artifact with experiment=427, confirmed_from=419, rerun=True
  - main(): success artifact crane_detection_rate present
  - main(): success artifact all required REQUIRED_RESULT_FIELDS

Spec: REQ-BENCH-003, SCENARIO-BENCH-020
"""

from __future__ import annotations

import json
import sys
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import scripts.experiment_427_precision_live_confirmed as exp427  # noqa: E402
from carnot.pipeline.crane_extractor import CRANEExtractionGate  # noqa: E402
from carnot.pipeline.dual_gpu_health import DualGPUHealthResult  # noqa: E402
from carnot.pipeline.precision_benchmark import (  # noqa: E402
    PipelineVariant,
    PrecisionStackResult,
    compute_signed_improvement,
)
from scripts.experiment_template import (  # noqa: E402
    ExperimentTemplate,
    REQUIRED_RESULT_FIELDS,
)


# ===========================================================================
# compute_crane_detection_rate
# ===========================================================================


class TestComputeCraneDetectionRate:
    def test_empty_list_returns_zero(self):
        assert exp427.compute_crane_detection_rate([]) == 0.0

    def test_all_true_returns_one(self):
        assert exp427.compute_crane_detection_rate([True, True, True]) == pytest.approx(1.0)

    def test_all_false_returns_zero(self):
        assert exp427.compute_crane_detection_rate([False, False]) == pytest.approx(0.0)

    def test_mixed_returns_fraction(self):
        # 2 out of 4 → 0.5
        assert exp427.compute_crane_detection_rate([True, False, True, False]) == pytest.approx(0.5)

    def test_single_true(self):
        assert exp427.compute_crane_detection_rate([True]) == pytest.approx(1.0)

    def test_single_false(self):
        assert exp427.compute_crane_detection_rate([False]) == pytest.approx(0.0)

    def test_three_of_five(self):
        hits = [True, True, True, False, False]
        assert exp427.compute_crane_detection_rate(hits) == pytest.approx(0.6)


# ===========================================================================
# build_exp427_artifact
# ===========================================================================


def _make_results(inference_mode: str = "live_gpu") -> list[PrecisionStackResult]:
    results = []
    for model in ("Gemma4-E4B-it", "Qwen3.5-0.8B"):
        for variant in PipelineVariant:
            results.append(
                PrecisionStackResult(
                    model_id=model,
                    n_questions=10,
                    baseline_accuracy=0.50,
                    precision_stack_accuracy=0.55,
                    signed_improvement=compute_signed_improvement(0.50, 0.55),
                    pipeline_variant=variant,
                    inference_mode=inference_mode,
                )
            )
    return results


class TestBuildExp427Artifact:
    def test_experiment_id_is_427(self):
        artifact = exp427.build_exp427_artifact([], "live_gpu", [], rerun=True)
        assert artifact["experiment"] == 427

    def test_confirmed_from_is_419(self):
        artifact = exp427.build_exp427_artifact([], "live_gpu", [], rerun=True)
        assert artifact["confirmed_from"] == 419

    def test_rerun_true(self):
        artifact = exp427.build_exp427_artifact([], "live_gpu", [], rerun=True)
        assert artifact["rerun"] is True

    def test_rerun_false(self):
        artifact = exp427.build_exp427_artifact([], "live_gpu", [], rerun=False)
        assert artifact["rerun"] is False

    def test_crane_detection_rate_zero_for_empty_hits(self):
        artifact = exp427.build_exp427_artifact([], "live_gpu", [], rerun=True)
        assert artifact["crane_detection_rate"] == pytest.approx(0.0)

    def test_crane_detection_rate_computed(self):
        artifact = exp427.build_exp427_artifact(
            [], "live_gpu", [True, True, False, False], rerun=True
        )
        assert artifact["crane_detection_rate"] == pytest.approx(0.5)

    def test_schema_v2(self):
        artifact = exp427.build_exp427_artifact([], "live_gpu", [], rerun=True)
        assert artifact["precision_schema"] == "carnot.precision_benchmark.v2"

    def test_honest_verdict_live_improvement(self):
        results = _make_results("live_gpu")
        artifact = exp427.build_exp427_artifact(results, "live_gpu", [], rerun=True)
        assert artifact["honest_verdict"] == "live_improvement"

    def test_honest_verdict_blocked(self):
        artifact = exp427.build_exp427_artifact([], "blocked", [], rerun=True)
        assert artifact["honest_verdict"] == "blocked"

    def test_inference_mode_propagated(self):
        artifact = exp427.build_exp427_artifact([], "live_gpu", [], rerun=True)
        assert artifact["inference_mode"] == "live_gpu"


# ===========================================================================
# main() helpers
# ===========================================================================

_CI_HEALTH = DualGPUHealthResult(
    gpu0_util_pct=0.0,
    gpu1_util_pct=0.0,
    gpu0_temp_c=0.0,
    gpu1_temp_c=0.0,
    gpu0_vram_mb=0.0,
    gpu1_vram_mb=0.0,
    gpu1_is_zombie=False,
    temperature_warning=False,
    recommended_batch_size_factor=1.0,
)

_ZOMBIE_HEALTH = DualGPUHealthResult(
    gpu0_util_pct=88.0,
    gpu1_util_pct=0.0,
    gpu0_temp_c=75.0,
    gpu1_temp_c=40.0,
    gpu0_vram_mb=10000.0,
    gpu1_vram_mb=1800.0,
    gpu1_is_zombie=True,
    temperature_warning=False,
    recommended_batch_size_factor=1.0,
)

_THERMAL_HEALTH = DualGPUHealthResult(
    gpu0_util_pct=88.0,
    gpu1_util_pct=80.0,
    gpu0_temp_c=82.0,
    gpu1_temp_c=81.0,
    gpu0_vram_mb=10000.0,
    gpu1_vram_mb=10000.0,
    gpu1_is_zombie=False,
    temperature_warning=True,
    recommended_batch_size_factor=0.75,
)


def _write_exp413(tmp_path: Path, verdict: str) -> None:
    results_dir = tmp_path / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "experiment_413_env_autofix.json").write_text(
        json.dumps({"honest_verdict": verdict})
    )


def _write_exp419(tmp_path: Path, data: dict) -> None:
    results_dir = tmp_path / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "experiment_419_precision_live.json").write_text(json.dumps(data))


def _patch_repo_root(tmp_path: Path):
    return patch.object(
        sys.modules["scripts.experiment_427_precision_live_confirmed"],
        "_REPO_ROOT",
        tmp_path,
    )


def _common_rerun_patches(tmp_path: Path, gpu_health=None) -> list:
    """Return patches shared by all re-run path tests."""
    fake_model = MagicMock()
    if gpu_health is None:
        gpu_health = _CI_HEALTH
    return [
        _patch_repo_root(tmp_path),
        patch("scripts.experiment_template.ExperimentTemplate.setup"),
        patch(
            "carnot.pipeline.live_gpu_gate.LiveGPUGate.require_live_or_blocked",
            return_value=None,
        ),
        patch(
            "scripts.experiment_427_precision_live_confirmed.check_dual_gpu_health",
            return_value=gpu_health,
        ),
        patch(
            "scripts.experiment_template.ExperimentTemplate.setup_gpu",
            return_value={"all_healthy": True, "models": []},
        ),
        patch(
            "scripts.experiment_427_precision_live_confirmed._load_model_pipeline",
            return_value=fake_model,
        ),
        patch(
            "scripts.experiment_427_precision_live_confirmed.load_gsm8k_questions",
            return_value=[{"question": "q", "answer": "#### 4"}] * 2,
        ),
        patch(
            "scripts.experiment_427_precision_live_confirmed._apply_variant_with_crane",
            return_value=("resp", 0, 0),
        ),
        patch(
            "scripts.experiment_427_precision_live_confirmed._count_baseline_correct",
            return_value=1,
        ),
        patch(
            "scripts.experiment_427_precision_live_confirmed._call_model",
            return_value="#### 4",
        ),
        patch("scripts.experiment_template.ExperimentTemplate.checkpoint_save"),
    ]


def _run_main(tmp_path: Path, extra_patches: list | None = None, gpu_health=None):
    """Run main() with standard re-run patches; return written artifact."""
    written = {}

    def fake_write(tmpl, artifact):
        written["artifact"] = artifact

    patches = _common_rerun_patches(tmp_path, gpu_health=gpu_health)
    if extra_patches:
        patches.extend(extra_patches)

    with ExitStack() as stack:
        for cm in patches:
            stack.enter_context(cm)
        stack.enter_context(
            patch.object(exp427, "_write_artifact", side_effect=fake_write)
        )
        exp427.main()

    return written.get("artifact", {})


# ===========================================================================
# main() tests
# ===========================================================================


class TestMainConfirmPath:
    """Tests for the CONFIRM path (Exp 419 status='success')."""

    def test_confirm_copies_with_427_metadata(self, tmp_path):
        _write_exp419(
            tmp_path,
            {
                "status": "success",
                "inference_mode": "live_gpu",
                "honest_verdict": "live_improvement",
                "precision_schema": "carnot.precision_benchmark.v2",
                "headline_result": {"signed_improvement": 0.05},
            },
        )
        written = {}

        def fake_write(tmpl, artifact):
            written["artifact"] = artifact

        with _patch_repo_root(tmp_path), \
             patch("scripts.experiment_template.ExperimentTemplate.setup"), \
             patch.object(exp427, "_write_artifact", side_effect=fake_write):
            exp427.main()

        assert written["artifact"]["experiment"] == 427
        assert written["artifact"]["confirmed_from"] == 419
        assert written["artifact"]["rerun"] is False

    def test_confirm_no_improvement_also_confirms(self, tmp_path):
        _write_exp419(
            tmp_path,
            {
                "status": "success",
                "inference_mode": "live_gpu",
                "honest_verdict": "live_no_improvement",
            },
        )
        written = {}

        def fake_write(tmpl, artifact):
            written["artifact"] = artifact

        with _patch_repo_root(tmp_path), \
             patch("scripts.experiment_template.ExperimentTemplate.setup"), \
             patch.object(exp427, "_write_artifact", side_effect=fake_write):
            exp427.main()

        assert written["artifact"]["confirmed_from"] == 419
        assert written["artifact"]["rerun"] is False


class TestMainRerunPath:
    """Tests for the RERUN path (Exp 419 status='partial' or missing)."""

    def test_exp419_partial_triggers_rerun(self, tmp_path):
        _write_exp419(tmp_path, {"experiment": 419, "status": "partial"})
        _write_exp413(tmp_path, "auto_fix_applied")
        art = _run_main(tmp_path)
        assert art.get("rerun") is True
        assert art.get("confirmed_from") == 419

    def test_exp413_missing_writes_blocked(self, tmp_path):
        _write_exp419(tmp_path, {"status": "partial"})
        # No exp413 file
        written = {}

        def fake_write(tmpl, artifact):
            written["artifact"] = artifact

        with _patch_repo_root(tmp_path), \
             patch("scripts.experiment_template.ExperimentTemplate.setup"), \
             patch.object(exp427, "_write_artifact", side_effect=fake_write):
            exp427.main()

        assert written["artifact"]["honest_verdict"] == "blocked"

    def test_exp413_bad_verdict_writes_blocked(self, tmp_path):
        _write_exp419(tmp_path, {"status": "partial"})
        _write_exp413(tmp_path, "gpu_hardware_not_live")
        written = {}

        def fake_write(tmpl, artifact):
            written["artifact"] = artifact

        with _patch_repo_root(tmp_path), \
             patch("scripts.experiment_template.ExperimentTemplate.setup"), \
             patch.object(exp427, "_write_artifact", side_effect=fake_write):
            exp427.main()

        assert written["artifact"]["honest_verdict"] == "blocked"

    @pytest.mark.parametrize("verdict", [
        "gpu_confirmed_live",
        "auto_fix_applied",
        "gpu_detected_env_was_correct",
    ])
    def test_gate0_all_allowed_verdicts_pass(self, tmp_path, verdict):
        """All three Gate 0 allowed verdicts must proceed past the gate."""
        _write_exp419(tmp_path, {"status": "partial"})
        _write_exp413(tmp_path, verdict)
        art = _run_main(tmp_path)
        # Passed Gate 0 — reached inference loop (confirmed_from set, not just blocked)
        assert art.get("confirmed_from") == 419

    def test_live_gpu_gate_blocked(self, tmp_path):
        _write_exp419(tmp_path, {"status": "partial"})
        _write_exp413(tmp_path, "auto_fix_applied")
        written = {}

        def fake_write(tmpl, artifact):
            written["artifact"] = artifact

        with _patch_repo_root(tmp_path), \
             patch("scripts.experiment_template.ExperimentTemplate.setup"), \
             patch(
                 "carnot.pipeline.live_gpu_gate.LiveGPUGate.require_live_or_blocked",
                 return_value={"status": "blocked"},
             ), \
             patch.object(exp427, "_write_artifact", side_effect=fake_write):
            exp427.main()

        assert written["artifact"]["honest_verdict"] == "blocked"

    def test_gpu1_zombie_logs_warning_but_continues(self, tmp_path, caplog):
        import logging
        _write_exp419(tmp_path, {"status": "partial"})
        _write_exp413(tmp_path, "auto_fix_applied")

        with caplog.at_level(logging.WARNING, logger="scripts.experiment_427_precision_live_confirmed"):
            art = _run_main(tmp_path, gpu_health=_ZOMBIE_HEALTH)

        zombie_warned = any("zombie" in r.message.lower() for r in caplog.records)
        assert zombie_warned
        # Experiment should still produce an artifact (not blocked by zombie).
        assert art.get("confirmed_from") == 419

    def test_temperature_warning_logs_warning_but_continues(self, tmp_path, caplog):
        import logging
        _write_exp419(tmp_path, {"status": "partial"})
        _write_exp413(tmp_path, "auto_fix_applied")

        with caplog.at_level(logging.WARNING, logger="scripts.experiment_427_precision_live_confirmed"):
            art = _run_main(tmp_path, gpu_health=_THERMAL_HEALTH)

        thermal_warned = any("temperature" in r.message.lower() for r in caplog.records)
        assert thermal_warned
        assert art.get("confirmed_from") == 419

    def test_setup_gpu_not_healthy_writes_blocked(self, tmp_path):
        _write_exp419(tmp_path, {"status": "partial"})
        _write_exp413(tmp_path, "auto_fix_applied")
        written = {}

        def fake_write(tmpl, artifact):
            written["artifact"] = artifact

        with _patch_repo_root(tmp_path), \
             patch("scripts.experiment_template.ExperimentTemplate.setup"), \
             patch(
                 "carnot.pipeline.live_gpu_gate.LiveGPUGate.require_live_or_blocked",
                 return_value=None,
             ), \
             patch(
                 "scripts.experiment_427_precision_live_confirmed.check_dual_gpu_health",
                 return_value=_CI_HEALTH,
             ), \
             patch(
                 "scripts.experiment_template.ExperimentTemplate.setup_gpu",
                 return_value={"all_healthy": False, "models": []},
             ), \
             patch.object(exp427, "_write_artifact", side_effect=fake_write):
            exp427.main()

        assert written["artifact"]["honest_verdict"] == "blocked"
        assert "setup_gpu" in written["artifact"]["failure_reason"]

    def test_model_load_failure_writes_blocked(self, tmp_path):
        _write_exp419(tmp_path, {"status": "partial"})
        _write_exp413(tmp_path, "auto_fix_applied")
        written = {}

        def fake_write(tmpl, artifact):
            written["artifact"] = artifact

        with _patch_repo_root(tmp_path), \
             patch("scripts.experiment_template.ExperimentTemplate.setup"), \
             patch(
                 "carnot.pipeline.live_gpu_gate.LiveGPUGate.require_live_or_blocked",
                 return_value=None,
             ), \
             patch(
                 "carnot.pipeline.dual_gpu_health.check_dual_gpu_health",
                 return_value=_CI_HEALTH,
             ), \
             patch(
                 "scripts.experiment_template.ExperimentTemplate.setup_gpu",
                 return_value={"all_healthy": True, "models": []},
             ), \
             patch(
                 "scripts.experiment_427_precision_live_confirmed._load_model_pipeline",
                 side_effect=RuntimeError("GPU OOM"),
             ), \
             patch.object(exp427, "_write_artifact", side_effect=fake_write):
            exp427.main()

        assert written["artifact"]["honest_verdict"] == "blocked"
        assert "model load failed" in written["artifact"]["failure_reason"]

    def test_success_artifact_experiment_427(self, tmp_path):
        _write_exp419(tmp_path, {"status": "partial"})
        _write_exp413(tmp_path, "auto_fix_applied")
        art = _run_main(tmp_path)
        assert art["experiment"] == 427

    def test_success_artifact_confirmed_from_419(self, tmp_path):
        _write_exp419(tmp_path, {"status": "partial"})
        _write_exp413(tmp_path, "auto_fix_applied")
        art = _run_main(tmp_path)
        assert art["confirmed_from"] == 419

    def test_success_artifact_rerun_true(self, tmp_path):
        _write_exp419(tmp_path, {"status": "partial"})
        _write_exp413(tmp_path, "auto_fix_applied")
        art = _run_main(tmp_path)
        assert art["rerun"] is True

    def test_success_artifact_crane_detection_rate_present(self, tmp_path):
        _write_exp419(tmp_path, {"status": "partial"})
        _write_exp413(tmp_path, "auto_fix_applied")
        art = _run_main(tmp_path)
        assert "crane_detection_rate" in art

    def test_success_artifact_required_fields(self, tmp_path):
        _write_exp419(tmp_path, {"status": "partial"})
        _write_exp413(tmp_path, "auto_fix_applied")
        art = _run_main(tmp_path)
        for field in REQUIRED_RESULT_FIELDS:
            assert field in art, f"Missing required field: {field}"

    def test_success_artifact_inference_mode_live_gpu(self, tmp_path):
        _write_exp419(tmp_path, {"status": "partial"})
        _write_exp413(tmp_path, "auto_fix_applied")
        art = _run_main(tmp_path)
        assert art.get("inference_mode") == "live_gpu"

    def test_success_artifact_schema_v2(self, tmp_path):
        _write_exp419(tmp_path, {"status": "partial"})
        _write_exp413(tmp_path, "auto_fix_applied")
        art = _run_main(tmp_path)
        assert art.get("precision_schema") == "carnot.precision_benchmark.v2"

    def test_exp419_missing_file_triggers_rerun(self, tmp_path):
        # No exp419 file → warning → rerun path
        _write_exp413(tmp_path, "auto_fix_applied")
        art = _run_main(tmp_path)
        assert art.get("rerun") is True
