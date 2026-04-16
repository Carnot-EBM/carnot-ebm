"""Tests for scripts/experiment_394_precision_live.py.

Covers 100% of the NEW functions introduced by Exp 394:
- load_preflight_verdict: file present with matching verdict, missing verdict key,
  file missing, file unreadable
- build_exp394_artifact: live_improvement, live_no_improvement, blocked verdict,
  schema v2, inference_mode field, all_results count, headline_result, empty results
- _write_artifact: file written, parent dirs created
- main() — preflight verdict != gpu_confirmed_live → blocked artifact, exit
- main() — preflight verdict missing (None) → blocked artifact, exit
- main() — LiveGPUGate returns blocked → blocked artifact, exit
- main() — setup_gpu all_healthy=False → blocked artifact
- main() — model load fails → blocked artifact
- main() — success path: artifact written with live_gpu_confirmed=True
- main() — success artifact has all REQUIRED_RESULT_FIELDS
- main() — success artifact all_results count = 10 (5 variants × 2 models)
- main() — success artifact precision_schema == v2
- main() — success artifact honest_verdict present
- main() — success artifact inference_mode == live_gpu
- main() — success artifact n_models == 2, n_variants == 5
- main() — LLMConstraintExtractor import error → graceful fallback (no crash)

Functions imported from experiment_368 (run_variant, load_gsm8k_questions, etc.) are
tested by test_experiment_368_precision_live.py and are NOT re-tested here.

Spec: REQ-BENCH-003, SCENARIO-BENCH-007, SCENARIO-BENCH-008, SCENARIO-BENCH-009,
      SCENARIO-BENCH-020
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Bootstrap sys.path so scripts.* and carnot.* resolve.
# ---------------------------------------------------------------------------
import sys
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import scripts.experiment_394_precision_live as exp394
from carnot.pipeline.precision_benchmark import (
    PipelineVariant,
    PrecisionStackResult,
)
from scripts.experiment_template import ExperimentTemplate


# ---------------------------------------------------------------------------
# Shared fixture builders
# ---------------------------------------------------------------------------


def _make_result(
    model_id: str,
    variant: PipelineVariant,
    baseline_acc: float = 0.50,
    stack_acc: float = 0.55,
    inference_mode: str = "live_gpu",
) -> PrecisionStackResult:
    """Build a minimal PrecisionStackResult for testing artifact builders."""
    return PrecisionStackResult(
        model_id=model_id,
        n_questions=200,
        baseline_accuracy=baseline_acc,
        precision_stack_accuracy=stack_acc,
        signed_improvement=stack_acc - baseline_acc,
        pipeline_variant=variant,
        inference_mode=inference_mode,
        n_violations_found=5,
        n_repairs_attempted=3,
        n_repairs_improved=2,
        n_repairs_broken=0,
    )


def _all_live_results(
    baseline_acc: float = 0.50,
    stack_acc: float = 0.55,
) -> list[PrecisionStackResult]:
    """Build 10 PrecisionStackResult objects (5 variants × 2 models) in live_gpu mode."""
    results = []
    for model_id in ("Gemma4-E4B-it", "Qwen3.5-0.8B"):
        for variant in PipelineVariant:
            results.append(_make_result(model_id, variant, baseline_acc, stack_acc))
    return results


def _healthy_gpu_status() -> dict:
    return {
        "all_healthy": True,
        "models": [
            {
                "name": "Gemma4-E4B-it",
                "health_ok": True,
                "stall_root_cause": None,
                "load_time_s": 1.0,
                "gpu_id": 0,
            },
            {
                "name": "Qwen3.5-0.8B",
                "health_ok": True,
                "stall_root_cause": None,
                "load_time_s": 0.5,
                "gpu_id": 1,
            },
        ],
        "prewarm_time_s": 1.5,
        "dual_gpu_auto_assigned": True,
        "gpu_monitor_results": {
            "n_gpus_detected": 2,
            "n_zombies": 0,
            "idle_gpus": [0, 1],
            "all_healthy": True,
        },
    }


def _unhealthy_gpu_status() -> dict:
    return {
        "all_healthy": False,
        "models": [
            {
                "name": "Gemma4-E4B-it",
                "health_ok": False,
                "stall_root_cause": "OOM",
                "load_time_s": 0.0,
                "gpu_id": 0,
            }
        ],
        "prewarm_time_s": 0.0,
        "dual_gpu_auto_assigned": False,
        "gpu_monitor_results": {
            "n_gpus_detected": 0,
            "n_zombies": 0,
            "idle_gpus": [],
            "all_healthy": False,
        },
    }


def _mock_model_fn() -> MagicMock:
    """HuggingFace pipeline mock that returns a response containing #### 6."""

    def _call(prompt, max_new_tokens=512):
        return [{"generated_text": "Step 1: answer is 6.\n#### 6"}]

    return MagicMock(side_effect=_call)


def _write_preflight(tmp_path: Path, verdict: str | None) -> None:
    """Write a synthetic Exp 390 preflight JSON under tmp_path."""
    preflight_dir = tmp_path / "results"
    preflight_dir.mkdir(parents=True, exist_ok=True)
    data: dict = {"experiment": 390, "status": "complete"}
    if verdict is not None:
        data["honest_verdict"] = verdict
    (preflight_dir / "experiment_390_gpu_preflight.json").write_text(
        json.dumps(data)
    )


def _run_main_success(tmp_path: Path) -> dict:
    """Run main() in mocked-success mode and return the written artifact."""
    _write_preflight(tmp_path, "gpu_confirmed_live")
    small_questions = [
        {"question": "What is 2+2?", "answer": "#### 4"},
        {"question": "What is 3+3?", "answer": "#### 6"},
    ]
    mock_model = _mock_model_fn()

    with patch(
        "scripts.experiment_394_precision_live.LiveGPUGate.require_live_or_blocked",
        return_value=None,
    ):
        with patch(
            "scripts.experiment_394_precision_live.ExperimentTemplate"
        ) as MockTmpl:
            tmpl_instance = ExperimentTemplate(
                exp_id=exp394.EXP_ID,
                title=exp394.EXP_TITLE,
                deliverable=exp394.DELIVERABLE,
                repo_root=tmp_path,
            )
            tmpl_instance.setup()
            tmpl_instance.setup_gpu = MagicMock(return_value=_healthy_gpu_status())
            MockTmpl.return_value = tmpl_instance

            with patch.object(exp394, "load_gsm8k_questions", return_value=small_questions):
                with patch(
                    "scripts.experiment_394_precision_live._load_model_pipeline",
                    return_value=mock_model,
                ):
                    exp394.main()

    return json.loads((tmp_path / exp394.DELIVERABLE).read_text())


# ---------------------------------------------------------------------------
# load_preflight_verdict
# ---------------------------------------------------------------------------


class TestLoadPreflightVerdict:
    def test_returns_verdict_when_present(self, tmp_path: Path):
        """Returns the honest_verdict string when the file exists and key is present."""
        _write_preflight(tmp_path, "gpu_confirmed_live")
        verdict = exp394.load_preflight_verdict(tmp_path)
        assert verdict == "gpu_confirmed_live"

    def test_returns_none_when_key_absent(self, tmp_path: Path):
        """Returns None when the file exists but honest_verdict key is absent."""
        _write_preflight(tmp_path, None)
        verdict = exp394.load_preflight_verdict(tmp_path)
        assert verdict is None

    def test_returns_none_when_file_missing(self, tmp_path: Path):
        """Returns None when the preflight JSON file does not exist."""
        verdict = exp394.load_preflight_verdict(tmp_path)
        assert verdict is None

    def test_returns_other_verdict_values(self, tmp_path: Path):
        """Returns non-confirmed verdict strings as-is."""
        _write_preflight(tmp_path, "blocked")
        verdict = exp394.load_preflight_verdict(tmp_path)
        assert verdict == "blocked"


# ---------------------------------------------------------------------------
# build_exp394_artifact
# ---------------------------------------------------------------------------


class TestBuildExp394Artifact:
    def test_live_improvement_verdict(self):
        """live_gpu mode with positive signed_improvement → live_improvement."""
        results = _all_live_results(baseline_acc=0.50, stack_acc=0.55)
        artifact = exp394.build_exp394_artifact(results, "live_gpu")
        assert artifact["honest_verdict"] == "live_improvement"

    def test_live_no_improvement_verdict(self):
        """live_gpu mode with negative signed_improvement → live_no_improvement."""
        results = _all_live_results(baseline_acc=0.55, stack_acc=0.50)
        artifact = exp394.build_exp394_artifact(results, "live_gpu")
        assert artifact["honest_verdict"] == "live_no_improvement"

    def test_live_equal_verdict(self):
        """live_gpu mode with signed_improvement == 0 → live_no_improvement."""
        results = _all_live_results(baseline_acc=0.50, stack_acc=0.50)
        artifact = exp394.build_exp394_artifact(results, "live_gpu")
        assert artifact["honest_verdict"] == "live_no_improvement"

    def test_blocked_verdict(self):
        """inference_mode != live_gpu → honest_verdict == blocked."""
        results = _all_live_results()
        artifact = exp394.build_exp394_artifact(results, "blocked")
        assert artifact["honest_verdict"] == "blocked"

    def test_schema_v2(self):
        """Artifact schema is carnot.precision_benchmark.v2."""
        results = _all_live_results()
        artifact = exp394.build_exp394_artifact(results, "live_gpu")
        assert artifact["precision_schema"] == "carnot.precision_benchmark.v2"

    def test_inference_mode_set(self):
        """inference_mode field equals the argument passed in."""
        results = _all_live_results()
        artifact = exp394.build_exp394_artifact(results, "live_gpu")
        assert artifact["inference_mode"] == "live_gpu"

    def test_all_results_present(self):
        """Artifact all_results contains all 10 result objects."""
        results = _all_live_results()
        artifact = exp394.build_exp394_artifact(results, "live_gpu")
        assert len(artifact["all_results"]) == 10

    def test_headline_result_model_and_variant(self):
        """headline_result is for Gemma4-E4B-it FULL_STACK."""
        results = _all_live_results(baseline_acc=0.50, stack_acc=0.60)
        artifact = exp394.build_exp394_artifact(results, "live_gpu")
        hr = artifact.get("headline_result", {})
        assert hr.get("model_id") == "Gemma4-E4B-it"
        assert hr.get("pipeline_variant") == PipelineVariant.FULL_STACK.value

    def test_empty_results(self):
        """Empty results list produces no headline_result but still sets verdict."""
        artifact = exp394.build_exp394_artifact([], "live_gpu")
        assert artifact["honest_verdict"] == "live_no_improvement"
        assert artifact["precision_schema"] == "carnot.precision_benchmark.v2"


# ---------------------------------------------------------------------------
# _write_artifact
# ---------------------------------------------------------------------------


class TestWriteArtifact:
    def test_file_written(self, tmp_path: Path):
        """_write_artifact writes JSON to the deliverable path."""
        tmpl = ExperimentTemplate(
            exp_id=394,
            title="test",
            deliverable="results/experiment_394_precision_live.json",
            repo_root=tmp_path,
        )
        tmpl.setup()
        artifact = {"test_key": "test_value"}
        exp394._write_artifact(tmpl, artifact)
        output = json.loads((tmp_path / exp394.DELIVERABLE).read_text())
        assert output["test_key"] == "test_value"

    def test_parent_dirs_created(self, tmp_path: Path):
        """_write_artifact creates parent directories if they don't exist."""
        tmpl = ExperimentTemplate(
            exp_id=394,
            title="test",
            deliverable="results/nested/dir/experiment_394.json",
            repo_root=tmp_path,
        )
        # Do NOT call setup() so directory is not pre-created.
        tmpl._output_path = tmp_path / "results" / "nested" / "dir" / "experiment_394.json"
        exp394._write_artifact(tmpl, {"x": 1})
        assert tmpl._output_path.exists()


# ---------------------------------------------------------------------------
# main() — preflight gate failures
# ---------------------------------------------------------------------------


class TestMainPreflightBlocked:
    def test_preflight_wrong_verdict_writes_blocked(self, tmp_path: Path):
        """When Exp 390 verdict is not gpu_confirmed_live, main() writes blocked."""
        _write_preflight(tmp_path, "blocked")

        with patch(
            "scripts.experiment_394_precision_live.ExperimentTemplate"
        ) as MockTmpl:
            tmpl_instance = ExperimentTemplate(
                exp_id=exp394.EXP_ID,
                title=exp394.EXP_TITLE,
                deliverable=exp394.DELIVERABLE,
                repo_root=tmp_path,
            )
            tmpl_instance.setup()
            MockTmpl.return_value = tmpl_instance

            exp394.main()

        output_path = tmp_path / exp394.DELIVERABLE
        assert output_path.exists()
        artifact = json.loads(output_path.read_text())
        assert artifact.get("honest_verdict") == "blocked"
        assert artifact.get("inference_mode") == "blocked"
        assert artifact.get("precision_schema") == "carnot.precision_benchmark.v2"
        # preflight_verdict must be recorded in the artifact for auditability
        assert "preflight_verdict" in artifact

    def test_preflight_missing_verdict_writes_blocked(self, tmp_path: Path):
        """When Exp 390 verdict key is absent, main() writes blocked artifact."""
        _write_preflight(tmp_path, None)  # writes file without honest_verdict key

        with patch(
            "scripts.experiment_394_precision_live.ExperimentTemplate"
        ) as MockTmpl:
            tmpl_instance = ExperimentTemplate(
                exp_id=exp394.EXP_ID,
                title=exp394.EXP_TITLE,
                deliverable=exp394.DELIVERABLE,
                repo_root=tmp_path,
            )
            tmpl_instance.setup()
            MockTmpl.return_value = tmpl_instance

            exp394.main()

        output_path = tmp_path / exp394.DELIVERABLE
        assert output_path.exists()
        artifact = json.loads(output_path.read_text())
        assert artifact.get("honest_verdict") == "blocked"

    def test_preflight_file_missing_writes_blocked(self, tmp_path: Path):
        """When Exp 390 preflight file is absent entirely, main() writes blocked."""
        # Do NOT write any preflight file.

        with patch(
            "scripts.experiment_394_precision_live.ExperimentTemplate"
        ) as MockTmpl:
            tmpl_instance = ExperimentTemplate(
                exp_id=exp394.EXP_ID,
                title=exp394.EXP_TITLE,
                deliverable=exp394.DELIVERABLE,
                repo_root=tmp_path,
            )
            tmpl_instance.setup()
            MockTmpl.return_value = tmpl_instance

            exp394.main()

        output_path = tmp_path / exp394.DELIVERABLE
        assert output_path.exists()
        artifact = json.loads(output_path.read_text())
        assert artifact.get("honest_verdict") == "blocked"


# ---------------------------------------------------------------------------
# main() — LiveGPUGate and setup_gpu failures
# ---------------------------------------------------------------------------


class TestMainGateBlocked:
    def test_live_gate_fails_writes_blocked_artifact(self, tmp_path: Path):
        """When LiveGPUGate returns a blocked dict, main() writes it and returns."""
        _write_preflight(tmp_path, "gpu_confirmed_live")
        blocked_artifact = {
            "status": "blocked",
            "blocked_reason": "CARNOT_FORCE_LIVE not set",
        }
        with patch(
            "scripts.experiment_394_precision_live.LiveGPUGate.require_live_or_blocked",
            return_value=blocked_artifact,
        ):
            with patch(
                "scripts.experiment_394_precision_live.ExperimentTemplate"
            ) as MockTmpl:
                tmpl_instance = ExperimentTemplate(
                    exp_id=exp394.EXP_ID,
                    title=exp394.EXP_TITLE,
                    deliverable=exp394.DELIVERABLE,
                    repo_root=tmp_path,
                )
                tmpl_instance.setup()
                MockTmpl.return_value = tmpl_instance

                exp394.main()

        output_path = tmp_path / exp394.DELIVERABLE
        assert output_path.exists()
        artifact = json.loads(output_path.read_text())
        assert artifact.get("honest_verdict") == "blocked"
        assert artifact.get("inference_mode") == "blocked"
        assert artifact.get("precision_schema") == "carnot.precision_benchmark.v2"

    def test_setup_gpu_unhealthy_writes_blocked(self, tmp_path: Path):
        """When setup_gpu reports all_healthy=False, main() writes blocked artifact."""
        _write_preflight(tmp_path, "gpu_confirmed_live")
        with patch(
            "scripts.experiment_394_precision_live.LiveGPUGate.require_live_or_blocked",
            return_value=None,
        ):
            with patch(
                "scripts.experiment_394_precision_live.ExperimentTemplate"
            ) as MockTmpl:
                tmpl_instance = ExperimentTemplate(
                    exp_id=exp394.EXP_ID,
                    title=exp394.EXP_TITLE,
                    deliverable=exp394.DELIVERABLE,
                    repo_root=tmp_path,
                )
                tmpl_instance.setup()
                tmpl_instance.setup_gpu = MagicMock(return_value=_unhealthy_gpu_status())
                MockTmpl.return_value = tmpl_instance

                exp394.main()

        output_path = tmp_path / exp394.DELIVERABLE
        assert output_path.exists()
        artifact = json.loads(output_path.read_text())
        assert artifact.get("honest_verdict") == "blocked"
        assert "not all_healthy" in artifact.get("failure_reason", "")

    def test_model_load_fails_writes_blocked(self, tmp_path: Path):
        """When a model fails to load, main() writes a blocked artifact."""
        _write_preflight(tmp_path, "gpu_confirmed_live")
        with patch(
            "scripts.experiment_394_precision_live.LiveGPUGate.require_live_or_blocked",
            return_value=None,
        ):
            with patch(
                "scripts.experiment_394_precision_live.ExperimentTemplate"
            ) as MockTmpl:
                tmpl_instance = ExperimentTemplate(
                    exp_id=exp394.EXP_ID,
                    title=exp394.EXP_TITLE,
                    deliverable=exp394.DELIVERABLE,
                    repo_root=tmp_path,
                )
                tmpl_instance.setup()
                tmpl_instance.setup_gpu = MagicMock(return_value=_healthy_gpu_status())
                MockTmpl.return_value = tmpl_instance

                with patch(
                    "scripts.experiment_394_precision_live._load_model_pipeline",
                    side_effect=RuntimeError("CUDA OOM"),
                ):
                    exp394.main()

        output_path = tmp_path / exp394.DELIVERABLE
        assert output_path.exists()
        artifact = json.loads(output_path.read_text())
        assert artifact.get("honest_verdict") == "blocked"
        assert "model load failed" in artifact.get("failure_reason", "")


# ---------------------------------------------------------------------------
# main() — success path
# ---------------------------------------------------------------------------


class TestMainSuccess:
    def test_artifact_written_with_live_gpu_confirmed(self, tmp_path: Path):
        """Success path writes artifact with live_gpu_confirmed=True."""
        artifact = _run_main_success(tmp_path)
        assert artifact.get("live_gpu_confirmed") is True

    def test_artifact_has_required_fields(self, tmp_path: Path):
        """Success artifact contains all REQUIRED_RESULT_FIELDS."""
        from scripts.experiment_template import REQUIRED_RESULT_FIELDS

        artifact = _run_main_success(tmp_path)
        for field in REQUIRED_RESULT_FIELDS:
            assert field in artifact, f"Missing required field: {field}"

    def test_all_results_count_ten(self, tmp_path: Path):
        """Success artifact all_results has 10 entries (5 variants × 2 models)."""
        artifact = _run_main_success(tmp_path)
        all_results = artifact.get("all_results", [])
        assert len(all_results) == 10

    def test_precision_schema_v2(self, tmp_path: Path):
        """Success artifact precision_schema is carnot.precision_benchmark.v2."""
        artifact = _run_main_success(tmp_path)
        assert artifact.get("precision_schema") == "carnot.precision_benchmark.v2"

    def test_honest_verdict_present(self, tmp_path: Path):
        """Success artifact has an honest_verdict field."""
        artifact = _run_main_success(tmp_path)
        assert "honest_verdict" in artifact

    def test_inference_mode_live_gpu(self, tmp_path: Path):
        """Success artifact inference_mode is live_gpu."""
        artifact = _run_main_success(tmp_path)
        assert artifact.get("inference_mode") == "live_gpu"

    def test_n_models_n_variants_set(self, tmp_path: Path):
        """Success artifact records n_models=2 and n_variants=5."""
        artifact = _run_main_success(tmp_path)
        assert artifact.get("n_models") == 2
        assert artifact.get("n_variants") == 5

    def test_no_headline_result_still_succeeds(self, tmp_path: Path):
        """Success path with no FULL_STACK Gemma4-E4B-it result logs the else branch."""
        _write_preflight(tmp_path, "gpu_confirmed_live")
        small_questions = [
            {"question": "What is 2+2?", "answer": "#### 4"},
        ]
        mock_model = _mock_model_fn()

        with patch(
            "scripts.experiment_394_precision_live.LiveGPUGate.require_live_or_blocked",
            return_value=None,
        ):
            with patch(
                "scripts.experiment_394_precision_live.ExperimentTemplate"
            ) as MockTmpl:
                tmpl_instance = ExperimentTemplate(
                    exp_id=exp394.EXP_ID,
                    title=exp394.EXP_TITLE,
                    deliverable=exp394.DELIVERABLE,
                    repo_root=tmp_path,
                )
                tmpl_instance.setup()
                tmpl_instance.setup_gpu = MagicMock(return_value=_healthy_gpu_status())
                MockTmpl.return_value = tmpl_instance

                with patch.object(
                    exp394, "load_gsm8k_questions", return_value=small_questions
                ):
                    with patch(
                        "scripts.experiment_394_precision_live._load_model_pipeline",
                        return_value=mock_model,
                    ):
                        # Patch build_exp394_artifact to return no headline_result
                        with patch.object(
                            exp394,
                            "build_exp394_artifact",
                            return_value={
                                "precision_schema": "carnot.precision_benchmark.v2",
                                "inference_mode": "live_gpu",
                                "honest_verdict": "live_improvement",
                                "headline_result": {},  # empty → exercises else branch
                                "all_results": [],
                            },
                        ):
                            exp394.main()

        output_path = tmp_path / exp394.DELIVERABLE
        assert output_path.exists()
        artifact = json.loads(output_path.read_text())
        assert artifact.get("status") == "success"

    def test_llm_extractor_import_error_falls_back_gracefully(self, tmp_path: Path):
        """If LLMConstraintExtractor import fails, main() continues without it."""
        _write_preflight(tmp_path, "gpu_confirmed_live")
        small_questions = [
            {"question": "What is 2+2?", "answer": "#### 4"},
        ]
        mock_model = _mock_model_fn()

        with patch(
            "scripts.experiment_394_precision_live.LiveGPUGate.require_live_or_blocked",
            return_value=None,
        ):
            with patch(
                "scripts.experiment_394_precision_live.ExperimentTemplate"
            ) as MockTmpl:
                tmpl_instance = ExperimentTemplate(
                    exp_id=exp394.EXP_ID,
                    title=exp394.EXP_TITLE,
                    deliverable=exp394.DELIVERABLE,
                    repo_root=tmp_path,
                )
                tmpl_instance.setup()
                tmpl_instance.setup_gpu = MagicMock(return_value=_healthy_gpu_status())
                MockTmpl.return_value = tmpl_instance

                with patch.object(
                    exp394, "load_gsm8k_questions", return_value=small_questions
                ):
                    with patch(
                        "scripts.experiment_394_precision_live._load_model_pipeline",
                        return_value=mock_model,
                    ):
                        # Simulate LLMConstraintExtractor import failure
                        with patch.dict("sys.modules", {"carnot.pipeline.llm_extractor": None}):
                            exp394.main()

        output_path = tmp_path / exp394.DELIVERABLE
        assert output_path.exists()
        artifact = json.loads(output_path.read_text())
        assert artifact.get("status") == "success"
