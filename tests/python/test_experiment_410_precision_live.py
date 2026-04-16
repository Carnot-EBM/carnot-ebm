"""Tests for scripts/experiment_410_precision_live.py.

Covers 100% of the NEW functions introduced by Exp 410:
- load_preflight_verdict: file present, missing, malformed, no field
- build_exp410_artifact: live_improvement, live_no_improvement, blocked, schema v2
- _write_artifact: file written, parent dirs created
- main() — preflight blocked (honest_verdict != gpu_confirmed_live) → blocked artifact, no inference
- main() — preflight file missing → blocked artifact
- main() — LiveGPUGate returns blocked → blocked artifact
- main() — setup_gpu all_healthy=False → blocked artifact
- main() — model load fails → blocked artifact
- main() — CRANE import fails, LLMExtractor wired → success
- main() — both CRANE and LLM extractor import fail → success (graceful)
- main() — success path writes artifact with live_gpu_confirmed=True
- main() — success artifact has all REQUIRED_RESULT_FIELDS
- main() — success artifact all_results count = 10 (5 variants × 2 models)
- main() — success artifact precision_schema == v2
- main() — success artifact honest_verdict present

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

import scripts.experiment_410_precision_live as exp410
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
    baseline_acc: float = 0.50, stack_acc: float = 0.55
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


def _write_preflight(tmp_path: Path, verdict: str) -> None:
    """Write a minimal Exp 404 preflight JSON with the given verdict."""
    preflight_path = tmp_path / exp410.EXP_404_PREFLIGHT
    preflight_path.parent.mkdir(parents=True, exist_ok=True)
    preflight_path.write_text(json.dumps({"honest_verdict": verdict}))


def _run_main_success(tmp_path: Path) -> dict:
    """Run main() in mocked-success mode and return the written artifact."""
    _write_preflight(tmp_path, "gpu_confirmed_live")

    small_questions = [
        {"question": "What is 2+2?", "answer": "#### 4"},
        {"question": "What is 3+3?", "answer": "#### 6"},
    ]
    mock_model = _mock_model_fn()

    with patch.object(exp410, "load_preflight_verdict", return_value="gpu_confirmed_live"):
        with patch(
            "scripts.experiment_410_precision_live.LiveGPUGate.require_live_or_blocked",
            return_value=None,
        ):
            with patch(
                "scripts.experiment_410_precision_live.ExperimentTemplate"
            ) as MockTmpl:
                tmpl_instance = ExperimentTemplate(
                    exp_id=exp410.EXP_ID,
                    title=exp410.EXP_TITLE,
                    deliverable=exp410.DELIVERABLE,
                    repo_root=tmp_path,
                )
                tmpl_instance.setup()
                tmpl_instance.setup_gpu = MagicMock(return_value=_healthy_gpu_status())
                MockTmpl.return_value = tmpl_instance

                with patch.object(
                    exp410, "load_gsm8k_questions", return_value=small_questions
                ):
                    with patch(
                        "scripts.experiment_410_precision_live._load_model_pipeline",
                        return_value=mock_model,
                    ):
                        # Suppress CRANE and LLM extractor imports (not needed for success path)
                        with patch.dict(
                            "sys.modules",
                            {
                                "carnot.pipeline.crane_extractor": None,
                                "carnot.pipeline.llm_extractor": None,
                            },
                        ):
                            exp410.main()

    return json.loads((tmp_path / exp410.DELIVERABLE).read_text())


# ---------------------------------------------------------------------------
# load_preflight_verdict
# ---------------------------------------------------------------------------


class TestLoadPreflightVerdict:
    def test_reads_verdict_from_file(self, tmp_path: Path):
        """Returns the honest_verdict string when file is present and valid."""
        _write_preflight(tmp_path, "gpu_confirmed_live")
        with patch.object(exp410, "_REPO_ROOT", tmp_path):
            verdict = exp410.load_preflight_verdict(tmp_path)
        assert verdict == "gpu_confirmed_live"

    def test_returns_env_not_propagating(self, tmp_path: Path):
        """Returns 'env_not_propagating' for that specific verdict value."""
        _write_preflight(tmp_path, "env_not_propagating")
        verdict = exp410.load_preflight_verdict(tmp_path)
        assert verdict == "env_not_propagating"

    def test_file_missing_returns_missing(self, tmp_path: Path):
        """Returns 'missing' when the preflight file does not exist."""
        verdict = exp410.load_preflight_verdict(tmp_path)
        assert verdict == "missing"

    def test_malformed_json_returns_missing(self, tmp_path: Path):
        """Returns 'missing' when the preflight file contains invalid JSON."""
        preflight_path = tmp_path / exp410.EXP_404_PREFLIGHT
        preflight_path.parent.mkdir(parents=True, exist_ok=True)
        preflight_path.write_text("{not valid json")
        verdict = exp410.load_preflight_verdict(tmp_path)
        assert verdict == "missing"

    def test_no_honest_verdict_field_returns_missing(self, tmp_path: Path):
        """Returns 'missing' when JSON is valid but has no 'honest_verdict' key."""
        preflight_path = tmp_path / exp410.EXP_404_PREFLIGHT
        preflight_path.parent.mkdir(parents=True, exist_ok=True)
        preflight_path.write_text(json.dumps({"status": "success"}))
        verdict = exp410.load_preflight_verdict(tmp_path)
        assert verdict == "missing"


# ---------------------------------------------------------------------------
# build_exp410_artifact
# ---------------------------------------------------------------------------


class TestBuildExp410Artifact:
    def test_live_improvement_verdict(self):
        """live_gpu with positive signed_improvement → live_improvement."""
        results = _all_live_results(baseline_acc=0.50, stack_acc=0.55)
        artifact = exp410.build_exp410_artifact(results, "live_gpu")
        assert artifact["honest_verdict"] == "live_improvement"

    def test_live_no_improvement_verdict(self):
        """live_gpu with negative signed_improvement → live_no_improvement."""
        results = _all_live_results(baseline_acc=0.55, stack_acc=0.50)
        artifact = exp410.build_exp410_artifact(results, "live_gpu")
        assert artifact["honest_verdict"] == "live_no_improvement"

    def test_live_equal_verdict(self):
        """live_gpu with signed_improvement == 0 → live_no_improvement."""
        results = _all_live_results(baseline_acc=0.50, stack_acc=0.50)
        artifact = exp410.build_exp410_artifact(results, "live_gpu")
        assert artifact["honest_verdict"] == "live_no_improvement"

    def test_blocked_verdict(self):
        """inference_mode != live_gpu → honest_verdict == blocked."""
        results = _all_live_results()
        artifact = exp410.build_exp410_artifact(results, "blocked")
        assert artifact["honest_verdict"] == "blocked"

    def test_schema_v2(self):
        """Artifact precision_schema is carnot.precision_benchmark.v2."""
        results = _all_live_results()
        artifact = exp410.build_exp410_artifact(results, "live_gpu")
        assert artifact["precision_schema"] == "carnot.precision_benchmark.v2"

    def test_inference_mode_set(self):
        """inference_mode field equals the argument passed in."""
        results = _all_live_results()
        artifact = exp410.build_exp410_artifact(results, "live_gpu")
        assert artifact["inference_mode"] == "live_gpu"

    def test_all_results_present(self):
        """Artifact all_results contains all 10 result objects."""
        results = _all_live_results()
        artifact = exp410.build_exp410_artifact(results, "live_gpu")
        assert len(artifact["all_results"]) == 10

    def test_headline_result_model(self):
        """headline_result is for Gemma4-E4B-it FULL_STACK."""
        results = _all_live_results(baseline_acc=0.50, stack_acc=0.60)
        artifact = exp410.build_exp410_artifact(results, "live_gpu")
        hr = artifact.get("headline_result", {})
        assert hr.get("model_id") == "Gemma4-E4B-it"
        assert hr.get("pipeline_variant") == PipelineVariant.FULL_STACK.value

    def test_empty_results(self):
        """Empty results list produces no headline_result but still sets schema and verdict."""
        artifact = exp410.build_exp410_artifact([], "live_gpu")
        assert artifact["honest_verdict"] == "live_no_improvement"
        assert artifact["precision_schema"] == "carnot.precision_benchmark.v2"


# ---------------------------------------------------------------------------
# _write_artifact
# ---------------------------------------------------------------------------


class TestWriteArtifact:
    def test_file_written(self, tmp_path: Path):
        """_write_artifact writes JSON to the deliverable path."""
        tmpl = ExperimentTemplate(
            exp_id=410,
            title="test",
            deliverable="results/experiment_410_precision_live.json",
            repo_root=tmp_path,
        )
        tmpl.setup()
        artifact = {"test_key": "test_value"}
        exp410._write_artifact(tmpl, artifact)
        output = json.loads((tmp_path / exp410.DELIVERABLE).read_text())
        assert output["test_key"] == "test_value"

    def test_parent_dirs_created(self, tmp_path: Path):
        """_write_artifact creates parent directories if they don't exist."""
        tmpl = ExperimentTemplate(
            exp_id=410,
            title="test",
            deliverable="results/nested/dir/experiment_410.json",
            repo_root=tmp_path,
        )
        # Do NOT call setup() so the directory is not pre-created.
        tmpl._output_path = tmp_path / "results" / "nested" / "dir" / "experiment_410.json"
        exp410._write_artifact(tmpl, {"x": 1})
        assert tmpl._output_path.exists()


# ---------------------------------------------------------------------------
# main() — preflight blocked paths
# ---------------------------------------------------------------------------


class TestMainPreflightBlocked:
    def _run_main_with_verdict(self, verdict: str, tmp_path: Path) -> dict:
        """Run main() with a specific preflight verdict and return the artifact."""
        with patch.object(exp410, "load_preflight_verdict", return_value=verdict):
            with patch(
                "scripts.experiment_410_precision_live.ExperimentTemplate"
            ) as MockTmpl:
                tmpl_instance = ExperimentTemplate(
                    exp_id=exp410.EXP_ID,
                    title=exp410.EXP_TITLE,
                    deliverable=exp410.DELIVERABLE,
                    repo_root=tmp_path,
                )
                tmpl_instance.setup()
                MockTmpl.return_value = tmpl_instance
                exp410.main()

        return json.loads((tmp_path / exp410.DELIVERABLE).read_text())

    def test_env_not_propagating_writes_blocked(self, tmp_path: Path):
        """honest_verdict='env_not_propagating' (Exp 404 actual result) → blocked."""
        artifact = self._run_main_with_verdict("env_not_propagating", tmp_path)
        assert artifact.get("honest_verdict") == "blocked"
        assert artifact.get("inference_mode") == "blocked"
        assert artifact.get("precision_schema") == "carnot.precision_benchmark.v2"

    def test_blocked_verdict_writes_blocked(self, tmp_path: Path):
        """honest_verdict='blocked' → blocked artifact."""
        artifact = self._run_main_with_verdict("blocked", tmp_path)
        assert artifact.get("honest_verdict") == "blocked"

    def test_missing_preflight_writes_blocked(self, tmp_path: Path):
        """honest_verdict='missing' (file not found) → blocked artifact."""
        artifact = self._run_main_with_verdict("missing", tmp_path)
        assert artifact.get("honest_verdict") == "blocked"

    def test_blocked_artifact_has_preflight_verdict_field(self, tmp_path: Path):
        """Blocked artifact records the actual preflight_verdict value."""
        artifact = self._run_main_with_verdict("env_not_propagating", tmp_path)
        assert artifact.get("preflight_verdict") == "env_not_propagating"

    def test_blocked_artifact_has_failure_reason(self, tmp_path: Path):
        """Blocked artifact includes failure_reason string."""
        artifact = self._run_main_with_verdict("env_not_propagating", tmp_path)
        assert "preflight blocked" in artifact.get("failure_reason", "")

    def test_no_inference_runs_when_preflight_blocked(self, tmp_path: Path):
        """LiveGPUGate is NOT called when preflight verdict blocks execution."""
        with patch.object(exp410, "load_preflight_verdict", return_value="env_not_propagating"):
            with patch(
                "scripts.experiment_410_precision_live.LiveGPUGate.require_live_or_blocked"
            ) as mock_gate:
                with patch(
                    "scripts.experiment_410_precision_live.ExperimentTemplate"
                ) as MockTmpl:
                    tmpl_instance = ExperimentTemplate(
                        exp_id=exp410.EXP_ID,
                        title=exp410.EXP_TITLE,
                        deliverable=exp410.DELIVERABLE,
                        repo_root=tmp_path,
                    )
                    tmpl_instance.setup()
                    MockTmpl.return_value = tmpl_instance
                    exp410.main()

        mock_gate.assert_not_called()


# ---------------------------------------------------------------------------
# main() — LiveGPUGate and GPU setup blocked paths
# ---------------------------------------------------------------------------


class TestMainGPUBlocked:
    def test_live_gate_fails_writes_blocked_artifact(self, tmp_path: Path):
        """When LiveGPUGate returns a blocked dict, main() writes it and returns."""
        blocked_artifact = {
            "status": "blocked",
            "blocked_reason": "CARNOT_FORCE_LIVE not set",
        }
        with patch.object(exp410, "load_preflight_verdict", return_value="gpu_confirmed_live"):
            with patch(
                "scripts.experiment_410_precision_live.LiveGPUGate.require_live_or_blocked",
                return_value=blocked_artifact,
            ):
                with patch(
                    "scripts.experiment_410_precision_live.ExperimentTemplate"
                ) as MockTmpl:
                    tmpl_instance = ExperimentTemplate(
                        exp_id=exp410.EXP_ID,
                        title=exp410.EXP_TITLE,
                        deliverable=exp410.DELIVERABLE,
                        repo_root=tmp_path,
                    )
                    tmpl_instance.setup()
                    MockTmpl.return_value = tmpl_instance
                    exp410.main()

        output_path = tmp_path / exp410.DELIVERABLE
        assert output_path.exists()
        artifact = json.loads(output_path.read_text())
        assert artifact.get("honest_verdict") == "blocked"
        assert artifact.get("inference_mode") == "blocked"
        assert artifact.get("precision_schema") == "carnot.precision_benchmark.v2"

    def test_setup_gpu_unhealthy_writes_blocked(self, tmp_path: Path):
        """When setup_gpu reports all_healthy=False, main() writes blocked artifact."""
        with patch.object(exp410, "load_preflight_verdict", return_value="gpu_confirmed_live"):
            with patch(
                "scripts.experiment_410_precision_live.LiveGPUGate.require_live_or_blocked",
                return_value=None,
            ):
                with patch(
                    "scripts.experiment_410_precision_live.ExperimentTemplate"
                ) as MockTmpl:
                    tmpl_instance = ExperimentTemplate(
                        exp_id=exp410.EXP_ID,
                        title=exp410.EXP_TITLE,
                        deliverable=exp410.DELIVERABLE,
                        repo_root=tmp_path,
                    )
                    tmpl_instance.setup()
                    tmpl_instance.setup_gpu = MagicMock(return_value=_unhealthy_gpu_status())
                    MockTmpl.return_value = tmpl_instance
                    exp410.main()

        output_path = tmp_path / exp410.DELIVERABLE
        assert output_path.exists()
        artifact = json.loads(output_path.read_text())
        assert artifact.get("honest_verdict") == "blocked"
        assert "not all_healthy" in artifact.get("failure_reason", "")

    def test_model_load_fails_writes_blocked(self, tmp_path: Path):
        """When a model fails to load, main() writes a blocked artifact."""
        with patch.object(exp410, "load_preflight_verdict", return_value="gpu_confirmed_live"):
            with patch(
                "scripts.experiment_410_precision_live.LiveGPUGate.require_live_or_blocked",
                return_value=None,
            ):
                with patch(
                    "scripts.experiment_410_precision_live.ExperimentTemplate"
                ) as MockTmpl:
                    tmpl_instance = ExperimentTemplate(
                        exp_id=exp410.EXP_ID,
                        title=exp410.EXP_TITLE,
                        deliverable=exp410.DELIVERABLE,
                        repo_root=tmp_path,
                    )
                    tmpl_instance.setup()
                    tmpl_instance.setup_gpu = MagicMock(return_value=_healthy_gpu_status())
                    MockTmpl.return_value = tmpl_instance

                    with patch(
                        "scripts.experiment_410_precision_live._load_model_pipeline",
                        side_effect=RuntimeError("CUDA OOM"),
                    ):
                        exp410.main()

        output_path = tmp_path / exp410.DELIVERABLE
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
        """Success artifact records n_models and n_variants."""
        artifact = _run_main_success(tmp_path)
        assert artifact.get("n_models") == 2
        assert artifact.get("n_variants") == 5

    def test_crane_import_fails_falls_back_to_llm_extractor(self, tmp_path: Path):
        """When CRANE import fails, main() wires LLMExtractor and succeeds."""
        _write_preflight(tmp_path, "gpu_confirmed_live")
        small_questions = [{"question": "What is 2+2?", "answer": "#### 4"}]
        mock_model = _mock_model_fn()

        with patch.object(exp410, "load_preflight_verdict", return_value="gpu_confirmed_live"):
            with patch(
                "scripts.experiment_410_precision_live.LiveGPUGate.require_live_or_blocked",
                return_value=None,
            ):
                with patch(
                    "scripts.experiment_410_precision_live.ExperimentTemplate"
                ) as MockTmpl:
                    tmpl_instance = ExperimentTemplate(
                        exp_id=exp410.EXP_ID,
                        title=exp410.EXP_TITLE,
                        deliverable=exp410.DELIVERABLE,
                        repo_root=tmp_path,
                    )
                    tmpl_instance.setup()
                    tmpl_instance.setup_gpu = MagicMock(return_value=_healthy_gpu_status())
                    MockTmpl.return_value = tmpl_instance

                    with patch.object(
                        exp410, "load_gsm8k_questions", return_value=small_questions
                    ):
                        with patch(
                            "scripts.experiment_410_precision_live._load_model_pipeline",
                            return_value=mock_model,
                        ):
                            # CRANE fails, LLM extractor succeeds
                            with patch.dict(
                                "sys.modules",
                                {"carnot.pipeline.crane_extractor": None},
                            ):
                                exp410.main()

        output_path = tmp_path / exp410.DELIVERABLE
        assert output_path.exists()
        artifact = json.loads(output_path.read_text())
        assert artifact.get("status") == "success"

    def test_both_extractors_fail_proceeds_gracefully(self, tmp_path: Path):
        """When both CRANE and LLM extractor imports fail, main() still succeeds."""
        small_questions = [{"question": "What is 2+2?", "answer": "#### 4"}]
        mock_model = _mock_model_fn()

        with patch.object(exp410, "load_preflight_verdict", return_value="gpu_confirmed_live"):
            with patch(
                "scripts.experiment_410_precision_live.LiveGPUGate.require_live_or_blocked",
                return_value=None,
            ):
                with patch(
                    "scripts.experiment_410_precision_live.ExperimentTemplate"
                ) as MockTmpl:
                    tmpl_instance = ExperimentTemplate(
                        exp_id=exp410.EXP_ID,
                        title=exp410.EXP_TITLE,
                        deliverable=exp410.DELIVERABLE,
                        repo_root=tmp_path,
                    )
                    tmpl_instance.setup()
                    tmpl_instance.setup_gpu = MagicMock(return_value=_healthy_gpu_status())
                    MockTmpl.return_value = tmpl_instance

                    with patch.object(
                        exp410, "load_gsm8k_questions", return_value=small_questions
                    ):
                        with patch(
                            "scripts.experiment_410_precision_live._load_model_pipeline",
                            return_value=mock_model,
                        ):
                            with patch.dict(
                                "sys.modules",
                                {
                                    "carnot.pipeline.crane_extractor": None,
                                    "carnot.pipeline.llm_extractor": None,
                                },
                            ):
                                exp410.main()

        output_path = tmp_path / exp410.DELIVERABLE
        assert output_path.exists()
        artifact = json.loads(output_path.read_text())
        assert artifact.get("status") == "success"
