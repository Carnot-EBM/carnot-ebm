"""Tests for scripts/experiment_1742_swe_bench.py.

Spec: REQ-BENCH-1742, SCENARIO-BENCH-1742
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

from carnot.pipeline.swebench_harness import DEFAULT_TARGET_INSTANCE_IDS, PatchEvaluation

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "experiment_1742_swe_bench.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("experiment_1742_swe_bench", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["experiment_1742_swe_bench"] = module
    spec.loader.exec_module(module)
    return module


def _row(instance_id: str) -> dict[str, object]:
    return {
        "instance_id": instance_id,
        "repo": "django/django",
        "base_commit": "abc123",
        "patch": "",
        "test_patch": "",
        "problem_statement": "Fix the issue.",
        "hints_text": "",
        "created_at": "2026-01-01T00:00:00Z",
        "version": "1.0",
        "FAIL_TO_PASS": '["tests/test_bug.py::test_regression"]',
        "PASS_TO_PASS": "[]",
        "environment_setup_commit": "def456",
    }


def _diff() -> str:
    return (
        "diff --git a/pkg/mod.py b/pkg/mod.py\n"
        "--- a/pkg/mod.py\n"
        "+++ b/pkg/mod.py\n"
        "@@ -1 +1 @@\n"
        "-old = 1\n"
        "+new = 2\n"
    )


def test_run_experiment_writes_blocked_artifact_without_cached_sota(tmp_path: Path) -> None:
    """SCENARIO-BENCH-1742: missing SOTA weights produce an honest blocked artifact."""
    module = _load_script()
    output_path = tmp_path / "experiment_1742_swe_bench.json"

    payload = module.run_experiment(
        output_path=output_path,
        rows=[_row(DEFAULT_TARGET_INSTANCE_IDS[0])],
        model_specs_provider=lambda: None,
    )

    assert output_path.exists()
    assert json.loads(output_path.read_text()) == payload
    assert payload["status"] == "blocked"
    assert payload["honest_verdict"] == "blocked_no_sota_gguf"
    assert payload["config"]["eqm_decoding_enabled"] is False
    assert payload["dataset"]["selected_instance_ids"] == [DEFAULT_TARGET_INSTANCE_IDS[0]]
    assert payload["metrics"]["n_instances"] == 1
    assert payload["metrics"]["headline_resolve_rates_available"] is False


def test_run_experiment_with_injected_backends_writes_complete_artifact(tmp_path: Path) -> None:
    """REQ-BENCH-1742: injected model and evaluator backends exercise the complete path."""
    module = _load_script()
    output_path = tmp_path / "experiment_1742_swe_bench.json"

    model_specs = [
        {"name": "Qwen3.6-35B-A3B", "hf_id": "unsloth/Qwen3.6-35B-A3B-GGUF"},
        {"name": "Gemma4-31B-it", "hf_id": "unsloth/gemma-4-31B-it-GGUF"},
    ]

    def generator(prompt: str, *, model_name: str, eqm_decoding_enabled: bool) -> str:
        assert "SWE-Bench Lite" in prompt
        assert model_name in {"Qwen3.6-35B-A3B", "Gemma4-31B-it"}
        assert eqm_decoding_enabled is False
        return _diff()

    def evaluator(problem, patch: str, model_name: str) -> PatchEvaluation:
        assert problem.instance_id == DEFAULT_TARGET_INSTANCE_IDS[0]
        assert patch.startswith("diff --git")
        assert model_name in {"Qwen3.6-35B-A3B", "Gemma4-31B-it"}
        return PatchEvaluation(
            resolved=True,
            status="complete",
            fail_to_pass_passed=True,
            pass_to_pass_passed=True,
            report_path="reports/result.json",
        )

    payload = module.run_experiment(
        output_path=output_path,
        rows=[_row(DEFAULT_TARGET_INSTANCE_IDS[0])],
        model_specs_provider=lambda: model_specs,
        generator=generator,
        evaluator=evaluator,
    )

    assert payload["status"] == "complete"
    assert payload["honest_verdict"] == "baseline_complete"
    assert payload["metrics"]["n_models"] == 2
    assert payload["metrics"]["baseline_resolve_rate"] == 1.0
    assert len(payload["per_model_results"]) == 2
