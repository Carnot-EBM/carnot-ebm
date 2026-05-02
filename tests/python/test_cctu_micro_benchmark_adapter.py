"""Tests for the Exp 1144 CCTU micro-benchmark adapter.

Spec: REQ-VERIFY-1144, SCENARIO-VERIFY-1144
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_DIR = REPO_ROOT / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

from carnot.eval.cctu_micro_benchmark_adapter import (  # noqa: E402
    ALLOWED_HONEST_VERDICTS,
    CCTU_SOURCE_DATASET_ID,
    MANDATED_SOTA_MODELS,
    REQUIRED_ARTIFACT_FIELDS,
    MockCCTURunner,
    ResponseEvaluation,
    build_exp1144_artifact,
    build_micro_benchmark_tasks,
    compliant_response_for_task,
    evaluate_response,
    load_or_write_tasks,
    repair_prompt,
    run_micro_benchmark,
    write_tasks_json,
)
import carnot.eval.cctu_micro_benchmark_adapter as adapter  # noqa: E402


def _load_script_module():
    spec = importlib.util.spec_from_file_location(
        "experiment_1144_cctu_micro_benchmark_adapter",
        REPO_ROOT / "scripts" / "experiment_1144_cctu_micro_benchmark_adapter.py",
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_task_generation_writes_25_executable_validator_specs(tmp_path: Path) -> None:
    """REQ-VERIFY-1144-1: task JSON has 25 sourced tasks and validator specs."""

    tasks = build_micro_benchmark_tasks()
    output_path = tmp_path / "cctu_micro_benchmark_25.json"
    written = write_tasks_json(tasks, output_path)

    assert len(tasks) == 25
    assert written == output_path
    assert json.loads(output_path.read_text(encoding="utf-8")) == tasks

    task_ids = {task["task_id"] for task in tasks}
    assert len(task_ids) == 25
    for task in tasks:
        assert task["source"]["dataset"] == CCTU_SOURCE_DATASET_ID
        assert task["system_prompt"]
        assert task["user_request"]
        assert task["expected_answer"]
        assert 3 <= len(task["constraints"]) <= 5
        for constraint in task["constraints"]:
            assert constraint["id"].startswith(task["task_id"])
            assert constraint["type"]
            assert constraint["description"]
            assert constraint["validator"]["name"]

        result = evaluate_response(task, compliant_response_for_task(task))
        assert result.passed is True
        assert result.completion_rate == pytest.approx(1.0)


def test_cascade_routes_numeric_semantic_and_format_verifiers() -> None:
    """REQ-VERIFY-1144-3: cascade invokes Z3, SemEnergy, and AST structure paths."""

    task = build_micro_benchmark_tasks()[2]
    good = compliant_response_for_task(task)
    bad = good.replace("SCORE: 22", "SCORE: 4").replace(task["expected_answer"], "wrong target")

    good_result = evaluate_response(task, good)
    bad_result = evaluate_response(task, bad)

    assert good_result.verifier_scores["z3_math"] == pytest.approx(0.0)
    assert "semenergy" in good_result.verifier_scores
    assert "ast_structure" in good_result.verifier_scores
    assert good_result.by_type["numeric"] == (1, 1)
    assert good_result.by_type["semantic"] == (1, 1)
    assert bad_result.passed is False
    assert "numeric" in bad_result.violated_types
    assert "semantic" in bad_result.violated_types


def test_repair_prompt_names_failed_constraints() -> None:
    """REQ-VERIFY-1144-4: Carnot-guided repair prompt carries verifier feedback."""

    task = build_micro_benchmark_tasks()[0]
    result = evaluate_response(task, "TOOL_CALL lookup {}\nFINAL: wrong\nSCORE: 4")
    prompt = repair_prompt(task, result)

    assert "Repair the response" in prompt
    assert task["task_id"] in prompt
    assert "numeric" in prompt
    assert "semantic" in prompt


def test_mock_run_writes_required_artifact_fields(tmp_path: Path) -> None:
    """REQ-VERIFY-1144-2/4/5: mock run is honest and reports required rates."""

    data_path = tmp_path / "cctu_micro_benchmark_25.json"
    output_path = tmp_path / "experiment_1144_cctu_micro_benchmark_adapter.json"

    artifact = run_micro_benchmark(
        data_path=data_path,
        output_path=output_path,
        force_mock=True,
    )

    assert output_path.exists()
    assert json.loads(output_path.read_text(encoding="utf-8")) == artifact
    assert set(REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert artifact["cctu_adapter_written"] is True
    assert artifact["n_tasks_defined"] == 25
    assert artifact["n_tasks_evaluated"] == 25
    assert artifact["model_used"] in MANDATED_SOTA_MODELS
    assert artifact["inference_mode"] == "mock"
    assert artifact["honest_verdict"] == "mock_inference_only"
    assert artifact["honest_verdict"] in ALLOWED_HONEST_VERDICTS
    assert artifact["baseline_completion_rate"] < artifact["carnot_guided_completion_rate"]
    assert artifact["carnot_delta_pp"] > 0.0
    assert set(artifact["constraint_types_tested"]) >= {"numeric", "semantic", "format"}
    assert all(0.0 <= rate <= 1.0 for rate in artifact["per_constraint_tp_rate"].values())


def test_edge_paths_for_existing_data_and_validator_failures(tmp_path: Path) -> None:
    """REQ-VERIFY-1144-1/3: loader and validators expose deterministic failures."""

    tasks = build_micro_benchmark_tasks()
    data_path = tmp_path / "tasks.json"
    write_tasks_json(tasks, data_path)

    assert load_or_write_tasks(data_path) == tasks
    assert ResponseEvaluation(False, (), {}, {}).completion_rate == 0.0

    no_score = compliant_response_for_task(tasks[0]).replace("SCORE: 20.", "")
    no_score_eval = evaluate_response(tasks[0], no_score)
    assert "score=None" in no_score_eval.checks[0].detail

    unknown_validator_task = json.loads(json.dumps(tasks[0]))
    unknown_validator_task["constraints"][0]["validator"] = {"name": "not_a_validator"}
    unknown_validator_eval = evaluate_response(
        unknown_validator_task, compliant_response_for_task(unknown_validator_task)
    )
    assert "unknown validator" in unknown_validator_eval.checks[0].detail

    unknown_style_task = json.loads(json.dumps(tasks[0]))
    unknown_style_task["constraints"][3]["validator"]["style"] = "not_a_style"
    unknown_style_eval = evaluate_response(
        unknown_style_task, compliant_response_for_task(unknown_style_task)
    )
    assert "unknown style" in unknown_style_eval.checks[3].detail


def test_run_can_skip_repair_when_baseline_already_passes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-VERIFY-1144-4: guided condition reuses already-valid baseline output."""

    class PassingRunner(MockCCTURunner):
        def generate(self, task: dict, prompt: str, *, guided: bool) -> str:
            del prompt, guided
            return compliant_response_for_task(task)

    monkeypatch.setattr(
        adapter,
        "_select_runner",
        lambda force_mock: (PassingRunner(), {"reason": f"force_mock={force_mock}"}),
    )
    artifact = adapter.run_micro_benchmark(
        data_path=tmp_path / "tasks.json",
        output_path=tmp_path / "artifact.json",
        force_mock=False,
    )

    assert artifact["baseline_completion_rate"] == pytest.approx(1.0)
    assert artifact["carnot_guided_completion_rate"] == pytest.approx(1.0)
    assert all(row["repair_used"] is False for row in artifact["guided_results"])


def test_build_exp1144_artifact_verdicts_are_deterministic() -> None:
    """REQ-VERIFY-1144-5: live verdict enum follows the measured delta sign."""

    base = {
        "model_used": MANDATED_SOTA_MODELS[0],
        "constraint_types_tested": ["numeric"],
        "per_constraint_tp_rate": {"numeric": 1.0},
        "baseline_results": [],
        "guided_results": [],
    }

    positive = build_exp1144_artifact(
        n_tasks_defined=25,
        n_tasks_evaluated=25,
        inference_mode="live_gpu",
        baseline_completion_rate=0.25,
        carnot_guided_completion_rate=0.5,
        **base,
    )
    neutral = build_exp1144_artifact(
        n_tasks_defined=25,
        n_tasks_evaluated=25,
        inference_mode="live_gpu",
        baseline_completion_rate=0.5,
        carnot_guided_completion_rate=0.5,
        **base,
    )
    negative = build_exp1144_artifact(
        n_tasks_defined=25,
        n_tasks_evaluated=25,
        inference_mode="live_gpu",
        baseline_completion_rate=0.75,
        carnot_guided_completion_rate=0.5,
        **base,
    )
    mock = build_exp1144_artifact(
        n_tasks_defined=25,
        n_tasks_evaluated=25,
        inference_mode="mock",
        baseline_completion_rate=0.0,
        carnot_guided_completion_rate=1.0,
        **base,
    )

    assert positive["honest_verdict"] == "carnot_positive_delta"
    assert neutral["honest_verdict"] == "carnot_neutral"
    assert negative["honest_verdict"] == "carnot_negative"
    assert mock["honest_verdict"] == "mock_inference_only"


def test_script_run_experiment_accepts_tmp_paths(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-1144: script entrypoint delegates to the adapter runner."""

    script = _load_script_module()
    data_path = tmp_path / "tasks.json"
    output_path = tmp_path / "artifact.json"

    artifact = script.run_experiment(
        data_path=data_path,
        output_path=output_path,
        force_mock=True,
    )

    assert artifact["cctu_adapter_written"] is True
    assert output_path.exists()
