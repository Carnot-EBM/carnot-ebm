"""Tests for Exp 2919 ConstraintBench mini direct optimization.

Spec: REQ-BENCH-2919, SCENARIO-BENCH-2919.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot.eval import constraintbench_mini_direct_optimization as exp


MANDATED = "unsloth/Qwen3.6-35B-A3B-GGUF"


def test_req_bench_2919_manifest_has_three_exact_task_families(tmp_path: Path) -> None:
    """REQ-BENCH-2919: manifest materializes 15-30 tiny exact-verifier tasks."""

    tasks = exp.build_task_manifest()
    manifest_path = tmp_path / exp.MANIFEST_FILENAME
    persisted = exp.write_task_manifest(tasks, manifest_path)

    assert len(tasks) == 18
    assert json.loads(manifest_path.read_text(encoding="utf-8")) == persisted
    assert {task.task_type for task in tasks} == {
        "graph_coloring",
        "knapsack_binary",
        "linear_integer",
    }
    assert {task.exact_verifier_type for task in tasks} == set(exp.EXACT_VERIFIER_TYPES)
    assert all(exp.solve_task(task).feasible_count > 0 for task in tasks)
    assert all(
        task.prompt.startswith("Solve this direct constrained optimization task.") for task in tasks
    )


def test_scenario_bench_2919_verifier_separates_feasible_suboptimal_and_infeasible() -> None:
    """SCENARIO-BENCH-2919: feasibility and exact optimality are separate outcomes."""

    linear_task = next(
        task for task in exp.build_task_manifest() if task.task_type == "linear_integer"
    )
    graph_task = next(
        task for task in exp.build_task_manifest() if task.task_type == "graph_coloring"
    )

    optimal = exp.evaluate_model_output(linear_task, exp.compliant_answer_for_task(linear_task))
    suboptimal = exp.evaluate_model_output(
        linear_task,
        json.dumps({"solution": exp.suboptimal_answer_for_task(linear_task)}),
    )
    infeasible = exp.evaluate_model_output(
        graph_task,
        json.dumps(
            {
                "solution": {
                    "colors": {str(node): 0 for node in range(graph_task.payload["n_nodes"])}
                }
            }
        ),
    )

    assert optimal["syntax_valid"] is True
    assert optimal["feasible"] is True
    assert optimal["optimal"] is True
    assert optimal["violation_class"] == "none"
    assert optimal["objective_value"] == optimal["optimum_value"]

    assert suboptimal["syntax_valid"] is True
    assert suboptimal["feasible"] is True
    assert suboptimal["optimal"] is False
    assert suboptimal["violation_class"] == "suboptimal"

    assert infeasible["syntax_valid"] is True
    assert infeasible["feasible"] is False
    assert infeasible["optimal"] is False
    assert infeasible["violation_class"] == "infeasible"
    assert any(reason.startswith("edge_conflict:") for reason in infeasible["violation_reasons"])


def test_req_bench_2919_parser_accepts_wrappers_and_fails_closed() -> None:
    """REQ-BENCH-2919: structured JSON parsing is tolerant but schema-bound."""

    linear_task, knapsack_task, graph_task = exp.build_task_manifest()[:3]
    wrapped = f"<think>scratch</think>\n```json\n{exp.compliant_answer_for_task(linear_task)}\n```"
    parsed = exp.parse_model_output(linear_task, wrapped)
    no_json = exp.parse_model_output(linear_task, "no JSON here")
    bad_linear = exp.parse_model_output(linear_task, '{"solution": {"variables": {"x": true}}}')
    bad_knapsack = exp.parse_model_output(knapsack_task, '{"solution": {"selected_items": "all"}}')
    bad_graph = exp.parse_model_output(graph_task, '{"solution": {"colors": {"0": "red"}}}')

    assert parsed.syntax_valid is True
    assert parsed.solution == exp.solve_task(linear_task).optimal_solution
    assert no_json.parse_error == "no_json_object"
    assert bad_linear.parse_error == "variable_not_integer:x"
    assert bad_knapsack.parse_error == "selected_items_not_list"
    assert bad_graph.parse_error == "color_not_integer:0"


def test_req_bench_2919_run_blocks_when_mandated_gguf_cache_is_missing(tmp_path: Path) -> None:
    """REQ-BENCH-2919: missing mandated GGUFs writes the required blocked verdict."""

    calls: list[dict[str, Any]] = []

    def cached_pair(**kwargs: Any) -> None:
        calls.append(kwargs)
        return None

    artifact = exp.run_experiment(
        exp.ExperimentConfig(
            output_path=tmp_path / exp.OUTPUT_FILENAME,
            manifest_path=tmp_path / exp.MANIFEST_FILENAME,
            started_at=5.0,
            clock=lambda: 6.25,
            tests_run=("focused pytest",),
        ),
        cached_pair_provider=cached_pair,
        individual_model_resolver=lambda _hf_id: None,
        collect_model_outputs_fn=lambda _spec, _tasks: pytest.fail("collector must not run"),
    )

    assert calls == [{"gpu_indices": (0, 1)}]
    assert json.loads((tmp_path / exp.OUTPUT_FILENAME).read_text(encoding="utf-8")) == artifact
    assert artifact["honest_verdict"] == "blocked_sota_gguf_cache_missing"
    assert artifact["constraintbench_mini_ready"] is False
    assert artifact["cached_sota_pair_used"] is False
    assert artifact["models_used"] == []
    assert artifact["n_tasks"] == 18
    assert artifact["per_task_results"] == []
    assert artifact["feasibility_rate"] == 0.0
    assert artifact["optimality_rate"] == 0.0
    assert artifact["syntax_valid_rate"] == 0.0
    assert artifact["duration_s"] == pytest.approx(1.25)
    assert Path(artifact["task_manifest_path"]).name == exp.MANIFEST_FILENAME


def test_scenario_bench_2919_run_scores_fake_live_rows_and_counts_violations(
    tmp_path: Path,
) -> None:
    """SCENARIO-BENCH-2919: live rows produce separate feasibility and optimality rates."""

    def fake_collection(spec: dict[str, Any], tasks: list[exp.OptimizationTask]) -> dict[str, Any]:
        rows = []
        for index, task in enumerate(tasks):
            if index % 3 == 0:
                output_text = exp.compliant_answer_for_task(task)
            elif index % 3 == 1:
                output_text = json.dumps({"solution": exp.infeasible_answer_for_task(task)})
            else:
                output_text = json.dumps({"solution": exp.suboptimal_answer_for_task(task)})
            rows.append(
                {
                    "task_id": task.task_id,
                    "model_hf_id": spec["hf_id"],
                    "model_name": spec["name"],
                    "generation_source": "live_sota_llamacpp",
                    "output_text": output_text,
                    "elapsed_seconds": 0.01,
                    "blocker": None,
                }
            )
        return {
            "summary": {
                "hf_id": spec["hf_id"],
                "model_name": spec["name"],
                "model_path": spec["model_path"],
                "model_used": True,
                "blocker": None,
            },
            "rows": rows,
        }

    artifact = exp.run_experiment(
        exp.ExperimentConfig(
            output_path=tmp_path / exp.OUTPUT_FILENAME,
            manifest_path=tmp_path / exp.MANIFEST_FILENAME,
            started_at=10.0,
            clock=lambda: 13.0,
        ),
        cached_pair_provider=lambda **_: [
            {"name": "Qwen3.6-35B-A3B", "hf_id": MANDATED, "gpu": 0, "model_path": "/tmp/qwen.gguf"}
        ],
        individual_model_resolver=lambda _hf_id: None,
        collect_model_outputs_fn=fake_collection,
    )

    assert set(exp.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["constraintbench_mini_ready"] is True
    assert artifact["models_used"] == [MANDATED]
    assert artifact["cached_sota_pair_used"] is True
    assert artifact["syntax_valid_rate"] == pytest.approx(1.0)
    assert artifact["feasibility_rate"] == pytest.approx(12 / 18)
    assert artifact["optimality_rate"] == pytest.approx(6 / 18)
    assert artifact["violation_classes"] == {"infeasible": 6, "none": 6, "suboptimal": 6}
    assert len(artifact["per_task_results"]) == 18
    assert {row["exact_verifier_type"] for row in artifact["per_task_results"]} == set(
        exp.EXACT_VERIFIER_TYPES
    )


def test_req_bench_2919_collect_live_model_outputs_has_injectable_runtime_hooks() -> None:
    """REQ-BENCH-2919: live llama.cpp collection is testable without loading a real GGUF."""

    class FakeLlama:
        prompts: list[str] = []
        closed = False

        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

        def __call__(self, prompt: str, **_kwargs: Any) -> dict[str, Any]:
            self.prompts.append(prompt)
            task = exp.build_task_manifest()[len(self.prompts) - 1]
            return {"choices": [{"text": exp.compliant_answer_for_task(task)}]}

        def close(self) -> None:
            type(self).closed = True

    tasks = exp.build_task_manifest()[:2]
    spec = {"hf_id": MANDATED, "name": "Qwen3.6-35B-A3B", "gpu": 0, "model_path": "/tmp/qwen.gguf"}
    ok = exp.collect_live_model_outputs(
        spec,
        tasks,
        llama_importer=lambda: (True, FakeLlama, None),
    )
    missing_path = exp.collect_live_model_outputs(
        {**spec, "model_path": ""},
        tasks,
        llama_importer=lambda: (True, FakeLlama, None),
    )
    import_failed = exp.collect_live_model_outputs(
        spec,
        tasks,
        llama_importer=lambda: (False, None, "llama_cpp missing"),
    )

    assert ok["summary"]["model_used"] is True
    assert ok["rows"][0]["generation_source"] == "live_sota_llamacpp"
    assert FakeLlama.prompts == [task.prompt for task in tasks]
    assert FakeLlama.closed is True
    assert missing_path["summary"]["blocker"] == "model_not_cached"
    assert import_failed["summary"]["blocker"] == "llama_cpp missing"


def test_req_bench_2919_live_collector_records_load_generation_and_completion_edges() -> None:
    """REQ-BENCH-2919: collector records runtime failures and completion variants."""

    class LoadFails:
        def __init__(self, **_kwargs: Any) -> None:
            raise RuntimeError("load failed")

    class GenerateFails:
        def __init__(self, **_kwargs: Any) -> None:
            pass

        def __call__(self, _prompt: str, **_kwargs: Any) -> dict[str, Any]:
            raise ValueError("generation failed")

        def close(self) -> None:
            return None

    class MessageLlama:
        def __init__(self, **_kwargs: Any) -> None:
            pass

        def __call__(self, _prompt: str, **_kwargs: Any) -> dict[str, Any]:
            return {"choices": [{"message": {"content": '{"solution": {}}'}}]}

    tasks = exp.build_task_manifest()[:1]
    spec = {"hf_id": MANDATED, "name": "Qwen3.6-35B-A3B", "gpu": 0, "model_path": "/tmp/qwen.gguf"}

    load_failed = exp.collect_live_model_outputs(
        spec, tasks, llama_importer=lambda: (True, LoadFails, None)
    )
    gen_failed = exp.collect_live_model_outputs(
        spec, tasks, llama_importer=lambda: (True, GenerateFails, None)
    )
    message = exp.collect_live_model_outputs(
        spec, tasks, llama_importer=lambda: (True, MessageLlama, None)
    )

    assert load_failed["summary"]["blocker"] == "RuntimeError: load failed"
    assert gen_failed["summary"]["model_used"] is False
    assert gen_failed["rows"][0]["blocker"] == "ValueError: generation failed"
    assert message["rows"][0]["output_text"] == '{"solution": {}}'
    assert exp._completion_text({"choices": []}) == ""
    assert exp._completion_text("raw text") == "raw text"


def test_req_bench_2919_helper_branches_and_model_selection_are_deterministic() -> None:
    """REQ-BENCH-2919: helper branches fail closed and model fallback stays mandated."""

    calls: list[str] = []
    specs, used, error = exp.resolve_model_specs(
        cached_pair_provider=lambda **_: None,
        individual_model_resolver=lambda hf_id: (
            calls.append(hf_id) or ("/tmp/gemma.gguf" if "gemma-4-31B" in hf_id else None)
        ),
    )

    assert error is None
    assert used is False
    assert calls == list(exp.MANDATED_MODEL_IDS)
    assert specs == [
        {
            "name": "Gemma4-31B-it",
            "hf_id": "unsloth/gemma-4-31B-it-GGUF",
            "gpu": 0,
            "model_path": "/tmp/gemma.gguf",
        }
    ]

    _specs, _used, cache_error = exp.resolve_model_specs(
        cached_pair_provider=lambda **_: (_ for _ in ()).throw(RuntimeError("bad cache")),
        individual_model_resolver=lambda _hf_id: None,
    )
    assert cache_error == "RuntimeError: bad cache"

    linear_task = next(
        task for task in exp.build_task_manifest() if task.task_type == "linear_integer"
    )
    with pytest.raises(ValueError, match="unsupported operator"):
        exp._constraint_holds({"x": 0}, {"coefficients": {"x": 1}, "op": "!=", "rhs": 0})

    broken_task = exp.OptimizationTask(
        task_id="broken",
        task_type="unknown",
        exact_verifier_type="unknown",
        objective_sense="max",
        payload={},
        prompt="",
    )
    with pytest.raises(ValueError, match="unknown task type"):
        exp.solve_task(broken_task)
    assert (
        exp.parse_model_output(broken_task, '{"solution": {}}').parse_error == "unknown_task_type"
    )
    assert exp.infeasible_answer_for_task(linear_task)["variables"]


def test_req_bench_2919_defensive_parser_and_feasibility_branches() -> None:
    """REQ-BENCH-2919: closed defensive branches remain deterministic."""

    tasks = exp.build_task_manifest()
    linear_task = next(task for task in tasks if task.task_type == "linear_integer")
    knapsack_task = next(task for task in tasks if task.task_type == "knapsack_binary")
    graph_task = next(task for task in tasks if task.task_type == "graph_coloring")

    one_feasible = exp.OptimizationTask(
        task_id="one-feasible",
        task_type="linear_integer",
        exact_verifier_type="bounded_integer_exhaustive",
        objective_sense="max",
        payload={
            "variables": {"x": [0, 0]},
            "objective": {"x": 1},
            "constraints": [{"coefficients": {"x": 1}, "op": "==", "rhs": 0}],
        },
        prompt="",
    )
    impossible = exp.OptimizationTask(
        task_id="impossible",
        task_type="linear_integer",
        exact_verifier_type="bounded_integer_exhaustive",
        objective_sense="max",
        payload={
            "variables": {"x": [0, 0]},
            "objective": {"x": 1},
            "constraints": [{"coefficients": {"x": 1}, "op": ">=", "rhs": 1}],
        },
        prompt="",
    )

    assert exp.suboptimal_answer_for_task(one_feasible) == {"variables": {"x": 0}}
    assert exp.solve_task(impossible).feasible_count == 0
    assert (
        exp.parse_model_output(linear_task, '{"solution": []}').parse_error == "solution_not_object"
    )
    assert (
        exp.evaluate_model_output(linear_task, "{not json}")["violation_class"] == "syntax_invalid"
    )
    assert (
        exp.parse_model_output(linear_task, '{"solution": {"variables": []}}').parse_error
        == "variables_not_object"
    )
    assert exp.parse_model_output(
        linear_task, '{"solution": {"variables": {"x": 1}}}'
    ).parse_error.startswith("missing_variable:")
    assert exp.parse_model_output(
        knapsack_task, '{"solution": {"selected_items": [1]}}'
    ).parse_error == ("selected_item_not_string")
    assert (
        exp.parse_model_output(graph_task, '{"solution": {"colors": []}}').parse_error
        == "colors_not_object"
    )
    assert exp.parse_model_output(graph_task, '{"solution": {"colors": {"0": 0}}}').parse_error == (
        "missing_color:1"
    )

    graph_infeasible = exp.evaluate_model_output(
        graph_task,
        json.dumps(
            {
                "solution": {
                    "colors": {str(node): 999 for node in range(graph_task.payload["n_nodes"])}
                }
            }
        ),
    )
    assert any(
        reason.startswith("color_range:") for reason in graph_infeasible["violation_reasons"]
    )
    assert exp._check_graph_feasibility(graph_task, {"colors": {}})[1][0] == "missing_color:0"
    linear_infeasible = exp.evaluate_model_output(
        linear_task,
        json.dumps({"solution": exp.infeasible_answer_for_task(linear_task)}),
    )
    assert "domain:" in ",".join(linear_infeasible["violation_reasons"])
    assert exp._check_linear_feasibility(linear_task, {"variables": {}})[1][0].startswith(
        "missing_variable:"
    )
    duplicate_unknown = exp.evaluate_model_output(
        knapsack_task,
        json.dumps({"solution": {"selected_items": ["bogus", "bogus"]}}),
    )
    assert "duplicate_item" in duplicate_unknown["violation_reasons"]
    assert "unknown_item:bogus" in duplicate_unknown["violation_reasons"]
    knapsack_optimal = exp.evaluate_model_output(
        knapsack_task, exp.compliant_answer_for_task(knapsack_task)
    )
    assert knapsack_optimal["objective_value"] == knapsack_optimal["optimum_value"]

    assert exp.infeasible_answer_for_task(graph_task)["colors"]
    with pytest.raises(ValueError, match="unknown task type"):
        exp.infeasible_answer_for_task(exp.OptimizationTask("bad", "bad", "bad", "max", {}, ""))
    assert exp._solution_schema(exp.OptimizationTask("bad", "bad", "bad", "max", {}, "")) == {}
    assert exp._check_feasibility(exp.OptimizationTask("bad", "bad", "bad", "max", {}, ""), {}) == (
        False,
        ("unknown_task_type",),
    )
    assert exp._completion_text(123) == ""
    assert exp._completion_text({"choices": [7]}) == ""
    assert exp._completion_text({"choices": [{"message": {}}]}) == ""


def test_req_bench_2919_run_records_runtime_budget_skips_and_unknown_rows(tmp_path: Path) -> None:
    """REQ-BENCH-2919: unattempted models and unknown task rows are recorded safely."""

    def fake_collection(spec: dict[str, Any], tasks: list[exp.OptimizationTask]) -> dict[str, Any]:
        return {
            "summary": {
                "hf_id": spec["hf_id"],
                "model_name": spec["name"],
                "model_path": spec["model_path"],
                "model_used": True,
                "blocker": None,
            },
            "rows": [
                {
                    "task_id": "not-in-manifest",
                    "model_hf_id": spec["hf_id"],
                    "model_name": spec["name"],
                    "generation_source": "live_sota_llamacpp",
                    "output_text": "{}",
                    "elapsed_seconds": 0.01,
                    "blocker": None,
                },
                {
                    "task_id": tasks[0].task_id,
                    "model_hf_id": spec["hf_id"],
                    "model_name": spec["name"],
                    "generation_source": "live_sota_llamacpp",
                    "output_text": exp.compliant_answer_for_task(tasks[0]),
                    "elapsed_seconds": 0.01,
                    "blocker": None,
                },
            ],
        }

    artifact = exp.run_experiment(
        exp.ExperimentConfig(
            output_path=tmp_path / exp.OUTPUT_FILENAME,
            manifest_path=tmp_path / exp.MANIFEST_FILENAME,
            max_models=1,
        ),
        cached_pair_provider=lambda **_: [
            {
                "name": "Qwen3.6-35B-A3B",
                "hf_id": MANDATED,
                "gpu": 0,
                "model_path": "/tmp/qwen.gguf",
            },
            {
                "name": "Gemma4-31B-it",
                "hf_id": "unsloth/gemma-4-31B-it-GGUF",
                "gpu": 1,
                "model_path": "/tmp/gemma.gguf",
            },
        ],
        individual_model_resolver=lambda _hf_id: None,
        collect_model_outputs_fn=fake_collection,
    )

    assert len(artifact["per_task_results"]) == 1
    assert artifact["model_attempts"][1]["blocker"] == "not_attempted_runtime_budget"
