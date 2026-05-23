"""Tests for Exp 2926 ConstraintBench constrained-output rerun.

Spec: REQ-BENCH-2926, SCENARIO-BENCH-2926.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot.eval import constraintbench_constrained_output_rerun as exp
from carnot.eval import constraintbench_mini_direct_optimization as base


MANDATED = "unsloth/gemma-4-26B-A4B-it-GGUF"


def _spec() -> dict[str, Any]:
    return {
        "name": "Gemma4-26B-A4B-it",
        "hf_id": MANDATED,
        "gpu": 1,
        "model_path": "/tmp/gemma.gguf",
    }


def test_req_bench_2926_manifest_records_30_exact_tasks(tmp_path: Path) -> None:
    """REQ-BENCH-2926: the rerun materializes >=30 exact-verifier tasks."""

    tasks = exp.build_task_manifest()
    manifest = exp.write_task_manifest(tasks, tmp_path / exp.MANIFEST_FILENAME)
    persisted = json.loads((tmp_path / exp.MANIFEST_FILENAME).read_text(encoding="utf-8"))

    assert persisted == manifest
    assert len(tasks) == 30
    assert manifest["random_seed"] == 2926
    assert manifest["n_tasks"] == 30
    assert len({task.task_id for task in tasks}) == 30
    assert all(task.task_id.startswith("cbmini-2926-") for task in tasks)
    assert {task.task_type for task in tasks} == {
        "graph_coloring",
        "knapsack_binary",
        "linear_integer",
    }
    assert all(base.solve_task(task).feasible_count > 0 for task in tasks)
    assert all(row["objective_function"] for row in manifest["tasks"])
    assert all(row["constraints"] for row in manifest["tasks"])
    assert all(row["reference_answer"] for row in manifest["tasks"])
    assert all(row["prompt_hash"] == exp.prompt_hash(row["prompt"]) for row in manifest["tasks"])

    with pytest.raises(ValueError, match="at least 30"):
        exp.build_task_manifest(n_tasks=29)


def test_scenario_bench_2926_parser_repair_and_rates_are_non_tautological() -> None:
    """SCENARIO-BENCH-2926: syntax, feasibility, and optimality are separate rates."""

    tasks = exp.build_task_manifest()
    linear_task = next(task for task in tasks if task.task_type == "linear_integer")
    knapsack_task = next(task for task in tasks if task.task_type == "knapsack_binary")
    graph_task = next(task for task in tasks if task.task_type == "graph_coloring")

    optimal = base.solve_task(linear_task).optimal_solution
    stringified = {
        "solution": {
            "variables": {key: str(value) for key, value in optimal["variables"].items()}
        }
    }
    repaired = exp.parse_structured_output(linear_task, json.dumps(stringified))
    repaired_knapsack = exp.parse_structured_output(
        knapsack_task,
        json.dumps(
            {
                "solution": {
                    "selected_items": ",".join(
                        base.solve_task(knapsack_task).optimal_solution["selected_items"]
                    )
                }
            }
        ),
    )
    bad_solution = exp.parse_structured_output(linear_task, '{"solution": []}')
    assert repaired["syntax_valid"] is True
    assert repaired["parser_repair_applied"] is True
    assert repaired["parsed_output"] == optimal
    assert repaired_knapsack["syntax_valid"] is True
    assert repaired_knapsack["parser_repair_note"] == "comma_string_to_item_list"
    assert bad_solution["parse_error"] == "solution_not_object"

    optimal_row = exp.evaluate_raw_output(
        linear_task, json.dumps({"solution": optimal}), generation_metadata={}
    )
    infeasible_row = exp.evaluate_raw_output(
        graph_task,
        json.dumps({"solution": base.infeasible_answer_for_task(graph_task)}),
        generation_metadata={},
    )
    suboptimal_row = exp.evaluate_raw_output(
        linear_task,
        json.dumps({"solution": base.suboptimal_answer_for_task(linear_task)}),
        generation_metadata={},
    )
    syntax_row = exp.evaluate_raw_output(linear_task, "no json", generation_metadata={})
    metrics = exp.aggregate_results([optimal_row, infeasible_row, suboptimal_row, syntax_row])

    assert optimal_row["verifier_result"]["optimal"] is True
    assert infeasible_row["syntax_valid"] is True
    assert infeasible_row["feasible"] is False
    assert suboptimal_row["feasible"] is True
    assert suboptimal_row["optimal"] is False
    assert syntax_row["violation_class"] == "syntax_invalid"
    assert metrics["syntax_valid_rate"] == pytest.approx(0.75)
    assert metrics["feasibility_rate_overall"] == pytest.approx(0.5)
    assert metrics["feasibility_rate_given_syntax"] == pytest.approx(2 / 3)
    assert metrics["optimality_rate_given_feasible"] == pytest.approx(0.5)
    assert metrics["syntax_feasibility_not_tautological"] is True
    assert exp.aggregate_results([])["syntax_feasibility_not_tautological"] is False


def test_req_bench_2926_run_blocks_when_sota_gguf_cache_is_missing(tmp_path: Path) -> None:
    """REQ-BENCH-2926: no mandated GGUF produces the required honest blocked artifact."""

    calls: list[dict[str, Any]] = []

    def cached_pair(**kwargs: Any) -> None:
        calls.append(kwargs)
        return None

    artifact = exp.run_experiment(
        exp.ExperimentConfig(
            output_path=tmp_path / exp.OUTPUT_FILENAME,
            manifest_path=tmp_path / exp.MANIFEST_FILENAME,
            raw_response_dir=tmp_path / "raw",
            started_at=10.0,
            clock=lambda: 12.0,
            tests_run=("pytest focused",),
        ),
        cached_pair_provider=cached_pair,
        individual_model_resolver=lambda _hf_id: None,
        collect_model_outputs_fn=lambda *_args, **_kwargs: pytest.fail("collector must not run"),
        probe_llguidance=False,
    )

    assert calls == [{"gpu_indices": (0, 1)}]
    assert set(exp.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert json.loads((tmp_path / exp.OUTPUT_FILENAME).read_text(encoding="utf-8")) == artifact
    assert artifact["honest_verdict"] == "blocked_sota_gguf_cache_missing"
    assert artifact["constraintbench_corrigendum_ready"] is False
    assert artifact["flagged_adversarial"] is True
    assert artifact["cached_sota_pair_used"] is False
    assert artifact["constrained_decoder_available"] is False
    assert artifact["models_used"] == []
    assert artifact["n_tasks"] == 30
    assert artifact["syntax_valid_rate"] == 0.0
    assert artifact["feasibility_rate_overall"] == 0.0
    assert artifact["live_inference_duration_s"] == 0.0
    assert artifact["inference_substrate"] == "live_llm_inference"
    assert len(artifact["reproducibility_checksum"]) == 64


def test_scenario_bench_2926_run_writes_clean_corrected_row(tmp_path: Path) -> None:
    """SCENARIO-BENCH-2926: mixed fake live rows produce non-tautological metrics."""

    def fake_collection(
        spec: dict[str, Any],
        tasks: list[base.OptimizationTask],
        config: exp.ExperimentConfig,
        _decoder_status: dict[str, Any],
    ) -> dict[str, Any]:
        rows = [
            {
                "task_id": "not-in-manifest",
                "model_hf_id": spec["hf_id"],
                "model_name": spec["name"],
                "model_path": spec["model_path"],
                "gpu_index": spec["gpu"],
                "prompt_hash": "unknown",
                "per_task_seed": config.random_seed - 1,
                "generation_source": "live_sota_llamacpp_prompt_schema",
                "output_text": "{}",
                "raw_response_path": "",
                "raw_response_sha256": exp.sha256_text("{}"),
                "elapsed_seconds": 0.0,
                "blocker": None,
            }
        ]
        config.response_dir().mkdir(parents=True, exist_ok=True)
        for index, task in enumerate(tasks):
            if index % 4 == 0:
                output_text = base.compliant_answer_for_task(task)
            elif index % 4 == 1:
                output_text = json.dumps({"solution": base.infeasible_answer_for_task(task)})
            elif index % 4 == 2:
                output_text = json.dumps({"solution": base.suboptimal_answer_for_task(task)})
            else:
                output_text = "not json"
            raw_path = config.response_dir() / f"{task.task_id}.json"
            raw_path.write_text(output_text, encoding="utf-8")
            rows.append(
                {
                    "task_id": task.task_id,
                    "model_hf_id": spec["hf_id"],
                    "model_name": spec["name"],
                    "model_path": spec["model_path"],
                    "gpu_index": spec["gpu"],
                    "prompt_hash": exp.prompt_hash(task.prompt),
                    "per_task_seed": config.random_seed + index,
                    "generation_source": "live_sota_llamacpp_prompt_schema",
                    "output_text": output_text,
                    "raw_response_path": str(raw_path),
                    "raw_response_sha256": exp.sha256_text(output_text),
                    "elapsed_seconds": 2.0,
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
                "live_inference_duration_s": 61.2,
            },
            "rows": rows,
        }

    artifact = exp.run_experiment(
        exp.ExperimentConfig(
            output_path=tmp_path / exp.OUTPUT_FILENAME,
            manifest_path=tmp_path / exp.MANIFEST_FILENAME,
            raw_response_dir=tmp_path / "raw",
            started_at=1.0,
            clock=lambda: 66.0,
        ),
        cached_pair_provider=lambda **_: [
            _spec(),
            {
                "name": "Qwen3.6-35B-A3B",
                "hf_id": "unsloth/Qwen3.6-35B-A3B-GGUF",
                "gpu": 0,
                "model_path": "/tmp/qwen.gguf",
            },
        ],
        individual_model_resolver=lambda _hf_id: None,
        collect_model_outputs_fn=fake_collection,
        probe_llguidance=False,
    )

    manifest = json.loads((tmp_path / exp.MANIFEST_FILENAME).read_text(encoding="utf-8"))
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["constraintbench_corrigendum_ready"] is True
    assert artifact["flagged_adversarial"] is False
    assert artifact["models_used"] == [MANDATED]
    assert artifact["syntax_valid_rate"] == pytest.approx(23 / 30)
    assert artifact["feasibility_rate_overall"] == pytest.approx(15 / 30)
    assert artifact["feasibility_rate_given_syntax"] == pytest.approx(15 / 23)
    assert artifact["optimality_rate_given_feasible"] == pytest.approx(8 / 15)
    assert artifact["syntax_feasibility_not_tautological"] is True
    assert artifact["live_inference_duration_s"] == pytest.approx(61.2)
    assert len(artifact["per_task_results"]) == 30
    assert artifact["per_task_results"][0]["prompt_hash"] == manifest["tasks"][0]["prompt_hash"]
    assert artifact["per_task_results"][0]["per_task_seed"] == 2926
    assert artifact["per_task_results"][0]["model_path"] == "/tmp/gemma.gguf"
    assert artifact["per_task_results"][0]["parsed_output"]
    assert artifact["per_task_results"][0]["verifier_result"]["feasible"] is True
    assert artifact["model_attempts"][1]["blocker"] == "not_attempted_runtime_budget"
    assert (
        exp.compute_reproducibility_checksum(
            task_manifest=manifest["tasks"],
            model_specs=artifact["model_specs"],
            per_task_results=artifact["per_task_results"],
        )
        == artifact["reproducibility_checksum"]
    )
    mutated = [dict(row) for row in artifact["per_task_results"]]
    mutated[0]["raw_response_sha256"] = "changed"
    assert (
        exp.compute_reproducibility_checksum(
            task_manifest=manifest["tasks"],
            model_specs=artifact["model_specs"],
            per_task_results=mutated,
        )
        != artifact["reproducibility_checksum"]
    )


def test_req_bench_2926_duration_gate_blocks_short_live_rows(tmp_path: Path) -> None:
    """REQ-BENCH-2926: live rows under 60 seconds are honestly blocked."""

    def fake_short_collection(
        spec: dict[str, Any],
        tasks: list[base.OptimizationTask],
        _config: exp.ExperimentConfig,
        _decoder_status: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "summary": {
                "hf_id": spec["hf_id"],
                "model_name": spec["name"],
                "model_path": spec["model_path"],
                "model_used": True,
                "blocker": None,
                "live_inference_duration_s": 12.0,
            },
            "rows": [
                {
                    "task_id": task.task_id,
                    "model_hf_id": spec["hf_id"],
                    "model_name": spec["name"],
                    "model_path": spec["model_path"],
                    "gpu_index": spec["gpu"],
                    "prompt_hash": exp.prompt_hash(task.prompt),
                    "per_task_seed": 2926 + index,
                    "generation_source": "live_sota_llamacpp_prompt_schema",
                    "output_text": base.compliant_answer_for_task(task),
                    "raw_response_path": "",
                    "raw_response_sha256": "",
                    "elapsed_seconds": 0.1,
                    "blocker": None,
                }
                for index, task in enumerate(tasks)
            ],
        }

    artifact = exp.run_experiment(
        exp.ExperimentConfig(
            output_path=tmp_path / exp.OUTPUT_FILENAME,
            manifest_path=tmp_path / exp.MANIFEST_FILENAME,
            raw_response_dir=tmp_path / "raw",
        ),
        cached_pair_provider=lambda **_: [_spec()],
        individual_model_resolver=lambda _hf_id: None,
        collect_model_outputs_fn=fake_short_collection,
        probe_llguidance=False,
    )

    assert artifact["honest_verdict"] == "blocked_duration_gate_failed"
    assert artifact["constraintbench_corrigendum_ready"] is False
    assert artifact["flagged_adversarial"] is True
    assert artifact["live_inference_duration_s"] == pytest.approx(12.0)

    runtime_blocked = exp.run_experiment(
        exp.ExperimentConfig(
            output_path=tmp_path / "runtime_blocked.json",
            manifest_path=tmp_path / "runtime_manifest.json",
            raw_response_dir=tmp_path / "runtime_raw",
        ),
        cached_pair_provider=lambda **_: [_spec()],
        individual_model_resolver=lambda _hf_id: None,
        collect_model_outputs_fn=lambda spec, _tasks, _config, _decoder_status: {
            "summary": {
                "hf_id": spec["hf_id"],
                "model_name": spec["name"],
                "model_path": spec["model_path"],
                "model_used": False,
                "blocker": "no_usable_generations",
                "live_inference_duration_s": 0.0,
            },
            "rows": [],
        },
        probe_llguidance=False,
    )
    assert runtime_blocked["honest_verdict"] == "blocked_sota_runtime_unavailable"

    tautological = exp.run_experiment(
        exp.ExperimentConfig(
            output_path=tmp_path / "tautological.json",
            manifest_path=tmp_path / "tautological_manifest.json",
            raw_response_dir=tmp_path / "tautological_raw",
        ),
        cached_pair_provider=lambda **_: [_spec()],
        individual_model_resolver=lambda _hf_id: None,
        collect_model_outputs_fn=lambda spec, tasks, _config, _decoder_status: {
            "summary": {
                "hf_id": spec["hf_id"],
                "model_name": spec["name"],
                "model_path": spec["model_path"],
                "model_used": True,
                "blocker": None,
                "live_inference_duration_s": 61.0,
            },
            "rows": [
                {
                    "task_id": task.task_id,
                    "model_hf_id": spec["hf_id"],
                    "model_name": spec["name"],
                    "model_path": spec["model_path"],
                    "gpu_index": spec["gpu"],
                    "prompt_hash": exp.prompt_hash(task.prompt),
                    "per_task_seed": 2926 + index,
                    "generation_source": "live_sota_llamacpp_prompt_schema",
                    "output_text": base.compliant_answer_for_task(task),
                    "raw_response_path": "",
                    "raw_response_sha256": "",
                    "elapsed_seconds": 0.1,
                    "blocker": None,
                }
                for index, task in enumerate(tasks)
            ],
        },
        probe_llguidance=False,
    )
    assert tautological["honest_verdict"] == "blocked_tautological_syntax_feasibility_metrics"


def test_req_bench_2926_collect_live_outputs_captures_raw_provenance(tmp_path: Path) -> None:
    """REQ-BENCH-2926: live collection records prompt hashes, seeds, and raw files."""

    class FakeLlama:
        prompts: list[str] = []
        kwargs_seen: list[dict[str, Any]] = []
        closed = False

        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

        def __call__(self, prompt: str, **kwargs: Any) -> dict[str, Any]:
            self.prompts.append(prompt)
            self.kwargs_seen.append(kwargs)
            task = tasks[len(self.prompts) - 1]
            return {"choices": [{"text": base.compliant_answer_for_task(task)}]}

        def close(self) -> None:
            type(self).closed = True

    tasks = exp.build_task_manifest()[:2]
    config = exp.ExperimentConfig(raw_response_dir=tmp_path / "raw")
    collection = exp.collect_live_model_outputs(
        _spec(),
        tasks,
        config,
        {"constrained_decoder_available": False, "backend_name": "prompt_schema"},
        llama_importer=lambda: (True, FakeLlama, None),
    )

    first = collection["rows"][0]
    raw_payload = json.loads(Path(first["raw_response_path"]).read_text(encoding="utf-8"))
    assert collection["summary"]["model_used"] is True
    assert collection["summary"]["live_inference_duration_s"] >= 0.0
    assert FakeLlama.prompts == [task.prompt for task in tasks]
    assert FakeLlama.kwargs_seen[0]["seed"] == 2926
    assert FakeLlama.closed is True
    assert first["prompt_hash"] == exp.prompt_hash(tasks[0].prompt)
    assert first["per_task_seed"] == 2926
    assert first["gpu_index"] == 1
    assert first["raw_response_sha256"] == exp.sha256_text(raw_payload["raw_response"])
    assert raw_payload["model_path"] == "/tmp/gemma.gguf"
    assert exp._generation_source({"constrained_decoder_available": True}) == (
        "live_sota_llamacpp_llguidance_schema"
    )


def test_req_bench_2926_collector_records_runtime_failures(tmp_path: Path) -> None:
    """REQ-BENCH-2926: loader/import/generation edges fail closed."""

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
    config = exp.ExperimentConfig(raw_response_dir=tmp_path / "raw")
    decoder_status = {"constrained_decoder_available": False, "backend_name": "prompt_schema"}

    missing_path = exp.collect_live_model_outputs(
        {**_spec(), "model_path": ""},
        tasks,
        config,
        decoder_status,
        llama_importer=lambda: (True, MessageLlama, None),
    )
    import_failed = exp.collect_live_model_outputs(
        _spec(),
        tasks,
        config,
        decoder_status,
        llama_importer=lambda: (False, None, "llama_cpp missing"),
    )
    load_failed = exp.collect_live_model_outputs(
        _spec(),
        tasks,
        config,
        decoder_status,
        llama_importer=lambda: (True, LoadFails, None),
    )
    gen_failed = exp.collect_live_model_outputs(
        _spec(),
        tasks,
        config,
        decoder_status,
        llama_importer=lambda: (True, GenerateFails, None),
    )
    message = exp.collect_live_model_outputs(
        _spec(),
        tasks,
        config,
        decoder_status,
        llama_importer=lambda: (True, MessageLlama, None),
    )

    assert missing_path["summary"]["blocker"] == "model_not_cached"
    assert import_failed["summary"]["blocker"] == "llama_cpp missing"
    assert load_failed["summary"]["blocker"] == "RuntimeError: load failed"
    assert gen_failed["summary"]["model_used"] is False
    assert gen_failed["rows"][0]["blocker"] == "ValueError: generation failed"
    assert message["rows"][0]["output_text"] == '{"solution": {}}'


def test_req_bench_2926_constrained_decoder_probe_is_optional(monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-BENCH-2926: llguidance is preferred when present and fallback is explicit."""

    class NoMatcher:
        pass

    class BadMatcher:
        @staticmethod
        def grammar_from_json_schema(_schema: dict[str, Any], **_kwargs: Any) -> str:
            raise RuntimeError("bad grammar")

    class GoodMatcher:
        @staticmethod
        def grammar_from_json_schema(_schema: dict[str, Any], **_kwargs: Any) -> str:
            return "grammar"

        @staticmethod
        def validate_grammar(_grammar: str) -> str:
            return ""

    class GoodModule:
        LLMatcher = GoodMatcher

        @staticmethod
        def get_version() -> str:
            return "test-version"

    class ValidationFailsMatcher:
        @staticmethod
        def grammar_from_json_schema(_schema: dict[str, Any], **_kwargs: Any) -> str:
            return "bad grammar"

        @staticmethod
        def validate_grammar(_grammar: str) -> str:
            return "ERROR"

    def import_missing(_name: str) -> Any:
        raise ImportError("missing")

    monkeypatch.setattr(exp.importlib, "import_module", import_missing)
    assert exp.probe_constrained_decoder()["decoder_error"] == "llguidance not installed"
    assert exp.probe_constrained_decoder(probe_llguidance=False)[
        "constrained_decoder_available"
    ] is False
    assert exp.probe_constrained_decoder(llguidance_module=NoMatcher)[
        "constrained_decoder_available"
    ] is False
    assert exp.probe_constrained_decoder(llguidance_module=type("BadModule", (), {"LLMatcher": BadMatcher}))[
        "constrained_decoder_available"
    ] is False
    validation_fails = exp.probe_constrained_decoder(
        llguidance_module=type("ValidationFailsModule", (), {"LLMatcher": ValidationFailsMatcher})
    )
    assert validation_fails["decoder_error"] == "ERROR"
    assert validation_fails["grammar"] == "bad grammar"
    good = exp.probe_constrained_decoder(llguidance_module=GoodModule)
    assert good["constrained_decoder_available"] is True
    assert good["grammar"] == "grammar"
    assert good["llguidance_version"] == "test-version"
