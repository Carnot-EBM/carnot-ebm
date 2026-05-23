"""Tests for Exp 2931 LLMEval-Logic-style Z3 mini benchmark.

Spec: REQ-BENCH-2931, SCENARIO-BENCH-2931.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot.eval import llmeval_logic_z3_mini as exp


MANDATED = "unsloth/Qwen3.6-35B-A3B-GGUF"


def _spec() -> dict[str, Any]:
    return {
        "name": "Qwen3.6-35B-A3B",
        "hf_id": MANDATED,
        "gpu": 0,
        "model_path": "/tmp/qwen.gguf",
    }


def _completion(item: exp.LogicItem, answer: str | None = None) -> str:
    return json.dumps(
        {
            "formalization": item.formalization,
            "answer": answer or item.gold_answer,
        },
        sort_keys=True,
    )


def test_req_bench_2931_fixture_and_prompts_are_z3_checkable() -> None:
    """REQ-BENCH-2931: the fallback fixture has 12-20 Z3-checkable items."""

    items, scope = exp.build_or_load_logic_items(cache_paths=())
    prompts = [exp.prompt_for_item(item) for item in items]
    repeated, repeated_scope = exp.build_or_load_logic_items(cache_paths=())

    assert scope == repeated_scope
    assert "LLMEval-Logic-style" in scope
    assert 12 <= len(items) <= 20
    assert [item.item_id for item in items] == [item.item_id for item in repeated]
    assert len({item.item_id for item in items}) == len(items)
    assert all("Return exactly one JSON object" in prompt for prompt in prompts)
    assert all('"formalization"' in prompt for prompt in prompts)

    z3_rows = [exp.execute_z3_checks(item.formalization) for item in items]
    assert all(row["z3_executed"] is True for row in z3_rows)
    assert [row["solver_answer"] for row in z3_rows] == [item.gold_answer for item in items]
    assert exp.canonical_formalization(items[0].formalization) == exp.canonical_formalization(
        repeated[0].formalization
    )


def test_scenario_bench_2931_parse_z3_and_semantic_metrics_are_separate() -> None:
    """SCENARIO-BENCH-2931: parse, Z3, answer, and faithfulness rates split."""

    items, _scope = exp.build_or_load_logic_items(cache_paths=())
    faithful = exp.evaluate_raw_output(
        items[0],
        _completion(items[0]),
        generation_metadata={"model_hf_id": MANDATED, "raw_response_path": "a.json"},
    )
    wrong_answer = exp.evaluate_raw_output(
        items[1],
        _completion(items[1], answer="necessary"),
        generation_metadata={"model_hf_id": MANDATED, "raw_response_path": "b.json"},
    )
    syntax_error = exp.evaluate_raw_output(
        items[2],
        "not json",
        generation_metadata={"model_hf_id": MANDATED, "raw_response_path": "c.json"},
    )
    metrics = exp.aggregate_results([faithful, wrong_answer, syntax_error])

    assert faithful["parseable"] is True
    assert faithful["z3_executed"] is True
    assert faithful["answer_correct"] is True
    assert faithful["semantic_faithful"] is True
    assert faithful["z3_result"]["necessary"] is True

    assert wrong_answer["parseable"] is True
    assert wrong_answer["z3_executed"] is True
    assert wrong_answer["solver_answer"] == items[1].gold_answer
    assert wrong_answer["model_answer"] == "necessary"
    assert wrong_answer["answer_correct"] is False
    assert wrong_answer["semantic_faithful"] is True

    assert syntax_error["parseable"] is False
    assert syntax_error["z3_executed"] is False
    assert syntax_error["violation_class"] == "parse_error"

    assert metrics["parseability_rate"] == pytest.approx(2 / 3)
    assert metrics["z3_execution_rate"] == pytest.approx(2 / 3)
    assert metrics["answer_accuracy"] == pytest.approx(1 / 3)
    assert metrics["formalization_faithfulness_rate"] == pytest.approx(2 / 3)


def test_req_bench_2931_run_blocks_when_sota_gguf_cache_is_missing(tmp_path: Path) -> None:
    """REQ-BENCH-2931: missing mandated GGUF writes an honest blocked artifact."""

    calls: list[dict[str, Any]] = []

    def cached_pair(**kwargs: Any) -> None:
        calls.append(kwargs)
        return None

    artifact = exp.run_experiment(
        exp.ExperimentConfig(
            output_path=tmp_path / exp.OUTPUT_FILENAME,
            raw_response_dir=tmp_path / exp.RAW_RESPONSE_DIRNAME,
            started_at=10.0,
            clock=lambda: 12.5,
        ),
        cached_pair_provider=cached_pair,
        individual_model_resolver=lambda _hf_id: None,
        collect_model_outputs_fn=lambda *_args, **_kwargs: pytest.fail("collector must not run"),
    )

    assert calls == [{"gpu_indices": (0, 1)}]
    assert json.loads((tmp_path / exp.OUTPUT_FILENAME).read_text(encoding="utf-8")) == artifact
    assert set(exp.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert artifact["honest_verdict"] == "blocked_sota_gguf_cache_missing"
    assert artifact["logic_verifier_mini_ready"] is False
    assert artifact["models_used"] == []
    assert artifact["n_items"] == 12
    assert artifact["parseability_rate"] == 0.0
    assert artifact["z3_execution_rate"] == 0.0
    assert artifact["answer_accuracy"] == 0.0
    assert artifact["formalization_faithfulness_rate"] == 0.0
    assert artifact["per_item_results"] == []
    assert artifact["raw_response_dir"].endswith(exp.RAW_RESPONSE_DIRNAME)
    assert artifact["inference_substrate"] == "live_llm_inference_plus_z3"
    assert artifact["duration_s"] == pytest.approx(2.5)
    assert len(artifact["reproducibility_checksum"]) == 64


def test_scenario_bench_2931_runner_writes_live_z3_metrics(tmp_path: Path) -> None:
    """SCENARIO-BENCH-2931: fake live GGUF rows write required metrics."""

    def fake_collection(
        spec: dict[str, Any],
        items: list[exp.LogicItem],
        config: exp.ExperimentConfig,
    ) -> dict[str, Any]:
        rows = [
            {
                "item_id": "not-in-manifest",
                "model_hf_id": spec["hf_id"],
                "model_name": spec["name"],
                "model_path": spec["model_path"],
                "gpu_index": spec["gpu"],
                "prompt_hash": "unknown",
                "per_item_seed": exp.RANDOM_SEED - 1,
                "generation_source": "live_sota_llamacpp_logic_json",
                "output_text": "{}",
                "raw_response_path": "",
                "raw_response_sha256": exp.sha256_text("{}"),
                "elapsed_seconds": 0.0,
                "blocker": None,
            }
        ]
        config.response_dir().mkdir(parents=True, exist_ok=True)
        for index, item in enumerate(items):
            output_text = _completion(item, answer="possible" if index == 0 else None)
            raw_path = config.response_dir() / f"{item.item_id}.json"
            raw_path.write_text(output_text, encoding="utf-8")
            rows.append(
                {
                    "item_id": item.item_id,
                    "model_hf_id": spec["hf_id"],
                    "model_name": spec["name"],
                    "model_path": spec["model_path"],
                    "gpu_index": spec["gpu"],
                    "prompt_hash": exp.sha256_text(exp.prompt_for_item(item)),
                    "per_item_seed": exp.RANDOM_SEED + index,
                    "generation_source": "live_sota_llamacpp_logic_json",
                    "output_text": output_text,
                    "raw_response_path": str(raw_path),
                    "raw_response_sha256": exp.sha256_text(output_text),
                    "elapsed_seconds": 1.0,
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
                "live_inference_duration_s": 12.0,
            },
            "rows": rows,
        }

    artifact = exp.run_experiment(
        exp.ExperimentConfig(
            output_path=tmp_path / exp.OUTPUT_FILENAME,
            raw_response_dir=tmp_path / exp.RAW_RESPONSE_DIRNAME,
            max_models=1,
            started_at=1.0,
            clock=lambda: 20.0,
        ),
        cached_pair_provider=lambda **_: [
            _spec(),
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

    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["logic_verifier_mini_ready"] is True
    assert artifact["models_used"] == [MANDATED]
    assert artifact["parseability_rate"] == pytest.approx(1.0)
    assert artifact["z3_execution_rate"] == pytest.approx(1.0)
    assert artifact["answer_accuracy"] == pytest.approx(11 / 12)
    assert artifact["formalization_faithfulness_rate"] == pytest.approx(1.0)
    assert len(artifact["per_item_results"]) == 12
    assert artifact["model_attempts"][1]["blocker"] == "not_attempted_runtime_budget"
    assert artifact["per_item_results"][0]["answer_correct"] is False
    assert (
        exp.compute_reproducibility_checksum(
            items=artifact["item_manifest"],
            prompts=artifact["prompts"],
            model_specs=artifact["model_specs"],
            raw_outputs=[row["raw_output_sha256"] for row in artifact["per_item_results"]],
            parsed_formulas=[row["parsed_formalization"] for row in artifact["per_item_results"]],
            z3_results=[row["z3_result"] for row in artifact["per_item_results"]],
        )
        == artifact["reproducibility_checksum"]
    )


def test_req_bench_2931_model_resolution_and_live_collector_edges(tmp_path: Path) -> None:
    """REQ-BENCH-2931: SOTA resolution and llama.cpp collection fail closed."""

    calls: list[str] = []
    specs, cached_pair_used, cache_error = exp.resolve_model_specs(
        cached_pair_provider=lambda **_: None,
        individual_model_resolver=lambda hf_id: (
            calls.append(hf_id) or ("/tmp/gemma.gguf" if "gemma-4-31B" in hf_id else None)
        ),
    )
    _missing, _used, failed_cache = exp.resolve_model_specs(
        cached_pair_provider=lambda **_: (_ for _ in ()).throw(RuntimeError("cache boom")),
        individual_model_resolver=lambda _hf_id: None,
    )

    assert cached_pair_used is False
    assert cache_error is None
    assert calls == list(exp.MANDATED_MODEL_IDS)
    assert specs == [
        {
            "name": "Gemma4-31B-it",
            "hf_id": "unsloth/gemma-4-31B-it-GGUF",
            "gpu": 0,
            "model_path": "/tmp/gemma.gguf",
        }
    ]
    assert failed_cache == "RuntimeError: cache boom"

    class FakeLlama:
        prompts: list[str] = []
        kwargs_seen: list[dict[str, Any]] = []
        closed = False

        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

        def __call__(self, prompt: str, **kwargs: Any) -> dict[str, Any]:
            self.prompts.append(prompt)
            self.kwargs_seen.append(kwargs)
            item = exp.build_or_load_logic_items(cache_paths=())[0][len(self.prompts) - 1]
            return {"choices": [{"message": {"content": _completion(item)}}]}

        def close(self) -> None:
            type(self).closed = True

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

    items = exp.build_or_load_logic_items(cache_paths=())[0][:1]
    config = exp.ExperimentConfig(raw_response_dir=tmp_path / "raw")
    ok = exp.collect_live_model_outputs(
        _spec(),
        items,
        config,
        llama_importer=lambda: (True, FakeLlama, None),
    )
    missing_path = exp.collect_live_model_outputs(
        {**_spec(), "model_path": ""},
        items,
        config,
        llama_importer=lambda: (True, FakeLlama, None),
    )
    import_failed = exp.collect_live_model_outputs(
        _spec(),
        items,
        config,
        llama_importer=lambda: (False, None, "llama_cpp missing"),
    )
    load_failed = exp.collect_live_model_outputs(
        _spec(),
        items,
        config,
        llama_importer=lambda: (True, LoadFails, None),
    )
    gen_failed = exp.collect_live_model_outputs(
        _spec(),
        items,
        config,
        llama_importer=lambda: (True, GenerateFails, None),
    )

    assert ok["summary"]["model_used"] is True
    assert ok["rows"][0]["output_text"].startswith("{")
    assert ok["rows"][0]["raw_response_sha256"] == exp.sha256_text(ok["rows"][0]["output_text"])
    assert FakeLlama.prompts == [items[0].prompt]
    assert FakeLlama.kwargs_seen[0]["seed"] == exp.RANDOM_SEED
    assert FakeLlama.closed is True
    assert missing_path["summary"]["blocker"] == "model_not_cached"
    assert import_failed["summary"]["blocker"] == "llama_cpp missing"
    assert load_failed["summary"]["blocker"] == "RuntimeError: load failed"
    assert gen_failed["summary"]["model_used"] is False
    assert gen_failed["rows"][0]["blocker"] == "ValueError: generation failed"
    assert exp.completion_text({"choices": [{"text": "plain"}]}) == "plain"
    assert exp.completion_text("raw") == "raw"
    assert exp.completion_text({"choices": []}) == ""
    assert exp.completion_text({"choices": [7]}) == ""
    assert exp.completion_text({"choices": [{"message": {}}]}) == ""
    assert exp.completion_text(123) == ""


def test_req_bench_2931_defensive_parser_and_z3_branches(tmp_path: Path) -> None:
    """REQ-BENCH-2931: malformed formalizations fail closed."""

    item = exp.build_or_load_logic_items(cache_paths=())[0][0]
    cache_file = tmp_path / "llmeval_logic_cache.jsonl"
    cache_file.write_text(
        json.dumps(
            {
                "item_id": "cache-1",
                "problem": item.problem,
                "gold_answer": item.gold_answer,
                "formalization": item.formalization,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    loaded, scope = exp.build_or_load_logic_items(cache_paths=(cache_file,))
    assert loaded[0].item_id == "cache-1"
    assert "local LLMEval-Logic cache" in scope

    no_formalization = exp.parse_model_response('{"answer": "necessary"}')
    no_answer = exp.parse_model_response('{"formalization": {}}')
    invalid_answer = exp.parse_model_response('{"formalization": {}, "answer": "yes"}')
    bad_json = exp.parse_model_response("{not json}")

    assert no_formalization.error == "formalization_not_object"
    assert no_answer.error == "answer_not_string"
    assert invalid_answer.error == "answer_not_in_allowed_set"
    assert bad_json.error == "no_json_object"

    bad_rule = {"facts": [], "rules": [{"if": [], "then": []}], "exclusions": [], "query": []}
    bad_z3 = exp.execute_z3_checks(bad_rule)
    bad_atom = exp.execute_z3_checks(
        {"facts": [["A", "x"]], "rules": [], "exclusions": [], "query": ["A"]}
    )
    bad_atom_part = exp.execute_z3_checks(
        {"facts": [["", "x"]], "rules": [], "exclusions": [], "query": ["A", "x"]}
    )
    unsat = exp.execute_z3_checks(
        {"facts": [["A", "x"], ["B", "x"]], "rules": [], "exclusions": [[["A", "x"], ["B", "x"]]], "query": ["A", "x"]}
    )
    z3_error_row = exp.evaluate_raw_output(
        item,
        json.dumps(
            {
                "formalization": {
                    "facts": [["A", "x"]],
                    "rules": [],
                    "exclusions": [],
                    "query": ["A"],
                },
                "answer": item.gold_answer,
            },
            sort_keys=True,
        ),
        generation_metadata={},
    )
    extra_fact = exp.evaluate_raw_output(
        item,
        json.dumps(
            {
                "formalization": {
                    "facts": item.formalization["facts"] + [["Irrelevant", "milo"]],
                    "rules": item.formalization["rules"],
                    "exclusions": item.formalization["exclusions"],
                    "query": item.formalization["query"],
                },
                "answer": item.gold_answer,
            },
            sort_keys=True,
        ),
        generation_metadata={},
    )
    semantic_miss = exp.evaluate_raw_output(
        item,
        json.dumps(
            {
                "formalization": {
                    "facts": item.formalization["facts"],
                    "rules": [],
                    "exclusions": item.formalization["exclusions"],
                    "query": item.formalization["query"],
                },
                "answer": item.gold_answer,
            },
            sort_keys=True,
        ),
        generation_metadata={},
    )

    assert bad_z3["z3_executed"] is False
    assert bad_z3["z3_error"] == "atom_must_be_nonempty_list"
    assert bad_atom["z3_executed"] is False
    assert bad_atom["z3_error"] == "atom_needs_predicate_and_argument"
    assert bad_atom_part["z3_error"] == "atom_parts_must_be_nonempty_strings"
    assert unsat["z3_executed"] is True
    assert unsat["knowledge_base_consistent"] is False
    assert unsat["solver_answer"] == "inconsistent"
    assert z3_error_row["violation_class"] == "z3_error"
    assert extra_fact["violation_class"] == "semantic_mismatch"
    assert semantic_miss["parseable"] is True
    assert semantic_miss["z3_executed"] is True
    assert semantic_miss["semantic_faithful"] is False
    assert (
        exp._honest_verdict(  # noqa: SLF001
            specs=[_spec()],
            ready=False,
            models_used=[],
            z3_execution_rate=1.0,
        )
        == "blocked_sota_runtime_unavailable"
    )
    assert (
        exp._honest_verdict(  # noqa: SLF001
            specs=[_spec()],
            ready=False,
            models_used=[MANDATED],
            z3_execution_rate=0.5,
        )
        == "blocked_z3_execution_incomplete"
    )
    assert (
        exp._honest_verdict(  # noqa: SLF001
            specs=[_spec()],
            ready=False,
            models_used=[MANDATED],
            z3_execution_rate=1.0,
        )
        == "blocked_logic_verifier_mini_not_ready"
    )
