"""Tests for Exp 2967 SOTA DCCD NL-to-Z3 frontier formalization.

Spec: REQ-BENCH-2967, SCENARIO-BENCH-2967.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot.eval import logic_frontier_materializer as exp2966
from carnot.eval import sota_nl_to_z3_dccd_formalization as exp


MANDATED = "unsloth/gemma-4-26B-A4B-it-GGUF"


def _spec() -> dict[str, Any]:
    return {
        "name": "Gemma4-26B-A4B-it",
        "hf_id": MANDATED,
        "gpu": 0,
        "model_path": "/tmp/gemma.gguf",
    }


def _qwen_spec() -> dict[str, Any]:
    return {
        "name": "Qwen3.6-35B-A3B",
        "hf_id": "unsloth/Qwen3.6-35B-A3B-GGUF",
        "gpu": 1,
        "model_path": "/tmp/qwen.gguf",
    }


def _materialize_frontier(tmp_path: Path) -> Path:
    output_path = tmp_path / "results" / exp2966.OUTPUT_FILENAME
    exp2966.run_experiment(
        exp2966.ExperimentConfig(
            repo_root=tmp_path,
            output_path=output_path,
            manifest_path=tmp_path / "data" / "research" / exp2966.MANIFEST_FILENAME,
            started_at=3.0,
            clock=lambda: 4.0,
        )
    )
    return output_path


def _proposal(
    item: exp.FrontierItem,
    *,
    assertions: list[str] | None = None,
    expected_status: str | None = None,
) -> str:
    return json.dumps(
        {
            "variables": [{"name": "x", "sort": "Entity"}],
            "predicates": [{"name": "Predicate", "signature": ["Entity"], "returns": "Bool"}],
            "assertions": assertions if assertions is not None else [item.reference_smt2],
            "query": "(check-sat)",
            "expected_status": expected_status or item.expected_solver_status,
            "answer_extraction": {"symbols": list(item.expected_answer_values)},
        },
        sort_keys=True,
    )


def _raw_row(item: exp.FrontierItem, text: str) -> dict[str, Any]:
    return {
        "item_id": item.item_id,
        "model_hf_id": MANDATED,
        "model_name": "Gemma4-26B-A4B-it",
        "model_path": "/tmp/gemma.gguf",
        "gpu_index": 0,
        "prompt_hash": exp.sha256_text(item.prompt),
        "per_item_seed": exp.RANDOM_SEED,
        "generation_source": "live_sota_dccd_structured_output",
        "draft_text": "Variables, predicates, assertions, query, and answer are identified.",
        "output_text": text,
        "raw_response_path": f"/tmp/{item.item_id}.json",
        "elapsed_seconds": 0.1,
        "blocker": None,
    }


def test_req_bench_2967_spec_anchor_exists() -> None:
    """REQ-BENCH-2967: the live frontier formalizer is OpenSpec anchored."""

    spec = Path("openspec/capabilities/benchmarks/spec.md").read_text(encoding="utf-8")

    assert "REQ-BENCH-2967" in spec
    assert "SCENARIO-BENCH-2967" in spec
    assert exp.OUTPUT_FILENAME in spec
    assert 'inference_substrate="live_llm_inference"' in spec
    assert "variables`, `predicates`, `assertions`, `query`" in spec


def test_req_bench_2967_parser_executes_z3_and_splits_failure_categories(
    tmp_path: Path,
) -> None:
    """REQ-BENCH-2967: every parseable structured proposal is judged by Z3."""

    frontier_path = _materialize_frontier(tmp_path)
    items = exp.load_frontier_items(exp.ExperimentConfig(frontier_artifact_path=frontier_path))
    item = items[0]
    wrong_answer = "sat" if item.expected_solver_status == "unsat" else "unsat"
    rows = [
        exp.evaluate_model_row(item, _proposal(item), generation_metadata={}),
        exp.evaluate_model_row(
            item,
            _proposal(item, expected_status=wrong_answer),
            generation_metadata={},
        ),
        exp.evaluate_model_row(
            item,
            _proposal(item, assertions=["(assert true)"], expected_status="sat"),
            generation_metadata={},
        ),
        exp.evaluate_model_row(
            item,
            _proposal(item, assertions=["(assert"], expected_status=item.expected_solver_status),
            generation_metadata={},
        ),
        exp.evaluate_model_row(item, "not json", generation_metadata={}),
    ]
    metrics = exp.aggregate_results(rows)

    assert rows[0]["failure_category"] == "solver_verified_correct"
    assert rows[1]["failure_category"] == "wrong_answer"
    assert rows[2]["failure_category"] == "wrong_formula"
    assert rows[3]["failure_category"] == "z3_exception"
    assert rows[4]["failure_category"] == "unparseable"
    assert rows[0]["structured_proposal"]["expected_status"] == item.expected_solver_status
    assert rows[0]["z3_result"]["actual_solver_status"] == item.expected_solver_status
    assert metrics["parseability_rate"] == pytest.approx(4 / 5)
    assert metrics["z3_execution_rate"] == pytest.approx(3 / 5)
    assert metrics["solver_verified_accuracy"] == pytest.approx(2 / 5)
    assert metrics["answer_accuracy"] == pytest.approx(2 / 5)
    assert metrics["failure_categories"] == {
        "unparseable": 1,
        "z3_exception": 1,
        "wrong_formula": 1,
        "wrong_answer": 1,
        "solver_verified_correct": 1,
    }
    nested = json.dumps({"formalization": json.loads(_proposal(item))})
    assert exp.parse_structured_proposal(nested)[0] is not None
    assert exp.parse_structured_proposal('{"variables": []}')[1] == (
        "missing_schema_field:predicates,assertions,query,expected_status,answer_extraction"
    )
    invalid_payloads = [
        (
            {
                "variables": {},
                "predicates": [],
                "assertions": ["(assert true)"],
                "query": "(check-sat)",
                "expected_status": "sat",
                "answer_extraction": {},
            },
            "variables_not_list",
        ),
        (
            {
                "variables": [],
                "predicates": {},
                "assertions": ["(assert true)"],
                "query": "(check-sat)",
                "expected_status": "sat",
                "answer_extraction": {},
            },
            "predicates_not_list",
        ),
        (
            {
                "variables": [],
                "predicates": [],
                "assertions": [],
                "query": "(check-sat)",
                "expected_status": "sat",
                "answer_extraction": {},
            },
            "assertions_not_nonempty_string_list",
        ),
        (
            {
                "variables": [],
                "predicates": [],
                "assertions": ["(assert true)"],
                "query": "",
                "expected_status": "sat",
                "answer_extraction": {},
            },
            "query_not_string",
        ),
        (
            {
                "variables": [],
                "predicates": [],
                "assertions": ["(assert true)"],
                "query": "(check-sat)",
                "expected_status": "unknown",
                "answer_extraction": {},
            },
            "expected_status_not_sat_or_unsat",
        ),
        (
            {
                "variables": [],
                "predicates": [],
                "assertions": ["(assert true)"],
                "query": "(check-sat)",
                "expected_status": "sat",
                "answer_extraction": [],
            },
            "answer_extraction_not_object",
        ),
    ]
    for payload, error in invalid_payloads:
        assert exp.parse_structured_proposal(json.dumps(payload))[1] == error
    mixed_shape = {
        "variables": ["loose"],
        "predicates": ["loose"],
        "assertions": ["(assert true)"],
        "query": "(check-sat)",
        "expected_status": "sat",
        "answer_extraction": {},
    }
    parsed, error = exp.parse_structured_proposal("{bad\n" + json.dumps(mixed_shape))
    assert error is None
    assert parsed is not None
    assert parsed.variables == [{"value": "loose"}]
    assert (
        exp.execute_structured_proposal(parsed, item, z3_module=None)["z3_error"]
        == "z3_unavailable"
    )

    answer_item = exp.FrontierItem(
        item_id="answer",
        prompt="Find answer.",
        expected_label="answer=1",
        check_kind="answer_extraction",
        expected_solver_status="sat",
        skill_labels=("answer extraction",),
        reference_smt2="",
        expected_answer_values={"answer": "1"},
    )
    answer_proposal, _ = exp.parse_structured_proposal(
        json.dumps(
            {
                "variables": [],
                "predicates": [],
                "assertions": ["(declare-const answer Int)", "(assert (= answer 1))"],
                "query": "(check-sat)",
                "expected_status": "sat",
                "answer_extraction": {"symbols": ["answer"]},
            }
        )
    )
    assert answer_proposal is not None
    answer_z3 = exp.execute_structured_proposal(answer_proposal, answer_item)
    assert answer_z3["actual_answer_values"] == {"answer": "1"}
    unsat_answer = exp.execute_structured_proposal(
        exp.StructuredProposal([], [], ["(assert false)"], "(check-sat)", "unsat", {}),
        answer_item,
    )
    assert unsat_answer["answer_extraction_matches_expected"] is False


def test_scenario_bench_2967_runner_writes_live_artifact_and_skill_metrics(
    tmp_path: Path,
) -> None:
    """SCENARIO-BENCH-2967: DCCD proposals are reported aggregate and skill-wise."""

    calls: list[dict[str, Any]] = []
    frontier_path = _materialize_frontier(tmp_path)

    def cached_pair(**kwargs: Any) -> list[dict[str, Any]]:
        calls.append(kwargs)
        return [_spec(), _qwen_spec()]

    def collect(
        _spec_arg: dict[str, Any],
        items: list[exp.FrontierItem],
        _config: exp.ExperimentConfig,
    ) -> dict[str, Any]:
        return {
            "summary": {
                "hf_id": MANDATED,
                "model_name": "Gemma4-26B-A4B-it",
                "model_path": "/tmp/gemma.gguf",
                "model_used": True,
                "blocker": None,
            },
            "rows": [_raw_row(item, _proposal(item)) for item in items],
        }

    artifact = exp.run_experiment(
        exp.ExperimentConfig(
            repo_root=tmp_path,
            output_path=tmp_path / "results" / exp.OUTPUT_FILENAME,
            frontier_artifact_path=frontier_path,
            raw_response_dir=tmp_path / "raw",
            started_at=10.0,
            clock=lambda: 22.5,
        ),
        cached_pair_provider=cached_pair,
        individual_model_resolver=lambda _hf_id: None,
        collect_model_outputs_fn=collect,
    )
    saved = json.loads((tmp_path / "results" / exp.OUTPUT_FILENAME).read_text(encoding="utf-8"))

    assert saved == artifact
    assert calls == [{"gpu_indices": (0, 1)}]
    assert set(exp.REQUIRED_ARTIFACT_FIELDS) <= artifact.keys()
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["inference_substrate"] == "live_llm_inference"
    assert artifact["legacy_models_only_for_smoke"] is False
    assert artifact["headline_models_used"] == [MANDATED]
    assert artifact["n_items"] == 24
    assert artifact["parseability_rate"] == pytest.approx(1.0)
    assert artifact["z3_execution_rate"] == pytest.approx(1.0)
    assert artifact["solver_verified_accuracy"] == pytest.approx(1.0)
    assert artifact["answer_accuracy"] == pytest.approx(1.0)
    assert artifact["baseline_parseability_rate"] == pytest.approx(0.083333)
    assert artifact["baseline_solver_verified_accuracy"] == pytest.approx(0.0)
    assert artifact["formalization_delta_clean"] is True
    assert artifact["failure_categories"]["solver_verified_correct"] == 24
    assert artifact["skill_wise_metrics"]["symbolization"]["n_items"] == 24
    assert artifact["skill_wise_metrics"]["symbolization"]["solver_verified_accuracy"] == 1.0
    assert artifact["formalization_manifest_sha256"] == exp.formalization_manifest_sha256(
        artifact["item_manifest"], artifact["per_item_results"]
    )
    assert artifact["duration_s"] == pytest.approx(12.5)
    assert artifact["model_attempts"][1]["blocker"] == "not_attempted_runtime_budget"


def test_req_bench_2967_preconditions_and_blocked_artifacts(tmp_path: Path) -> None:
    """REQ-BENCH-2967: missing source, Z3, or headline GGUFs fail closed."""

    missing_frontier = exp.run_experiment(
        exp.ExperimentConfig(
            repo_root=tmp_path,
            output_path=tmp_path / "missing.json",
            frontier_artifact_path=tmp_path / "absent.json",
            started_at=1.0,
            clock=lambda: 2.0,
        ),
        cached_pair_provider=lambda **_: [_spec()],
        individual_model_resolver=lambda _hf_id: None,
    )
    frontier_path = _materialize_frontier(tmp_path)
    no_z3 = exp.run_experiment(
        exp.ExperimentConfig(
            repo_root=tmp_path,
            output_path=tmp_path / "no_z3.json",
            frontier_artifact_path=frontier_path,
            started_at=2.0,
            clock=lambda: 3.0,
        ),
        cached_pair_provider=lambda **_: [_spec()],
        individual_model_resolver=lambda _hf_id: None,
        z3_module=None,
    )
    no_model = exp.run_experiment(
        exp.ExperimentConfig(
            repo_root=tmp_path,
            output_path=tmp_path / "no_model.json",
            frontier_artifact_path=frontier_path,
            started_at=3.0,
            clock=lambda: 4.0,
        ),
        cached_pair_provider=lambda **_: None,
        individual_model_resolver=lambda _hf_id: None,
    )
    recovered_single = exp.check_preconditions(
        exp.ExperimentConfig(frontier_artifact_path=frontier_path),
        cached_pair_provider=lambda **_: None,
        individual_model_resolver=lambda hf_id: "/tmp/gemma.gguf" if hf_id == MANDATED else None,
    )
    cache_exception = exp.check_preconditions(
        exp.ExperimentConfig(frontier_artifact_path=frontier_path),
        cached_pair_provider=lambda **_: (_ for _ in ()).throw(RuntimeError("cache down")),
        individual_model_resolver=lambda hf_id: "/tmp/gemma.gguf" if hf_id == MANDATED else None,
    )
    llama_missing = exp.check_preconditions(
        exp.ExperimentConfig(frontier_artifact_path=frontier_path),
        cached_pair_provider=lambda **_: [_spec()],
        individual_model_resolver=lambda _hf_id: None,
        module_importer=lambda name: (_ for _ in ()).throw(ImportError(name)),
    )
    relative_manifest = tmp_path / "relative_artifact.json"
    manifest = json.loads(
        (tmp_path / "data" / "research" / exp2966.MANIFEST_FILENAME).read_text(encoding="utf-8")
    )
    (tmp_path / "relative_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    relative_manifest.write_text(
        json.dumps(
            {"logic_frontier_materialized": True, "manifest_path": "relative_manifest.json"}
        ),
        encoding="utf-8",
    )
    not_materialized = tmp_path / "not_materialized.json"
    not_materialized.write_text(
        json.dumps(
            {"logic_frontier_materialized": False, "manifest_path": "relative_manifest.json"}
        ),
        encoding="utf-8",
    )
    non_object = tmp_path / "array.json"
    non_object.write_text("[]", encoding="utf-8")

    assert missing_frontier["honest_verdict"] == "blocked_precondition: exp2966_not_materialized"
    assert missing_frontier["n_items"] == 0
    assert no_z3["honest_verdict"] == "blocked_precondition: z3_import_failed"
    assert no_z3["z3_execution_rate"] == 0.0
    assert no_model["honest_verdict"] == "blocked_precondition: headline_gguf_missing"
    assert no_model["model_specs"]
    assert recovered_single.block_reason is None
    assert recovered_single.cached_sota_pair_used is False
    assert recovered_single.model_specs == [{**_spec(), "model_path": "/tmp/gemma.gguf"}]
    assert any("cache down" in row["detail"] for row in cache_exception.rows)
    assert llama_missing.block_reason == "llama_cpp_import_failed"
    assert (
        len(
            exp.load_frontier_items(
                exp.ExperimentConfig(repo_root=tmp_path, frontier_artifact_path=relative_manifest)
            )
        )
        == 24
    )
    with pytest.raises(ValueError, match="exp2966_not_materialized"):
        exp.load_frontier_items(exp.ExperimentConfig(frontier_artifact_path=not_materialized))
    with pytest.raises(ValueError, match="did not contain a JSON object"):
        exp.load_frontier_items(exp.ExperimentConfig(frontier_artifact_path=non_object))

    with pytest.raises(ValueError, match="missing required fields"):
        exp.validate_artifact({"honest_verdict": "x"})
    valid = json.loads((tmp_path / "no_model.json").read_text(encoding="utf-8"))
    with pytest.raises(ValueError, match="inference_substrate"):
        exp.validate_artifact(valid | {"inference_substrate": "deterministic_wiring"})
    with pytest.raises(ValueError, match="model_specs"):
        exp.validate_artifact(valid | {"model_specs": [{"hf_id": "legacy"}]})
    with pytest.raises(ValueError, match="formalization_delta_clean"):
        exp.validate_artifact(
            valid
            | {
                "parseability_rate": 1.0,
                "z3_execution_rate": 1.0,
                "solver_verified_accuracy": 1.0,
            }
        )


def test_req_bench_2967_collect_live_dccd_outputs_edges(tmp_path: Path) -> None:
    """REQ-BENCH-2967: live collection records draft/schema calls and blockers."""

    frontier_path = _materialize_frontier(tmp_path)
    item = exp.load_frontier_items(exp.ExperimentConfig(frontier_artifact_path=frontier_path))[0]

    class FakeLlama:
        prompts: list[str] = []
        closed = False

        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

        def __call__(self, prompt: str, **_kwargs: Any) -> dict[str, Any]:
            type(self).prompts.append(prompt)
            if len(type(self).prompts) % 2:
                return {"choices": [{"text": "Draft: use the reference constraints."}]}
            return {"choices": [{"message": {"content": _proposal(item)}}]}

        def close(self) -> None:
            type(self).closed = True

    class LoadFails:
        def __init__(self, **_kwargs: Any) -> None:
            raise RuntimeError("load failed")

    class GenerateFails:
        def __init__(self, **_kwargs: Any) -> None:
            return None

        def __call__(self, _prompt: str, **_kwargs: Any) -> dict[str, Any]:
            raise ValueError("generation failed")

        def close(self) -> None:
            return None

    class EmptyStructured:
        def __init__(self, **_kwargs: Any) -> None:
            return None

        def __call__(self, prompt: str, **_kwargs: Any) -> dict[str, Any]:
            if "Return exactly one JSON object" in prompt:
                return {"choices": [{"text": ""}]}
            return {"choices": [{"text": "Draft only."}]}

        def close(self) -> None:
            return None

    cfg = exp.ExperimentConfig(raw_response_dir=tmp_path / "raw")
    ok = exp.collect_live_dccd_outputs(
        _spec(),
        [item],
        cfg,
        llama_importer=lambda: (True, FakeLlama, None),
    )
    missing_path = exp.collect_live_dccd_outputs(
        {**_spec(), "model_path": ""},
        [item],
        cfg,
        llama_importer=lambda: (True, FakeLlama, None),
    )
    import_failed = exp.collect_live_dccd_outputs(
        _spec(),
        [item],
        cfg,
        llama_importer=lambda: (False, None, "llama_cpp missing"),
    )
    load_failed = exp.collect_live_dccd_outputs(
        _spec(),
        [item],
        cfg,
        llama_importer=lambda: (True, LoadFails, None),
    )
    gen_failed = exp.collect_live_dccd_outputs(
        _spec(),
        [item],
        cfg,
        llama_importer=lambda: (True, GenerateFails, None),
    )
    empty = exp.collect_live_dccd_outputs(
        _spec(),
        [item],
        cfg,
        llama_importer=lambda: (True, EmptyStructured, None),
    )

    assert ok["summary"]["model_used"] is True
    assert ok["rows"][0]["draft_text"].startswith("Draft:")
    assert ok["rows"][0]["output_text"].startswith("{")
    assert ok["rows"][0]["raw_response_sha256"] == exp.sha256_text(ok["rows"][0]["output_text"])
    assert "Return exactly one JSON object" in FakeLlama.prompts[1]
    assert FakeLlama.closed is True
    assert missing_path["summary"]["blocker"] == "model_not_cached"
    assert import_failed["summary"]["blocker"] == "llama_cpp missing"
    assert load_failed["summary"]["blocker"] == "RuntimeError: load failed"
    assert gen_failed["summary"]["model_used"] is False
    assert gen_failed["rows"][0]["blocker"] == "ValueError: generation failed"
    assert empty["rows"][0]["blocker"] == "empty_generation"
    assert exp.completion_text({"choices": [{"text": "plain"}]}) == "plain"
    assert exp.completion_text("raw") == "raw"
    assert exp.completion_text({"choices": []}) == ""
    assert exp.completion_text({"choices": [7]}) == ""
    assert exp.completion_text({"choices": [{"message": {}}]}) == ""
    assert exp.completion_text(123) == ""
