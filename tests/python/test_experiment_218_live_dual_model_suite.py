"""Spec: REQ-VERIFY-025, REQ-VERIFY-026, REQ-VERIFY-027, REQ-VERIFY-028, REQ-VERIFY-029,
SCENARIO-VERIFY-025, SCENARIO-VERIFY-026, SCENARIO-VERIFY-027,
SCENARIO-VERIFY-028, SCENARIO-VERIFY-029.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import runpy
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest


def load_module():
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "scripts" / "experiment_218_live_dual_model_suite.py"
    spec = importlib.util.spec_from_file_location(
        "experiment_218_live_dual_model_suite", module_path
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    return repo


def sample_records() -> list[dict[str, object]]:
    return [
        {"case_id": "case-a", "prompt": "Alpha", "task_slice": "instruction_surface_only"},
        {"case_id": "case-b", "prompt": "Beta", "task_slice": "code_typed_properties"},
        {"case_id": "case-c", "prompt": "Gamma", "task_slice": "live_gsm8k_semantic_failure"},
    ]


def make_constraint_case(
    *,
    schema_type: str = "json_object",
    constraint_types: list[str] | None = None,
    expected_answer_schema: dict[str, object] | None = None,
    source_family: str = "instruction_following",
    task_slice: str = "instruction_grounded",
    gold_atomic_constraints: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    return {
        "case_id": "constraint-case",
        "example_id": "constraint-case",
        "source_family": source_family,
        "task_slice": task_slice,
        "constraint_types": constraint_types or ["literal"],
        "prompt": "Prompt",
        "expected_answer_schema": expected_answer_schema or {"type": schema_type},
        "gold_atomic_constraints": gold_atomic_constraints or [],
    }


def make_constraint_context(**overrides: object) -> dict[str, object]:
    base = {
        "schema_type": "json_object",
        "raw_response": "",
        "structured_payload": None,
        "answer_candidate": "",
        "answer_text": "",
        "normalized_answer": "",
        "normalized_response": "",
        "checks": [],
        "json_answer": None,
        "yaml_answer": None,
        "bullet_answer": None,
        "comma_items": None,
        "sections": [],
        "sentences": [],
        "numbered_steps": [],
        "identifier": None,
        "parsed_number": None,
        "code_text": "",
        "function_node": None,
        "function_obj": object(),
        "function_name": "",
        "function_signature": "",
        "code_probes_pass": True,
        "code_probe_details": {},
        "resolved_values": {},
        "output_style": "answer_only_terse",
    }
    base.update(overrides)
    return base


# REQ-VERIFY-025
def test_build_parser_and_supported_models_are_fixed():
    module = load_module()

    parser = module.build_parser()
    benchmark_action = next(action for action in parser._actions if action.dest == "benchmark")

    assert list(benchmark_action.choices) == [
        "gsm8k_semantic",
        "humaneval_property",
        "constraint_ir",
    ]
    assert [model["name"] for model in module.MODEL_SPECS] == [
        "Qwen3.5-0.8B",
        "Gemma4-E4B-it",
    ]
    assert [model["hf_id"] for model in module.MODEL_SPECS] == [
        "Qwen/Qwen3.5-0.8B",
        "google/gemma-4-E4B-it",
    ]

    args = parser.parse_args(["--benchmark", "gsm8k_semantic"])
    assert args.sample_seed == module.DEFAULT_SAMPLE_SEED
    assert args.sample_size == module.BENCHMARK_SPECS["gsm8k_semantic"]["default_sample_size"]
    assert args.output == module.default_output_path("gsm8k_semantic")


# REQ-VERIFY-025, SCENARIO-VERIFY-026
def test_build_cohort_manifest_is_deterministic_and_reuses_prompt_seed_across_modes():
    module = load_module()

    first = module.build_cohort_manifest(sample_records(), sample_size=2, sample_seed=11)
    second = module.build_cohort_manifest(sample_records(), sample_size=2, sample_seed=11)

    assert first == second
    assert [case["sample_position"] for case in first] == [1, 2]
    assert {case["case_id"] for case in first}.issubset({"case-a", "case-b", "case-c"})
    for case in first:
        prompt_seeds = case["prompt_seeds"]
        assert prompt_seeds["baseline"] == prompt_seeds["verify_only"]
        assert prompt_seeds["verify_only"] == prompt_seeds["verify_repair"]


# REQ-VERIFY-025
def test_recommended_response_mode_uses_exp213_policy_defaults():
    module = load_module()

    assert module.recommended_response_mode("live_gsm8k_semantic_failure") == "structured_json"
    assert module.recommended_response_mode("code_typed_properties") == "answer_only_terse"
    assert module.recommended_response_mode("unknown-task-slice") == "answer_only_terse"


# REQ-VERIFY-025
def test_repo_override_policy_fallback_checkpoint_edges_and_utc_now(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    module = load_module()
    repo = make_repo(tmp_path)
    monkeypatch.setenv("CARNOT_REPO_ROOT", str(repo))

    assert module.get_repo_root() == repo.resolve()
    assert module.load_monitorability_policy(repo / "results" / "missing.json") == {}

    missing = module.load_checkpoint(tmp_path / "missing.json", ["case-a"])
    assert missing == {"case_ids": ["case-a"], "results_by_case": {}}

    mismatch_path = tmp_path / "mismatch.json"
    mismatch_path.write_text(
        json.dumps({"case_ids": ["other-case"], "results_by_case": {"other-case": {}}}),
        encoding="utf-8",
    )
    mismatch = module.load_checkpoint(mismatch_path, ["case-a"])
    assert mismatch == {"case_ids": ["case-a"], "results_by_case": {}}

    invalid_results_path = tmp_path / "invalid-results.json"
    invalid_results_path.write_text(
        json.dumps({"case_ids": ["case-a"], "results_by_case": []}),
        encoding="utf-8",
    )
    invalid = module.load_checkpoint(invalid_results_path, ["case-a"])
    assert invalid == {"case_ids": ["case-a"], "results_by_case": {}}

    assert "T" in module.utc_now()


# SCENARIO-VERIFY-025
def test_run_mode_resumes_only_pending_cases_and_preserves_order(tmp_path: Path):
    module = load_module()
    checkpoint_dir = tmp_path / "checkpoints"
    cases = module.build_cohort_manifest(sample_records(), sample_size=2, sample_seed=7)
    checkpoint_path = module.checkpoint_path(
        checkpoint_dir,
        benchmark="constraint_ir",
        model_name="Qwen3.5-0.8B",
        mode="baseline",
    )

    module.save_checkpoint(
        checkpoint_path,
        {
            "benchmark": "constraint_ir",
            "model_name": "Qwen3.5-0.8B",
            "mode": "baseline",
            "case_ids": [case["case_id"] for case in cases],
            "results_by_case": {
                cases[0]["case_id"]: {
                    "case_id": cases[0]["case_id"],
                    "mode": "baseline",
                    "value": "cached",
                }
            },
        },
    )

    executed: list[str] = []

    def execute_case(case: dict[str, object]) -> dict[str, object]:
        executed.append(str(case["case_id"]))
        return {
            "case_id": case["case_id"],
            "mode": "baseline",
            "value": f"fresh-{case['case_id']}",
        }

    results = module.run_mode(
        benchmark="constraint_ir",
        model_name="Qwen3.5-0.8B",
        mode="baseline",
        cases=cases,
        checkpoint_dir=checkpoint_dir,
        execute_case=execute_case,
    )

    assert executed == [cases[1]["case_id"]]
    assert [result["case_id"] for result in results] == [case["case_id"] for case in cases]
    assert results[0]["value"] == "cached"
    assert results[1]["value"] == f"fresh-{cases[1]['case_id']}"


# REQ-VERIFY-026, SCENARIO-VERIFY-026
def test_build_artifact_payload_has_stable_schema_and_order():
    module = load_module()
    cohort = module.build_cohort_manifest(sample_records(), sample_size=2, sample_seed=5)
    paired_runs = [
        {
            "benchmark": "constraint_ir",
            "mode": "baseline",
            "model_name": "Qwen3.5-0.8B",
            "model_hf_id": "Qwen/Qwen3.5-0.8B",
            "summary": {"n_cases": 2},
            "cases": [{"case_id": cohort[0]["case_id"]}, {"case_id": cohort[1]["case_id"]}],
        },
        {
            "benchmark": "constraint_ir",
            "mode": "verify_only",
            "model_name": "Qwen3.5-0.8B",
            "model_hf_id": "Qwen/Qwen3.5-0.8B",
            "summary": {"n_cases": 2},
            "cases": [{"case_id": cohort[0]["case_id"]}, {"case_id": cohort[1]["case_id"]}],
        },
    ]

    payload = module.build_artifact_payload(
        benchmark="constraint_ir",
        output_path=Path("results/experiment_219_results.json"),
        cohort=cohort,
        paired_runs=paired_runs,
        statistics={"Qwen3.5-0.8B": {"baseline": {"n_cases": 2}}},
        sample_seed=5,
        sample_size=2,
        started_at="2026-04-12T12:00:00Z",
        finished_at="2026-04-12T12:01:00Z",
        runtime_seconds=60.0,
        checkpoint_dir=Path("results/checkpoints/experiment_218"),
        max_repairs=3,
        policy_path=Path("results/monitorability_policy_213.json"),
        inference_mode="live_gpu",
    )

    assert payload["experiment"] == 219
    assert payload["run_date"] == "20260412"
    assert payload["schema"]["artifact"] == "carnot.live_dual_model_suite.v1"
    assert payload["schema"]["benchmark_case_schema"] == "constraint_ir.v1"
    assert payload["metadata"]["output_path"] == "results/experiment_219_results.json"
    assert payload["metadata"]["checkpoint_dir"] == "results/checkpoints/experiment_218"
    assert payload["metadata"]["max_repairs"] == 3
    assert payload["metadata"]["policy_source"] == "results/monitorability_policy_213.json"
    assert payload["metadata"]["inference_mode"] == "live_gpu"
    assert payload["cohort"]["case_ids"] == [case["case_id"] for case in cohort]
    assert payload["paired_runs"] == paired_runs
    assert (
        payload["cohort"]["cases"][0]["prompt_seeds"]["baseline"]
        == payload["cohort"]["cases"][0]["prompt_seeds"]["verify_repair"]
    )


# REQ-VERIFY-027
def test_artifact_experiment_id_follows_output_filename():
    module = load_module()

    assert module.artifact_experiment_id(Path("results/experiment_219_results.json")) == 219
    assert module.artifact_experiment_id(Path("results/experiment_220_results.json")) == 220
    assert module.artifact_experiment_id(Path("results/experiment_221_results.json")) == 221
    assert module.artifact_experiment_id(Path("results/custom.json")) == 218


# REQ-VERIFY-027
def test_extract_final_number_ignores_punctuation_only_matches():
    module = load_module()

    assert module._extract_final_number("Result: ,,,") is None
    assert module._extract_final_number("Answer: -12") == -12


# REQ-VERIFY-027, SCENARIO-VERIFY-027
def test_summarize_gsm8k_runs_reports_semantic_metrics_parse_coverage_and_overhead():
    module = load_module()

    baseline_runs = [
        {
            "correct": True,
            "latency_seconds": 1.2,
            "prompt_tokens": 18,
            "response_tokens": 7,
            "total_tokens": 25,
            "typed_reasoning_parse_status": "direct_json",
        },
        {
            "correct": False,
            "latency_seconds": 1.8,
            "prompt_tokens": 20,
            "response_tokens": 9,
            "total_tokens": 29,
            "typed_reasoning_parse_status": "fallback_text",
        },
    ]
    verify_only_runs = [
        {
            "accepted_correct": True,
            "flagged": False,
            "correct": True,
            "semantic_violation_count": 0,
            "typed_reasoning_parse_status": "direct_json",
            "latency_seconds": 0.4,
            "total_tokens": 0,
        },
        {
            "accepted_correct": False,
            "flagged": True,
            "correct": False,
            "semantic_violation_count": 2,
            "typed_reasoning_parse_status": "unavailable",
            "latency_seconds": 0.6,
            "total_tokens": 0,
        },
    ]
    verify_repair_runs = [
        {
            "initial_correct": True,
            "correct": True,
            "repaired": False,
            "n_repairs": 0,
            "latency_seconds": 0.0,
            "total_tokens": 0,
        },
        {
            "initial_correct": False,
            "correct": True,
            "repaired": True,
            "n_repairs": 2,
            "latency_seconds": 2.5,
            "total_tokens": 34,
        },
    ]

    summary = module._summarize_runs(
        "gsm8k_semantic",
        baseline_runs,
        verify_only_runs,
        verify_repair_runs,
    )

    assert summary["baseline"]["accuracy"] == 0.5
    assert summary["baseline"]["mean_total_tokens"] == 27.0
    assert summary["verify_only"]["accuracy"] == 0.5
    assert summary["verify_only"]["n_wrong_answers"] == 1
    assert summary["verify_only"]["n_wrong_detected"] == 1
    assert summary["verify_only"]["semantic_violation_count"] == 2
    assert summary["verify_only"]["parse_coverage"] == 0.5
    assert summary["verify_only"]["false_positives"] == 0
    assert summary["verify_only"]["mean_additional_latency_seconds"] == 0.5
    assert summary["verify_repair"]["accuracy"] == 1.0
    assert summary["verify_repair"]["n_repaired"] == 1
    assert summary["verify_repair"]["repair_yield"] == 1.0
    assert summary["verify_repair"]["mean_additional_tokens"] == 17.0
    assert summary["paired_deltas"]["repair_minus_baseline"] == 0.5
    assert summary["paired_deltas"]["verify_only_minus_baseline"] == 0.0


# REQ-VERIFY-028, SCENARIO-VERIFY-028
def test_summarize_humaneval_runs_reports_execution_vs_property_metrics_and_traces():
    module = load_module()

    baseline_runs = [
        {"passed": True, "latency_seconds": 1.0},
        {"passed": False, "latency_seconds": 1.4},
        {"passed": True, "latency_seconds": 1.6},
    ]
    verify_only_runs = [
        {
            "passed": True,
            "execution_only": {"detected": False},
            "execution_plus_property": {"detected": False, "n_property_violations": 0},
            "execution_only_accepted": True,
            "execution_plus_property_accepted": True,
            "property_only_detected": False,
            "official_test_miss_caught_by_property": False,
            "execution_only_latency_seconds": 0.2,
            "execution_plus_property_latency_seconds": 0.4,
            "latency_seconds": 0.7,
        },
        {
            "passed": False,
            "execution_only": {"detected": True},
            "execution_plus_property": {"detected": True, "n_property_violations": 2},
            "execution_only_accepted": False,
            "execution_plus_property_accepted": False,
            "property_only_detected": False,
            "official_test_miss_caught_by_property": False,
            "execution_only_latency_seconds": 0.3,
            "execution_plus_property_latency_seconds": 0.5,
            "latency_seconds": 0.9,
        },
        {
            "passed": True,
            "execution_only": {"detected": False},
            "execution_plus_property": {"detected": True, "n_property_violations": 1},
            "execution_only_accepted": True,
            "execution_plus_property_accepted": False,
            "property_only_detected": True,
            "official_test_miss_caught_by_property": True,
            "execution_only_latency_seconds": 0.4,
            "execution_plus_property_latency_seconds": 0.8,
            "latency_seconds": 1.1,
        },
    ]
    verify_repair_runs = [
        {"passed": True, "repaired": False, "n_repairs": 0, "latency_seconds": 0.0},
        {"passed": True, "repaired": True, "n_repairs": 1, "latency_seconds": 2.5},
        {"passed": True, "repaired": False, "n_repairs": 0, "latency_seconds": 0.0},
    ]

    summary = module._summarize_runs(
        "humaneval_property",
        baseline_runs,
        verify_only_runs,
        verify_repair_runs,
    )

    assert summary["baseline"]["pass_at_1"] == 2 / 3
    assert summary["baseline"]["mean_latency_seconds"] == 1.333
    assert summary["verify_only"]["execution_only"]["pass_at_1"] == 2 / 3
    assert summary["verify_only"]["execution_only"]["n_wrong_answers"] == 1
    assert summary["verify_only"]["execution_only"]["n_wrong_detected"] == 1
    assert summary["verify_only"]["execution_only"]["false_positives"] == 0
    assert summary["verify_only"]["execution_only"]["mean_latency_seconds"] == 0.3
    assert summary["verify_only"]["execution_plus_property"]["pass_at_1"] == 1 / 3
    assert summary["verify_only"]["execution_plus_property"]["n_wrong_detected"] == 1
    assert summary["verify_only"]["execution_plus_property"]["false_positives"] == 1
    assert summary["verify_only"]["execution_plus_property"]["property_violation_total"] == 3
    assert (
        summary["verify_only"]["execution_plus_property"]["official_test_misses_caught_by_property"]
        == 1
    )
    assert (
        summary["verify_only"]["execution_plus_property"][
            "execution_only_misses_caught_by_property"
        ]
        == 1
    )
    assert summary["verify_only"]["execution_plus_property"]["mean_latency_seconds"] == 0.567
    assert summary["verify_only"]["mean_total_latency_seconds"] == 0.9
    assert summary["verify_repair"]["pass_at_1"] == 1.0
    assert summary["verify_repair"]["n_repaired"] == 1
    assert summary["verify_repair"]["repair_success_rate"] == 1.0
    assert summary["verify_repair"]["mean_latency_seconds"] == 0.833
    assert summary["paired_deltas"]["execution_only_minus_baseline"] == 0.0
    assert summary["paired_deltas"]["execution_plus_property_minus_baseline"] == pytest.approx(
        -(1 / 3)
    )
    assert summary["paired_deltas"]["repair_minus_baseline"] == pytest.approx(1 / 3)
    assert summary["paired_deltas"][
        "execution_plus_property_minus_execution_only"
    ] == pytest.approx(-(1 / 3))


# REQ-VERIFY-029
def test_load_constraint_ir_records_enriches_case_id_and_task_slice(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    module = load_module()
    repo = make_repo(tmp_path)
    benchmark_path = repo / "data" / "research" / "constraint_ir_benchmark_211.jsonl"
    benchmark_path.parent.mkdir(parents=True, exist_ok=True)
    benchmark_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "example_id": "exp211-live-gsm8k-923",
                        "source_family": "live_gsm8k_semantic_failure",
                        "constraint_types": ["compositional", "semantic_grounding"],
                        "prompt": "Question",
                        "gold_atomic_constraints": [],
                        "expected_answer_schema": {"type": "number"},
                    }
                ),
                json.dumps(
                    {
                        "example_id": "exp211-instruction-bullets-1",
                        "source_family": "instruction_following",
                        "constraint_types": ["literal"],
                        "prompt": "Bullets",
                        "gold_atomic_constraints": [],
                        "expected_answer_schema": {"type": "bullet_list"},
                    }
                ),
                json.dumps(
                    {
                        "example_id": "exp211-instruction-grounded-1",
                        "source_family": "instruction_following",
                        "constraint_types": ["compositional", "semantic_grounding"],
                        "prompt": "Grounded",
                        "gold_atomic_constraints": [],
                        "expected_answer_schema": {"type": "comma_separated_list"},
                    }
                ),
                json.dumps(
                    {
                        "example_id": "exp211-code-dedupe-1",
                        "source_family": "code_typed_properties",
                        "constraint_types": ["typed_property", "compositional"],
                        "prompt": "Code",
                        "gold_atomic_constraints": [],
                        "expected_answer_schema": {"type": "python_function"},
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("CARNOT_REPO_ROOT", str(repo))

    records = module._load_benchmark_records("constraint_ir")

    assert [record["case_id"] for record in records] == [
        "exp211-live-gsm8k-923",
        "exp211-instruction-bullets-1",
        "exp211-instruction-grounded-1",
        "exp211-code-dedupe-1",
    ]
    assert [record["task_slice"] for record in records] == [
        "live_gsm8k_semantic_failure",
        "instruction_surface_only",
        "instruction_grounded",
        "code_typed_properties",
    ]


# REQ-VERIFY-029, SCENARIO-VERIFY-029
def test_evaluate_constraint_ir_response_reports_exact_partial_and_semantic_failures():
    module = load_module()
    case = {
        "case_id": "exp221-decision",
        "example_id": "exp221-decision",
        "source_family": "instruction_following",
        "task_slice": "instruction_grounded",
        "constraint_types": ["literal", "semantic_grounding"],
        "prompt": (
            "Choose the strongest launch option and return JSON with keys choice and evidence.\n"
            "O1 | reach 7 | cost 4 | risk high\n"
            "O2 | reach 6 | cost 3 | risk medium\n"
            "O3 | reach 5 | cost 2 | risk low\n"
            "Prefer lower risk, then higher reach. "
            "evidence must list the chosen row ID and the risk label."
        ),
        "expected_answer_schema": {"type": "json_object", "required_keys": ["choice", "evidence"]},
        "gold_atomic_constraints": [
            {
                "constraint_id": "c1",
                "type": "json_exact_keys",
                "target": "json_keys",
                "relation": "equals",
                "value": ["choice", "evidence"],
            },
            {
                "constraint_id": "c2",
                "type": "grounded_selection",
                "target": "choice",
                "relation": "equals",
                "value": "O3",
            },
            {
                "constraint_id": "c3",
                "type": "grounded_evidence_ids",
                "target": "evidence",
                "relation": "equals",
                "value": ["O3", "risk low"],
            },
        ],
    }

    evaluation = module._evaluate_constraint_ir_response(
        case,
        "structured_json",
        '{"choice": "O2", "evidence": ["O2", "risk low"]}',
    )

    assert evaluation["parseable"] is True
    assert evaluation["output_style"] == "structured_json"
    assert evaluation["constraint_extraction_coverage"] == 1.0
    assert evaluation["constraint_coverage"] == 1.0
    assert evaluation["exact_satisfaction"] is False
    assert evaluation["partial_satisfaction"] == pytest.approx(1 / 3)
    assert evaluation["answer_quality"] == pytest.approx(1 / 3)
    assert evaluation["semantic_violation_count"] == 2
    assert evaluation["failure_breakdown"] == {
        "literal": 0,
        "semantic": 2,
        "search_optimization_limited": 0,
    }
    assert [item["status"] for item in evaluation["constraint_results"]] == [
        "satisfied",
        "violated",
        "violated",
    ]
    assert {item["judge"] for item in evaluation["constraint_results"]} == {"deterministic"}


# REQ-VERIFY-029, SCENARIO-VERIFY-029
def test_evaluate_constraint_ir_response_bounds_non_terminating_code_probes(
    monkeypatch: pytest.MonkeyPatch,
):
    module = load_module()
    monkeypatch.setattr(module, "_CONSTRAINT_IR_CODE_PROBE_TIMEOUT_SECONDS", 0.01)
    case = {
        "case_id": "exp211-code-toposort-timeout",
        "example_id": "exp211-code-toposort-timeout",
        "source_family": "code_typed_properties",
        "task_slice": "code_typed_properties",
        "constraint_types": ["typed_property"],
        "prompt": "Write topo_sort(edges).",
        "expected_answer_schema": {"type": "python_function", "name": "topo_sort"},
        "gold_atomic_constraints": [
            {
                "constraint_id": "name",
                "type": "function_name",
                "value": "topo_sort",
            },
            {
                "constraint_id": "typed-property",
                "type": "typed_property",
                "value": "probe",
            },
        ],
    }
    response = """```python
def topo_sort(edges):
    while True:
        pass
```"""

    evaluation = module._evaluate_constraint_ir_response(
        case,
        "answer_only_terse",
        response,
    )

    assert evaluation["parseable"] is True
    assert evaluation["exact_satisfaction"] is False
    assert evaluation["partial_satisfaction"] == 0.5
    assert evaluation["failure_breakdown"]["search_optimization_limited"] == 1
    probe_result = next(
        item
        for item in evaluation["constraint_results"]
        if item["constraint_id"] == "typed-property"
    )
    assert probe_result["status"] == "violated"
    assert probe_result["details"]["failure_mode"] == "timeout"
    assert probe_result["details"]["stage"] == "probe_call"
    assert probe_result["details"]["timeout_seconds"] == pytest.approx(0.01)


# REQ-VERIFY-029
def test_constraint_ir_helpers_cover_parser_and_probe_edge_cases(
    monkeypatch: pytest.MonkeyPatch,
):
    module = load_module()
    monkeypatch.setattr(module, "_CONSTRAINT_IR_CODE_PROBE_TIMEOUT_SECONDS", 0.01)

    assert module._parse_json_payload('prefix {"a": 1} suffix') == {"a": 1}
    assert module._parse_json_payload("prefix {bad json} suffix") is None
    assert module._unwrap_answer_candidate('{"final_answer": {"value": 3}, "checks": ["ok"]}') == (
        {"value": 3},
        ["ok"],
    )
    assert module._stringify_answer_value(None) == ""
    assert module._stringify_answer_value(9) == "9"
    assert module._parse_number_answer(True) is None
    assert module._parse_number_answer(7) == 7
    assert module._parse_number_answer(7.9) == 7
    assert module._parse_bullets(["alpha", " ", "beta"]) == ["alpha", "beta"]
    assert module._parse_bullets("- alpha\n* beta") == ["alpha", "beta"]
    assert module._parse_comma_items(["alpha", " ", "beta"]) == ["alpha", "beta"]
    assert module._parse_comma_items('["alpha", "beta"]') == ["alpha", "beta"]
    assert module._parse_comma_items("[bad json]") == ["[bad json]"]
    assert module._parse_flat_yaml_object({"count": 2}) == {"count": 2}
    assert module._parse_flat_yaml_object("") is None
    assert module._parse_flat_yaml_object("not yaml") is None
    assert module._parse_flat_yaml_object(": value") is None
    assert module._parse_flat_yaml_object("\ncount: 3\nenabled: true\nowner: ops") == {
        "count": 3,
        "enabled": True,
        "owner": "ops",
    }
    assert module._parse_flat_yaml_object("count: 3\n\nowner: ops") == {
        "count": 3,
        "owner": "ops",
    }
    assert module._parse_numbered_steps("1. collect\nignore\n2. ship") == ["collect", "ship"]
    assert module._extract_python_code(None) is None
    assert module._extract_python_code("```\ndef helper():\n    return 1\n```") == (
        "def helper():\n    return 1"
    )
    assert module._extract_python_code("before\ndef helper():\n    return 1") == (
        "def helper():\n    return 1"
    )

    function_node = ast.parse("def helper(x: int) -> str:\n    return str(x)\n").body[0]
    assert module._function_signature_from_ast(function_node) == "helper(x: int) -> str"
    assert (
        module._constraint_family(
            make_constraint_case(schema_type="bullet_list"), {"type": "count_exact"}
        )
        == "literal"
    )
    assert (
        module._constraint_family(
            make_constraint_case(constraint_types=["semantic_grounding"]),
            {"type": "grounded_selection"},
        )
        == "semantic"
    )
    assert (
        module._constraint_family(
            make_constraint_case(
                schema_type="python_function",
                constraint_types=["typed_property"],
                expected_answer_schema={"type": "python_function", "name": "custom"},
                source_family="code_typed_properties",
                task_slice="code_typed_properties",
            ),
            {"type": "typed_property"},
        )
        == "search_optimization_limited"
    )

    assert module._ast_safe_eval(ast.parse("-(x + 2) / 2", mode="eval"), {"x": 2}) == pytest.approx(
        -2.0
    )
    assert module._ast_safe_eval(ast.parse("x * 3 - 1", mode="eval"), {"x": 2}) == pytest.approx(
        5.0
    )
    with pytest.raises(ValueError):
        module._ast_safe_eval(ast.parse("missing", mode="eval"), {"x": 2})
    with pytest.raises(ValueError):
        module._ast_safe_eval(ast.parse("x ** 2", mode="eval"), {"x": 2})

    assert module._resolved_constraint_values(
        {
            "gold_atomic_constraints": [
                {"target": "total", "value": 2},
                {"target": "delta", "value": "total + 3"},
                {"target": "blocked", "value": "missing + 1"},
            ]
        }
    ) == {"total": 2.0, "delta": 5.0}
    assert module._looks_calm_professional("Calm handoff.")
    assert not module._looks_calm_professional("panic mode")
    topo_probe = {"nodes": ["a", "b"], "edges": [("a", "b")]}
    assert not module._topological_order_valid("oops", topo_probe)
    assert not module._topological_order_valid(["a"], topo_probe)
    assert not module._topological_order_valid(["b", "a"], topo_probe)
    assert module._run_with_timeout(lambda: "ok", timeout_seconds=0.0) == "ok"

    def raises_expected(_value: object) -> None:
        raise ValueError("bad")

    assert module._run_code_probe(
        raises_expected, {"args": ["x"], "expect_exception": "ValueError"}
    ) == (
        True,
        None,
    )
    assert module._run_code_probe(
        lambda value: value, {"args": ["x"], "expect_exception": "ValueError"}
    ) == (
        False,
        {"failure_mode": "missing_expected_exception", "stage": "probe_call"},
    )
    assert module._run_code_probe(lambda value: value + 1, {"args": [1], "expected": 2}) == (
        True,
        None,
    )
    assert module._run_code_probe(lambda value: value + 2, {"args": [1], "expected": 2}) == (
        False,
        {"failure_mode": "wrong_result", "stage": "probe_call"},
    )

    def raises_unexpected(_value: object) -> None:
        raise RuntimeError("bad")

    passed, details = module._run_code_probe(raises_unexpected, {"args": ["x"]})
    assert passed is False
    assert details == {
        "failure_mode": "exception",
        "stage": "probe_call",
        "exception_type": "RuntimeError",
    }

    def mutates(items: list[str]) -> list[str]:
        items.append("z")
        return items

    passed, details = module._run_code_probe(mutates, {"args": [["a"]], "immutable_arg_index": 0})
    assert passed is False
    assert details == {"failure_mode": "mutated_input", "stage": "probe_call"}

    def wrong_topo(_edges: list[tuple[str, str]]) -> list[str]:
        return ["b", "a"]

    passed, details = module._run_code_probe(
        wrong_topo,
        {
            "args": [[("a", "b")]],
            "validator": "topological_order",
            "nodes": ["a", "b"],
            "edges": [("a", "b")],
        },
    )
    assert passed is False
    assert details == {"failure_mode": "wrong_result", "stage": "probe_call"}

    def loops(_edges: list[tuple[str, str]]) -> list[str]:
        while True:
            pass

    passed, details = module._run_code_probe(
        loops,
        {
            "args": [[("a", "b")]],
            "validator": "topological_order",
            "nodes": ["a", "b"],
            "edges": [("a", "b")],
        },
    )
    assert passed is False
    assert details == {
        "failure_mode": "timeout",
        "stage": "probe_call",
        "timeout_seconds": 0.01,
    }

    assert module._code_probe_success("unknown", lambda: None) == (
        False,
        {"failure_mode": "missing_probe", "stage": "probe_setup"},
    )

    def good_topo(edges: list[tuple[str, str]]) -> list[str]:
        if ("a", "b") in edges and ("b", "a") in edges:
            raise ValueError("cycle")
        return ["a", "b", "c"]

    assert module._code_probe_success("topo_sort", good_topo) == (True, {})
    assert (
        module._classify_constraint_ir_output_style(
            "REASONING:\nFINAL: answer",
            structured_payload=None,
            schema_type="number",
            code_text=None,
        )
        == "free_form_reasoning"
    )
    assert (
        module._classify_constraint_ir_output_style(
            "line one\nline two",
            structured_payload=None,
            schema_type="number",
            code_text=None,
        )
        == "free_form_reasoning"
    )
    assert (
        module._classify_constraint_ir_output_style(
            "answer",
            structured_payload=None,
            schema_type="number",
            code_text=None,
        )
        == "answer_only_terse"
    )
    assert (
        module._classify_constraint_ir_output_style(
            "",
            structured_payload=None,
            schema_type="number",
            code_text=None,
        )
        == "other_unstructured"
    )


# REQ-VERIFY-029
def test_build_constraint_ir_context_covers_exec_failures_and_missing_probes(
    monkeypatch: pytest.MonkeyPatch,
):
    module = load_module()
    monkeypatch.setattr(module, "_CONSTRAINT_IR_CODE_PROBE_TIMEOUT_SECONDS", 0.01)
    case = make_constraint_case(
        schema_type="python_function",
        constraint_types=["typed_property"],
        expected_answer_schema={"type": "python_function", "name": "topo_sort"},
        source_family="code_typed_properties",
        task_slice="code_typed_properties",
    )

    timeout_context = module._build_constraint_ir_context(
        case,
        "answer_only_terse",
        "```python\ndef topo_sort(edges):\n    return []\nwhile True:\n    pass\n```",
    )
    assert timeout_context["code_probe_details"]["failure_mode"] == "timeout"
    assert timeout_context["code_probe_details"]["stage"] == "exec"
    assert timeout_context["function_obj"] is None

    exception_context = module._build_constraint_ir_context(
        case,
        "answer_only_terse",
        "```python\ndef topo_sort(edges):\n    return []\nraise RuntimeError('boom')\n```",
    )
    assert exception_context["code_probe_details"] == {
        "failure_mode": "exception",
        "stage": "exec",
    }
    assert exception_context["function_obj"] is None

    syntax_context = module._build_constraint_ir_context(
        case,
        "answer_only_terse",
        "```python\ndef topo_sort(\n```",
    )
    assert syntax_context["function_node"] is None
    assert syntax_context["function_obj"] is None

    missing_probe_case = make_constraint_case(
        schema_type="python_function",
        constraint_types=["typed_property"],
        expected_answer_schema={"type": "python_function", "name": "custom"},
        source_family="code_typed_properties",
        task_slice="code_typed_properties",
    )
    missing_probe_context = module._build_constraint_ir_context(
        missing_probe_case,
        "answer_only_terse",
        "```python\ndef custom(value):\n    return value\n```",
    )
    assert missing_probe_context["code_probe_details"] == {
        "failure_mode": "missing_probe",
        "stage": "probe_setup",
    }


# REQ-VERIFY-029
def test_constraint_ir_constraint_evaluator_covers_remaining_branches():
    module = load_module()
    function_node = ast.parse(
        "def topo_sort(edges: list[tuple[str, str]]) -> list[str]:\n    return []\n"
    ).body[0]
    python_case = make_constraint_case(
        schema_type="python_function",
        constraint_types=["typed_property"],
        expected_answer_schema={"type": "python_function", "name": "topo_sort"},
        source_family="code_typed_properties",
        task_slice="code_typed_properties",
    )
    python_context = make_constraint_context(
        schema_type="python_function",
        function_node=function_node,
        function_obj=object(),
        function_name="topo_sort",
        function_signature=module._function_signature_from_ast(function_node),
        code_text="def topo_sort(edges):\n    return sorted(edges)\n",
        code_probes_pass=True,
        code_probe_details={},
    )

    assert (
        module._evaluate_constraint_ir_constraint(
            python_case,
            {
                "constraint_id": "signature",
                "type": "signature",
                "value": "topo_sort(edges: list[tuple[str, str]]) -> list[str]",
            },
            python_context,
        )["status"]
        == "satisfied"
    )
    return_type_result = module._evaluate_constraint_ir_constraint(
        python_case,
        {"constraint_id": "return", "type": "return_type", "value": "list[int]"},
        python_context,
    )
    assert return_type_result["status"] == "violated"
    assert return_type_result["details"]["observed"] == "list[str]"
    assert (
        module._evaluate_constraint_ir_constraint(
            python_case,
            {"constraint_id": "forbidden", "type": "forbidden_api", "value": ["sorted("]},
            python_context,
        )["status"]
        == "violated"
    )
    time_complexity = module._evaluate_constraint_ir_constraint(
        python_case,
        {"constraint_id": "time", "type": "time_complexity", "value": "O(n)"},
        dict(
            python_context, code_probes_pass=False, code_probe_details={"failure_mode": "timeout"}
        ),
    )
    assert time_complexity["status"] == "violated"
    assert time_complexity["details"]["failure_mode"] == "timeout"
    assert (
        module._evaluate_constraint_ir_constraint(
            python_case,
            {"constraint_id": "typed", "type": "typed_property", "value": "probe"},
            dict(
                python_context,
                function_obj=None,
                code_probe_details={"failure_mode": "exception", "stage": "exec"},
            ),
        )["details"]["stage"]
        == "exec"
    )

    list_context = make_constraint_context(
        schema_type="bullet_list",
        bullet_answer=["alpha beta", "gamma delta"],
        answer_text="- alpha beta\n- gamma delta",
        raw_response="- alpha beta\n- gamma delta",
        normalized_answer=module._normalize_surface("- alpha beta gamma delta"),
        normalized_response=module._normalize_surface("- alpha beta gamma delta"),
        sections=[("Status", "Ready now."), ("Notes", "Escalate slowly.")],
        sentences=["Use the launch checklist.", "Keep operators calm."],
        numbered_steps=["collect baseline metrics", "validate result"],
        yaml_answer={"owner": "ops", "count": 3, "enabled": True},
        json_answer={"mode": "ship", "evidence": "O3 risk low"},
        comma_items=["O3", "O1"],
        identifier="O3",
        parsed_number=5,
        resolved_values={"final_total": 5.0, "total": 5.0},
    )
    instruction_case = make_constraint_case()
    semantic_case = make_constraint_case(constraint_types=["semantic_grounding"])

    assert (
        module._evaluate_constraint_ir_constraint(
            instruction_case,
            {"constraint_id": "count-bullets", "type": "count_exact", "value": 2},
            list_context,
        )["status"]
        == "satisfied"
    )
    assert (
        module._evaluate_constraint_ir_constraint(
            make_constraint_case(schema_type="comma_separated_list"),
            {"constraint_id": "count-commas", "type": "count_exact", "value": 1},
            dict(list_context, schema_type="comma_separated_list"),
        )["status"]
        == "violated"
    )
    assert (
        module._evaluate_constraint_ir_constraint(
            make_constraint_case(schema_type="number"),
            {"constraint_id": "count-missing", "type": "count_exact", "value": 1},
            make_constraint_context(schema_type="number"),
        )["status"]
        == "violated"
    )
    assert (
        module._evaluate_constraint_ir_constraint(
            instruction_case,
            {
                "constraint_id": "words-bullets",
                "type": "word_count_range",
                "target": "bullet_word_count",
                "value": [2, 2],
            },
            list_context,
        )["status"]
        == "satisfied"
    )
    assert (
        module._evaluate_constraint_ir_constraint(
            make_constraint_case(schema_type="two_sentences"),
            {"constraint_id": "words-response", "type": "word_count_range", "value": [1, 2]},
            make_constraint_context(
                schema_type="two_sentences",
                answer_text="This answer is too long.",
                raw_response="This answer is too long.",
            ),
        )["status"]
        == "violated"
    )
    assert (
        module._evaluate_constraint_ir_constraint(
            instruction_case,
            {
                "constraint_id": "must-s1",
                "type": "must_include_token",
                "target": "sentence_1",
                "value": "launch",
            },
            list_context,
        )["status"]
        == "satisfied"
    )
    assert (
        module._evaluate_constraint_ir_constraint(
            instruction_case,
            {
                "constraint_id": "must-section",
                "type": "must_include_phrase",
                "target": "notes",
                "value": "escalate",
            },
            list_context,
        )["status"]
        == "satisfied"
    )
    assert (
        module._evaluate_constraint_ir_constraint(
            instruction_case,
            {"constraint_id": "forbidden", "type": "forbidden_token", "value": "panic"},
            list_context,
        )["status"]
        == "satisfied"
    )
    assert (
        module._evaluate_constraint_ir_constraint(
            semantic_case,
            {"constraint_id": "no-extra", "type": "no_extra_keys", "value": ["mode"]},
            list_context,
        )["status"]
        == "violated"
    )
    assert (
        module._evaluate_constraint_ir_constraint(
            semantic_case,
            {
                "constraint_id": "enum",
                "type": "enum_membership",
                "target": "mode",
                "value": ["hold", "ship"],
            },
            list_context,
        )["status"]
        == "satisfied"
    )
    assert (
        module._evaluate_constraint_ir_constraint(
            make_constraint_case(schema_type="markdown_sections"),
            {
                "constraint_id": "section-order",
                "type": "section_order",
                "value": ["Status", "Notes"],
            },
            dict(list_context, schema_type="markdown_sections"),
        )["status"]
        == "satisfied"
    )
    assert (
        module._evaluate_constraint_ir_constraint(
            make_constraint_case(schema_type="markdown_sections"),
            {
                "constraint_id": "sentences-per-section",
                "type": "sentence_count_per_section",
                "value": 1,
            },
            dict(list_context, schema_type="markdown_sections"),
        )["status"]
        == "satisfied"
    )
    assert (
        module._evaluate_constraint_ir_constraint(
            semantic_case,
            {"constraint_id": "select-list", "type": "grounded_selection", "value": ["O3", "O1"]},
            dict(list_context, schema_type="comma_separated_list"),
        )["status"]
        == "satisfied"
    )
    assert (
        module._evaluate_constraint_ir_constraint(
            semantic_case,
            {
                "constraint_id": "select-json",
                "type": "grounded_selection",
                "target": "mode",
                "value": "ship",
            },
            list_context,
        )["status"]
        == "satisfied"
    )
    assert (
        module._evaluate_constraint_ir_constraint(
            make_constraint_case(schema_type="identifier"),
            {"constraint_id": "select-id", "type": "grounded_selection", "value": "O3"},
            dict(list_context, schema_type="identifier", json_answer=None, comma_items=["O3"]),
        )["status"]
        == "satisfied"
    )
    assert (
        module._evaluate_constraint_ir_constraint(
            semantic_case,
            {
                "constraint_id": "evidence",
                "type": "grounded_evidence_ids",
                "value": ["O3", "risk", "low"],
            },
            dict(list_context, json_answer={"evidence": "O3 risk low"}),
        )["status"]
        == "satisfied"
    )

    ordering_case = make_constraint_case(
        schema_type="comma_separated_list",
        gold_atomic_constraints=[
            {"constraint_id": "choose", "type": "grounded_selection", "value": ["O3", "O1"]}
        ],
    )
    assert (
        module._evaluate_constraint_ir_constraint(
            ordering_case,
            {"constraint_id": "ordering", "type": "ordering", "value": "match"},
            dict(list_context, schema_type="comma_separated_list"),
        )["status"]
        == "satisfied"
    )
    assert (
        module._evaluate_constraint_ir_constraint(
            make_constraint_case(schema_type="numbered_list"),
            {"constraint_id": "step-count", "type": "step_count", "value": 2},
            dict(list_context, schema_type="numbered_list"),
        )["status"]
        == "satisfied"
    )
    assert (
        module._evaluate_constraint_ir_constraint(
            make_constraint_case(schema_type="numbered_list"),
            {
                "constraint_id": "step-roles-short",
                "type": "step_roles",
                "value": ["baseline_metrics"],
            },
            dict(list_context, schema_type="numbered_list"),
        )["status"]
        == "violated"
    )
    assert (
        module._evaluate_constraint_ir_constraint(
            make_constraint_case(schema_type="numbered_list"),
            {
                "constraint_id": "step-roles",
                "type": "step_roles",
                "value": ["baseline_metrics", "validate_result"],
            },
            dict(list_context, schema_type="numbered_list"),
        )["status"]
        == "satisfied"
    )
    assert (
        module._evaluate_constraint_ir_constraint(
            make_constraint_case(schema_type="numbered_list"),
            {
                "constraint_id": "step-roles-bad",
                "type": "step_roles",
                "value": ["baseline_metrics", "validate_result"],
            },
            dict(
                list_context,
                schema_type="numbered_list",
                numbered_steps=["collect evidence", "validate result"],
            ),
        )["status"]
        == "violated"
    )
    assert (
        module._evaluate_constraint_ir_constraint(
            make_constraint_case(schema_type="yaml_object"),
            {
                "constraint_id": "yaml-keys",
                "type": "yaml_exact_keys",
                "value": ["owner", "count", "enabled"],
            },
            dict(list_context, schema_type="yaml_object"),
        )["status"]
        == "satisfied"
    )
    assert (
        module._evaluate_constraint_ir_constraint(
            make_constraint_case(schema_type="yaml_object"),
            {"constraint_id": "derived", "type": "derived_value", "target": "count", "value": 3},
            dict(list_context, schema_type="yaml_object"),
        )["status"]
        == "satisfied"
    )

    negation_case = make_constraint_case(
        schema_type="comma_separated_list",
        gold_atomic_constraints=[
            {"constraint_id": "choose", "type": "grounded_selection", "value": ["O3", "O1"]}
        ],
    )
    assert (
        module._evaluate_constraint_ir_constraint(
            negation_case,
            {"constraint_id": "negation", "type": "negation_scope", "value": "match"},
            dict(list_context, schema_type="comma_separated_list"),
        )["status"]
        == "satisfied"
    )
    assert (
        module._evaluate_constraint_ir_constraint(
            make_constraint_case(schema_type="two_sentences"),
            {"constraint_id": "sentence-count", "type": "sentence_count", "value": 2},
            dict(list_context, schema_type="two_sentences"),
        )["status"]
        == "satisfied"
    )
    assert (
        module._evaluate_constraint_ir_constraint(
            instruction_case,
            {"constraint_id": "tone", "type": "tone", "value": "calm"},
            dict(list_context, answer_text="panic mode"),
        )["status"]
        == "violated"
    )

    live_context = make_constraint_context(
        schema_type="number",
        parsed_number=None,
        normalized_response=module._normalize_surface("final total 5 units"),
        resolved_values={"total": 5.0},
    )
    assert (
        module._evaluate_live_prompt_constraint(
            semantic_case,
            {"constraint_id": "binding-missing", "type": "final_answer_binding", "target": "total"},
            live_context,
        )["status"]
        == "violated"
    )
    assert (
        module._evaluate_live_prompt_constraint(
            semantic_case,
            {"constraint_id": "binding-match", "type": "final_answer_binding", "target": "total"},
            dict(live_context, parsed_number=5),
        )["status"]
        == "satisfied"
    )
    assert (
        module._evaluate_live_prompt_constraint(
            semantic_case,
            {
                "constraint_id": "fragment-match",
                "type": "entity_binding",
                "target": "total",
                "value": 5,
                "unit": "units",
            },
            live_context,
        )["status"]
        == "satisfied"
    )
    assert (
        module._evaluate_constraint_ir_constraint(
            semantic_case,
            {
                "constraint_id": "fallback-unjudged",
                "type": "entity_binding",
                "target": "other",
                "value": "absent",
            },
            dict(live_context, normalized_response="nothing relevant"),
        )["status"]
        == "unjudged"
    )


# REQ-VERIFY-029, SCENARIO-VERIFY-029
def test_summarize_constraint_ir_runs_reports_prompt_side_metrics_and_breakdowns():
    module = load_module()

    baseline_runs = [
        {
            "evaluation": {
                "parseable": True,
                "output_style": "structured_json",
                "constraint_extraction_coverage": 1.0,
                "exact_satisfaction": True,
                "partial_satisfaction": 1.0,
                "semantic_violation_count": 0,
                "failure_breakdown": {
                    "literal": 0,
                    "semantic": 0,
                    "search_optimization_limited": 0,
                },
            }
        },
        {
            "evaluation": {
                "parseable": True,
                "output_style": "free_form_reasoning",
                "constraint_extraction_coverage": 0.5,
                "exact_satisfaction": False,
                "partial_satisfaction": 0.25,
                "semantic_violation_count": 1,
                "failure_breakdown": {
                    "literal": 0,
                    "semantic": 1,
                    "search_optimization_limited": 1,
                },
            }
        },
    ]
    verify_only_runs = [
        {
            "verified": True,
            "flagged": False,
            "evaluation": dict(baseline_runs[0]["evaluation"]),
        },
        {
            "verified": False,
            "flagged": True,
            "evaluation": dict(baseline_runs[1]["evaluation"]),
        },
    ]
    verify_repair_runs = [
        {
            "verified": True,
            "repaired": False,
            "n_repairs": 0,
            "evaluation": dict(baseline_runs[0]["evaluation"]),
        },
        {
            "verified": True,
            "repaired": True,
            "n_repairs": 1,
            "evaluation": {
                "parseable": True,
                "output_style": "structured_json",
                "constraint_extraction_coverage": 1.0,
                "exact_satisfaction": True,
                "partial_satisfaction": 1.0,
                "semantic_violation_count": 0,
                "failure_breakdown": {
                    "literal": 0,
                    "semantic": 0,
                    "search_optimization_limited": 0,
                },
            },
        },
    ]

    summary = module._summarize_runs(
        "constraint_ir",
        baseline_runs,
        verify_only_runs,
        verify_repair_runs,
    )

    assert summary["baseline"]["parse_success_rate"] == 1.0
    assert summary["baseline"]["mean_constraint_extraction_coverage"] == 0.75
    assert summary["baseline"]["exact_satisfaction_rate"] == 0.5
    assert summary["baseline"]["mean_partial_satisfaction"] == 0.625
    assert summary["baseline"]["semantic_violation_count"] == 1
    assert summary["baseline"]["failures_by_constraint_family"] == {
        "literal": 0,
        "semantic": 1,
        "search_optimization_limited": 1,
    }
    assert summary["baseline"]["behavior_by_output_style"]["structured_json"]["n_cases"] == 1
    assert (
        summary["baseline"]["behavior_by_output_style"]["structured_json"][
            "exact_satisfaction_rate"
        ]
        == 1.0
    )
    assert (
        summary["baseline"]["behavior_by_output_style"]["free_form_reasoning"][
            "exact_satisfaction_rate"
        ]
        == 0.0
    )
    assert summary["verify_only"]["verified_rate"] == 0.5
    assert summary["verify_only"]["semantic_violation_count"] == 1
    assert summary["verify_only"]["cases_with_semantic_violations"] == 1
    assert summary["verify_repair"]["verified_rate"] == 1.0
    assert summary["verify_repair"]["n_repaired"] == 1
    assert summary["verify_repair"]["repair_yield"] == 1.0
    assert summary["verify_repair"]["mean_partial_satisfaction"] == 1.0
    assert summary["paired_deltas"]["repair_minus_baseline_exact"] == 0.5


@dataclass
class _FakeSerializable:
    payload: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        return dict(self.payload)


@dataclass
class _FakeDataclassPayload:
    label: str


# REQ-VERIFY-027, SCENARIO-VERIFY-027
def test_serialize_verification_result_preserves_semantic_trace_artifacts():
    module = load_module()

    verification = SimpleNamespace(
        verified=False,
        energy=1.25,
        constraints=[],
        violations=[],
        certificate={"n_constraints": 2, "n_violations": 1},
        typed_reasoning=_FakeSerializable(
            {
                "provenance": {
                    "extraction_method": "direct_json",
                    "parser_version": "20260412",
                }
            }
        ),
        semantic_grounding=_FakeSerializable(
            {
                "violations": [
                    {
                        "violation_type": "answer_target_mismatch",
                        "description": "Answered the wrong target.",
                    }
                ]
            }
        ),
    )

    serialized = module._serialize_verification_result(verification)

    assert serialized["verified"] is False
    assert serialized["energy"] == 1.25
    assert serialized["certificate"] == {"n_constraints": 2, "n_violations": 1}
    assert serialized["typed_reasoning"]["provenance"]["extraction_method"] == "direct_json"
    assert serialized["semantic_grounding"]["violations"][0]["violation_type"] == (
        "answer_target_mismatch"
    )


# REQ-VERIFY-028
def test_run_humaneval_baseline_preserves_generation_trace(monkeypatch: pytest.MonkeyPatch):
    module = load_module()
    from carnot.pipeline import humaneval_live_benchmark as humaneval_module

    class FakeTokenizer:
        def __call__(self, text: str, **_: object) -> dict[str, list[int]]:
            return {"input_ids": list(range(len(text.split())))}

    monkeypatch.setattr(module, "_generate_text", lambda **_: "return x + 1")
    monkeypatch.setattr(
        humaneval_module,
        "build_candidate_code",
        lambda prompt, body: f"{prompt.strip()}::{body}",
    )
    monkeypatch.setattr(
        humaneval_module,
        "execute_humaneval",
        lambda code, problem, timeout=5.0: humaneval_module.HarnessResult(
            passed=True,
            error_type="none",
            error_message="",
            stdout="",
        ),
    )
    monkeypatch.setattr(
        humaneval_module,
        "run_instrumentation",
        lambda code, prompt, entry_point, official_tests=None: {
            "detected": False,
            "n_property_violations": 0,
        },
    )
    perf = iter([1.0, 1.4])
    monkeypatch.setattr(module.time, "perf_counter", lambda: next(perf))

    result = module._run_humaneval_baseline(
        {
            "case_id": "humaneval-1",
            "prompt": 'def add_one(x: int) -> int:\n    """Return x + 1."""\n',
            "entry_point": "add_one",
            "test": "def check(candidate):\n    assert candidate(1) == 2\n",
            "prompt_seeds": {"baseline": 17},
        },
        model=object(),
        tokenizer=FakeTokenizer(),
    )

    assert result["generation_trace"]["attempts"][0]["response"] == "return x + 1"
    assert result["prompt_tokens"] > 0
    assert result["response_tokens"] == 4
    assert result["total_tokens"] == result["prompt_tokens"] + result["response_tokens"]
    assert result["latency_seconds"] == 0.4


# REQ-VERIFY-028, SCENARIO-VERIFY-028
def test_run_humaneval_verify_only_records_property_only_catches_and_latency(
    monkeypatch: pytest.MonkeyPatch,
):
    module = load_module()
    from carnot.pipeline import humaneval_live_benchmark as humaneval_module

    execution_only = {"detected": False, "n_property_violations": 0}
    execution_plus_property = {"detected": True, "n_property_violations": 2}

    def fake_run_instrumentation(
        code: str,
        prompt: str,
        entry_point: str,
        official_tests: str | None = None,
    ) -> dict[str, object]:
        return execution_only if official_tests is None else execution_plus_property

    monkeypatch.setattr(humaneval_module, "run_instrumentation", fake_run_instrumentation)
    monkeypatch.setattr(
        humaneval_module,
        "execute_humaneval",
        lambda code, problem, timeout=5.0: humaneval_module.HarnessResult(
            passed=True,
            error_type="none",
            error_message="",
            stdout="",
        ),
    )
    perf = iter([10.0, 10.2, 10.5, 10.9])
    monkeypatch.setattr(module.time, "perf_counter", lambda: next(perf))

    result = module._run_humaneval_verify_only(
        {
            "case_id": "humaneval-2",
            "prompt": "def sort_numbers(nums):\n",
            "entry_point": "sort_numbers",
            "test": "def check(candidate):\n    assert candidate([1, 2]) == [1, 2]\n",
            "prompt_seeds": {"verify_only": 29},
        },
        {
            "candidate_code": "def sort_numbers(nums):\n    return nums\n",
            "passed": True,
        },
    )

    assert result["execution_only_accepted"] is True
    assert result["execution_plus_property_accepted"] is False
    assert result["property_only_detected"] is True
    assert result["official_test_miss_caught_by_property"] is True
    assert result["execution_only_latency_seconds"] == 0.2
    assert result["execution_plus_property_latency_seconds"] == 0.3
    assert result["latency_seconds"] == 0.9


# REQ-VERIFY-028
def test_run_humaneval_verify_repair_records_history_and_generation_trace(
    monkeypatch: pytest.MonkeyPatch,
):
    module = load_module()
    from carnot.pipeline import humaneval_live_benchmark as humaneval_module

    class FakeTokenizer:
        def __call__(self, text: str, **_: object) -> dict[str, list[int]]:
            return {"input_ids": list(range(len(text.split())))}

    monkeypatch.setattr(module, "_generate_text", lambda **_: "return x + 1")
    monkeypatch.setattr(
        humaneval_module, "build_repair_prompt", lambda *args, **kwargs: "repair prompt"
    )
    monkeypatch.setattr(
        humaneval_module,
        "build_candidate_code",
        lambda prompt, body: f"CODE::{body}",
    )
    monkeypatch.setattr(
        humaneval_module,
        "execute_humaneval",
        lambda code, problem, timeout=5.0: humaneval_module.HarnessResult(
            passed=True,
            error_type="none",
            error_message="",
            stdout="",
        ),
    )
    monkeypatch.setattr(
        humaneval_module,
        "run_instrumentation",
        lambda code, prompt, entry_point, official_tests=None: {
            "detected": False,
            "n_property_violations": 0,
        },
    )
    perf = iter([20.0, 21.25])
    monkeypatch.setattr(module.time, "perf_counter", lambda: next(perf))

    result = module._run_humaneval_verify_repair(
        {
            "case_id": "humaneval-3",
            "prompt": 'def add_one(x: int) -> int:\n    """Return x + 1."""\n',
            "entry_point": "add_one",
            "test": "def check(candidate):\n    assert candidate(1) == 2\n",
            "prompt_seeds": {"verify_repair": 101},
        },
        {
            "body": "return x - 1",
            "candidate_code": "CODE::return x - 1",
            "passed": False,
            "error_type": "failure",
            "error_message": "AssertionError",
            "instrumentation": {"detected": True, "n_property_violations": 1},
        },
        model=object(),
        tokenizer=FakeTokenizer(),
        max_repairs=3,
    )

    assert result["passed"] is True
    assert result["repaired"] is True
    assert result["n_repairs"] == 1
    assert len(result["history"]) == 2
    assert result["history"][0]["harness"]["error_message"] == "AssertionError"
    assert result["history"][1]["repair_prompt"] == "repair prompt"
    assert result["history"][1]["generation_trace"]["attempts"][0]["response"] == "return x + 1"
    assert result["prompt_tokens"] > 0
    assert result["total_tokens"] == result["prompt_tokens"] + result["response_tokens"]
    assert result["latency_seconds"] == 1.25


# REQ-VERIFY-027
def test_helper_serializers_cover_env_token_and_generation_branches(
    monkeypatch: pytest.MonkeyPatch,
):
    module = load_module()

    monkeypatch.delenv("CARNOT_FORCE_LIVE", raising=False)
    monkeypatch.delenv("CARNOT_FORCE_CPU", raising=False)
    assert module.live_inference_mode() == "simulated"

    monkeypatch.setenv("CARNOT_FORCE_LIVE", "1")
    monkeypatch.setenv("CARNOT_FORCE_CPU", "0")
    assert module.live_inference_mode() == "live_gpu"

    monkeypatch.setenv("CARNOT_FORCE_CPU", "1")
    assert module.live_inference_mode() == "live_cpu"

    class FakeTokenizer:
        def __call__(self, text: str, **_: object) -> dict[str, list[int]]:
            return {"input_ids": list(range(len(text.split())))}

    class RaisingTokenizer:
        def __call__(self, text: str, **_: object) -> dict[str, list[int]]:
            raise RuntimeError(text)

    assert module._token_count(None, "") == 0
    assert module._token_count(None, "alpha beta") == 2
    assert module._token_count(FakeTokenizer(), "alpha beta gamma") == 3
    assert module._token_count(RaisingTokenizer(), "alpha beta") == 2
    assert module._round_mean([]) == 0.0
    assert module._round_mean([1.0, 2.0]) == 1.5

    typed_with_attr = SimpleNamespace(provenance=SimpleNamespace(extraction_method="direct_json"))
    assert module._typed_reasoning_parse_status(None) == "unavailable"
    assert module._typed_reasoning_parse_status(typed_with_attr) == "direct_json"
    assert module._typed_reasoning_parse_status(_FakeSerializable({})) == "parsed"

    assert module._serialize_jsonable(("alpha", 2)) == ["alpha", 2]
    assert module._serialize_jsonable(_FakeDataclassPayload(label="beta")) == {"label": "beta"}
    assert module._serialize_jsonable(object()).startswith("<object object at ")
    assert module._serialize_constraint_result({"kind": "cached"}) == {"kind": "cached"}
    constraint = SimpleNamespace(
        constraint_type="semantic_grounding",
        description="needs review",
        metadata={"satisfied": False},
    )
    assert module._serialize_constraint_result(constraint) == {
        "constraint_type": "semantic_grounding",
        "description": "needs review",
        "metadata": {"satisfied": False},
    }

    trace = module._build_generation_trace(
        tokenizer=FakeTokenizer(),
        attempts=[
            module._serialize_generation_attempt(
                prompt="prompt one",
                response="answer one",
                tokenizer=FakeTokenizer(),
                valid=True,
                error="schema",
            )
        ],
        fallback_record={"prompt": "retry prompt", "response": "retry answer"},
    )

    assert trace["fallback_used"] is True
    assert trace["attempts"][0]["valid"] is True
    assert trace["attempts"][0]["error"] == "schema"
    assert trace["total_tokens"] == 8


# REQ-VERIFY-026
def test_write_artifact_creates_parent_directories_and_trailing_newline(tmp_path: Path):
    module = load_module()
    output_path = tmp_path / "nested" / "artifact.json"
    payload = {"experiment": 218}

    module.write_artifact(output_path, payload)

    assert output_path.exists()
    assert output_path.read_text(encoding="utf-8").endswith("\n")
    assert json.loads(output_path.read_text(encoding="utf-8")) == payload


# REQ-VERIFY-025
def test_cli_help_entrypoint_exits_zero(monkeypatch: pytest.MonkeyPatch):
    module_path = (
        Path(__file__).resolve().parents[2] / "scripts" / "experiment_218_live_dual_model_suite.py"
    )
    monkeypatch.setattr("sys.argv", [str(module_path), "--help"])

    with pytest.raises(SystemExit) as excinfo:
        runpy.run_path(str(module_path), run_name="__main__")

    assert excinfo.value.code == 0


# REQ-VERIFY-026
def test_main_writes_payload_and_returns_zero(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    module = load_module()
    repo = make_repo(tmp_path)
    output_path = repo / "results" / "exp218.json"
    captured: dict[str, object] = {}

    monkeypatch.setenv("CARNOT_REPO_ROOT", str(repo))
    monkeypatch.setattr(module, "_run_live_benchmark", lambda args: {"experiment": 218})
    monkeypatch.setattr(
        module,
        "write_artifact",
        lambda path, payload: captured.update({"path": path, "payload": payload}),
    )

    result = module.main(["--benchmark", "gsm8k_semantic", "--output", str(output_path)])

    assert result == 0
    assert captured["path"] == output_path
    assert captured["payload"] == {"experiment": 218}
    assert f"Saved {output_path}" in capsys.readouterr().out
