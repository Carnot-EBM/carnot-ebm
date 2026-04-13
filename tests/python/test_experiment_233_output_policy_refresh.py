"""Spec: REQ-VERIFY-044, REQ-VERIFY-045, SCENARIO-VERIFY-045, SCENARIO-VERIFY-046."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


def load_module():
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "scripts" / "experiment_233_output_policy_refresh.py"
    spec = importlib.util.spec_from_file_location(
        "experiment_233_output_policy_refresh", module_path
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


def write_exp211_fixture(repo: Path) -> None:
    source = (
        Path(__file__).resolve().parents[2]
        / "data"
        / "research"
        / "constraint_ir_benchmark_211.jsonl"
    )
    target = repo / "data" / "research" / "constraint_ir_benchmark_211.jsonl"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")


def load_subset(module):
    return module.build_representative_subset(module.load_benchmark_records(module.BENCHMARK_PATH))


# REQ-VERIFY-044, SCENARIO-VERIFY-046
def test_build_representative_subset_includes_benchmark_and_repo_spec_slices():
    module = load_module()
    subset = load_subset(module)

    assert [item["example_id"] for item in subset] == module.SUBSET_EXAMPLE_IDS
    assert len(subset) >= 13
    assert {item["source_family"] for item in subset} == {
        "live_gsm8k_semantic_failure",
        "instruction_following",
        "code_typed_properties",
        "repo_spec_grounding",
    }
    assert {item["task_slice"] for item in subset} == {
        "live_gsm8k_semantic_failure",
        "instruction_surface_only",
        "instruction_grounded",
        "code_typed_properties",
        "repo_spec_grounding",
    }
    spec_examples = [item for item in subset if item["source_family"] == "repo_spec_grounding"]
    assert all(item["prompt_source_path"] for item in spec_examples)
    assert all(item["gold_answer"] for item in spec_examples)


# REQ-VERIFY-044, SCENARIO-VERIFY-046
def test_prompt_and_token_helpers_cover_new_mode_branches():
    module = load_module()
    subset = load_subset(module)

    live_example = next(
        item for item in subset if item["task_slice"] == "live_gsm8k_semantic_failure"
    )
    code_example = next(item for item in subset if item["task_slice"] == "code_typed_properties")
    spec_example = next(item for item in subset if item["task_slice"] == "repo_spec_grounding")

    free_form_prompt = module.build_mode_prompt(live_example, "free_form_reasoning")
    code_free_form_prompt = module.build_mode_prompt(code_example, "free_form_reasoning")
    terse_prompt = module.build_mode_prompt(live_example, "answer_only_terse")
    code_terse_prompt = module.build_mode_prompt(code_example, "answer_only_terse")
    spec_terse_prompt = module.build_mode_prompt(spec_example, "answer_only_terse")
    minimal_prompt = module.build_mode_prompt(spec_example, "minimal_json")
    grammar_prompt = module.build_mode_prompt(code_example, "grammar_gated_json")

    assert "REASONING:" in free_form_prompt
    assert "```python" in code_free_form_prompt
    assert "No explanation" in terse_prompt
    assert "Python function definition" in code_terse_prompt
    assert "native format" in spec_terse_prompt
    assert '"final_answer"' in minimal_prompt
    assert '"claims"' in minimal_prompt
    assert "single-line JSON object" in grammar_prompt
    assert "strict key order" in grammar_prompt

    assert module.max_new_tokens_for(code_example, "answer_only_terse") < module.max_new_tokens_for(
        code_example, "minimal_json"
    )
    assert module.max_new_tokens_for(
        live_example, "grammar_gated_json"
    ) > module.max_new_tokens_for(live_example, "minimal_json")
    assert module.max_new_tokens_for(spec_example, "free_form_reasoning") > 0


# REQ-VERIFY-044
def test_summarize_records_aggregates_retry_claim_and_repair_metrics():
    module = load_module()
    records = [
        {
            "model_name": "Qwen3.5-0.8B",
            "mode": "minimal_json",
            "task_slice": "live_gsm8k_semantic_failure",
            "parse_success": True,
            "answer_quality": 1.0,
            "exact_satisfaction": 1.0,
            "partial_satisfaction": 1.0,
            "claim_coverage": 1.0,
            "repair_useful": 0.0,
            "retries_used": 0,
            "retry_budget": 1,
            "prompt_tokens": 20,
            "completion_tokens": 12,
            "total_tokens": 32,
            "latency_seconds": 0.4,
        },
        {
            "model_name": "Qwen3.5-0.8B",
            "mode": "minimal_json",
            "task_slice": "live_gsm8k_semantic_failure",
            "parse_success": False,
            "answer_quality": 0.5,
            "exact_satisfaction": 0.0,
            "partial_satisfaction": 0.5,
            "claim_coverage": 0.5,
            "repair_useful": 1.0,
            "retries_used": 1,
            "retry_budget": 1,
            "prompt_tokens": 22,
            "completion_tokens": 16,
            "total_tokens": 38,
            "latency_seconds": 0.6,
        },
        {
            "model_name": "Gemma4-E4B-it",
            "mode": "answer_only_terse",
            "task_slice": "code_typed_properties",
            "parse_success": True,
            "answer_quality": 1.0,
            "exact_satisfaction": 1.0,
            "partial_satisfaction": 1.0,
            "claim_coverage": 0.0,
            "repair_useful": 0.0,
            "retries_used": 0,
            "retry_budget": 0,
            "prompt_tokens": 10,
            "completion_tokens": 30,
            "total_tokens": 40,
            "latency_seconds": 0.3,
        },
    ]

    summary = module.summarize_records(records)

    live_metrics = summary["by_task_slice"]["live_gsm8k_semantic_failure"]["minimal_json"]
    assert live_metrics["n_records"] == 2
    assert live_metrics["parse_success_rate"] == 0.5
    assert live_metrics["retry_rate"] == 0.5
    assert live_metrics["mean_retries_used"] == 0.5
    assert live_metrics["repair_usefulness_rate"] == 0.5
    assert live_metrics["claim_coverage_mean"] == 0.75
    assert live_metrics["mean_total_tokens"] == 35.0
    assert summary["by_model"]["Qwen3.5-0.8B"]["minimal_json"]["retry_rate"] == 0.5
    assert summary["n_responses"] == 3


# REQ-VERIFY-045, SCENARIO-VERIFY-045
def test_derive_policy_prefers_terse_minimal_and_grammar_modes_by_slice():
    module = load_module()
    summary = {
        "by_task_slice": {
            "live_gsm8k_semantic_failure": {
                "free_form_reasoning": {
                    "parse_success_rate": 1.0,
                    "answer_quality_mean": 0.2,
                    "exact_satisfaction_rate": 0.2,
                    "partial_satisfaction_mean": 0.55,
                    "claim_coverage_mean": 0.45,
                    "retry_rate": 0.0,
                    "repair_usefulness_rate": 0.0,
                    "mean_total_tokens": 120.0,
                },
                "answer_only_terse": {
                    "parse_success_rate": 1.0,
                    "answer_quality_mean": 0.3,
                    "exact_satisfaction_rate": 0.3,
                    "partial_satisfaction_mean": 0.3,
                    "claim_coverage_mean": 0.0,
                    "retry_rate": 0.0,
                    "repair_usefulness_rate": 0.0,
                    "mean_total_tokens": 15.0,
                },
                "minimal_json": {
                    "parse_success_rate": 0.85,
                    "answer_quality_mean": 0.45,
                    "exact_satisfaction_rate": 0.45,
                    "partial_satisfaction_mean": 0.72,
                    "claim_coverage_mean": 0.8,
                    "retry_rate": 0.3,
                    "repair_usefulness_rate": 0.25,
                    "mean_total_tokens": 60.0,
                },
                "grammar_gated_json": {
                    "parse_success_rate": 0.95,
                    "answer_quality_mean": 0.45,
                    "exact_satisfaction_rate": 0.45,
                    "partial_satisfaction_mean": 0.74,
                    "claim_coverage_mean": 0.82,
                    "retry_rate": 0.5,
                    "repair_usefulness_rate": 0.2,
                    "mean_total_tokens": 78.0,
                },
            },
            "instruction_surface_only": {
                "free_form_reasoning": {
                    "parse_success_rate": 0.7,
                    "answer_quality_mean": 0.7,
                    "exact_satisfaction_rate": 0.3,
                    "partial_satisfaction_mean": 0.7,
                    "claim_coverage_mean": 0.4,
                    "retry_rate": 0.0,
                    "repair_usefulness_rate": 0.0,
                    "mean_total_tokens": 80.0,
                },
                "answer_only_terse": {
                    "parse_success_rate": 1.0,
                    "answer_quality_mean": 1.0,
                    "exact_satisfaction_rate": 1.0,
                    "partial_satisfaction_mean": 1.0,
                    "claim_coverage_mean": 0.75,
                    "retry_rate": 0.0,
                    "repair_usefulness_rate": 0.0,
                    "mean_total_tokens": 20.0,
                },
                "minimal_json": {
                    "parse_success_rate": 1.0,
                    "answer_quality_mean": 1.0,
                    "exact_satisfaction_rate": 1.0,
                    "partial_satisfaction_mean": 1.0,
                    "claim_coverage_mean": 0.75,
                    "retry_rate": 0.1,
                    "repair_usefulness_rate": 0.0,
                    "mean_total_tokens": 42.0,
                },
                "grammar_gated_json": {
                    "parse_success_rate": 1.0,
                    "answer_quality_mean": 1.0,
                    "exact_satisfaction_rate": 1.0,
                    "partial_satisfaction_mean": 1.0,
                    "claim_coverage_mean": 0.75,
                    "retry_rate": 0.2,
                    "repair_usefulness_rate": 0.0,
                    "mean_total_tokens": 60.0,
                },
            },
            "repo_spec_grounding": {
                "free_form_reasoning": {
                    "parse_success_rate": 0.9,
                    "answer_quality_mean": 0.6,
                    "exact_satisfaction_rate": 0.4,
                    "partial_satisfaction_mean": 0.7,
                    "claim_coverage_mean": 0.5,
                    "retry_rate": 0.0,
                    "repair_usefulness_rate": 0.0,
                    "mean_total_tokens": 90.0,
                },
                "answer_only_terse": {
                    "parse_success_rate": 0.95,
                    "answer_quality_mean": 0.7,
                    "exact_satisfaction_rate": 0.5,
                    "partial_satisfaction_mean": 0.75,
                    "claim_coverage_mean": 0.1,
                    "retry_rate": 0.0,
                    "repair_usefulness_rate": 0.0,
                    "mean_total_tokens": 24.0,
                },
                "minimal_json": {
                    "parse_success_rate": 0.98,
                    "answer_quality_mean": 0.9,
                    "exact_satisfaction_rate": 0.8,
                    "partial_satisfaction_mean": 0.92,
                    "claim_coverage_mean": 0.85,
                    "retry_rate": 0.1,
                    "repair_usefulness_rate": 0.1,
                    "mean_total_tokens": 48.0,
                },
                "grammar_gated_json": {
                    "parse_success_rate": 0.99,
                    "answer_quality_mean": 0.9,
                    "exact_satisfaction_rate": 0.8,
                    "partial_satisfaction_mean": 0.92,
                    "claim_coverage_mean": 0.86,
                    "retry_rate": 0.4,
                    "repair_usefulness_rate": 0.1,
                    "mean_total_tokens": 68.0,
                },
            },
        },
        "by_model": {
            "Qwen3.5-0.8B": {
                "minimal_json": {"exact_satisfaction_rate": 0.6},
                "grammar_gated_json": {"exact_satisfaction_rate": 0.58},
            },
            "Gemma4-E4B-it": {
                "minimal_json": {"exact_satisfaction_rate": 0.75},
                "grammar_gated_json": {"exact_satisfaction_rate": 0.78},
            },
        },
    }

    policy = module.derive_policy(summary)

    assert policy["per_task_slice"]["live_gsm8k_semantic_failure"]["recommended_mode"] == (
        "minimal_json"
    )
    assert policy["per_task_slice"]["live_gsm8k_semantic_failure"]["retry_budget"] == 1
    assert policy["per_task_slice"]["instruction_surface_only"]["recommended_mode"] == (
        "answer_only_terse"
    )
    assert policy["per_task_slice"]["repo_spec_grounding"]["recommended_mode"] == "minimal_json"
    assert policy["mode_defaults"]["fallback_mode"] == "answer_only_terse"
    assert "minimal_json" in policy["global_policy"]["request_json_when"][0]
    assert "grammar_gated_json" in policy["mode_guidance"]["Gemma4-E4B-it"]


# REQ-VERIFY-044, REQ-VERIFY-045, SCENARIO-VERIFY-046
def test_main_writes_results_and_policy_idempotently(tmp_path: Path, monkeypatch):
    module = load_module()
    repo = make_repo(tmp_path)
    write_exp211_fixture(repo)

    monkeypatch.setattr(module, "REPO_ROOT", repo)
    monkeypatch.setattr(
        module,
        "BENCHMARK_PATH",
        repo / "data" / "research" / "constraint_ir_benchmark_211.jsonl",
    )
    monkeypatch.setattr(module, "RESULTS_PATH", repo / "results" / "experiment_233_results.json")
    monkeypatch.setattr(module, "POLICY_PATH", repo / "results" / "output_policy_233.json")
    monkeypatch.setattr(module, "get_run_timestamp", lambda: "2026-04-13T09:00:00Z")

    fake_responses = [
        {
            "model_name": "Qwen3.5-0.8B",
            "hf_id": "Qwen/Qwen3.5-0.8B",
            "example_id": "exp211-live-gsm8k-923",
            "task_slice": "live_gsm8k_semantic_failure",
            "source_family": "live_gsm8k_semantic_failure",
            "mode": "minimal_json",
            "parse_success": True,
            "answer_quality": 1.0,
            "exact_satisfaction": 1.0,
            "partial_satisfaction": 1.0,
            "claim_coverage": 1.0,
            "repair_useful": 0.0,
            "retries_used": 0,
            "retry_budget": 1,
            "prompt_tokens": 30,
            "completion_tokens": 18,
            "total_tokens": 48,
            "latency_seconds": 0.2,
            "raw_response": (
                '{"final_answer": 2, "claims": ["subtract 15", "divide by 3", "halve"]}'
            ),
        },
        {
            "model_name": "Gemma4-E4B-it",
            "hf_id": "google/gemma-4-E4B-it",
            "example_id": "exp233-spec-req-ids",
            "task_slice": "repo_spec_grounding",
            "source_family": "repo_spec_grounding",
            "mode": "answer_only_terse",
            "parse_success": True,
            "answer_quality": 1.0,
            "exact_satisfaction": 1.0,
            "partial_satisfaction": 1.0,
            "claim_coverage": 0.0,
            "repair_useful": 0.0,
            "retries_used": 0,
            "retry_budget": 0,
            "prompt_tokens": 28,
            "completion_tokens": 6,
            "total_tokens": 34,
            "latency_seconds": 0.1,
            "raw_response": "REQ-VERIFY-022, REQ-VERIFY-023, REQ-VERIFY-024",
        },
    ]
    monkeypatch.setattr(module, "run_live_audit", lambda subset: fake_responses)

    assert module.main() == 0
    assert module.main() == 0

    results = json.loads((repo / "results" / "experiment_233_results.json").read_text())
    policy = json.loads((repo / "results" / "output_policy_233.json").read_text())

    assert results["experiment"] == "Exp 233"
    assert results["run_date"] == "20260413"
    assert results["metadata"]["timestamp"] == "2026-04-13T09:00:00Z"
    assert results["metadata"]["inference_mode"] == "live_gpu"
    assert results["metadata"]["modes"] == module.MODE_ORDER
    assert len(results["responses"]) == 2
    assert policy["experiment"] == "Exp 233"
    assert policy["run_date"] == "20260413"
    assert policy["derived_from"] == "results/experiment_233_results.json"


# REQ-VERIFY-044
def test_helper_branches_cover_repo_root_extractors_and_mode_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    module = load_module()
    repo = tmp_path / "override"
    repo.mkdir()
    monkeypatch.setenv("CARNOT_REPO_ROOT", str(repo))

    assert module.get_repo_root() == repo.resolve()
    assert "T" in module.get_run_timestamp()

    with pytest.raises(KeyError):
        module.build_representative_subset([])

    assert module._json_answer_hint({"expected_answer_schema": {"type": "json_object"}}) == (
        "final_answer must be a JSON object matching the task schema."
    )
    assert module._json_answer_hint(
        {"expected_answer_schema": {"type": "comma_separated_list"}}
    ) == ("final_answer must be a JSON array of strings in order.")
    assert module._json_answer_hint({"expected_answer_schema": {"type": "bullet_list"}}) == (
        "final_answer must be a JSON array of bullet strings."
    )
    assert module._json_answer_hint({"expected_answer_schema": {"type": "number"}}).startswith(
        "final_answer must contain"
    )

    with pytest.raises(ValueError):
        module.build_mode_prompt(
            {
                "example_id": "x",
                "task_slice": "repo_spec_grounding",
                "prompt": "",
                "expected_answer_schema": {"type": "number"},
            },
            "unknown_mode",
        )

    assert module.parse_structured_payload('{"final_answer": 2, "claims": []}') == {
        "final_answer": 2,
        "claims": [],
    }
    assert module.extract_final_section("not json", "minimal_json") == ""
    assert module.extract_final_section('{"final_answer": ["A", "B"]}', "minimal_json") == (
        '["A", "B"]'
    )
    assert module.extract_final_section('{"final_answer": 2}', "minimal_json") == "2"
    assert module.constraint_ratio(1, 2) == 0.5


# REQ-VERIFY-044
def test_claim_spec_and_code_helpers_cover_remaining_branches(monkeypatch: pytest.MonkeyPatch):
    module = load_module()

    marker_source = module._claim_marker_source("raw", {"claims": ["a"], "checks": ["b"]})
    assert "claims" not in marker_source
    assert '["a"]' in marker_source
    assert '["b"]' in marker_source
    assert module._claim_coverage({"claim_markers": []}, "raw", None) == 0.0

    spec_example = {
        "gold_answer": ["REQ-VERIFY-022", "REQ-VERIFY-023", "REQ-VERIFY-024"],
        "claim_markers": ["req verify 022"],
    }
    exact_spec = module._evaluate_repo_spec_grounding(
        spec_example,
        "answer_only_terse",
        "REQ-VERIFY-022, REQ-VERIFY-023, REQ-VERIFY-024",
    )
    partial_spec = module._evaluate_repo_spec_grounding(
        spec_example,
        "answer_only_terse",
        "REQ-VERIFY-022, REQ-VERIFY-999, REQ-VERIFY-024",
    )
    assert exact_spec["exact_satisfaction"] == 1.0
    assert partial_spec["partial_satisfaction"] == 0.6667

    signature = module._function_signature_from_ast(
        __import__("ast").parse("def f(a: int) -> int:\n    return a\n").body[0]
    )
    assert signature == "f(a: int) -> int"

    assert module._topological_order_valid(["a", "b", "a"], ["a", "b"], [("a", "b")]) is False
    assert module._topological_order_valid(["a"], ["a", "b"], [("a", "b")]) is False
    assert module._topological_order_valid(["a", "b"], ["a", "b"], [("a", "b")]) is True
    assert module._topological_order_valid(None, ["a", "b"], [("a", "b")]) is False

    def boom(_edges):
        raise ValueError("cycle")

    def mutating(items):
        items.pop()
        return items

    def topo(_edges):
        return ["a", "b", "c"]

    assert (
        module._run_code_probe(boom, {"args": [[("a", "b")]], "expect_exception": "ValueError"})
        is True
    )
    assert (
        module._run_code_probe(
            mutating, {"args": [["a", "b"]], "expected": ["a"], "immutable_arg_index": 0}
        )
        is False
    )
    assert (
        module._run_code_probe(
            topo,
            {
                "args": [[("a", "b"), ("b", "c")]],
                "validator": "topological_order",
                "nodes": ["a", "b", "c"],
                "edges": [("a", "b"), ("b", "c")],
            },
        )
        is True
    )
    assert module._run_code_probe(lambda x: x, {"args": [5], "expected": 5}) is True
    assert (
        module._run_code_probe(lambda x: x, {"args": [5], "expect_exception": "ValueError"})
        is False
    )

    code_example = {
        "expected_answer_schema": {
            "name": "dedupe_preserve_order",
            "signature": "dedupe_preserve_order(items: list[str]) -> list[str]",
        },
        "probe_cases": [
            {
                "args": [["a", "b", "a"]],
                "expected": ["a", "b"],
                "immutable_arg_index": 0,
            }
        ],
    }
    assert (
        module._evaluate_code_typed_properties(code_example, "answer_only_terse", "not code")[
            "parse_success"
        ]
        is False
    )
    assert (
        module._evaluate_code_typed_properties(
            code_example,
            "answer_only_terse",
            "```python\ndef dedupe_preserve_order(\n```",
        )["parse_success"]
        is False
    )
    assert (
        module._evaluate_code_typed_properties(
            code_example,
            "answer_only_terse",
            "```python\ndef other(items):\n    return items\n```",
        )["parse_success"]
        is False
    )
    assert (
        module._evaluate_code_typed_properties(
            code_example,
            "answer_only_terse",
            "```python\n"
            "def dedupe_preserve_order(items: list[str]) -> list[str]:\n"
            "    return items\n"
            "del dedupe_preserve_order\n```",
        )["parse_success"]
        is False
    )
    surface_json_example = {
        "expected_answer_schema": {"type": "json_object"},
        "gold_atomic_constraints": [1, 2, 3, 4],
        "claim_markers": [],
    }
    assert (
        module._evaluate_instruction_surface_only(
            surface_json_example,
            "minimal_json",
            (
                '{"final_answer": {"action": "approve", '
                '"reason": "clear fit", "confidence": "high"}, '
                '"claims": []}'
            ),
            {
                "final_answer": {
                    "action": "approve",
                    "reason": "clear fit",
                    "confidence": "high",
                },
                "claims": [],
            },
        )["exact_satisfaction"]
        == 1.0
    )
    with pytest.raises(ValueError):
        module._evaluate_instruction_surface_only(
            {"expected_answer_schema": {"type": "yaml_object"}, "gold_atomic_constraints": []},
            "answer_only_terse",
            "x",
            None,
        )
    grounded_json_example = {
        "expected_answer_schema": {"type": "json_object"},
        "gold_answer": {"choice": "O3", "evidence": ["O3", "risk low"]},
        "claim_markers": [],
    }
    assert (
        module._evaluate_instruction_grounded(
            grounded_json_example,
            "minimal_json",
            '{"final_answer": {"choice": "O3", "evidence": ["O3", "risk low"]}, "claims": []}',
            {"final_answer": {"choice": "O3", "evidence": ["O3", "risk low"]}, "claims": []},
        )["exact_satisfaction"]
        == 1.0
    )
    assert (
        module._evaluate_instruction_grounded(
            grounded_json_example,
            "minimal_json",
            '{"final_answer": {"choice": "O3", "evidence": "O3 with risk low"}, "claims": []}',
            {"final_answer": {"choice": "O3", "evidence": "O3 with risk low"}, "claims": []},
        )["exact_satisfaction"]
        == 1.0
    )
    with pytest.raises(ValueError):
        module._evaluate_instruction_grounded(
            {
                "expected_answer_schema": {"type": "identifier"},
                "gold_answer": "X",
                "claim_markers": [],
            },
            "answer_only_terse",
            "X",
            None,
        )

    monkeypatch.setattr(module, "count_tokens", lambda tokenizer, text: len(str(text).split()))
    attempt = module._record_attempt(
        tokenizer=object(),
        prompt="one two",
        raw_response="three four five",
        evaluation={
            "parse_success": True,
            "answer_quality": 1.0,
            "exact_satisfaction": 1.0,
            "partial_satisfaction": 1.0,
            "claim_coverage": 0.5,
        },
        latency_seconds=0.25,
    )
    assert attempt["total_tokens"] == 5


# REQ-VERIFY-044, REQ-VERIFY-045
def test_evaluate_response_retry_and_policy_helpers_cover_remaining_paths():
    module = load_module()
    subset = load_subset(module)

    surface_example = next(
        item for item in subset if item["task_slice"] == "instruction_surface_only"
    )
    grounded_example = next(item for item in subset if item["task_slice"] == "instruction_grounded")
    code_example = next(item for item in subset if item["task_slice"] == "code_typed_properties")
    spec_example = next(item for item in subset if item["task_slice"] == "repo_spec_grounding")
    live_example = next(
        item for item in subset if item["task_slice"] == "live_gsm8k_semantic_failure"
    )

    live = module.evaluate_response(
        live_example,
        "minimal_json",
        '{"final_answer": 2, "claims": ["remaining", "row", "mint"]}',
    )
    surface = module.evaluate_response(
        surface_example,
        "minimal_json",
        (
            '{"final_answer": ["risk owner deadline now", '
            '"owner tracks token risk", "deadline closes token risk"], '
            '"claims": ["risk", "owner", "deadline"]}'
        ),
    )
    grounded = module.evaluate_response(
        grounded_example,
        "minimal_json",
        '{"final_answer": ["P3", "P1"], "claims": ["under 50k", "before June"]}',
    )
    code = module.evaluate_response(
        code_example,
        "answer_only_terse",
        "```python\n"
        "def dedupe_preserve_order(items: list[str]) -> list[str]:\n"
        "    seen = set()\n"
        "    out = []\n"
        "    for item in items:\n"
        "        if item not in seen:\n"
        "            seen.add(item)\n"
        "            out.append(item)\n"
        "    return out\n```",
    )
    spec = module.evaluate_response(
        spec_example,
        "minimal_json",
        (
            '{"final_answer": ["REQ-VERIFY-022", '
            '"REQ-VERIFY-023", "REQ-VERIFY-024"], '
            '"claims": ["structured reasoning"]}'
        ),
    )

    assert live["exact_satisfaction"] == 1.0
    assert surface["parse_success"] is True
    assert grounded["exact_satisfaction"] == 1.0
    assert code["parse_success"] is True
    assert spec["exact_satisfaction"] == 1.0

    bad = dict(spec_example)
    bad["task_slice"] = "unsupported"
    with pytest.raises(ValueError):
        module.evaluate_response(bad, "answer_only_terse", "x")

    assert module._needs_retry(spec_example, "answer_only_terse", spec) is False
    assert (
        module._needs_retry(
            spec_example,
            "minimal_json",
            {
                "parse_success": False,
                "exact_satisfaction": 0.0,
                "partial_satisfaction": 0.0,
                "claim_coverage": 0.0,
            },
        )
        is True
    )
    assert (
        module._needs_retry(
            grounded_example,
            "minimal_json",
            {
                "parse_success": True,
                "exact_satisfaction": 0.0,
                "partial_satisfaction": 0.5,
                "claim_coverage": 0.2,
            },
        )
        is True
    )
    assert (
        module._needs_retry(
            live_example,
            "minimal_json",
            {
                "parse_success": True,
                "exact_satisfaction": 1.0,
                "partial_satisfaction": 1.0,
                "claim_coverage": 0.1,
            },
        )
        is True
    )
    assert (
        module._needs_retry(
            live_example,
            "minimal_json",
            {
                "parse_success": True,
                "exact_satisfaction": 1.0,
                "partial_satisfaction": 1.0,
                "claim_coverage": 1.0,
            },
        )
        is False
    )
    assert module._evaluation_rank(spec) == (1.0, 1.0, 1.0, spec["claim_coverage"])
    prompt = module._build_retry_prompt(
        spec_example,
        "minimal_json",
        {
            "parse_success": True,
            "exact_satisfaction": 1.0,
            "claim_coverage": 1.0,
        },
    )
    assert "response quality needs improvement" in prompt
    failing_prompt = module._build_retry_prompt(
        spec_example,
        "minimal_json",
        {
            "parse_success": False,
            "exact_satisfaction": 0.0,
            "claim_coverage": 0.0,
        },
    )
    assert "did not parse cleanly" in failing_prompt
    assert "incomplete or wrong" in failing_prompt
    assert "missed verifier-visible evidence" in failing_prompt
    assert module.safe_mean([]) == 0.0

    assert module._json_mode_choice({"minimal_json": {"exact_satisfaction_rate": 0.7}})[0] == (
        "minimal_json"
    )
    assert module._json_mode_choice({"grammar_gated_json": {"exact_satisfaction_rate": 0.7}})[
        0
    ] == ("grammar_gated_json")
    assert (
        module._json_mode_choice(
            {
                "minimal_json": {
                    "exact_satisfaction_rate": 0.6,
                    "parse_success_rate": 0.7,
                    "partial_satisfaction_mean": 0.7,
                },
                "grammar_gated_json": {
                    "exact_satisfaction_rate": 0.7,
                    "parse_success_rate": 0.9,
                    "partial_satisfaction_mean": 0.8,
                },
            }
        )[0]
        == "grammar_gated_json"
    )

    fallback_policy = module.derive_policy(
        {
            "by_task_slice": {
                "repo_spec_grounding": {
                    "free_form_reasoning": {
                        "parse_success_rate": 0.0,
                        "exact_satisfaction_rate": 0.0,
                    },
                }
            },
            "by_model": {},
        }
    )
    assert fallback_policy["per_task_slice"]["repo_spec_grounding"]["recommended_mode"] == (
        "answer_only_terse"
    )
