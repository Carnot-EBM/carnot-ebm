"""Spec: REQ-VERIFY-013, REQ-VERIFY-014, SCENARIO-VERIFY-013, SCENARIO-VERIFY-014."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


def load_module():
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "scripts" / "experiment_213_monitorability_audit.py"
    spec = importlib.util.spec_from_file_location(
        "experiment_213_monitorability_audit", module_path
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


# REQ-VERIFY-013, SCENARIO-VERIFY-013
def test_build_representative_subset_covers_models_and_task_slices():
    module = load_module()
    subset = load_subset(module)

    assert [item["example_id"] for item in subset] == module.SUBSET_EXAMPLE_IDS
    assert len(subset) == 11
    assert {item["source_family"] for item in subset} == {
        "live_gsm8k_semantic_failure",
        "instruction_following",
        "code_typed_properties",
    }
    assert {item["task_slice"] for item in subset} == {
        "live_gsm8k_semantic_failure",
        "instruction_surface_only",
        "instruction_grounded",
        "code_typed_properties",
    }
    assert sum(1 for item in subset if item["free_form_reasoning_monitorable"]) == 2
    assert subset[0]["gold_answer"] == 2
    assert subset[-1]["gold_answer"] is None


# REQ-VERIFY-013, SCENARIO-VERIFY-013
def test_prompt_and_subset_helpers_cover_override_and_mode_branches(tmp_path: Path, monkeypatch):
    module = load_module()
    subset = load_subset(module)

    repo_override = tmp_path / "override"
    repo_override.mkdir()
    monkeypatch.setenv("CARNOT_REPO_ROOT", str(repo_override))
    assert module.get_repo_root() == repo_override.resolve()

    with pytest.raises(KeyError):
        module.build_representative_subset([])

    code_example = next(item for item in subset if item["task_slice"] == "code_typed_properties")
    live_example = next(
        item for item in subset if item["task_slice"] == "live_gsm8k_semantic_failure"
    )
    instruction_example = next(
        item for item in subset if item["task_slice"] == "instruction_surface_only"
    )

    free_form_prompt = module.build_mode_prompt(code_example, "free_form_reasoning")
    assert "```python" in free_form_prompt

    terse_prompt = module.build_mode_prompt(live_example, "answer_only_terse")
    assert "final numeric answer" in terse_prompt

    code_terse_prompt = module.build_mode_prompt(code_example, "answer_only_terse")
    assert "complete Python function definition" in code_terse_prompt

    instruction_terse_prompt = module.build_mode_prompt(instruction_example, "answer_only_terse")
    assert "native format" in instruction_terse_prompt

    non_code_free_form = module.build_mode_prompt(live_example, "free_form_reasoning")
    assert "<final answer only in the task's native format>" in non_code_free_form

    structured_prompt = module.build_mode_prompt(live_example, "structured_json")
    assert "strict JSON" in structured_prompt

    with pytest.raises(ValueError):
        module.build_mode_prompt(live_example, "unknown_mode")

    assert module.max_new_tokens_for(code_example, "answer_only_terse") == 220
    assert module.max_new_tokens_for(code_example, "free_form_reasoning") == 320
    assert module.max_new_tokens_for(code_example, "structured_json") == 360
    assert module.max_new_tokens_for(live_example, "answer_only_terse") == 96
    assert module.max_new_tokens_for(live_example, "free_form_reasoning") == 180
    assert module.max_new_tokens_for(live_example, "structured_json") == 220
    assert "T" in module.get_run_timestamp()


# REQ-VERIFY-013, SCENARIO-VERIFY-013
def test_parser_helpers_cover_fences_json_lists_numbers_and_code_edges():
    module = load_module()

    assert module.strip_fence('```json\n{"a": 1}\n```') == '{"a": 1}'
    assert module.parse_structured_payload('```json\n{"a": 1}\n```') == {"a": 1}
    assert module.parse_structured_payload('prefix {"a": 1} suffix') == {"a": 1}
    assert module.parse_structured_payload("not json at all") is None

    assert module.extract_final_section('{"final_answer": ["P3", "P1"]}', "structured_json") == (
        '["P3", "P1"]'
    )
    assert (
        module.extract_final_section("REASONING:\nwork\nFINAL:\n42", "free_form_reasoning") == "42"
    )
    assert module.extract_final_section("raw text", "answer_only_terse") == "raw text"

    assert module.parse_number_answer(True) is None
    assert module.parse_number_answer(7) == 7
    assert module.parse_number_answer(7.9) == 7
    assert module.parse_number_answer("no digits") is None

    assert module.parse_bullet_answer(["a", "b"]) == ["a", "b"]
    assert module.parse_bullet_answer("* one\n* two") == ["one", "two"]
    assert module.parse_json_object_answer({"x": 1}) == {"x": 1}
    assert module.parse_json_object_answer(5) is None
    assert module.parse_comma_list_answer(["A", "B"]) == ["A", "B"]
    assert module.parse_comma_list_answer('["A", "B"]') == ["A", "B"]
    assert module.parse_comma_list_answer("[not valid json") == ["[not valid json"]
    assert module.parse_comma_list_answer('["A", bad]') == ['["A"', "bad]"]

    assert module.extract_python_code(1) is None
    assert module.extract_python_code("```python\ndef f():\n    return 1\n```").startswith("def f")
    assert module.extract_python_code("```\ndef f():\n    return 1\n```").startswith("def f")
    assert module.extract_python_code("prefix def f():\n    return 1").startswith("def f")
    assert module.extract_python_code("plain text only") is None

    assert module.constraint_ratio(2, 0) == 0.0


# REQ-VERIFY-013, REQ-VERIFY-014, SCENARIO-VERIFY-014
def test_evaluate_response_scores_numeric_modes_by_quality_visibility_and_coverage():
    module = load_module()
    example = next(
        item for item in load_subset(module) if item["example_id"] == "exp211-live-gsm8k-923"
    )

    terse = module.evaluate_response(example, "answer_only_terse", "2")
    free_form = module.evaluate_response(
        example,
        "free_form_reasoning",
        "REASONING:\nRemaining cups become mint/chamomile rows before the final split.\n"
        "Each row gets cups before mint is halved.\nFINAL:\n4",
    )
    structured = module.evaluate_response(
        example,
        "structured_json",
        json.dumps(
            {
                "final_answer": 2,
                "checks": [
                    {"constraint": "remaining cups", "evidence": "subtract cinnamon first"},
                    {"constraint": "row split", "evidence": "divide by rows"},
                    {"constraint": "mint split", "evidence": "halve the row cups"},
                ],
                "confidence": "high",
            }
        ),
    )

    assert terse["parseable"] is True
    assert terse["answer_quality"] == 1.0
    assert terse["constraint_coverage"] < structured["constraint_coverage"]
    assert terse["semantic_visibility"] is None

    assert free_form["parseable"] is True
    assert free_form["answer_quality"] == 0.0
    assert free_form["constraint_coverage"] > terse["constraint_coverage"]
    assert free_form["semantic_visibility"] == 1.0

    assert structured["parseable"] is True
    assert structured["answer_quality"] == 1.0
    assert structured["constraint_coverage"] == 1.0
    assert structured["semantic_visibility"] is None

    unparsable = module.evaluate_response(example, "answer_only_terse", "not a number")
    assert unparsable["parseable"] is False
    assert unparsable["semantic_visibility"] == 0.0


# REQ-VERIFY-013, REQ-VERIFY-014, SCENARIO-VERIFY-014
def test_evaluate_response_handles_surface_and_grounded_instruction_modes():
    module = load_module()
    subset = load_subset(module)
    bullet_example = next(
        item for item in subset if item["example_id"] == "exp211-instruction-bullets-1"
    )
    grounded_example = next(
        item for item in subset if item["example_id"] == "exp211-instruction-grounded-1"
    )
    json_surface_example = next(
        item for item in subset if item["example_id"] == "exp211-instruction-json-1"
    )

    bullets = module.evaluate_response(
        bullet_example,
        "answer_only_terse",
        "- risk owner deadline now\n- owner tracks token risk\n- deadline closes token risk",
    )
    grounded_terse = module.evaluate_response(grounded_example, "answer_only_terse", "P3, P1")
    grounded_structured = module.evaluate_response(
        grounded_example,
        "structured_json",
        json.dumps(
            {
                "final_answer": ["P3", "P1"],
                "checks": [
                    {"constraint": "under $50k", "evidence": "P1 and P3 only"},
                    {"constraint": "before June", "evidence": "P1 May 20, P3 April 30"},
                ],
                "confidence": "high",
            }
        ),
    )
    invalid_structured = module.evaluate_response(
        grounded_example,
        "structured_json",
        '{"final_answer": ["P3", "P1"], "checks": [}',
    )

    assert bullets["parseable"] is True
    assert bullets["answer_quality"] == 1.0
    assert bullets["constraint_coverage"] == 1.0

    json_surface = module.evaluate_response(
        json_surface_example,
        "structured_json",
        json.dumps(
            {
                "final_answer": {
                    "action": "approve",
                    "reason": "clear fit",
                    "confidence": "medium",
                },
                "checks": [{"constraint": "exact keys", "evidence": "present"}],
                "confidence": "high",
            }
        ),
    )
    assert json_surface["parseable"] is True
    assert json_surface["answer_quality"] == 1.0

    assert grounded_terse["parseable"] is True
    assert grounded_terse["answer_quality"] == 1.0
    assert grounded_terse["constraint_coverage"] < grounded_structured["constraint_coverage"]

    assert grounded_structured["parseable"] is True
    assert grounded_structured["answer_quality"] == 1.0
    assert grounded_structured["constraint_coverage"] == 1.0

    assert invalid_structured["parseable"] is False
    assert invalid_structured["answer_quality"] == 0.0
    assert invalid_structured["constraint_coverage"] == 0.0

    decision_example = next(
        item for item in subset if item["example_id"] == "exp211-instruction-decision-1"
    )
    decision = module.evaluate_response(
        decision_example,
        "structured_json",
        json.dumps(
            {
                "final_answer": {"choice": "O3", "evidence": ["O3", "risk low"]},
                "checks": [{"constraint": "lower risk", "evidence": "O3 is low risk"}],
                "confidence": "high",
            }
        ),
    )
    assert decision["parseable"] is True
    assert decision["answer_quality"] == 1.0

    decision_string_evidence = module.evaluate_response(
        decision_example,
        "structured_json",
        json.dumps(
            {
                "final_answer": {"choice": "O3", "evidence": "O3 with risk low"},
                "checks": [{"constraint": "lower risk", "evidence": "O3 is low risk"}],
                "confidence": "high",
            }
        ),
    )
    assert decision_string_evidence["parseable"] is True
    assert decision_string_evidence["answer_quality"] == 1.0

    with pytest.raises(ValueError):
        unsupported_surface = dict(bullet_example)
        unsupported_surface["expected_answer_schema"] = {"type": "yaml_object"}
        module.evaluate_instruction_surface_only(
            unsupported_surface,
            "answer_only_terse",
            "owner: a",
        )

    with pytest.raises(ValueError):
        unsupported_grounded = dict(grounded_example)
        unsupported_grounded["expected_answer_schema"] = {"type": "identifier"}
        module.evaluate_instruction_grounded(
            unsupported_grounded,
            "answer_only_terse",
            "P3",
            None,
        )

    assert module.safe_mean([]) == 0.0


# REQ-VERIFY-013, REQ-VERIFY-014, SCENARIO-VERIFY-014
def test_evaluate_response_executes_code_probes_and_surfaces_failures():
    module = load_module()
    example = next(
        item for item in load_subset(module) if item["example_id"] == "exp211-code-score-1"
    )

    correct = module.evaluate_response(
        example,
        "answer_only_terse",
        "```python\n"
        "def score_keywords(text: str, weights: dict[str, int]) -> int:\n"
        "    words = text.split()\n"
        "    total = 0\n"
        "    for key, value in weights.items():\n"
        "        if key in words:\n"
        "            total += value\n"
        "    return total\n"
        "```",
    )
    broken = module.evaluate_response(
        example,
        "free_form_reasoning",
        "REASONING:\nUse every matching token, even repeated ones.\nFINAL:\n"
        "```python\n"
        "def score_keywords(text: str, weights: dict[str, int]) -> int:\n"
        "    total = 0\n"
        "    for word in text.split():\n"
        "        total += weights.get(word, 0)\n"
        "    return total\n"
        "```",
    )
    parse_failure = module.evaluate_response(example, "answer_only_terse", "return 0")

    assert correct["parseable"] is True
    assert correct["answer_quality"] == 1.0
    assert correct["constraint_coverage"] == 1.0
    assert correct["semantic_visibility"] is None

    assert broken["parseable"] is True
    assert broken["answer_quality"] == 0.5
    assert broken["semantic_visibility"] == 1.0

    assert parse_failure["parseable"] is False
    assert parse_failure["answer_quality"] == 0.0
    assert parse_failure["constraint_coverage"] == 0.0


# REQ-VERIFY-013, SCENARIO-VERIFY-014
def test_code_edge_cases_and_dispatch_error_paths_are_exercised():
    module = load_module()
    subset = load_subset(module)

    dedupe_example = next(item for item in subset if item["example_id"] == "exp211-code-dedupe-1")

    mutating = module.evaluate_response(
        dedupe_example,
        "answer_only_terse",
        "```python\n"
        "def dedupe_preserve_order(items: list[str]) -> list[str]:\n"
        "    items.pop()\n"
        "    return items\n"
        "```",
    )
    assert mutating["parseable"] is True
    assert mutating["answer_quality"] == 0.0
    assert mutating["semantic_visibility"] == 1.0

    wrong_name = module.evaluate_response(
        dedupe_example,
        "answer_only_terse",
        "```python\ndef other_name(items: list[str]) -> list[str]:\n    return items\n```",
    )
    assert wrong_name["parseable"] is False

    deleted_function = module.evaluate_response(
        dedupe_example,
        "answer_only_terse",
        "```python\n"
        "def dedupe_preserve_order(items: list[str]) -> list[str]:\n"
        "    return items\n"
        "del dedupe_preserve_order\n"
        "```",
    )
    assert deleted_function["parseable"] is False

    syntax_error = module.evaluate_response(
        dedupe_example,
        "answer_only_terse",
        "```python\ndef dedupe_preserve_order(items: list[str]) -> list[str]:\n    return [\n```",
    )
    assert syntax_error["parseable"] is False

    signature = module.function_signature_from_ast(
        __import__("ast").parse("def f(a):\n    return a\n").body[0]
    )
    assert signature == "f(a)"

    with pytest.raises(ValueError):
        bad_example = dict(dedupe_example)
        bad_example["task_slice"] = "unsupported"
        module.evaluate_response(bad_example, "answer_only_terse", "x")


# REQ-VERIFY-014, SCENARIO-VERIFY-014
def test_derive_policy_prefers_structured_or_terse_from_summary_metrics():
    module = load_module()

    summary = {
        "by_task_slice": {
            "live_gsm8k_semantic_failure": {
                "free_form_reasoning": {
                    "answer_quality_mean": 0.25,
                    "constraint_coverage_mean": 0.58,
                    "semantic_visibility_mean": 0.4,
                    "parseability_rate": 1.0,
                    "mean_completion_tokens": 120.0,
                },
                "answer_only_terse": {
                    "answer_quality_mean": 0.5,
                    "constraint_coverage_mean": 0.25,
                    "semantic_visibility_mean": 0.0,
                    "parseability_rate": 1.0,
                    "mean_completion_tokens": 12.0,
                },
                "structured_json": {
                    "answer_quality_mean": 0.5,
                    "constraint_coverage_mean": 0.92,
                    "semantic_visibility_mean": 0.85,
                    "parseability_rate": 0.9,
                    "mean_completion_tokens": 85.0,
                },
            },
            "instruction_surface_only": {
                "free_form_reasoning": {
                    "answer_quality_mean": 0.5,
                    "constraint_coverage_mean": 0.5,
                    "semantic_visibility_mean": 0.5,
                    "parseability_rate": 0.5,
                    "mean_completion_tokens": 80.0,
                },
                "answer_only_terse": {
                    "answer_quality_mean": 1.0,
                    "constraint_coverage_mean": 1.0,
                    "semantic_visibility_mean": 1.0,
                    "parseability_rate": 1.0,
                    "mean_completion_tokens": 18.0,
                },
                "structured_json": {
                    "answer_quality_mean": 1.0,
                    "constraint_coverage_mean": 1.0,
                    "semantic_visibility_mean": 1.0,
                    "parseability_rate": 1.0,
                    "mean_completion_tokens": 45.0,
                },
            },
            "instruction_grounded": {
                "free_form_reasoning": {
                    "answer_quality_mean": 0.6,
                    "constraint_coverage_mean": 0.8,
                    "semantic_visibility_mean": 0.7,
                    "parseability_rate": 1.0,
                    "mean_completion_tokens": 90.0,
                },
                "answer_only_terse": {
                    "answer_quality_mean": 0.5,
                    "constraint_coverage_mean": 0.1,
                    "semantic_visibility_mean": 0.2,
                    "parseability_rate": 1.0,
                    "mean_completion_tokens": 18.0,
                },
                "structured_json": {
                    "answer_quality_mean": 0.45,
                    "constraint_coverage_mean": 0.75,
                    "semantic_visibility_mean": 0.75,
                    "parseability_rate": 0.8,
                    "mean_completion_tokens": 70.0,
                },
            },
            "code_typed_properties": {
                "free_form_reasoning": {
                    "answer_quality_mean": 0.5,
                    "constraint_coverage_mean": 0.66,
                    "semantic_visibility_mean": 0.66,
                    "parseability_rate": 0.66,
                    "mean_completion_tokens": 140.0,
                },
                "answer_only_terse": {
                    "answer_quality_mean": 1.0,
                    "constraint_coverage_mean": 1.0,
                    "semantic_visibility_mean": 1.0,
                    "parseability_rate": 1.0,
                    "mean_completion_tokens": 70.0,
                },
                "structured_json": {
                    "answer_quality_mean": 0.33,
                    "constraint_coverage_mean": 0.33,
                    "semantic_visibility_mean": 0.33,
                    "parseability_rate": 0.33,
                    "mean_completion_tokens": 110.0,
                },
            },
        },
        "by_model": {
            "Model A": {
                "free_form_reasoning": {"semantic_visibility_mean": 0.2},
                "structured_json": {"constraint_coverage_mean": 0.8},
            },
            "Model B": {
                "free_form_reasoning": {"semantic_visibility_mean": 0.7},
                "structured_json": {"constraint_coverage_mean": 0.2},
            },
        },
    }

    policy = module.derive_policy(summary)

    assert (
        policy["per_task_slice"]["live_gsm8k_semantic_failure"]["recommended_mode"]
        == "structured_json"
    )
    assert (
        policy["per_task_slice"]["instruction_surface_only"]["recommended_mode"]
        == "answer_only_terse"
    )
    assert policy["per_task_slice"]["instruction_grounded"]["recommended_mode"] == (
        "structured_json"
    )
    assert (
        policy["per_task_slice"]["code_typed_properties"]["recommended_mode"] == "answer_only_terse"
    )
    assert "free_form_reasoning" in policy["global_policy"]["distrust_free_form_traces_when"]
    assert "Prefer structured_json" in policy["model_guidance"]["Model A"]
    assert "Use terse or structured outputs" in policy["model_guidance"]["Model B"]

    terse_fallback = module.derive_policy(
        {
            "by_task_slice": {
                "instruction_grounded": {
                    "free_form_reasoning": {
                        "answer_quality_mean": 0.6,
                        "constraint_coverage_mean": 0.4,
                        "semantic_visibility_mean": 0.6,
                        "parseability_rate": 1.0,
                    },
                    "answer_only_terse": {
                        "answer_quality_mean": 0.55,
                        "constraint_coverage_mean": 0.3,
                        "semantic_visibility_mean": 0.2,
                        "parseability_rate": 1.0,
                    },
                    "structured_json": {
                        "answer_quality_mean": 0.2,
                        "constraint_coverage_mean": 0.35,
                        "semantic_visibility_mean": 0.35,
                        "parseability_rate": 0.7,
                    },
                }
            },
            "by_model": {},
        }
    )
    assert terse_fallback["per_task_slice"]["instruction_grounded"]["recommended_mode"] == (
        "answer_only_terse"
    )


# REQ-VERIFY-013, REQ-VERIFY-014, SCENARIO-VERIFY-013
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
    monkeypatch.setattr(module, "RESULTS_PATH", repo / "results" / "experiment_213_results.json")
    monkeypatch.setattr(
        module,
        "POLICY_PATH",
        repo / "results" / "monitorability_policy_213.json",
    )
    monkeypatch.setattr(module, "get_run_timestamp", lambda: "2026-04-12T09:00:00Z")

    fake_responses = [
        {
            "model_name": "Qwen3.5-0.8B",
            "hf_id": "Qwen/Qwen3.5-0.8B",
            "mode": "answer_only_terse",
            "example_id": "exp211-live-gsm8k-923",
            "task_slice": "live_gsm8k_semantic_failure",
            "source_family": "live_gsm8k_semantic_failure",
            "parseable": True,
            "answer_quality": 1.0,
            "constraint_coverage": 0.25,
            "semantic_visibility": None,
            "prompt_tokens": 20,
            "completion_tokens": 3,
            "latency_seconds": 0.1,
            "raw_response": "2",
        },
        {
            "model_name": "Qwen3.5-0.8B",
            "hf_id": "Qwen/Qwen3.5-0.8B",
            "mode": "structured_json",
            "example_id": "exp211-live-gsm8k-923",
            "task_slice": "live_gsm8k_semantic_failure",
            "source_family": "live_gsm8k_semantic_failure",
            "parseable": True,
            "answer_quality": 1.0,
            "constraint_coverage": 1.0,
            "semantic_visibility": None,
            "prompt_tokens": 35,
            "completion_tokens": 30,
            "latency_seconds": 0.2,
            "raw_response": '{"final_answer": 2, "checks": []}',
        },
    ]
    monkeypatch.setattr(module, "run_live_audit", lambda subset: fake_responses)

    assert module.main() == 0
    assert module.main() == 0

    results = json.loads((repo / "results" / "experiment_213_results.json").read_text())
    policy = json.loads((repo / "results" / "monitorability_policy_213.json").read_text())

    assert results["experiment"] == "Exp 213"
    assert results["run_date"] == "20260412"
    assert results["metadata"]["timestamp"] == "2026-04-12T09:00:00Z"
    assert len(results["responses"]) == 2
    assert policy["experiment"] == "Exp 213"
    assert policy["run_date"] == "20260412"
