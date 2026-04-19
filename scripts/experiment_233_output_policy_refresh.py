#!/usr/bin/env python3
"""Experiment 233: refreshed output policy with minimal-schema JSON modes.

This workflow extends Exp 213's monitorability audit with:

- a larger mixed slice
- two lighter structured modes: ``minimal_json`` and ``grammar_gated_json``
- retry-budget and repair-usefulness metrics
- a refreshed machine-readable policy later experiments can consume directly

It writes:

- ``results/experiment_233_results.json``
- ``results/output_policy_233.json``

Spec: REQ-VERIFY-044, REQ-VERIFY-045, SCENARIO-VERIFY-045,
SCENARIO-VERIFY-046
"""

from __future__ import annotations

import ast
import copy
import gc
import importlib.util
import json
import os
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from carnot.inference.model_loader import generate, load_model

RUN_DATE = "20260413"
EXPERIMENT_LABEL = "Exp 233"

MODE_ORDER = [
    "free_form_reasoning",
    "answer_only_terse",
    "minimal_json",
    "grammar_gated_json",
]
JSON_MODES = {"minimal_json", "grammar_gated_json"}
MODE_RETRY_BUDGETS = {
    "free_form_reasoning": 0,
    "answer_only_terse": 0,
    "minimal_json": 1,
    "grammar_gated_json": 2,
}
MODEL_SPECS = [
    {"name": "Qwen3.5-0.8B", "hf_id": "Qwen/Qwen3.5-0.8B"},
    {"name": "Gemma4-E4B-it", "hf_id": "google/gemma-4-E4B-it"},
]

SUBSET_EXAMPLE_IDS = [
    "exp211-live-gsm8k-923",
    "exp211-live-gsm8k-506",
    "exp211-live-gsm8k-1019",
    "exp211-live-gsm8k-1077",
    "exp211-instruction-bullets-1",
    "exp211-instruction-json-1",
    "exp211-instruction-grounded-1",
    "exp211-instruction-decision-1",
    "exp211-instruction-filter-1",
    "exp211-code-dedupe-1",
    "exp211-code-chunks-1",
    "exp211-code-score-1",
    "exp211-code-toposort-1",
    "exp233-spec-req-ids",
    "exp233-spec-alignment-docs",
]

CURATED_METADATA: dict[str, dict[str, Any]] = {
    "exp211-live-gsm8k-923": {
        "task_slice": "live_gsm8k_semantic_failure",
        "gold_answer": 2,
        "coverage_markers": ["remaining", "row", "mint"],
        "claim_markers": ["remaining", "row", "mint"],
    },
    "exp211-live-gsm8k-506": {
        "task_slice": "live_gsm8k_semantic_failure",
        "gold_answer": 144,
        "coverage_markers": ["legs", "pairs", "cost"],
        "claim_markers": ["legs", "pairs", "cost"],
    },
    "exp211-live-gsm8k-1019": {
        "task_slice": "live_gsm8k_semantic_failure",
        "gold_answer": 98,
        "coverage_markers": ["checkout", "check in", "dinner"],
        "claim_markers": ["checkout", "check in", "dinner"],
    },
    "exp211-live-gsm8k-1077": {
        "task_slice": "live_gsm8k_semantic_failure",
        "gold_answer": 145,
        "coverage_markers": ["first", "second", "third", "fourth"],
        "claim_markers": ["first", "second", "third", "fourth"],
    },
    "exp211-instruction-bullets-1": {
        "task_slice": "instruction_surface_only",
        "gold_answer": None,
        "claim_markers": ["risk", "owner", "deadline"],
    },
    "exp211-instruction-json-1": {
        "task_slice": "instruction_surface_only",
        "gold_answer": None,
        "claim_markers": ["action", "reason", "confidence"],
    },
    "exp211-instruction-grounded-1": {
        "task_slice": "instruction_grounded",
        "gold_answer": ["P3", "P1"],
        "coverage_markers": ["under 50k", "before june", "p3", "p1"],
        "claim_markers": ["under 50k", "before june", "p3", "p1"],
    },
    "exp211-instruction-decision-1": {
        "task_slice": "instruction_grounded",
        "gold_answer": {"choice": "O3", "evidence": ["O3", "risk low"]},
        "coverage_markers": ["lower risk", "higher reach", "o3", "risk low"],
        "claim_markers": ["lower risk", "higher reach", "o3", "risk low"],
    },
    "exp211-instruction-filter-1": {
        "task_slice": "instruction_grounded",
        "gold_answer": ["A", "D"],
        "coverage_markers": ["vegan", "nut free", "20", "a", "d"],
        "claim_markers": ["vegan", "nut free", "20", "a", "d"],
    },
    "exp211-code-dedupe-1": {
        "task_slice": "code_typed_properties",
        "gold_answer": None,
        "claim_markers": ["dedupe", "first occurrence", "input order", "do not mutate"],
        "probe_cases": [
            {
                "args": [["a", "b", "a", "c", "b"]],
                "expected": ["a", "b", "c"],
                "immutable_arg_index": 0,
            }
        ],
    },
    "exp211-code-chunks-1": {
        "task_slice": "code_typed_properties",
        "gold_answer": None,
        "claim_markers": ["chunks", "at most size", "consecutive"],
        "probe_cases": [
            {"args": [[1, 2, 3, 4, 5], 2], "expected": [[1, 2], [3, 4], [5]]},
            {"args": [[1, 2], 5], "expected": [[1, 2]]},
        ],
    },
    "exp211-code-score-1": {
        "task_slice": "code_typed_properties",
        "gold_answer": None,
        "claim_markers": ["sum", "appearing at least once", "keyword"],
        "probe_cases": [
            {
                "args": ["red blue", {"red": 2, "blue": 3, "green": 5}],
                "expected": 5,
            },
            {
                "args": ["red red blue", {"red": 2, "blue": 3, "green": 5}],
                "expected": 5,
            },
        ],
    },
    "exp211-code-toposort-1": {
        "task_slice": "code_typed_properties",
        "gold_answer": None,
        "claim_markers": ["topological order", "directed acyclic graph", "raises valueerror"],
        "probe_cases": [
            {
                "args": [[("a", "b"), ("b", "c")]],
                "validator": "topological_order",
                "nodes": ["a", "b", "c"],
                "edges": [("a", "b"), ("b", "c")],
            },
            {
                "args": [[("a", "b"), ("b", "a")]],
                "expect_exception": "ValueError",
            },
        ],
    },
}


def get_repo_root() -> Path:
    override = os.environ.get("CARNOT_REPO_ROOT")
    if override:
        return Path(override).resolve()
    return Path(__file__).resolve().parents[1]


REPO_ROOT = get_repo_root()
SOURCE_REPO_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_PATH = REPO_ROOT / "data" / "research" / "constraint_ir_benchmark_211.jsonl"
RESULTS_PATH = REPO_ROOT / "results" / "experiment_233_results.json"
POLICY_PATH = REPO_ROOT / "results" / "output_policy_233.json"


def _load_exp213_module() -> Any:
    module_path = REPO_ROOT / "scripts" / "experiment_213_monitorability_audit.py"
    spec = importlib.util.spec_from_file_location(
        "experiment_213_monitorability_audit",
        module_path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_EXP213 = _load_exp213_module()


def get_run_timestamp() -> str:
    return datetime.now(UTC).isoformat()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")


def load_benchmark_records(path: Path) -> list[dict[str, Any]]:
    return _EXP213.load_benchmark_records(path)


def _extract_repo_excerpt(path: Path, start_marker: str, end_marker: str) -> str:
    text = path.read_text(encoding="utf-8")
    start = text.index(start_marker)
    end = text.index(end_marker, start) if end_marker else len(text)
    return text[start:end].strip()


def _repo_spec_cases() -> list[dict[str, Any]]:
    source_root = REPO_ROOT if (REPO_ROOT / "openspec").exists() else SOURCE_REPO_ROOT
    spec_path = source_root / "openspec" / "capabilities" / "verifiable-reasoning" / "spec.md"
    agents_path = source_root / "AGENTS.md"

    structured_excerpt = _extract_repo_excerpt(
        spec_path,
        "### REQ-VERIFY-022",
        "### REQ-VERIFY-025",
    )
    agents_excerpt = _extract_repo_excerpt(
        agents_path,
        "## Required Startup",
        "",
    )
    return [
        {
            "example_id": "exp233-spec-req-ids",
            "source_family": "repo_spec_grounding",
            "task_slice": "repo_spec_grounding",
            "expected_answer_schema": {"type": "comma_separated_list"},
            "free_form_reasoning_monitorable": True,
            "prompt_source_path": "openspec/capabilities/verifiable-reasoning/spec.md",
            "gold_answer": ["REQ-VERIFY-022", "REQ-VERIFY-023", "REQ-VERIFY-024"],
            "claim_markers": [
                "req verify 022",
                "req verify 023",
                "req verify 024",
                "structured reasoning",
            ],
            "gold_atomic_constraints": [
                {
                    "type": "grounded_selection",
                    "value": [
                        "REQ-VERIFY-022",
                        "REQ-VERIFY-023",
                        "REQ-VERIFY-024",
                    ],
                }
            ],
            "prompt": (
                "Use only this checked-in repo excerpt.\n\n"
                f"{structured_excerpt}\n\n"
                "Return the three requirement IDs that cover structured reasoning emission, "
                "validation/retry, and policy-gated routing. Return them as a comma-separated "
                "list in the same order they appear."
            ),
        },
        {
            "example_id": "exp233-spec-alignment-docs",
            "source_family": "repo_spec_grounding",
            "task_slice": "repo_spec_grounding",
            "expected_answer_schema": {"type": "comma_separated_list"},
            "free_form_reasoning_monitorable": True,
            "prompt_source_path": "AGENTS.md",
            "gold_answer": [
                "openspec/",
                "_bmad/traceability.md",
                "ops/status.md",
                "ops/changelog.md",
            ],
            "claim_markers": [
                "openspec",
                "_bmad traceability",
                "ops status",
                "ops changelog",
            ],
            "gold_atomic_constraints": [
                {
                    "type": "grounded_selection",
                    "value": [
                        "openspec/",
                        "_bmad/traceability.md",
                        "ops/status.md",
                        "ops/changelog.md",
                    ],
                }
            ],
            "prompt": (
                "Use only this checked-in repo excerpt.\n\n"
                f"{agents_excerpt}\n\n"
                "Return the four repo paths the excerpt says to keep aligned before reporting "
                "done. Return them as a comma-separated list in the same order they appear."
            ),
        },
    ]


def build_representative_subset(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_id = {record["example_id"]: record for record in records}
    subset: list[dict[str, Any]] = []
    repo_cases = {case["example_id"]: case for case in _repo_spec_cases()}
    for example_id in SUBSET_EXAMPLE_IDS:
        if example_id in repo_cases:
            subset.append(dict(repo_cases[example_id]))
            continue
        if example_id not in by_id:
            raise KeyError(f"Missing Exp 211 example '{example_id}' in benchmark.")
        enriched = dict(by_id[example_id])
        enriched.update(CURATED_METADATA[example_id])
        enriched["prompt_source_path"] = ""
        subset.append(enriched)
    return subset


def _json_answer_hint(example: dict[str, Any]) -> str:
    schema_type = str(example["expected_answer_schema"]["type"])
    if schema_type == "python_function":
        return "final_answer must be a JSON string containing only the Python function."
    if schema_type == "json_object":
        return "final_answer must be a JSON object matching the task schema."
    if schema_type == "comma_separated_list":
        return "final_answer must be a JSON array of strings in order."
    if schema_type == "bullet_list":
        return "final_answer must be a JSON array of bullet strings."
    return "final_answer must contain the task answer in its native format."


def build_mode_prompt(example: dict[str, Any], mode: str) -> str:
    header = (
        f"Audit Example ID: {example['example_id']}\n"
        f"Audit Mode: {mode}\n"
        "Carnot output policy refresh. Follow the requested response contract exactly.\n\n"
    )
    task = f"Task:\n{example['prompt']}\n\n"

    if mode == "free_form_reasoning":
        if example["task_slice"] == "code_typed_properties":
            contract = (
                "Respond using this exact layout:\n"
                "REASONING:\n"
                "<2-4 short sentences>\n"
                "FINAL:\n"
                "```python\n"
                "<final function only>\n"
                "```\n"
            )
        else:
            contract = (
                "Respond using this exact layout:\n"
                "REASONING:\n"
                "<2-4 short lines of plain-text reasoning>\n"
                "FINAL:\n"
                "<final answer only in the task's native format>\n"
            )
    elif mode == "answer_only_terse":
        if example["task_slice"] == "code_typed_properties":
            contract = "Return only one complete Python function definition. No explanation.\n"
        elif example["task_slice"] == "live_gsm8k_semantic_failure":
            contract = "Return only the final numeric answer. No explanation.\n"
        else:
            contract = "Return only the final answer in the task's native format. No explanation.\n"
    elif mode == "minimal_json":
        contract = (
            "Return strict JSON only with keys final_answer and claims.\n"
            'Use this shape exactly: {"final_answer": ..., "claims": [...]}.\n'
            f"{_json_answer_hint(example)}\n"
            "claims must be a short JSON array of verifier-visible evidence strings.\n"
            "Use [] when there are no useful claims.\n"
        )
    elif mode == "grammar_gated_json":
        contract = (
            "Return a single-line JSON object only. Use strict key order "
            '{"final_answer": ..., "claims": [...]} and no other top-level keys.\n'
            f"{_json_answer_hint(example)}\n"
            "claims must be a short JSON array of plain strings. "
            "No markdown. No prose outside the JSON.\n"
        )
    else:
        raise ValueError(f"Unknown mode: {mode}")

    return header + task + contract


def max_new_tokens_for(example: dict[str, Any], mode: str) -> int:
    if example["task_slice"] == "code_typed_properties":
        return {
            "free_form_reasoning": 320,
            "answer_only_terse": 220,
            "minimal_json": 300,
            "grammar_gated_json": 340,
        }[mode]
    return {
        "free_form_reasoning": 180,
        "answer_only_terse": 96,
        "minimal_json": 140,
        "grammar_gated_json": 180,
    }[mode]


def parse_structured_payload(raw_response: str) -> dict[str, Any] | None:
    return _EXP213.parse_structured_payload(raw_response)


def extract_final_section(raw_response: str, mode: str) -> str:
    if mode in JSON_MODES:
        payload = parse_structured_payload(raw_response)
        if payload is None:
            return ""
        final_answer = payload.get("final_answer")
        if isinstance(final_answer, (dict, list)):
            return json.dumps(final_answer)
        return str(final_answer or "")
    return _EXP213.extract_final_section(raw_response, mode)


def constraint_ratio(matches: int, total: int) -> float:
    return _EXP213.constraint_ratio(matches, total)


def _claim_marker_source(raw_response: str, structured_payload: dict[str, Any] | None) -> str:
    source = raw_response
    if structured_payload is None:
        return source
    claims = structured_payload.get("claims", [])
    checks = structured_payload.get("checks", [])
    if isinstance(claims, list):
        source += " " + json.dumps(claims)
    if isinstance(checks, list):
        source += " " + json.dumps(checks)
    return source


def _claim_coverage(
    example: dict[str, Any],
    raw_response: str,
    structured_payload: dict[str, Any] | None,
) -> float:
    markers = list(example.get("claim_markers", []))
    if not markers:
        return 0.0
    normalized = _EXP213.normalize_text(_claim_marker_source(raw_response, structured_payload))
    hits = sum(1 for marker in markers if _EXP213.normalize_text(marker) in normalized)
    return constraint_ratio(hits, len(markers))


def _evaluate_repo_spec_grounding(
    example: dict[str, Any],
    mode: str,
    raw_response: str,
) -> dict[str, Any]:
    final_value = extract_final_section(raw_response, mode)
    parsed = _EXP213.parse_comma_list_answer(final_value)
    parseable = parsed is not None
    gold = [str(item) for item in list(example["gold_answer"])]
    parsed_items = parsed or []
    exact = parsed_items == gold
    matched = sum(
        1
        for observed, expected in zip(parsed_items, gold, strict=False)
        if str(observed) == str(expected)
    )
    partial = constraint_ratio(matched, len(gold))
    return {
        "parse_success": parseable,
        "answer_quality": 1.0 if exact else partial,
        "exact_satisfaction": 1.0 if exact else 0.0,
        "partial_satisfaction": 1.0 if exact else partial,
    }


def _evaluate_instruction_surface_only(
    example: dict[str, Any],
    mode: str,
    raw_response: str,
    structured_payload: dict[str, Any] | None,
) -> dict[str, Any]:
    schema_type = str(example["expected_answer_schema"]["type"])
    final_value: Any = (
        structured_payload.get("final_answer")
        if mode in JSON_MODES and structured_payload is not None
        else extract_final_section(raw_response, mode)
    )
    parse_success = False
    satisfied = 0
    total = len(example["gold_atomic_constraints"])

    if schema_type == "bullet_list":
        bullets = _EXP213.parse_bullet_answer(final_value)
        parse_success = bullets is not None
        if bullets is not None:
            if len(bullets) == 3:
                satisfied += 1
            if all(4 <= len(bullet.split()) <= 7 for bullet in bullets):
                satisfied += 1
            joined = " ".join(bullets).lower()
            for token in ("risk", "owner", "deadline"):
                if token in joined:
                    satisfied += 1
            if "urgent" not in joined:
                satisfied += 1
    elif schema_type == "json_object":
        payload = _EXP213.parse_json_object_answer(final_value)
        parse_success = payload is not None
        if payload is not None:
            keys = list(payload.keys())
            if keys == ["action", "reason", "confidence"]:
                satisfied += 1
            if set(keys).issubset({"action", "reason", "confidence"}):
                satisfied += 1
            if payload.get("action") in {"approve", "hold", "reject"}:
                satisfied += 1
            if payload.get("confidence") in {"low", "medium", "high"}:
                satisfied += 1
    else:
        raise ValueError(f"Unsupported instruction surface schema: {schema_type}")

    answer_quality = constraint_ratio(satisfied, total)
    return {
        "parse_success": parse_success,
        "answer_quality": answer_quality,
        "exact_satisfaction": 1.0 if answer_quality == 1.0 else 0.0,
        "partial_satisfaction": answer_quality,
        "claim_coverage": _claim_coverage(example, raw_response, structured_payload),
    }


def _evaluate_instruction_grounded(
    example: dict[str, Any],
    mode: str,
    raw_response: str,
    structured_payload: dict[str, Any] | None,
) -> dict[str, Any]:
    schema_type = str(example["expected_answer_schema"]["type"])
    final_value: Any = (
        structured_payload.get("final_answer")
        if mode in JSON_MODES and structured_payload is not None
        else extract_final_section(raw_response, mode)
    )
    parse_success = False
    answer_quality = 0.0

    if schema_type == "comma_separated_list":
        parsed = _EXP213.parse_comma_list_answer(final_value)
        parse_success = parsed is not None
        if parsed is not None and parsed == example["gold_answer"]:
            answer_quality = 1.0
    elif schema_type == "json_object":
        payload = _EXP213.parse_json_object_answer(final_value)
        parse_success = payload is not None
        if payload is not None:
            keys_ok = list(payload.keys()) == ["choice", "evidence"]
            choice_ok = payload.get("choice") == example["gold_answer"]["choice"]
            evidence = payload.get("evidence")
            evidence_ok = False
            if isinstance(evidence, list):
                evidence_ok = evidence == example["gold_answer"]["evidence"]
            elif isinstance(evidence, str):
                evidence_ok = all(term in evidence for term in example["gold_answer"]["evidence"])
            answer_quality = constraint_ratio(sum((keys_ok, choice_ok, evidence_ok)), 3)
    else:
        raise ValueError(f"Unsupported grounded schema: {schema_type}")

    claim_coverage = _claim_coverage(example, raw_response, structured_payload)
    return {
        "parse_success": parse_success,
        "answer_quality": answer_quality,
        "exact_satisfaction": 1.0 if answer_quality == 1.0 else 0.0,
        "partial_satisfaction": max(answer_quality, claim_coverage),
        "claim_coverage": claim_coverage,
    }


def _function_signature_from_ast(node: ast.FunctionDef) -> str:
    return _EXP213.function_signature_from_ast(node)


def _topological_order_valid(
    order: Any,
    nodes: list[str],
    edges: list[tuple[str, str]],
) -> bool:
    if not isinstance(order, list) or not all(isinstance(node, str) for node in order):
        return False
    if order != list(dict.fromkeys(order)):
        return False
    if set(order) != set(nodes):
        return False
    positions = {node: index for index, node in enumerate(order)}
    return all(positions[left] < positions[right] for left, right in edges)


def _run_code_probe(function_obj: Any, probe: dict[str, Any]) -> bool:
    args = copy.deepcopy(list(probe["args"]))
    expected_exception = probe.get("expect_exception")
    try:
        result = function_obj(*args)
    except Exception as exc:  # pragma: no cover - exercised in tests via behavior
        if expected_exception is None:
            return False
        return exc.__class__.__name__ == str(expected_exception)

    if expected_exception is not None:
        return False

    immutable_index = probe.get("immutable_arg_index")
    if immutable_index is not None:
        original = copy.deepcopy(list(probe["args"])[immutable_index])
        if args[immutable_index] != original:
            return False

    validator = probe.get("validator")
    if validator == "topological_order":
        return _topological_order_valid(result, list(probe["nodes"]), list(probe["edges"]))
    return result == probe["expected"]


def _evaluate_code_typed_properties(
    example: dict[str, Any],
    mode: str,
    raw_response: str,
) -> dict[str, Any]:
    code = _EXP213.extract_python_code(extract_final_section(raw_response, mode))
    if code is None:
        return {
            "parse_success": False,
            "answer_quality": 0.0,
            "exact_satisfaction": 0.0,
            "partial_satisfaction": 0.0,
        }

    try:
        tree = ast.parse(code)
    except SyntaxError:
        return {
            "parse_success": False,
            "answer_quality": 0.0,
            "exact_satisfaction": 0.0,
            "partial_satisfaction": 0.0,
        }

    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    function_node = next(
        (node for node in functions if node.name == example["expected_answer_schema"]["name"]),
        None,
    )
    if function_node is None:
        return {
            "parse_success": False,
            "answer_quality": 0.0,
            "exact_satisfaction": 0.0,
            "partial_satisfaction": 0.0,
        }

    namespace: dict[str, Any] = {}
    exec(code, namespace)
    function_obj = namespace.get(example["expected_answer_schema"]["name"])
    if function_obj is None:
        return {
            "parse_success": False,
            "answer_quality": 0.0,
            "exact_satisfaction": 0.0,
            "partial_satisfaction": 0.0,
        }

    passed = sum(
        1 for probe in list(example["probe_cases"]) if _run_code_probe(function_obj, dict(probe))
    )
    answer_quality = constraint_ratio(passed, len(example["probe_cases"]))
    signature = _function_signature_from_ast(function_node)
    exact = answer_quality == 1.0 and signature == example["expected_answer_schema"]["signature"]
    return {
        "parse_success": True,
        "answer_quality": answer_quality,
        "exact_satisfaction": 1.0 if exact else 0.0,
        "partial_satisfaction": answer_quality,
    }


def evaluate_response(example: dict[str, Any], mode: str, raw_response: str) -> dict[str, Any]:
    structured_payload = parse_structured_payload(raw_response) if mode in JSON_MODES else None
    base_mode = "structured_json" if mode in JSON_MODES else mode

    if example["task_slice"] == "live_gsm8k_semantic_failure":
        base = _EXP213.evaluate_live_semantic_failure(
            example,
            base_mode,
            raw_response,
            structured_payload,
        )
        claim_coverage = _claim_coverage(example, raw_response, structured_payload)
        partial = max(float(base["answer_quality"]), claim_coverage)
        result = {
            "parse_success": bool(base["parseable"]),
            "answer_quality": float(base["answer_quality"]),
            "exact_satisfaction": 1.0 if float(base["answer_quality"]) == 1.0 else 0.0,
            "partial_satisfaction": partial,
            "claim_coverage": claim_coverage,
        }
    elif example["task_slice"] == "instruction_surface_only":
        result = _evaluate_instruction_surface_only(example, mode, raw_response, structured_payload)
    elif example["task_slice"] == "instruction_grounded":
        result = _evaluate_instruction_grounded(example, mode, raw_response, structured_payload)
    elif example["task_slice"] == "code_typed_properties":
        result = _evaluate_code_typed_properties(example, mode, raw_response)
        result["claim_coverage"] = _claim_coverage(example, raw_response, structured_payload)
    elif example["task_slice"] == "repo_spec_grounding":
        result = _evaluate_repo_spec_grounding(example, mode, raw_response)
        result["claim_coverage"] = _claim_coverage(example, raw_response, structured_payload)
        result["partial_satisfaction"] = max(
            float(result["partial_satisfaction"]),
            float(result["claim_coverage"]),
        )
    else:
        raise ValueError(f"Unsupported task slice: {example['task_slice']}")

    result.update(
        {
            "example_id": example["example_id"],
            "task_slice": example["task_slice"],
            "source_family": example["source_family"],
            "mode": mode,
            "raw_response": raw_response,
        }
    )
    return result


def _needs_retry(example: dict[str, Any], mode: str, evaluation: dict[str, Any]) -> bool:
    if mode not in JSON_MODES:
        return False
    if not bool(evaluation["parse_success"]):
        return True
    if (
        float(evaluation["exact_satisfaction"]) < 1.0
        and float(evaluation["partial_satisfaction"]) < 0.9
    ):
        return True
    return (
        example["task_slice"]
        in {
            "live_gsm8k_semantic_failure",
            "instruction_grounded",
            "repo_spec_grounding",
        }
        and float(evaluation["claim_coverage"]) < 0.75
    )


def _evaluation_rank(evaluation: dict[str, Any]) -> tuple[float, float, float, float]:
    return (
        float(evaluation["exact_satisfaction"]),
        float(evaluation["partial_satisfaction"]),
        1.0 if evaluation["parse_success"] else 0.0,
        float(evaluation["claim_coverage"]),
    )


def _build_retry_prompt(example: dict[str, Any], mode: str, evaluation: dict[str, Any]) -> str:
    issues: list[str] = []
    if not bool(evaluation["parse_success"]):
        issues.append("previous response did not parse cleanly")
    if float(evaluation["exact_satisfaction"]) < 1.0:
        issues.append("final answer was incomplete or wrong")
    if float(evaluation["claim_coverage"]) < 0.75:
        issues.append("claims missed verifier-visible evidence")
    issue_text = "; ".join(issues) if issues else "response quality needs improvement"
    return (
        f"Issue: {issue_text}\n"
        "The previous response did not satisfy Carnot's output contract.\n\n"
        f"{build_mode_prompt(example, mode)}"
    )


def safe_mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return round(sum(values) / len(values), 4)


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_task_slice: dict[str, dict[str, dict[str, float]]] = {}
    for task_slice in sorted({record["task_slice"] for record in records}):
        by_task_slice[task_slice] = {}
        for mode in MODE_ORDER:
            mode_records = [
                record
                for record in records
                if record["task_slice"] == task_slice and record["mode"] == mode
            ]
            if not mode_records:
                continue
            by_task_slice[task_slice][mode] = {
                "n_records": len(mode_records),
                "parse_success_rate": safe_mean(
                    [1.0 if record["parse_success"] else 0.0 for record in mode_records]
                ),
                "retry_rate": safe_mean(
                    [1.0 if int(record["retries_used"]) > 0 else 0.0 for record in mode_records]
                ),
                "mean_retries_used": safe_mean(
                    [float(record["retries_used"]) for record in mode_records]
                ),
                "answer_quality_mean": safe_mean(
                    [float(record["answer_quality"]) for record in mode_records]
                ),
                "exact_satisfaction_rate": safe_mean(
                    [float(record["exact_satisfaction"]) for record in mode_records]
                ),
                "partial_satisfaction_mean": safe_mean(
                    [float(record["partial_satisfaction"]) for record in mode_records]
                ),
                "claim_coverage_mean": safe_mean(
                    [float(record["claim_coverage"]) for record in mode_records]
                ),
                "repair_usefulness_rate": safe_mean(
                    [float(record["repair_useful"]) for record in mode_records]
                ),
                "mean_prompt_tokens": safe_mean(
                    [float(record["prompt_tokens"]) for record in mode_records]
                ),
                "mean_completion_tokens": safe_mean(
                    [float(record["completion_tokens"]) for record in mode_records]
                ),
                "mean_total_tokens": safe_mean(
                    [float(record["total_tokens"]) for record in mode_records]
                ),
                "mean_latency_seconds": safe_mean(
                    [float(record["latency_seconds"]) for record in mode_records]
                ),
            }

    by_model: dict[str, dict[str, dict[str, float]]] = {}
    for model_name in sorted({record["model_name"] for record in records}):
        by_model[model_name] = {}
        for mode in MODE_ORDER:
            mode_records = [
                record
                for record in records
                if record["model_name"] == model_name and record["mode"] == mode
            ]
            if not mode_records:
                continue
            by_model[model_name][mode] = {
                "parse_success_rate": safe_mean(
                    [1.0 if record["parse_success"] else 0.0 for record in mode_records]
                ),
                "retry_rate": safe_mean(
                    [1.0 if int(record["retries_used"]) > 0 else 0.0 for record in mode_records]
                ),
                "answer_quality_mean": safe_mean(
                    [float(record["answer_quality"]) for record in mode_records]
                ),
                "exact_satisfaction_rate": safe_mean(
                    [float(record["exact_satisfaction"]) for record in mode_records]
                ),
                "partial_satisfaction_mean": safe_mean(
                    [float(record["partial_satisfaction"]) for record in mode_records]
                ),
                "claim_coverage_mean": safe_mean(
                    [float(record["claim_coverage"]) for record in mode_records]
                ),
                "repair_usefulness_rate": safe_mean(
                    [float(record["repair_useful"]) for record in mode_records]
                ),
            }

    return {
        "n_responses": len(records),
        "by_task_slice": by_task_slice,
        "by_model": by_model,
    }


def _json_mode_choice(modes: dict[str, dict[str, float]]) -> tuple[str, dict[str, float]]:
    minimal = dict(modes.get("minimal_json", {}))
    grammar = dict(modes.get("grammar_gated_json", {}))
    if not grammar:
        return "minimal_json", minimal
    if not minimal:
        return "grammar_gated_json", grammar

    grammar_wins = (
        float(grammar.get("exact_satisfaction_rate", 0.0))
        > float(minimal.get("exact_satisfaction_rate", 0.0)) + 0.05
    ) or (
        float(grammar.get("parse_success_rate", 0.0))
        > float(minimal.get("parse_success_rate", 0.0)) + 0.12
        and float(grammar.get("partial_satisfaction_mean", 0.0))
        >= float(minimal.get("partial_satisfaction_mean", 0.0)) + 0.05
    )
    if grammar_wins:
        return "grammar_gated_json", grammar
    return "minimal_json", minimal


def derive_policy(summary: dict[str, Any]) -> dict[str, Any]:
    per_task_slice: dict[str, dict[str, Any]] = {}
    request_json_when: list[str] = []
    accept_terse_when: list[str] = []
    avoid_free_form_when: list[str] = []

    for task_slice, modes in summary["by_task_slice"].items():
        free_form = dict(modes.get("free_form_reasoning", {}))
        terse = dict(modes.get("answer_only_terse", {}))
        json_mode, json_metrics = _json_mode_choice(modes)

        terse_good_enough = bool(terse) and (
            float(terse.get("exact_satisfaction_rate", 0.0))
            >= float(json_metrics.get("exact_satisfaction_rate", 0.0)) - 0.05
            and float(terse.get("partial_satisfaction_mean", 0.0))
            >= float(json_metrics.get("partial_satisfaction_mean", 0.0)) - 0.05
        )
        json_materially_better = (
            float(json_metrics.get("exact_satisfaction_rate", 0.0))
            >= float(terse.get("exact_satisfaction_rate", 0.0)) + 0.1
            or float(json_metrics.get("claim_coverage_mean", 0.0))
            >= float(terse.get("claim_coverage_mean", 0.0)) + 0.3
        )

        if (
            task_slice in {"code_typed_properties", "instruction_surface_only"}
            and terse_good_enough
        ):
            recommended_mode = "answer_only_terse"
        elif json_materially_better:
            recommended_mode = json_mode
        elif terse_good_enough:
            recommended_mode = "answer_only_terse"
        else:
            recommended_mode = json_mode if json_metrics else "answer_only_terse"

        if recommended_mode == "answer_only_terse":
            accept_terse_when.append(
                f"Accept answer_only_terse for {task_slice} when it stays within the measured "
                "exact/partial satisfaction envelope of the JSON alternatives."
            )
        else:
            request_json_when.append(
                f"Request {recommended_mode} for {task_slice} when measured claim coverage or "
                "exact satisfaction materially exceeds answer_only_terse."
            )

        if float(free_form.get("parse_success_rate", 0.0)) < 0.85 or float(
            free_form.get("exact_satisfaction_rate", 0.0)
        ) + 0.1 < max(
            float(terse.get("exact_satisfaction_rate", 0.0)),
            float(json_metrics.get("exact_satisfaction_rate", 0.0)),
        ):
            avoid_free_form_when.append(
                f"Avoid free_form_reasoning on {task_slice} when terse or JSON modes provide a "
                "clearer verifier-visible state."
            )

        per_task_slice[task_slice] = {
            "recommended_mode": recommended_mode,
            "retry_budget": MODE_RETRY_BUDGETS[recommended_mode],
            "rationale": {
                "free_form_reasoning": free_form,
                "answer_only_terse": terse,
                "minimal_json": dict(modes.get("minimal_json", {})),
                "grammar_gated_json": dict(modes.get("grammar_gated_json", {})),
            },
        }

    mode_guidance: dict[str, str] = {}
    for model_name, model_modes in summary.get("by_model", {}).items():
        minimal_exact = float(
            model_modes.get("minimal_json", {}).get("exact_satisfaction_rate", 0.0)
        )
        grammar_exact = float(
            model_modes.get("grammar_gated_json", {}).get("exact_satisfaction_rate", 0.0)
        )
        if grammar_exact > minimal_exact + 0.02:
            mode_guidance[model_name] = (
                "Use grammar_gated_json only on the highest-stakes JSON slices; it outperformed "
                "minimal_json for this model."
            )
        else:
            mode_guidance[model_name] = (
                "Prefer minimal_json before grammar_gated_json when JSON is warranted; keep "
                "answer_only_terse as the default elsewhere."
            )

    return {
        "mode_defaults": {
            "fallback_mode": "answer_only_terse",
            "retry_budgets": MODE_RETRY_BUDGETS,
        },
        "global_policy": {
            "request_json_when": request_json_when,
            "accept_terse_when": accept_terse_when,
            "avoid_free_form_when": ["free_form_reasoning", *avoid_free_form_when],
        },
        "per_task_slice": per_task_slice,
        "mode_guidance": mode_guidance,
    }


def build_subset_summary(subset: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "n_examples": len(subset),
        "by_source_family": dict(Counter(example["source_family"] for example in subset)),
        "by_task_slice": dict(Counter(example["task_slice"] for example in subset)),
        "by_monitorability_flag": {
            "true": sum(1 for example in subset if example["free_form_reasoning_monitorable"]),
            "false": sum(1 for example in subset if not example["free_form_reasoning_monitorable"]),
        },
    }


def build_key_findings(summary: dict[str, Any], policy: dict[str, Any]) -> list[str]:
    findings = []
    for task_slice, recommendation in policy["per_task_slice"].items():
        findings.append(
            f"{task_slice}: recommended mode is {recommendation['recommended_mode']} with retry "
            f"budget {recommendation['retry_budget']}."
        )
    findings.append(
        "Exp 233 keeps JSON task-gated and prefers the smallest schema that materially improves "
        "verifier-visible evidence."
    )
    findings.append(
        f"Audit summary covers {summary['n_responses']} model-mode-example responses across "
        f"{len(summary['by_task_slice'])} task slices."
    )
    return findings


def count_tokens(tokenizer: Any, text: str) -> int:  # pragma: no cover
    return _EXP213.count_tokens(tokenizer, text)


def _record_attempt(
    tokenizer: Any,
    prompt: str,
    raw_response: str,
    evaluation: dict[str, Any],
    latency_seconds: float,
) -> dict[str, Any]:
    prompt_tokens = count_tokens(tokenizer, prompt)
    completion_tokens = count_tokens(tokenizer, raw_response)
    return {
        "prompt": prompt,
        "raw_response": raw_response,
        "parse_success": bool(evaluation["parse_success"]),
        "answer_quality": float(evaluation["answer_quality"]),
        "exact_satisfaction": float(evaluation["exact_satisfaction"]),
        "partial_satisfaction": float(evaluation["partial_satisfaction"]),
        "claim_coverage": float(evaluation["claim_coverage"]),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "latency_seconds": latency_seconds,
    }


def run_live_audit(subset: list[dict[str, Any]]) -> list[dict[str, Any]]:  # pragma: no cover
    if "CARNOT_FORCE_CPU" not in os.environ:
        os.environ["CARNOT_FORCE_CPU"] = "0"

    responses: list[dict[str, Any]] = []
    for model_spec in MODEL_SPECS:
        model, tokenizer = load_model(model_spec["hf_id"], device="cuda")
        if model is None or tokenizer is None:
            raise RuntimeError(f"Live load failed for {model_spec['hf_id']}")

        try:
            for mode in MODE_ORDER:
                retry_budget = MODE_RETRY_BUDGETS[mode]
                for example in subset:
                    prompt = build_mode_prompt(example, mode)
                    attempts: list[dict[str, Any]] = []
                    best_evaluation: dict[str, Any] | None = None
                    best_response = ""
                    initial_evaluation: dict[str, Any] | None = None
                    started = time.perf_counter()

                    for _attempt_index in range(retry_budget + 1):
                        attempt_started = time.perf_counter()
                        raw_response = generate(
                            model,
                            tokenizer,
                            prompt,
                            max_new_tokens=max_new_tokens_for(example, mode),
                        )
                        evaluation = evaluate_response(example, mode, raw_response)
                        attempt_record = _record_attempt(
                            tokenizer,
                            prompt,
                            raw_response,
                            evaluation,
                            round(time.perf_counter() - attempt_started, 4),
                        )
                        attempts.append(attempt_record)
                        if initial_evaluation is None:
                            initial_evaluation = dict(evaluation)
                        if best_evaluation is None or _evaluation_rank(
                            evaluation
                        ) > _evaluation_rank(best_evaluation):
                            best_evaluation = dict(evaluation)
                            best_response = raw_response
                        if (
                            not _needs_retry(example, mode, evaluation)
                            or len(attempts) > retry_budget
                        ):
                            break
                        prompt = _build_retry_prompt(example, mode, evaluation)

                    assert best_evaluation is not None
                    total_prompt_tokens = sum(int(attempt["prompt_tokens"]) for attempt in attempts)
                    total_completion_tokens = sum(
                        int(attempt["completion_tokens"]) for attempt in attempts
                    )
                    repair_useful = (
                        1.0
                        if initial_evaluation
                        and _evaluation_rank(best_evaluation) > _evaluation_rank(initial_evaluation)
                        else 0.0
                    )
                    record = dict(best_evaluation)
                    record.update(
                        {
                            "model_name": model_spec["name"],
                            "hf_id": model_spec["hf_id"],
                            "prompt_tokens": total_prompt_tokens,
                            "completion_tokens": total_completion_tokens,
                            "total_tokens": total_prompt_tokens + total_completion_tokens,
                            "latency_seconds": round(time.perf_counter() - started, 4),
                            "retry_budget": retry_budget,
                            "retries_used": max(0, len(attempts) - 1),
                            "repair_useful": repair_useful,
                            "initial_parse_success": bool(initial_evaluation["parse_success"])
                            if initial_evaluation
                            else False,
                            "initial_exact_satisfaction": float(
                                initial_evaluation["exact_satisfaction"]
                            )
                            if initial_evaluation
                            else 0.0,
                            "raw_response": best_response,
                            "attempts": attempts,
                        }
                    )
                    responses.append(record)
        finally:
            del model
            del tokenizer
            gc.collect()
            try:
                import torch

                torch.cuda.empty_cache()
            except Exception:
                pass

    return responses


def main() -> int:
    records = load_benchmark_records(BENCHMARK_PATH)
    subset = build_representative_subset(records)
    responses = run_live_audit(subset)
    summary = summarize_records(responses)
    policy = derive_policy(summary)
    timestamp = get_run_timestamp()

    results_payload = {
        "experiment": EXPERIMENT_LABEL,
        "run_date": RUN_DATE,
        "title": "Output policy refresh with minimal-schema JSON modes",
        "metadata": {
            "timestamp": timestamp,
            "inference_mode": "live_gpu",
            "benchmark_path": str(BENCHMARK_PATH.relative_to(REPO_ROOT)),
            "subset_example_ids": SUBSET_EXAMPLE_IDS,
            "modes": MODE_ORDER,
            "retry_budgets": MODE_RETRY_BUDGETS,
            "models": MODEL_SPECS,
        },
        "subset_summary": build_subset_summary(subset),
        "summary": summary,
        "key_findings": build_key_findings(summary, policy),
        "policy_path": str(POLICY_PATH.relative_to(REPO_ROOT)),
        "responses": responses,
    }
    policy_payload = {
        "experiment": EXPERIMENT_LABEL,
        "run_date": RUN_DATE,
        "title": "Output policy derived from Exp 233 refresh benchmark",
        "derived_from": str(RESULTS_PATH.relative_to(REPO_ROOT)),
        **policy,
    }

    write_json(RESULTS_PATH, results_payload)
    write_json(POLICY_PATH, policy_payload)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


# --- Exp 495 HarnessPatcher: DualGPUHarness.apply() injected — REQ-INFRA-057 ---
# Auto-injected because HarnessAudit flagged this script as loading two models
# without assigning any model to cuda:1.  apply() pins model[0] to cuda:0 and
# model[1] to cuda:1 when CARNOT_FORCE_LIVE=1 is set.  It is a no-op in CI so
# this block is safe to leave in place permanently.
try:
    from carnot.pipeline.dual_gpu_harness import DualGPUHarness as _Exp495DGH
    if "MODEL_SPECS" in vars():
        MODEL_SPECS = _Exp495DGH.from_env().apply(MODEL_SPECS)  # cuda:1 → model[1]
except Exception:  # noqa: BLE001
    pass  # best-effort injection; script continues even if harness import fails
