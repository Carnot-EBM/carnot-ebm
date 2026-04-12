#!/usr/bin/env python3
# ruff: noqa: E501
"""Experiment 218: shared dual-model live benchmark harness.

Builds one checkpointed CLI for the next live benchmark milestone:

- ``gsm8k_semantic``
- ``humaneval_property``
- ``constraint_ir``

The harness keeps paired comparisons honest by:

- restricting the model set to the two target small instruction-tuned models
- sampling cohorts deterministically
- recording one shared prompt seed per case and reusing it across
  ``baseline``, ``verify_only``, and ``verify_repair``
- checkpointing every benchmark/model/mode cell independently so long runs
  can resume without scrambling case order or duplicating results

Usage:
    .venv/bin/python scripts/experiment_218_live_dual_model_suite.py --help

Spec: REQ-VERIFY-025, REQ-VERIFY-026, REQ-VERIFY-027, REQ-VERIFY-028,
REQ-VERIFY-029, SCENARIO-VERIFY-025, SCENARIO-VERIFY-026,
SCENARIO-VERIFY-027, SCENARIO-VERIFY-028, SCENARIO-VERIFY-029
"""

from __future__ import annotations

import argparse
import ast
import copy
import gc
import hashlib
import importlib.util
import json
import os
import random
import re
import signal
import threading
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

RUN_DATE = "20260412"
EXPERIMENT = 218
SCHEMA_ARTIFACT = "carnot.live_dual_model_suite.v1"
DEFAULT_SAMPLE_SEED = 218
DEFAULT_MAX_REPAIRS = 3
MODE_ORDER = ("baseline", "verify_only", "verify_repair")
_EXPERIMENT_OUTPUT_RE = re.compile(r"experiment_(\d+)_results\.json$")
_CONSTRAINT_IR_CODE_PROBE_TIMEOUT_SECONDS = 0.25

MODEL_SPECS: list[dict[str, str]] = [
    {"name": "Qwen3.5-0.8B", "hf_id": "Qwen/Qwen3.5-0.8B"},
    {"name": "Gemma4-E4B-it", "hf_id": "google/gemma-4-E4B-it"},
]

BENCHMARK_SPECS: dict[str, dict[str, Any]] = {
    "gsm8k_semantic": {
        "title": "Live GSM8K semantic benchmark",
        "case_schema": "gsm8k_semantic.v1",
        "default_sample_size": 200,
        "source_artifacts": [
            "results/experiment_206_results.json",
            "results/experiment_207_results.json",
            "results/monitorability_policy_213.json",
        ],
    },
    "humaneval_property": {
        "title": "Live HumanEval property benchmark",
        "case_schema": "humaneval_property.v1",
        "default_sample_size": 50,
        "source_artifacts": [
            "results/experiment_208_results.json",
            "results/monitorability_policy_213.json",
        ],
    },
    "constraint_ir": {
        "title": "Live prompt-side constraint benchmark",
        "case_schema": "constraint_ir.v1",
        "default_sample_size": 100,
        "source_artifacts": [
            "data/research/constraint_ir_benchmark_211.jsonl",
            "results/experiment_213_results.json",
            "results/monitorability_policy_213.json",
        ],
    },
}

_CONSTRAINT_IR_LITERAL_TYPES = {
    "count_exact",
    "word_count_range",
    "must_include_token",
    "must_include_phrase",
    "forbidden_token",
    "forbidden_phrase",
    "json_exact_keys",
    "no_extra_keys",
    "enum_membership",
    "section_order",
    "sentence_count_per_section",
    "step_count",
    "step_roles",
    "yaml_exact_keys",
    "sentence_count",
    "function_name",
    "signature",
    "return_type",
    "forbidden_api",
}
_CONSTRAINT_IR_OUTPUT_STYLES = (
    "structured_json",
    "free_form_reasoning",
    "answer_only_terse",
    "code_only",
    "other_unstructured",
)
_PLAN_ROLE_KEYWORDS: dict[str, tuple[str, ...]] = {
    "baseline_metrics": ("baseline", "metric"),
    "stage_change": ("stage", "change"),
    "validate_result": ("validate", "result"),
    "rollback": ("rollback",),
    "inventory_dependencies": ("inventory", "dependenc"),
    "dry_run": ("dry run",),
    "execute_change": ("execute", "change"),
    "gather_logs": ("gather", "log"),
    "isolate_scope": ("isolate", "scope"),
    "apply_fix": ("apply", "fix"),
}
_TONE_FORBIDDEN_MARKERS = {
    "panic mode",
    "total mess",
    "bad surprise",
    "!",
}
_CODE_PROBES: dict[str, list[dict[str, Any]]] = {
    "dedupe_preserve_order": [
        {
            "args": [["a", "b", "a", "c", "b"]],
            "expected": ["a", "b", "c"],
            "immutable_arg_index": 0,
        }
    ],
    "dedupe_casefold": [{"args": [["A", "b", "a", "B", "c"]], "expected": ["A", "b", "c"]}],
    "dedupe_tuples": [{"args": [[(1, 2), (1, 2), (2, 3), (1, 2)]], "expected": [(1, 2), (2, 3)]}],
    "slugify": [{"args": ["Hello,  World!!"], "expected": "hello-world"}],
    "slugify_filename": [{"args": ["  Q1 Report!  "], "expected": "q1-report"}],
    "slugify_tag": [{"args": ["Hello World!"], "expected": "hello_world"}],
    "merge_intervals": [
        {
            "args": [[(1, 3), (2, 5), (8, 9)]],
            "expected": [(1, 5), (8, 9)],
            "immutable_arg_index": 0,
        }
    ],
    "merge_touching_intervals": [
        {"args": [[(1, 2), (2, 4), (6, 7)]], "expected": [(1, 4), (6, 7)]}
    ],
    "insert_interval": [{"args": [[(1, 2), (5, 7)], (2, 6)], "expected": [(1, 7)]}],
    "topo_sort": [
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
    "topo_sort_nodes": [
        {
            "args": [["a", "b", "c"], [("a", "b")]],
            "validator": "topological_order",
            "nodes": ["a", "b", "c"],
            "edges": [("a", "b")],
        },
        {
            "args": [["a", "b"], [("a", "b"), ("b", "a")]],
            "expect_exception": "ValueError",
        },
    ],
    "course_order": [
        {
            "args": [[("math", "ai"), ("ai", "ml")]],
            "validator": "topological_order",
            "nodes": ["math", "ai", "ml"],
            "edges": [("math", "ai"), ("ai", "ml")],
        },
        {
            "args": [[("math", "ai"), ("ai", "math")]],
            "expect_exception": "ValueError",
        },
    ],
    "normalize_us_phone": [
        {"args": ["(555) 123-4567"], "expected": "555-123-4567"},
        {"args": ["555-12"], "expect_exception": "ValueError"},
    ],
    "normalize_ext_phone": [{"args": ["555-123-4567 x89"], "expected": ("555-123-4567", "89")}],
    "normalize_digits_only": [
        {"args": ["tel:+1 (555) 123-4567"], "expected": "5551234567"},
        {"args": ["555-12"], "expect_exception": "ValueError"},
    ],
    "parse_user_row": [
        {
            "args": [" 42, Ada Lovelace, "],
            "expected": {"id": "42", "name": "Ada Lovelace", "email": None},
        }
    ],
    "parse_metric_row": [
        {"args": ["mon, 5, 120"], "expected": {"day": "mon", "errors": 5, "latency_ms": 120}}
    ],
    "parse_flag_row": [
        {
            "args": ["feature-x,true,ops"],
            "expected": {"name": "feature-x", "enabled": True, "owner": "ops"},
        }
    ],
    "rolling_average": [{"args": [[1.0, 2.0, 3.0, 4.0], 2], "expected": [1.5, 2.5, 3.5]}],
    "rolling_sum": [{"args": [[1, 2, 3, 4], 3], "expected": [6, 9]}],
    "rolling_max": [{"args": [[1, 3, 2, 5], 2], "expected": [3, 3, 5]}],
    "load_config": [
        {
            "args": [{"HOST": "localhost", "PORT": "8080"}],
            "expected": {"HOST": "localhost", "PORT": 8080, "MODE": "safe"},
        },
        {"args": [{"PORT": "8080"}], "expect_exception": "ValueError"},
    ],
    "load_retry_config": [
        {"args": [{"RETRIES": "2"}], "expected": {"RETRIES": 2, "TIMEOUT": 30}},
        {"args": [{}], "expect_exception": "ValueError"},
    ],
    "load_feature_config": [
        {
            "args": [{"OWNER": "ian", "ENABLED": "true"}],
            "expected": {"OWNER": "ian", "CHANNEL": "general", "ENABLED": True},
        },
        {"args": [{"ENABLED": "true"}], "expect_exception": "ValueError"},
    ],
    "group_by_team": [
        {
            "args": [
                [
                    {"team": "blue", "id": "1"},
                    {"team": "red", "id": "2"},
                    {"team": "blue", "id": "3"},
                ]
            ],
            "expected": {
                "blue": [{"team": "blue", "id": "1"}, {"team": "blue", "id": "3"}],
                "red": [{"team": "red", "id": "2"}],
            },
        }
    ],
    "group_by_priority": [
        {
            "args": [
                [
                    {"priority": "p1", "id": "a"},
                    {"priority": "p2", "id": "b"},
                    {"priority": "p1", "id": "c"},
                ]
            ],
            "expected": {"p1": ["a", "c"], "p2": ["b"]},
        }
    ],
    "group_words_by_initial": [
        {"args": [["Apple", "ant", "Boat"]], "expected": {"a": ["Apple", "ant"], "b": ["Boat"]}}
    ],
    "to_roman": [
        {"args": [944], "expected": "CMXLIV"},
        {"args": [0], "expect_exception": "ValueError"},
    ],
    "from_roman": [
        {"args": ["MCMXCIV"], "expected": 1994},
        {"args": ["IIII"], "expect_exception": "ValueError"},
    ],
    "is_valid_roman": [
        {"args": ["MCMXCIV"], "expected": True},
        {"args": ["IIII"], "expected": False},
    ],
    "chunk_list": [{"args": [[1, 2, 3, 4, 5], 2], "expected": [[1, 2], [3, 4], [5]]}],
    "chunk_text": [{"args": ["abcdef", 2], "expected": ["ab", "cd", "ef"]}],
    "chunk_pairs": [
        {"args": [[(1, 2), (3, 4), (5, 6)], 2], "expected": [[(1, 2), (3, 4)], [(5, 6)]]}
    ],
    "score_keywords": [
        {"args": ["red red blue", {"red": 2, "blue": 3, "green": 5}], "expected": 5}
    ],
    "score_casefold_keywords": [{"args": ["Red BLUE", {"red": 2, "blue": 3}], "expected": 5}],
    "score_tag_overlap": [{"args": [["p1", "p1", "p2"], {"p1": 2, "p2": 3}], "expected": 5}],
}


def get_repo_root() -> Path:
    """Resolve the repository root, honoring the usual test override."""
    override = os.environ.get("CARNOT_REPO_ROOT")
    if override:
        return Path(override).resolve()
    return Path(__file__).resolve().parents[1]


def safe_slug(text: str) -> str:
    """Convert a label into a filesystem-safe slug."""
    cleaned = text.strip().lower().replace("/", "_").replace(" ", "_")
    return "".join(char if char.isalnum() or char in {"_", "-"} else "_" for char in cleaned)


def default_output_path(benchmark: str) -> Path:
    """Return the default artifact path for a benchmark."""
    return get_repo_root() / "results" / f"experiment_218_{benchmark}_results.json"


def default_checkpoint_dir() -> Path:
    """Return the default checkpoint directory."""
    return get_repo_root() / "results" / "checkpoints" / "experiment_218"


def artifact_experiment_id(output_path: Path) -> int:
    """Infer the deliverable experiment id from the output filename when present."""
    match = _EXPERIMENT_OUTPUT_RE.search(output_path.name)
    if match is None:
        return EXPERIMENT
    return int(match.group(1))


def live_inference_mode() -> str:
    """Describe the current run's intended inference mode from env defaults."""
    if os.environ.get("CARNOT_FORCE_LIVE") == "1":
        return "live_cpu" if os.environ.get("CARNOT_FORCE_CPU") == "1" else "live_gpu"
    return "simulated"


def _token_count(tokenizer: Any, text: str) -> int:
    """Count tokens conservatively, falling back to whitespace if needed."""
    if not text:
        return 0
    if tokenizer is None:
        return len(text.split())
    try:
        encoded = tokenizer(
            text,
            add_special_tokens=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        input_ids = encoded.get("input_ids")
        if isinstance(input_ids, list):
            return len(input_ids)
    except Exception:
        pass
    return len(text.split())


def _round_mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return round(sum(values) / len(values), 3)


def _typed_reasoning_parse_status(typed_reasoning: Any) -> str:
    """Return a stable parse-status label for typed reasoning artifacts."""
    if typed_reasoning is None:
        return "unavailable"
    provenance = getattr(typed_reasoning, "provenance", None)
    extraction_method = getattr(provenance, "extraction_method", None)
    if isinstance(extraction_method, str) and extraction_method:
        return extraction_method
    if hasattr(typed_reasoning, "to_dict"):
        payload = typed_reasoning.to_dict()
        provenance_payload = payload.get("provenance", {})
        if isinstance(provenance_payload, dict):
            method = provenance_payload.get("extraction_method")
            if isinstance(method, str) and method:
                return method
    return "parsed"


def _serialize_jsonable(value: Any) -> Any:
    """Best-effort conversion into JSON-friendly structures."""
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(key): _serialize_jsonable(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_serialize_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_serialize_jsonable(item) for item in value]
    if hasattr(value, "to_dict"):
        return _serialize_jsonable(value.to_dict())
    if is_dataclass(value) and not isinstance(value, type):
        return _serialize_jsonable(asdict(value))
    return str(value)


def _serialize_constraint_result(constraint: Any) -> dict[str, Any]:
    """Serialize one pipeline constraint result without leaking opaque objects."""
    if isinstance(constraint, dict):
        return dict(constraint)
    return {
        "constraint_type": getattr(constraint, "constraint_type", None),
        "description": getattr(constraint, "description", ""),
        "metadata": _serialize_jsonable(getattr(constraint, "metadata", {})),
    }


def _serialize_verification_result(verification: Any) -> dict[str, Any]:
    """Serialize verification output for later trace-learning and audit work."""
    typed_reasoning = getattr(verification, "typed_reasoning", None)
    semantic_grounding = getattr(verification, "semantic_grounding", None)
    constraints = [
        _serialize_constraint_result(constraint)
        for constraint in list(getattr(verification, "constraints", []))
    ]
    violations = [
        _serialize_constraint_result(violation)
        for violation in list(getattr(verification, "violations", []))
    ]
    return {
        "verified": bool(getattr(verification, "verified", False)),
        "energy": round(float(getattr(verification, "energy", 0.0)), 6),
        "certificate": _serialize_jsonable(getattr(verification, "certificate", {})),
        "constraints": constraints,
        "violations": violations,
        "n_constraints": len(constraints),
        "n_violations": len(violations),
        "typed_reasoning_parse_status": _typed_reasoning_parse_status(typed_reasoning),
        "typed_reasoning": _serialize_jsonable(typed_reasoning),
        "semantic_grounding": _serialize_jsonable(semantic_grounding),
    }


def _serialize_generation_attempt(
    *,
    prompt: str,
    response: str,
    tokenizer: Any,
    valid: bool | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    """Serialize one generation attempt with prompt/response token counts."""
    prompt_tokens = _token_count(tokenizer, prompt)
    response_tokens = _token_count(tokenizer, response)
    payload = {
        "prompt": prompt,
        "response": response,
        "prompt_tokens": prompt_tokens,
        "response_tokens": response_tokens,
        "total_tokens": prompt_tokens + response_tokens,
    }
    if valid is not None:
        payload["valid"] = valid
    if error is not None:
        payload["error"] = error
    return payload


def _build_generation_trace(
    *,
    tokenizer: Any,
    attempts: list[dict[str, Any]],
    fallback_record: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Collect prompt/response attempts and aggregate token usage."""
    serialized_attempts = list(attempts)
    if fallback_record:
        serialized_attempts.append(
            _serialize_generation_attempt(
                prompt=str(fallback_record.get("prompt", "")),
                response=str(fallback_record.get("response", "")),
                tokenizer=tokenizer,
            )
        )
    prompt_tokens = sum(int(attempt.get("prompt_tokens", 0)) for attempt in serialized_attempts)
    response_tokens = sum(int(attempt.get("response_tokens", 0)) for attempt in serialized_attempts)
    return {
        "attempts": serialized_attempts,
        "prompt_tokens": prompt_tokens,
        "response_tokens": response_tokens,
        "total_tokens": prompt_tokens + response_tokens,
        "fallback_used": bool(fallback_record),
    }


def load_monitorability_policy(path: Path | None = None) -> dict[str, Any]:
    """Load the Exp 213 monitorability policy when it is present."""
    policy_path = path or (get_repo_root() / "results" / "monitorability_policy_213.json")
    if not policy_path.exists():
        return {}
    payload = json.loads(policy_path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def recommended_response_mode(
    task_slice: str,
    policy: dict[str, Any] | None = None,
) -> str:
    """Return the policy-recommended response mode for a task slice."""
    payload = policy if policy is not None else load_monitorability_policy()
    per_task_slice = payload.get("per_task_slice", {})
    if isinstance(per_task_slice, dict):
        entry = per_task_slice.get(task_slice, {})
        if isinstance(entry, dict):
            mode = entry.get("recommended_mode")
            if isinstance(mode, str):
                return mode
    return "answer_only_terse"


def _constraint_ir_task_slice(record: dict[str, Any]) -> str:
    source_family = str(record.get("source_family", ""))
    constraint_types = {str(item) for item in list(record.get("constraint_types", []))}
    if source_family == "live_gsm8k_semantic_failure":
        return "live_gsm8k_semantic_failure"
    if source_family == "code_typed_properties":
        return "code_typed_properties"
    if "semantic_grounding" in constraint_types:
        return "instruction_grounded"
    return "instruction_surface_only"


def _normalize_surface(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def _parse_json_payload(raw_response: str) -> dict[str, Any] | None:
    candidate = raw_response.strip()
    if candidate.startswith("```") and candidate.endswith("```"):
        lines = candidate.splitlines()
        if len(lines) >= 2:
            candidate = "\n".join(lines[1:-1]).strip()
    for text in (candidate,):
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            payload = None
        if isinstance(payload, dict):
            return payload
    start = candidate.find("{")
    end = candidate.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        payload = json.loads(candidate[start : end + 1])
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _unwrap_answer_candidate(raw_response: str) -> tuple[Any, list[Any]]:
    payload = _parse_json_payload(raw_response)
    if payload is None:
        return raw_response.strip(), []
    checks = payload.get("checks", [])
    structured_checks = checks if isinstance(checks, list) else []
    if "final_answer" in payload:
        return payload.get("final_answer"), structured_checks
    return payload, structured_checks


def _stringify_answer_value(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=True)
    return str(value).strip()


def _parse_number_answer(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    text = _stringify_answer_value(value)
    matches = re.findall(r"-?\d+(?:\.\d+)?", text)
    if not matches:
        return None
    return int(float(matches[-1]))


def _parse_bullets(value: Any) -> list[str] | None:
    if isinstance(value, list):
        bullets = [str(item).strip() for item in value if str(item).strip()]
        return bullets or None
    text = _stringify_answer_value(value)
    bullets = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("- ") or stripped.startswith("* "):
            bullets.append(stripped[2:].strip())
    return bullets or None


def _parse_comma_items(value: Any) -> list[str] | None:
    if isinstance(value, list):
        items = [str(item).strip() for item in value if str(item).strip()]
        return items or None
    text = _stringify_answer_value(value)
    if text.startswith("[") and text.endswith("]"):
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            payload = None
        if isinstance(payload, list):
            items = [str(item).strip() for item in payload if str(item).strip()]
            return items or None
    items = [item.strip() for item in text.split(",") if item.strip()]
    return items or None


def _parse_flat_yaml_object(value: Any) -> dict[str, Any] | None:
    if isinstance(value, dict):
        return {str(key): val for key, val in value.items()}
    text = _stringify_answer_value(value)
    if not text:
        return None
    payload: dict[str, Any] = {}
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if ":" not in stripped:
            return None
        key, raw_value = stripped.split(":", 1)
        key = key.strip()
        raw_value = raw_value.strip().strip("'\"")
        if not key:
            return None
        if re.fullmatch(r"-?\d+", raw_value):
            payload[key] = int(raw_value)
        elif raw_value.lower() in {"true", "false"}:
            payload[key] = raw_value.lower() == "true"
        else:
            payload[key] = raw_value
    return payload or None


def _parse_markdown_sections(text: str) -> list[tuple[str, str]]:
    sections: list[tuple[str, str]] = []
    current_heading: str | None = None
    body_lines: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        heading_match = re.match(
            r"^(?:#{1,6}\s*)?([A-Za-z][A-Za-z0-9 ]*[A-Za-z0-9])\s*:?\s*$",
            stripped,
        )
        if heading_match and len(heading_match.group(1).split()) <= 3:
            if current_heading is not None:
                sections.append((current_heading, " ".join(body_lines).strip()))
            current_heading = heading_match.group(1)
            body_lines = []
            continue
        if current_heading is not None and stripped:
            body_lines.append(stripped)
    if current_heading is not None:
        sections.append((current_heading, " ".join(body_lines).strip()))
    return sections


def _split_sentences(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+|\n+", text) if part.strip()]


def _parse_numbered_steps(text: str) -> list[str]:
    steps: list[str] = []
    for line in text.splitlines():
        match = re.match(r"^\s*\d+\.\s+(.*\S)\s*$", line)
        if match:
            steps.append(match.group(1).strip())
    return steps


def _extract_python_code(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    if "```python" in text:
        fragment = text.split("```python", 1)[1]
        return fragment.split("```", 1)[0].strip()
    if text.startswith("```") and text.endswith("```"):
        lines = text.splitlines()
        if len(lines) >= 2:
            return "\n".join(lines[1:-1]).strip()
    if "def " in text:
        return text[text.find("def ") :].strip()
    return None


def _function_signature_from_ast(node: ast.FunctionDef) -> str:
    args = []
    for arg in node.args.args:
        if arg.annotation is None:
            args.append(arg.arg)
        else:
            args.append(f"{arg.arg}: {ast.unparse(arg.annotation)}")
    signature = f"{node.name}({', '.join(args)})"
    if node.returns is not None:
        signature += f" -> {ast.unparse(node.returns)}"
    return signature


def _constraint_family(case: dict[str, Any], constraint: dict[str, Any]) -> str:
    constraint_type = str(constraint.get("type", ""))
    if constraint_type in _CONSTRAINT_IR_LITERAL_TYPES:
        return "literal"
    if (
        "semantic_grounding" in {str(item) for item in list(case.get("constraint_types", []))}
        and case.get("expected_answer_schema", {}).get("type") != "python_function"
    ):
        return "semantic"
    return "search_optimization_limited"


def _ast_safe_eval(node: ast.AST, variables: dict[str, float]) -> float:
    if isinstance(node, ast.Expression):
        return _ast_safe_eval(node.body, variables)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.Name):
        if node.id not in variables:
            raise ValueError(node.id)
        return float(variables[node.id])
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -_ast_safe_eval(node.operand, variables)
    if isinstance(node, ast.BinOp):
        left = _ast_safe_eval(node.left, variables)
        right = _ast_safe_eval(node.right, variables)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            return left / right
    raise ValueError("unsupported-expression")


def _resolved_constraint_values(case: dict[str, Any]) -> dict[str, float]:
    resolved: dict[str, float] = {}
    pending = [dict(item) for item in list(case.get("gold_atomic_constraints", []))]
    while pending:
        progressed = False
        next_pending: list[dict[str, Any]] = []
        for constraint in pending:
            target = str(constraint.get("target", ""))
            value = constraint.get("value")
            try:
                if isinstance(value, (int, float)):
                    resolved[target] = float(value)
                    progressed = True
                    continue
                if isinstance(value, str):
                    parsed = ast.parse(value, mode="eval")
                    resolved[target] = _ast_safe_eval(parsed, resolved)
                    progressed = True
                    continue
            except Exception:
                next_pending.append(constraint)
                continue
            next_pending.append(constraint)
        if not progressed:
            break
        pending = next_pending
    return resolved


def _looks_calm_professional(text: str) -> bool:
    lowered = text.lower()
    return not any(marker in lowered for marker in _TONE_FORBIDDEN_MARKERS)


def _topological_order_valid(result: Any, probe: dict[str, Any]) -> bool:
    if not isinstance(result, list):
        return False
    nodes = [str(item) for item in list(probe.get("nodes", []))]
    if set(result) != set(nodes) or len(result) != len(nodes):
        return False
    positions = {str(node): index for index, node in enumerate(result)}
    for edge in list(probe.get("edges", [])):
        if positions[str(edge[0])] >= positions[str(edge[1])]:
            return False
    return True


class _ConstraintIRProbeTimeoutError(RuntimeError):
    """Raised when a prompt-derived code probe exceeds the bounded budget."""


def _run_with_timeout(callback: Any, *, timeout_seconds: float) -> Any:
    if timeout_seconds <= 0 or threading.current_thread() is not threading.main_thread():
        return callback()

    def _raise_timeout(_signum: int, _frame: Any) -> None:
        raise _ConstraintIRProbeTimeoutError("constraint-ir code probe timed out")

    previous_handler = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, _raise_timeout)
    signal.setitimer(signal.ITIMER_REAL, timeout_seconds)
    try:
        return callback()
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous_handler)


def _run_code_probe(function_obj: Any, probe: dict[str, Any]) -> tuple[bool, dict[str, Any] | None]:
    args = copy.deepcopy(list(probe.get("args", [])))
    immutable_index = (
        probe.get("immutable_arg_index")
        if isinstance(probe.get("immutable_arg_index"), int)
        else None
    )
    frozen_arg = None
    if immutable_index is not None and immutable_index < len(args):
        frozen_arg = copy.deepcopy(args[immutable_index])
    try:
        result = _run_with_timeout(
            lambda: function_obj(*args),
            timeout_seconds=_CONSTRAINT_IR_CODE_PROBE_TIMEOUT_SECONDS,
        )
    except _ConstraintIRProbeTimeoutError:
        return False, {
            "failure_mode": "timeout",
            "stage": "probe_call",
            "timeout_seconds": _CONSTRAINT_IR_CODE_PROBE_TIMEOUT_SECONDS,
        }
    except Exception as exc:
        expected_exception = probe.get("expect_exception")
        if expected_exception is not None and exc.__class__.__name__ == expected_exception:
            return True, None
        return False, {
            "failure_mode": "exception",
            "stage": "probe_call",
            "exception_type": exc.__class__.__name__,
        }
    if probe.get("expect_exception") is not None:
        return False, {"failure_mode": "missing_expected_exception", "stage": "probe_call"}
    if frozen_arg is not None:
        assert immutable_index is not None
        if args[immutable_index] != frozen_arg:
            return False, {"failure_mode": "mutated_input", "stage": "probe_call"}
    validator = str(probe.get("validator", "equals"))
    if validator == "topological_order":
        if _topological_order_valid(result, probe):
            return True, None
        return False, {"failure_mode": "wrong_result", "stage": "probe_call"}
    if result == probe.get("expected"):
        return True, None
    return False, {"failure_mode": "wrong_result", "stage": "probe_call"}


def _code_probe_success(function_name: str, function_obj: Any) -> tuple[bool, dict[str, Any]]:
    probes = _CODE_PROBES.get(function_name, [])
    if not probes:
        return False, {"failure_mode": "missing_probe", "stage": "probe_setup"}
    for probe in probes:
        passed, details = _run_code_probe(function_obj, probe)
        if not passed:
            return False, details or {"failure_mode": "unknown", "stage": "probe_call"}
    return True, {}


def _classify_constraint_ir_output_style(
    raw_response: str,
    *,
    structured_payload: dict[str, Any] | None,
    schema_type: str,
    code_text: str | None,
) -> str:
    if structured_payload is not None:
        return "structured_json"
    if schema_type == "python_function" and code_text is not None:
        stripped = raw_response.strip()
        if stripped.startswith("def ") or stripped.startswith("```"):
            return "code_only"
    if "REASONING:" in raw_response or "FINAL:" in raw_response:
        return "free_form_reasoning"
    if "\n" in raw_response and schema_type not in {
        "bullet_list",
        "markdown_sections",
        "numbered_list",
        "python_function",
    }:
        return "free_form_reasoning"
    if raw_response.strip():
        return "answer_only_terse"
    return "other_unstructured"


def _build_constraint_ir_context(
    case: dict[str, Any],
    response_mode: str,
    raw_response: str,
) -> dict[str, Any]:
    del response_mode
    schema_type = str(case["expected_answer_schema"]["type"])
    structured_payload = _parse_json_payload(raw_response)
    answer_candidate, checks = _unwrap_answer_candidate(raw_response)
    answer_text = _stringify_answer_value(answer_candidate)
    json_answer = (
        answer_candidate if isinstance(answer_candidate, dict) else _parse_json_payload(answer_text)
    )
    yaml_answer = _parse_flat_yaml_object(
        answer_candidate if schema_type == "yaml_object" else answer_text
    )
    bullet_answer = _parse_bullets(
        answer_candidate if schema_type == "bullet_list" else answer_text
    )
    comma_items = _parse_comma_items(
        answer_candidate if schema_type in {"comma_separated_list", "identifier"} else answer_text
    )
    code_text = _extract_python_code(
        answer_text if schema_type == "python_function" else raw_response
    )
    code_tree = None
    function_node = None
    function_obj = None
    function_name = ""
    function_signature = ""
    code_probe_details: dict[str, Any] = {}
    if code_text is not None:
        try:
            code_tree = ast.parse(code_text)
        except SyntaxError:
            code_tree = None
        if code_tree is not None:
            expected_name = str(case["expected_answer_schema"].get("name", ""))
            functions = [node for node in code_tree.body if isinstance(node, ast.FunctionDef)]
            function_node = next((node for node in functions if node.name == expected_name), None)
            if function_node is not None:
                function_name = function_node.name
                function_signature = _function_signature_from_ast(function_node)
                namespace: dict[str, Any] = {}
                try:
                    _run_with_timeout(
                        lambda: exec(code_text, namespace),
                        timeout_seconds=_CONSTRAINT_IR_CODE_PROBE_TIMEOUT_SECONDS,
                    )
                    function_obj = namespace.get(expected_name)
                except _ConstraintIRProbeTimeoutError:
                    code_probe_details = {
                        "failure_mode": "timeout",
                        "stage": "exec",
                        "timeout_seconds": _CONSTRAINT_IR_CODE_PROBE_TIMEOUT_SECONDS,
                    }
                except Exception:
                    function_obj = None
                    code_probe_details = {
                        "failure_mode": "exception",
                        "stage": "exec",
                    }
    code_probes_pass = False
    if function_obj is not None:
        code_probes_pass, probe_details = _code_probe_success(function_name, function_obj)
        if probe_details:
            code_probe_details = probe_details
    return {
        "schema_type": schema_type,
        "raw_response": raw_response,
        "structured_payload": structured_payload,
        "answer_candidate": answer_candidate,
        "answer_text": answer_text,
        "normalized_answer": _normalize_surface(answer_text),
        "normalized_response": _normalize_surface(
            raw_response + (" " + json.dumps(checks, ensure_ascii=True) if checks else "")
        ),
        "checks": checks,
        "json_answer": json_answer,
        "yaml_answer": yaml_answer,
        "bullet_answer": bullet_answer,
        "comma_items": comma_items,
        "sections": _parse_markdown_sections(answer_text or raw_response),
        "sentences": _split_sentences(answer_text or raw_response),
        "numbered_steps": _parse_numbered_steps(answer_text or raw_response),
        "identifier": (
            (comma_items[0] if comma_items and schema_type == "identifier" else None)
            or next(iter(re.findall(r"[A-Za-z][A-Za-z0-9_-]*", answer_text or raw_response)), None)
        ),
        "parsed_number": _parse_number_answer(
            answer_candidate if schema_type == "number" else answer_text
        ),
        "code_text": code_text,
        "function_node": function_node,
        "function_obj": function_obj,
        "function_name": function_name,
        "function_signature": function_signature,
        "code_probes_pass": code_probes_pass,
        "code_probe_details": code_probe_details,
        "resolved_values": _resolved_constraint_values(case),
        "output_style": _classify_constraint_ir_output_style(
            raw_response,
            structured_payload=structured_payload,
            schema_type=schema_type,
            code_text=code_text,
        ),
    }


def _constraint_result(
    case: dict[str, Any],
    constraint: dict[str, Any],
    *,
    status: str,
    judge: str,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "constraint_id": str(constraint.get("constraint_id", "")),
        "type": str(constraint.get("type", "")),
        "family": _constraint_family(case, constraint),
        "status": status,
        "judge": judge,
        "details": details or {},
    }


def _evaluate_live_prompt_constraint(
    case: dict[str, Any],
    constraint: dict[str, Any],
    context: dict[str, Any],
) -> dict[str, Any]:
    normalized = str(context["normalized_response"])
    resolved_values = dict(context["resolved_values"])
    expected_value = resolved_values.get(str(constraint.get("target", "")))
    fragments: list[str] = []
    if expected_value is not None:
        numeric_text = (
            str(int(expected_value)) if float(expected_value).is_integer() else str(expected_value)
        )
        fragments.append(_normalize_surface(numeric_text))
    raw_value = constraint.get("value")
    if isinstance(raw_value, (int, float)):
        numeric_text = str(int(raw_value)) if float(raw_value).is_integer() else str(raw_value)
        fragments.append(_normalize_surface(numeric_text))
    elif isinstance(raw_value, str):
        fragments.append(_normalize_surface(raw_value.replace("_", " ")))
    fragments.append(_normalize_surface(str(constraint.get("target", "")).replace("_", " ")))
    if constraint.get("unit") is not None:
        fragments.append(_normalize_surface(str(constraint["unit"])))
    compact = [fragment for fragment in fragments if fragment]
    if str(constraint.get("type")) == "final_answer_binding" and expected_value is not None:
        parsed_number = context.get("parsed_number")
        if parsed_number is None:
            return _constraint_result(case, constraint, status="violated", judge="heuristic_rule")
        return _constraint_result(
            case,
            constraint,
            status="satisfied" if float(parsed_number) == float(expected_value) else "violated",
            judge="heuristic_rule",
            details={"expected": expected_value, "observed": parsed_number},
        )
    if any(fragment and fragment in normalized for fragment in compact):
        return _constraint_result(case, constraint, status="satisfied", judge="heuristic_rule")
    return _constraint_result(case, constraint, status="unjudged", judge="heuristic_rule")


def _evaluate_constraint_ir_constraint(
    case: dict[str, Any],
    constraint: dict[str, Any],
    context: dict[str, Any],
) -> dict[str, Any]:
    constraint_type = str(constraint.get("type", ""))
    schema_type = str(context["schema_type"])
    response_text = str(context["answer_text"] or context["raw_response"])
    normalized_response = str(context["normalized_answer"] or context["normalized_response"])

    if schema_type == "python_function":
        function_node = context.get("function_node")
        function_obj = context.get("function_obj")
        if constraint_type == "function_name":
            return _constraint_result(
                case,
                constraint,
                status=(
                    "satisfied"
                    if function_node is not None
                    and context["function_name"] == constraint.get("value")
                    else "violated"
                ),
                judge="deterministic",
            )
        if constraint_type == "signature":
            return _constraint_result(
                case,
                constraint,
                status=(
                    "satisfied"
                    if function_node is not None
                    and context["function_signature"] == str(constraint.get("value"))
                    else "violated"
                ),
                judge="deterministic",
            )
        if constraint_type == "return_type":
            observed = (
                ast.unparse(function_node.returns)
                if function_node is not None and function_node.returns is not None
                else None
            )
            return _constraint_result(
                case,
                constraint,
                status="satisfied" if observed == str(constraint.get("value")) else "violated",
                judge="deterministic",
                details={"observed": observed},
            )
        if constraint_type == "forbidden_api":
            code_text = str(context.get("code_text") or "")
            forbidden_values = constraint.get("value")
            forbidden_list = (
                forbidden_values if isinstance(forbidden_values, list) else [forbidden_values]
            )
            satisfied = all(str(item) not in code_text for item in forbidden_list)
            return _constraint_result(
                case,
                constraint,
                status="satisfied" if satisfied else "violated",
                judge="deterministic",
            )
        if constraint_type == "time_complexity":
            code_text = str(context.get("code_text") or "")
            heuristic_ok = (
                bool(context.get("code_probes_pass"))
                and "sorted(" not in code_text
                and ".sort(" not in code_text
            )
            return _constraint_result(
                case,
                constraint,
                status="satisfied" if heuristic_ok else "violated",
                judge="heuristic_rule",
                details=dict(context.get("code_probe_details") or {}),
            )
        if function_obj is None:
            return _constraint_result(
                case,
                constraint,
                status="violated",
                judge="deterministic",
                details=dict(context.get("code_probe_details") or {}),
            )
        return _constraint_result(
            case,
            constraint,
            status="satisfied" if bool(context.get("code_probes_pass")) else "violated",
            judge="deterministic",
            details=dict(context.get("code_probe_details") or {}),
        )

    if constraint_type == "count_exact":
        expected = int(constraint.get("value", 0))
        if schema_type == "bullet_list" and context["bullet_answer"] is not None:
            observed = len(context["bullet_answer"])
        elif (
            schema_type in {"comma_separated_list", "identifier"}
            and context["comma_items"] is not None
        ):
            observed = len(context["comma_items"])
        else:
            observed = None
        if observed is None:
            return _constraint_result(case, constraint, status="violated", judge="deterministic")
        return _constraint_result(
            case,
            constraint,
            status="satisfied" if observed == expected else "violated",
            judge="deterministic",
            details={"observed": observed},
        )
    if constraint_type == "word_count_range":
        bounds = list(constraint.get("value", [0, 0]))
        low, high = int(bounds[0]), int(bounds[1])
        if (
            str(constraint.get("target")) == "bullet_word_count"
            and context["bullet_answer"] is not None
        ):
            satisfied = all(low <= len(item.split()) <= high for item in context["bullet_answer"])
            observed = [len(item.split()) for item in context["bullet_answer"]]
        else:
            observed = len(response_text.split())
            satisfied = low <= observed <= high
        return _constraint_result(
            case,
            constraint,
            status="satisfied" if satisfied else "violated",
            judge="deterministic",
            details={"observed": observed},
        )
    if constraint_type in {"must_include_token", "must_include_phrase"}:
        required = _normalize_surface(str(constraint.get("value", "")))
        if str(constraint.get("target")) == "sentence_1":
            haystack = _normalize_surface(context["sentences"][0]) if context["sentences"] else ""
        else:
            section_target = _normalize_surface(str(constraint.get("target", "")))
            section_map = {
                _normalize_surface(name): _normalize_surface(body)
                for name, body in list(context["sections"])
            }
            haystack = section_map.get(section_target, normalized_response)
        return _constraint_result(
            case,
            constraint,
            status="satisfied" if required and required in haystack else "violated",
            judge="deterministic",
        )
    if constraint_type in {"forbidden_token", "forbidden_phrase"}:
        forbidden = _normalize_surface(str(constraint.get("value", "")))
        return _constraint_result(
            case,
            constraint,
            status="satisfied" if forbidden not in normalized_response else "violated",
            judge="deterministic",
        )
    if constraint_type == "json_exact_keys":
        keys = (
            list(context["json_answer"].keys())
            if isinstance(context["json_answer"], dict)
            else None
        )
        expected = list(constraint.get("value", []))
        return _constraint_result(
            case,
            constraint,
            status="satisfied" if keys == expected else "violated",
            judge="deterministic",
            details={"observed": keys},
        )
    if constraint_type == "no_extra_keys":
        keys = (
            set(context["json_answer"].keys()) if isinstance(context["json_answer"], dict) else None
        )
        expected = set(constraint.get("value", []))
        satisfied = keys is not None and keys.issubset(expected)
        return _constraint_result(
            case, constraint, status="satisfied" if satisfied else "violated", judge="deterministic"
        )
    if constraint_type == "enum_membership":
        payload = context["json_answer"]
        field_value = (
            payload.get(str(constraint.get("target"))) if isinstance(payload, dict) else None
        )
        allowed = {str(item) for item in list(constraint.get("value", []))}
        return _constraint_result(
            case,
            constraint,
            status="satisfied" if str(field_value) in allowed else "violated",
            judge="deterministic",
            details={"observed": field_value},
        )
    if constraint_type == "section_order":
        observed = [name for name, _body in list(context["sections"])]
        expected = [str(item) for item in list(constraint.get("value", []))]
        return _constraint_result(
            case,
            constraint,
            status="satisfied" if observed == expected else "violated",
            judge="deterministic",
            details={"observed": observed},
        )
    if constraint_type == "sentence_count_per_section":
        expected = int(constraint.get("value", 0))
        satisfied = bool(context["sections"]) and all(
            len(_split_sentences(body)) == expected for _name, body in list(context["sections"])
        )
        return _constraint_result(
            case, constraint, status="satisfied" if satisfied else "violated", judge="deterministic"
        )
    if constraint_type == "grounded_selection":
        expected = constraint.get("value")
        if isinstance(expected, list):
            observed = context["comma_items"]
            satisfied = observed == expected
        elif (
            isinstance(context["json_answer"], dict)
            and str(constraint.get("target")) in context["json_answer"]
        ):
            observed = context["json_answer"].get(str(constraint.get("target")))
            satisfied = observed == expected
        else:
            observed = (
                context["identifier"]
                if schema_type == "identifier"
                else (context["comma_items"][0] if context["comma_items"] else None)
            )
            satisfied = observed == expected
        return _constraint_result(
            case,
            constraint,
            status="satisfied" if satisfied else "violated",
            judge="deterministic",
            details={"observed": observed},
        )
    if constraint_type == "grounded_evidence_ids":
        payload = context["json_answer"]
        evidence = payload.get("evidence") if isinstance(payload, dict) else None
        expected = [str(item) for item in list(constraint.get("value", []))]
        if isinstance(evidence, list):
            observed = [str(item) for item in evidence]
            satisfied = observed == expected
        else:
            observed = str(evidence)
            normalized = _normalize_surface(observed)
            satisfied = all(_normalize_surface(item) in normalized for item in expected)
        return _constraint_result(
            case,
            constraint,
            status="satisfied" if satisfied else "violated",
            judge="deterministic",
            details={"observed": observed},
        )
    if constraint_type == "ordering":
        expected = next(
            (
                list(item.get("value", []))
                for item in list(case.get("gold_atomic_constraints", []))
                if str(item.get("type")) == "grounded_selection"
                and isinstance(item.get("value"), list)
            ),
            None,
        )
        observed = context["comma_items"]
        satisfied = observed is not None and expected is not None and observed == expected
        return _constraint_result(
            case,
            constraint,
            status="satisfied" if satisfied else "violated",
            judge="deterministic",
            details={"observed": observed},
        )
    if constraint_type == "step_count":
        observed = len(context["numbered_steps"])
        expected = int(constraint.get("value", 0))
        return _constraint_result(
            case,
            constraint,
            status="satisfied" if observed == expected else "violated",
            judge="deterministic",
            details={"observed": observed},
        )
    if constraint_type == "step_roles":
        steps = list(context["numbered_steps"])
        roles = [str(item) for item in list(constraint.get("value", []))]
        if len(steps) != len(roles):
            return _constraint_result(case, constraint, status="violated", judge="heuristic_rule")
        satisfied = True
        for step, role in zip(steps, roles, strict=True):
            lowered = step.lower()
            for keyword in _PLAN_ROLE_KEYWORDS.get(role, (role.replace("_", " "),)):
                if keyword not in lowered:
                    satisfied = False
                    break
            if not satisfied:
                break
        return _constraint_result(
            case,
            constraint,
            status="satisfied" if satisfied else "violated",
            judge="heuristic_rule",
        )
    if constraint_type == "yaml_exact_keys":
        keys = (
            list(context["yaml_answer"].keys())
            if isinstance(context["yaml_answer"], dict)
            else None
        )
        expected = list(constraint.get("value", []))
        return _constraint_result(
            case,
            constraint,
            status="satisfied" if keys == expected else "violated",
            judge="deterministic",
            details={"observed": keys},
        )
    if constraint_type == "derived_value":
        payload = (
            context["yaml_answer"]
            if isinstance(context["yaml_answer"], dict)
            else context["json_answer"]
        )
        observed = payload.get(str(constraint.get("target"))) if isinstance(payload, dict) else None
        expected = constraint.get("value")
        return _constraint_result(
            case,
            constraint,
            status="satisfied" if observed == expected else "violated",
            judge="deterministic",
            details={"observed": observed},
        )
    if constraint_type == "negation_scope":
        expected = next(
            (
                list(item.get("value", []))
                for item in list(case.get("gold_atomic_constraints", []))
                if str(item.get("type")) == "grounded_selection"
            ),
            None,
        )
        observed = context["comma_items"]
        return _constraint_result(
            case,
            constraint,
            status="satisfied" if observed == expected else "violated",
            judge="deterministic",
            details={"observed": observed},
        )
    if constraint_type == "sentence_count":
        observed = len(context["sentences"])
        expected = int(constraint.get("value", 0))
        return _constraint_result(
            case,
            constraint,
            status="satisfied" if observed == expected else "violated",
            judge="deterministic",
            details={"observed": observed},
        )
    if constraint_type == "tone":
        return _constraint_result(
            case,
            constraint,
            status="satisfied" if _looks_calm_professional(response_text) else "violated",
            judge="heuristic_rule",
        )
    return _evaluate_live_prompt_constraint(case, constraint, context)


def sample_records(
    records: list[dict[str, Any]],
    sample_size: int,
    sample_seed: int,
) -> list[dict[str, Any]]:
    """Return a deterministic seeded shuffle followed by first-N selection."""
    shuffled = list(records)
    random.Random(sample_seed).shuffle(shuffled)
    return shuffled[:sample_size]


def shared_prompt_seed(sample_seed: int, case_id: str) -> int:
    """Derive one stable prompt seed for a case from the sample seed."""
    digest = hashlib.sha256(f"{sample_seed}:{case_id}".encode()).hexdigest()
    return int(digest[:8], 16)


def build_cohort_manifest(
    records: list[dict[str, Any]],
    sample_size: int,
    sample_seed: int,
) -> list[dict[str, Any]]:
    """Build the sampled cohort manifest with shared prompt seeds."""
    sampled = sample_records(records, sample_size=sample_size, sample_seed=sample_seed)
    cohort: list[dict[str, Any]] = []
    for sample_position, record in enumerate(sampled, start=1):
        enriched = dict(record)
        case_id = str(enriched["case_id"])
        prompt_seed = shared_prompt_seed(sample_seed, case_id)
        enriched["sample_position"] = sample_position
        enriched["prompt_seeds"] = {
            "baseline": prompt_seed,
            "verify_only": prompt_seed,
            "verify_repair": prompt_seed,
        }
        cohort.append(enriched)
    return cohort


def checkpoint_path(
    checkpoint_dir: Path,
    *,
    benchmark: str,
    model_name: str,
    mode: str,
) -> Path:
    """Return the per-benchmark/model/mode checkpoint path."""
    return checkpoint_dir / (
        f"{safe_slug(benchmark)}__{safe_slug(model_name)}__{safe_slug(mode)}.json"
    )


def load_checkpoint(path: Path, expected_case_ids: list[str]) -> dict[str, Any]:
    """Load a checkpoint when the cohort metadata still matches."""
    fresh = {
        "case_ids": list(expected_case_ids),
        "results_by_case": {},
    }
    if not path.exists():
        return fresh

    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("case_ids") != expected_case_ids:
        return fresh
    results_by_case = payload.get("results_by_case", {})
    if not isinstance(results_by_case, dict):
        return fresh
    return {
        **payload,
        "case_ids": list(expected_case_ids),
        "results_by_case": dict(results_by_case),
    }


def save_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    """Write a checkpoint atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    tmp_path.replace(path)


def run_mode(
    *,
    benchmark: str,
    model_name: str,
    mode: str,
    cases: list[dict[str, Any]],
    checkpoint_dir: Path,
    execute_case: Any,
) -> list[dict[str, Any]]:
    """Resume-aware execution helper for one benchmark/model/mode cell."""
    case_ids = [str(case["case_id"]) for case in cases]
    ckpt_path = checkpoint_path(
        checkpoint_dir,
        benchmark=benchmark,
        model_name=model_name,
        mode=mode,
    )
    checkpoint = load_checkpoint(ckpt_path, case_ids)
    results_by_case: dict[str, Any] = dict(checkpoint["results_by_case"])

    for case in cases:
        case_id = str(case["case_id"])
        if case_id in results_by_case:
            continue
        result = dict(execute_case(case))
        result.setdefault("case_id", case_id)
        result.setdefault("mode", mode)
        results_by_case[case_id] = result
        save_checkpoint(
            ckpt_path,
            {
                "benchmark": benchmark,
                "model_name": model_name,
                "mode": mode,
                "case_ids": case_ids,
                "results_by_case": results_by_case,
            },
        )

    return [dict(results_by_case[case_id]) for case_id in case_ids]


def build_artifact_payload(
    *,
    benchmark: str,
    output_path: Path,
    cohort: list[dict[str, Any]],
    paired_runs: list[dict[str, Any]],
    statistics: dict[str, Any],
    sample_seed: int,
    sample_size: int,
    started_at: str,
    finished_at: str,
    runtime_seconds: float,
    checkpoint_dir: Path,
    max_repairs: int,
    policy_path: Path,
    inference_mode: str,
) -> dict[str, Any]:
    """Build the stable top-level Exp 218 artifact payload."""
    spec = BENCHMARK_SPECS[benchmark]
    return {
        "experiment": artifact_experiment_id(output_path),
        "benchmark": benchmark,
        "title": spec["title"],
        "run_date": RUN_DATE,
        "schema": {
            "artifact": SCHEMA_ARTIFACT,
            "benchmark_case_schema": spec["case_schema"],
        },
        "metadata": {
            "started_at": started_at,
            "finished_at": finished_at,
            "runtime_seconds": round(runtime_seconds, 3),
            "sample_seed": sample_seed,
            "sample_size": sample_size,
            "sample_strategy": "seeded_shuffle_first_n",
            "modes": list(MODE_ORDER),
            "models": [dict(model) for model in MODEL_SPECS],
            "source_artifacts": list(spec["source_artifacts"]),
            "output_path": str(output_path),
            "checkpoint_dir": str(checkpoint_dir),
            "checkpoint_pattern": (
                "results/checkpoints/experiment_218/<benchmark>__<model>__<mode>.json"
            ),
            "max_repairs": max_repairs,
            "policy_source": str(policy_path),
            "inference_mode": inference_mode,
            "force_live": os.environ.get("CARNOT_FORCE_LIVE") == "1",
            "force_cpu": os.environ.get("CARNOT_FORCE_CPU") == "1",
        },
        "cohort": {
            "case_count": len(cohort),
            "case_ids": [str(case["case_id"]) for case in cohort],
            "cases": [dict(case) for case in cohort],
        },
        "paired_runs": list(paired_runs),
        "statistics": dict(statistics),
    }


def write_artifact(path: Path, payload: dict[str, Any]) -> None:
    """Write an artifact with parent directory creation and a trailing newline."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")


def utc_now() -> str:
    """Return the current UTC timestamp in ISO-8601 format."""
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


class HarnessArgumentParser(argparse.ArgumentParser):
    """Parser that fills benchmark-specific defaults after parsing."""

    def parse_args(  # type: ignore[override]
        self,
        args: list[str] | None = None,
        namespace: argparse.Namespace | None = None,
    ) -> argparse.Namespace:
        parsed = super().parse_args(args=args, namespace=namespace)
        if parsed.sample_size is None:
            parsed.sample_size = BENCHMARK_SPECS[parsed.benchmark]["default_sample_size"]
        if parsed.output is None:
            parsed.output = default_output_path(parsed.benchmark)
        if parsed.checkpoint_dir is None:
            parsed.checkpoint_dir = default_checkpoint_dir()
        return parsed


def build_parser() -> argparse.ArgumentParser:
    """Build the unified Exp 218 CLI parser."""
    parser = HarnessArgumentParser(
        description="Checkpointed shared live benchmark harness for Exp 219-221.",
    )
    parser.add_argument(
        "--benchmark",
        choices=list(BENCHMARK_SPECS),
        required=True,
        help="Benchmark to run.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Override the benchmark-specific default sample size.",
    )
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=DEFAULT_SAMPLE_SEED,
        help="Seed for deterministic cohort sampling.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Artifact path. Defaults to results/experiment_218_<benchmark>_results.json.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=None,
        help="Directory for per-benchmark/model/mode checkpoints.",
    )
    parser.add_argument(
        "--max-repairs",
        type=int,
        default=DEFAULT_MAX_REPAIRS,
        help="Maximum verify-repair iterations per case.",
    )
    return parser


def _load_script_module(module_name: str, relative_path: str) -> Any:  # pragma: no cover
    module_path = get_repo_root() / relative_path
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load script module: {relative_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _seed_runtime(seed: int) -> None:  # pragma: no cover
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def _load_live_model(model_spec: dict[str, str]) -> tuple[Any, Any]:  # pragma: no cover
    os.environ.setdefault("CARNOT_FORCE_LIVE", "1")
    os.environ.setdefault("CARNOT_FORCE_CPU", "0")

    from carnot.inference.model_loader import load_model  # type: ignore[import-untyped]

    model, tokenizer = load_model(model_spec["hf_id"], device="cuda")
    if model is None or tokenizer is None:
        raise RuntimeError(f"Failed to load live model: {model_spec['hf_id']}")
    return model, tokenizer


def _unload_live_model(model: Any, tokenizer: Any) -> None:  # pragma: no cover
    del model, tokenizer
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _generate_text(
    *,
    model: Any,
    tokenizer: Any,
    prompt: str,
    prompt_seed: int,
    max_new_tokens: int,
) -> str:  # pragma: no cover
    _seed_runtime(prompt_seed)
    from carnot.inference.model_loader import generate

    return str(generate(model, tokenizer, prompt, max_new_tokens=max_new_tokens))


def _extract_final_number(text: str) -> int | None:  # pragma: no cover
    numeric_token = r"-?(?:\d[\d,]*)"

    match = re.search(rf"####\s*({numeric_token})", text)
    if match:
        return int(match.group(1).replace(",", ""))

    match = re.search(rf"[Aa]nswer[:\s]+({numeric_token})", text)
    if match:
        return int(match.group(1).replace(",", ""))

    numbers = re.findall(numeric_token, text)
    if numbers:
        return int(numbers[-1].replace(",", ""))
    return None


def _load_benchmark_records(benchmark: str) -> list[dict[str, Any]]:  # pragma: no cover
    if benchmark == "gsm8k_semantic":
        from datasets import load_dataset  # type: ignore[import-untyped]

        dataset = None
        for dataset_name in ("openai/gsm8k", "gsm8k"):
            try:
                dataset = load_dataset(dataset_name, "main", split="test")
                break
            except Exception:
                continue
        if dataset is None:
            raise RuntimeError("Could not load GSM8K test split.")

        records: list[dict[str, Any]] = []
        for dataset_idx in range(len(dataset)):
            row = dataset[dataset_idx]
            ground_truth = _extract_final_number(str(row["answer"]))
            if ground_truth is None:
                continue
            records.append(
                {
                    "case_id": f"gsm8k-{dataset_idx}",
                    "dataset_idx": dataset_idx,
                    "question": row["question"],
                    "ground_truth": ground_truth,
                    "task_slice": "live_gsm8k_semantic_failure",
                }
            )
        return records

    if benchmark == "humaneval_property":
        from datasets import load_dataset

        dataset = load_dataset("openai_humaneval", split="test")
        return [
            {
                "case_id": f"humaneval-{idx}",
                "dataset_idx": idx,
                "task_id": row["task_id"],
                "prompt": row["prompt"],
                "test": row["test"],
                "entry_point": row["entry_point"],
                "task_slice": "code_typed_properties",
            }
            for idx, row in enumerate(dataset)
        ]

    if benchmark == "constraint_ir":
        path = get_repo_root() / "data" / "research" / "constraint_ir_benchmark_211.jsonl"
        records = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        enriched: list[dict[str, Any]] = []
        for index, record in enumerate(records):
            row = dict(record)
            row["case_id"] = str(row.get("example_id", f"constraint-ir-{index}"))
            row["dataset_idx"] = index
            row["task_slice"] = _constraint_ir_task_slice(row)
            enriched.append(row)
        return enriched

    raise ValueError(f"Unsupported benchmark: {benchmark}")


def _build_constraint_ir_prompt(
    case: dict[str, Any],
    response_mode: str,
) -> tuple[str, int]:  # pragma: no cover
    module = _load_script_module(
        "experiment_213_monitorability_audit",
        "scripts/experiment_213_monitorability_audit.py",
    )
    return (
        str(module.build_mode_prompt(case, response_mode)),
        int(module.max_new_tokens_for(case, response_mode)),
    )


def _evaluate_constraint_ir_response(
    case: dict[str, Any],
    response_mode: str,
    response: str,
) -> dict[str, Any]:  # pragma: no cover
    context = _build_constraint_ir_context(case, response_mode, response)
    constraint_results = [
        _evaluate_constraint_ir_constraint(case, dict(constraint), context)
        for constraint in list(case.get("gold_atomic_constraints", []))
    ]
    total_constraints = len(constraint_results)
    judged_constraints = [item for item in constraint_results if item["status"] != "unjudged"]
    satisfied_constraints = [item for item in constraint_results if item["status"] == "satisfied"]
    failures_by_family = {
        "literal": 0,
        "semantic": 0,
        "search_optimization_limited": 0,
    }
    coverage_gaps_by_family = {
        "literal": 0,
        "semantic": 0,
        "search_optimization_limited": 0,
    }
    judging_summary = {
        "deterministic": 0,
        "heuristic_rule": 0,
        "model_assisted": 0,
    }
    for item in constraint_results:
        judge = str(item["judge"])
        if judge in judging_summary:
            judging_summary[judge] += 1
        family = str(item["family"])
        if item["status"] == "violated":
            failures_by_family[family] += 1
        elif item["status"] == "unjudged":
            coverage_gaps_by_family[family] += 1

    parseable = bool(judged_constraints) and context["answer_text"] != ""
    if str(context["schema_type"]) == "python_function":
        parseable = context["function_obj"] is not None
    elif str(context["schema_type"]) == "json_object":
        parseable = context["json_answer"] is not None
    elif str(context["schema_type"]) == "yaml_object":
        parseable = context["yaml_answer"] is not None
    elif str(context["schema_type"]) == "bullet_list":
        parseable = context["bullet_answer"] is not None
    elif str(context["schema_type"]) == "number":
        parseable = context["parsed_number"] is not None
    elif str(context["schema_type"]) == "numbered_list":
        parseable = bool(context["numbered_steps"])
    elif str(context["schema_type"]) == "markdown_sections":
        parseable = bool(context["sections"])
    elif str(context["schema_type"]) == "two_sentences":
        parseable = len(context["sentences"]) == 2
    elif str(context["schema_type"]) == "identifier":
        parseable = context["identifier"] is not None
    elif str(context["schema_type"]) == "comma_separated_list":
        parseable = context["comma_items"] is not None

    extraction_coverage = (
        round(len(judged_constraints) / total_constraints, 6) if total_constraints else 0.0
    )
    partial_satisfaction = (
        round(len(satisfied_constraints) / total_constraints, 6) if total_constraints else 0.0
    )
    exact_satisfaction = (
        parseable
        and total_constraints > 0
        and len(judged_constraints) == total_constraints
        and len(satisfied_constraints) == total_constraints
    )
    return {
        "parseable": parseable,
        "answer_quality": partial_satisfaction,
        "constraint_coverage": extraction_coverage,
        "constraint_extraction_coverage": extraction_coverage,
        "exact_satisfaction": exact_satisfaction,
        "partial_satisfaction": partial_satisfaction,
        "semantic_violation_count": failures_by_family["semantic"],
        "failure_breakdown": failures_by_family,
        "coverage_gap_breakdown": coverage_gaps_by_family,
        "constraint_results": constraint_results,
        "judging_summary": judging_summary,
        "output_style": str(context["output_style"]),
        "example_id": str(case.get("example_id", case["case_id"])),
        "task_slice": str(case["task_slice"]),
        "source_family": str(case["source_family"]),
        "mode": response_mode,
        "raw_response": response,
    }


def _run_gsm8k_baseline(
    case: dict[str, Any],
    *,
    model_spec: dict[str, str],
    model: Any,
    tokenizer: Any,
    policy: dict[str, Any],
) -> dict[str, Any]:  # pragma: no cover
    from carnot.pipeline.verify_repair import VerifyRepairPipeline  # type: ignore[import-untyped]

    typed_reasoning = None
    response_mode = recommended_response_mode(str(case["task_slice"]), policy)
    prompt_seed = int(case["prompt_seeds"]["baseline"])
    task = f"Question: {case['question']}\nSolve the problem and give the final answer as a number."
    prompt = (
        task + "\nReturn only the final numeric answer. No explanation."
        if response_mode != "structured_json"
        else task
    )
    started = time.perf_counter()
    if response_mode == "structured_json":
        from carnot.pipeline.structured_reasoning import (  # type: ignore[import-untyped]
            StructuredReasoningController,
        )

        fallback_record: dict[str, Any] = {}

        def fallback_generate(generated_prompt: str, max_new_tokens: int) -> str:
            response = _generate_text(
                model=model,
                tokenizer=tokenizer,
                prompt=generated_prompt,
                prompt_seed=prompt_seed,
                max_new_tokens=max_new_tokens,
            )
            fallback_record.update(
                {
                    "prompt": generated_prompt,
                    "response": response,
                }
            )
            return response

        controller = StructuredReasoningController(policy=policy)
        emission = controller.emit(
            question=task,
            task_slice=str(case["task_slice"]),
            model_name=model_spec["hf_id"],
            model=model,
            tokenizer=tokenizer,
            fallback_generate=fallback_generate,
        )
        response = emission.response
        typed_reasoning = emission.typed_reasoning
        effective_mode = emission.response_mode
        generation_trace = _build_generation_trace(
            tokenizer=tokenizer,
            attempts=[
                _serialize_generation_attempt(
                    prompt=str(attempt.prompt),
                    response=str(attempt.raw_response),
                    tokenizer=tokenizer,
                    valid=bool(attempt.valid),
                    error=attempt.error,
                )
                for attempt in emission.attempts
            ],
            fallback_record=fallback_record if emission.fallback_used else None,
        )
    else:
        response = _generate_text(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            prompt_seed=prompt_seed,
            max_new_tokens=96,
        )
        typed_reasoning = VerifyRepairPipeline(model=None).extract_typed_reasoning(task, response)
        effective_mode = response_mode
        generation_trace = _build_generation_trace(
            tokenizer=tokenizer,
            attempts=[
                _serialize_generation_attempt(
                    prompt=prompt,
                    response=response,
                    tokenizer=tokenizer,
                )
            ],
        )

    extracted_answer = _extract_final_number(response)
    correct = extracted_answer == int(case["ground_truth"])
    return {
        "case_id": str(case["case_id"]),
        "mode": "baseline",
        "prompt_seed": prompt_seed,
        "response_mode": effective_mode,
        "response": response,
        "extracted_answer": extracted_answer,
        "ground_truth": int(case["ground_truth"]),
        "correct": correct,
        "typed_reasoning_available": typed_reasoning is not None,
        "typed_reasoning_parse_status": _typed_reasoning_parse_status(typed_reasoning),
        "typed_reasoning": _serialize_jsonable(typed_reasoning),
        "generation_trace": generation_trace,
        "prompt_tokens": int(generation_trace["prompt_tokens"]),
        "response_tokens": int(generation_trace["response_tokens"]),
        "total_tokens": int(generation_trace["total_tokens"]),
        "latency_seconds": round(time.perf_counter() - started, 3),
    }


def _run_gsm8k_verify_only(
    case: dict[str, Any],
    baseline: dict[str, Any],
) -> dict[str, Any]:  # pragma: no cover
    from carnot.pipeline.verify_repair import VerifyRepairPipeline

    pipeline = VerifyRepairPipeline(model=None, domains=["arithmetic"], timeout_seconds=30.0)
    started = time.perf_counter()
    verification = pipeline.verify(
        str(case["question"]),
        str(baseline["response"]),
        domain="arithmetic",
    )
    verification_trace = _serialize_verification_result(verification)
    semantic_grounding = verification_trace.get("semantic_grounding", {})
    semantic_violations = []
    if isinstance(semantic_grounding, dict):
        violations = semantic_grounding.get("violations", [])
        if isinstance(violations, list):
            semantic_violations = violations
    semantic_violation_count = len(semantic_violations)
    flagged = (not verification.verified) or bool(verification.violations)
    return {
        "case_id": str(case["case_id"]),
        "mode": "verify_only",
        "prompt_seed": int(case["prompt_seeds"]["verify_only"]),
        "response_mode": baseline["response_mode"],
        "response": baseline["response"],
        "verified": verification.verified,
        "flagged": flagged,
        "n_constraints": len(verification.constraints),
        "n_violations": len(verification.violations),
        "semantic_violation_count": semantic_violation_count,
        "typed_reasoning_available": bool(verification.typed_reasoning),
        "typed_reasoning_parse_status": str(verification_trace["typed_reasoning_parse_status"]),
        "parseable": verification_trace["typed_reasoning_parse_status"] != "unavailable",
        "verification": verification_trace,
        "correct": bool(baseline["correct"]),
        "accepted_correct": bool(baseline["correct"]) and not flagged,
        "prompt_tokens": 0,
        "response_tokens": 0,
        "total_tokens": 0,
        "latency_seconds": round(time.perf_counter() - started, 3),
    }


def _run_gsm8k_verify_repair(
    case: dict[str, Any],
    baseline: dict[str, Any],
    *,
    model_spec: dict[str, str],
    model: Any,
    tokenizer: Any,
    policy: dict[str, Any],
    max_repairs: int,
) -> dict[str, Any]:  # pragma: no cover
    from carnot.pipeline.verify_repair import VerifyRepairPipeline

    pipeline = VerifyRepairPipeline(model=None, domains=["arithmetic"], timeout_seconds=30.0)
    current_response = str(baseline["response"])
    started = time.perf_counter()
    verification = pipeline.verify(str(case["question"]), current_response, domain="arithmetic")
    initial_trace = _serialize_verification_result(verification)
    history = [
        {
            "iteration": 0,
            "response_mode": baseline["response_mode"],
            "response": current_response,
            "verification": initial_trace,
        }
    ]
    if verification.verified:
        return {
            "case_id": str(case["case_id"]),
            "mode": "verify_repair",
            "prompt_seed": int(case["prompt_seeds"]["verify_repair"]),
            "response_mode": baseline["response_mode"],
            "initial_response": baseline["response"],
            "final_response": current_response,
            "initial_correct": bool(baseline["correct"]),
            "correct": bool(baseline["correct"]),
            "verified": True,
            "repaired": False,
            "n_repairs": 0,
            "typed_reasoning_parse_status": str(initial_trace["typed_reasoning_parse_status"]),
            "initial_verification": initial_trace,
            "final_verification": initial_trace,
            "prompt_tokens": 0,
            "response_tokens": 0,
            "total_tokens": 0,
            "latency_seconds": round(time.perf_counter() - started, 3),
            "history": history,
        }

    task_slice = str(case["task_slice"])
    task_prefix = (
        f"Question: {case['question']}\n\n"
        f"Your previous answer was:\n{current_response}\n\n"
        f"The following issues were found:\n"
        f"{VerifyRepairPipeline._format_violations(verification.violations)}\n\n"
        "Please provide a corrected answer."
    )
    n_repairs = 0
    total_prompt_tokens = 0
    total_response_tokens = 0
    for repair_idx in range(1, max_repairs + 1):
        response_mode = str(baseline["response_mode"])
        if response_mode == "structured_json":
            from carnot.pipeline.structured_reasoning import StructuredReasoningController

            controller = StructuredReasoningController(policy=policy)
            repair_seed = int(case["prompt_seeds"]["verify_repair"]) + repair_idx
            fallback_record: dict[str, Any] = {}

            def fallback_generate(
                generated_prompt: str,
                max_new_tokens: int,
                *,
                repair_seed: int = repair_seed,
                fallback_record: dict[str, Any] = fallback_record,
            ) -> str:
                response = _generate_text(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=generated_prompt,
                    prompt_seed=repair_seed,
                    max_new_tokens=max_new_tokens,
                )
                fallback_record.update(
                    {
                        "prompt": generated_prompt,
                        "response": response,
                    }
                )
                return response

            emission = controller.emit(
                question=task_prefix,
                task_slice=task_slice,
                model_name=model_spec["hf_id"],
                model=model,
                tokenizer=tokenizer,
                fallback_generate=fallback_generate,
            )
            current_response = emission.response
            generation_trace = _build_generation_trace(
                tokenizer=tokenizer,
                attempts=[
                    _serialize_generation_attempt(
                        prompt=str(attempt.prompt),
                        response=str(attempt.raw_response),
                        tokenizer=tokenizer,
                        valid=bool(attempt.valid),
                        error=attempt.error,
                    )
                    for attempt in emission.attempts
                ],
                fallback_record=fallback_record if emission.fallback_used else None,
            )
            current_response_mode = emission.response_mode
        else:
            repair_prompt = task_prefix + "\nReturn only the final numeric answer."
            current_response = _generate_text(
                model=model,
                tokenizer=tokenizer,
                prompt=repair_prompt,
                prompt_seed=int(case["prompt_seeds"]["verify_repair"]) + repair_idx,
                max_new_tokens=96,
            )
            generation_trace = _build_generation_trace(
                tokenizer=tokenizer,
                attempts=[
                    _serialize_generation_attempt(
                        prompt=repair_prompt,
                        response=current_response,
                        tokenizer=tokenizer,
                    )
                ],
            )
            current_response_mode = response_mode
        total_prompt_tokens += int(generation_trace["prompt_tokens"])
        total_response_tokens += int(generation_trace["response_tokens"])
        verification = pipeline.verify(str(case["question"]), current_response, domain="arithmetic")
        verification_trace = _serialize_verification_result(verification)
        history.append(
            {
                "iteration": repair_idx,
                "response_mode": current_response_mode,
                "response": current_response,
                "generation_trace": generation_trace,
                "verification": verification_trace,
            }
        )
        n_repairs = repair_idx
        if verification.verified:
            break

    final_answer = _extract_final_number(current_response)
    final_correct = final_answer == int(case["ground_truth"])
    return {
        "case_id": str(case["case_id"]),
        "mode": "verify_repair",
        "prompt_seed": int(case["prompt_seeds"]["verify_repair"]),
        "response_mode": baseline["response_mode"],
        "initial_response": baseline["response"],
        "final_response": current_response,
        "initial_correct": bool(baseline["correct"]),
        "correct": final_correct,
        "verified": verification.verified,
        "repaired": (not bool(baseline["correct"])) and final_correct,
        "n_repairs": n_repairs,
        "typed_reasoning_parse_status": str(
            history[-1]["verification"]["typed_reasoning_parse_status"]
        ),
        "initial_verification": initial_trace,
        "final_verification": history[-1]["verification"],
        "prompt_tokens": total_prompt_tokens,
        "response_tokens": total_response_tokens,
        "total_tokens": total_prompt_tokens + total_response_tokens,
        "latency_seconds": round(time.perf_counter() - started, 3),
        "history": history,
    }


def _run_humaneval_baseline(
    case: dict[str, Any],
    *,
    model: Any,
    tokenizer: Any,
) -> dict[str, Any]:  # pragma: no cover
    from carnot.pipeline.humaneval_live_benchmark import (  # type: ignore[import-untyped]
        build_candidate_code,
        execute_humaneval,
        run_instrumentation,
    )

    prompt_seed = int(case["prompt_seeds"]["baseline"])
    generation_prompt = (
        "You are an expert Python programmer.\n"
        "Complete the following function.\n"
        "Return ONLY the function body lines. No def line. No markdown fences.\n"
        "Indent with 4 spaces.\n\n"
        f"{case['prompt']}"
    )
    started = time.perf_counter()
    body = _generate_text(
        model=model,
        tokenizer=tokenizer,
        prompt=generation_prompt,
        prompt_seed=prompt_seed,
        max_new_tokens=220,
    )
    generation_trace = _build_generation_trace(
        tokenizer=tokenizer,
        attempts=[
            _serialize_generation_attempt(
                prompt=generation_prompt,
                response=body,
                tokenizer=tokenizer,
            )
        ],
    )
    candidate_code = build_candidate_code(str(case["prompt"]), body)
    harness = execute_humaneval(candidate_code, case, timeout=5.0)
    instrumentation = run_instrumentation(
        candidate_code,
        str(case["prompt"]),
        str(case["entry_point"]),
        official_tests=str(case["test"]),
    )
    return {
        "case_id": str(case["case_id"]),
        "mode": "baseline",
        "prompt_seed": prompt_seed,
        "response_mode": "answer_only_terse",
        "body": body,
        "candidate_code": candidate_code,
        "passed": harness.passed,
        "error_type": harness.error_type,
        "error_message": harness.error_message,
        "harness": {
            "passed": harness.passed,
            "error_type": harness.error_type,
            "error_message": harness.error_message,
            "stdout": harness.stdout,
        },
        "instrumentation": instrumentation,
        "generation_trace": generation_trace,
        "prompt_tokens": int(generation_trace["prompt_tokens"]),
        "response_tokens": int(generation_trace["response_tokens"]),
        "total_tokens": int(generation_trace["total_tokens"]),
        "latency_seconds": round(time.perf_counter() - started, 3),
    }


def _run_humaneval_verify_only(
    case: dict[str, Any],
    baseline: dict[str, Any],
) -> dict[str, Any]:  # pragma: no cover
    from carnot.pipeline.humaneval_live_benchmark import (
        execute_humaneval,
        run_instrumentation,
    )

    candidate_code = str(baseline["candidate_code"])
    started = time.perf_counter()
    execution_only = run_instrumentation(
        candidate_code,
        str(case["prompt"]),
        str(case["entry_point"]),
        official_tests=None,
    )
    after_execution_only = time.perf_counter()
    execution_plus_property = run_instrumentation(
        candidate_code,
        str(case["prompt"]),
        str(case["entry_point"]),
        official_tests=str(case["test"]),
    )
    after_execution_plus_property = time.perf_counter()
    harness = execute_humaneval(candidate_code, case, timeout=5.0)
    passed = harness.passed
    execution_only_detected = bool(execution_only["detected"])
    execution_plus_property_detected = bool(execution_plus_property["detected"])
    property_violation_count = int(execution_plus_property.get("n_property_violations", 0))
    execution_only_accepted = passed and not execution_only_detected
    execution_plus_property_accepted = passed and not execution_plus_property_detected
    property_only_detected = (
        execution_plus_property_detected
        and not execution_only_detected
        and property_violation_count > 0
    )
    finished = time.perf_counter()
    return {
        "case_id": str(case["case_id"]),
        "mode": "verify_only",
        "prompt_seed": int(case["prompt_seeds"]["verify_only"]),
        "response_mode": "answer_only_terse",
        "passed": passed,
        "harness": {
            "passed": harness.passed,
            "error_type": harness.error_type,
            "error_message": harness.error_message,
            "stdout": harness.stdout,
        },
        "execution_only": execution_only,
        "execution_plus_property": execution_plus_property,
        "execution_only_accepted": execution_only_accepted,
        "execution_plus_property_accepted": execution_plus_property_accepted,
        "property_only_detected": property_only_detected,
        "official_test_miss_caught_by_property": passed and property_violation_count > 0,
        "execution_only_latency_seconds": round(after_execution_only - started, 3),
        "execution_plus_property_latency_seconds": round(
            after_execution_plus_property - after_execution_only,
            3,
        ),
        "harness_latency_seconds": round(finished - after_execution_plus_property, 3),
        "prompt_tokens": 0,
        "response_tokens": 0,
        "total_tokens": 0,
        "latency_seconds": round(finished - started, 3),
    }


def _run_humaneval_verify_repair(
    case: dict[str, Any],
    baseline: dict[str, Any],
    *,
    model: Any,
    tokenizer: Any,
    max_repairs: int,
) -> dict[str, Any]:  # pragma: no cover
    from carnot.pipeline.humaneval_live_benchmark import (
        HarnessResult,
        build_candidate_code,
        build_repair_prompt,
        execute_humaneval,
        run_instrumentation,
    )

    current_body = str(baseline["body"])
    current_code = str(baseline["candidate_code"])
    harness = HarnessResult(
        passed=False,
        error_type=str(baseline["error_type"]),
        error_message=str(baseline["error_message"]),
        stdout="",
    )
    instrumentation = dict(baseline["instrumentation"])
    n_repairs = 0
    started = time.perf_counter()
    history = [
        {
            "iteration": 0,
            "body": current_body,
            "candidate_code": current_code,
            "harness": {
                "passed": bool(baseline["passed"]),
                "error_type": str(baseline["error_type"]),
                "error_message": str(baseline["error_message"]),
                "stdout": str(baseline.get("harness", {}).get("stdout", "")),
            },
            "instrumentation": dict(instrumentation),
            "generation_trace": _serialize_jsonable(baseline.get("generation_trace")),
        }
    ]
    total_prompt_tokens = 0
    total_response_tokens = 0

    if bool(baseline["passed"]):
        return {
            "case_id": str(case["case_id"]),
            "mode": "verify_repair",
            "prompt_seed": int(case["prompt_seeds"]["verify_repair"]),
            "response_mode": "answer_only_terse",
            "initial_passed": True,
            "passed": True,
            "repaired": False,
            "n_repairs": 0,
            "final_body": baseline["body"],
            "final_code": baseline["candidate_code"],
            "final_harness": history[0]["harness"],
            "final_instrumentation": history[0]["instrumentation"],
            "prompt_tokens": 0,
            "response_tokens": 0,
            "total_tokens": 0,
            "latency_seconds": round(time.perf_counter() - started, 3),
            "history": history,
        }

    for repair_idx in range(max_repairs):
        repair_prompt = build_repair_prompt(
            str(case["prompt"]),
            current_body,
            harness,
            instrumentation,
            repair_idx=repair_idx,
        )
        current_body = _generate_text(
            model=model,
            tokenizer=tokenizer,
            prompt=repair_prompt,
            prompt_seed=int(case["prompt_seeds"]["verify_repair"]) + repair_idx + 1,
            max_new_tokens=220,
        )
        generation_trace = _build_generation_trace(
            tokenizer=tokenizer,
            attempts=[
                _serialize_generation_attempt(
                    prompt=repair_prompt,
                    response=current_body,
                    tokenizer=tokenizer,
                )
            ],
        )
        current_code = build_candidate_code(str(case["prompt"]), current_body)
        harness = execute_humaneval(current_code, case, timeout=5.0)
        instrumentation = run_instrumentation(
            current_code,
            str(case["prompt"]),
            str(case["entry_point"]),
            official_tests=str(case["test"]),
        )
        total_prompt_tokens += int(generation_trace["prompt_tokens"])
        total_response_tokens += int(generation_trace["response_tokens"])
        history.append(
            {
                "iteration": repair_idx + 1,
                "repair_prompt": repair_prompt,
                "body": current_body,
                "candidate_code": current_code,
                "generation_trace": generation_trace,
                "harness": {
                    "passed": harness.passed,
                    "error_type": harness.error_type,
                    "error_message": harness.error_message,
                    "stdout": harness.stdout,
                },
                "instrumentation": instrumentation,
            }
        )
        n_repairs = repair_idx + 1
        if harness.passed:
            break

    return {
        "case_id": str(case["case_id"]),
        "mode": "verify_repair",
        "prompt_seed": int(case["prompt_seeds"]["verify_repair"]),
        "response_mode": "answer_only_terse",
        "initial_passed": bool(baseline["passed"]),
        "passed": harness.passed,
        "repaired": (not bool(baseline["passed"])) and harness.passed,
        "n_repairs": n_repairs,
        "final_body": current_body,
        "final_code": current_code,
        "final_harness": history[-1]["harness"],
        "final_instrumentation": history[-1]["instrumentation"],
        "prompt_tokens": total_prompt_tokens,
        "response_tokens": total_response_tokens,
        "total_tokens": total_prompt_tokens + total_response_tokens,
        "latency_seconds": round(time.perf_counter() - started, 3),
        "history": history,
        "error_type": harness.error_type,
        "error_message": harness.error_message,
    }


def _run_constraint_ir_baseline(
    case: dict[str, Any],
    *,
    model: Any,
    tokenizer: Any,
    policy: dict[str, Any],
) -> dict[str, Any]:  # pragma: no cover
    response_mode = recommended_response_mode(str(case["task_slice"]), policy)
    prompt, max_new_tokens = _build_constraint_ir_prompt(case, response_mode)
    started = time.perf_counter()
    response = _generate_text(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        prompt_seed=int(case["prompt_seeds"]["baseline"]),
        max_new_tokens=max_new_tokens,
    )
    evaluation = _evaluate_constraint_ir_response(case, response_mode, response)
    return {
        "case_id": str(case["case_id"]),
        "mode": "baseline",
        "prompt_seed": int(case["prompt_seeds"]["baseline"]),
        "response_mode": response_mode,
        "response": response,
        "output_style": str(evaluation["output_style"]),
        "constraint_extraction_coverage": float(evaluation["constraint_extraction_coverage"]),
        "exact_satisfaction": bool(evaluation["exact_satisfaction"]),
        "partial_satisfaction": float(evaluation["partial_satisfaction"]),
        "semantic_violation_count": int(evaluation["semantic_violation_count"]),
        "evaluation": evaluation,
        "latency_seconds": round(time.perf_counter() - started, 3),
    }


def _constraint_ir_verified(evaluation: dict[str, Any]) -> bool:  # pragma: no cover
    return bool(evaluation.get("exact_satisfaction"))


def _run_constraint_ir_verify_only(
    case: dict[str, Any],
    baseline: dict[str, Any],
) -> dict[str, Any]:  # pragma: no cover
    evaluation = dict(baseline["evaluation"])
    verified = _constraint_ir_verified(evaluation)
    return {
        "case_id": str(case["case_id"]),
        "mode": "verify_only",
        "prompt_seed": int(case["prompt_seeds"]["verify_only"]),
        "response_mode": str(baseline["response_mode"]),
        "response": baseline["response"],
        "verified": verified,
        "flagged": not verified,
        "output_style": str(evaluation["output_style"]),
        "constraint_extraction_coverage": float(evaluation["constraint_extraction_coverage"]),
        "exact_satisfaction": bool(evaluation["exact_satisfaction"]),
        "partial_satisfaction": float(evaluation["partial_satisfaction"]),
        "semantic_violation_count": int(evaluation["semantic_violation_count"]),
        "evaluation": evaluation,
    }


def _run_constraint_ir_verify_repair(
    case: dict[str, Any],
    baseline: dict[str, Any],
    *,
    model: Any,
    tokenizer: Any,
    max_repairs: int,
) -> dict[str, Any]:  # pragma: no cover
    evaluation = dict(baseline["evaluation"])
    if _constraint_ir_verified(evaluation):
        return {
            "case_id": str(case["case_id"]),
            "mode": "verify_repair",
            "prompt_seed": int(case["prompt_seeds"]["verify_repair"]),
            "response_mode": str(baseline["response_mode"]),
            "initial_verified": True,
            "verified": True,
            "repaired": False,
            "n_repairs": 0,
            "final_response": baseline["response"],
            "output_style": str(evaluation["output_style"]),
            "constraint_extraction_coverage": float(evaluation["constraint_extraction_coverage"]),
            "exact_satisfaction": bool(evaluation["exact_satisfaction"]),
            "partial_satisfaction": float(evaluation["partial_satisfaction"]),
            "semantic_violation_count": int(evaluation["semantic_violation_count"]),
            "evaluation": evaluation,
            "history": [
                {
                    "iteration": 0,
                    "response": baseline["response"],
                    "evaluation": evaluation,
                }
            ],
        }

    current_response = str(baseline["response"])
    response_mode = str(baseline["response_mode"])
    prompt, max_new_tokens = _build_constraint_ir_prompt(case, response_mode)
    issue_lines = [
        f"- parseable={evaluation.get('parseable')}",
        f"- answer_quality={evaluation.get('answer_quality')}",
        f"- constraint_coverage={evaluation.get('constraint_coverage')}",
    ]
    n_repairs = 0
    history = [
        {
            "iteration": 0,
            "response": current_response,
            "evaluation": dict(evaluation),
        }
    ]
    for repair_idx in range(max_repairs):
        repair_prompt = (
            f"{prompt}\n\n"
            "Your previous response did not satisfy the required contract.\n"
            f"Previous response:\n{current_response}\n\n"
            "Issues:\n"
            + "\n".join(issue_lines)
            + "\n\nAnswer again using the same response contract.\n"
        )
        current_response = _generate_text(
            model=model,
            tokenizer=tokenizer,
            prompt=repair_prompt,
            prompt_seed=int(case["prompt_seeds"]["verify_repair"]) + repair_idx + 1,
            max_new_tokens=max_new_tokens,
        )
        evaluation = _evaluate_constraint_ir_response(case, response_mode, current_response)
        issue_lines = [
            f"- parseable={evaluation.get('parseable')}",
            f"- answer_quality={evaluation.get('answer_quality')}",
            f"- constraint_coverage={evaluation.get('constraint_coverage')}",
        ]
        n_repairs = repair_idx + 1
        history.append(
            {
                "iteration": repair_idx + 1,
                "repair_prompt": repair_prompt,
                "response": current_response,
                "evaluation": dict(evaluation),
            }
        )
        if _constraint_ir_verified(evaluation):
            break

    verified = _constraint_ir_verified(evaluation)
    return {
        "case_id": str(case["case_id"]),
        "mode": "verify_repair",
        "prompt_seed": int(case["prompt_seeds"]["verify_repair"]),
        "response_mode": response_mode,
        "initial_verified": _constraint_ir_verified(dict(baseline["evaluation"])),
        "verified": verified,
        "repaired": (not _constraint_ir_verified(dict(baseline["evaluation"]))) and verified,
        "n_repairs": n_repairs,
        "final_response": current_response,
        "output_style": str(evaluation["output_style"]),
        "constraint_extraction_coverage": float(evaluation["constraint_extraction_coverage"]),
        "exact_satisfaction": bool(evaluation["exact_satisfaction"]),
        "partial_satisfaction": float(evaluation["partial_satisfaction"]),
        "semantic_violation_count": int(evaluation["semantic_violation_count"]),
        "evaluation": evaluation,
        "history": history,
    }


def _summarize_runs(
    benchmark: str,
    baseline_runs: list[dict[str, Any]],
    verify_only_runs: list[dict[str, Any]],
    verify_repair_runs: list[dict[str, Any]],
) -> dict[str, Any]:  # pragma: no cover
    n_cases = len(baseline_runs)
    if benchmark == "gsm8k_semantic":
        baseline_accuracy = sum(1 for run in baseline_runs if run["correct"]) / n_cases
        verify_accuracy = sum(1 for run in verify_only_runs if run["accepted_correct"]) / n_cases
        repair_accuracy = sum(1 for run in verify_repair_runs if run["correct"]) / n_cases
        n_wrong_answers = sum(1 for run in baseline_runs if not run["correct"])
        n_wrong_detected = sum(
            1 for run in verify_only_runs if run["flagged"] and not run["correct"]
        )
        false_positives = sum(1 for run in verify_only_runs if run["flagged"] and run["correct"])
        n_repaired = sum(1 for run in verify_repair_runs if run["repaired"])
        semantic_violation_count = sum(
            int(run.get("semantic_violation_count", 0)) for run in verify_only_runs
        )
        parse_coverage = (
            sum(
                1
                for run in verify_only_runs
                if str(run.get("typed_reasoning_parse_status", "unavailable")) != "unavailable"
            )
            / n_cases
        )
        return {
            "baseline": {
                "n_cases": n_cases,
                "accuracy": baseline_accuracy,
                "n_correct": sum(1 for run in baseline_runs if run["correct"]),
                "mean_latency_seconds": _round_mean(
                    [float(run.get("latency_seconds", 0.0)) for run in baseline_runs]
                ),
                "mean_prompt_tokens": _round_mean(
                    [float(run.get("prompt_tokens", 0.0)) for run in baseline_runs]
                ),
                "mean_response_tokens": _round_mean(
                    [float(run.get("response_tokens", 0.0)) for run in baseline_runs]
                ),
                "mean_total_tokens": _round_mean(
                    [float(run.get("total_tokens", 0.0)) for run in baseline_runs]
                ),
            },
            "verify_only": {
                "n_cases": n_cases,
                "accuracy": verify_accuracy,
                "n_flagged": sum(1 for run in verify_only_runs if run["flagged"]),
                "n_wrong_answers": n_wrong_answers,
                "n_wrong_detected": n_wrong_detected,
                "wrong_detection_rate": round(
                    n_wrong_detected / n_wrong_answers if n_wrong_answers else 0.0,
                    6,
                ),
                "semantic_violation_count": semantic_violation_count,
                "parse_coverage": round(parse_coverage, 6),
                "false_positives": false_positives,
                "false_positive_rate": round(
                    false_positives / max(1, sum(1 for run in baseline_runs if run["correct"])),
                    6,
                ),
                "mean_additional_latency_seconds": _round_mean(
                    [float(run.get("latency_seconds", 0.0)) for run in verify_only_runs]
                ),
                "mean_additional_tokens": _round_mean(
                    [float(run.get("total_tokens", 0.0)) for run in verify_only_runs]
                ),
            },
            "verify_repair": {
                "n_cases": n_cases,
                "accuracy": repair_accuracy,
                "n_repaired": n_repaired,
                "repair_yield": round(n_repaired / n_wrong_answers if n_wrong_answers else 0.0, 6),
                "avg_repairs": round(
                    sum(int(run["n_repairs"]) for run in verify_repair_runs) / n_cases,
                    3,
                ),
                "mean_additional_latency_seconds": _round_mean(
                    [float(run.get("latency_seconds", 0.0)) for run in verify_repair_runs]
                ),
                "mean_additional_tokens": _round_mean(
                    [float(run.get("total_tokens", 0.0)) for run in verify_repair_runs]
                ),
            },
            "paired_deltas": {
                "verify_only_minus_baseline": verify_accuracy - baseline_accuracy,
                "repair_minus_baseline": repair_accuracy - baseline_accuracy,
            },
        }

    if benchmark == "humaneval_property":
        baseline_pass = sum(1 for run in baseline_runs if run["passed"]) / n_cases
        n_wrong_answers = sum(1 for run in baseline_runs if not run["passed"])
        execution_only_pass = (
            sum(1 for run in verify_only_runs if run["execution_only_accepted"]) / n_cases
        )
        execution_plus_property_pass = (
            sum(1 for run in verify_only_runs if run["execution_plus_property_accepted"]) / n_cases
        )
        execution_only_wrong_detected = sum(
            1
            for run in verify_only_runs
            if (not run["passed"]) and run["execution_only"]["detected"]
        )
        execution_plus_property_wrong_detected = sum(
            1
            for run in verify_only_runs
            if (not run["passed"]) and run["execution_plus_property"]["detected"]
        )
        execution_only_false_positives = sum(
            1 for run in verify_only_runs if run["passed"] and run["execution_only"]["detected"]
        )
        execution_plus_property_false_positives = sum(
            1
            for run in verify_only_runs
            if run["passed"] and run["execution_plus_property"]["detected"]
        )
        n_repaired = sum(1 for run in verify_repair_runs if run["repaired"])
        repair_pass = sum(1 for run in verify_repair_runs if run["passed"]) / n_cases
        return {
            "baseline": {
                "n_cases": n_cases,
                "pass_at_1": baseline_pass,
                "mean_latency_seconds": _round_mean(
                    [float(run.get("latency_seconds", 0.0)) for run in baseline_runs]
                ),
            },
            "verify_only": {
                "n_cases": n_cases,
                "execution_only": {
                    "pass_at_1": execution_only_pass,
                    "n_wrong_answers": n_wrong_answers,
                    "n_wrong_detected": execution_only_wrong_detected,
                    "wrong_detection_rate": round(
                        execution_only_wrong_detected / n_wrong_answers if n_wrong_answers else 0.0,
                        6,
                    ),
                    "false_positives": execution_only_false_positives,
                    "false_positive_rate": round(
                        execution_only_false_positives
                        / max(1, sum(1 for run in baseline_runs if run["passed"])),
                        6,
                    ),
                    "mean_latency_seconds": _round_mean(
                        [
                            float(run.get("execution_only_latency_seconds", 0.0))
                            for run in verify_only_runs
                        ]
                    ),
                },
                "execution_plus_property": {
                    "pass_at_1": execution_plus_property_pass,
                    "n_wrong_answers": n_wrong_answers,
                    "n_wrong_detected": execution_plus_property_wrong_detected,
                    "wrong_detection_rate": round(
                        (
                            execution_plus_property_wrong_detected / n_wrong_answers
                            if n_wrong_answers
                            else 0.0
                        ),
                        6,
                    ),
                    "false_positives": execution_plus_property_false_positives,
                    "false_positive_rate": round(
                        execution_plus_property_false_positives
                        / max(1, sum(1 for run in baseline_runs if run["passed"])),
                        6,
                    ),
                    "property_violation_total": sum(
                        int(run["execution_plus_property"]["n_property_violations"])
                        for run in verify_only_runs
                    ),
                    "problems_with_property_violations": sum(
                        1
                        for run in verify_only_runs
                        if int(run["execution_plus_property"]["n_property_violations"]) > 0
                    ),
                    "official_test_misses_caught_by_property": sum(
                        1
                        for run in verify_only_runs
                        if run["official_test_miss_caught_by_property"]
                    ),
                    "execution_only_misses_caught_by_property": sum(
                        1 for run in verify_only_runs if run["property_only_detected"]
                    ),
                    "mean_latency_seconds": _round_mean(
                        [
                            float(run.get("execution_plus_property_latency_seconds", 0.0))
                            for run in verify_only_runs
                        ]
                    ),
                },
                "mean_total_latency_seconds": _round_mean(
                    [float(run.get("latency_seconds", 0.0)) for run in verify_only_runs]
                ),
            },
            "verify_repair": {
                "n_cases": n_cases,
                "pass_at_1": repair_pass,
                "n_repaired": n_repaired,
                "repair_success_rate": round(
                    n_repaired / n_wrong_answers if n_wrong_answers else 0.0,
                    6,
                ),
                "avg_repairs": round(
                    sum(int(run["n_repairs"]) for run in verify_repair_runs) / n_cases,
                    3,
                ),
                "mean_latency_seconds": _round_mean(
                    [float(run.get("latency_seconds", 0.0)) for run in verify_repair_runs]
                ),
            },
            "paired_deltas": {
                "execution_only_minus_baseline": execution_only_pass - baseline_pass,
                "execution_plus_property_minus_baseline": execution_plus_property_pass
                - baseline_pass,
                "execution_plus_property_minus_execution_only": (
                    execution_plus_property_pass - execution_only_pass
                ),
                "repair_minus_baseline": repair_pass - baseline_pass,
            },
        }

    if benchmark == "constraint_ir":

        def extract(run: dict[str, Any]) -> dict[str, Any]:
            return dict(run.get("evaluation", {}))

        def sum_breakdowns(runs: list[dict[str, Any]], field: str) -> dict[str, int]:
            totals = {
                "literal": 0,
                "semantic": 0,
                "search_optimization_limited": 0,
            }
            for run in runs:
                breakdown = extract(run).get(field, {})
                if isinstance(breakdown, dict):
                    for family in totals:
                        totals[family] += int(breakdown.get(family, 0))
            return totals

        def summarize_by_output_style(runs: list[dict[str, Any]]) -> dict[str, Any]:
            summary: dict[str, Any] = {}
            for style in _CONSTRAINT_IR_OUTPUT_STYLES:
                matching = [
                    extract(run) for run in runs if extract(run).get("output_style") == style
                ]
                if not matching:
                    continue
                count = len(matching)
                summary[style] = {
                    "n_cases": count,
                    "parse_success_rate": round(
                        sum(1 for item in matching if item.get("parseable")) / count,
                        6,
                    ),
                    "mean_constraint_extraction_coverage": _round_mean(
                        [
                            float(
                                item.get(
                                    "constraint_extraction_coverage",
                                    item.get("constraint_coverage", 0.0),
                                )
                            )
                            for item in matching
                        ]
                    ),
                    "exact_satisfaction_rate": round(
                        sum(1 for item in matching if item.get("exact_satisfaction")) / count,
                        6,
                    ),
                    "mean_partial_satisfaction": _round_mean(
                        [
                            float(item.get("partial_satisfaction", item.get("answer_quality", 0.0)))
                            for item in matching
                        ]
                    ),
                }
            return summary

        baseline_exact = (
            sum(1 for run in baseline_runs if extract(run).get("exact_satisfaction")) / n_cases
        )
        verify_exact = sum(1 for run in verify_only_runs if run["verified"]) / n_cases
        repair_exact = sum(1 for run in verify_repair_runs if run["verified"]) / n_cases
        n_failures = sum(1 for run in baseline_runs if not extract(run).get("exact_satisfaction"))
        n_repaired = sum(1 for run in verify_repair_runs if run["repaired"])
        return {
            "baseline": {
                "n_cases": n_cases,
                "parse_success_rate": round(
                    sum(1 for run in baseline_runs if extract(run).get("parseable")) / n_cases,
                    6,
                ),
                "mean_constraint_extraction_coverage": _round_mean(
                    [
                        float(
                            extract(run).get(
                                "constraint_extraction_coverage",
                                extract(run).get("constraint_coverage", 0.0),
                            )
                        )
                        for run in baseline_runs
                    ]
                ),
                "exact_satisfaction_rate": round(baseline_exact, 6),
                "mean_partial_satisfaction": _round_mean(
                    [
                        float(
                            extract(run).get(
                                "partial_satisfaction", extract(run).get("answer_quality", 0.0)
                            )
                        )
                        for run in baseline_runs
                    ]
                ),
                "semantic_violation_count": sum(
                    int(extract(run).get("semantic_violation_count", 0)) for run in baseline_runs
                ),
                "failures_by_constraint_family": sum_breakdowns(baseline_runs, "failure_breakdown"),
                "coverage_gaps_by_constraint_family": sum_breakdowns(
                    baseline_runs,
                    "coverage_gap_breakdown",
                ),
                "behavior_by_output_style": summarize_by_output_style(baseline_runs),
            },
            "verify_only": {
                "n_cases": n_cases,
                "verified_rate": round(verify_exact, 6),
                "n_flagged": sum(1 for run in verify_only_runs if run["flagged"]),
                "flagged_rate": round(
                    sum(1 for run in verify_only_runs if run["flagged"]) / n_cases,
                    6,
                ),
                "mean_constraint_extraction_coverage": _round_mean(
                    [
                        float(
                            extract(run).get(
                                "constraint_extraction_coverage",
                                extract(run).get("constraint_coverage", 0.0),
                            )
                        )
                        for run in verify_only_runs
                    ]
                ),
                "exact_satisfaction_rate": round(verify_exact, 6),
                "mean_partial_satisfaction": _round_mean(
                    [
                        float(
                            extract(run).get(
                                "partial_satisfaction", extract(run).get("answer_quality", 0.0)
                            )
                        )
                        for run in verify_only_runs
                    ]
                ),
                "semantic_violation_count": sum(
                    int(extract(run).get("semantic_violation_count", 0)) for run in verify_only_runs
                ),
                "cases_with_semantic_violations": sum(
                    1
                    for run in verify_only_runs
                    if int(extract(run).get("semantic_violation_count", 0)) > 0
                ),
                "failures_by_constraint_family": sum_breakdowns(
                    verify_only_runs,
                    "failure_breakdown",
                ),
            },
            "verify_repair": {
                "n_cases": n_cases,
                "verified_rate": round(repair_exact, 6),
                "n_repaired": n_repaired,
                "repair_yield": round(n_repaired / n_failures if n_failures else 0.0, 6),
                "avg_repairs": round(
                    sum(int(run.get("n_repairs", 0)) for run in verify_repair_runs) / n_cases,
                    3,
                ),
                "mean_partial_satisfaction": _round_mean(
                    [
                        float(
                            extract(run).get(
                                "partial_satisfaction", extract(run).get("answer_quality", 0.0)
                            )
                        )
                        for run in verify_repair_runs
                    ]
                ),
                "semantic_violation_count": sum(
                    int(extract(run).get("semantic_violation_count", 0))
                    for run in verify_repair_runs
                ),
                "failures_by_constraint_family": sum_breakdowns(
                    verify_repair_runs,
                    "failure_breakdown",
                ),
            },
            "paired_deltas": {
                "verify_only_minus_baseline_exact": verify_exact - baseline_exact,
                "repair_minus_baseline_exact": repair_exact - baseline_exact,
                "repair_minus_baseline_partial": (
                    _round_mean(
                        [
                            float(
                                extract(run).get(
                                    "partial_satisfaction", extract(run).get("answer_quality", 0.0)
                                )
                            )
                            for run in verify_repair_runs
                        ]
                    )
                    - _round_mean(
                        [
                            float(
                                extract(run).get(
                                    "partial_satisfaction", extract(run).get("answer_quality", 0.0)
                                )
                            )
                            for run in baseline_runs
                        ]
                    )
                ),
                "repair_minus_verify_only_exact": repair_exact - verify_exact,
            },
        }

    raise ValueError(f"Unsupported benchmark for summary: {benchmark}")


def _run_live_benchmark(args: argparse.Namespace) -> dict[str, Any]:  # pragma: no cover
    benchmark = str(args.benchmark)
    started_at = utc_now()
    started = time.perf_counter()
    policy_path = get_repo_root() / "results" / "monitorability_policy_213.json"
    policy = load_monitorability_policy(policy_path)
    records = _load_benchmark_records(benchmark)
    cohort = build_cohort_manifest(
        records,
        sample_size=args.sample_size,
        sample_seed=args.sample_seed,
    )
    checkpoint_dir = Path(args.checkpoint_dir)
    paired_runs: list[dict[str, Any]] = []
    statistics: dict[str, Any] = {}

    for model_spec in MODEL_SPECS:
        print(f"Running {benchmark} on {model_spec['name']} ({model_spec['hf_id']})")
        model, tokenizer = _load_live_model(model_spec)
        try:

            def execute_baseline_case(
                case: dict[str, Any],
                *,
                benchmark: str = benchmark,
                model_spec: dict[str, str] = model_spec,
                model: Any = model,
                tokenizer: Any = tokenizer,
                policy: dict[str, Any] = policy,
            ) -> dict[str, Any]:
                if benchmark == "gsm8k_semantic":
                    return _run_gsm8k_baseline(
                        case,
                        model_spec=model_spec,
                        model=model,
                        tokenizer=tokenizer,
                        policy=policy,
                    )
                if benchmark == "humaneval_property":
                    return _run_humaneval_baseline(case, model=model, tokenizer=tokenizer)
                return _run_constraint_ir_baseline(
                    case,
                    model=model,
                    tokenizer=tokenizer,
                    policy=policy,
                )

            baseline_runs = run_mode(
                benchmark=benchmark,
                model_name=model_spec["name"],
                mode="baseline",
                cases=cohort,
                checkpoint_dir=checkpoint_dir,
                execute_case=execute_baseline_case,
            )
            baseline_by_case = {run["case_id"]: run for run in baseline_runs}

            def execute_verify_only_case(
                case: dict[str, Any],
                *,
                benchmark: str = benchmark,
                baseline_by_case: dict[str, dict[str, Any]] = baseline_by_case,
            ) -> dict[str, Any]:
                baseline = baseline_by_case[str(case["case_id"])]
                if benchmark == "gsm8k_semantic":
                    return _run_gsm8k_verify_only(case, baseline)
                if benchmark == "humaneval_property":
                    return _run_humaneval_verify_only(case, baseline)
                return _run_constraint_ir_verify_only(case, baseline)

            verify_only_runs = run_mode(
                benchmark=benchmark,
                model_name=model_spec["name"],
                mode="verify_only",
                cases=cohort,
                checkpoint_dir=checkpoint_dir,
                execute_case=execute_verify_only_case,
            )

            def execute_verify_repair_case(
                case: dict[str, Any],
                *,
                benchmark: str = benchmark,
                baseline_by_case: dict[str, dict[str, Any]] = baseline_by_case,
                model_spec: dict[str, str] = model_spec,
                model: Any = model,
                tokenizer: Any = tokenizer,
                policy: dict[str, Any] = policy,
                max_repairs: int = args.max_repairs,
            ) -> dict[str, Any]:
                baseline = baseline_by_case[str(case["case_id"])]
                if benchmark == "gsm8k_semantic":
                    return _run_gsm8k_verify_repair(
                        case,
                        baseline,
                        model_spec=model_spec,
                        model=model,
                        tokenizer=tokenizer,
                        policy=policy,
                        max_repairs=max_repairs,
                    )
                if benchmark == "humaneval_property":
                    return _run_humaneval_verify_repair(
                        case,
                        baseline,
                        model=model,
                        tokenizer=tokenizer,
                        max_repairs=max_repairs,
                    )
                return _run_constraint_ir_verify_repair(
                    case,
                    baseline,
                    model=model,
                    tokenizer=tokenizer,
                    max_repairs=max_repairs,
                )

            verify_repair_runs = run_mode(
                benchmark=benchmark,
                model_name=model_spec["name"],
                mode="verify_repair",
                cases=cohort,
                checkpoint_dir=checkpoint_dir,
                execute_case=execute_verify_repair_case,
            )
        finally:
            _unload_live_model(model, tokenizer)

        model_summary = _summarize_runs(
            benchmark,
            baseline_runs,
            verify_only_runs,
            verify_repair_runs,
        )
        statistics[model_spec["name"]] = model_summary
        paired_runs.extend(
            [
                {
                    "benchmark": benchmark,
                    "mode": "baseline",
                    "model_name": model_spec["name"],
                    "model_hf_id": model_spec["hf_id"],
                    "summary": model_summary["baseline"],
                    "cases": baseline_runs,
                },
                {
                    "benchmark": benchmark,
                    "mode": "verify_only",
                    "model_name": model_spec["name"],
                    "model_hf_id": model_spec["hf_id"],
                    "summary": model_summary["verify_only"],
                    "cases": verify_only_runs,
                },
                {
                    "benchmark": benchmark,
                    "mode": "verify_repair",
                    "model_name": model_spec["name"],
                    "model_hf_id": model_spec["hf_id"],
                    "summary": model_summary["verify_repair"],
                    "cases": verify_repair_runs,
                },
            ]
        )

    finished_at = utc_now()
    return build_artifact_payload(
        benchmark=benchmark,
        output_path=Path(args.output),
        cohort=cohort,
        paired_runs=paired_runs,
        statistics=statistics,
        sample_seed=args.sample_seed,
        sample_size=args.sample_size,
        started_at=started_at,
        finished_at=finished_at,
        runtime_seconds=time.perf_counter() - started,
        checkpoint_dir=Path(args.checkpoint_dir),
        max_repairs=int(args.max_repairs),
        policy_path=policy_path,
        inference_mode=live_inference_mode(),
    )


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = build_parser()
    args = parser.parse_args(argv)
    payload = _run_live_benchmark(args)
    write_artifact(Path(args.output), payload)
    print(f"Saved {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
