"""Build the Exp6649 direct exact-certificate proposal corpus.

The task freezes a small feasible slice of the Exp6604 plan corpus. It asks two
admitted local MoE models for one direct proposal per task. Raw output remains
available before parsing. The exact executor then localizes the first failed
step without changing or regenerating the proposal.

Spec refs: REQ-REPORT-6649, SCENARIO-REPORT-6649-*,
REQ-CONSTRAINT-6649, SCENARIO-CONSTRAINT-6649-*, REQ-INFER-SOTA-6649,
and SCENARIO-INFER-SOTA-6649-*.
"""

from __future__ import annotations

import argparse
import base64
from collections.abc import Mapping, Sequence
from copy import deepcopy
import datetime
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import platform
import signal
import socket
import subprocess
import tempfile
import time
from typing import Any
import urllib.error
import urllib.request

from carnot import experiment_6604_exact_two_level_plan_corpus as exact_corpus
from carnot import experiment_6648_three_family_gguf_canaries as canaries
from carnot.inference.sota_models import cached_sota_pair


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_PATH = Path("results/experiment_6649_exact_certificate_proposal_corpus.json")
UPSTREAM_PATH = Path("results/experiment_6648_three_family_gguf_canaries.json")
EXP6604_PATH = Path("results/experiment_6604_exact_two_level_plan_corpus.json")
MODULE_PATH = Path("python/carnot/experiment_6649_exact_certificate_proposal_corpus.py")
TEST_PATH = Path("tests/python/test_experiment_6649_exact_certificate_proposal_corpus.py")
REPORT_SPEC_PATH = REPO_ROOT / "openspec/capabilities/research-reporting/spec.md"
CONSTRAINT_SPEC_PATH = REPO_ROOT / "openspec/capabilities/constraint-verification/spec.md"
INFERENCE_SPEC_PATH = REPO_ROOT / "openspec/capabilities/llm-ebm-inference/spec.md"
PROTECTED_PATHS = (Path("research-roadmap.yaml"), Path("scripts/research_conductor.py"))

INFERENCE_SUBSTRATE = "local_llama_cpp_cuda_direct_exact_certificate_generation"
VERIFIER_IS_ORACLE = False
PARSER_VERSION = "carnot.exact_certificate_plan_parser.v1"
MANIFEST_VERSION = "carnot.experiment_6649.frozen_task_manifest.v1"
ROW_VERSION = "carnot.experiment_6649.candidate_row.v1"
RANDOM_SEED = 6_649_000
SELECTED_STRATUM_INDEXES = (0, 1, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13)
FROZEN_TASK_IDS = tuple(
    [f"plan-calibration-{index:02d}" for index in SELECTED_STRATUM_INDEXES]
    + [f"plan-held-{index:02d}" for index in SELECTED_STRATUM_INDEXES]
)
EXPECTED_TASK_COUNT = len(FROZEN_TASK_IDS)

DEFINED_MODEL_SPECS = [
    {
        "family_id": "qwen36_flagship_moe",
        "hf_id": "unsloth/Qwen3.6-35B-A3B-GGUF",
        "role": "flagship_moe",
        "quantization": "Q4_K_M",
        "device_index": 0,
        "resolution_method": "cached_sota_pair",
        "headline_eligible": True,
    },
    {
        "family_id": "gemma4_26b_middle_moe",
        "hf_id": "unsloth/gemma-4-26B-A4B-it-GGUF",
        "role": "middle_moe",
        "quantization": "Q4_K_M",
        "device_index": 1,
        "resolution_method": "cached_sota_pair",
        "headline_eligible": True,
    },
]
EXPECTED_ROW_COUNT = EXPECTED_TASK_COUNT * len(DEFINED_MODEL_SPECS)

DECODE_PARAMETERS = {
    "chat_serialization": "llama_cpp_v1_chat_completions_embedded_gguf_template",
    "context_size": 4096,
    "max_output_tokens": 128,
    "temperature": 0.0,
    "top_p": 1.0,
    "top_k": 1,
    "repeat_penalty": 1.0,
    "stop_rules": ["</s>", "<|im_end|>", "<|endoftext|>"],
    "attempt_count": 1,
    "repair": False,
    "retry": False,
    "regeneration": False,
}

LOAD_TIMEOUT_S = 900.0
REQUEST_TIMEOUT_S = 120.0
SHUTDOWN_TIMEOUT_S = 30.0
RECOVERY_TIMEOUT_S = 120.0

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "honest_verdict",
    "verdict_class",
    "gate_check_summary",
    "upstream_gate_receipt",
    "defined_model_specs",
    "frozen_task_manifest",
    "candidate_rows",
    "parse_failure_rows",
    "model_level_metrics",
    "regeneration_headroom_rows",
    "candidate_corpus_complete",
    "regeneration_headroom_count",
    "per_unit_rows",
    "aggregate_row_recomputation",
    "preconditions_checked",
    "protected_files_unchanged",
    "inference_substrate",
    "verifier_is_oracle",
    "field_provenance",
    "random_seed",
    "duration_s",
    "tests_run",
    "reproducibility_checksum",
)

FIELD_PRINCIPLES = {
    "status": "The task ends with complete evidence or one named blocking condition.",
    "honest_verdict": "The conclusion separates direct evidence from any unattempted repair claim.",
    "verdict_class": "The closed class follows authentic row-derived direct evidence.",
    "gate_check_summary": "A block names the exact failed upstream, identity, row, parser, or checker value.",
    "upstream_gate_receipt": "The exact Exp6648 admission field and artifact hash authorize generation.",
    "defined_model_specs": "Only the flagship Qwen MoE and middle Gemma MoE own headline rows.",
    "frozen_task_manifest": "Tasks, prompts, targets, seeds, budgets, parser, and checker freeze before output.",
    "candidate_rows": "Every model-task unit retains raw, parsed, exact, prefix, token, timing, and receipt data.",
    "parse_failure_rows": "Missing parses remain explicit and never become invalid zero values.",
    "model_level_metrics": "Direct rates and uncertainty derive only from retained model rows.",
    "regeneration_headroom_rows": "Only invalid parsed rows with a useful exact prefix enter measured headroom.",
    "candidate_corpus_complete": "Completeness depends on expected authentic recheckable rows, not success rate.",
    "regeneration_headroom_count": "The integer is the exact size of the row-derived headroom subset.",
    "per_unit_rows": "Every expected headline model-task unit remains available for independent replay.",
    "aggregate_row_recomputation": "Counts, rates, completeness, and headroom rebuild from rows.",
    "preconditions_checked": "Inputs, cache, hardware, tools, identities, and hashes are checked before generation.",
    "protected_files_unchanged": "The active roadmap and conductor retain their before-run hashes.",
    "inference_substrate": "The artifact declares local llama.cpp CUDA direct generation.",
    "verifier_is_oracle": "The checker evaluates a separate proposal model and performs no intervention.",
    "field_provenance": "Every field carries source, hash, reducer, and schema lineage.",
    "random_seed": "The fixed task schedule gives both models the same per-task generation seed.",
    "duration_s": "Monotonic runtime makes the real workload visible.",
    "tests_run": "Commands, exits, and summaries show which checks ran.",
    "reproducibility_checksum": "The final content hash detects any artifact change.",
}

CLOSED_VERDICT_CLASSES = {
    "positive",
    "circular_positive",
    "null",
    "blocked",
    "disqualified",
    "partial",
}

DEFAULT_TESTS_RUN = [
    {
        "command": ".venv/bin/pytest tests/python/test_experiment_6649_exact_certificate_proposal_corpus.py -q --no-cov -n 0",
        "exit_code": 0,
        "summary": "focused tests passed",
    },
    {
        "command": "COVERAGE_FILE=/tmp/carnot_exp6649.coverage .venv/bin/coverage report --include='*/experiment_6649_exact_certificate_proposal_corpus.py' --fail-under=100 -m",
        "exit_code": 0,
        "summary": "new module statement coverage is 100%",
    },
]


def canonical_json(value: Any) -> str:
    """Return stable compact JSON for content-addressed receipts."""

    return json.dumps(value, separators=(",", ":"), sort_keys=True, ensure_ascii=False)


def sha256_bytes(value: bytes) -> str:
    """Return the project SHA-256 receipt format for bytes."""

    return "sha256:" + hashlib.sha256(value).hexdigest()


def sha256_json(value: Any) -> str:
    """Hash one canonical JSON value."""

    return sha256_bytes(canonical_json(value).encode("utf-8"))


def sha256_file(path: str | Path) -> str:
    """Hash one existing file or return an explicit missing marker."""

    target = Path(path)
    return sha256_bytes(target.read_bytes()) if target.is_file() else "missing"


def artifact_checksum(payload: Mapping[str, Any]) -> str:
    """Hash every final artifact field except the checksum itself."""

    return sha256_json(
        {key: value for key, value in payload.items() if key != "reproducibility_checksum"}
    )


def candidate_row_checksum(row: Mapping[str, Any]) -> str:
    """Hash one candidate row without its self-referential receipt."""

    return sha256_json({key: value for key, value in row.items() if key != "row_sha256"})


def manifest_checksum(manifest: Mapping[str, Any]) -> str:
    """Hash the frozen manifest without its self-referential receipt."""

    return sha256_json({key: value for key, value in manifest.items() if key != "manifest_sha256"})


def _exact_identity() -> JsonDict:
    """Bind parser, compiler, and checker source identities."""

    executor_source = inspect.getsource(exact_corpus.IndependentExactExecutor)
    step_source = inspect.getsource(exact_corpus._exact_step)  # noqa: SLF001
    return {
        "parser_version": PARSER_VERSION,
        "parser_module_sha256": sha256_file(REPO_ROOT / MODULE_PATH),
        "token_syntax_compiler_version": exact_corpus.TOKEN_COMPILER_VERSION,
        "action_semantic_compiler_version": exact_corpus.SEMANTIC_COMPILER_VERSION,
        "exact_executor_version": exact_corpus.EXECUTOR_VERSION,
        "exact_executor_source_sha256": sha256_bytes(executor_source.encode("utf-8")),
        "exact_step_source_sha256": sha256_bytes(step_source.encode("utf-8")),
        "exp6604_module_sha256": sha256_file(
            REPO_ROOT / "python/carnot/experiment_6604_exact_two_level_plan_corpus.py"
        ),
    }


def build_frozen_task_manifest() -> JsonDict:
    """Freeze 24 feasible Exp6604 tasks before any proposal exists."""

    source_by_id = {task["task_id"]: task for task in exact_corpus.generate_plan_tasks()}
    tasks = []
    for index, task_id in enumerate(FROZEN_TASK_IDS):
        source = deepcopy(source_by_id[task_id])
        if source.get("known_feasible") is not True or not source.get("gold_witness"):
            raise ValueError(f"frozen task is not exactly feasible: {task_id}")
        exact_target = str(source["gold_witness"])
        prompt = str(source["model_prompt_bytes"])
        tasks.append(
            {
                "task_id": task_id,
                "split": source["split"],
                "stratum": deepcopy(source["stratum"]),
                "known_feasible": True,
                "task_source_seed": source["seed"],
                "generation_seed": RANDOM_SEED + index + 1,
                "task_source_sha256": source["source_sha256"],
                "task_source_bytes_b64": base64.b64encode(
                    source["source_bytes"].encode("utf-8")
                ).decode("ascii"),
                "task_payload": source,
                "prompt": prompt,
                "prompt_sha256": sha256_bytes(prompt.encode("utf-8")),
                "exact_target": exact_target,
                "exact_target_sha256": sha256_bytes(exact_target.encode("utf-8")),
                "target_step_count": len(exact_target.splitlines()),
            }
        )
    identity = _exact_identity()
    manifest: JsonDict = {
        "schema": MANIFEST_VERSION,
        "task_count": len(tasks),
        "ordered_task_ids": [task["task_id"] for task in tasks],
        "split_counts": {
            split: sum(task["split"] == split for task in tasks)
            for split in ("calibration", "held")
        },
        "tasks": tasks,
        "decode_parameters": deepcopy(DECODE_PARAMETERS),
        "parser_version": PARSER_VERSION,
        "compiler_checker_identity": identity,
        "randomization_plan": {
            "selection": "fixed ordered task IDs; no outcome-conditioned selection",
            "model_launch_order": [spec["family_id"] for spec in DEFINED_MODEL_SPECS],
            "task_order": [task["task_id"] for task in tasks],
            "generation_seed_schedule": [task["generation_seed"] for task in tasks],
            "same_task_seed_for_both_models": True,
            "shuffle": False,
        },
        "manifest_sha256": "",
    }
    manifest["manifest_sha256"] = manifest_checksum(manifest)
    return manifest


def resolve_model_specs() -> list[JsonDict]:
    """Resolve the exact admitted Qwen and middle-Gemma cached GGUF pair."""

    pair = cached_sota_pair(gpu_indices=(0, 1), model_indices=(0, 1)) or []
    by_id = {str(row.get("hf_id")): row for row in pair if isinstance(row, Mapping)}
    rows = []
    for defined in DEFINED_MODEL_SPECS:
        resolved = by_id.get(defined["hf_id"], {})
        model_path = str(resolved.get("model_path", ""))
        rows.append(
            {
                **deepcopy(defined),
                "model_path": model_path,
                "model_sha256": sha256_file(model_path) if model_path else "missing",
                "resolved": bool(model_path and Path(model_path).is_file()),
            }
        )
    return rows


def _task_payload(task: Mapping[str, Any]) -> Mapping[str, Any]:
    payload = task.get("task_payload")
    if isinstance(payload, Mapping):
        return payload
    raw = base64.b64decode(str(task["task_source_bytes_b64"]), validate=True)
    return json.loads(raw.decode("utf-8"))


def parse_proposal(task: Mapping[str, Any], raw: bytes) -> JsonDict:
    """Parse only canonical task actions and preserve all failures explicitly."""

    base = {
        "parser_version": PARSER_VERSION,
        "parse_succeeded": False,
        "parsed_plan": None,
        "parsed_step_count": None,
        "parse_failure": None,
    }
    if not raw:
        return {**base, "parse_failure": "empty_output"}
    try:
        decoded = raw.decode("utf-8", "strict")
    except UnicodeDecodeError:
        return {**base, "parse_failure": "invalid_utf8"}
    candidate = decoded.strip()
    if not candidate:
        return {**base, "parse_failure": "empty_output"}
    if "```" in candidate:
        return {**base, "parse_failure": "code_fence_not_allowed"}
    allowed = {
        str(row["canonical_call"]) for row in _task_payload(task)["grounded_action_vocabulary"]
    }
    lines = candidate.splitlines()
    if any(line not in allowed for line in lines):
        return {**base, "parse_failure": "noncanonical_line"}
    return {
        **base,
        "parse_succeeded": True,
        "parsed_plan": "\n".join(lines),
        "parsed_step_count": len(lines),
        "parse_failure": None,
    }


def localize_exact_outcome(task: Mapping[str, Any], parsed_plan: str) -> JsonDict:
    """Replay exact steps and identify the first invalid step or final goal check."""

    payload = _task_payload(task)
    actions, parse_error = exact_corpus._executor_parse_plan(payload, parsed_plan)  # noqa: SLF001
    if parse_error is not None:
        raise ValueError(f"parsed plan failed exact parser: {parse_error}")
    state = frozenset(str(value) for value in payload["initial_state"])
    seen: frozenset[str] = frozenset()
    outcomes = []
    valid_calls = []
    first_failure: int | None = None
    for step, action in enumerate(actions):
        before = sorted(state)
        next_state, next_seen, reason, detail = exact_corpus._exact_step(  # noqa: SLF001
            payload, state, seen, action
        )
        accepted = next_state is not None and reason is None
        outcomes.append(
            {
                "step": step,
                "action": action["canonical_call"],
                "action_id": action["action_id"],
                "accepted": accepted,
                "reason": "step_valid" if accepted else reason,
                "detail": detail,
                "state_before": before,
                "state_after": sorted(next_state) if next_state is not None else before,
            }
        )
        if not accepted:
            first_failure = step
            break
        state, seen = next_state, next_seen
        valid_calls.append(str(action["canonical_call"]))
    exact_result = exact_corpus.IndependentExactExecutor().execute(payload, parsed_plan)
    if first_failure is None and exact_result["valid"] is not True:
        first_failure = len(actions)
        outcomes.append(
            {
                "step": len(actions),
                "action": None,
                "action_id": "final_goal_check",
                "accepted": False,
                "reason": exact_result["reason"],
                "detail": deepcopy(exact_result["detail"]),
                "state_before": deepcopy(exact_result["final_state"]),
                "state_after": deepcopy(exact_result["final_state"]),
            }
        )
    valid_prefix_length = len(valid_calls)
    return {
        "exact_final_validity": exact_result["valid"],
        "exact_final_result": exact_result,
        "per_step_exact_outcomes": outcomes,
        "first_failing_step": first_failure,
        "first_failure_reason": None if first_failure is None else outcomes[-1]["reason"],
        "valid_prefix_length": valid_prefix_length,
        "valid_prefix": "\n".join(valid_calls),
    }


def build_candidate_row(
    task: Mapping[str, Any],
    model: Mapping[str, Any],
    generation: Mapping[str, Any],
    process_receipt: Mapping[str, Any],
) -> JsonDict:
    """Seal one raw response before adding parser and exact-check results."""

    raw = bytes(generation.get("raw_output", b""))
    parsed = parse_proposal(task, raw)
    exact = (
        localize_exact_outcome(task, str(parsed["parsed_plan"]))
        if parsed["parse_succeeded"]
        else {
            "exact_final_validity": None,
            "exact_final_result": None,
            "per_step_exact_outcomes": [],
            "first_failing_step": None,
            "first_failure_reason": None,
            "valid_prefix_length": None,
            "valid_prefix": None,
        }
    )
    decoded = raw.decode("utf-8", "replace")
    row: JsonDict = {
        "schema": ROW_VERSION,
        "row_id": f"{model['family_id']}|{task['task_id']}",
        "model_family_id": model["family_id"],
        "model_hf_id": model["hf_id"],
        "model_role": model["role"],
        "model_path": model["model_path"],
        "model_sha256": model["model_sha256"],
        "headline_eligible": model.get("headline_eligible") is True,
        "task_id": task["task_id"],
        "split": task["split"],
        "task_source_sha256": task["task_source_sha256"],
        "prompt": task["prompt"],
        "prompt_sha256": task["prompt_sha256"],
        "exact_target_sha256": task["exact_target_sha256"],
        "target_step_count": task["target_step_count"],
        "generation_seed": task["generation_seed"],
        "decode_parameters": deepcopy(DECODE_PARAMETERS),
        "raw_output": decoded,
        "raw_output_bytes_b64": base64.b64encode(raw).decode("ascii"),
        "raw_output_byte_count": len(raw),
        "raw_output_sha256": sha256_bytes(raw),
        "raw_api_response_sha256": generation.get("raw_api_response_sha256"),
        "raw_recorded_before_parse": True,
        **parsed,
        **exact,
        "prompt_tokens": int(generation.get("prompt_tokens", 0)),
        "generated_tokens": int(generation.get("generated_tokens", 0)),
        "latency_s": round(float(generation.get("latency_s", 0.0)), 9),
        "started_monotonic_ns": int(generation.get("started_monotonic_ns", 0)),
        "finished_monotonic_ns": int(generation.get("finished_monotonic_ns", 0)),
        "http_status": int(generation.get("http_status", 0)),
        "finish_reason": str(generation.get("finish_reason", "unknown")),
        "generation_failure_kind": generation.get("failure_kind"),
        "process_and_accelerator_receipt": deepcopy(process_receipt),
        "lineage": {
            "manifest_schema": MANIFEST_VERSION,
            "task_source_sha256": task["task_source_sha256"],
            "parser_version": PARSER_VERSION,
            "exact_executor_version": exact_corpus.EXECUTOR_VERSION,
            "proposal_attempt": "direct_only",
        },
        "attempt_count": 1,
        "retry_count": 0,
        "regeneration_attempted": False,
        "repair_attempted": False,
        "row_sha256": "",
    }
    row["row_sha256"] = candidate_row_checksum(row)
    return row


def wilson_interval(successes: int, total: int) -> list[float]:
    """Return a two-sided 95 percent Wilson interval for one binomial rate."""

    if total <= 0:
        return [0.0, 0.0]
    z = 1.959963984540054
    rate = successes / total
    denominator = 1.0 + z * z / total
    center = (rate + z * z / (2.0 * total)) / denominator
    spread = (
        z * math.sqrt(rate * (1.0 - rate) / total + z * z / (4.0 * total * total)) / denominator
    )
    return [round(max(0.0, center - spread), 9), round(min(1.0, center + spread), 9)]


def _metrics(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    total = len(rows)
    parsed = sum(row.get("parse_succeeded") is True for row in rows)
    successes = sum(row.get("exact_final_validity") is True for row in rows)
    exact_invalid = sum(
        row.get("parse_succeeded") is True and row.get("exact_final_validity") is False
        for row in rows
    )
    return {
        "row_count": total,
        "parsed_row_count": parsed,
        "parse_failure_count": total - parsed,
        "exact_invalid_count": exact_invalid,
        "direct_exact_success_count": successes,
        "direct_exact_success_rate": round(successes / total, 9) if total else 0.0,
        "direct_exact_success_interval_95": wilson_interval(successes, total),
        "denominator_policy": "all expected direct rows, including explicit parse failures",
    }


def model_level_metrics(
    rows: Sequence[Mapping[str, Any]], model_specs: Sequence[Mapping[str, Any]]
) -> list[JsonDict]:
    """Recompute one direct-rate receipt per headline model."""

    result = []
    for model in model_specs:
        selected = [row for row in rows if row.get("model_family_id") == model.get("family_id")]
        result.append(
            {
                "model_family_id": model.get("family_id"),
                "model_hf_id": model.get("hf_id"),
                "expected_row_count": EXPECTED_TASK_COUNT,
                **_metrics(selected),
            }
        )
    return result


def pooled_metrics(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Recompute the pooled direct rate across both headline models."""

    return {"expected_row_count": EXPECTED_ROW_COUNT, **_metrics(rows)}


def parse_failure_rows(rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Return explicit failures without an invented exact-invalid value."""

    return [
        {
            "row_id": row.get("row_id"),
            "model_family_id": row.get("model_family_id"),
            "task_id": row.get("task_id"),
            "raw_output_sha256": row.get("raw_output_sha256"),
            "parse_failure": row.get("parse_failure"),
            "parsed_plan": None,
            "exact_final_validity": None,
            "valid_prefix_length": None,
        }
        for row in rows
        if row.get("parse_succeeded") is False
    ]


def regeneration_headroom_rows(rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Select only invalid parsed rows with a useful prefix and target suffix."""

    result = []
    for row in rows:
        prefix = row.get("valid_prefix_length")
        target_steps = row.get("target_step_count")
        if not (
            row.get("parse_succeeded") is True
            and row.get("exact_final_validity") is False
            and isinstance(prefix, int)
            and prefix > 0
            and isinstance(target_steps, int)
            and target_steps - prefix > 0
        ):
            continue
        result.append(
            {
                "row_id": row.get("row_id"),
                "model_family_id": row.get("model_family_id"),
                "task_id": row.get("task_id"),
                "raw_output_sha256": row.get("raw_output_sha256"),
                "first_failing_step": row.get("first_failing_step"),
                "first_failure_reason": row.get("first_failure_reason"),
                "valid_prefix_length": prefix,
                "target_step_count": target_steps,
                "remaining_step_count": target_steps - prefix,
                "reason": "non-empty exact-valid prefix with at least one target step remaining",
            }
        )
    return result


def _receipt_ready(receipt: Mapping[str, Any], model: Mapping[str, Any]) -> bool:
    return bool(
        receipt.get("authentic") is True
        and receipt.get("owned_process") is True
        and int(receipt.get("pid", 0) or 0) > 1
        and int(receipt.get("pid_start_ticks", 0) or 0) > 0
        and receipt.get("device_index") == model.get("device_index")
        and str(receipt.get("device_uuid", "")).startswith("GPU-")
        and receipt.get("model_sha256") == model.get("model_sha256")
        and receipt.get("cuda_offload") is True
        and int(receipt.get("offloaded_layers", 0) or 0) > 0
        and receipt.get("accelerator_observed") is True
    )


def _recheck_row(
    row: Mapping[str, Any], task: Mapping[str, Any], model: Mapping[str, Any]
) -> list[str]:
    failures = []
    try:
        raw = base64.b64decode(str(row.get("raw_output_bytes_b64", "")), validate=True)
    except (TypeError, ValueError):
        return ["raw_output_bytes_b64"]
    expected_parse = parse_proposal(task, raw)
    expected_exact = (
        localize_exact_outcome(task, str(expected_parse["parsed_plan"]))
        if expected_parse["parse_succeeded"]
        else {
            "exact_final_validity": None,
            "exact_final_result": None,
            "per_step_exact_outcomes": [],
            "first_failing_step": None,
            "first_failure_reason": None,
            "valid_prefix_length": None,
            "valid_prefix": None,
        }
    )
    equal_fields = {
        "model_family_id": model.get("family_id"),
        "model_hf_id": model.get("hf_id"),
        "model_path": model.get("model_path"),
        "model_sha256": model.get("model_sha256"),
        "task_id": task.get("task_id"),
        "split": task.get("split"),
        "task_source_sha256": task.get("task_source_sha256"),
        "prompt": task.get("prompt"),
        "prompt_sha256": task.get("prompt_sha256"),
        "exact_target_sha256": task.get("exact_target_sha256"),
        "target_step_count": task.get("target_step_count"),
        "generation_seed": task.get("generation_seed"),
        "raw_output": raw.decode("utf-8", "replace"),
        "raw_output_byte_count": len(raw),
        "raw_output_sha256": sha256_bytes(raw),
        **expected_parse,
        **expected_exact,
    }
    for field, expected in equal_fields.items():
        if row.get(field) != expected:
            failures.append(field)
    if row.get("decode_parameters") != DECODE_PARAMETERS:
        failures.append("decode_parameters")
    if row.get("raw_recorded_before_parse") is not True:
        failures.append("raw_recorded_before_parse")
    if row.get("attempt_count") != 1 or row.get("retry_count") != 0:
        failures.append("attempt_contract")
    if row.get("regeneration_attempted") is not False or row.get("repair_attempted") is not False:
        failures.append("no_intervention_contract")
    if not _receipt_ready(row.get("process_and_accelerator_receipt", {}), model):
        failures.append("process_and_accelerator_receipt")
    if row.get("row_sha256") != candidate_row_checksum(row):
        failures.append("row_sha256")
    return sorted(set(failures))


def recompute_aggregates(
    rows: Sequence[Mapping[str, Any]],
    manifest: Mapping[str, Any],
    model_specs: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Independently replay every expected row and rebuild all aggregates."""

    retained = [row for row in rows if isinstance(row, Mapping)]
    expected = [
        f"{model['family_id']}|{task['task_id']}"
        for model in model_specs
        for task in manifest.get("tasks", [])
    ]
    observed = [str(row.get("row_id")) for row in retained]
    failed_checks = []
    if len(retained) != len(expected):
        failed_checks.append("row_count")
    if observed != expected:
        failed_checks.append("row_order_or_membership")
    if len(observed) != len(set(observed)):
        failed_checks.append("duplicate_row_id")
    task_by_id = {
        str(task["task_id"]): task
        for task in manifest.get("tasks", [])
        if isinstance(task, Mapping)
    }
    model_by_id = {
        str(model["family_id"]): model for model in model_specs if isinstance(model, Mapping)
    }
    row_rechecks = []
    for row in retained:
        task = task_by_id.get(str(row.get("task_id")))
        model = model_by_id.get(str(row.get("model_family_id")))
        failures = (
            ["unknown_task_or_model"]
            if task is None or model is None
            else _recheck_row(row, task, model)
        )
        if failures:
            failed_checks.append(f"row:{row.get('row_id')}:{','.join(failures)}")
        row_rechecks.append(
            {
                "row_id": row.get("row_id"),
                "rechecked": not failures,
                "failed_fields": failures,
            }
        )
    models_ready = len(model_specs) == len(DEFINED_MODEL_SPECS) and all(
        model.get("resolved") is True
        and model.get("headline_eligible") is True
        and model.get("hf_id") == defined["hf_id"]
        and model.get("family_id") == defined["family_id"]
        and str(model.get("model_sha256", "")).startswith("sha256:")
        for model, defined in zip(model_specs, DEFINED_MODEL_SPECS, strict=False)
    )
    if not models_ready:
        failed_checks.append("model_specs")
    manifest_ready = bool(
        manifest.get("task_count") == EXPECTED_TASK_COUNT
        and manifest.get("ordered_task_ids") == list(FROZEN_TASK_IDS)
        and manifest.get("manifest_sha256") == manifest_checksum(manifest)
    )
    if not manifest_ready:
        failed_checks.append("frozen_task_manifest")
    headroom = regeneration_headroom_rows(retained)
    return {
        "candidate_corpus_complete": not failed_checks,
        "expected_row_count": len(expected),
        "observed_row_count": len(retained),
        "expected_row_ids": expected,
        "observed_row_ids": observed,
        "failed_checks": sorted(set(failed_checks)),
        "row_rechecks": row_rechecks,
        "model_level_metrics": model_level_metrics(retained, model_specs),
        "pooled_metrics": pooled_metrics(retained),
        "parse_failure_count": len(parse_failure_rows(retained)),
        "regeneration_headroom_count": len(headroom),
        "reducer": "expected ordered keys plus independent raw parse, exact replay, identity, receipt, and row hash checks",
    }


def _field_provenance(manifest: Mapping[str, Any]) -> dict[str, JsonDict]:
    """Give each required field source, hash, reducer, and schema lineage."""

    manifest_hash = str(manifest.get("manifest_sha256", "missing"))
    return {
        field: {
            "principle": FIELD_PRINCIPLES[field],
            "source": (
                "per_unit_rows"
                if field
                in {
                    "candidate_rows",
                    "parse_failure_rows",
                    "model_level_metrics",
                    "regeneration_headroom_rows",
                    "candidate_corpus_complete",
                    "regeneration_headroom_count",
                    "per_unit_rows",
                    "aggregate_row_recomputation",
                }
                else "frozen task contract and direct run receipt"
            ),
            "source_sha256": manifest_hash,
            "reducer": (
                "recompute_aggregates"
                if field
                in {
                    "model_level_metrics",
                    "candidate_corpus_complete",
                    "aggregate_row_recomputation",
                }
                else "direct immutable receipt"
            ),
            "schema": "carnot.experiment_6649.field_provenance.v1",
        }
        for field in REQUIRED_ARTIFACT_FIELDS
    }


def build_artifact(
    *,
    date: str,
    upstream_gate_receipt: Mapping[str, Any],
    model_specs: Sequence[Mapping[str, Any]],
    manifest: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    preconditions: Mapping[str, Any],
    protected_files: Mapping[str, Any],
    tests_run: Sequence[Mapping[str, Any]],
    duration_s: float,
) -> JsonDict:
    """Assemble a terminal artifact whose completeness ignores outcome quality."""

    candidates = [deepcopy(dict(row)) for row in rows]
    models = [deepcopy(dict(row)) for row in model_specs]
    reduction = recompute_aggregates(candidates, manifest, models)
    complete = reduction["candidate_corpus_complete"]
    failures = []
    if upstream_gate_receipt.get("passed") is not True:
        failures.append(
            {
                "check": "upstream_gate",
                "expected": True,
                "observed": upstream_gate_receipt.get("observed_value"),
            }
        )
    if preconditions.get("all_required_preconditions_available") is not True:
        failures.append(
            {
                "check": "preconditions",
                "expected": True,
                "observed": preconditions.get("all_required_preconditions_available"),
            }
        )
    if protected_files.get("all_unchanged") is not True:
        failures.append(
            {
                "check": "protected_files",
                "expected": True,
                "observed": protected_files.get("all_unchanged"),
            }
        )
    if not complete:
        failures.append(
            {
                "check": "candidate_rows",
                "expected": "all expected rows independently recheck",
                "observed": reduction["failed_checks"],
            }
        )
    complete = complete and not failures
    metrics = reduction["model_level_metrics"]
    headroom = regeneration_headroom_rows(candidates)
    pooled = reduction["pooled_metrics"]
    verdict_class = "positive" if complete and pooled["direct_exact_success_count"] > 0 else "null"
    if complete:
        status = "complete"
        honest_verdict = (
            f"complete: all {len(candidates)} direct proposal rows independently recheck; "
            f"pooled direct exact success is {pooled['direct_exact_success_count']}/"
            f"{pooled['row_count']} and measured regeneration headroom is {len(headroom)}; "
            "no repair was attempted and no repair claim is made"
        )
    else:
        status = "blocked_candidate_corpus_integrity"
        honest_verdict = (
            "blocked_candidate_corpus_integrity: expected complete authentic direct rows; "
            f"observed {failures!r}"
        )
        verdict_class = "blocked"
    payload: JsonDict = {
        "schema": "carnot.experiment_6649.exact_certificate_proposal_corpus.v1",
        "run_date": date,
        "status": status,
        "honest_verdict": honest_verdict,
        "verdict_class": verdict_class,
        "gate_check_summary": {"all_passed": complete, "failures": failures},
        "upstream_gate_receipt": deepcopy(dict(upstream_gate_receipt)),
        "defined_model_specs": models,
        "frozen_task_manifest": deepcopy(dict(manifest)),
        "candidate_rows": candidates,
        "parse_failure_rows": parse_failure_rows(candidates),
        "model_level_metrics": metrics,
        "regeneration_headroom_rows": headroom,
        "candidate_corpus_complete": complete,
        "regeneration_headroom_count": len(headroom),
        "per_unit_rows": deepcopy(candidates),
        "aggregate_row_recomputation": reduction,
        "preconditions_checked": deepcopy(dict(preconditions)),
        "protected_files_unchanged": deepcopy(dict(protected_files)),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": VERIFIER_IS_ORACLE,
        "field_provenance": _field_provenance(manifest),
        "random_seed": {
            "base": RANDOM_SEED,
            "task_generation_seed_schedule": [
                task.get("generation_seed") for task in manifest.get("tasks", [])
            ],
            "same_task_seed_for_both_models": True,
        },
        "duration_s": round(float(duration_s), 6),
        "tests_run": [deepcopy(dict(row)) for row in tests_run],
        "reproducibility_checksum": "",
    }
    payload["reproducibility_checksum"] = artifact_checksum(payload)
    return payload


def build_blocked_artifact(
    *,
    date: str,
    failed_condition: str,
    expected: Any,
    observed: Any,
    upstream_gate_receipt: Mapping[str, Any],
    model_specs: Sequence[Mapping[str, Any]],
    manifest: Mapping[str, Any],
    preconditions: Mapping[str, Any],
    protected_files: Mapping[str, Any],
    tests_run: Sequence[Mapping[str, Any]],
    duration_s: float,
) -> JsonDict:
    """Build a complete blocked schema without inventing missing unit rows."""

    payload: JsonDict = {
        "schema": "carnot.experiment_6649.exact_certificate_proposal_corpus.v1",
        "run_date": date,
        "status": f"blocked_{failed_condition}",
        "honest_verdict": (
            f"blocked_{failed_condition}: expected {expected!r}, observed {observed!r}"
        ),
        "verdict_class": "blocked",
        "gate_check_summary": {
            "all_passed": False,
            "failed_condition": failed_condition,
            "expected": expected,
            "observed": observed,
        },
        "upstream_gate_receipt": deepcopy(dict(upstream_gate_receipt)),
        "defined_model_specs": [deepcopy(dict(row)) for row in model_specs],
        "frozen_task_manifest": deepcopy(dict(manifest)),
        "candidate_rows": [],
        "parse_failure_rows": [],
        "model_level_metrics": model_level_metrics([], model_specs),
        "regeneration_headroom_rows": [],
        "candidate_corpus_complete": False,
        "regeneration_headroom_count": 0,
        "per_unit_rows": [],
        "aggregate_row_recomputation": {
            "candidate_corpus_complete": False,
            "expected_row_count": EXPECTED_ROW_COUNT,
            "observed_row_count": 0,
            "failed_checks": [failed_condition],
            "pooled_metrics": pooled_metrics([]),
            "regeneration_headroom_count": 0,
        },
        "preconditions_checked": deepcopy(dict(preconditions)),
        "protected_files_unchanged": deepcopy(dict(protected_files)),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": VERIFIER_IS_ORACLE,
        "field_provenance": _field_provenance(manifest),
        "random_seed": {
            "base": RANDOM_SEED,
            "task_generation_seed_schedule": [
                task.get("generation_seed") for task in manifest.get("tasks", [])
            ],
            "same_task_seed_for_both_models": True,
        },
        "duration_s": round(float(duration_s), 6),
        "tests_run": [deepcopy(dict(row)) for row in tests_run],
        "reproducibility_checksum": "",
    }
    payload["reproducibility_checksum"] = artifact_checksum(payload)
    return payload


def validate_artifact(payload: Mapping[str, Any]) -> list[str]:
    """Reject schema, row, aggregate, verdict, provenance, or checksum drift."""

    errors = []
    missing = sorted(set(REQUIRED_ARTIFACT_FIELDS) - set(payload))
    if missing:
        errors.append("missing_required_fields:" + ",".join(missing))
        return errors
    if payload.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate_mismatch")
    if payload.get("verifier_is_oracle") is not VERIFIER_IS_ORACLE:
        errors.append("verifier_is_oracle_mismatch")
    if payload.get("verdict_class") not in CLOSED_VERDICT_CLASSES:
        errors.append("verdict_class_invalid")
    if set(payload.get("field_provenance", {})) != set(REQUIRED_ARTIFACT_FIELDS):
        errors.append("field_provenance_mismatch")
    if payload.get("candidate_rows") != payload.get("per_unit_rows"):
        errors.append("candidate_per_unit_rows_mismatch")
    if payload.get("status") == "complete":
        reduction = recompute_aggregates(
            payload.get("candidate_rows", []),
            payload.get("frozen_task_manifest", {}),
            payload.get("defined_model_specs", []),
        )
        expected_headroom = regeneration_headroom_rows(payload.get("candidate_rows", []))
        if (
            reduction != payload.get("aggregate_row_recomputation")
            or payload.get("candidate_corpus_complete") is not True
            or payload.get("model_level_metrics") != reduction["model_level_metrics"]
            or payload.get("parse_failure_rows")
            != parse_failure_rows(payload.get("candidate_rows", []))
            or payload.get("regeneration_headroom_rows") != expected_headroom
            or payload.get("regeneration_headroom_count") != len(expected_headroom)
        ):
            errors.append("aggregate_recomputation_mismatch")
        if payload.get("verdict_class") not in {"positive", "null"}:
            errors.append("complete_verdict_class_invalid")
    else:
        if not str(payload.get("status", "")).startswith("blocked_"):
            errors.append("blocked_status_prefix_missing")
        if not str(payload.get("honest_verdict", "")).startswith("blocked_"):
            errors.append("blocked_verdict_prefix_missing")
        if payload.get("verdict_class") != "blocked":
            errors.append("blocked_verdict_class_mismatch")
        if payload.get("candidate_corpus_complete") is not False:
            errors.append("blocked_corpus_complete")
        if payload.get("regeneration_headroom_count") != 0:
            errors.append("blocked_headroom_nonzero")
    if payload.get("reproducibility_checksum") != artifact_checksum(payload):
        errors.append("reproducibility_checksum_mismatch")
    return errors


def write_artifact_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    """Synchronize complete bytes before one same-directory atomic replacement."""

    errors = validate_artifact(payload)
    if errors:
        raise ValueError(";".join(errors))
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode(
        "utf-8"
    )
    with tempfile.NamedTemporaryFile(dir=target.parent, prefix=".exp6649-", delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, target)
    directory = os.open(target.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def protected_hashes(root: Path) -> dict[str, str]:
    """Hash both orchestration files that this task cannot modify."""

    return {path.as_posix(): sha256_file(root / path) for path in PROTECTED_PATHS}


def protected_files_receipt(root: Path, before: Mapping[str, str]) -> JsonDict:
    """Compare protected before and after hashes without changing either file."""

    after = protected_hashes(root)
    rows = [
        {
            "path": path.as_posix(),
            "before_sha256": before.get(path.as_posix()),
            "after_sha256": after.get(path.as_posix()),
            "unchanged": before.get(path.as_posix()) == after.get(path.as_posix()),
        }
        for path in PROTECTED_PATHS
    ]
    return {"rows": rows, "all_unchanged": all(row["unchanged"] for row in rows)}


def build_upstream_gate_receipt(root: Path) -> JsonDict:
    """Bind the exact Exp6648 admission field and source hash."""

    path = root / UPSTREAM_PATH
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        payload = {}
    observed = payload.get("all_mandated_models_admitted")
    return {
        "path": UPSTREAM_PATH.as_posix(),
        "absolute_path": str(path.resolve()),
        "sha256": sha256_file(path),
        "field": "all_mandated_models_admitted",
        "expected_value": True,
        "observed_value": observed,
        "passed": observed is True,
    }


def _command_receipt(command: Sequence[str], root: Path, timeout_s: float) -> JsonDict:
    """Run one verification command and retain its exit and compact summary."""  # pragma: no cover

    started = time.monotonic()
    try:
        result = subprocess.run(
            list(command),
            cwd=root,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
        stdout = result.stdout
        stderr = result.stderr
        exit_code = result.returncode
    except (OSError, subprocess.TimeoutExpired) as exc:
        stdout = ""
        stderr = f"{type(exc).__name__}: {exc}"
        exit_code = 124 if isinstance(exc, subprocess.TimeoutExpired) else 127
    lines = [line for line in (stdout + "\n" + stderr).splitlines() if line.strip()]
    return {
        "command": " ".join(command),
        "exit_code": exit_code,
        "duration_s": round(time.monotonic() - started, 6),
        "summary": lines[-1] if lines else "no output",
        "stdout_sha256": sha256_bytes(stdout.encode("utf-8")),
        "stderr_sha256": sha256_bytes(stderr.encode("utf-8")),
    }


def run_verification_commands(root: Path) -> list[JsonDict]:
    """Run focused, coverage, spec, row, artifact, model-tier, and style checks."""  # pragma: no cover

    coverage_file = "/tmp/carnot_exp6649.coverage"
    commands = [
        (
            [
                ".venv/bin/pytest",
                TEST_PATH.as_posix(),
                "-q",
                "--no-cov",
                "-n",
                "0",
            ],
            300.0,
        ),
        (
            [
                ".venv/bin/coverage",
                "run",
                "--rcfile=/dev/null",
                f"--include=*/{MODULE_PATH.name}",
                "-m",
                "pytest",
                TEST_PATH.as_posix(),
                "-q",
                "--no-cov",
                "-n",
                "0",
            ],
            300.0,
        ),
        (
            [
                ".venv/bin/coverage",
                "report",
                "--rcfile=/dev/null",
                f"--include=*/{MODULE_PATH.name}",
                "--fail-under=100",
                "--show-missing",
            ],
            60.0,
        ),
        (
            [
                ".venv/bin/python",
                "scripts/check_spec_coverage.py",
                TEST_PATH.as_posix(),
            ],
            60.0,
        ),
        (
            [
                ".venv/bin/pytest",
                TEST_PATH.as_posix(),
                "-q",
                "--no-cov",
                "-n",
                "0",
                "-k",
                "completeness or validator or atomic",
            ],
            120.0,
        ),
        (
            [
                ".venv/bin/pytest",
                "tests/python/test_inference_sota_models.py",
                "-q",
                "--no-cov",
                "-n",
                "0",
            ],
            120.0,
        ),
        (
            [".venv/bin/ruff", "check", MODULE_PATH.as_posix(), TEST_PATH.as_posix()],
            60.0,
        ),
        (
            [
                ".venv/bin/ruff",
                "format",
                "--check",
                MODULE_PATH.as_posix(),
                TEST_PATH.as_posix(),
            ],
            60.0,
        ),
    ]
    return [_command_receipt(command, root, timeout) for command, timeout in commands]


def _host_resources(root: Path) -> JsonDict:
    """Record CPU, RAM, disk, runtime, cache, and platform identity."""  # pragma: no cover

    memory = {}
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            key, value = line.split(":", 1)
            memory[key] = int(value.strip().split()[0]) * 1024
    except (OSError, ValueError):
        memory = {}
    disk = os.statvfs(root)
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "cpu_count": os.cpu_count(),
        "ram_bytes": memory.get("MemTotal"),
        "ram_available_bytes": memory.get("MemAvailable"),
        "disk_free_bytes": disk.f_bavail * disk.f_frsize,
        "model_cache_root": str((Path.home() / ".cache/huggingface/hub").resolve()),
    }


def collect_preconditions(
    root: Path,
    upstream: Mapping[str, Any],
    models: Sequence[Mapping[str, Any]],
    manifest: Mapping[str, Any],
    protected_before: Mapping[str, str],
) -> JsonDict:
    """Check every required input before direct generation begins."""  # pragma: no cover

    llama = canaries.llama_cpp_receipt()
    gpus = canaries.gpu_inventory()
    checks = {
        "upstream_gate": upstream.get("passed") is True,
        "model_resolution": len(models) == 2 and all(row.get("resolved") is True for row in models),
        "model_hashes": all(
            str(row.get("model_sha256", "")).startswith("sha256:") for row in models
        ),
        "hardware": len(gpus) >= 2
        and all(int(row.get("memory_total_mb", 0)) > 0 for row in gpus[:2]),
        "tools": llama.get("exists") is True
        and llama.get("executable") is True
        and llama.get("cuda_linked") is True,
        "task_manifest": manifest.get("manifest_sha256") == manifest_checksum(manifest),
        "compiler_and_checker": all(
            str(value).startswith("sha256:")
            for key, value in manifest.get("compiler_checker_identity", {}).items()
            if key.endswith("sha256")
        ),
        "protected_hashes": all(
            str(value).startswith("sha256:") for value in protected_before.values()
        ),
    }
    return {
        "all_required_preconditions_available": all(checks.values()),
        "checks": checks,
        "failed_preconditions": [key for key, value in checks.items() if not value],
        "inputs": {
            "exp6604": {
                "path": EXP6604_PATH.as_posix(),
                "sha256": sha256_file(root / EXP6604_PATH),
            },
            "upstream": deepcopy(dict(upstream)),
            "module": {"path": MODULE_PATH.as_posix(), "sha256": sha256_file(root / MODULE_PATH)},
            "test": {"path": TEST_PATH.as_posix(), "sha256": sha256_file(root / TEST_PATH)},
            "specs": {
                path.relative_to(root).as_posix(): sha256_file(path)
                for path in (REPORT_SPEC_PATH, CONSTRAINT_SPEC_PATH, INFERENCE_SPEC_PATH)
            },
        },
        "models": [deepcopy(dict(row)) for row in models],
        "hardware": {"gpus": gpus, "resources": _host_resources(root)},
        "tools": {"llama_cpp": llama},
        "task_manifest": {
            "task_count": manifest.get("task_count"),
            "manifest_sha256": manifest.get("manifest_sha256"),
            "ordered_task_ids": manifest.get("ordered_task_ids"),
        },
        "compiler_checker_identity": deepcopy(manifest.get("compiler_checker_identity", {})),
        "randomization_plan": deepcopy(manifest.get("randomization_plan", {})),
        "protected_hashes_before": dict(protected_before),
        "download_allowed": False,
        "auto_tokenizer_allowed": False,
        "legacy_headline_rows_allowed": False,
    }


def _free_port() -> int:
    """Reserve one loopback port number for a short-lived model server."""  # pragma: no cover

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as handle:
        handle.bind(("127.0.0.1", 0))
        return int(handle.getsockname()[1])


def _server_command(model: Mapping[str, Any], port: int) -> list[str]:
    """Build the frozen direct llama-server command."""  # pragma: no cover

    server = Path.home() / ".cache/llama.cpp-master/build/bin/llama-server"
    return [
        str(server),
        "--model",
        str(model["model_path"]),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--ctx-size",
        str(DECODE_PARAMETERS["context_size"]),
        "--n-gpu-layers",
        "all",
        "--device",
        "CUDA0",
        "--split-mode",
        "none",
        "--main-gpu",
        "0",
        "--parallel",
        "1",
        "--batch-size",
        "128",
        "--ubatch-size",
        "128",
        "--offline",
        "--jinja",
        "--reasoning",
        "off",
        "--no-ui",
        "--log-verbosity",
        "4",
    ]


def _pid_start_ticks(pid: int) -> int:
    """Read Linux process start ticks for PID reuse protection."""  # pragma: no cover

    try:
        return int(Path(f"/proc/{pid}/stat").read_text().split()[21])
    except (OSError, ValueError, IndexError):
        return 0


def _http_bytes(url: str, payload: Mapping[str, Any] | None, timeout_s: float) -> tuple[int, bytes]:
    """Send one local JSON request and retain the exact response bytes."""  # pragma: no cover

    body = None if payload is None else canonical_json(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="GET" if body is None else "POST",
    )
    with urllib.request.urlopen(request, timeout=timeout_s) as response:  # noqa: S310
        return int(response.status), response.read()


def _generation_request(port: int, task: Mapping[str, Any]) -> JsonDict:
    """Run one frozen direct proposal request without retry or repair."""  # pragma: no cover

    started_ns = time.monotonic_ns()
    started = time.monotonic()
    payload = {
        "model": "local-gguf",
        "messages": [{"role": "user", "content": task["prompt"]}],
        "seed": int(task["generation_seed"]),
        "temperature": DECODE_PARAMETERS["temperature"],
        "top_p": DECODE_PARAMETERS["top_p"],
        "top_k": DECODE_PARAMETERS["top_k"],
        "repeat_penalty": DECODE_PARAMETERS["repeat_penalty"],
        "max_tokens": DECODE_PARAMETERS["max_output_tokens"],
        "stop": DECODE_PARAMETERS["stop_rules"],
        "stream": False,
    }
    try:
        status, raw_api = _http_bytes(
            f"http://127.0.0.1:{port}/v1/chat/completions", payload, REQUEST_TIMEOUT_S
        )
        decoded = json.loads(raw_api.decode("utf-8"))
        choice = decoded.get("choices", [{}])[0]
        text = str(choice.get("message", {}).get("content", ""))
        usage = decoded.get("usage", {})
        failure = None
        finish = str(choice.get("finish_reason", "unknown"))
    except (
        OSError,
        TimeoutError,
        urllib.error.URLError,
        UnicodeDecodeError,
        json.JSONDecodeError,
    ) as exc:
        status = 124
        raw_api = b""
        text = ""
        usage = {}
        failure = f"{type(exc).__name__}: {exc}"
        finish = "request_failure"
    return {
        "raw_output": text.encode("utf-8"),
        "raw_api_response_sha256": sha256_bytes(raw_api),
        "prompt_tokens": int(usage.get("prompt_tokens", 0) or 0),
        "generated_tokens": int(usage.get("completion_tokens", 0) or 0),
        "latency_s": time.monotonic() - started,
        "started_monotonic_ns": started_ns,
        "finished_monotonic_ns": time.monotonic_ns(),
        "http_status": status,
        "finish_reason": finish,
        "failure_kind": failure,
    }


def _compact_gpu(snapshot: Mapping[str, Any]) -> JsonDict:
    """Keep the selected accelerator facts needed by each unit row."""  # pragma: no cover

    return {
        "device_index": snapshot.get("index"),
        "device_uuid": snapshot.get("device_uuid") or snapshot.get("uuid"),
        "name": snapshot.get("name"),
        "memory_used_mb": snapshot.get("memory_used_mb"),
        "memory_free_mb": snapshot.get("memory_free_mb"),
        "utilization_pct": snapshot.get("utilization_pct"),
        "temperature_c": snapshot.get("temperature_c"),
        "model_pid_present": snapshot.get("model_pid_present"),
        "observed_monotonic_ns": snapshot.get("observed_monotonic_ns"),
    }


def _run_model_session(
    root: Path, model: Mapping[str, Any], tasks: Sequence[Mapping[str, Any]]
) -> tuple[list[JsonDict], JsonDict]:  # pragma: no cover - exercised by the required live E2E run
    """Load one model, collect all direct outputs, then prove unload."""  # pragma: no cover

    port = _free_port()
    command = _server_command(model, port)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(model["device_index"])
    before = canaries._gpu_snapshot(int(model["device_index"]), 0)  # noqa: SLF001
    baseline = int(before.get("memory_used_mb", 0) or 0)
    generations = []
    process: subprocess.Popen[bytes] | None = None
    process_start_ticks = 0
    loaded = False
    offloaded_layers = 0
    resident: JsonDict = {}
    started_ns = time.monotonic_ns()
    stdout_bytes = b""
    stderr_bytes = b""
    exit_code: int | None = None
    error = ""
    with tempfile.TemporaryDirectory(prefix=f"exp6649-{model['family_id']}-") as directory:
        stdout_path = Path(directory) / "stdout.bin"
        stderr_path = Path(directory) / "stderr.bin"
        try:
            with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
                process = subprocess.Popen(
                    command,
                    cwd=root,
                    env=env,
                    stdin=subprocess.DEVNULL,
                    stdout=stdout,
                    stderr=stderr,
                )
                process_start_ticks = _pid_start_ticks(process.pid)
            deadline = time.monotonic() + LOAD_TIMEOUT_S
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError(f"llama-server exited during load: {process.returncode}")
                try:
                    status, raw = _http_bytes(f"http://127.0.0.1:{port}/health", None, 0.5)
                    health = json.loads(raw.decode("utf-8"))
                    if status == 200 and health.get("status") == "ok":
                        loaded = True
                        break
                except (
                    OSError,
                    TimeoutError,
                    urllib.error.URLError,
                    UnicodeDecodeError,
                    json.JSONDecodeError,
                ):
                    pass
                time.sleep(0.5)
            if not loaded:
                raise TimeoutError("llama-server load timeout")
            for _ in range(40):
                stderr_bytes = stderr_path.read_bytes()
                offloaded_layers = canaries._offloaded_layers(  # noqa: SLF001
                    stderr_bytes.decode("utf-8", "replace")
                )
                if offloaded_layers > 0:
                    break
                time.sleep(0.25)
            resident = canaries._gpu_snapshot(int(model["device_index"]), process.pid)  # noqa: SLF001
            if offloaded_layers <= 0 or resident.get("model_pid_present") is not True:
                raise RuntimeError("CUDA accelerator receipt missing")
            generations = [_generation_request(port, task) for task in tasks]
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            while len(generations) < len(tasks):
                now = time.monotonic_ns()
                generations.append(
                    {
                        "raw_output": b"",
                        "raw_api_response_sha256": sha256_bytes(b""),
                        "prompt_tokens": 0,
                        "generated_tokens": 0,
                        "latency_s": 0.0,
                        "started_monotonic_ns": now,
                        "finished_monotonic_ns": now,
                        "http_status": 503,
                        "finish_reason": "process_failure",
                        "failure_kind": error,
                    }
                )
        finally:
            if process is not None and process.poll() is None:
                process.send_signal(signal.SIGTERM)
                try:
                    exit_code = process.wait(timeout=SHUTDOWN_TIMEOUT_S)
                except subprocess.TimeoutExpired:
                    process.kill()
                    exit_code = process.wait(timeout=5)
            elif process is not None:
                exit_code = process.returncode
            stdout_bytes = stdout_path.read_bytes() if stdout_path.is_file() else b""
            stderr_bytes = stderr_path.read_bytes() if stderr_path.is_file() else b""
    pid = 0 if process is None else process.pid
    after: JsonDict = {}
    recovered = False
    recovery_started = time.monotonic()
    while time.monotonic() - recovery_started < RECOVERY_TIMEOUT_S:
        after = canaries._gpu_snapshot(int(model["device_index"]), pid)  # noqa: SLF001
        recovered = (
            pid > 1
            and not Path(f"/proc/{pid}").exists()
            and after.get("model_pid_present") is False
            and abs(int(after.get("memory_used_mb", 0) or 0) - baseline) <= 512
        )
        if recovered:
            break
        time.sleep(0.5)
    receipt = {
        "session_id": f"exp6649-{model['family_id']}-{started_ns}",
        "pid": pid,
        "pid_start_ticks": process_start_ticks,
        "parent_pid": os.getpid(),
        "executable": command[0],
        "argv": command,
        "argv_sha256": sha256_json(command),
        "device_index": model["device_index"],
        "device_uuid": before.get("device_uuid") or before.get("uuid"),
        "model_sha256": model["model_sha256"],
        "cuda_offload": offloaded_layers > 0,
        "offloaded_layers": offloaded_layers,
        "accelerator_observed": resident.get("model_pid_present") is True,
        "owned_process": pid > 1,
        "started_monotonic_ns": started_ns,
        "ended_monotonic_ns": time.monotonic_ns(),
        "exit_code": exit_code,
        "absent_after_exit": pid > 1 and not Path(f"/proc/{pid}").exists(),
        "memory_recovered": recovered,
        "gpu_before": _compact_gpu(before),
        "gpu_resident": _compact_gpu(resident),
        "gpu_after": _compact_gpu(after),
        "stdout_sha256": sha256_bytes(stdout_bytes),
        "stderr_sha256": sha256_bytes(stderr_bytes),
        "error": error,
        "authentic": bool(
            loaded
            and offloaded_layers > 0
            and resident.get("model_pid_present") is True
            and pid > 1
            and not Path(f"/proc/{pid}").exists()
            and recovered
        ),
    }
    return generations, receipt


def _first_failed_precondition(preconditions: Mapping[str, Any]) -> tuple[str, Any]:
    """Return the first named false precondition in declared order."""  # pragma: no cover

    for name, value in preconditions.get("checks", {}).items():
        if value is not True:
            return str(name), value
    return "preconditions", preconditions.get("all_required_preconditions_available")


def run(date: str, root: Path = REPO_ROOT) -> JsonDict:
    """Run preconditions, direct generation, checks, and one atomic final write."""  # pragma: no cover

    started = time.monotonic()
    protected_before = protected_hashes(root)
    upstream = build_upstream_gate_receipt(root)
    manifest = build_frozen_task_manifest()
    models = resolve_model_specs()
    preconditions = collect_preconditions(root, upstream, models, manifest, protected_before)
    tests_run = run_verification_commands(root)
    if preconditions["all_required_preconditions_available"] is not True:
        condition, observed = _first_failed_precondition(preconditions)
        protected = protected_files_receipt(root, protected_before)
        artifact = build_blocked_artifact(
            date=date,
            failed_condition=condition,
            expected=True,
            observed=observed,
            upstream_gate_receipt=upstream,
            model_specs=models,
            manifest=manifest,
            preconditions=preconditions,
            protected_files=protected,
            tests_run=tests_run,
            duration_s=time.monotonic() - started,
        )
        write_artifact_atomic(root / RESULT_PATH, artifact)
        return artifact
    rows = []
    sessions = []
    for model in models:
        generations, receipt = _run_model_session(root, model, manifest["tasks"])
        sessions.append(receipt)
        for task, generation in zip(manifest["tasks"], generations, strict=True):
            rows.append(build_candidate_row(task, model, generation, receipt))
    preconditions["generation_sessions"] = sessions
    protected = protected_files_receipt(root, protected_before)
    artifact = build_artifact(
        date=date,
        upstream_gate_receipt=upstream,
        model_specs=models,
        manifest=manifest,
        rows=rows,
        preconditions=preconditions,
        protected_files=protected,
        tests_run=tests_run,
        duration_s=time.monotonic() - started,
    )
    write_artifact_atomic(root / RESULT_PATH, artifact)
    return artifact


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=datetime.datetime.now(datetime.UTC).strftime("%Y%m%d"))
    parser.add_argument("--validate", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run Exp6649 or validate one existing artifact."""  # pragma: no cover

    args = _parse_args(argv)
    if args.validate is not None:
        payload = json.loads(args.validate.read_text(encoding="utf-8"))
        errors = validate_artifact(payload)
        if errors:
            print("\n".join(errors))
            return 1
        print("valid")
        return 0
    artifact = run(args.date)
    print(
        canonical_json(
            {
                "status": artifact["status"],
                "candidate_corpus_complete": artifact["candidate_corpus_complete"],
                "regeneration_headroom_count": artifact["regeneration_headroom_count"],
                "result": RESULT_PATH.as_posix(),
            }
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
