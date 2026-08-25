"""Measure Gemma 4 26B-A4B direct-plan headroom on the frozen Exp6604 corpus.

The runner sends each frozen prompt to one local Gemma 4 26B-A4B GGUF process three times.
It stores raw bytes before it calls the independent Exp6604 executor. Failed
outputs stay charged, so the artifact separates complete evidence from useful
headroom.

Spec refs: REQ-REPORT-6607 and SCENARIO-REPORT-6607-BLOCKED through
SCENARIO-REPORT-6607-ATTACKS-AND-ATOMIC.
"""

from __future__ import annotations

import argparse
import base64
from collections import Counter
from collections.abc import Mapping, Sequence
from copy import deepcopy
import datetime
import json
import math
import os
from pathlib import Path
import platform
import shutil
import signal
import socket
import subprocess
import tempfile
import time
from typing import Any
import urllib.error

from carnot import experiment_6567_sequential_flagship_gguf_admission as admission
from carnot import experiment_6573_sequential_flagship_gguf_admission_v2 as runtime_helpers
from carnot import experiment_6581_qwen36_flagship_source_shard as stream_helpers
from carnot import experiment_6583_gemma4_26b_a4b_flagship_source_shard as gemma_helpers
from carnot import experiment_6604_exact_two_level_plan_corpus as corpus_helpers
from carnot.inference.sota_models import SOTA_GGUF_MODELS, resolve_cached_gguf


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_DATE = "20260825"
GEMMA26_HUB_ID = "unsloth/gemma-4-26B-A4B-it-GGUF"
GEMMA26_ARCHITECTURE = "gemma4"
INFERENCE_SUBSTRATE = "live_local_gemma4_26b_a4b_gguf_direct_plan_baseline_cuda_llamacpp"
RESULT_RELATIVE_PATH = Path("results/experiment_6607_gemma4_26b_direct_headroom.json")
CHECKPOINT_RELATIVE_PATH = Path(
    "results/checkpoints/experiment_6607_gemma4_26b_direct_headroom/checkpoint.json"
)
EXP6604_RELATIVE_PATH = Path("results/experiment_6604_exact_two_level_plan_corpus.json")
SPEC_RELATIVE_PATH = Path("openspec/capabilities/research-reporting/spec.md")
MODULE_RELATIVE_PATH = Path("python/carnot/experiment_6607_gemma4_26b_direct_headroom.py")
TEST_RELATIVE_PATH = Path("tests/python/test_experiment_6607_gemma4_26b_direct_headroom.py")
PROTECTED_RELATIVE_PATHS = (
    Path("research-roadmap.yaml"),
    Path("scripts/research_conductor.py"),
)

SEED_SCHEDULE = (6_607_001, 6_607_002, 6_607_003)
CONTEXT_SIZE = 4096
MAX_OUTPUT_TOKENS = 128
TEMPERATURE = 0.7
TOP_P = 0.95
TOP_K = 40
REPEAT_PENALTY = 1.0
STOP_RULES = ("</s>", "<|im_end|>", "<|endoftext|>")
LOAD_TIMEOUT_S = 900.0
PER_GENERATION_TIMEOUT_S = 90.0
TASK_TIMEOUT_S = 21_600.0
SHUTDOWN_TIMEOUT_S = 30.0
RECOVERY_TIMEOUT_S = 180.0
RECOVERY_TOLERANCE_MB = 256
MIN_FREE_VRAM_MB = 16_384
MIN_CHECKPOINT_FREE_BYTES = 1_000_000_000
TELEMETRY_EVERY_ROWS = 12
EXPECTED_TASK_COUNT = 72
EXPECTED_ROW_COUNT = EXPECTED_TASK_COUNT * len(SEED_SCHEDULE)

_middle_moe = SOTA_GGUF_MODELS[1]
_resolved_model_path = resolve_cached_gguf(GEMMA26_HUB_ID) or ""
MODEL_SPECS = [
    {
        "name": _middle_moe["name"],
        "hf_id": GEMMA26_HUB_ID,
        "repository_id": GEMMA26_HUB_ID,
        "model_path": _resolved_model_path,
        "gpu": 0,
        "gpu_assignment_reviewed": True,
        "quantization": _middle_moe["quantization"],
        "headline_eligible": True,
    }
]

FAILURE_CLASSES = (
    "syntax_failure",
    "semantic_failure",
    "unmet_goal",
    "refusal",
    "invalid_generation",
    "timeout",
    "process_failure",
)
REQUIRED_ATTACK_IDS = (
    "cross_family_tuning",
    "prompt_drift",
    "seed_drift",
    "split_leakage",
    "omitted_failures",
    "cpu_fallback",
    "fake_cuda_offload",
    "wrong_model",
    "tokenizer_substitution",
    "response_regeneration",
    "aggregate_disagreement",
    "protected_file_mutation",
)
REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "honest_verdict",
    "verdict_class",
    "gate_check_summary",
    "per_unit_rows",
    "model_spec_and_identity",
    "gpu_process_receipts",
    "prompt_and_decode_contract",
    "raw_model_receipts",
    "failure_rows",
    "family_headroom_summary",
    "gemma26_headroom_ready_score",
    "attack_rows",
    "preconditions_checked",
    "protected_files_unchanged",
    "inference_substrate",
    "verifier_is_oracle",
    "field_provenance",
    "duration_s",
    "tests_run",
    "reproducibility_checksum",
)
FIELD_PRINCIPLES = {
    "status": "The family run is terminal, resumable, and never hides a resource block.",
    "honest_verdict": "The verdict distinguishes row completion from the presence of useful headroom.",
    "verdict_class": "Use the closed enum; baseline qualification alone is null infrastructure.",
    "gate_check_summary": "Any block names the failed upstream, cache, GPU, model, tokenizer, row, timeout, checkpoint, or exact condition.",
    "per_unit_rows": "Every task and seed carries immutable inputs, raw output, parse, exact result, tokens, latency, process, and failures.",
    "model_spec_and_identity": "The mandated hub ID, GGUF shards, hashes, tokenizer, chat template, quantization, and context are bound.",
    "gpu_process_receipts": "Owned process, device, VRAM, CUDA layer offload, timing, and unload prove local GPU inference.",
    "prompt_and_decode_contract": "Direct prompt bytes, stop rules, sampling settings, and seed schedule are frozen before inference.",
    "raw_model_receipts": "Every raw response is retained before parsing or exact checking.",
    "failure_rows": "Timeouts, parse failures, refusals, invalid plans, and resource errors remain charged.",
    "family_headroom_summary": "Calibration and held exact success and failure modes recompute only from rows.",
    "gemma26_headroom_ready_score": "This binary field marks complete held success in the frozen 20 to 80 percent interval.",
    "attack_rows": "Drift, leakage, dropping, fallback, identity, regeneration, aggregate, and mutation attacks fail closed.",
    "preconditions_checked": "Upstream gate, files, hashes, GPU, cache, model, tokenizer, timeouts, and protected files are explicit.",
    "protected_files_unchanged": "Both protected orchestration files retain their original hashes.",
    "inference_substrate": "The task declares live local Gemma 4 26B-A4B GGUF inference through CUDA llama.cpp.",
    "verifier_is_oracle": "The independent exact executor defines plan success.",
    "field_provenance": "Every aggregate points to per-unit raw, model, process, and exact-check receipts.",
    "duration_s": "Monotonic duration is consistent with the declared live workload.",
    "tests_run": "Named validation commands include exits and durations.",
    "reproducibility_checksum": "A final content hash detects result mutation.",
}

FOCUSED_TEST_COMMAND = ".venv/bin/pytest tests/python/test_experiment_6607_gemma4_26b_direct_headroom.py -q --no-cov -n 0"
COVERAGE_RUN_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_6607_gemma4_26b_direct_headroom.py "
    "-m pytest tests/python/test_experiment_6607_gemma4_26b_direct_headroom.py "
    "-q --no-cov -n 0"
)
COVERAGE_REPORT_COMMAND = (
    ".venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_6607_gemma4_26b_direct_headroom.py "
    "--fail-under=100 --show-missing"
)
RUFF_CHECK_COMMAND = f".venv/bin/ruff check {MODULE_RELATIVE_PATH} {TEST_RELATIVE_PATH}"
RUFF_FORMAT_COMMAND = f".venv/bin/ruff format --check {MODULE_RELATIVE_PATH} {TEST_RELATIVE_PATH}"
SPEC_COVERAGE_COMMAND = f".venv/bin/python scripts/check_spec_coverage.py {TEST_RELATIVE_PATH}"

canonical_json = stream_helpers.canonical_json
sha256_bytes = stream_helpers.sha256_bytes
sha256_text = stream_helpers.sha256_text
sha256_json = stream_helpers.sha256_json
sha256_file = stream_helpers.sha256_file
row_hash = stream_helpers.row_hash
artifact_checksum = stream_helpers.artifact_checksum


def build_prompt_and_decode_contract(exp6604_artifact: Mapping[str, Any]) -> JsonDict:
    """Freeze every task prompt and all direct decoding choices."""

    task_rows = [
        dict(row)
        for row in exp6604_artifact.get("plan_fixture_rows", [])
        if isinstance(row, Mapping)
    ]
    prompts = []
    for row in task_rows:
        prompt_bytes = str(row.get("model_prompt_bytes", "")).encode("utf-8")
        source_bytes = str(row.get("source_bytes", "")).encode("utf-8")
        prompts.append(
            {
                "task_id": row.get("task_id"),
                "split": row.get("split"),
                "task_sha256": sha256_bytes(source_bytes),
                "prompt_bytes_b64": base64.b64encode(prompt_bytes).decode("ascii"),
                "prompt_sha256": sha256_bytes(prompt_bytes),
            }
        )
    split_hashes = {
        split: sha256_json([row["task_sha256"] for row in prompts if row.get("split") == split])
        for split in ("calibration", "held")
    }
    contract: JsonDict = {
        "schema": "carnot.experiment_6607.direct_prompt_decode_contract.v1",
        "source_artifact_schema": exp6604_artifact.get("schema"),
        "source_corpus_checksum": exp6604_artifact.get("reproducibility_checksum"),
        "task_count": len(prompts),
        "expected_row_count": len(prompts) * len(SEED_SCHEDULE),
        "task_order": [row["task_id"] for row in prompts],
        "task_prompts": prompts,
        "split_hashes": split_hashes,
        "chat_serialization": "llama_cpp_v1_chat_completions_embedded_gguf_template",
        "message_roles": ["user"],
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "top_k": TOP_K,
        "repeat_penalty": REPEAT_PENALTY,
        "context_size": CONTEXT_SIZE,
        "max_output_tokens": MAX_OUTPUT_TOKENS,
        "stop_rules": list(STOP_RULES),
        "seed_schedule": list(SEED_SCHEDULE),
        "per_generation_timeout_s": PER_GENERATION_TIMEOUT_S,
        "task_timeout_s": TASK_TIMEOUT_S,
        "grammar": None,
        "semantic_mask": None,
        "repair": False,
        "retry_count": 0,
        "cross_family_context": False,
        "prior_response_context": False,
        "outcome_conditioned_regeneration": False,
    }
    contract["contract_sha256"] = sha256_json(contract)
    return contract


def build_generation_jobs(
    exp6604_artifact: Mapping[str, Any],
    prompt_contract: Mapping[str, Any],
) -> list[JsonDict]:
    """Build the ordered Cartesian product of frozen tasks and three seeds."""

    prompt_by_task = {
        str(row["task_id"]): dict(row)
        for row in prompt_contract.get("task_prompts", [])
        if isinstance(row, Mapping)
    }
    jobs = []
    for fixture in exp6604_artifact.get("plan_fixture_rows", []):
        if not isinstance(fixture, Mapping):  # pragma: no cover - validated upstream shape.
            continue
        task_id = str(fixture.get("task_id"))
        prompt_row = prompt_by_task[task_id]
        source_bytes = str(fixture.get("source_bytes", "")).encode("utf-8")
        task = json.loads(source_bytes.decode("utf-8"))
        prompt_bytes = base64.b64decode(prompt_row["prompt_bytes_b64"], validate=True)
        for seed in SEED_SCHEDULE:
            jobs.append(
                {
                    "row_id": f"{task_id}|seed-{seed}",
                    "task_id": task_id,
                    "split": fixture.get("split"),
                    "seed": seed,
                    "task_sha256": sha256_bytes(source_bytes),
                    "task_source_bytes_b64": base64.b64encode(source_bytes).decode("ascii"),
                    "prompt_bytes": prompt_bytes,
                    "prompt_bytes_b64": prompt_row["prompt_bytes_b64"],
                    "prompt_sha256": prompt_row["prompt_sha256"],
                    "prompt_contract_sha256": prompt_contract.get("contract_sha256"),
                    "task": task,
                }
            )
    return jobs


def _looks_like_refusal(text: str) -> bool:
    lowered = text.casefold()
    return any(
        phrase in lowered
        for phrase in (
            "i cannot",
            "i can't",
            "i will not",
            "unable to provide",
            "cannot provide",
            "sorry",
        )
    )


def _score_raw_candidate(
    task: Mapping[str, Any],
    raw: bytes,
    *,
    failure_kind: str | None,
    finish_reason: str,
) -> JsonDict:
    """Classify one immutable response and call the exact executor once."""

    decoded: str | None
    try:
        decoded = raw.decode("utf-8", "strict")
    except UnicodeDecodeError:
        decoded = None
    executor_input = decoded if decoded is not None else ""
    exact_result = corpus_helpers.IndependentExactExecutor().execute(task, executor_input)
    if failure_kind == "timeout":
        failure_class = "timeout"
        parse_state = "timeout"
    elif failure_kind:
        failure_class = "process_failure"
        parse_state = "process_failure"
    elif finish_reason == "length":
        failure_class = "invalid_generation"
        parse_state = "bounded_output_truncated"
    elif decoded is None or not raw:
        failure_class = "invalid_generation"
        parse_state = "invalid_utf8" if decoded is None else "empty_response"
    elif _looks_like_refusal(decoded):
        failure_class = "refusal"
        parse_state = "refusal_detected"
    elif exact_result.get("valid") is True:
        failure_class = None
        parse_state = "parsed_canonical_candidate"
    elif exact_result.get("reason") == "syntax_error":
        failure_class = "syntax_failure"
        parse_state = "syntax_invalid"
    elif exact_result.get("reason") in {"precondition_violation", "ordering_violation"}:
        failure_class = "semantic_failure"
        parse_state = "syntax_valid_semantic_failure"
    elif exact_result.get("reason") == "unmet_goal":
        failure_class = "unmet_goal"
        parse_state = "syntax_valid_unmet_goal"
    else:  # pragma: no cover - the frozen executor has a closed reason set.
        failure_class = "invalid_generation"
        parse_state = "unknown_executor_rejection"
    return {
        "parsed_plan": decoded,
        "parse_state": parse_state,
        "exact_executor_result": exact_result,
        "exact_success": failure_class is None and exact_result.get("valid") is True,
        "failure_class": failure_class,
        "charged_failure": failure_class is not None,
        "failure_flags": {name: failure_class == name for name in FAILURE_CLASSES},
    }


def build_per_unit_row(
    job: Mapping[str, Any],
    generation: Mapping[str, Any],
    process_binding: Mapping[str, Any],
) -> JsonDict:
    """Seal raw bytes, then attach parse and exact-executor evidence."""

    raw = bytes(generation.get("raw_response_bytes", b""))
    score = _score_raw_candidate(
        job["task"],
        raw,
        failure_kind=(
            str(generation.get("failure_kind"))
            if generation.get("failure_kind") is not None
            else None
        ),
        finish_reason=str(generation.get("finish_reason", "unknown")),
    )
    row: JsonDict = {
        "schema": "carnot.experiment_6607.direct_plan_row.v1",
        "row_id": job.get("row_id"),
        "task_id": job.get("task_id"),
        "split": job.get("split"),
        "seed": int(job.get("seed", 0)),
        "task_sha256": job.get("task_sha256"),
        "task_source_bytes_b64": job.get("task_source_bytes_b64"),
        "prompt_bytes_b64": job.get("prompt_bytes_b64"),
        "prompt_sha256": job.get("prompt_sha256"),
        "prompt_contract_sha256": job.get("prompt_contract_sha256"),
        "prompt_contains_split_outcome": False,
        "raw_response_bytes_b64": base64.b64encode(raw).decode("ascii"),
        "raw_response_byte_count": len(raw),
        "raw_response_sha256": sha256_bytes(raw),
        "raw_api_response_sha256": generation.get("raw_api_response_sha256"),
        "raw_recorded_before_parse": True,
        **score,
        "exact_executor_call_count": 1,
        "model_process": dict(process_binding),
        "timing": {
            "started_monotonic_ns": int(generation.get("started_monotonic_ns", 0)),
            "finished_monotonic_ns": int(generation.get("finished_monotonic_ns", 0)),
            "latency_s": round(float(generation.get("latency_s", 0.0)), 9),
        },
        "token_count": {
            "prompt": int(generation.get("prompt_tokens", 0)),
            "completion": int(generation.get("completion_tokens", 0)),
            "total": int(generation.get("prompt_tokens", 0))
            + int(generation.get("completion_tokens", 0)),
        },
        "finish_reason": str(generation.get("finish_reason", "unknown")),
        "http_status": int(generation.get("http_status", 0)),
        "attempt_count": 1,
        "regeneration_count": 0,
        "response_regenerated": False,
    }
    row["row_hash"] = row_hash(row)
    return row


def _wilson_interval(successes: int, total: int) -> list[float]:
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


def family_headroom_summary(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Recompute split success and exclusive failure rates from rows."""

    summary: JsonDict = {}
    for split in ("calibration", "held"):
        split_rows = [row for row in rows if row.get("split") == split]
        total = len(split_rows)
        successes = sum(row.get("exact_success") is True for row in split_rows)
        failure_counts = Counter(
            str(row.get("failure_class"))
            for row in split_rows
            if row.get("failure_class") in FAILURE_CLASSES
        )
        summary[split] = {
            "row_count": total,
            "exact_success_count": successes,
            "exact_success_rate": round(successes / total, 9) if total else 0.0,
            "exact_success_interval_95": _wilson_interval(successes, total),
            "failure_counts": {name: int(failure_counts.get(name, 0)) for name in FAILURE_CLASSES},
            "failure_rates": {
                name: round(failure_counts.get(name, 0) / total, 9) if total else 0.0
                for name in FAILURE_CLASSES
            },
            "charged_failure_count": total - successes,
            "charged_failure_rate": round((total - successes) / total, 9) if total else 0.0,
        }
    summary["reducer"] = "exclusive row failure_class and exact_success fields"
    return summary


def gemma26_headroom_ready_score(
    rows_complete: bool,
    held_success_count: int,
    held_row_count: int,
) -> float:
    """Return one only for complete rows and held success in the closed band."""

    if not rows_complete or held_row_count <= 0:
        return 0.0
    rate = held_success_count / held_row_count
    return 1.0 if 0.20 <= rate <= 0.80 else 0.0


def raw_model_receipts(rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Flatten retained raw bytes without replacing them with parser output."""

    return [
        {
            "row_id": row.get("row_id"),
            "raw_response_bytes_b64": row.get("raw_response_bytes_b64"),
            "raw_response_byte_count": row.get("raw_response_byte_count"),
            "raw_response_sha256": row.get("raw_response_sha256"),
            "raw_api_response_sha256": row.get("raw_api_response_sha256"),
            "raw_recorded_before_parse": row.get("raw_recorded_before_parse"),
            "row_hash": row.get("row_hash"),
        }
        for row in rows
    ]


def failure_rows(rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Retain every charged row with its immutable response and exact receipt."""

    return [
        {
            "row_id": row.get("row_id"),
            "task_id": row.get("task_id"),
            "split": row.get("split"),
            "seed": row.get("seed"),
            "failure_class": row.get("failure_class"),
            "failure_flags": deepcopy(row.get("failure_flags", {})),
            "raw_response_sha256": row.get("raw_response_sha256"),
            "exact_reason": row.get("exact_executor_result", {}).get("reason"),
            "row_hash": row.get("row_hash"),
        }
        for row in rows
        if row.get("charged_failure") is True
    ]


def _model_identity_ready(identity: Mapping[str, Any]) -> bool:
    specs = identity.get("MODEL_SPECS")
    tokenizer = identity.get("embedded_tokenizer", {})
    template = identity.get("embedded_chat_template", {})
    shards = identity.get("gguf_shards", [])
    return bool(
        specs == MODEL_SPECS
        and identity.get("hub_id") == GEMMA26_HUB_ID
        and identity.get("model_path") == MODEL_SPECS[0]["model_path"]
        and str(identity.get("model_sha256", "")).startswith("sha256:")
        and shards
        and all(
            isinstance(row, Mapping)
            and row.get("path")
            and row.get("sha256") == identity.get("model_sha256")
            and int(row.get("byte_count", 0)) > 0
            for row in shards
        )
        and "Q4" in str(identity.get("quantization", "")).upper()
        and identity.get("architecture") == GEMMA26_ARCHITECTURE
        and tokenizer.get("source") == "embedded_gguf"
        and tokenizer.get("loadable") is True
        and int(tokenizer.get("token_count", 0)) > 0
        and str(tokenizer.get("identity_sha256", "")).startswith("sha256:")
        and template.get("source") == "tokenizer.chat_template"
        and template.get("present") is True
        and str(template.get("sha256", "")).startswith("sha256:")
        and identity.get("llama_cpp", {}).get("cuda_linked") is True
        and identity.get("auto_tokenizer_used") is False
        and identity.get("download_performed") is False
        and identity.get("legacy_headline_row_count") == 0
    )


def _gpu_receipts_ready(receipts: Mapping[str, Any], expected_rows: int) -> bool:
    sessions = [row for row in receipts.get("sessions", []) if isinstance(row, Mapping)]
    if not sessions or sum(int(row.get("row_count", 0)) for row in sessions) != expected_rows:
        return False
    for session in sessions:
        samples = [row for row in session.get("samples", []) if isinstance(row, Mapping)]
        before = [row for row in samples if row.get("stage") == "before"]
        during = [row for row in samples if row.get("stage") == "during"]
        after = [row for row in samples if row.get("stage") == "after"]
        if not (
            session.get("owned_child") is True
            and session.get("repository_id") == GEMMA26_HUB_ID
            and session.get("model_sha256")
            and session.get("selected_gpu") == MODEL_SPECS[0]["gpu"]
            and str(session.get("cuda_visible_devices")) == str(MODEL_SPECS[0]["gpu"])
            and session.get("cpu_fallback") is False
            and session.get("cuda_offload") is True
            and int(session.get("offloaded_layers", 0)) > 0
            and session.get("server_healthy") is True
            and before
            and len(during) >= 2
            and after
            and all(row.get("worker_pid_present") is True for row in during)
            and session.get("shutdown_requested") is True
            and session.get("normal_shutdown") is True
            and session.get("worker_absent_after_exit") is True
            and session.get("port_closed") is True
            and session.get("memory_recovered") is True
            and session.get("signals_sent_to_unrelated_pids") == []
        ):
            return False
    return receipts.get("all_sessions_authentic") is True


def _row_ready(
    row: Mapping[str, Any],
    expected_job: Mapping[str, Any],
    model_identity: Mapping[str, Any],
    sessions: Mapping[str, Mapping[str, Any]],
) -> bool:
    try:
        prompt = base64.b64decode(str(row.get("prompt_bytes_b64", "")), validate=True)
        raw = base64.b64decode(str(row.get("raw_response_bytes_b64", "")), validate=True)
        source = base64.b64decode(str(row.get("task_source_bytes_b64", "")), validate=True)
        task = json.loads(source.decode("utf-8"))
    except (TypeError, ValueError, UnicodeDecodeError, json.JSONDecodeError):
        return False
    process = row.get("model_process", {})
    session = sessions.get(str(process.get("session_id")))
    score = _score_raw_candidate(
        task,
        raw,
        failure_kind=(
            "timeout"
            if row.get("failure_class") == "timeout"
            else "process_failure"
            if row.get("failure_class") == "process_failure"
            else None
        ),
        finish_reason=str(row.get("finish_reason")),
    )
    return bool(
        row.get("row_id") == expected_job.get("row_id")
        and row.get("task_id") == expected_job.get("task_id")
        and row.get("split") == expected_job.get("split")
        and row.get("seed") == expected_job.get("seed")
        and row.get("task_sha256") == sha256_bytes(source)
        and row.get("task_sha256") == expected_job.get("task_sha256")
        and row.get("prompt_sha256") == sha256_bytes(prompt)
        and row.get("prompt_sha256") == expected_job.get("prompt_sha256")
        and row.get("prompt_contract_sha256") == expected_job.get("prompt_contract_sha256")
        and row.get("prompt_contains_split_outcome") is False
        and row.get("raw_response_sha256") == sha256_bytes(raw)
        and row.get("raw_response_byte_count") == len(raw)
        and row.get("raw_recorded_before_parse") is True
        and row.get("parsed_plan") == score["parsed_plan"]
        and row.get("parse_state") == score["parse_state"]
        and row.get("exact_executor_result") == score["exact_executor_result"]
        and row.get("exact_success") == score["exact_success"]
        and row.get("failure_class") == score["failure_class"]
        and row.get("charged_failure") == score["charged_failure"]
        and row.get("failure_flags") == score["failure_flags"]
        and row.get("exact_executor_call_count") == 1
        and row.get("attempt_count") == 1
        and row.get("regeneration_count") == 0
        and row.get("response_regenerated") is False
        and isinstance(session, Mapping)
        and process.get("pid") == session.get("pid")
        and process.get("owned_child") is True
        and process.get("repository_id") == GEMMA26_HUB_ID
        and process.get("model_sha256") == model_identity.get("model_sha256")
        and process.get("selected_gpu") == MODEL_SPECS[0]["gpu"]
        and process.get("cpu_fallback") is False
        and process.get("cuda_offload") is True
        and int(process.get("offloaded_layers", 0)) > 0
        and process.get("tokenizer_source") == "embedded_gguf"
        and process.get("chat_template_sha256")
        == model_identity.get("embedded_chat_template", {}).get("sha256")
        and row.get("token_count", {}).get("total")
        == int(row.get("token_count", {}).get("prompt", 0))
        + int(row.get("token_count", {}).get("completion", 0))
        and float(row.get("timing", {}).get("latency_s", -1.0)) >= 0.0
        and row.get("row_hash") == row_hash(row)
    )


def integrity_reducer(
    payload: Mapping[str, Any],
    *,
    require_attack_rows: bool = True,
) -> JsonDict:
    """Recompute baseline completeness without using the headroom outcome."""

    contract = payload.get("prompt_and_decode_contract", {})
    rows = [row for row in payload.get("per_unit_rows", []) if isinstance(row, Mapping)]
    expected_ids = [
        f"{task_id}|seed-{seed}"
        for task_id in contract.get("task_order", [])
        for seed in SEED_SCHEDULE
    ]
    prompt_rows = {
        str(row.get("task_id")): row
        for row in contract.get("task_prompts", [])
        if isinstance(row, Mapping)
    }
    expected_jobs = []
    source_by_task = {str(row.get("task_id")): row for row in rows if isinstance(row, Mapping)}
    for row_id in expected_ids:
        task_id, seed_text = row_id.rsplit("|seed-", 1)
        source_row = source_by_task.get(task_id, {})
        expected_jobs.append(
            {
                "row_id": row_id,
                "task_id": task_id,
                "split": prompt_rows.get(task_id, {}).get("split"),
                "seed": int(seed_text),
                "task_sha256": prompt_rows.get(task_id, {}).get("task_sha256"),
                "prompt_sha256": prompt_rows.get(task_id, {}).get("prompt_sha256"),
                "prompt_contract_sha256": contract.get("contract_sha256"),
                "task_source_bytes_b64": source_row.get("task_source_bytes_b64"),
            }
        )
    sessions = {
        str(row.get("session_id")): row
        for row in payload.get("gpu_process_receipts", {}).get("sessions", [])
        if isinstance(row, Mapping)
    }
    model_identity = payload.get("model_spec_and_identity", {})
    expected_summary = family_headroom_summary(rows)
    attacks = [row for row in payload.get("attack_rows", []) if isinstance(row, Mapping)]
    checkpoint = payload.get("checkpoint_receipts", {})
    contract_without_hash = {
        key: value for key, value in contract.items() if key != "contract_sha256"
    }
    checks = {
        "prompt_contract": contract.get("contract_sha256") == sha256_json(contract_without_hash)
        and contract.get("task_count") == EXPECTED_TASK_COUNT
        and contract.get("expected_row_count") == EXPECTED_ROW_COUNT
        and contract.get("seed_schedule") == list(SEED_SCHEDULE)
        and contract.get("grammar") is None
        and contract.get("semantic_mask") is None
        and contract.get("repair") is False
        and contract.get("cross_family_context") is False
        and contract.get("outcome_conditioned_regeneration") is False,
        "complete_row_order": len(rows) == EXPECTED_ROW_COUNT
        and [row.get("row_id") for row in rows] == expected_ids,
        "model_identity": _model_identity_ready(model_identity),
        "gpu_process_receipts": _gpu_receipts_ready(
            payload.get("gpu_process_receipts", {}), len(rows)
        ),
        "authentic_rows": len(rows) == len(expected_jobs)
        and all(
            _row_ready(row, job, model_identity, sessions)
            for row, job in zip(rows, expected_jobs, strict=True)
        ),
        "raw_model_receipts": payload.get("raw_model_receipts") == raw_model_receipts(rows),
        "failure_rows": payload.get("failure_rows") == failure_rows(rows),
        "family_headroom_summary": payload.get("family_headroom_summary") == expected_summary,
        "checkpoint": checkpoint.get("accepted_row_count") == len(rows)
        and checkpoint.get("completed_prefix_hash") == sha256_json(rows)
        and checkpoint.get("prompt_contract_sha256") == contract.get("contract_sha256")
        and checkpoint.get("model_sha256") == model_identity.get("model_sha256")
        and checkpoint.get("atomic_replace") is True
        and checkpoint.get("directory_fsync") is True,
        "preconditions": payload.get("preconditions_checked", {}).get(
            "all_required_preconditions_available"
        )
        is True,
        "protected_files": payload.get("protected_files_unchanged", {}).get("all_unchanged")
        is True,
        "tests": bool(payload.get("tests_run"))
        and all(row.get("exit_code") == 0 for row in payload.get("tests_run", [])),
        "substrate": payload.get("inference_substrate") == INFERENCE_SUBSTRATE,
        "oracle": payload.get("verifier_is_oracle") is True,
        "attacks": (
            {row.get("attack_id") for row in attacks} == set(REQUIRED_ATTACK_IDS)
            and all(
                row.get("candidate_integrity_ready_score") == 0.0
                and row.get("failed_closed") is True
                for row in attacks
            )
        )
        if require_attack_rows
        else True,
    }
    return {
        "complete": all(checks.values()),
        "checks": checks,
        "failed_checks": [name for name, passed in checks.items() if not passed],
        "expected_row_count": EXPECTED_ROW_COUNT,
        "observed_row_count": len(rows),
        "reducer": "conjunction of frozen prompts, authentic rows, local identity, CUDA sessions, checkpoints, aggregates, attacks, and protected hashes",
    }


def _rehash_first_row(payload: JsonDict) -> JsonDict:
    payload["per_unit_rows"][0]["row_hash"] = row_hash(payload["per_unit_rows"][0])
    return payload


def build_attack_rows(base_payload: Mapping[str, Any]) -> list[JsonDict]:
    """Apply the registered mutations and prove that integrity becomes zero."""

    def cross_family_tuning(value: JsonDict) -> None:
        contract = value["prompt_and_decode_contract"]
        contract["cross_family_context"] = True
        contract["contract_sha256"] = sha256_json(
            {key: item for key, item in contract.items() if key != "contract_sha256"}
        )

    def prompt_drift(value: JsonDict) -> None:
        value["per_unit_rows"][0]["prompt_sha256"] = "sha256:drift"
        _rehash_first_row(value)

    def seed_drift(value: JsonDict) -> None:
        value["per_unit_rows"][0]["seed"] += 1
        _rehash_first_row(value)

    def split_leakage(value: JsonDict) -> None:
        value["per_unit_rows"][0]["prompt_contains_split_outcome"] = True
        _rehash_first_row(value)

    def omitted_failures(value: JsonDict) -> None:
        value["failure_rows"] = []

    def cpu_fallback(value: JsonDict) -> None:
        value["per_unit_rows"][0]["model_process"]["cpu_fallback"] = True
        _rehash_first_row(value)

    def fake_cuda(value: JsonDict) -> None:
        value["per_unit_rows"][0]["model_process"]["offloaded_layers"] = 0
        _rehash_first_row(value)

    def wrong_model(value: JsonDict) -> None:
        value["model_spec_and_identity"]["hub_id"] = "Qwen/Qwen3.5-0.8B"

    def tokenizer_substitution(value: JsonDict) -> None:
        value["model_spec_and_identity"]["embedded_tokenizer"]["source"] = (
            "transformers.AutoTokenizer"
        )

    def response_regeneration(value: JsonDict) -> None:
        value["per_unit_rows"][0]["regeneration_count"] = 1
        value["per_unit_rows"][0]["response_regenerated"] = True
        _rehash_first_row(value)

    def aggregate_disagreement(value: JsonDict) -> None:
        value["family_headroom_summary"]["held"]["exact_success_count"] += 1

    def protected_mutation(value: JsonDict) -> None:
        value["protected_files_unchanged"]["all_unchanged"] = False

    mutations = {
        "cross_family_tuning": cross_family_tuning,
        "prompt_drift": prompt_drift,
        "seed_drift": seed_drift,
        "split_leakage": split_leakage,
        "omitted_failures": omitted_failures,
        "cpu_fallback": cpu_fallback,
        "fake_cuda_offload": fake_cuda,
        "wrong_model": wrong_model,
        "tokenizer_substitution": tokenizer_substitution,
        "response_regeneration": response_regeneration,
        "aggregate_disagreement": aggregate_disagreement,
        "protected_file_mutation": protected_mutation,
    }
    rows = []
    for attack_id in REQUIRED_ATTACK_IDS:
        candidate = deepcopy(base_payload)
        mutations[attack_id](candidate)
        reduction = integrity_reducer(candidate, require_attack_rows=False)
        score = 1.0 if reduction["complete"] else 0.0
        rows.append(
            {
                "attack_id": attack_id,
                "candidate_integrity_ready_score": score,
                "expected_integrity_ready_score": 0.0,
                "failed_checks": reduction["failed_checks"],
                "failed_closed": score == 0.0,
            }
        )
    return rows


def _field_provenance() -> dict[str, JsonDict]:
    """Map every required field to raw receipts and its reducer."""

    return {
        field: {
            "principle": FIELD_PRINCIPLES[field],
            "raw_sources": [
                "Exp6604 frozen task and prompt bytes",
                "per_unit_rows raw response bytes",
                "Gemma 4 26B-A4B GGUF identity and GPU process sessions",
                "IndependentExactExecutor results",
            ],
            "reducer": (
                "family_headroom_summary"
                if field in {"family_headroom_summary", "gemma26_headroom_ready_score"}
                else "integrity_reducer"
                if field in {"failure_rows", "raw_model_receipts", "attack_rows"}
                else "direct immutable receipt"
            ),
        }
        for field in REQUIRED_ARTIFACT_FIELDS
    }


def assemble_artifact(
    *,
    run_date: str,
    exp6604_artifact: Mapping[str, Any],
    prompt_contract: Mapping[str, Any],
    per_unit_rows: Sequence[Mapping[str, Any]],
    model_identity: Mapping[str, Any],
    gpu_receipts: Mapping[str, Any],
    checkpoint_receipts: Mapping[str, Any],
    preconditions: Mapping[str, Any],
    protected_files: Mapping[str, Any],
    tests_run: Sequence[Mapping[str, Any]],
    duration_s: float,
) -> JsonDict:
    """Assemble the terminal artifact and keep completeness outcome-blind."""

    rows = [dict(row) for row in per_unit_rows]
    summary = family_headroom_summary(rows)
    payload: JsonDict = {
        "schema": "carnot.experiment_6607.gemma4_26b_direct_headroom.v1",
        "run_date": run_date,
        "status": "assembling",
        "honest_verdict": "blocked_assembling",
        "verdict_class": "blocked",
        "gate_check_summary": {
            "all_passed": False,
            "failed_condition": "assembling",
            "expected": True,
            "observed": False,
            "upstream": {
                "path": EXP6604_RELATIVE_PATH.as_posix(),
                "headroom_fixture_ready_score": exp6604_artifact.get(
                    "headroom_fixture_ready_score"
                ),
                "task_count": len(exp6604_artifact.get("plan_fixture_rows", [])),
            },
        },
        "per_unit_rows": rows,
        "model_spec_and_identity": dict(model_identity),
        "gpu_process_receipts": dict(gpu_receipts),
        "prompt_and_decode_contract": dict(prompt_contract),
        "raw_model_receipts": raw_model_receipts(rows),
        "failure_rows": failure_rows(rows),
        "family_headroom_summary": summary,
        "gemma26_headroom_ready_score": 0.0,
        "attack_rows": [],
        "preconditions_checked": dict(preconditions),
        "protected_files_unchanged": dict(protected_files),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": True,
        "field_provenance": _field_provenance(),
        "duration_s": round(float(duration_s), 6),
        "tests_run": [dict(row) for row in tests_run],
        "checkpoint_receipts": dict(checkpoint_receipts),
        "reproducibility_checksum": "",
    }
    payload["attack_rows"] = build_attack_rows(payload)
    reduction = integrity_reducer(payload)
    complete = reduction["complete"]
    held = summary["held"]
    ready = gemma26_headroom_ready_score(
        complete,
        int(held["exact_success_count"]),
        int(held["row_count"]),
    )
    payload["gemma26_headroom_ready_score"] = ready
    payload["integrity_recomputation"] = reduction
    if complete:
        rate = float(held["exact_success_rate"])
        payload["status"] = "complete"
        payload["verdict_class"] = "null"
        if ready == 1.0:
            payload["honest_verdict"] = (
                f"complete: all {len(rows)} Gemma 4 26B-A4B direct rows are retained and held "
                f"exact success is {rate:.6f}, inside the frozen headroom interval; "
                "baseline qualification is null infrastructure"
            )
        else:
            payload["honest_verdict"] = (
                f"complete: all {len(rows)} Gemma 4 26B-A4B direct rows are retained but held "
                f"exact success is {rate:.6f}, outside the frozen headroom interval; "
                "no useful headroom qualified"
            )
        payload["gate_check_summary"].update(
            {
                "all_passed": True,
                "failed_condition": None,
                "expected": True,
                "observed": True,
            }
        )
    else:
        failed = reduction["failed_checks"][0] if reduction["failed_checks"] else "integrity"
        payload["status"] = f"blocked_{failed}"
        payload["honest_verdict"] = f"blocked_{failed}: direct baseline integrity did not complete"
        payload["gate_check_summary"].update(
            {
                "all_passed": False,
                "failed_condition": failed,
                "expected": True,
                "observed": reduction["checks"].get(failed),
            }
        )
    payload["reproducibility_checksum"] = artifact_checksum(payload)
    return payload


def build_blocked_artifact(
    *,
    run_date: str,
    failed_condition: str,
    expected: Any,
    observed: Any,
    model_identity: Mapping[str, Any],
    preconditions: Mapping[str, Any],
    protected_files: Mapping[str, Any],
    tests_run: Sequence[Mapping[str, Any]],
    duration_s: float,
) -> JsonDict:
    """Write all required fields when a live prerequisite blocks inference."""

    payload: JsonDict = {
        "schema": "carnot.experiment_6607.gemma4_26b_direct_headroom.v1",
        "run_date": run_date,
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
        "per_unit_rows": [],
        "model_spec_and_identity": dict(model_identity),
        "gpu_process_receipts": {"sessions": [], "all_sessions_authentic": False},
        "prompt_and_decode_contract": {},
        "raw_model_receipts": [],
        "failure_rows": [],
        "family_headroom_summary": family_headroom_summary([]),
        "gemma26_headroom_ready_score": 0.0,
        "attack_rows": [
            {
                "attack_id": attack_id,
                "candidate_integrity_ready_score": 0.0,
                "expected_integrity_ready_score": 0.0,
                "failed_checks": ["blocked_precondition"],
                "failed_closed": True,
                "not_run_due_to_block": True,
            }
            for attack_id in REQUIRED_ATTACK_IDS
        ],
        "preconditions_checked": dict(preconditions),
        "protected_files_unchanged": dict(protected_files),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": True,
        "field_provenance": _field_provenance(),
        "duration_s": round(float(duration_s), 6),
        "tests_run": [dict(row) for row in tests_run],
        "checkpoint_receipts": {},
        "reproducibility_checksum": "",
    }
    payload["reproducibility_checksum"] = artifact_checksum(payload)
    return payload


def validate_artifact(payload: Mapping[str, Any]) -> list[str]:
    """Reject schema, identity, row, aggregate, verdict, or checksum drift."""

    missing = sorted(set(REQUIRED_ARTIFACT_FIELDS) - set(payload))
    if missing:
        return ["missing_required_fields:" + ",".join(missing)]
    errors = []
    if payload.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate_mismatch")
    if payload.get("verifier_is_oracle") is not True:
        errors.append("verifier_is_oracle_mismatch")
    if payload.get("verdict_class") not in {
        "positive",
        "circular_positive",
        "null",
        "blocked",
        "disqualified",
        "partial",
    }:
        errors.append("verdict_class_invalid")
    if set(payload.get("field_provenance", {})) != set(REQUIRED_ARTIFACT_FIELDS):
        errors.append("field_provenance_mismatch")
    if payload.get("status") == "complete":
        reduction = integrity_reducer(payload)
        if not reduction["complete"]:
            if reduction["checks"].get("model_identity") is False:
                errors.append("model_identity_mismatch")
            if reduction["checks"].get("family_headroom_summary") is False:
                errors.append("family_headroom_summary_mismatch")
            errors.append("integrity_recomputation_failed")
        summary = family_headroom_summary(payload.get("per_unit_rows", []))
        expected_ready = gemma26_headroom_ready_score(
            reduction["complete"],
            int(summary["held"]["exact_success_count"]),
            int(summary["held"]["row_count"]),
        )
        if payload.get("gemma26_headroom_ready_score") != expected_ready:
            errors.append("gemma26_headroom_ready_score_mismatch")
        if payload.get("verdict_class") != "null":
            errors.append("complete_baseline_verdict_class_mismatch")
    else:
        if not str(payload.get("status", "")).startswith("blocked_"):
            errors.append("blocked_status_prefix_missing")
        if not str(payload.get("honest_verdict", "")).startswith("blocked_"):
            errors.append("blocked_verdict_prefix_missing")
        if payload.get("verdict_class") != "blocked":
            errors.append("blocked_verdict_class_mismatch")
        if payload.get("gate_check_summary", {}).get("failed_condition") is None:
            errors.append("blocked_gate_condition_missing")
        if payload.get("gemma26_headroom_ready_score") != 0.0:
            errors.append("blocked_ready_score_nonzero")
    if payload.get("reproducibility_checksum") != artifact_checksum(payload):
        errors.append("reproducibility_checksum_mismatch")
    return errors


def gpu_ownership_receipt(sample: Mapping[str, Any], *, controller_pid: int) -> JsonDict:
    """Scope the ownership gate to the selected GPU and this run's controller."""

    selected_uuid = sample.get("device", {}).get("uuid")
    selected = [
        dict(row)
        for row in sample.get("compute_processes", [])
        if isinstance(row, Mapping) and row.get("gpu_uuid") == selected_uuid
    ]
    foreign = [row for row in selected if int(row.get("pid", 0) or 0) != controller_pid]
    return {
        "available": not foreign,
        "controller_pid": controller_pid,
        "selected_gpu_uuid": selected_uuid,
        "selected_gpu_processes": selected,
        "foreign_selected_gpu_processes": foreign,
    }


def _atomic_write_json(path: Path, payload: Mapping[str, Any], prefix: str) -> JsonDict:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode(
        "utf-8"
    )
    with tempfile.NamedTemporaryFile(dir=target.parent, prefix=prefix, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, target)
    directory_fd = os.open(target.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    return {
        "path": str(target.resolve()),
        "sha256": sha256_file(target),
        "byte_count": len(encoded),
        "atomic_replace": True,
        "directory_fsync": True,
    }


def atomic_write_artifact(path: Path, payload: Mapping[str, Any]) -> JsonDict:
    """Validate and atomically replace one terminal artifact."""

    errors = validate_artifact(payload)
    if errors:  # pragma: no cover - callers test validation before writing.
        raise ValueError(";".join(errors))
    return _atomic_write_json(Path(path), payload, ".exp6607-final-")


def atomic_write_checkpoint(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    prompt_contract_sha256: str,
    exp6604_artifact_sha256: str,
    model_sha256: str,
    process_sessions: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Atomically preserve a complete raw-row prefix for exact resume."""

    payload: JsonDict = {
        "schema": "carnot.experiment_6607.checkpoint.v1",
        "prompt_contract_sha256": prompt_contract_sha256,
        "exp6604_artifact_sha256": exp6604_artifact_sha256,
        "model_sha256": model_sha256,
        "completed_row_count": len(rows),
        "completed_row_ids": [row.get("row_id") for row in rows],
        "completed_row_hashes": [row.get("row_hash") for row in rows],
        "completed_prefix_hash": sha256_json(list(rows)),
        "rows": [dict(row) for row in rows],
        "process_sessions": [dict(row) for row in process_sessions],
        "checkpoint_checksum": "",
    }
    payload["checkpoint_checksum"] = sha256_json(
        {key: value for key, value in payload.items() if key != "checkpoint_checksum"}
    )
    receipt = _atomic_write_json(Path(path), payload, ".exp6607-checkpoint-")
    return {
        **receipt,
        "accepted_row_count": len(rows),
        "completed_prefix_hash": payload["completed_prefix_hash"],
        "prompt_contract_sha256": prompt_contract_sha256,
        "model_sha256": model_sha256,
    }


def load_checkpoint(
    path: Path,
    *,
    expected_row_ids: Sequence[str],
    prompt_contract_sha256: str,
    exp6604_artifact_sha256: str,
    model_sha256: str,
) -> JsonDict:
    """Accept only an exact ordered prefix from the same frozen contracts."""

    target = Path(path)
    if not target.is_file():
        return {
            "accepted": True,
            "completed_row_count": 0,
            "rows": [],
            "process_sessions": [],
            "gate_check_summary": {"all_passed": True, "failed_condition": None},
        }
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):  # pragma: no cover - host corruption path.
        return {
            "accepted": False,
            "completed_row_count": 0,
            "rows": [],
            "process_sessions": [],
            "gate_check_summary": {
                "all_passed": False,
                "failed_condition": "checkpoint_json",
            },
        }
    checks = (
        ("prompt_contract_sha256", prompt_contract_sha256),
        ("exp6604_artifact_sha256", exp6604_artifact_sha256),
        ("model_sha256", model_sha256),
    )
    for field, expected in checks:
        if payload.get(field) != expected:
            return {
                "accepted": False,
                "completed_row_count": 0,
                "rows": [],
                "process_sessions": [],
                "gate_check_summary": {
                    "all_passed": False,
                    "failed_condition": field,
                    "expected": expected,
                    "observed": payload.get(field),
                },
            }
    rows = [row for row in payload.get("rows", []) if isinstance(row, Mapping)]
    count = len(rows)
    checksum = sha256_json(
        {key: value for key, value in payload.items() if key != "checkpoint_checksum"}
    )
    prefix_ready = (
        payload.get("completed_row_count") == count
        and payload.get("completed_row_ids") == list(expected_row_ids[:count])
        and payload.get("completed_row_hashes") == [row.get("row_hash") for row in rows]
        and payload.get("completed_prefix_hash") == sha256_json(rows)
        and payload.get("checkpoint_checksum") == checksum
        and all(row.get("row_hash") == row_hash(row) for row in rows)
    )
    if not prefix_ready:  # pragma: no cover - focused tamper tests use final validator.
        return {
            "accepted": False,
            "completed_row_count": 0,
            "rows": [],
            "process_sessions": [],
            "gate_check_summary": {
                "all_passed": False,
                "failed_condition": "checkpoint_prefix",
            },
        }
    return {
        "accepted": True,
        "completed_row_count": count,
        "rows": [dict(row) for row in rows],
        "process_sessions": [
            dict(row) for row in payload.get("process_sessions", []) if isinstance(row, Mapping)
        ],
        "gate_check_summary": {"all_passed": True, "failed_condition": None},
    }


def _utc_now() -> str:  # pragma: no cover - live receipt.
    return datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _run_command(
    command: Sequence[str], repo_root: Path, timeout_s: float
) -> JsonDict:  # pragma: no cover - live validation receipt.
    started = time.monotonic()
    try:
        result = subprocess.run(
            list(command),
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
        return {
            "command": " ".join(command),
            "exit_code": result.returncode,
            "duration_s": round(time.monotonic() - started, 6),
            "stdout": result.stdout[-8000:],
            "stderr": result.stderr[-8000:],
            "stdout_sha256": sha256_text(result.stdout),
            "stderr_sha256": sha256_text(result.stderr),
        }
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {
            "command": " ".join(command),
            "exit_code": 124 if isinstance(exc, subprocess.TimeoutExpired) else 127,
            "duration_s": round(time.monotonic() - started, 6),
            "stdout": "",
            "stderr": f"{type(exc).__name__}: {exc}",
            "stdout_sha256": sha256_text(""),
            "stderr_sha256": sha256_text(str(exc)),
        }


def _checkpoint_tests(repo_root: Path) -> list[JsonDict]:  # pragma: no cover
    commands = (
        (FOCUSED_TEST_COMMAND, 300.0),
        (COVERAGE_RUN_COMMAND, 300.0),
        (COVERAGE_REPORT_COMMAND, 60.0),
        (RUFF_CHECK_COMMAND, 60.0),
        (RUFF_FORMAT_COMMAND, 60.0),
        (SPEC_COVERAGE_COMMAND, 60.0),
    )
    return [_run_command(command.split(), repo_root, timeout) for command, timeout in commands]


def _protected_hashes(repo_root: Path) -> dict[str, str]:  # pragma: no cover
    return {path.as_posix(): sha256_file(repo_root / path) for path in PROTECTED_RELATIVE_PATHS}


def _protected_receipt(
    before: Mapping[str, str], after: Mapping[str, str]
) -> JsonDict:  # pragma: no cover
    rows = [
        {
            "path": path,
            "before_sha256": before.get(path),
            "after_sha256": after.get(path),
            "unchanged": before.get(path) == after.get(path),
        }
        for path in sorted(set(before) | set(after))
    ]
    return {
        "all_unchanged": bool(rows) and all(row["unchanged"] for row in rows),
        "rows": rows,
    }


def _host_resources(repo_root: Path) -> JsonDict:  # pragma: no cover
    memory: dict[str, int] = {}
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            key, value = line.split(":", 1)
            memory[key] = int(value.strip().split()[0])
    except (OSError, ValueError):
        pass
    disk = shutil.disk_usage(repo_root)
    return {
        "cpu": {
            "count": os.cpu_count(),
            "architecture": platform.machine(),
            "processor": platform.processor(),
        },
        "ram": {
            "total_kib": memory.get("MemTotal"),
            "available_kib": memory.get("MemAvailable"),
        },
        "disk": {
            "total_bytes": disk.total,
            "used_bytes": disk.used,
            "free_bytes": disk.free,
        },
    }


def _resolve_model_identity(gpu: int) -> JsonDict:  # pragma: no cover
    model_path = MODEL_SPECS[0]["model_path"]
    server = Path.home() / ".cache/llama.cpp-master/build/bin/llama-server"
    metadata = gemma_helpers._resolve_metadata_receipt()  # noqa: SLF001
    tokenizer = (
        admission.read_embedded_tokenizer_metadata(model_path)
        if model_path
        else {"embedded_tokenizer_ok": False, "error": "model_path_missing"}
    )
    full_hash = sha256_file(model_path) if model_path else "missing"
    provenance = metadata.get("provenance", {})
    ordered_shards = provenance.get("ordered_shards", []) if isinstance(provenance, Mapping) else []
    shard_rows = []
    for row in ordered_shards:
        path = Path(str(row.get("snapshot_path", "")))
        shard_rows.append(
            {
                "path": str(path),
                "resolved_path": str(path.resolve()) if path.exists() else "",
                "sha256": sha256_file(path),
                "byte_count": path.stat().st_size if path.is_file() else 0,
                "shard_number": row.get("shard_number"),
                "shard_count": row.get("shard_count"),
            }
        )
    content = metadata.get("content_metadata", {})
    build = runtime_helpers._llama_cpp_build_receipt(server)  # noqa: SLF001
    tokenizer_identity = sha256_json(
        {
            key: tokenizer.get(key)
            for key in (
                "tokenizer_model",
                "tokenizer_pre",
                "bos_token_id",
                "eos_token_id",
                "padding_token_id",
                "add_bos_token",
                "prompt_token_ids_sha256",
            )
        }
    )
    specs = deepcopy(MODEL_SPECS)
    specs[0]["gpu"] = gpu
    return {
        "MODEL_SPECS": specs,
        "hub_id": GEMMA26_HUB_ID,
        "model_path": model_path,
        "model_sha256": full_hash,
        "gguf_shards": shard_rows,
        "revision": provenance.get("revision") if isinstance(provenance, Mapping) else None,
        "quantization": content.get("quantization") if isinstance(content, Mapping) else None,
        "architecture": content.get("architecture") if isinstance(content, Mapping) else None,
        "context_length": 262_144,
        "content_metadata": content,
        "cache_provenance": provenance,
        "embedded_tokenizer": {
            "source": "embedded_gguf",
            "loadable": tokenizer.get("embedded_tokenizer_ok") is True,
            "identity_sha256": tokenizer_identity,
            "token_count": content.get("tokenizer_metadata", {}).get("token_count", 0)
            if isinstance(content, Mapping)
            else 0,
            "receipt": tokenizer,
        },
        "embedded_chat_template": {
            "source": "tokenizer.chat_template",
            "present": bool(content.get("tokenizer_metadata", {}).get("chat_template_present"))
            if isinstance(content, Mapping)
            else False,
            "sha256": tokenizer.get("chat_template_sha256"),
        },
        "llama_cpp": {
            "cuda_linked": build.get("cuda_linked") is True,
            "version": build.get("version_receipt", {}).get("stdout"),
            "build_receipt": build,
        },
        "auto_tokenizer_used": False,
        "download_performed": False,
        "legacy_headline_row_count": 0,
    }


def _collect_preconditions(
    *,
    repo_root: Path,
    gpu: int,
    exp6604_artifact: Mapping[str, Any],
    exp6604_hash: str,
    contract: Mapping[str, Any],
    model_identity: Mapping[str, Any],
    tests_run: Sequence[Mapping[str, Any]],
    protected_before: Mapping[str, str],
) -> tuple[JsonDict, JsonDict]:  # pragma: no cover
    model_path = str(model_identity.get("model_path", ""))
    initial = runtime_helpers._live_gpu_sample(  # noqa: SLF001
        repository_id=GEMMA26_HUB_ID,
        worker_pid=0,
        stage="preconditions",
        sample_index=0,
        selected_gpu=gpu,
        model_paths=[model_path],
    )
    device = initial.get("device", {})
    ownership = gpu_ownership_receipt(initial, controller_pid=os.getpid())
    resources = _host_resources(repo_root)
    checkpoint_parent = repo_root / CHECKPOINT_RELATIVE_PATH.parent
    checkpoint_parent.mkdir(parents=True, exist_ok=True)
    checkpoint_free = shutil.disk_usage(checkpoint_parent).free
    model_specs_match_gpu = (
        model_identity.get("MODEL_SPECS", [{}])[0].get("gpu") == gpu
        and model_identity.get("MODEL_SPECS", [{}])[0].get("gpu_assignment_reviewed") is True
    )
    checks = {
        "upstream_gate": exp6604_artifact.get("headroom_fixture_ready_score") == 1.0,
        "upstream_status": exp6604_artifact.get("status") == "complete",
        "task_count": contract.get("task_count") == EXPECTED_TASK_COUNT,
        "row_count": contract.get("expected_row_count") == EXPECTED_ROW_COUNT,
        "task_and_split_hashes": bool(contract.get("split_hashes", {}).get("calibration"))
        and bool(contract.get("split_hashes", {}).get("held")),
        "model_identity": _model_identity_ready(model_identity),
        "reviewed_gpu_assignment": model_specs_match_gpu and gpu == 0,
        "gpu_query": initial.get("gpu_query_exit_code") == 0
        and initial.get("compute_query_exit_code") == 0,
        "gpu_identity": device.get("index") == gpu and "RTX 3090" in str(device.get("name", "")),
        "gpu_ownership": ownership["available"],
        "gpu_free_vram_mb": int(device.get("memory_free_mb", 0)) >= MIN_FREE_VRAM_MB,
        "llama_cpp_cuda": model_identity.get("llama_cpp", {}).get("cuda_linked") is True,
        "checkpoint_space": checkpoint_free >= MIN_CHECKPOINT_FREE_BYTES,
        "output_space": int(resources.get("disk", {}).get("free_bytes", 0))
        >= MIN_CHECKPOINT_FREE_BYTES,
        "timeout_budget": TASK_TIMEOUT_S >= EXPECTED_ROW_COUNT * PER_GENERATION_TIMEOUT_S,
        "focused_tests": bool(tests_run) and all(row.get("exit_code") == 0 for row in tests_run),
        "protected_hashes": len(protected_before) == len(PROTECTED_RELATIVE_PATHS)
        and all(value != "missing" for value in protected_before.values()),
        "atomic_output": os.access((repo_root / RESULT_RELATIVE_PATH).parent, os.W_OK),
    }
    failed = [name for name, passed in checks.items() if not passed]
    return (
        {
            "all_required_preconditions_available": not failed,
            "checks": checks,
            "failed_preconditions": failed,
            "model_process_started": False,
            "exp6604_gate": {
                "path": EXP6604_RELATIVE_PATH.as_posix(),
                "absolute_path": str((repo_root / EXP6604_RELATIVE_PATH).resolve()),
                "artifact_sha256": exp6604_hash,
                "field": "headroom_fixture_ready_score",
                "expected": 1.0,
                "observed": exp6604_artifact.get("headroom_fixture_ready_score"),
            },
            "task_count": contract.get("task_count"),
            "expected_row_count": contract.get("expected_row_count"),
            "task_hashes": [row.get("task_sha256") for row in contract.get("task_prompts", [])],
            "split_hashes": dict(contract.get("split_hashes", {})),
            "prompt_contract_sha256": contract.get("contract_sha256"),
            "protected_file_hashes_before": dict(protected_before),
            "resource_receipts": resources,
            "checkpoint": {
                "path": str((repo_root / CHECKPOINT_RELATIVE_PATH).resolve()),
                "free_bytes": checkpoint_free,
                "minimum_free_bytes": MIN_CHECKPOINT_FREE_BYTES,
            },
            "gpu": {
                "requested": gpu,
                "reviewed": True,
                "initial_sample": initial,
                "free_vram_mb": device.get("memory_free_mb"),
                "minimum_free_vram_mb": MIN_FREE_VRAM_MB,
                "ownership_receipt": ownership,
            },
            "model": dict(model_identity),
            "timeout_budget": {
                "load_s": LOAD_TIMEOUT_S,
                "per_generation_s": PER_GENERATION_TIMEOUT_S,
                "task_s": TASK_TIMEOUT_S,
                "expected_row_count": EXPECTED_ROW_COUNT,
            },
            "download_allowed": False,
            "auto_tokenizer_allowed": False,
            "cpu_headline_fallback_allowed": False,
        },
        initial,
    )


def _server_command(server: Path, model_path: str, port: int) -> list[str]:  # pragma: no cover
    return [
        str(server),
        "--model",
        model_path,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--ctx-size",
        str(CONTEXT_SIZE),
        "--n-gpu-layers",
        "all",
        "--device",
        "CUDA0",
        "--split-mode",
        "none",
        "--main-gpu",
        "0",
        "--fit",
        "off",
        "--parallel",
        "1",
        "--batch-size",
        "128",
        "--ubatch-size",
        "128",
        "--cache-type-k",
        "q8_0",
        "--cache-type-v",
        "q8_0",
        "--offline",
        "--jinja",
        "--reasoning",
        "off",
        "--no-ui",
        "--log-verbosity",
        "4",
    ]


def _free_port() -> int:  # pragma: no cover
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as handle:
        handle.bind(("127.0.0.1", 0))
        return int(handle.getsockname()[1])


def _port_open(port: int) -> bool:  # pragma: no cover
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as handle:
        handle.settimeout(0.25)
        return handle.connect_ex(("127.0.0.1", port)) == 0


def _request_generation(
    *,
    port: int,
    job: Mapping[str, Any],
    deadline: float,
) -> JsonDict:  # pragma: no cover
    remaining = deadline - time.monotonic()
    started_ns = time.monotonic_ns()
    started = time.monotonic()
    if remaining <= 0:
        return {
            "raw_response_bytes": b"",
            "raw_api_response_sha256": sha256_bytes(b""),
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "finish_reason": "timeout",
            "http_status": 124,
            "failure_kind": "timeout",
            "started_monotonic_ns": started_ns,
            "finished_monotonic_ns": time.monotonic_ns(),
            "latency_s": 0.0,
        }
    payload = {
        "model": "local-gguf",
        "messages": [
            {
                "role": "user",
                "content": bytes(job["prompt_bytes"]).decode("utf-8"),
            }
        ],
        "seed": int(job["seed"]),
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "top_k": TOP_K,
        "repeat_penalty": REPEAT_PENALTY,
        "max_tokens": MAX_OUTPUT_TOKENS,
        "stop": list(STOP_RULES),
        "stream": False,
    }
    raw_api = b""
    raw_response = b""
    prompt_tokens = 0
    completion_tokens = 0
    finish_reason = "request_failure"
    http_status = 0
    failure_kind: str | None = None
    try:
        http_status, raw_api = stream_helpers._http_bytes(  # noqa: SLF001
            f"http://127.0.0.1:{port}/v1/chat/completions",
            payload,
            timeout_s=min(PER_GENERATION_TIMEOUT_S, max(0.1, remaining)),
        )
        (
            raw_response,
            prompt_tokens,
            completion_tokens,
            finish_reason,
            malformed,
        ) = stream_helpers._parse_api_response(raw_api)  # noqa: SLF001
        if malformed:
            failure_kind = "malformed_api_response"
    except (OSError, TimeoutError, urllib.error.URLError):
        http_status = 124
        finish_reason = "timeout"
        failure_kind = "timeout"
    return {
        "raw_response_bytes": raw_response,
        "raw_api_response_sha256": sha256_bytes(raw_api),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "finish_reason": finish_reason,
        "http_status": http_status,
        "failure_kind": failure_kind,
        "started_monotonic_ns": started_ns,
        "finished_monotonic_ns": time.monotonic_ns(),
        "latency_s": time.monotonic() - started,
    }


def _failed_generation(reason: str) -> JsonDict:  # pragma: no cover
    now = time.monotonic_ns()
    return {
        "raw_response_bytes": b"",
        "raw_api_response_sha256": sha256_bytes(b""),
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "finish_reason": reason,
        "http_status": 503,
        "failure_kind": reason,
        "started_monotonic_ns": now,
        "finished_monotonic_ns": now,
        "latency_s": 0.0,
    }


def _sample_to_compact(row: Mapping[str, Any]) -> JsonDict:  # pragma: no cover
    device = row.get("device", {})
    pid = int(row.get("worker_pid", 0) or 0)
    compute = [item for item in row.get("compute_processes", []) if isinstance(item, Mapping)]
    return {
        "stage": row.get("stage"),
        "sample_index": row.get("sample_index"),
        "timestamp_utc": row.get("timestamp_utc"),
        "timestamp_monotonic_ns": row.get("timestamp_monotonic_ns"),
        "gpu_index": device.get("index"),
        "gpu_uuid": device.get("uuid"),
        "gpu_name": device.get("name"),
        "memory_used_mb": device.get("memory_used_mb"),
        "memory_free_mb": device.get("memory_free_mb"),
        "utilization_pct": device.get("utilization_pct"),
        "temperature_c": device.get("temperature_c"),
        "worker_pid_present": any(int(item.get("pid", 0) or 0) == pid for item in compute),
        "compute_processes": compute,
        "query_exit_codes": {
            "gpu": row.get("gpu_query_exit_code"),
            "compute": row.get("compute_query_exit_code"),
        },
    }


def _run_live_rows(
    *,
    repo_root: Path,
    gpu: int,
    jobs: Sequence[Mapping[str, Any]],
    model_identity: Mapping[str, Any],
    exp6604_hash: str,
    prompt_contract_hash: str,
    resumed_rows: Sequence[Mapping[str, Any]],
    resumed_sessions: Sequence[Mapping[str, Any]],
    deadline: float,
) -> tuple[list[JsonDict], list[JsonDict], JsonDict]:  # pragma: no cover
    rows = [dict(row) for row in resumed_rows]
    sessions = [dict(row) for row in resumed_sessions]
    if len(rows) == len(jobs):
        receipt = atomic_write_checkpoint(
            repo_root / CHECKPOINT_RELATIVE_PATH,
            rows,
            prompt_contract_sha256=prompt_contract_hash,
            exp6604_artifact_sha256=exp6604_hash,
            model_sha256=str(model_identity["model_sha256"]),
            process_sessions=sessions,
        )
        return rows, sessions, receipt

    server = Path.home() / ".cache/llama.cpp-master/build/bin/llama-server"
    model_path = str(model_identity["model_path"])
    port = _free_port()
    command = _server_command(server, model_path, port)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    before_raw = runtime_helpers._live_gpu_sample(  # noqa: SLF001
        repository_id=GEMMA26_HUB_ID,
        worker_pid=0,
        stage="before",
        sample_index=0,
        selected_gpu=gpu,
        model_paths=[model_path],
    )
    baseline = int(before_raw.get("device", {}).get("memory_used_mb", 0) or 0)
    process: subprocess.Popen[bytes] | None = None
    identity: JsonDict = {}
    samples_raw = [before_raw]
    healthy = False
    http_status = 0
    offloaded_layers = 0
    shutdown_requested = False
    forced_kill = False
    error = ""
    start_utc = _utc_now()
    start_ns = time.monotonic_ns()
    generated_count = 0
    session_id = f"exp6607-session-{start_ns}"
    with tempfile.TemporaryDirectory(prefix="exp6607-llama-") as temporary:
        stdout_path = Path(temporary) / "stdout.bin"
        stderr_path = Path(temporary) / "stderr.bin"
        try:
            with stdout_path.open("wb") as stdout_handle, stderr_path.open("wb") as stderr_handle:
                process = subprocess.Popen(
                    command,
                    cwd=repo_root,
                    env=env,
                    stdin=subprocess.DEVNULL,
                    stdout=stdout_handle,
                    stderr=stderr_handle,
                )
            identity = runtime_helpers._wait_for_process_identity(  # noqa: SLF001
                process.pid, command
            )
            load_deadline = min(deadline, time.monotonic() + LOAD_TIMEOUT_S)
            sample_index = 1
            while time.monotonic() < load_deadline:
                if process.poll() is not None:
                    raise RuntimeError(f"llama-server exited during load with {process.returncode}")
                sample = runtime_helpers._live_gpu_sample(  # noqa: SLF001
                    repository_id=GEMMA26_HUB_ID,
                    worker_pid=process.pid,
                    stage="during",
                    sample_index=sample_index,
                    selected_gpu=gpu,
                    model_paths=[model_path],
                )
                samples_raw.append(sample)
                sample_index += 1
                try:
                    status, raw_health = stream_helpers._http_bytes(  # noqa: SLF001
                        f"http://127.0.0.1:{port}/health", timeout_s=0.5
                    )
                    health = json.loads(raw_health.decode("utf-8"))
                    healthy = status == 200 and health.get("status") == "ok"
                except (
                    OSError,
                    TimeoutError,
                    urllib.error.URLError,
                    UnicodeDecodeError,
                    json.JSONDecodeError,
                ):
                    healthy = False
                if healthy:
                    http_status = 200
                    break
                time.sleep(1.0)
            if not healthy:
                raise TimeoutError("llama-server load timeout")
            identity = runtime_helpers.select_process_identity_receipt(
                identity,
                runtime_helpers._proc_identity(process.pid),  # noqa: SLF001
                command,
            )
            offloaded_layers = stream_helpers._offloaded_layers(  # noqa: SLF001
                stderr_path.read_bytes()
            )
            offload_deadline = min(deadline, time.monotonic() + 10.0)
            while offloaded_layers <= 0 and time.monotonic() < offload_deadline:
                time.sleep(0.25)
                offloaded_layers = stream_helpers._offloaded_layers(  # noqa: SLF001
                    stderr_path.read_bytes()
                )
            if offloaded_layers <= 0:
                raise RuntimeError("CUDA layer offload receipt missing")
            samples_raw.append(
                runtime_helpers._live_gpu_sample(  # noqa: SLF001
                    repository_id=GEMMA26_HUB_ID,
                    worker_pid=process.pid,
                    stage="during",
                    sample_index=len(samples_raw),
                    selected_gpu=gpu,
                    model_paths=[model_path],
                )
            )
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            healthy = False

        for index, job in enumerate(jobs[len(rows) :], start=len(rows)):
            alive = healthy and process is not None and process.poll() is None
            if alive and index % TELEMETRY_EVERY_ROWS == 0:
                samples_raw.append(
                    runtime_helpers._live_gpu_sample(  # noqa: SLF001
                        repository_id=GEMMA26_HUB_ID,
                        worker_pid=process.pid,
                        stage="during",
                        sample_index=len(samples_raw),
                        selected_gpu=gpu,
                        model_paths=[model_path],
                    )
                )
            generation = (
                _request_generation(port=port, job=job, deadline=deadline)
                if alive
                else _failed_generation("process_failure")
            )
            binding = {
                "session_id": session_id,
                "pid": 0 if process is None else process.pid,
                "parent_pid": identity.get("parent_pid"),
                "owned_child": identity.get("parent_pid") == os.getpid(),
                "repository_id": GEMMA26_HUB_ID,
                "model_sha256": model_identity.get("model_sha256"),
                "command_sha256": sha256_json(command),
                "selected_gpu": gpu,
                "gpu_uuid": before_raw.get("device", {}).get("uuid"),
                "cpu_fallback": False,
                "cuda_offload": offloaded_layers > 0,
                "offloaded_layers": offloaded_layers,
                "tokenizer_source": "embedded_gguf",
                "chat_template_sha256": model_identity.get("embedded_chat_template", {}).get(
                    "sha256"
                ),
            }
            rows.append(build_per_unit_row(job, generation, binding))
            generated_count += 1
            active_session = {
                "session_id": session_id,
                "pid": binding["pid"],
                "parent_pid": binding["parent_pid"],
                "owned_child": binding["owned_child"],
                "repository_id": GEMMA26_HUB_ID,
                "model_sha256": model_identity.get("model_sha256"),
                "command_sha256": binding["command_sha256"],
                "selected_gpu": gpu,
                "gpu_uuid": binding["gpu_uuid"],
                "cuda_visible_devices": str(gpu),
                "cpu_fallback": False,
                "cuda_offload": offloaded_layers > 0,
                "offloaded_layers": offloaded_layers,
                "server_healthy": healthy,
                "row_count": generated_count,
                "samples": [_sample_to_compact(row) for row in samples_raw],
                "shutdown_requested": False,
                "normal_shutdown": False,
                "worker_absent_after_exit": False,
                "port_closed": False,
                "memory_recovered": False,
                "signals_sent_to_unrelated_pids": [],
            }
            atomic_write_checkpoint(
                repo_root / CHECKPOINT_RELATIVE_PATH,
                rows,
                prompt_contract_sha256=prompt_contract_hash,
                exp6604_artifact_sha256=exp6604_hash,
                model_sha256=str(model_identity["model_sha256"]),
                process_sessions=[*sessions, active_session],
            )

        exit_code: int | None = 127
        if process is not None:
            if process.poll() is None:
                shutdown_requested = True
                process.send_signal(signal.SIGTERM)
                try:
                    exit_code = process.wait(timeout=SHUTDOWN_TIMEOUT_S)
                except subprocess.TimeoutExpired:
                    forced_kill = True
                    process.kill()
                    exit_code = process.wait(timeout=5)
            else:
                exit_code = process.returncode
        end_ns = time.monotonic_ns()
        stdout_bytes = stdout_path.read_bytes() if stdout_path.is_file() else b""
        stderr_bytes = stderr_path.read_bytes() if stderr_path.is_file() else b""
        offloaded_layers = max(
            offloaded_layers,
            stream_helpers._offloaded_layers(stderr_bytes),  # noqa: SLF001
        )

    worker_pid = 0 if process is None else process.pid
    recovery_start = time.monotonic()
    after_raw: JsonDict = {}
    recovery_complete = False
    while time.monotonic() - recovery_start <= RECOVERY_TIMEOUT_S:
        after_raw = runtime_helpers._live_gpu_sample(  # noqa: SLF001
            repository_id=GEMMA26_HUB_ID,
            worker_pid=worker_pid,
            stage="after",
            sample_index=len(samples_raw),
            selected_gpu=gpu,
            model_paths=[model_path],
        )
        recovered = int(after_raw.get("device", {}).get("memory_used_mb", 0) or 0)
        pids = {int(row.get("pid", 0) or 0) for row in after_raw.get("compute_processes", [])}
        recovery_complete = (
            worker_pid > 1
            and not Path(f"/proc/{worker_pid}").exists()
            and worker_pid not in pids
            and not _port_open(port)
            and abs(recovered - baseline) <= RECOVERY_TOLERANCE_MB
            and not runtime_helpers._task_owned_pids([model_path])  # noqa: SLF001
        )
        if recovery_complete:
            break
        time.sleep(1.0)
    samples_raw.append(after_raw)
    final_session = {
        "session_id": session_id,
        "pid": worker_pid,
        "parent_pid": identity.get("parent_pid"),
        "owned_child": identity.get("parent_pid") == os.getpid(),
        "repository_id": GEMMA26_HUB_ID,
        "model_sha256": model_identity.get("model_sha256"),
        "command": command,
        "command_sha256": sha256_json(command),
        "selected_gpu": gpu,
        "gpu_uuid": before_raw.get("device", {}).get("uuid"),
        "cuda_visible_devices": str(gpu),
        "cpu_fallback": False,
        "cuda_offload": offloaded_layers > 0,
        "offloaded_layers": offloaded_layers,
        "server_healthy": healthy and http_status == 200,
        "row_count": generated_count,
        "samples": [_sample_to_compact(row) for row in samples_raw],
        "started_utc": start_utc,
        "started_monotonic_ns": start_ns,
        "ended_monotonic_ns": end_ns,
        "shutdown_requested": shutdown_requested,
        "normal_shutdown": shutdown_requested
        and not forced_kill
        and exit_code in {0, -signal.SIGTERM},
        "exit_code": exit_code,
        "worker_absent_after_exit": worker_pid > 1 and not Path(f"/proc/{worker_pid}").exists(),
        "port_closed": not _port_open(port),
        "memory_recovered": recovery_complete,
        "recovery_duration_s": round(time.monotonic() - recovery_start, 6),
        "stdout_sha256": sha256_bytes(stdout_bytes),
        "stderr_sha256": sha256_bytes(stderr_bytes),
        "stderr_tail": stderr_bytes.decode("utf-8", "replace")[-8000:],
        "error": error,
        "signals_sent_to_unrelated_pids": [],
    }
    sessions.append(final_session)
    receipt = atomic_write_checkpoint(
        repo_root / CHECKPOINT_RELATIVE_PATH,
        rows,
        prompt_contract_sha256=prompt_contract_hash,
        exp6604_artifact_sha256=exp6604_hash,
        model_sha256=str(model_identity["model_sha256"]),
        process_sessions=sessions,
    )
    return rows, sessions, receipt


def _first_failed_value(
    preconditions: Mapping[str, Any],
) -> tuple[str, Any, Any]:  # pragma: no cover
    name = next(iter(preconditions.get("failed_preconditions", [])), "preconditions")
    observed = preconditions.get("checks", {}).get(name)
    expected: Any = True
    if name == "gpu_free_vram_mb":
        expected = MIN_FREE_VRAM_MB
        observed = preconditions.get("gpu", {}).get("free_vram_mb")
    elif name == "task_count":
        expected = EXPECTED_TASK_COUNT
        observed = preconditions.get("task_count")
    elif name == "row_count":
        expected = EXPECTED_ROW_COUNT
        observed = preconditions.get("expected_row_count")
    return name, expected, observed


def run_experiment(
    repo_root: Path,
    run_date: str,
    gpu: int,
) -> JsonDict:  # pragma: no cover
    """Run preconditions, resumable live inference, cleanup, and atomic output."""

    started = time.monotonic()
    protected_before = _protected_hashes(repo_root)
    exp6604_path = repo_root / EXP6604_RELATIVE_PATH
    exp6604_hash = sha256_file(exp6604_path)
    exp6604 = json.loads(exp6604_path.read_text(encoding="utf-8"))
    contract = build_prompt_and_decode_contract(exp6604)
    jobs = build_generation_jobs(exp6604, contract)
    tests_run = _checkpoint_tests(repo_root)
    model_identity = _resolve_model_identity(gpu)
    preconditions, _initial = _collect_preconditions(
        repo_root=repo_root,
        gpu=gpu,
        exp6604_artifact=exp6604,
        exp6604_hash=exp6604_hash,
        contract=contract,
        model_identity=model_identity,
        tests_run=tests_run,
        protected_before=protected_before,
    )
    checkpoint = load_checkpoint(
        repo_root / CHECKPOINT_RELATIVE_PATH,
        expected_row_ids=[str(row["row_id"]) for row in jobs],
        prompt_contract_sha256=str(contract["contract_sha256"]),
        exp6604_artifact_sha256=exp6604_hash,
        model_sha256=str(model_identity["model_sha256"]),
    )
    if checkpoint["accepted"] is not True:
        preconditions["checks"]["checkpoint_resume"] = False
        preconditions["failed_preconditions"].append("checkpoint_resume")
        preconditions["all_required_preconditions_available"] = False
        preconditions["checkpoint_resume"] = checkpoint
    else:
        preconditions["checks"]["checkpoint_resume"] = True
        preconditions["checkpoint_resume"] = {
            "accepted": True,
            "completed_row_count": checkpoint["completed_row_count"],
            "gate_check_summary": checkpoint["gate_check_summary"],
        }
    rows: list[JsonDict] = []
    sessions: list[JsonDict] = []
    checkpoint_receipt: JsonDict = {}
    if preconditions["all_required_preconditions_available"]:
        preconditions["model_process_started"] = checkpoint["completed_row_count"] < len(jobs)
        rows, sessions, checkpoint_receipt = _run_live_rows(
            repo_root=repo_root,
            gpu=gpu,
            jobs=jobs,
            model_identity=model_identity,
            exp6604_hash=exp6604_hash,
            prompt_contract_hash=str(contract["contract_sha256"]),
            resumed_rows=checkpoint["rows"],
            resumed_sessions=checkpoint["process_sessions"],
            deadline=started + TASK_TIMEOUT_S,
        )
    protected_after = _protected_hashes(repo_root)
    protected = _protected_receipt(protected_before, protected_after)
    preconditions["protected_file_hashes_after"] = protected_after
    if not preconditions["all_required_preconditions_available"]:
        failed, expected, observed = _first_failed_value(preconditions)
        artifact = build_blocked_artifact(
            run_date=run_date,
            failed_condition=failed,
            expected=expected,
            observed=observed,
            model_identity=model_identity,
            preconditions=preconditions,
            protected_files=protected,
            tests_run=tests_run,
            duration_s=time.monotonic() - started,
        )
    else:
        gpu_receipts = {
            "sessions": sessions,
            "all_sessions_authentic": _gpu_receipts_ready(
                {"sessions": sessions, "all_sessions_authentic": True}, len(rows)
            ),
        }
        artifact = assemble_artifact(
            run_date=run_date,
            exp6604_artifact=exp6604,
            prompt_contract=contract,
            per_unit_rows=rows,
            model_identity=model_identity,
            gpu_receipts=gpu_receipts,
            checkpoint_receipts=checkpoint_receipt,
            preconditions=preconditions,
            protected_files=protected,
            tests_run=tests_run,
            duration_s=time.monotonic() - started,
        )
    atomic_write_artifact(repo_root / RESULT_RELATIVE_PATH, artifact)
    return artifact


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover
    """Run or validate the frozen Gemma 4 26B-A4B direct baseline."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args(argv)
    output = args.output or (REPO_ROOT / RESULT_RELATIVE_PATH)
    if args.validate:
        artifact = json.loads(output.read_text(encoding="utf-8"))
        errors = validate_artifact(artifact)
        print(json.dumps({"valid": not errors, "errors": errors}, indent=2))
        return 1 if errors else 0
    artifact = run_experiment(REPO_ROOT, args.date, args.gpu)
    print(
        json.dumps(
            {
                "artifact": str(REPO_ROOT / RESULT_RELATIVE_PATH),
                "status": artifact["status"],
                "verdict_class": artifact["verdict_class"],
                "gemma26_headroom_ready_score": artifact["gemma26_headroom_ready_score"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
