"""Compare three certificate transports on three local GGUF model families.

The exact task executors decide semantic success. Grammar acceptance only
decides whether an output has the required surface form. This separation keeps
syntax improvements from becoming unsupported correctness claims.

Spec refs: REQ-CONSTRAINT-6676, SCENARIO-CONSTRAINT-6676-ONE-GENERATION,
SCENARIO-CONSTRAINT-6676-LAZY-SYNTAX,
SCENARIO-CONSTRAINT-6676-EXACT-AUTHORITY, REQ-INFER-SOTA-6676,
SCENARIO-INFER-SOTA-6676-COMPLETE-MATRIX,
SCENARIO-INFER-SOTA-6676-CUDA-BLOCK, REQ-INFRA-6676,
SCENARIO-INFRA-6676-OWNER-BOUND-SESSION,
SCENARIO-INFRA-6676-CLEAN-RELEASE, REQ-VERIFY-6676,
SCENARIO-VERIFY-6676-PAIRED-EVIDENCE,
SCENARIO-VERIFY-6676-MISSING-IS-MISSING, REQ-SAFE-6676,
SCENARIO-SAFE-6676-HARMFUL-FLIP,
SCENARIO-SAFE-6676-NO-SEMANTIC-PROMOTION, REQ-REPORT-6676,
SCENARIO-REPORT-6676-COMPLETE-ROWS,
SCENARIO-REPORT-6676-BLOCKED-ARTIFACT, and
SCENARIO-REPORT-6676-ATOMIC-PROVENANCE.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import shutil
import signal
import socket
import subprocess
import sys
import time
from typing import Any
import urllib.error
import urllib.request

from carnot import experiment_6648_three_family_gguf_canaries as runtime_api
from carnot import experiment_6661_triggered_tail_fixture as fixture_api
from carnot import experiment_6675_triggered_tail_scope_receipt as upstream_api
from carnot import gpu_lease_phase_journal as lease_api
from carnot.inference.sota_models import (
    cached_sota_pair,
    gguf_tokenizer_loadable,
    resolve_cached_gguf,
)


JsonDict = dict[str, Any]
MODULE_NAME = "carnot.experiment_6676_three_family_triggered_tail_ab"
REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_PATH = Path("results/experiment_6676_three_family_triggered_tail_ab.json")
WORK_PATH = Path("results/.experiment_6676_three_family_triggered_tail_ab")
UPSTREAM_PATH = Path("results/experiment_6675_triggered_tail_scope_receipt.json")
MODULE_PATH = Path("python/carnot/experiment_6676_three_family_triggered_tail_ab.py")
TEST_PATH = Path("tests/python/test_experiment_6676_three_family_triggered_tail_ab.py")
RUN_DATE = "20260827"
RANDOM_SEED = 6_676_001
TRIGGER_TOKEN = fixture_api.TRIGGER_TOKEN
ARM_ORDER = ("natural", "immediate_json", "triggered_tail")
INFERENCE_SUBSTRATE = "local_llamacpp_cuda_mandated_gguf_three_family"
LEGACY_MODEL_CAN_SATISFY_HEADLINE = False
VERDICT_CLASSES = frozenset(
    {"positive", "circular_positive", "null", "blocked", "disqualified", "partial"}
)
COMPLETE_PHASE_SEQUENCE = lease_api.COMPLETE_PHASE_SEQUENCE
PROTECTED_PATHS = (Path("research-roadmap.yaml"), Path("scripts/research_conductor.py"))

MODEL_SPECS = [
    {
        "family_id": "qwen36_flagship_moe",
        "hf_id": "unsloth/Qwen3.6-35B-A3B-GGUF",
        "role": "flagship_moe",
        "quantization": "Q4_K_M",
        "device_index": 0,
        "resolution_method": "cached_sota_pair",
    },
    {
        "family_id": "gemma4_31b_flagship_dense",
        "hf_id": "unsloth/gemma-4-31B-it-GGUF",
        "role": "flagship_dense",
        "quantization": "Q4_K_M",
        "device_index": 0,
        "resolution_method": "resolve_cached_gguf",
    },
    {
        "family_id": "gemma4_26b_middle_moe",
        "hf_id": "unsloth/gemma-4-26B-A4B-it-GGUF",
        "role": "middle_moe",
        "quantization": "Q4_K_M",
        "device_index": 1,
        "resolution_method": "cached_sota_pair",
    },
]

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "honest_verdict",
    "verdict_class",
    "gate_check_summary",
    "model_specs",
    "models_used",
    "frozen_run_manifest",
    "per_model_process_receipts",
    "per_unit_rows",
    "held_family_rows",
    "harmful_flip_rows",
    "exact_success_summary",
    "parse_transport_summary",
    "triggered_tail_ab_ready",
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

FOCUSED_TEST_COMMAND = (
    ".venv/bin/pytest tests/python/test_experiment_6676_three_family_triggered_tail_ab.py "
    "-q --no-cov -n 0 -o addopts="
)
COVERAGE_RUN_COMMAND = (
    "PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 COVERAGE_FILE=/tmp/carnot_exp6676.coverage "
    ".venv/bin/coverage run "
    "--include='*/experiment_6676_three_family_triggered_tail_ab.py' -m pytest "
    "--noconftest tests/python/test_experiment_6676_three_family_triggered_tail_ab.py "
    "-q -o addopts="
)
COVERAGE_REPORT_COMMAND = (
    "COVERAGE_FILE=/tmp/carnot_exp6676.coverage .venv/bin/coverage report "
    "--include='*/experiment_6676_three_family_triggered_tail_ab.py' -m --fail-under=100"
)
SPEC_COVERAGE_COMMAND = f".venv/bin/python scripts/check_spec_coverage.py {TEST_PATH}"
RUFF_COMMAND = f".venv/bin/ruff check {MODULE_PATH} {TEST_PATH}"
FORMAT_COMMAND = f".venv/bin/ruff format --check {MODULE_PATH} {TEST_PATH}"
VALIDATE_COMMAND = f".venv/bin/python -m {MODULE_NAME} --validate"
ADVERSARIAL_COMMAND = f".venv/bin/python scripts/adversarial_verify.py {RESULT_PATH}"

LOAD_TIMEOUT_S = 900.0
REQUEST_TIMEOUT_S = 180.0
SHUTDOWN_TIMEOUT_S = 30.0
RECOVERY_TIMEOUT_S = 180.0
SESSION_TIMEOUT_S = 7_200.0
RECOVERY_TOLERANCE_MB = 512


def canonical_json(value: Any) -> str:
    """Return stable JSON text for hashes and byte comparisons."""

    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


def sha256_text(value: str) -> str:
    """Hash UTF-8 text with an explicit algorithm prefix."""

    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_json(value: Any) -> str:
    """Hash one canonical JSON value."""

    return sha256_text(canonical_json(value))


def sha256_file(path: str | Path) -> str:
    """Hash a complete file, or preserve that the file is missing."""

    candidate = Path(path)
    if not candidate.is_file():
        return "missing"
    digest = hashlib.sha256()
    with candidate.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _hash_without(value: Mapping[str, Any], field: str) -> str:
    return sha256_json({key: item for key, item in value.items() if key != field})


def frozen_manifest_hash(manifest: Mapping[str, Any]) -> str:
    """Hash a frozen run manifest without its self-referential hash."""

    return _hash_without(manifest, "manifest_sha256")


def unit_row_hash(row: Mapping[str, Any]) -> str:
    """Hash one retained unit row without its self-referential hash."""

    return _hash_without(row, "row_sha256")


def process_receipt_hash(receipt: Mapping[str, Any]) -> str:
    """Hash one process receipt without its self-referential hash."""

    return _hash_without(receipt, "receipt_sha256")


def artifact_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash a terminal artifact while excluding only its checksum."""

    return _hash_without(artifact, "reproducibility_checksum")


def _read_json(path: Path) -> JsonDict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(value) if isinstance(value, Mapping) else {}


def _gguf_magic(path: str | Path) -> tuple[str, bool]:
    try:
        magic = Path(path).read_bytes()[:4]
    except OSError:
        return "", False
    return magic.decode("ascii", "replace"), magic == b"GGUF"


def resolve_model_specs(
    *,
    pair_resolver: Callable[..., list[JsonDict] | None] = cached_sota_pair,
    gguf_resolver: Callable[..., str | None] = resolve_cached_gguf,
    tokenizer_probe: Callable[[str | None], tuple[bool, str]] = gguf_tokenizer_loadable,
) -> list[JsonDict]:
    """Resolve all mandated files through the frozen pair and dense helpers."""

    pair = pair_resolver(gpu_indices=(0, 1), model_indices=(0, 1)) or []
    pair_by_id = {str(row.get("hf_id")): dict(row) for row in pair if isinstance(row, Mapping)}
    dense_path = gguf_resolver(MODEL_SPECS[1]["hf_id"], "Q4_K_M")
    rows: list[JsonDict] = []
    for spec in MODEL_SPECS:
        resolved = (
            pair_by_id.get(spec["hf_id"], {}).get("model_path")
            if spec["resolution_method"] == "cached_sota_pair"
            else dense_path
        )
        path_text = str(resolved or "")
        candidate = Path(path_text) if path_text else None
        exists = bool(candidate and candidate.is_file())
        magic, magic_valid = _gguf_magic(candidate or "")
        tokenizer_ok, tokenizer_detail = tokenizer_probe(path_text if exists else None)
        rows.append(
            {
                **spec,
                "resolver_call": (
                    "cached_sota_pair(gpu_indices=(0, 1), model_indices=(0, 1))"
                    if spec["resolution_method"] == "cached_sota_pair"
                    else f"resolve_cached_gguf({spec['hf_id']!r}, 'Q4_K_M')"
                ),
                "model_path": path_text,
                "resolved_path": str(candidate.resolve()) if exists and candidate else "",
                "model_sha256": sha256_file(candidate or ""),
                "byte_count": candidate.stat().st_size if exists and candidate else 0,
                "gguf_magic": magic,
                "gguf_magic_valid": magic_valid,
                "tokenizer_source": "llama.cpp_embedded_gguf",
                "embedded_tokenizer_loadable": bool(tokenizer_ok),
                "embedded_tokenizer_detail": tokenizer_detail,
                "resolved": exists,
                "download_performed": False,
            }
        )
    return rows


def model_resolution_failures(rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Return each cache, magic, hash, or embedded-tokenizer blocker."""

    failures: list[JsonDict] = []
    by_id = {str(row.get("hf_id")): row for row in rows}
    for spec in MODEL_SPECS:
        row = by_id.get(spec["hf_id"])
        if row is None:
            failures.append(_failure(spec["hf_id"], True, None, "model_row_missing"))
            continue
        checks = {
            "resolved": row.get("resolved") is True,
            "gguf_magic": row.get("gguf_magic_valid") is True,
            "model_hash": str(row.get("model_sha256", "")).startswith("sha256:"),
            "embedded_tokenizer": row.get("embedded_tokenizer_loadable") is True,
            "no_download": row.get("download_performed") is False,
        }
        for name, passed in checks.items():
            if not passed:
                failures.append(
                    _failure(
                        f"{spec['family_id']}.{name}",
                        True,
                        row.get(name),
                        f"model_{name}_failed",
                    )
                )
    return failures


def load_upstream_gate(root: Path = REPO_ROOT) -> tuple[JsonDict, JsonDict]:
    """Load the producer-owned Exp6675 readiness field and validate its artifact."""

    path = root / UPSTREAM_PATH
    payload = _read_json(path)
    validator_errors = upstream_api.validate_artifact(payload) if payload else ["artifact_missing"]
    observed = payload.get("triggered_tail_fixture_ready")
    return (
        {
            "path": UPSTREAM_PATH.as_posix(),
            "absolute_path": str(path.resolve()),
            "sha256": sha256_file(path),
            "field": "triggered_tail_fixture_ready",
            "expected_value": True,
            "observed_value": observed,
            "upstream_status": payload.get("status"),
            "validator_errors": validator_errors,
            "passed": observed is True and not validator_errors,
        },
        payload,
    )


def unit_id(hf_id: str, task_id: str, arm: str) -> str:
    """Return the readable unique identity used by rows and receipts."""

    return f"{hf_id}::{task_id}::{arm}"


def build_frozen_run_manifest(upstream: Mapping[str, Any], *, ports: Sequence[int]) -> JsonDict:
    """Freeze all comparison choices before a model request can run."""

    tasks = deepcopy(list(upstream.get("frozen_task_manifest", [])))
    source_arms = upstream.get("arm_contracts", {})
    grammar = deepcopy(dict(upstream.get("syntax_only_grammar_receipt", {})))
    arms: dict[str, JsonDict] = {}
    for arm in ARM_ORDER:
        contract = deepcopy(dict(source_arms.get(arm, {})))
        if arm != "natural":
            contract["runtime_grammar"] = grammar.get("grammar")
            contract["runtime_grammar_sha256"] = grammar.get("grammar_sha256")
        arms[arm] = contract
    model_order = [row["hf_id"] for row in MODEL_SPECS]
    unit_seeds: dict[str, int] = {}
    for model_index, model_id in enumerate(model_order):
        for task_index, task in enumerate(tasks):
            for arm_index, arm in enumerate(ARM_ORDER):
                key = unit_id(model_id, str(task.get("task_id")), arm)
                unit_seeds[key] = RANDOM_SEED + model_index * 100_000 + task_index * 10 + arm_index
    manifest: JsonDict = {
        "schema": "carnot.experiment_6676.frozen_run_manifest.v1",
        "source_artifact": UPSTREAM_PATH.as_posix(),
        "source_artifact_checksum": upstream.get("reproducibility_checksum"),
        "tasks": tasks,
        "arms": arms,
        "arm_order": list(ARM_ORDER),
        "model_order": model_order,
        "ports": [int(port) for port in ports],
        "base_seed": RANDOM_SEED,
        "unit_seeds": unit_seeds,
        "generation_settings": {
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": 0,
            "max_tokens": 256,
            "stream": False,
            "planned_generations_per_unit": 1,
            "suffix_regeneration_allowed": False,
            "repair_allowed": False,
        },
        "stopping_rules": {"stop": [], "server_eos_allowed": True, "request_timeout_s": 180},
        "parsers": deepcopy(
            dict(upstream.get("frozen_input_receipts", {})).get("parser_hashes", {})
        ),
        "grammar": grammar,
        "checker_hashes": deepcopy(
            dict(upstream.get("frozen_input_receipts", {})).get("checker_hashes", {})
        ),
        "model_request_order": "model_then_task_then_arm",
        "missing_row_policy": "explicit_row_with_cause_excluded_from_rates",
        "finite_answer_id_transport": False,
        "hidden_gold_in_prompts": False,
        "solver_generated_answers_in_prompts": False,
    }
    manifest["manifest_sha256"] = frozen_manifest_hash(manifest)
    return manifest


def build_generation_request(
    task: Mapping[str, Any], arm_contract: Mapping[str, Any], seed: int
) -> JsonDict:
    """Build one answer-blind request from the visible task prompt only."""

    arm = str(arm_contract.get("arm"))
    prompt = str(arm_contract.get("prompt_template", "")).format(
        task_prompt=str(task.get("prompt", ""))
    )
    request: JsonDict = {
        "model": "local-mandated-gguf",
        "messages": [{"role": "user", "content": prompt}],
        "seed": int(seed),
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": 0,
        "max_tokens": int(arm_contract.get("total_token_budget", 256)),
        "stream": False,
        "stop": [],
        "cache_prompt": False,
    }
    if arm == "immediate_json":
        request.update(
            {
                "grammar": arm_contract.get("runtime_grammar"),
                "grammar_lazy": False,
            }
        )
    elif arm == "triggered_tail":
        request.update(
            {
                "grammar": arm_contract.get("runtime_grammar"),
                "grammar_lazy": True,
                "grammar_triggers": [{"type": 1, "value": TRIGGER_TOKEN}],
            }
        )
    return request


def decompose_output(raw_text: str, arm: str) -> JsonDict:
    """Separate the delayed tail without changing the retained raw text."""

    count = raw_text.count(TRIGGER_TOKEN)
    position = raw_text.find(TRIGGER_TOKEN) if count else -1
    if arm == "triggered_tail" and position >= 0:
        reasoning = raw_text[:position]
        tail = raw_text[position + len(TRIGGER_TOKEN) :]
        trigger_position: int | None = position
    else:
        reasoning = raw_text if arm == "natural" else ""
        tail = ""
        trigger_position = None
    return {
        "reasoning_text": reasoning,
        "trigger_position": trigger_position,
        "tail_text": tail,
        "trigger_count": count,
    }


def _response_parts(response: Mapping[str, Any]) -> JsonDict:
    body = response.get("body") if isinstance(response.get("body"), Mapping) else {}
    choices = body.get("choices", []) if isinstance(body, Mapping) else []
    choice = choices[0] if isinstance(choices, list) and choices else {}
    message = choice.get("message", {}) if isinstance(choice, Mapping) else {}
    content = str(message.get("content", "")) if isinstance(message, Mapping) else ""
    reasoning = str(message.get("reasoning_content", "")) if isinstance(message, Mapping) else ""
    raw_text = content
    if reasoning and reasoning not in content:
        raw_text = reasoning + ("\n" if content else "") + content
    usage = body.get("usage", {}) if isinstance(body, Mapping) else {}
    return {
        "raw_text": raw_text,
        "content_text": content,
        "reasoning_content": reasoning,
        "prompt_tokens": int(usage.get("prompt_tokens", 0) or 0)
        if isinstance(usage, Mapping)
        else 0,
        "completion_tokens": int(usage.get("completion_tokens", 0) or 0)
        if isinstance(usage, Mapping)
        else 0,
        "finish_reason": choice.get("finish_reason") if isinstance(choice, Mapping) else None,
        "body_sha256": sha256_json(body),
    }


def build_unit_row(
    *,
    model: Mapping[str, Any],
    task: Mapping[str, Any],
    arm: str,
    response: Mapping[str, Any],
    process_receipt_id: str,
    seed: int,
) -> JsonDict:
    """Parse and execute-check one completed, single-generation response."""

    parts = _response_parts(response)
    raw_text = str(parts["raw_text"])
    transport = decompose_output(raw_text, arm)
    parse = fixture_api.parse_arm_output(arm, raw_text)
    if parse.get("parsed") is True:
        checked = fixture_api.check_certificate(task, str(parse.get("certificate", "")))
        exact = {
            "checker_invoked": True,
            "exact_success": checked.get("exact_valid") is True,
            "reason": checked.get("reason"),
            "detail": deepcopy(checked.get("detail", [])),
            "authority": checked.get("authority"),
            "checker_sha256": dict(task.get("checker", {})).get("sha256"),
        }
    else:
        exact = {
            "checker_invoked": False,
            "exact_success": False,
            "reason": "parse_failed",
            "detail": [parse.get("failure")],
            "authority": dict(task.get("checker", {})).get("name"),
            "checker_sha256": dict(task.get("checker", {})).get("sha256"),
        }
    row: JsonDict = {
        "unit_id": unit_id(str(model.get("hf_id")), str(task.get("task_id")), arm),
        "row_status": "completed",
        "missing_cause": None,
        "family_id": model.get("family_id"),
        "hf_id": model.get("hf_id"),
        "model_path": model.get("model_path"),
        "model_sha256": model.get("model_sha256"),
        "task_id": task.get("task_id"),
        "task_sha256": task.get("task_sha256"),
        "family": task.get("family"),
        "arm": arm,
        "seed": int(seed),
        "request_count": 1,
        "raw_output": raw_text,
        "raw_output_sha256": sha256_text(raw_text),
        "content_text": parts["content_text"],
        "reasoning_text": transport["reasoning_text"],
        "api_reasoning_text": parts["reasoning_content"],
        "trigger_position": transport["trigger_position"],
        "trigger_count": transport["trigger_count"],
        "tail_text": transport["tail_text"],
        "parse_outcome": deepcopy(parse),
        "exact_outcome": exact,
        "harmful_flip": False,
        "baseline_unit_id": None,
        "prompt_token_count": parts["prompt_tokens"],
        "proposal_token_count": parts["completion_tokens"],
        "latency_s": float(response.get("latency_s", 0.0) or 0.0),
        "http_status": int(response.get("http_status", 0) or 0),
        "finish_reason": parts["finish_reason"],
        "api_response_sha256": parts["body_sha256"],
        "process_receipt_id": process_receipt_id,
    }
    row["row_sha256"] = unit_row_hash(row)
    return row


def build_missing_unit_row(
    *,
    model: Mapping[str, Any],
    task: Mapping[str, Any],
    arm: str,
    process_receipt_id: str,
    seed: int,
    cause: str,
) -> JsonDict:
    """Retain one failed request as missing without inventing a zero label."""

    row: JsonDict = {
        "unit_id": unit_id(str(model.get("hf_id")), str(task.get("task_id")), arm),
        "row_status": "missing",
        "missing_cause": str(cause),
        "family_id": model.get("family_id"),
        "hf_id": model.get("hf_id"),
        "model_path": model.get("model_path"),
        "model_sha256": model.get("model_sha256"),
        "task_id": task.get("task_id"),
        "task_sha256": task.get("task_sha256"),
        "family": task.get("family"),
        "arm": arm,
        "seed": int(seed),
        "request_count": 1,
        "raw_output": "",
        "raw_output_sha256": sha256_text(""),
        "content_text": "",
        "reasoning_text": "",
        "api_reasoning_text": "",
        "trigger_position": None,
        "trigger_count": 0,
        "tail_text": "",
        "parse_outcome": {"parsed": False, "failure": str(cause), "certificate": None},
        "exact_outcome": {
            "checker_invoked": False,
            "exact_success": None,
            "reason": "unit_missing",
            "detail": [str(cause)],
            "authority": dict(task.get("checker", {})).get("name"),
            "checker_sha256": dict(task.get("checker", {})).get("sha256"),
        },
        "harmful_flip": False,
        "baseline_unit_id": None,
        "prompt_token_count": 0,
        "proposal_token_count": 0,
        "latency_s": 0.0,
        "http_status": 0,
        "finish_reason": "request_failed",
        "api_response_sha256": sha256_json({}),
        "process_receipt_id": process_receipt_id,
    }
    row["row_sha256"] = unit_row_hash(row)
    return row


def annotate_harmful_flips(rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Mark constrained exact losses against each matched natural baseline."""

    result = [deepcopy(dict(row)) for row in rows]
    natural = {
        (str(row.get("hf_id")), str(row.get("task_id"))): row
        for row in result
        if row.get("arm") == "natural" and row.get("row_status") == "completed"
    }
    for row in result:
        baseline = natural.get((str(row.get("hf_id")), str(row.get("task_id"))))
        harmful = bool(
            row.get("arm") != "natural"
            and row.get("row_status") == "completed"
            and baseline
            and dict(baseline.get("exact_outcome", {})).get("exact_success") is True
            and dict(row.get("exact_outcome", {})).get("exact_success") is False
        )
        row["harmful_flip"] = harmful
        row["baseline_unit_id"] = baseline.get("unit_id") if harmful and baseline else None
        row["row_sha256"] = unit_row_hash(row)
    return result


def _arm_metrics(rows: Sequence[Mapping[str, Any]], outcome: str) -> dict[str, JsonDict]:
    metrics: dict[str, JsonDict] = {}
    for arm in ARM_ORDER:
        arm_rows = [row for row in rows if row.get("arm") == arm]
        complete = [row for row in arm_rows if row.get("row_status") == "completed"]
        if outcome == "exact":
            successes = sum(
                dict(row.get("exact_outcome", {})).get("exact_success") is True for row in complete
            )
            metrics[arm] = {
                "successes": successes,
                "total": len(complete),
                "rate": successes / len(complete) if complete else None,
                "missing": len(arm_rows) - len(complete),
            }
        else:
            successes = sum(
                dict(row.get("parse_outcome", {})).get("parsed") is True for row in complete
            )
            metrics[arm] = {
                "parsed": successes,
                "total": len(complete),
                "rate": successes / len(complete) if complete else None,
                "missing": len(arm_rows) - len(complete),
            }
    return metrics


def _paired_interval(differences: Sequence[int]) -> list[float] | None:
    if not differences:
        return None
    mean = sum(differences) / len(differences)
    if len(differences) == 1:
        return [mean, mean]
    variance = sum((value - mean) ** 2 for value in differences) / (len(differences) - 1)
    half_width = 1.96 * math.sqrt(variance / len(differences))
    return [max(-1.0, mean - half_width), min(1.0, mean + half_width)]


def _paired_metrics(rows: Sequence[Mapping[str, Any]], constrained_arm: str) -> JsonDict:
    by_key: dict[tuple[str, str], dict[str, Mapping[str, Any]]] = {}
    for row in rows:
        if row.get("row_status") != "completed":
            continue
        key = (str(row.get("hf_id")), str(row.get("task_id")))
        by_key.setdefault(key, {})[str(row.get("arm"))] = row
    differences: list[int] = []
    wins = losses = ties = no_headroom = 0
    for arms in by_key.values():
        baseline = arms.get("natural")
        constrained = arms.get(constrained_arm)
        if baseline is None or constrained is None:
            continue
        baseline_value = bool(dict(baseline.get("exact_outcome", {})).get("exact_success"))
        constrained_value = bool(dict(constrained.get("exact_outcome", {})).get("exact_success"))
        difference = int(constrained_value) - int(baseline_value)
        differences.append(difference)
        wins += difference == 1
        losses += difference == -1
        ties += difference == 0
        no_headroom += baseline_value
    return {
        "pairs": len(differences),
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "no_headroom_rows": no_headroom,
        "delta": sum(differences) / len(differences) if differences else None,
        "interval_95": _paired_interval(differences),
    }


def recompute_summaries(
    rows: Sequence[Mapping[str, Any]],
    manifest: Mapping[str, Any],
    model_specs: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Rebuild all exact, parse, family, paired, safety, and coverage headlines."""

    model_ids = [str(model.get("hf_id")) for model in model_specs]
    families = list(dict.fromkeys(str(task.get("family")) for task in manifest.get("tasks", [])))
    complete = [row for row in rows if row.get("row_status") == "completed"]
    expected_ids = {
        unit_id(model_id, str(task.get("task_id")), arm)
        for model_id in model_ids
        for task in manifest.get("tasks", [])
        for arm in ARM_ORDER
    }
    observed_ids = [str(row.get("unit_id")) for row in rows]
    exact_by_model = {
        model_id: _arm_metrics([row for row in rows if row.get("hf_id") == model_id], "exact")
        for model_id in model_ids
    }
    exact_by_family = {
        family: _arm_metrics([row for row in rows if row.get("family") == family], "exact")
        for family in families
    }
    parse_by_model = {
        model_id: _arm_metrics([row for row in rows if row.get("hf_id") == model_id], "parse")
        for model_id in model_ids
    }
    parse_by_family = {
        family: _arm_metrics([row for row in rows if row.get("family") == family], "parse")
        for family in families
    }
    paired: dict[str, JsonDict] = {}
    for model_id in [*model_ids, "overall"]:
        scoped = (
            rows if model_id == "overall" else [row for row in rows if row.get("hf_id") == model_id]
        )
        paired[model_id] = {
            arm: _paired_metrics(scoped, arm) for arm in ARM_ORDER if arm != "natural"
        }
    held_rows: list[JsonDict] = []
    for model_id in model_ids:
        for family in families:
            scoped = [
                row for row in rows if row.get("hf_id") == model_id and row.get("family") == family
            ]
            held_rows.append(
                {
                    "hf_id": model_id,
                    "family": family,
                    "exact_outcomes": _arm_metrics(scoped, "exact"),
                    "paired_deltas": {
                        arm: _paired_metrics(scoped, arm) for arm in ARM_ORDER if arm != "natural"
                    },
                }
            )
    harmful = [
        {
            "unit_id": row.get("unit_id"),
            "baseline_unit_id": row.get("baseline_unit_id"),
            "hf_id": row.get("hf_id"),
            "task_id": row.get("task_id"),
            "family": row.get("family"),
            "arm": row.get("arm"),
            "baseline_exact_success": True,
            "constrained_exact_success": False,
            "checker_sha256": dict(row.get("exact_outcome", {})).get("checker_sha256"),
        }
        for row in rows
        if row.get("harmful_flip") is True
    ]
    counts = Counter(observed_ids)
    missing_rows = sum(row.get("row_status") == "missing" for row in rows)
    exact_summary = {
        "overall": _arm_metrics(rows, "exact"),
        "per_model": exact_by_model,
        "per_held_family": exact_by_family,
        "paired_deltas": paired,
        "missing_rows": missing_rows,
        "completed_rows": len(complete),
    }
    parse_summary = {
        "overall": _arm_metrics(rows, "parse"),
        "per_model": parse_by_model,
        "per_held_family": parse_by_family,
        "missing_rows": missing_rows,
        "completed_rows": len(complete),
    }
    aggregate = {
        "expected_unit_count": len(expected_ids),
        "observed_unit_count": len(rows),
        "unique_unit_count": len(counts),
        "missing_expected_unit_ids": sorted(expected_ids - set(observed_ids)),
        "unexpected_unit_ids": sorted(set(observed_ids) - expected_ids),
        "duplicate_unit_ids": sorted(key for key, count in counts.items() if count != 1),
        "explicit_missing_row_count": missing_rows,
        "completed_row_count": len(complete),
        "all_rows_terminal": all(
            row.get("row_status") == "completed"
            or (row.get("row_status") == "missing" and bool(row.get("missing_cause")))
            for row in rows
        ),
        "headlines_sha256": sha256_json(
            {
                "exact_success_summary": exact_summary,
                "parse_transport_summary": parse_summary,
                "held_family_rows": held_rows,
                "harmful_flip_rows": harmful,
            }
        ),
    }
    return {
        "exact_success_summary": exact_summary,
        "parse_transport_summary": parse_summary,
        "held_family_rows": held_rows,
        "harmful_flip_rows": harmful,
        "aggregate_row_recomputation": aggregate,
    }


def process_receipt_failures(
    receipt: Mapping[str, Any],
    model: Mapping[str, Any],
    *,
    expected_inference_count: int,
) -> list[str]:
    """Validate process identity, ownership, CUDA residence, unload, and release."""

    failures: list[str] = []
    if receipt.get("receipt_sha256") != process_receipt_hash(receipt):
        failures.append("receipt_hash_mismatch")
    if receipt.get("hf_id") != model.get("hf_id") or receipt.get("model_sha256") != model.get(
        "model_sha256"
    ):
        failures.append("model_identity_mismatch")
    if not isinstance(receipt.get("pid"), int) or not isinstance(
        receipt.get("pid_start_ticks"), int
    ):
        failures.append("process_identity_missing")
    if receipt.get("parent_pid") != receipt.get("worker_pid"):
        failures.append("process_not_owned_child")
    if receipt.get("port_owner_pid") != receipt.get("pid"):
        failures.append("port_owner_mismatch")
    if (
        not str(receipt.get("owner_token_digest", "")).startswith("sha256:")
        or receipt.get("owner_token_opaque") is not True
    ):
        failures.append("owner_token_missing")
    if list(receipt.get("phase_sequence", [])) != list(COMPLETE_PHASE_SEQUENCE):
        failures.append("phase_sequence_mismatch")
    if receipt.get("cuda_offload") is not True or not receipt.get("cuda_uuid"):
        failures.append("cuda_residency_missing")
    if int(receipt.get("vram_resident_mb", 0) or 0) <= int(receipt.get("vram_before_mb", 0) or 0):
        failures.append("resident_vram_missing")
    if receipt.get("inference_count") != expected_inference_count:
        failures.append("inference_count_mismatch")
    if (
        receipt.get("server_exit_code") is None
        or receipt.get("server_absent_after_exit") is not True
    ):
        failures.append("server_exit_missing")
    if receipt.get("port_released") is not True:
        failures.append("port_not_released")
    if receipt.get("vram_recovered") is not True or receipt.get("unload_observed") is not True:
        failures.append("unload_not_proved")
    if receipt.get("lease_released") is not True:
        failures.append("lease_not_released")
    if receipt.get("release_phase") != "terminal_complete":
        failures.append("release_phase_invalid")
    if receipt.get("errors") not in ([], ()):
        failures.append("runtime_errors_present")
    return list(dict.fromkeys(failures))


def classify_verdict(summary: Mapping[str, Any], *, missing_count: int) -> str:
    """Classify evidence without promoting an executable-oracle result."""

    if missing_count:
        return "partial"
    exact = dict(summary.get("exact_success_summary", {}))
    paired = dict(exact.get("paired_deltas", {}))
    overall = dict(paired.get("overall", {}))
    triggered = dict(overall.get("triggered_tail", {}))
    interval = triggered.get("interval_95")
    if isinstance(interval, list) and len(interval) == 2 and interval[0] > 0:
        return "circular_positive"
    return "null"


def build_field_provenance(upstream: Mapping[str, Any]) -> dict[str, JsonDict]:
    """Name the raw source and reducer that owns every required field."""

    descriptions: dict[str, JsonDict] = {}
    for field in REQUIRED_ARTIFACT_FIELDS:
        row: JsonDict = {
            "source_path": str(UPSTREAM_PATH if field == "frozen_run_manifest" else MODULE_PATH),
            "function": "build_artifact",
            "raw_source": "retained Exp6676 rows and measured runtime receipts",
            "parser": "not_applicable",
            "checker": "not_applicable",
        }
        if field == "per_unit_rows":
            row.update(
                {
                    "raw_source": "llama.cpp /v1/chat/completions response bytes",
                    "parser": f"{fixture_api.__name__}.parse_arm_output",
                    "checker": f"{fixture_api.__name__}.check_certificate",
                    "function": "build_unit_row",
                }
            )
        elif field in {"exact_success_summary", "held_family_rows", "harmful_flip_rows"}:
            row.update(
                {
                    "raw_source": "per_unit_rows",
                    "parser": f"{fixture_api.__name__}.parse_arm_output",
                    "checker": f"{fixture_api.__name__}.check_certificate",
                    "function": "recompute_summaries",
                }
            )
        elif field == "parse_transport_summary":
            row.update(
                {
                    "raw_source": "per_unit_rows.parse_outcome",
                    "parser": f"{fixture_api.__name__}.parse_arm_output",
                    "function": "recompute_summaries",
                }
            )
        elif field == "per_model_process_receipts":
            row.update(
                {
                    "raw_source": "/proc, nvidia-smi, socket ownership, and GpuLease journal",
                    "function": "run_model_session",
                }
            )
        row["source_artifact_checksum"] = upstream.get("reproducibility_checksum")
        row["sha256"] = sha256_json({key: value for key, value in row.items() if key != "sha256"})
        descriptions[field] = row
    return descriptions


def _failure(check: str, expected: Any, observed: Any, reason: str) -> JsonDict:
    return {
        "check": check,
        "reason": reason,
        "expected_value": expected,
        "observed_value": deepcopy(observed),
    }


def _coverage_ready(aggregate: Mapping[str, Any]) -> bool:
    return bool(
        aggregate.get("expected_unit_count")
        == aggregate.get("observed_unit_count")
        == aggregate.get("unique_unit_count")
        and not aggregate.get("missing_expected_unit_ids")
        and not aggregate.get("unexpected_unit_ids")
        and not aggregate.get("duplicate_unit_ids")
        and aggregate.get("all_rows_terminal") is True
    )


def _models_used(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    completed = {str(row.get("hf_id")) for row in rows if row.get("row_status") == "completed"}
    return [spec["hf_id"] for spec in MODEL_SPECS if spec["hf_id"] in completed]


def build_artifact(
    *,
    date: str,
    duration_s: float,
    upstream_gate_receipt: Mapping[str, Any],
    model_specs: Sequence[Mapping[str, Any]],
    manifest: Mapping[str, Any],
    process_receipts: Sequence[Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
    preconditions: Mapping[str, Any],
    protected_receipt: Mapping[str, Any],
    tests_run: Sequence[Mapping[str, Any]],
    upstream_payload: Mapping[str, Any] | None = None,
) -> JsonDict:
    """Build a complete comparison artifact only from retained evidence."""

    annotated = annotate_harmful_flips(rows)
    summaries = recompute_summaries(annotated, manifest, model_specs)
    aggregate = summaries["aggregate_row_recomputation"]
    expected_per_model = len(manifest.get("tasks", [])) * len(ARM_ORDER)
    receipt_by_id = {str(row.get("hf_id")): row for row in process_receipts}
    receipt_failures: list[JsonDict] = []
    for model in model_specs:
        receipt = receipt_by_id.get(str(model.get("hf_id")))
        failures = (
            ["process_receipt_missing"]
            if receipt is None
            else process_receipt_failures(
                receipt, model, expected_inference_count=expected_per_model
            )
        )
        for reason in failures:
            receipt_failures.append(
                _failure(f"{model.get('family_id')}.{reason}", True, False, reason)
            )
    gate_failures: list[JsonDict] = []
    if upstream_gate_receipt.get("passed") is not True:
        gate_failures.append(
            _failure(
                "upstream.triggered_tail_fixture_ready",
                True,
                upstream_gate_receipt.get("observed_value"),
                "upstream_gate_failed",
            )
        )
    if preconditions.get("all_required_preconditions_available") is not True:
        for check in preconditions.get("failed_preconditions", []):
            gate_failures.append(
                _failure(
                    f"precondition.{check}",
                    True,
                    dict(preconditions.get("checks", {})).get(check),
                    "runtime_precondition_failed",
                )
            )
    if protected_receipt.get("all_unchanged") is not True:
        gate_failures.append(
            _failure("protected_files_unchanged", True, False, "protected_file_changed")
        )
    if not _coverage_ready(aggregate):
        gate_failures.append(_failure("unit_coverage", True, aggregate, "unit_coverage_incomplete"))
    gate_failures.extend(receipt_failures)
    ready = not gate_failures
    missing_count = int(aggregate.get("explicit_missing_row_count", 0) or 0)
    verdict_class = classify_verdict(summaries, missing_count=missing_count) if ready else "blocked"
    status = (
        "complete_partial"
        if ready and verdict_class == "partial"
        else f"complete_{verdict_class}"
        if ready
        else "blocked_runtime_or_evidence"
    )
    triggered = dict(
        dict(dict(summaries["exact_success_summary"]).get("paired_deltas", {})).get("overall", {})
    ).get("triggered_tail", {})
    honest = (
        "complete: three-family transport rows are complete; exact paired delta "
        f"is {dict(triggered).get('delta')} with executable checkers as oracle"
        if ready
        else "blocked_runtime_or_evidence: one or more deterministic gates failed"
    )
    upstream_source = dict(upstream_payload or {})
    artifact: JsonDict = {
        "status": status,
        "honest_verdict": honest,
        "verdict_class": verdict_class,
        "gate_check_summary": gate_failures,
        "model_specs": [deepcopy(dict(row)) for row in model_specs],
        "models_used": _models_used(annotated),
        "frozen_run_manifest": deepcopy(dict(manifest)),
        "per_model_process_receipts": [deepcopy(dict(row)) for row in process_receipts],
        "per_unit_rows": annotated,
        "held_family_rows": summaries["held_family_rows"],
        "harmful_flip_rows": summaries["harmful_flip_rows"],
        "exact_success_summary": summaries["exact_success_summary"],
        "parse_transport_summary": summaries["parse_transport_summary"],
        "triggered_tail_ab_ready": ready,
        "aggregate_row_recomputation": aggregate,
        "preconditions_checked": deepcopy(dict(preconditions)),
        "protected_files_unchanged": deepcopy(dict(protected_receipt)),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": True,
        "field_provenance": build_field_provenance(upstream_source),
        "random_seed": {
            "base": RANDOM_SEED,
            "per_unit": deepcopy(dict(manifest.get("unit_seeds", {}))),
        },
        "duration_s": round(float(duration_s), 6),
        "tests_run": [deepcopy(dict(row)) for row in tests_run],
    }
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return artifact


def build_blocked_artifact(
    *,
    date: str,
    duration_s: float,
    upstream_gate_receipt: Mapping[str, Any],
    model_specs: Sequence[Mapping[str, Any]],
    manifest: Mapping[str, Any],
    preconditions: Mapping[str, Any],
    protected_receipt: Mapping[str, Any],
    tests_run: Sequence[Mapping[str, Any]],
    blocker: str,
    expected: Any,
    observed: Any,
    upstream_payload: Mapping[str, Any] | None = None,
) -> JsonDict:
    """Build a terminal blocker without inventing model or unit evidence."""

    summaries = recompute_summaries([], manifest, model_specs)
    source = dict(upstream_payload or {})
    artifact: JsonDict = {
        "status": "blocked_runtime_precondition",
        "honest_verdict": f"blocked_{blocker}: expected {expected!r}, observed {observed!r}",
        "verdict_class": "blocked",
        "gate_check_summary": [
            _failure(blocker, expected, observed, "deterministic_precondition_failed")
        ],
        "model_specs": [deepcopy(dict(row)) for row in model_specs],
        "models_used": [],
        "frozen_run_manifest": deepcopy(dict(manifest)),
        "per_model_process_receipts": [],
        "per_unit_rows": [],
        "held_family_rows": summaries["held_family_rows"],
        "harmful_flip_rows": [],
        "exact_success_summary": summaries["exact_success_summary"],
        "parse_transport_summary": summaries["parse_transport_summary"],
        "triggered_tail_ab_ready": False,
        "aggregate_row_recomputation": summaries["aggregate_row_recomputation"],
        "preconditions_checked": deepcopy(dict(preconditions)),
        "protected_files_unchanged": deepcopy(dict(protected_receipt)),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": True,
        "field_provenance": build_field_provenance(source),
        "random_seed": {
            "base": RANDOM_SEED,
            "per_unit": deepcopy(dict(manifest.get("unit_seeds", {}))),
        },
        "duration_s": round(float(duration_s), 6),
        "tests_run": [deepcopy(dict(row)) for row in tests_run],
    }
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> list[str]:
    """Fail closed on schema, row, aggregate, process, or checksum drift."""

    errors: list[str] = []
    missing = sorted(set(REQUIRED_ARTIFACT_FIELDS) - set(artifact))
    if missing:
        errors.extend(f"missing_field:{field}" for field in missing)
        return errors
    if artifact.get("verdict_class") not in VERDICT_CLASSES:
        errors.append("verdict_class_invalid")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate_invalid")
    if artifact.get("verifier_is_oracle") is not True:
        errors.append("oracle_boundary_missing")
    manifest = artifact.get("frozen_run_manifest")
    manifest = manifest if isinstance(manifest, Mapping) else {}
    if manifest.get("manifest_sha256") != frozen_manifest_hash(manifest):
        errors.append("manifest_hash_mismatch")
    rows = artifact.get("per_unit_rows")
    rows = rows if isinstance(rows, list) else []
    if any(
        not isinstance(row, Mapping) or row.get("row_sha256") != unit_row_hash(row) for row in rows
    ):
        errors.append("unit_row_invalid")
    models = artifact.get("model_specs")
    models = models if isinstance(models, list) else []
    if artifact.get("triggered_tail_ab_ready") is True:
        recomputed = recompute_summaries(rows, manifest, models)
        fields = (
            "exact_success_summary",
            "parse_transport_summary",
            "held_family_rows",
            "harmful_flip_rows",
            "aggregate_row_recomputation",
        )
        if any(artifact.get(field) != recomputed.get(field) for field in fields):
            errors.append("aggregate_recomputation_mismatch")
        expected_per_model = len(manifest.get("tasks", [])) * len(ARM_ORDER)
        receipts = artifact.get("per_model_process_receipts")
        receipts = receipts if isinstance(receipts, list) else []
        by_id = {
            str(receipt.get("hf_id")): receipt
            for receipt in receipts
            if isinstance(receipt, Mapping)
        }
        for model in models:
            receipt = by_id.get(str(model.get("hf_id")))
            if receipt is None or process_receipt_failures(
                receipt, model, expected_inference_count=expected_per_model
            ):
                errors.append("process_receipt_invalid")
                break
        if artifact.get("gate_check_summary"):
            errors.append("ready_artifact_has_gate_failures")
    else:
        if artifact.get("verdict_class") != "blocked" or not artifact.get("gate_check_summary"):
            errors.append("blocked_artifact_gate_invalid")
    provenance = artifact.get("field_provenance")
    provenance = provenance if isinstance(provenance, Mapping) else {}
    if set(REQUIRED_ARTIFACT_FIELDS) - set(provenance):
        errors.append("field_provenance_incomplete")
    if artifact.get("protected_files_unchanged", {}).get("all_unchanged") is not True:
        errors.append("protected_files_changed")
    if artifact.get("reproducibility_checksum") != artifact_checksum(artifact):
        errors.append("reproducibility_checksum_mismatch")
    return list(dict.fromkeys(errors))


def write_artifact_atomic(path: Path, artifact: Mapping[str, Any]) -> None:
    """Publish one complete JSON object with file and directory sync."""

    lease_api.write_json_atomic(path, artifact)


def protected_hashes(root: Path) -> dict[str, str]:
    """Hash files that this task is not authorized to change."""

    return {path.as_posix(): sha256_file(root / path) for path in PROTECTED_PATHS}


def protected_files_receipt(root: Path, before: Mapping[str, str]) -> JsonDict:
    """Prove byte identity of the active roadmap and conductor."""

    after = protected_hashes(root)
    return {
        "before_hashes": dict(before),
        "after_hashes": after,
        "rows": [
            {
                "path": path,
                "before_sha256": before.get(path),
                "after_sha256": after.get(path),
                "unchanged": before.get(path) == after.get(path),
            }
            for path in sorted(set(before) | set(after))
        ],
        "all_unchanged": bool(before) and dict(before) == after,
    }


def choose_ports(count: int) -> list[int]:  # pragma: no cover - host socket allocation.
    """Choose distinct loopback ports and verify that each is initially free."""

    ports: list[int] = []
    while len(ports) < count:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as handle:
            handle.bind(("127.0.0.1", 0))
            port = int(handle.getsockname()[1])
        if port not in ports:
            ports.append(port)
    return ports


def server_command(server: str, model: Mapping[str, Any], port: int) -> list[str]:
    """Build one single-slot CUDA server command for a frozen model file."""

    return [
        server,
        "--model",
        str(model.get("model_path", "")),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--ctx-size",
        "2048",
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
        "256",
        "--ubatch-size",
        "256",
        "--offline",
        "--jinja",
        "--reasoning",
        "auto",
        "--reasoning-format",
        "none",
        "--no-ui",
        "--log-verbosity",
        "4",
    ]


def _command_receipt(command: str, root: Path, timeout_s: float) -> JsonDict:
    started = time.monotonic()
    try:
        result = subprocess.run(
            command,
            cwd=root,
            shell=True,
            executable="/bin/bash",
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
        output = (result.stdout + "\n" + result.stderr).strip().splitlines()
        return {
            "command": command,
            "exit_code": result.returncode,
            "summary": output[-1] if output else "no output",
            "duration_s": round(time.monotonic() - started, 6),
            "stdout_sha256": sha256_text(result.stdout),
            "stderr_sha256": sha256_text(result.stderr),
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "command": command,
            "exit_code": 124,
            "summary": f"TimeoutExpired after {exc.timeout}s",
            "duration_s": round(time.monotonic() - started, 6),
            "stdout_sha256": sha256_text(""),
            "stderr_sha256": sha256_text(str(exc)),
        }


def run_verification_commands(root: Path) -> list[JsonDict]:  # pragma: no cover - real tools.
    """Run focused, coverage, spec, lint, and format checks before inference."""

    commands = (
        (FOCUSED_TEST_COMMAND, 300.0),
        (COVERAGE_RUN_COMMAND, 300.0),
        (COVERAGE_REPORT_COMMAND, 60.0),
        (SPEC_COVERAGE_COMMAND, 60.0),
        (RUFF_COMMAND, 60.0),
        (FORMAT_COMMAND, 60.0),
    )
    return [_command_receipt(command, root, timeout) for command, timeout in commands]


def _port_open(port: int) -> bool:  # pragma: no cover - host socket receipt.
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as handle:
        handle.settimeout(0.25)
        return handle.connect_ex(("127.0.0.1", int(port))) == 0


def _port_owner_pid(port: int) -> int | None:  # pragma: no cover - host process receipt.
    receipt = runtime_api._run_command(("ss", "-ltnpH", "sport", "=", f":{int(port)}"))
    if receipt.get("exit_code") != 0:
        return None
    matches = re.findall(r"pid=(\d+)", str(receipt.get("stdout", "")))
    return int(matches[0]) if len(set(matches)) == 1 else None


def _lease_locks_available(
    runtime_dir: Path, gpus: Sequence[Mapping[str, Any]]
) -> tuple[bool, list[JsonDict]]:  # pragma: no cover - kernel lock receipt.
    rows: list[JsonDict] = []
    available = True
    lease_dir = runtime_dir / "leases"
    lease_dir.mkdir(parents=True, exist_ok=True)
    for gpu in gpus:
        uuid = str(gpu.get("uuid", ""))
        path = lease_api.lock_path_for(lease_dir, uuid)
        descriptor = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
        locked = False
        try:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                locked = True
            except BlockingIOError:
                available = False
            rows.append({"device_uuid": uuid, "lock_path": str(path), "available": locked})
        finally:
            if locked:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)
    return available, rows


def _input_receipts(
    root: Path, upstream_gate: Mapping[str, Any], manifest: Mapping[str, Any]
) -> JsonDict:
    paths = (
        UPSTREAM_PATH,
        Path("results/experiment_6661_triggered_tail_fixture.json"),
        Path("python/carnot/experiment_6661_triggered_tail_fixture.py"),
        Path("python/carnot/experiment_6675_triggered_tail_scope_receipt.py"),
    )
    rows = [{"path": path.as_posix(), "sha256": sha256_file(root / path)} for path in paths]
    return {
        "rows": rows,
        "all_present": all(row["sha256"].startswith("sha256:") for row in rows),
        "upstream_artifact_sha256": upstream_gate.get("sha256"),
        "manifest_sha256": manifest.get("manifest_sha256"),
    }


def collect_preconditions(
    *,
    root: Path,
    runtime_dir: Path,
    upstream_gate: Mapping[str, Any],
    manifest: Mapping[str, Any],
    model_specs: Sequence[Mapping[str, Any]],
    protected_before: Mapping[str, str],
) -> JsonDict:  # pragma: no cover - host and model probes.
    """Measure cache, CUDA, resources, locks, ports, and workload exclusion."""

    gpus = runtime_api.gpu_inventory()
    compute = runtime_api._compute_processes()
    llama_cpp = runtime_api.llama_cpp_receipt()
    resources = runtime_api._host_resources(root)
    inputs = _input_receipts(root, upstream_gate, manifest)
    relevant_devices = {int(spec.get("device_index", -1)) for spec in model_specs}
    gpu_by_index = {int(row.get("index", -1)): row for row in gpus}
    relevant_uuids = {
        str(gpu_by_index[index].get("uuid")) for index in relevant_devices if index in gpu_by_index
    }
    conflicts = [row for row in compute if str(row.get("gpu_uuid")) in relevant_uuids]
    lock_ok, lock_rows = _lease_locks_available(
        runtime_dir, [gpu_by_index[i] for i in relevant_devices if i in gpu_by_index]
    )
    ports = [int(port) for port in manifest.get("ports", [])]
    port_rows = [{"port": port, "free_before_launch": not _port_open(port)} for port in ports]
    vram_rows: list[JsonDict] = []
    for device_index in sorted(relevant_devices):
        assigned = [
            spec for spec in model_specs if int(spec.get("device_index", -1)) == device_index
        ]
        required_mb = max(
            (
                math.ceil(int(spec.get("byte_count", 0) or 0) / (1024 * 1024)) + 1536
                for spec in assigned
            ),
            default=0,
        )
        free_mb = int(gpu_by_index.get(device_index, {}).get("memory_free_mb", 0) or 0)
        vram_rows.append(
            {
                "device_index": device_index,
                "device_uuid": gpu_by_index.get(device_index, {}).get("uuid"),
                "required_free_mb": required_mb,
                "observed_free_mb": free_mb,
                "sufficient": free_mb >= required_mb > 0,
            }
        )
    resolution_failures = model_resolution_failures(model_specs)
    checks = {
        "upstream_gate": upstream_gate.get("passed") is True,
        "input_receipts": inputs["all_present"] is True,
        "model_cache": len(model_specs) == 3
        and all(spec.get("resolved") is True for spec in model_specs),
        "gguf_magic": len(model_specs) == 3
        and all(spec.get("gguf_magic_valid") is True for spec in model_specs),
        "model_hashes": len(model_specs) == 3
        and all(str(spec.get("model_sha256", "")).startswith("sha256:") for spec in model_specs),
        "embedded_tokenizers": len(model_specs) == 3
        and all(spec.get("embedded_tokenizer_loadable") is True for spec in model_specs),
        "cuda_visibility": relevant_devices <= set(gpu_by_index)
        and llama_cpp.get("cuda_linked") is True,
        "vram": len(vram_rows) == len(relevant_devices)
        and all(row["sufficient"] for row in vram_rows),
        "ram": int(resources.get("ram_available_bytes", 0) or 0) >= 8 * 1024**3,
        "disk": int(resources.get("disk_free_bytes", 0) or 0) >= 1024**3,
        "lease_ownership": lock_ok,
        "port_ownership": len(port_rows) == 3
        and all(row["free_before_launch"] for row in port_rows),
        "no_conflicting_workload": not conflicts,
        "protected_hashes": len(protected_before) == len(PROTECTED_PATHS)
        and all(str(value).startswith("sha256:") for value in protected_before.values()),
    }
    failed = [name for name, passed in checks.items() if not passed]
    return {
        "all_required_preconditions_available": not failed and not resolution_failures,
        "checks": checks,
        "failed_preconditions": [
            *failed,
            *(str(row.get("check")) for row in resolution_failures),
        ],
        "upstream_gate": deepcopy(dict(upstream_gate)),
        "input_receipts": inputs,
        "cache": [deepcopy(dict(row)) for row in model_specs],
        "hardware": {
            "gpus": gpus,
            "compute_processes": compute,
            "conflicting_workload_rows": conflicts,
            "nvidia_smi_path": shutil.which("nvidia-smi"),
        },
        "vram_rows": vram_rows,
        "lease_probe_rows": lock_rows,
        "port_probe_rows": port_rows,
        "resources": resources,
        "tools": {
            "python_executable": sys.executable,
            "python_version": platform.python_version(),
            "llama_cpp": llama_cpp,
            "ss_path": shutil.which("ss"),
        },
        "protected_hashes_before": dict(protected_before),
        "runtime_dir": str(runtime_dir.resolve()),
        "download_allowed": False,
        "auto_tokenizer_allowed": False,
        "legacy_model_can_satisfy_headline": False,
    }


def _http_json(
    url: str, payload: Mapping[str, Any] | None = None, timeout_s: float = 1.0
) -> tuple[int, JsonDict]:  # pragma: no cover - live server request.
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="GET" if payload is None else "POST",
    )
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        raw = response.read()
        value = json.loads(raw.decode("utf-8"))
        return int(response.status), dict(value) if isinstance(value, Mapping) else {}


def _generation_response(port: int, request: Mapping[str, Any]) -> JsonDict:  # pragma: no cover
    started = time.monotonic()
    try:
        status, body = _http_json(
            f"http://127.0.0.1:{port}/v1/chat/completions",
            request,
            REQUEST_TIMEOUT_S,
        )
        return {
            "http_status": status,
            "latency_s": round(time.monotonic() - started, 6),
            "body": body,
        }
    except (OSError, TimeoutError, urllib.error.URLError, json.JSONDecodeError) as exc:
        return {
            "http_status": 0,
            "latency_s": round(time.monotonic() - started, 6),
            "body": {},
            "error": f"{type(exc).__name__}: {exc}",
        }


def _process_identity(pid: int, command: Sequence[str]) -> JsonDict:  # pragma: no cover
    try:
        executable = os.readlink(f"/proc/{pid}/exe")
    except OSError:
        executable = str(command[0])
    return {
        "pid": pid,
        "pid_start_ticks": lease_api.proc_start_ticks(pid),
        "parent_pid": os.getpid(),
        "executable": executable,
        "argv": list(command),
        "argv_sha256": sha256_json(list(command)),
    }


def run_model_session(
    *,
    model: Mapping[str, Any],
    manifest: Mapping[str, Any],
    port: int,
    runtime_dir: Path,
    root: Path,
) -> tuple[JsonDict, list[JsonDict]]:  # pragma: no cover - required live CUDA path.
    """Own one lease and server while collecting the complete task-arm matrix."""

    worker = lease_api.current_process_identity()
    family = str(model.get("family_id"))
    device_index = int(model.get("device_index", -1))
    gpus = runtime_api.gpu_inventory()
    gpu = next((row for row in gpus if row.get("index") == device_index), {})
    device_uuid = str(gpu.get("uuid", ""))
    before = runtime_api._gpu_snapshot(device_index, 0)
    server_info = runtime_api.llama_cpp_receipt()
    command = server_command(str(server_info.get("path", "")), model, port)
    receipt_id = f"process:{family}"
    process: subprocess.Popen[bytes] | None = None
    process_identity: JsonDict = {
        "pid": 0,
        "pid_start_ticks": None,
        "parent_pid": worker["pid"],
    }
    lease: lease_api.GpuLease | None = None
    owner: JsonDict = {}
    release: JsonDict = {}
    resident: JsonDict = {}
    after: JsonDict = {}
    port_owner: int | None = None
    inference_count = 0
    errors: list[str] = []
    rows: list[JsonDict] = []
    vram_recovered = False
    port_released = False
    stdout_path = runtime_dir / f"{family}.llama.stdout"
    stderr_path = runtime_dir / f"{family}.llama.stderr"
    try:
        lease = lease_api.GpuLease.acquire(
            runtime_dir=runtime_dir / "leases",
            task_id=f"exp6676-{family}",
            device_uuid=device_uuid,
            expected_model=str(model.get("model_path", "")),
            vram_before_mb=int(before.get("memory_used_mb", 0) or 0),
            ttl_s=SESSION_TIMEOUT_S,
        )
        owner = lease.owner_receipt()
        lease.transition("admitted")
        lease.transition("loading")
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(device_index)
        with stdout_path.open("wb") as stdout_handle, stderr_path.open("wb") as stderr_handle:
            process = subprocess.Popen(
                command,
                cwd=root,
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=stdout_handle,
                stderr=stderr_handle,
            )
        process_identity = _process_identity(process.pid, command)
        deadline = time.monotonic() + LOAD_TIMEOUT_S
        healthy = False
        next_heartbeat = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError(f"llama_server_load_exit:{process.returncode}")
            try:
                status, health = _http_json(f"http://127.0.0.1:{port}/health", timeout_s=0.5)
                healthy = status == 200 and health.get("status") == "ok"
            except (OSError, TimeoutError, urllib.error.URLError, json.JSONDecodeError):
                healthy = False
            if healthy:
                break
            if time.monotonic() >= next_heartbeat:
                lease.heartbeat()
                next_heartbeat = time.monotonic() + 10.0
            time.sleep(1.0)
        if not healthy:
            raise TimeoutError("llama_server_load_timeout")
        port_owner = _port_owner_pid(port)
        if port_owner != process.pid:
            raise RuntimeError(f"port_owner_mismatch:{port_owner}:{process.pid}")
        resident = runtime_api._gpu_snapshot(device_index, process.pid)
        if (
            server_info.get("cuda_linked") is not True
            or resident.get("model_pid_present") is not True
            or int(resident.get("memory_used_mb", 0) or 0)
            <= int(before.get("memory_used_mb", 0) or 0)
        ):
            raise RuntimeError("owner_bound_cuda_residency_missing")
        lease.transition("resident", vram_mb=int(resident.get("memory_used_mb", 0) or 0))
        lease.transition("inferencing")
        for task in manifest.get("tasks", []):
            for arm in ARM_ORDER:
                key = unit_id(str(model.get("hf_id")), str(task.get("task_id")), arm)
                seed = int(dict(manifest.get("unit_seeds", {}))[key])
                request = build_generation_request(task, dict(manifest.get("arms", {}))[arm], seed)
                response = _generation_response(port, request)
                inference_count += 1
                body = response.get("body")
                choices = body.get("choices", []) if isinstance(body, Mapping) else []
                if response.get("http_status") == 200 and choices:
                    rows.append(
                        build_unit_row(
                            model=model,
                            task=task,
                            arm=arm,
                            response=response,
                            process_receipt_id=receipt_id,
                            seed=seed,
                        )
                    )
                else:
                    rows.append(
                        build_missing_unit_row(
                            model=model,
                            task=task,
                            arm=arm,
                            process_receipt_id=receipt_id,
                            seed=seed,
                            cause=str(response.get("error", "invalid_http_response")),
                        )
                    )
                if inference_count % 6 == 0:
                    lease.heartbeat()
                if process.poll() is not None:
                    raise RuntimeError(f"llama_server_inference_exit:{process.returncode}")
    except Exception as exc:
        errors.append(f"{type(exc).__name__}: {exc}")
    finally:
        expected_keys = [(task, arm) for task in manifest.get("tasks", []) for arm in ARM_ORDER]
        present = {str(row.get("unit_id")) for row in rows}
        for task, arm in expected_keys:
            key = unit_id(str(model.get("hf_id")), str(task.get("task_id")), arm)
            if key not in present:
                rows.append(
                    build_missing_unit_row(
                        model=model,
                        task=task,
                        arm=arm,
                        process_receipt_id=receipt_id,
                        seed=int(dict(manifest.get("unit_seeds", {}))[key]),
                        cause=errors[-1] if errors else "session_ended_before_request",
                    )
                )
        if lease is not None and lease.document.get("phase") in {"resident", "inferencing"}:
            try:
                lease.transition("unloading")
            except lease_api.LeaseError as exc:
                errors.append(f"{type(exc).__name__}: {exc}")
        if process is not None:
            if process.poll() is None:
                process.send_signal(signal.SIGTERM)
                try:
                    process.wait(timeout=SHUTDOWN_TIMEOUT_S)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5.0)
            process_identity["exit_code"] = process.returncode
            process_identity["absent_after_exit"] = not Path(f"/proc/{process.pid}").exists()
        else:
            process_identity.update({"exit_code": None, "absent_after_exit": True})
        recovery_started = time.monotonic()
        baseline = int(before.get("memory_used_mb", 0) or 0)
        while time.monotonic() - recovery_started <= RECOVERY_TIMEOUT_S:
            after = runtime_api._gpu_snapshot(
                device_index, int(process_identity.get("pid", 0) or 0)
            )
            vram_recovered = (
                after.get("model_pid_present") is False
                and abs(int(after.get("memory_used_mb", 0) or 0) - baseline)
                <= RECOVERY_TOLERANCE_MB
            )
            if vram_recovered:
                break
            if lease is not None and lease.document.get("phase") not in lease_api.TERMINAL_PHASES:
                try:
                    lease.heartbeat()
                except lease_api.LeaseError:
                    pass
            time.sleep(1.0)
        port_released = not _port_open(port)
        if lease is not None:
            try:
                phase = str(lease.document.get("phase"))
                if phase == "unloading":
                    lease.transition(
                        "validating",
                        vram_mb=int(after.get("memory_used_mb", 0) or 0),
                        exit_code=int(process_identity.get("exit_code", 127) or 0),
                        unload_observed=bool(
                            process_identity.get("absent_after_exit") and port_released
                        ),
                    )
                    success = not errors and vram_recovered and port_released
                    lease.transition("terminal_complete" if success else "terminal_blocked")
                elif phase in {"preflight", "admitted", "loading"}:
                    lease.transition("terminal_blocked")
                if lease.document.get("phase") in lease_api.TERMINAL_PHASES:
                    release = lease.release()
                else:
                    lease.close()
            except lease_api.LeaseError as exc:
                errors.append(f"{type(exc).__name__}: {exc}")
                lease.close()
    phase_sequence: list[Any] = []
    release_phase = None
    if lease is not None:
        try:
            journal = lease_api.read_journal(lease.journal_path)
            phase_sequence = [event.get("phase") for event in journal.get("phase_history", [])]
            release_phase = journal.get("phase")
        except lease_api.LeaseError as exc:
            errors.append(f"{type(exc).__name__}: {exc}")
    receipt: JsonDict = {
        "receipt_id": receipt_id,
        "family_id": family,
        "hf_id": model.get("hf_id"),
        "model_path": model.get("model_path"),
        "model_sha256": model.get("model_sha256"),
        "worker_pid": worker["pid"],
        "worker_pid_start_ticks": worker["pid_start_ticks"],
        "pid": process_identity.get("pid"),
        "pid_start_ticks": process_identity.get("pid_start_ticks"),
        "parent_pid": process_identity.get("parent_pid"),
        "port": port,
        "port_owner_pid": port_owner,
        "owner_token_digest": owner.get("token_digest"),
        "owner_token_opaque": owner.get("token_opaque"),
        "phase_sequence": phase_sequence,
        "cuda_device_index": device_index,
        "cuda_uuid": device_uuid,
        "cuda_offload": resident.get("model_pid_present") is True,
        "vram_before_mb": int(before.get("memory_used_mb", 0) or 0),
        "vram_resident_mb": int(resident.get("memory_used_mb", 0) or 0),
        "vram_after_mb": int(after.get("memory_used_mb", 0) or 0),
        "inference_count": inference_count,
        "expected_inference_count": len(manifest.get("tasks", [])) * len(ARM_ORDER),
        "server_exit_code": process_identity.get("exit_code"),
        "server_absent_after_exit": process_identity.get("absent_after_exit"),
        "port_released": port_released,
        "vram_recovered": vram_recovered,
        "unload_observed": bool(process_identity.get("absent_after_exit") and port_released),
        "lease_released": release.get("released") is True,
        "release_phase": release_phase,
        "command": command,
        "command_sha256": sha256_json(command),
        "stdout_path": str(stdout_path),
        "stdout_sha256": sha256_file(stdout_path),
        "stderr_path": str(stderr_path),
        "stderr_sha256": sha256_file(stderr_path),
        "errors": errors,
    }
    receipt["receipt_sha256"] = process_receipt_hash(receipt)
    return receipt, rows


def run_model_sessions(
    *,
    model_specs: Sequence[Mapping[str, Any]],
    manifest: Mapping[str, Any],
    runtime_dir: Path,
    root: Path,
) -> tuple[list[JsonDict], list[JsonDict]]:  # pragma: no cover - required live CUDA path.
    """Run model sessions sequentially so a reused GPU never overlaps."""

    receipts: list[JsonDict] = []
    rows: list[JsonDict] = []
    ports = [int(port) for port in manifest.get("ports", [])]
    for model, port in zip(model_specs, ports, strict=True):
        receipt, model_rows = run_model_session(
            model=model,
            manifest=manifest,
            port=port,
            runtime_dir=runtime_dir,
            root=root,
        )
        receipts.append(receipt)
        rows.extend(model_rows)
    return receipts, rows


def _minimal_preconditions(upstream_gate: Mapping[str, Any]) -> JsonDict:
    passed = upstream_gate.get("passed") is True
    return {
        "all_required_preconditions_available": passed,
        "checks": {"upstream_gate": passed},
        "failed_preconditions": [] if passed else ["upstream_gate"],
        "upstream_gate": deepcopy(dict(upstream_gate)),
    }


def run(
    *,
    date: str,
    root: Path = REPO_ROOT,
    result_path: Path = RESULT_PATH,
    work_dir: Path = WORK_PATH,
) -> JsonDict:  # pragma: no cover - exercised by the required live command.
    """Run deterministic gates, matched inference, reduction, and atomic write."""

    started = time.monotonic()
    work_dir.mkdir(parents=True, exist_ok=True)
    protected_before = protected_hashes(root)
    upstream_gate, upstream = load_upstream_gate(root)
    models = resolve_model_specs()
    ports = choose_ports(len(MODEL_SPECS))
    manifest = build_frozen_run_manifest(upstream, ports=ports)
    if upstream_gate.get("passed") is not True:
        protected = protected_files_receipt(root, protected_before)
        artifact = build_blocked_artifact(
            date=date,
            duration_s=time.monotonic() - started,
            upstream_gate_receipt=upstream_gate,
            model_specs=models,
            manifest=manifest,
            preconditions=_minimal_preconditions(upstream_gate),
            protected_receipt=protected,
            tests_run=[],
            blocker="upstream.triggered_tail_fixture_ready",
            expected=True,
            observed=upstream_gate.get("observed_value"),
            upstream_payload=upstream,
        )
        errors = validate_artifact(artifact)
        if errors:
            raise ValueError("artifact_validation_failed:" + ",".join(errors))
        write_artifact_atomic(result_path, artifact)
        return artifact
    preconditions = collect_preconditions(
        root=root,
        runtime_dir=work_dir,
        upstream_gate=upstream_gate,
        manifest=manifest,
        model_specs=models,
        protected_before=protected_before,
    )
    tests_run = run_verification_commands(root)
    tests_passed = all(row.get("exit_code") == 0 for row in tests_run)
    if not preconditions.get("all_required_preconditions_available") or not tests_passed:
        failed = list(preconditions.get("failed_preconditions", []))
        if not tests_passed:
            failed.append("owned_verification")
        blocker = str(failed[0] if failed else "unknown_precondition")
        observed = (
            dict(preconditions.get("checks", {})).get(blocker)
            if blocker != "owned_verification"
            else [row for row in tests_run if row.get("exit_code") != 0]
        )
        protected = protected_files_receipt(root, protected_before)
        artifact = build_blocked_artifact(
            date=date,
            duration_s=time.monotonic() - started,
            upstream_gate_receipt=upstream_gate,
            model_specs=models,
            manifest=manifest,
            preconditions=preconditions,
            protected_receipt=protected,
            tests_run=tests_run,
            blocker=blocker,
            expected=True,
            observed=observed,
            upstream_payload=upstream,
        )
        errors = validate_artifact(artifact)
        if errors:
            raise ValueError("artifact_validation_failed:" + ",".join(errors))
        write_artifact_atomic(result_path, artifact)
        return artifact
    receipts, rows = run_model_sessions(
        model_specs=models,
        manifest=manifest,
        runtime_dir=work_dir,
        root=root,
    )
    protected = protected_files_receipt(root, protected_before)
    artifact = build_artifact(
        date=date,
        duration_s=time.monotonic() - started,
        upstream_gate_receipt=upstream_gate,
        model_specs=models,
        manifest=manifest,
        process_receipts=receipts,
        rows=rows,
        preconditions=preconditions,
        protected_receipt=protected,
        tests_run=tests_run,
        upstream_payload=upstream,
    )
    errors = validate_artifact(artifact)
    if errors:
        raise ValueError("artifact_validation_failed:" + ",".join(errors))
    write_artifact_atomic(result_path, artifact)
    return artifact


def _validate_path(path: Path) -> tuple[int, JsonDict]:
    artifact = _read_json(path)
    errors = validate_artifact(artifact) if artifact else ["artifact_missing_or_unreadable"]
    return (0 if not errors else 1), {"valid": not errors, "errors": errors}


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI wrapper.
    """Run Exp6676 or validate an existing terminal artifact."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--output", type=Path, default=RESULT_PATH)
    parser.add_argument("--work-dir", type=Path, default=WORK_PATH)
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args(argv)
    output = args.output if args.output.is_absolute() else REPO_ROOT / args.output
    work_dir = args.work_dir if args.work_dir.is_absolute() else REPO_ROOT / args.work_dir
    if args.validate:
        code, receipt = _validate_path(output)
        print(json.dumps(receipt, sort_keys=True))
        return code
    artifact = run(
        date=args.date,
        root=REPO_ROOT,
        result_path=output,
        work_dir=work_dir,
    )
    print(
        json.dumps(
            {
                "status": artifact["status"],
                "triggered_tail_ab_ready": artifact["triggered_tail_ab_ready"],
                "output": str(output),
            },
            sort_keys=True,
        )
    )
    return 0 if artifact["triggered_tail_ab_ready"] else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
