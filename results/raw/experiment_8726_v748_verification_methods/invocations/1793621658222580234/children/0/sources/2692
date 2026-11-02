"""Measure two bounded requests through the local ARC E3 generation path.

This experiment asks one transport-only action question of each required model.
It records whether the model generated a schema-valid action. It does not start
an ARC environment, execute an action, or claim a solve.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import tempfile
import time
from typing import Any

from carnot.experiment_artifacts import atomic_write_json
from carnot.inference.sota_models import resolve_cached_gguf


JsonDict = dict[str, Any]
EXPERIMENT_ID = "exp7113-arc-generation-liveness-recovery"
SCHEMA = "carnot.exp7113.v624_arc_generation_liveness.v1"
RUN_DATE = "20260907"
RESULT_PATH = Path("results/experiment_7113_v624_arc_generation_liveness.json")
RAW_ROOT = Path("/tmp/carnot-exp7113-arc-generation-liveness")
REGISTRY_PATH = Path("ops/arc_solve_registry.yaml")
SPEC_PATH = Path("openspec/capabilities/research-reporting/spec.md")
MODULE_PATH = Path("python/carnot/experiment_7113_v624_arc_generation_liveness.py")
SCRIPT_PATH = Path("scripts/experiments/experiment_7113_v624_arc_generation_liveness.py")
TEST_PATH = Path("tests/python/test_experiment_7113_v624_arc_generation_liveness.py")
E3_POLICY_PATH = Path("python/carnot/agentic/arc_competition_agent.py")
E3_RUNTIME_PATH = Path("python/carnot/agentic/arc_executable_world_model.py")
LEASE_PATH = Path("python/carnot/gpu_lease_phase_journal.py")
LEASE_RUNTIME_DIR = Path(os.environ.get("CARNOT_GPU_LEASE_RUNTIME_DIR", "/tmp/carnot-gpu-leases"))
INFERENCE_SUBSTRATE = "two bounded local E3 generation requests"
E3_REQUEST_PATH = "E3AgentPolicy._proposer -> LocalGGUFProposer.complete_text"
RANDOM_SEED = 7_113_202_609_07
TOKEN_BUDGET = 96
CONTEXT_BUDGET = 4096
REQUEST_TIMEOUT_S = 300
MODEL_LOAD_TIMEOUT_S = 900
SUBSTRATE_FLOORS = {"model_load_no_generation": 2.0, "model_bounded_generation": 10.0}
_MODEL_HASH_CACHE: dict[tuple[str, int, int], str | None] = {}

QWEN38_REPO_ID = "unsloth/Qwen3.8-27B-GGUF"
QWEN36_REPO_ID = "unsloth/Qwen3.6-35B-A3B-GGUF"
REQUIRED_MODEL_IDS = (QWEN38_REPO_ID, QWEN36_REPO_ID)
MODEL_SPECS: list[JsonDict] = [
    {
        "key": "arc_generator",
        "name": "Qwen3.8-27B",
        "repo_id": QWEN38_REPO_ID,
        "preferred_quantization": "Q4_K_M",
        "role": "current_arc_generator",
        "arc_generator_cell": True,
        "headline_cell": False,
        "legacy_small_model": False,
    },
    {
        "key": "sota_headline",
        "name": "Qwen3.6-35B-A3B",
        "repo_id": QWEN36_REPO_ID,
        "preferred_quantization": "Q4_K_M",
        "role": "mandated_sota_headline",
        "arc_generator_cell": False,
        "headline_cell": True,
        "legacy_small_model": False,
    },
]

ACTION_PROMPT = (
    "/no_think\n"
    "This is a bounded ARC E3 action-schema transport check, not a game solve. "
    "Return one JSON object and no other text. Use exactly the keys action and data. "
    "Choose any integer action from 1 through 5 and set data to null. Do not use markdown."
)

REQUIRED_ARTIFACT_FIELDS = (
    "field_principles",
    "preconditions_checked",
    "run_date",
    "MODEL_SPECS",
    "inference_substrate",
    "inference_substrate_class",
    "execution_venue",
    "duration_s",
    "source_artifact_hashes",
    "rows",
    "per_model_request_rows",
    "model_repo_ids",
    "resolved_model_paths",
    "resolved_model_hashes",
    "quantization_rows",
    "model_load_rows",
    "prompt_hashes",
    "generated_token_rows",
    "finish_reason_rows",
    "raw_output_hashes",
    "action_parse_rows",
    "action_schema_validity_rows",
    "request_timing_rows",
    "process_rows",
    "gpu_telemetry_rows",
    "legacy_smoke_rows",
    "headline_model_rows",
    "solve_provenance",
    "offline_reproduced",
    "arc_registry_hash_before",
    "arc_registry_hash_after",
    "arc_registry_delta",
    "arc_generation_liveness_ready_score",
    "random_seed",
    "reproducibility_checksum",
    "gate_check_summary",
    "verifier_is_oracle",
    "verdict_class",
    "honest_verdict",
)

FIELD_PRINCIPLES: JsonDict = {
    "field_principles": "Each field states why its evidence is needed.",
    "preconditions_checked": "Unavailable resources must stop calls before evidence can be invented.",
    "run_date": "A real date places this result after the reporting cutover.",
    "MODEL_SPECS": "Pinned model declarations prevent silent substitution.",
    "inference_substrate": "The method description separates request liveness from game evaluation.",
    "inference_substrate_class": "The closed class selects the correct duration floor.",
    "execution_venue": "The venue identifies where process and GPU evidence was observed.",
    "duration_s": "Wall time makes local model execution plausible and auditable.",
    "source_artifact_hashes": "Source hashes bind the measurement to its implementation and inputs.",
    "rows": "Canonical rows let an independent reader recompute all claims.",
    "per_model_request_rows": "One row per required model prevents aggregate completion from hiding a missing call.",
    "model_repo_ids": "Repository identities distinguish required models from substitutes.",
    "resolved_model_paths": "Exact local paths bind requests to cached files.",
    "resolved_model_hashes": "File hashes bind model names to exact bytes.",
    "quantization_rows": "Quantization affects memory use and must be known for reproduction.",
    "model_load_rows": "Load and cleanup receipts distinguish a resident process from a generation.",
    "prompt_hashes": "Prompt hashes prove both bounded requests used the intended input.",
    "generated_token_rows": "Nonzero decoded tokens are the direct liveness gate.",
    "finish_reason_rows": "Finish reasons distinguish normal completion, limits, and timeouts.",
    "raw_output_hashes": "Output hashes bind parser claims to external raw text.",
    "action_parse_rows": "Parse rows show whether output became an action without a fallback.",
    "action_schema_validity_rows": "Schema checks prevent prose or malformed actions from passing.",
    "request_timing_rows": "Per-request timings expose missing or implausible execution.",
    "process_rows": "PID evidence ties requests to real local server processes.",
    "gpu_telemetry_rows": "Before and after samples prove the server occupied the leased GPU during each request.",
    "legacy_smoke_rows": "An explicit empty list proves no legacy model satisfied a required cell.",
    "headline_model_rows": "Role rows identify the mandated headline and ARC generator cells.",
    "solve_provenance": "Development-proxy provenance prevents liveness from becoming hidden-game credit.",
    "offline_reproduced": "False states that no action or solve replay occurred.",
    "arc_registry_hash_before": "The starting hash anchors the no-registry-mutation check.",
    "arc_registry_hash_after": "The ending hash exposes any registry mutation.",
    "arc_registry_delta": "Zero prevents request liveness from adding solve credit.",
    "arc_generation_liveness_ready_score": "Readiness needs both real requests, tokens, valid actions, telemetry, and cleanup.",
    "random_seed": "A fixed sampler seed makes the two requests reproducible.",
    "reproducibility_checksum": "A canonical digest detects later artifact-field drift.",
    "gate_check_summary": "The first failed check preserves expected and observed values.",
    "verifier_is_oracle": "False prevents a structural validator from becoming a game oracle.",
    "verdict_class": "A closed verdict supports deterministic downstream routing.",
    "honest_verdict": "The terminal statement reports liveness without claiming a solve.",
}


def canonical_json(value: Any) -> str:
    """Return stable JSON for all content-addressed evidence."""

    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


def sha256_text(value: str) -> str:
    """Hash text after its exact UTF-8 encoding."""

    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: str | Path) -> str | None:
    """Hash exact file bytes, or return null when no file can be read."""

    try:
        digest = hashlib.sha256()
        with Path(path).open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return "sha256:" + digest.hexdigest()
    except OSError:
        return None


def _model_sha256(path: str | Path) -> str | None:
    """Hash a large model once per exact resolved-path, size, and mtime tuple."""

    try:
        source = Path(path).resolve(strict=True)
        stat = source.stat()
    except OSError:
        return None
    key = (str(source), stat.st_size, stat.st_mtime_ns)
    if key not in _MODEL_HASH_CACHE:
        _MODEL_HASH_CACHE[key] = sha256_file(source)
    return _MODEL_HASH_CACHE[key]


def payload_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash the artifact without its self-referential checksum field."""

    payload = {key: value for key, value in artifact.items() if key != "reproducibility_checksum"}
    return sha256_text(canonical_json(payload))


def gate_row(check: str, expected: Any, observed: Any, *, passed: bool | None = None) -> JsonDict:
    """Keep one exact expected-versus-observed gate result."""

    return {
        "check": check,
        "expected_value": deepcopy(expected),
        "observed_value": deepcopy(observed),
        "passed": bool(expected == observed) if passed is None else bool(passed),
        "terminal": True,
    }


def gate_summary(checks: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Name the first failure while retaining every gate row."""

    rows = [deepcopy(dict(row)) for row in checks]
    failed = next((row for row in rows if row.get("passed") is not True), None)
    return {
        "passed": failed is None,
        "failed_check": None if failed is None else failed.get("check"),
        "expected_value": True if failed is None else failed.get("expected_value"),
        "observed_value": True if failed is None else failed.get("observed_value"),
        "checks": rows,
    }


def _same_file(left: str | Path, right: str | Path) -> bool:
    try:
        return Path(left).resolve(strict=True) == Path(right).resolve(strict=True)
    except OSError:
        return False


def _quantization_from_name(name: str, preferred: str) -> str:
    match = re.search(r"(?i)(UD-)?Q(?:[2-8]|I\d)[A-Z0-9_\-]*", name)
    return match.group(0).upper() if match else preferred


def resolved_model_row(
    declared: Mapping[str, Any],
    path: str | Path,
    *,
    gpu_index: int,
    gpu_uuid: str,
) -> JsonDict:
    """Bind one declared repository to exact cached GGUF bytes."""

    source = Path(path).absolute()
    resolved = source.resolve(strict=False)
    size = source.stat().st_size if source.is_file() else None
    return {
        **deepcopy(dict(declared)),
        "cached_model_path": str(source),
        "resolved_model_path": str(resolved),
        "resolved_model_filename": source.name,
        "quantization": _quantization_from_name(
            source.name, str(declared["preferred_quantization"])
        ),
        "model_size_bytes": size,
        "model_hash": _model_sha256(source),
        "gpu_index": int(gpu_index),
        "gpu_uuid": str(gpu_uuid),
        "download_attempted": False,
    }


def _gguf_header_ok(path: str | Path) -> bool:
    try:
        with Path(path).open("rb") as handle:
            return handle.read(4) == b"GGUF"
    except OSError:
        return False


def model_spec_errors(specs: Sequence[Mapping[str, Any]]) -> list[str]:
    """Reject missing, substituted, changed, or non-GGUF model files."""

    errors: list[str] = []
    rows = list(specs)
    if [row.get("repo_id") for row in rows] != list(REQUIRED_MODEL_IDS):
        errors.append("required_model_order_mismatch")
    paths = [str(row.get("resolved_model_path") or "") for row in rows]
    if len(rows) != 2 or any(not path or not Path(path).is_file() for path in paths):
        errors.append("model_file_missing")
    if len(paths) != len(set(paths)):
        errors.append("resolved_model_paths_not_distinct")
    if len({row.get("gpu_uuid") for row in rows}) != len(rows):
        errors.append("gpu_assignments_not_distinct")
    for row, path in zip(rows, paths, strict=False):
        if path and Path(path).is_file() and not _gguf_header_ok(path):
            errors.append("gguf_header_invalid")
        if (
            path
            and Path(path).is_file()
            and row.get("model_size_bytes") != Path(path).stat().st_size
        ):
            errors.append("model_size_mismatch")
        if path and Path(path).is_file() and row.get("model_hash") != _model_sha256(path):
            errors.append("model_hash_mismatch")
        if row.get("download_attempted") is not False or row.get("legacy_small_model") is not False:
            errors.append("model_policy_mismatch")
        if not isinstance(row.get("quantization"), str) or not row.get("quantization"):
            errors.append("quantization_missing")
    return list(dict.fromkeys(errors))


def _json_action_candidates(text: str) -> list[Mapping[str, Any]]:
    decoder = json.JSONDecoder()
    rows: list[Mapping[str, Any]] = []
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, Mapping) and "action" in value:
            rows.append(value)
    return rows


def action_schema_errors(action: Any) -> list[str]:
    """Validate the bounded ARC action object without executing it."""

    if not isinstance(action, Mapping):
        return ["action_not_object"]
    errors: list[str] = []
    if set(action) != {"action", "data"}:
        errors.append("action_keys_invalid")
    action_id = action.get("action")
    if (
        isinstance(action_id, bool)
        or not isinstance(action_id, int)
        or action_id not in range(1, 7)
    ):
        errors.append("action_id_invalid")
    data = action.get("data")
    if isinstance(action_id, int) and not isinstance(action_id, bool) and action_id == 6:
        if (
            not isinstance(data, Mapping)
            or set(data) != {"x", "y"}
            or any(
                isinstance(data.get(key), bool) or not isinstance(data.get(key), int)
                for key in ("x", "y")
            )
        ):
            errors.append("click_data_invalid")
    elif data is not None:
        errors.append("non_click_data_must_be_null")
    return errors


def parse_action(raw_output: str) -> JsonDict:
    """Parse one action only from model output and never install a fallback."""

    candidates = _json_action_candidates(str(raw_output))
    if len(candidates) != 1:
        reason = "no_json_action" if not candidates else "multiple_json_actions"
        return {
            "parse_status": "rejected",
            "parse_reason": reason,
            "proposed_action": None,
            "action_schema_valid": False,
            "action_schema_errors": [reason],
            "action_source": None,
        }
    action = deepcopy(dict(candidates[0]))
    errors = action_schema_errors(action)
    if errors:
        return {
            "parse_status": "rejected",
            "parse_reason": errors[0],
            "proposed_action": None,
            "action_schema_valid": False,
            "action_schema_errors": errors,
            "action_source": None,
        }
    return {
        "parse_status": "parsed",
        "parse_reason": "schema_valid",
        "proposed_action": action,
        "action_schema_valid": True,
        "action_schema_errors": [],
        "action_source": "parsed_raw_output",
    }


def write_raw_trace(path: str | Path, payload: Mapping[str, Any]) -> Path:
    """Write bulky prompt and output text outside the aggregate artifact."""

    target = Path(path)
    # REQ-REPORT-7113 deliberately keeps these traces outside results/. They are not
    # published artifacts, so a suite-level artifact-root override must not relocate them.
    atomic_write_json(target, dict(payload), allow_override=False)
    return target


def make_request_row(
    *,
    request_id: str,
    model: Mapping[str, Any],
    prompt: str,
    raw_output: str,
    prompt_tokens: int,
    generated_tokens: int,
    finish_reason: str,
    timed_out: bool,
    model_loaded: bool,
    observed_model_path: str | None,
    server_pid: int | None,
    process_identity: Mapping[str, Any],
    gpu_telemetry: Sequence[Mapping[str, Any]],
    load_started_s: float,
    load_finished_s: float,
    request_started_s: float,
    request_finished_s: float,
    raw_trace_path: str | Path,
    lease: Mapping[str, Any],
) -> JsonDict:
    """Create one request receipt whose action comes only from raw output."""

    parsed = (
        parse_action(raw_output)
        if not timed_out
        else {
            "parse_status": "rejected",
            "parse_reason": "request_timeout",
            "proposed_action": None,
            "action_schema_valid": False,
            "action_schema_errors": ["request_timeout"],
            "action_source": None,
        }
    )
    path = Path(raw_trace_path).absolute()
    return {
        "request_id": str(request_id),
        "model_repo_id": model.get("repo_id"),
        "model_role": model.get("role"),
        "resolved_model_path": model.get("resolved_model_path"),
        "resolved_model_hash": model.get("model_hash"),
        "observed_model_path": observed_model_path,
        "model_identity_match": bool(
            observed_model_path
            and _same_file(str(model.get("resolved_model_path") or ""), observed_model_path)
        ),
        "gpu_index": model.get("gpu_index"),
        "gpu_uuid": model.get("gpu_uuid"),
        "e3_request_path": E3_REQUEST_PATH,
        "request_count": 1,
        "generation_invoked": True,
        "model_loaded": bool(model_loaded),
        "model_download_attempted": False,
        "prompt_hash": sha256_text(prompt),
        "prompt_token_count": int(prompt_tokens),
        "generated_token_count": int(generated_tokens),
        "finish_reason": str(finish_reason),
        "timed_out": bool(timed_out),
        "raw_output_hash": sha256_text(raw_output),
        **parsed,
        "load_started_s": float(load_started_s),
        "load_finished_s": float(load_finished_s),
        "load_duration_s": round(float(load_finished_s) - float(load_started_s), 6),
        "request_started_s": float(request_started_s),
        "request_finished_s": float(request_finished_s),
        "request_duration_s": round(float(request_finished_s) - float(request_started_s), 6),
        "server_pid": server_pid,
        "process_identity": deepcopy(dict(process_identity)),
        "gpu_telemetry": [deepcopy(dict(row)) for row in gpu_telemetry],
        "raw_trace_path": str(path),
        "raw_trace_hash": sha256_file(path),
        "lease_id": lease.get("lease_id"),
        "lease_released": lease.get("released") is True,
        "unload_observed": lease.get("unload_observed") is True,
        "cleanup_signals_sent": deepcopy(list(lease.get("signals_sent") or [])),
    }


def _raw_binding_errors(row: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    path = Path(str(row.get("raw_trace_path") or ""))
    if "results" in path.parts:
        errors.append("raw_trace_inside_results")
    if sha256_file(path) != row.get("raw_trace_hash"):
        errors.append("raw_trace_hash_mismatch")
        return errors
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        errors.append("raw_trace_unreadable")
        return errors
    output = raw.get("raw_output")
    prompt = raw.get("prompt")
    if not isinstance(output, str) or sha256_text(output) != row.get("raw_output_hash"):
        errors.append("raw_output_hash_mismatch")
    if not isinstance(prompt, str) or sha256_text(prompt) != row.get("prompt_hash"):
        errors.append("prompt_hash_mismatch")
    parsed = (
        parse_action(output)
        if isinstance(output, str) and row.get("timed_out") is not True
        else {
            "parse_status": "rejected",
            "parse_reason": "request_timeout",
            "proposed_action": None,
            "action_schema_valid": False,
            "action_schema_errors": ["request_timeout"],
            "action_source": None,
        }
    )
    for field in (
        "parse_status",
        "parse_reason",
        "proposed_action",
        "action_schema_valid",
        "action_schema_errors",
        "action_source",
    ):
        if row.get(field) != parsed[field]:
            errors.append("raw_parse_receipt_mismatch")
            break
    return errors


def request_structure_errors(
    rows: Sequence[Mapping[str, Any]], model_specs: Sequence[Mapping[str, Any]]
) -> list[str]:
    """Find evidence contradictions that disqualify a measurement."""

    errors = model_spec_errors(model_specs)
    request_rows = list(rows)
    specs = list(model_specs)
    if len(request_rows) != 2:
        errors.append("required_request_count_mismatch")
    if [row.get("model_repo_id") for row in request_rows] != list(REQUIRED_MODEL_IDS):
        errors.append("request_model_order_mismatch")
    for row, model in zip(request_rows, specs, strict=False):
        observed_identity_match = bool(
            row.get("observed_model_path")
            and _same_file(
                str(model.get("resolved_model_path") or ""),
                str(row.get("observed_model_path")),
            )
        )
        if (
            row.get("resolved_model_path") != model.get("resolved_model_path")
            or row.get("resolved_model_hash") != model.get("model_hash")
            or row.get("model_identity_match") is not observed_identity_match
            or observed_identity_match is not True
        ):
            errors.append("runtime_model_path_mismatch")
        if row.get("request_count") != 1 or row.get("generation_invoked") is not True:
            errors.append("one_request_contract_mismatch")
        if row.get("e3_request_path") != E3_REQUEST_PATH:
            errors.append("e3_request_path_mismatch")
        if row.get("model_download_attempted") is not False:
            errors.append("model_download_policy_mismatch")
        if row.get("prompt_hash") != sha256_text(ACTION_PROMPT):
            errors.append("prompt_hash_mismatch")
        if (
            isinstance(row.get("prompt_token_count"), bool)
            or not isinstance(row.get("prompt_token_count"), int)
            or int(row.get("prompt_token_count", -1)) < 0
        ):
            errors.append("prompt_token_count_missing")
        if (
            isinstance(row.get("generated_token_count"), bool)
            or not isinstance(row.get("generated_token_count"), int)
            or int(row.get("generated_token_count", -1)) < 0
        ):
            errors.append("generated_token_count_invalid")
        parsed = row.get("parse_status") == "parsed"
        if parsed:
            if (
                row.get("action_source") != "parsed_raw_output"
                or row.get("action_schema_valid") is not True
                or action_schema_errors(row.get("proposed_action"))
            ):
                errors.append("parsed_action_receipt_invalid")
        elif row.get("proposed_action") is not None or row.get("action_source") is not None:
            errors.append("synthetic_action_leakage")
        pid = row.get("server_pid")
        process = row.get("process_identity")
        if (
            not isinstance(pid, int)
            or not isinstance(process, Mapping)
            or process.get("pid") != pid
        ):
            errors.append("process_telemetry_missing")
        telemetry = row.get("gpu_telemetry")
        telemetry = list(telemetry) if isinstance(telemetry, list) else []
        if [sample.get("phase") for sample in telemetry] != [
            "before_request",
            "after_request",
        ] or any(
            sample.get("sample_ok") is not True
            or sample.get("server_pid") != pid
            or sample.get("server_pid_visible") is not True
            or sample.get("gpu_uuid") != model.get("gpu_uuid")
            for sample in telemetry
        ):
            errors.append("gpu_telemetry_missing")
        load_started = row.get("load_started_s")
        load_finished = row.get("load_finished_s")
        request_started = row.get("request_started_s")
        request_finished = row.get("request_finished_s")
        numeric = all(
            isinstance(value, (int, float)) and not isinstance(value, bool)
            for value in (load_started, load_finished, request_started, request_finished)
        )
        if (
            not numeric
            or not (
                float(load_started)
                <= float(load_finished)
                <= float(request_started)
                <= float(request_finished)
            )
            or row.get("load_duration_s") != round(float(load_finished) - float(load_started), 6)
            or row.get("request_duration_s")
            != round(float(request_finished) - float(request_started), 6)
        ):
            errors.append("request_timing_invalid")
        if row.get("model_loaded") is not True:
            errors.append("model_load_incomplete")
        if row.get("lease_released") is not True or row.get("unload_observed") is not True:
            errors.append("gpu_cleanup_incomplete")
        if row.get("cleanup_signals_sent") != []:
            errors.append("unrelated_process_signal_detected")
        errors.extend(_raw_binding_errors(row))
    return list(dict.fromkeys(errors))


def liveness_failures(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    """Find honest liveness failures after receipt integrity is established."""

    errors: list[str] = []
    for row in rows:
        if int(row.get("generated_token_count") or 0) <= 0:
            errors.append("zero_generated_tokens")
        if row.get("timed_out") is True or row.get("finish_reason") == "timeout":
            errors.append("request_timeout")
        if row.get("parse_status") != "parsed" or row.get("action_schema_valid") is not True:
            errors.append("action_parser_rejection")
    return list(dict.fromkeys(errors))


def _substrate_class(rows: Sequence[Mapping[str, Any]]) -> str:
    return (
        "model_bounded_generation"
        if any(int(row.get("generated_token_count") or 0) > 0 for row in rows)
        else "model_load_no_generation"
    )


def _project_rows(
    model_specs: Sequence[Mapping[str, Any]], rows: Sequence[Mapping[str, Any]]
) -> JsonDict:
    specs = [deepcopy(dict(row)) for row in model_specs]
    requests = [deepcopy(dict(row)) for row in rows]
    return {
        "rows": requests,
        "per_model_request_rows": requests,
        "model_repo_ids": [row.get("repo_id") for row in specs],
        "resolved_model_paths": [row.get("resolved_model_path") for row in specs],
        "resolved_model_hashes": [row.get("model_hash") for row in specs],
        "quantization_rows": [
            {
                "model_repo_id": row.get("repo_id"),
                "resolved_model_filename": row.get("resolved_model_filename"),
                "quantization": row.get("quantization"),
                "model_size_bytes": row.get("model_size_bytes"),
            }
            for row in specs
        ],
        "model_load_rows": [
            {
                "model_repo_id": row.get("model_repo_id"),
                "model_loaded": row.get("model_loaded"),
                "model_load_duration_s": row.get("load_duration_s"),
                "observed_model_path": row.get("observed_model_path"),
                "model_identity_match": row.get("model_identity_match"),
                "lease_id": row.get("lease_id"),
                "lease_released": row.get("lease_released"),
                "unload_observed": row.get("unload_observed"),
            }
            for row in requests
        ],
        "prompt_hashes": [
            {"model_repo_id": row.get("model_repo_id"), "prompt_hash": row.get("prompt_hash")}
            for row in requests
        ],
        "generated_token_rows": [
            {
                "model_repo_id": row.get("model_repo_id"),
                "prompt_token_count": row.get("prompt_token_count"),
                "generated_token_count": row.get("generated_token_count"),
            }
            for row in requests
        ],
        "finish_reason_rows": [
            {
                "model_repo_id": row.get("model_repo_id"),
                "finish_reason": row.get("finish_reason"),
                "timed_out": row.get("timed_out"),
            }
            for row in requests
        ],
        "raw_output_hashes": [
            {
                "model_repo_id": row.get("model_repo_id"),
                "raw_output_hash": row.get("raw_output_hash"),
                "raw_trace_path": row.get("raw_trace_path"),
                "raw_trace_hash": row.get("raw_trace_hash"),
            }
            for row in requests
        ],
        "action_parse_rows": [
            {
                "model_repo_id": row.get("model_repo_id"),
                "parse_status": row.get("parse_status"),
                "parse_reason": row.get("parse_reason"),
                "proposed_action": deepcopy(row.get("proposed_action")),
                "action_source": row.get("action_source"),
            }
            for row in requests
        ],
        "action_schema_validity_rows": [
            {
                "model_repo_id": row.get("model_repo_id"),
                "action_schema_valid": row.get("action_schema_valid"),
                "action_schema_errors": deepcopy(row.get("action_schema_errors")),
            }
            for row in requests
        ],
        "request_timing_rows": [
            {
                "model_repo_id": row.get("model_repo_id"),
                "load_started_s": row.get("load_started_s"),
                "load_finished_s": row.get("load_finished_s"),
                "load_duration_s": row.get("load_duration_s"),
                "request_started_s": row.get("request_started_s"),
                "request_finished_s": row.get("request_finished_s"),
                "request_duration_s": row.get("request_duration_s"),
            }
            for row in requests
        ],
        "process_rows": [
            {
                "model_repo_id": row.get("model_repo_id"),
                "server_pid": row.get("server_pid"),
                **deepcopy(dict(row.get("process_identity") or {})),
            }
            for row in requests
        ],
        "gpu_telemetry_rows": [
            deepcopy(sample) for row in requests for sample in row.get("gpu_telemetry", [])
        ],
        "legacy_smoke_rows": [],
        "headline_model_rows": [
            {
                "model_repo_id": row.get("repo_id"),
                "role": row.get("role"),
                "arc_generator_cell": row.get("arc_generator_cell") is True,
                "headline_cell": row.get("headline_cell") is True,
                "legacy_small_model": row.get("legacy_small_model") is True,
            }
            for row in specs
        ],
    }


def build_artifact(
    *,
    run_date: str,
    duration_s: float,
    preconditions_checked: Sequence[Mapping[str, Any]],
    source_artifact_hashes: Mapping[str, Any],
    model_specs: Sequence[Mapping[str, Any]],
    request_rows: Sequence[Mapping[str, Any]],
    registry_hash_before: str | None,
    registry_hash_after: str | None,
) -> JsonDict:
    """Build a complete terminal artifact from immutable request receipts."""

    checks = [deepcopy(dict(row)) for row in preconditions_checked]
    preflight = gate_summary(checks)
    specs = [deepcopy(dict(row)) for row in model_specs]
    rows = [deepcopy(dict(row)) for row in request_rows]
    projections = _project_rows(specs, rows)
    blocked = preflight["passed"] is not True
    structural: list[str] = []
    liveness: list[str] = []
    if blocked:
        substrate_class = "blocked_no_run"
        verdict_class = "blocked"
        ready = 0
        rows = []
        projections = _project_rows(specs, rows)
        summary = preflight
    else:
        substrate_class = _substrate_class(rows)
        structural = request_structure_errors(rows, specs)
        if registry_hash_before is None or registry_hash_before != registry_hash_after:
            structural.append("arc_registry_hash_mismatch")
        floor = SUBSTRATE_FLOORS[substrate_class]
        if (
            not isinstance(duration_s, (int, float))
            or isinstance(duration_s, bool)
            or float(duration_s) < floor
        ):
            structural.append("duration_substrate_mismatch")
        structural = list(dict.fromkeys(structural))
        liveness = liveness_failures(rows)
        if structural:
            verdict_class = "disqualified"
        elif liveness:
            verdict_class = "null"
        else:
            verdict_class = "positive"
        ready = int(verdict_class == "positive")
        runtime_checks = [
            gate_row("request_evidence_integrity", [], structural),
            gate_row("both_required_models_live", [], liveness),
            gate_row("arc_generation_liveness_ready_score", 1, ready),
        ]
        summary = gate_summary([*checks, *runtime_checks])
    honest = {
        "positive": "complete_positive_arc_generation_liveness_ready_no_solve_claim",
        "null": "complete_null_arc_generation_liveness_not_ready_no_solve_claim",
        "blocked": "complete_blocked_arc_generation_liveness_precondition_failed",
        "disqualified": "complete_disqualified_arc_generation_liveness_evidence_invalid",
    }[verdict_class]
    artifact: JsonDict = {
        "field_principles": deepcopy(FIELD_PRINCIPLES),
        "preconditions_checked": checks,
        "run_date": str(run_date),
        "MODEL_SPECS": specs,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "inference_substrate_class": substrate_class,
        "execution_venue": "host",
        "duration_s": round(float(duration_s), 6),
        "source_artifact_hashes": deepcopy(dict(source_artifact_hashes)),
        **projections,
        "solve_provenance": "development_proxy",
        "offline_reproduced": False,
        "arc_registry_hash_before": registry_hash_before,
        "arc_registry_hash_after": registry_hash_after,
        "arc_registry_delta": 0,
        "arc_generation_liveness_ready_score": ready,
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
        "gate_check_summary": summary,
        "verifier_is_oracle": False,
        "verdict_class": verdict_class,
        "honest_verdict": honest,
    }
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    return artifact


def recompute_artifact(artifact: Mapping[str, Any]) -> JsonDict:
    """Rebuild every projection, score, class, verdict, gate, and checksum."""

    return build_artifact(
        run_date=str(artifact.get("run_date") or ""),
        duration_s=float(artifact.get("duration_s") or 0.0),
        preconditions_checked=list(artifact.get("preconditions_checked") or []),
        source_artifact_hashes=dict(artifact.get("source_artifact_hashes") or {}),
        model_specs=list(artifact.get("MODEL_SPECS") or []),
        request_rows=list(artifact.get("per_model_request_rows") or []),
        registry_hash_before=artifact.get("arc_registry_hash_before"),
        registry_hash_after=artifact.get("arc_registry_hash_after"),
    )


def validate_artifact(
    value: Mapping[str, Any] | str | Path, *, verify_raw_traces: bool = False
) -> list[str]:
    """Independently replay the complete Exp7113 artifact contract."""

    if isinstance(value, (str, Path)):
        try:
            artifact = json.loads(Path(value).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return ["artifact_unreadable"]
    elif isinstance(value, Mapping):
        artifact = deepcopy(dict(value))
    else:
        return ["artifact_not_object"]
    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    extra = [field for field in artifact if field not in REQUIRED_ARTIFACT_FIELDS]
    if missing or extra:
        return [
            *[f"missing_field:{field}" for field in missing],
            *[f"extra_field:{field}" for field in extra],
        ]
    errors: list[str] = []
    if artifact.get("field_principles") != FIELD_PRINCIPLES:
        errors.append("field_principles_mismatch")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate_mismatch")
    if artifact.get("execution_venue") != "host":
        errors.append("execution_venue_mismatch")
    if (
        artifact.get("solve_provenance") != "development_proxy"
        or artifact.get("offline_reproduced") is not False
    ):
        errors.append("solve_nonclaim_mismatch")
    if artifact.get("arc_registry_delta") != 0 or artifact.get("verifier_is_oracle") is not False:
        errors.append("authority_boundary_mismatch")
    recomputed = recompute_artifact(artifact)
    for field in REQUIRED_ARTIFACT_FIELDS:
        if field in {"source_artifact_hashes", "reproducibility_checksum"}:
            continue
        if artifact.get(field) != recomputed.get(field):
            if field == "inference_substrate_class":
                errors.append("inference_substrate_class_mismatch")
            else:
                errors.append(f"{field}_mismatch")
    if verify_raw_traces:
        recorded_structural = {
            item
            for check in artifact.get("gate_check_summary", {}).get("checks", [])
            if check.get("check") == "request_evidence_integrity"
            for item in check.get("observed_value", [])
        }
        for row in artifact.get("per_model_request_rows") or []:
            errors.extend(
                error for error in _raw_binding_errors(row) if error not in recorded_structural
            )
    if artifact.get("reproducibility_checksum") != payload_checksum(artifact):
        errors.append("reproducibility_checksum_mismatch")
    return list(dict.fromkeys(errors))


def _writable_target(path: Path) -> bool:  # pragma: no cover - host boundary
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, name = tempfile.mkstemp(prefix=".exp7113-write-", dir=path.parent)
        os.close(descriptor)
        Path(name).unlink()
        return True
    except OSError:
        return False


def _llama_server_receipt() -> JsonDict:  # pragma: no cover - binary boundary
    configured = os.environ.get("CARNOT_LLAMA_SERVER")
    path = (
        Path(configured)
        if configured
        else Path.home() / ".cache/llama.cpp-master/build/bin/llama-server"
    )
    if not path.is_file() or not os.access(path, os.X_OK):
        return {
            "path": str(path),
            "executable": False,
            "version_exit_code": None,
            "version": "",
            "cuda_linked": False,
            "sha256": sha256_file(path),
        }
    version = subprocess.run([str(path), "--version"], capture_output=True, text=True, check=False)
    linked = subprocess.run(["ldd", str(path)], capture_output=True, text=True, check=False)
    return {
        "path": str(path),
        "executable": path.is_file() and os.access(path, os.X_OK),
        "version_exit_code": version.returncode,
        "version": (version.stdout + version.stderr).strip()[:300],
        "cuda_linked": "libggml-cuda" in linked.stdout or "libcuda.so" in linked.stdout,
        "sha256": sha256_file(path),
    }


def nvidia_snapshot(
    phase: str, *, server_pid: int | None = None
) -> list[JsonDict]:  # pragma: no cover - GPU boundary
    try:
        gpu = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,uuid,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        apps = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=gpu_uuid,pid,used_memory",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return []
    process_by_uuid: dict[str, list[JsonDict]] = {}
    for line in apps.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 3:
            continue
        try:
            process_by_uuid.setdefault(parts[0], []).append(
                {"pid": int(parts[1]), "used_memory_mb": int(parts[2])}
            )
        except ValueError:
            continue
    rows = []
    for line in gpu.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 8:
            continue
        try:
            index, total, used, free, util, temperature = (
                int(parts[0]),
                int(parts[3]),
                int(parts[4]),
                int(parts[5]),
                int(parts[6]),
                int(parts[7]),
            )
        except ValueError:
            continue
        processes = process_by_uuid.get(parts[2], [])
        rows.append(
            {
                "phase": phase,
                "sample_ok": gpu.returncode == 0 and apps.returncode == 0,
                "gpu_index": index,
                "gpu_name": parts[1],
                "gpu_uuid": parts[2],
                "memory_total_mb": total,
                "memory_used_mb": used,
                "memory_free_mb": free,
                "utilization_pct": util,
                "temperature_c": temperature,
                "server_pid": server_pid,
                "server_pid_visible": server_pid is not None
                and any(row["pid"] == server_pid for row in processes),
                "process_rows": processes,
            }
        )
    return rows


def _resolve_models(
    gpus: Sequence[Mapping[str, Any]],
) -> list[JsonDict]:  # pragma: no cover - cache boundary
    try:
        paths = [
            resolve_cached_gguf(str(declared["repo_id"]), str(declared["preferred_quantization"]))
            for declared in MODEL_SPECS
        ]
    except (OSError, RuntimeError, ValueError):
        return []
    if any(path is None for path in paths) or len(gpus) < 2:
        return []
    return [
        resolved_model_row(
            declared,
            str(path),
            gpu_index=int(gpu["gpu_index"]),
            gpu_uuid=str(gpu["gpu_uuid"]),
        )
        for declared, path, gpu in zip(MODEL_SPECS, paths, gpus, strict=True)
    ]


def _source_hashes(root: Path) -> JsonDict:  # pragma: no cover - host boundary
    paths = (
        SPEC_PATH,
        MODULE_PATH,
        SCRIPT_PATH,
        TEST_PATH,
        E3_POLICY_PATH,
        E3_RUNTIME_PATH,
        LEASE_PATH,
        Path("results/experiment_7099_v623_adapter_withheld_preflight.json"),
    )
    return {path.as_posix(): sha256_file(root / path) for path in paths}


def gpu_is_idle_healthy(row: Mapping[str, Any]) -> bool:
    """Report whether one GPU telemetry row shows a cool, idle, empty, roomy card.

    An idle GPU reports exactly 0 percent utilization, and that is the state this
    experiment needs. Reading the field as `value or 999` turns that 0 into 999,
    because 0 is falsy, so the check rejected every truly idle card and could
    never pass. Read each field once and test for None, which keeps "the driver
    did not report this" apart from "the reported number is zero".
    """

    if row.get("sample_ok") is not True or row.get("process_rows"):
        return False
    temperature = row.get("temperature_c")
    utilization = row.get("utilization_pct")
    free_mb = row.get("memory_free_mb")
    if temperature is None or utilization is None or free_mb is None:
        return False
    return int(temperature) < 90 and int(utilization) <= 10 and int(free_mb) >= 20_000


def collect_preconditions(
    root: Path, output_path: Path, raw_root: Path
) -> JsonDict:  # pragma: no cover - host boundary
    """Check immutable inputs and allocate exact models to idle GPUs."""

    checks: list[JsonDict] = []
    initial = nvidia_snapshot("preflight")
    healthy = [row for row in initial if gpu_is_idle_healthy(row)]
    checks.append(gate_row("healthy_idle_gpus", 2, len(healthy), passed=len(healthy) >= 2))
    models = _resolve_models(healthy[:2])
    checks.append(gate_row("exact_cached_model_files", [], model_spec_errors(models)))
    runner = _llama_server_receipt()
    checks.append(
        gate_row(
            "cuda_llama_server",
            True,
            bool(
                runner["executable"] and runner["cuda_linked"] and runner["version_exit_code"] == 0
            ),
        )
    )
    runtime_paths = [root / E3_POLICY_PATH, root / E3_RUNTIME_PATH, root / LEASE_PATH]
    checks.append(
        gate_row(
            "readable_e3_runtime",
            True,
            all(path.is_file() and os.access(path, os.R_OK) for path in runtime_paths),
        )
    )
    checks.append(gate_row("writable_raw_path", True, _writable_target(raw_root / "probe.json")))
    checks.append(gate_row("writable_artifact_path", True, _writable_target(output_path)))
    registry = root / REGISTRY_PATH
    checks.append(
        gate_row(
            "readable_arc_registry",
            True,
            registry.is_file()
            and os.access(registry, os.R_OK)
            and sha256_file(registry) is not None,
        )
    )
    checks.append(gate_row("model_download_forbidden", False, False))
    checks.append(
        gate_row("gpu_lease_path_writable", True, _writable_target(LEASE_RUNTIME_DIR / "probe"))
    )
    return {
        "checks": checks,
        "summary": gate_summary(checks),
        "models": models,
        "initial_gpu_rows": initial,
        "runner": runner,
        "source_artifact_hashes": _source_hashes(root),
        "registry_hash": sha256_file(registry),
    }


def _free_port() -> int:  # pragma: no cover - operating-system boundary
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def _process_identity(pid: int) -> JsonDict:  # pragma: no cover - process boundary
    from carnot.gpu_lease_phase_journal import proc_start_ticks

    try:
        executable = os.readlink(f"/proc/{pid}/exe")
        cmdline = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").strip()
    except OSError:
        executable = ""
        cmdline = b""
    return {
        "pid": int(pid),
        "pid_start_ticks": proc_start_ticks(pid),
        "executable": executable,
        "cmdline_hash": "sha256:" + hashlib.sha256(cmdline).hexdigest(),
    }


def _gpu_sample_for_model(
    phase: str, model: Mapping[str, Any], server_pid: int
) -> JsonDict:  # pragma: no cover - GPU boundary
    rows = nvidia_snapshot(phase, server_pid=server_pid)
    row = next((item for item in rows if item.get("gpu_uuid") == model.get("gpu_uuid")), None)
    if row is None:
        return {
            "model_repo_id": model.get("repo_id"),
            "phase": phase,
            "gpu_index": model.get("gpu_index"),
            "gpu_uuid": model.get("gpu_uuid"),
            "sample_ok": False,
            "server_pid": server_pid,
            "server_pid_visible": False,
            "process_rows": [],
        }
    return {"model_repo_id": model.get("repo_id"), **row}


def _wait_for_unload(
    model: Mapping[str, Any], pid: int
) -> JsonDict:  # pragma: no cover - GPU boundary
    deadline = time.monotonic() + 60.0
    last = _gpu_sample_for_model("after_unload", model, pid)
    while last.get("server_pid_visible") is True and time.monotonic() < deadline:
        time.sleep(1.0)
        last = _gpu_sample_for_model("after_unload", model, pid)
    return last


def _run_model_request(
    model: Mapping[str, Any], raw_root: Path, lease: Any
) -> JsonDict:  # pragma: no cover - live local-model boundary
    """Load one model, call the real E3 completion seam once, then unload it."""

    from carnot.agentic.arc_competition_agent import E3AgentPolicy
    from carnot.agentic.arc_executable_world_model import LocalGGUFProposer

    old_env = {
        key: os.environ.get(key)
        for key in (
            "CARNOT_ARC_GENERATOR_CUDA_GPU",
            "CARNOT_ARC_GENERATOR_REQUIRE_CUDA",
            "CARNOT_ARC_GENERATOR_SEED",
            "CARNOT_LLAMA_SERVER",
            "HF_HUB_OFFLINE",
            "TRANSFORMERS_OFFLINE",
        )
    }
    os.environ.update(
        {
            "CARNOT_ARC_GENERATOR_CUDA_GPU": str(model["gpu_index"]),
            "CARNOT_ARC_GENERATOR_REQUIRE_CUDA": "1",
            "CARNOT_ARC_GENERATOR_SEED": str(RANDOM_SEED),
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        }
    )
    proposer = LocalGGUFProposer(
        repo_substr=str(model["name"]),
        model_path=str(model["resolved_model_path"]),
        model_repository=str(model["repo_id"]),
        model_filename=str(model["resolved_model_filename"]),
        requested_model_path=str(model["resolved_model_path"]),
        requested_model_filename=str(model["resolved_model_filename"]),
        port=_free_port(),
        n_ctx=CONTEXT_BUDGET,
        max_tokens=TOKEN_BUDGET,
        timeout=REQUEST_TIMEOUT_S,
        mtp=False,
        kv_quant="q8_0",
        n_gpu_layers=999,
        no_think_prefix="",
        use_chat_template=True,
    )
    policy = object.__new__(E3AgentPolicy)
    policy.proposer = proposer
    load_started = time.monotonic()
    server_pid: int | None = None
    raw_output = ""
    prompt_tokens = -1
    generated_tokens = -1
    finish_reason = "load_failed"
    timed_out = False
    before: list[JsonDict] = []
    after: list[JsonDict] = []
    process_identity: JsonDict = {}
    observed_path: str | None = None
    load_finished = load_started
    request_started = load_started
    request_finished = load_started
    unload = {"server_pid_visible": True, "memory_used_mb": 0}
    try:
        lease.transition("loading")
        if not proposer._ensure_server():
            raise RuntimeError("cuda_llama_server_load_failed")
        load_finished = time.monotonic()
        server_pid = int(proposer._proc.pid)
        observed_path = proposer.observed_model_path()
        process_identity = _process_identity(server_pid)
        before = [_gpu_sample_for_model("before_request", model, server_pid)]
        resident_mb = int(before[0].get("memory_used_mb") or 0)
        lease.transition("resident", vram_mb=resident_mb)
        lease.transition("inferencing")
        request_started = time.monotonic()
        calls_before = int(proposer.n_completion_calls)
        ok, raw_output = E3AgentPolicy._proposer(policy).complete_text(
            ACTION_PROMPT,
            max_tokens=TOKEN_BUDGET,
            temperature=0.0,
            stop=None,
        )
        request_finished = time.monotonic()
        if int(proposer.n_completion_calls) - calls_before != 1:
            raise RuntimeError("e3_request_count_mismatch")
        prompt_tokens = int(proposer.last_prompt_tokens)
        generated_tokens = max(0, int(proposer.last_generated_tokens))
        finish_reason = str(proposer.last_stop_type or ("error" if not ok else "unknown"))
        timed_out = bool(proposer.channel_totals.get("request_timeouts"))
        after = [_gpu_sample_for_model("after_request", model, server_pid)]
        raw_path = raw_root / str(model["key"]) / "request.json"
        write_raw_trace(
            raw_path,
            {
                "schema": SCHEMA + ".raw.v1",
                "request_id": str(model["key"]),
                "model_repo_id": model["repo_id"],
                "prompt": ACTION_PROMPT,
                "raw_output": raw_output,
            },
        )
    except Exception as exc:
        request_finished = time.monotonic()
        if request_started == load_started:
            load_finished = request_finished
            request_started = request_finished
        finish_reason = "timeout" if isinstance(exc, TimeoutError) else "error"
        timed_out = isinstance(exc, TimeoutError)
        raw_output = ""
        raw_path = raw_root / str(model["key"]) / "request.json"
        write_raw_trace(
            raw_path,
            {
                "schema": SCHEMA + ".raw.v1",
                "request_id": str(model["key"]),
                "model_repo_id": model["repo_id"],
                "prompt": ACTION_PROMPT,
                "raw_output": raw_output,
                "runtime_error": f"{type(exc).__name__}: {exc}",
            },
        )
    finally:
        if server_pid is not None:
            try:
                lease.transition("unloading")
            except Exception:
                pass
        proposer.stop()
        if server_pid is not None:
            unload = _wait_for_unload(model, server_pid)
        try:
            if server_pid is not None:
                lease.transition(
                    "validating",
                    vram_mb=int(unload.get("memory_used_mb") or 0),
                    exit_code=0,
                    unload_observed=unload.get("server_pid_visible") is False,
                )
                lease.transition("terminal_complete")
            else:
                lease.transition("terminal_blocked")
            release = lease.release()
        except Exception as exc:
            lease.close()
            release = {
                "lease_id": getattr(lease, "lease_id", None),
                "released": False,
                "signals_sent": [],
                "error": f"{type(exc).__name__}: {exc}",
            }
        for key, value in old_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    lease_receipt = {
        "lease_id": release.get("lease_id"),
        "released": release.get("released") is True,
        "unload_observed": unload.get("server_pid_visible") is False,
        "signals_sent": list(release.get("signals_sent") or []),
    }
    return make_request_row(
        request_id=str(model["key"]),
        model=model,
        prompt=ACTION_PROMPT,
        raw_output=raw_output,
        prompt_tokens=max(0, prompt_tokens),
        generated_tokens=max(0, generated_tokens),
        finish_reason=finish_reason,
        timed_out=timed_out,
        model_loaded=server_pid is not None,
        observed_model_path=observed_path,
        server_pid=server_pid,
        process_identity=process_identity,
        gpu_telemetry=[*before, *after],
        load_started_s=load_started,
        load_finished_s=load_finished,
        request_started_s=request_started,
        request_finished_s=request_finished,
        raw_trace_path=raw_path,
        lease=lease_receipt,
    )


def _acquire_all_leases(
    models: Sequence[Mapping[str, Any]],
) -> tuple[list[Any], str | None]:  # pragma: no cover - lease boundary
    from carnot import gpu_lease_phase_journal as lease_api

    leases = []
    try:
        for model in models:
            lease = lease_api.GpuLease.acquire(
                runtime_dir=LEASE_RUNTIME_DIR,
                task_id=EXPERIMENT_ID,
                device_uuid=str(model["gpu_uuid"]),
                expected_model=str(model["resolved_model_path"]),
                vram_before_mb=0,
                ttl_s=3600.0,
            )
            lease.transition("admitted")
            leases.append(lease)
        return leases, None
    except Exception as exc:
        for lease in leases:
            try:
                lease.transition("terminal_blocked")
                lease.release()
            except Exception:
                lease.close()
        return [], f"{type(exc).__name__}: {exc}"


def run(
    root: Path,
    *,
    run_date: str,
    output_path: Path,
    raw_root: Path,
) -> JsonDict:  # pragma: no cover - end-to-end hardware boundary
    """Preflight, acquire both leases, execute two requests, and publish once."""

    started = time.monotonic()
    preflight = collect_preconditions(root, output_path, raw_root)
    checks = list(preflight["checks"])
    models = list(preflight["models"])
    rows: list[JsonDict] = []
    leases: list[Any] = []
    if preflight["summary"]["passed"] is True:
        leases, lease_error = _acquire_all_leases(models)
        checks.append(
            gate_row(
                "gpu_leases_acquired",
                2,
                len(leases),
                passed=lease_error is None and len(leases) == 2,
            )
        )
    if gate_summary(checks)["passed"] is True:
        for model, lease in zip(models, leases, strict=True):
            rows.append(_run_model_request(model, raw_root, lease))
    registry_after = sha256_file(root / REGISTRY_PATH)
    artifact = build_artifact(
        run_date=run_date,
        duration_s=time.monotonic() - started,
        preconditions_checked=checks,
        source_artifact_hashes=preflight["source_artifact_hashes"],
        model_specs=models,
        request_rows=rows,
        registry_hash_before=preflight["registry_hash"],
        registry_hash_after=registry_after,
    )
    write_artifact(output_path, artifact)
    return artifact


def write_artifact(
    path: Path, artifact: Mapping[str, Any]
) -> Path:  # pragma: no cover - publication boundary
    errors = validate_artifact(
        artifact, verify_raw_traces=artifact.get("verdict_class") != "blocked"
    )
    if errors:
        raise ValueError("invalid Exp7113 artifact: " + "; ".join(errors))
    atomic_write_json(path, dict(artifact))
    return path


def _parser() -> argparse.ArgumentParser:  # pragma: no cover - CLI boundary
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--raw-root", type=Path)
    parser.add_argument("--validate", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI boundary
    args = _parser().parse_args(argv)
    if args.validate:
        errors = validate_artifact(args.validate, verify_raw_traces=True)
        print(json.dumps({"valid": not errors, "errors": errors}, sort_keys=True))
        return int(bool(errors))
    root = Path(__file__).resolve().parents[2]
    output = (args.output or root / RESULT_PATH).resolve()
    raw_root = (args.raw_root or RAW_ROOT / str(args.date)).resolve()
    artifact = run(root, run_date=str(args.date), output_path=output, raw_root=raw_root)
    print(f"Exp7113 {artifact['verdict_class']}: {artifact['honest_verdict']} -> {output}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
