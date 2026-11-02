#!/usr/bin/env python3
"""Exp5615 native llama.cpp CUDA runtime certificate.

Spec refs: REQ-VERIFY-5615, SCENARIO-VERIFY-5615.

This module changes only the inference substrate certificate.  It does not
rerun or interpret solve-versus-verify task accuracy.  The live path either
records two short native llama.cpp controls for each mandated GGUF model with
CUDA offload evidence, or writes a blocked certificate naming the missing
precondition before model loading.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
import hashlib
import json
import os
from pathlib import Path
import re
import selectors
import shutil
import subprocess
import threading
import time
from typing import Any

from carnot import experiment_5605_raw_response_evidence_envelope as exp5605
from carnot.inference.sota_models import SOTA_GGUF_MODELS, resolve_cached_gguf


JsonDict = dict[str, Any]
NativeRunner = Callable[..., JsonDict]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_5615_native_llamacpp_cuda_runtime_certificate.json")
RESPONSE_ENVELOPE_RELATIVE_PATH = Path(
    "results/experiment_5615_native_llamacpp_cuda_runtime_certificate.responses.jsonl"
)

SCHEMA = "carnot.experiment_5615.native_llamacpp_cuda_runtime_certificate.v507"
EXPERIMENT = 5615
EXPERIMENT_ID = "exp5615-native-llamacpp-cuda-runtime-certificate"
MILESTONE = "2026.07.507"
RUN_DATE = "20260714"
RANDOM_SEED = 5615
INFERENCE_SUBSTRATE = "live_llm_inference"
PARSER_NAME = "carnot.exp5615.native_runtime_certificate_parser"
PARSER_VERSION = SCHEMA
ENVELOPE_SCHEMA_VERSION = exp5605.ENVELOPE_SCHEMA_VERSION
SPEC_REFS = ("REQ-VERIFY-5615", "SCENARIO-VERIFY-5615", "REQ-VERIFY-5605")

QWEN_ID = "unsloth/Qwen3.6-35B-A3B-GGUF"
GEMMA31_ID = "unsloth/gemma-4-31B-it-GGUF"
GEMMA26_ID = "unsloth/gemma-4-26B-A4B-it-GGUF"
MANDATED_HEADLINE_IDS = (QWEN_ID, GEMMA31_ID, GEMMA26_ID)
LEGACY_SMOKE_IDS = ("Qwen/Qwen3.5-0.8B", "google/gemma-4-E4B-it")

POSITIVE_N_PREDICT = 64
TRUNCATED_N_PREDICT = 1
DEFAULT_CTX = 512
DEFAULT_BATCH = 256
DEFAULT_UBATCH = 128
DEFAULT_TIMEOUT_S = 900.0
CONTROL_JSON_SCHEMA = json.dumps(
    {
        "type": "object",
        "properties": {"certificate_control": {"const": "ok"}},
        "required": ["certificate_control"],
        "additionalProperties": False,
    },
    separators=(",", ":"),
)

FIELD_PRINCIPLES: JsonDict = {
    "field_principles": "Every certificate field names the evidence boundary it protects.",
    "model_specs": "All mandated identities, cache paths, and hashes are explicit so legacy smoke models cannot satisfy the certificate.",
    "native_binary_receipt": "The changed native llama.cpp substrate is authenticated before model load.",
    "cuda_build_capability": "Compiled CUDA offload support is real rather than inferred from a CPU binary.",
    "gpu_device_receipts": "Device identity, driver, free memory, PID memory, and process cleanup evidence are preserved.",
    "offload_layers_by_model": "Requested and observed offloaded layers stay separate.",
    "gpu_memory_delta_by_model": "CPU fallback cannot masquerade as offload.",
    "response_envelope_path": "Raw prompt and response evidence is replayable.",
    "lossless_replay_rate": "Preservation is exact for every row.",
    "stop_control_pass_rate": "Termination behavior is bounded by explicit controls.",
    "semantic_false_accept_count": "Parsing fails closed.",
    "orphan_process_count": "The native runtime cleans up.",
    "models_certified_count": "The denominator is three mandated models.",
    "runtime_certificate_ready_score": "Only a complete three-model native CUDA certificate may score 1.0.",
    "inference_substrate": "Declares that real local generation occurred when controls run.",
    "random_seed": "Fixed sampling makes the native arguments repeatable.",
    "reproducibility_checksum": "The method and evidence can be replayed.",
    "honest_verdict": "No-offload or failed-control evidence is terminal retirement evidence for this runtime certificate.",
}

REQUIRED_ARTIFACT_FIELDS = (
    "field_principles",
    "model_specs",
    "native_binary_receipt",
    "cuda_build_capability",
    "gpu_device_receipts",
    "offload_layers_by_model",
    "gpu_memory_delta_by_model",
    "response_envelope_path",
    "lossless_replay_rate",
    "stop_control_pass_rate",
    "semantic_false_accept_count",
    "orphan_process_count",
    "models_certified_count",
    "runtime_certificate_ready_score",
    "inference_substrate",
    "random_seed",
    "reproducibility_checksum",
    "honest_verdict",
)

CUDA_LIBRARY_RE = re.compile(r"(libggml-cuda|libcuda|libcublas|libcudart|CUDA\d)", re.I)
OFFLOAD_RE = re.compile(r"offloaded\s+(\d+)\s*/\s*(\d+)\s+layers", re.I)
LAYER_CUDA_RE = re.compile(r"layer\s+(\d+):\s+dev\s*=\s*CUDA", re.I)
PROMPT_TOKENS_RE = re.compile(r"prompt eval time\s*=\s*[0-9.]+\s*ms\s*/\s*(\d+)\s+tokens", re.I)
EVAL_RUNS_RE = re.compile(r"eval time\s*=\s*[0-9.]+\s*ms\s*/\s*(\d+)\s+runs", re.I)

encode_lossless_payload = exp5605.encode_lossless_payload
decode_lossless_payload = exp5605.decode_lossless_payload
encode_prompt = exp5605.encode_prompt
decode_prompt = exp5605.decode_prompt


class EnvelopeReplayError(ValueError):
    """Raised when a certificate response-envelope row cannot replay exactly."""


def canonical_json(value: Any) -> str:
    """Serialize JSON deterministically before hashing."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_bytes(payload: bytes) -> str:
    """Return a SHA-256 hex digest for byte evidence."""

    return hashlib.sha256(payload).hexdigest()


def sha256_text(value: str) -> str:
    """Return a SHA-256 hex digest for text evidence."""

    return sha256_bytes(value.encode("utf-8"))


def sha256_json(value: Any) -> str:
    """Return a SHA-256 hex digest for JSON-compatible evidence."""

    return sha256_text(canonical_json(value))


def sha256_file(path: str | Path) -> str:
    """Hash a local file in chunks so large GGUFs do not load into memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model_family(hf_id: str) -> str:
    """Return the stable per-model family label for the certificate."""

    if hf_id == QWEN_ID:
        return "qwen3.6-35b-a3b"
    if hf_id == GEMMA31_ID:
        return "gemma-4-31b-it"
    if hf_id == GEMMA26_ID:
        return "gemma-4-26b-a4b-it"
    return hf_id.rsplit("/", 1)[-1].replace("-GGUF", "").lower()


def normalize_model_specs(model_specs: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Normalize the three mandated model specs with cache path and hash evidence."""

    registry = {row["hf_id"]: row for row in SOTA_GGUF_MODELS}
    by_id = {str(row.get("hf_id")): row for row in model_specs if isinstance(row, Mapping)}
    normalized: list[JsonDict] = []
    for index, hf_id in enumerate(MANDATED_HEADLINE_IDS):
        source = by_id.get(hf_id, {})
        registry_row = registry.get(hf_id, {})
        path = str(source.get("model_path") or source.get("cache_path") or "")
        path_obj = Path(path).expanduser() if path else Path()
        present = bool(path and path_obj.is_file())
        normalized.append(
            {
                "name": str(
                    source.get("name") or registry_row.get("name") or hf_id.rsplit("/", 1)[-1]
                ),
                "hf_id": hf_id,
                "family": model_family(hf_id),
                "role": str(source.get("role") or registry_row.get("role") or ""),
                "gpu": int(source.get("gpu", index % 2) or 0),
                "model_path": path,
                "cache_path": path,
                "local_path_hash": exp5605.local_path_hash(path) if path else "",
                "model_sha256": str(source.get("model_sha256") or (sha256_file(path_obj) if present else "")),
                "local_model_present": present,
                "headline_eligible": source.get("headline_eligible") is not False,
                "active_params_b": source.get("active_params_b", registry_row.get("active_params_b")),
                "total_params_b": source.get("total_params_b", registry_row.get("total_params_b")),
                "quantization": str(
                    source.get("quantization") or registry_row.get("quantization") or "Q4_K_M"
                ),
                "legacy_smoke_label": None,
            }
        )
    return normalized


def resolve_all_headline_model_specs() -> list[JsonDict]:  # pragma: no cover - host dependent.
    """Resolve all three mandated GGUF files without downloading."""

    registry = {row["hf_id"]: row for row in SOTA_GGUF_MODELS}
    specs: list[JsonDict] = []
    for index, hf_id in enumerate(MANDATED_HEADLINE_IDS):
        row = registry[hf_id]
        path = resolve_cached_gguf(hf_id, str(row.get("quantization") or "Q4_K_M"))
        specs.append(
            {
                "name": row["name"],
                "hf_id": hf_id,
                "family": model_family(hf_id),
                "role": row["role"],
                "gpu": index % 2,
                "model_path": path or "",
                "headline_eligible": True,
                "active_params_b": row["active_params_b"],
                "total_params_b": row["total_params_b"],
                "quantization": row["quantization"],
            }
        )
    return normalize_model_specs(specs)


def build_native_cli_command(
    *,
    binary_path: str,
    model_path: str,
    prompt: str,
    control_kind: str,
    seed: int,
) -> list[str]:
    """Build the repeatable native llama.cpp CLI command for one control."""

    n_predict = TRUNCATED_N_PREDICT if control_kind == "truncated_control" else POSITIVE_N_PREDICT
    return [
        binary_path,
        "--model",
        model_path,
        "--prompt",
        prompt,
        "--predict",
        str(n_predict),
        "--ctx-size",
        str(DEFAULT_CTX),
        "--batch-size",
        str(DEFAULT_BATCH),
        "--ubatch-size",
        str(DEFAULT_UBATCH),
        "--gpu-layers",
        "all",
        "--split-mode",
        "layer",
        "--temp",
        "0",
        "--top-p",
        "1",
        "--seed",
        str(seed),
        "--json-schema",
        CONTROL_JSON_SCHEMA,
        "--no-display-prompt",
        "--simple-io",
        "--single-turn",
        "--perf",
        "--offline",
    ]


def run_native_controls(
    *,
    model_specs: Sequence[Mapping[str, Any]],
    native_binary_receipt: Mapping[str, Any],
    native_runner: NativeRunner,
) -> list[JsonDict]:
    """Run or inject one positive and one truncated native control per model."""

    results: list[JsonDict] = []
    for spec in normalize_model_specs(model_specs):
        for control in controls_for_model(spec, start_index=len(results)):
            raw = native_runner(
                model_spec=spec,
                control=control,
                native_binary_receipt=native_binary_receipt,
            )
            results.append(_normalize_control_result(raw, model_spec=spec, control=control))
    return results


def controls_for_model(model_spec: Mapping[str, Any], *, start_index: int = 0) -> list[JsonDict]:
    """Return the two certificate controls for one mandated model."""

    hf_id = str(model_spec["hf_id"])
    base = f"Model {hf_id} native CUDA certificate control."
    return [
        {
            "control_kind": "positive_control",
            "prompt": (
                base
                + ' Return exactly one JSON object matching {"certificate_control":"ok"}.'
            ),
            "seed": RANDOM_SEED + start_index,
            "n_predict": POSITIVE_N_PREDICT,
            "sampling_parameters": {
                "temperature": 0.0,
                "top_p": 1.0,
                "repeat_penalty": 1.0,
                "json_schema_sha256": sha256_text(CONTROL_JSON_SCHEMA),
                "n_predict": POSITIVE_N_PREDICT,
            },
        },
        {
            "control_kind": "truncated_control",
            "prompt": (
                base
                + ' Begin the JSON object matching {"certificate_control":"ok"}, but the runner will allow one token only.'
            ),
            "seed": RANDOM_SEED + start_index + 1,
            "n_predict": TRUNCATED_N_PREDICT,
            "sampling_parameters": {
                "temperature": 0.0,
                "top_p": 1.0,
                "repeat_penalty": 1.0,
                "json_schema_sha256": sha256_text(CONTROL_JSON_SCHEMA),
                "n_predict": TRUNCATED_N_PREDICT,
            },
        },
    ]


def default_native_runner(
    *,
    model_spec: Mapping[str, Any],
    control: Mapping[str, Any],
    native_binary_receipt: Mapping[str, Any],
) -> JsonDict:  # pragma: no cover - live native process path.
    """Execute one native llama.cpp CLI control and collect CUDA process receipts."""

    binary_path = str(native_binary_receipt.get("path") or "")
    command = build_native_cli_command(
        binary_path=binary_path,
        model_path=str(model_spec["model_path"]),
        prompt=str(control["prompt"]),
        control_kind=str(control["control_kind"]),
        seed=int(control["seed"]),
    )
    timeout_s = float(os.environ.get("CARNOT_5615_NATIVE_TIMEOUT_S", DEFAULT_TIMEOUT_S))
    before = _gpu_snapshot()
    started = time.perf_counter()
    proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stop_monitor = threading.Event()
    samples: list[list[JsonDict]] = []
    pid_memory_samples: list[float] = []

    def _monitor() -> None:
        while not stop_monitor.is_set():
            samples.append(_gpu_snapshot())
            pid_memory_samples.append(_query_pid_gpu_memory(proc.pid))
            time.sleep(0.25)

    monitor = threading.Thread(target=_monitor, daemon=True)
    monitor.start()
    stdout = b""
    stderr = b""
    timed_out = False
    try:
        stdout, stderr = proc.communicate(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        timed_out = True
        proc.kill()
        stdout, stderr = proc.communicate(timeout=10)
    finally:
        stop_monitor.set()
        monitor.join(timeout=2)
    after = _gpu_snapshot()
    wall_time_s = round(time.perf_counter() - started, 6)
    stdout_text = stdout.decode("utf-8", "replace")
    stderr_text = stderr.decode("utf-8", "replace")
    log_text = stdout_text + "\n" + stderr_text
    observed, total = parse_offload_layers(log_text)
    delta_total = max(
        0,
        max((_total_gpu_used(sample) for sample in samples), default=_total_gpu_used(before))
        - _total_gpu_used(before),
    )
    pid_delta = max(pid_memory_samples, default=0.0)
    gpu_delta = int(max(delta_total, pid_delta))
    prompt_tokens = _extract_int(PROMPT_TOKENS_RE, log_text, len(str(control["prompt"]).split()))
    completion_tokens = _extract_int(EVAL_RUNS_RE, log_text, len(stdout_text.split()))
    stop_reason = classify_stop_reason(
        control_kind=str(control["control_kind"]),
        returncode=proc.returncode,
        timed_out=timed_out,
        completion_tokens=completion_tokens,
        n_predict=int(control["n_predict"]),
        raw_response=stdout_text,
    )
    return {
        "model_hf_id": model_spec["hf_id"],
        "control_kind": control["control_kind"],
        "prompt": control["prompt"],
        "raw_response": stdout_text,
        "command": command,
        "sampling_parameters": dict(control["sampling_parameters"]),
        "token_counts": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "source": "llama_cpp_perf_or_whitespace",
        },
        "stop_reason": stop_reason,
        "truncation_flag": stop_reason in {"length", "timeout"},
        "returncode": proc.returncode,
        "exit_status": "timeout" if timed_out else ("completed" if proc.returncode == 0 else "failed"),
        "wall_time_s": wall_time_s,
        "pid": proc.pid,
        "port": None,
        "requested_offload_layers": "all",
        "observed_offloaded_layers": observed,
        "observed_total_layers": total,
        "gpu_memory_before": before,
        "gpu_memory_during": samples,
        "gpu_memory_after": after,
        "gpu_memory_delta_mb": gpu_delta,
        "stdout_tail": stdout_text[-2000:],
        "stderr_tail": stderr_text[-4000:],
    }


def classify_stop_reason(
    *,
    control_kind: str,
    returncode: int | None,
    timed_out: bool,
    completion_tokens: int,
    n_predict: int,
    raw_response: str,
) -> str:
    """Classify native termination without turning it into task accuracy."""

    if timed_out:
        return "timeout"
    if returncode not in (0, None):
        return "error"
    if control_kind == "truncated_control":
        return "length"
    parsed = parse_control_response(raw_response)
    if parsed["accepted"]:
        return "stop_sequence"
    if completion_tokens >= n_predict:
        return "length"
    return "stop_sequence"


def parse_offload_layers(text: str) -> tuple[int, int | None]:
    """Extract observed offloaded layers from llama.cpp logs."""

    matches = OFFLOAD_RE.findall(text)
    if matches:
        observed, total = matches[-1]
        return int(observed), int(total)
    layers = {int(match.group(1)) for match in LAYER_CUDA_RE.finditer(text)}
    if layers:
        return len(layers), None
    return 0, None


def parse_control_response(text: str) -> JsonDict:
    """Parse the certificate JSON control and fail closed on malformed text."""

    stripped = text.strip()
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError as exc:
        return {
            "parser_ok": False,
            "parsed_object": None,
            "accepted": False,
            "parser_error_type": "json_decode_error",
            "parser_error": str(exc),
        }
    if not isinstance(parsed, Mapping):
        return {
            "parser_ok": False,
            "parsed_object": None,
            "accepted": False,
            "parser_error_type": "not_object",
            "parser_error": "",
        }
    accepted = parsed.get("certificate_control") == "ok"
    return {
        "parser_ok": accepted,
        "parsed_object": dict(parsed) if accepted else None,
        "accepted": accepted,
        "parser_error_type": "" if accepted else "schema_mismatch",
        "parser_error": "",
    }


def _normalize_control_result(
    raw: Mapping[str, Any],
    *,
    model_spec: Mapping[str, Any],
    control: Mapping[str, Any],
) -> JsonDict:
    text = str(raw.get("raw_response", ""))
    parsed = parse_control_response(text)
    control_kind = str(control["control_kind"])
    stop_reason = str(raw.get("stop_reason") or "")
    truncation_flag = bool(raw.get("truncation_flag") is True)
    if control_kind == "positive_control":
        control_passed = bool(
            parsed["accepted"] and stop_reason != "length" and raw.get("returncode") == 0
        )
        observed = "positive_json" if parsed["accepted"] else "parse_failed"
    else:
        control_passed = bool(
            not parsed["accepted"] and (truncation_flag or stop_reason in {"length", "timeout"})
        )
        observed = "truncated_or_stopped" if control_passed else "unexpected_accept"
    outcome = {
        "validator": "exp5615_certificate_control_v1",
        "accepted": bool(parsed["accepted"]),
        "expected_control": control_kind,
        "observed_control": observed,
        "parser_ok": bool(parsed["parser_ok"]),
        "parser_error_type": str(parsed["parser_error_type"]),
        "control_passed": control_passed,
    }
    return {
        "model_hf_id": str(model_spec["hf_id"]),
        "model_family": str(model_spec["family"]),
        "control_kind": control_kind,
        "prompt": str(control["prompt"]),
        "raw_response": text,
        "sampling_parameters": dict(raw.get("sampling_parameters") or control["sampling_parameters"]),
        "seed": int(control["seed"]),
        "token_counts": dict(raw.get("token_counts") or {}),
        "stop_reason": stop_reason,
        "truncation_flag": truncation_flag,
        "parser_result": parsed,
        "parsed_object": parsed["parsed_object"],
        "exact_control_outcome": outcome,
        "native_process_receipt": {
            "command": list(raw.get("command") or []),
            "returncode": raw.get("returncode"),
            "exit_status": str(raw.get("exit_status") or ""),
            "wall_time_s": float(raw.get("wall_time_s", 0.0) or 0.0),
            "pid": int(raw.get("pid", 0) or 0),
            "port": raw.get("port"),
            "requested_offload_layers": raw.get("requested_offload_layers", "all"),
            "observed_offloaded_layers": int(raw.get("observed_offloaded_layers", 0) or 0),
            "observed_total_layers": raw.get("observed_total_layers"),
            "gpu_memory_before": list(raw.get("gpu_memory_before") or []),
            "gpu_memory_during": list(raw.get("gpu_memory_during") or []),
            "gpu_memory_after": list(raw.get("gpu_memory_after") or []),
            "gpu_memory_delta_mb": int(raw.get("gpu_memory_delta_mb", 0) or 0),
            "stdout_tail": str(raw.get("stdout_tail", ""))[-2000:],
            "stderr_tail": str(raw.get("stderr_tail", ""))[-4000:],
        },
    }


def build_response_envelope_rows(
    *,
    control_results: Sequence[Mapping[str, Any]],
    model_specs: Sequence[Mapping[str, Any]],
) -> list[JsonDict]:
    """Convert native controls into Exp5605-compatible lossless envelope rows."""

    specs = {str(row["hf_id"]): dict(row) for row in normalize_model_specs(model_specs)}
    rows: list[JsonDict] = []
    previous_hash = ""
    for sequence_index, result in enumerate(control_results):
        hf_id = str(result["model_hf_id"])
        spec = specs[hf_id]
        prompt = str(result["prompt"])
        raw = str(result["raw_response"]).encode("utf-8")
        process_receipt = dict(result["native_process_receipt"])
        row: JsonDict = {
            "envelope_schema_version": ENVELOPE_SCHEMA_VERSION,
            "sequence_index": sequence_index,
            "call_id": f"{model_family(hf_id)}-{result['control_kind']}-{sequence_index}",
            "control_kind": str(result["control_kind"]),
            "model_family": model_family(hf_id),
            "model_hf_id": hf_id,
            "model_local_path_hash": str(spec["local_path_hash"]),
            "model_file_sha256": str(spec["model_sha256"]),
            "prompt_payload": encode_prompt(prompt),
            "prompt_hash": sha256_bytes(prompt.encode("utf-8")),
            "raw_response_payload": encode_lossless_payload(raw),
            "payload_hash": sha256_bytes(raw),
            "llama_cpp_version": "",
            "llama_cpp_arguments": _arguments_from_command(process_receipt.get("command", [])),
            "device_offload_receipt": process_receipt,
            "native_process_receipt": process_receipt,
            "sampling_parameters": dict(result.get("sampling_parameters") or {}),
            "seed": int(result.get("seed", RANDOM_SEED) or RANDOM_SEED),
            "stop_reason": str(result.get("stop_reason") or ""),
            "token_counts": dict(result.get("token_counts") or {}),
            "truncation_flag": bool(result.get("truncation_flag") is True),
            "parser_name": PARSER_NAME,
            "parser_version": PARSER_VERSION,
            "parsed_object": result.get("parsed_object"),
            "exact_validator_outcome": dict(result["exact_control_outcome"]),
            "exact_control_outcome": dict(result["exact_control_outcome"]),
            "timestamp_utc": f"2026-07-14T00:00:{sequence_index:02d}Z",
            "previous_row_hash": previous_hash,
            "row_hash": "",
        }
        row["row_hash"] = row_hash(row)
        previous_hash = row["row_hash"]
        rows.append(row)
    return rows


def replay_response_envelope_rows(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Verify prompt/response bytes, hashes, parser replay, and row chaining."""

    previous_hash = ""
    replayed = 0
    stop_controls = 0
    stop_controls_passed = 0
    semantic_false_accepts = 0
    for row in rows:
        if row.get("previous_row_hash") != previous_hash:
            raise EnvelopeReplayError("previous_row_hash")
        if row.get("row_hash") != row_hash(row):
            raise EnvelopeReplayError("row_hash")
        prompt = decode_prompt(_mapping(row.get("prompt_payload"), "prompt_payload"))
        raw = decode_lossless_payload(_mapping(row.get("raw_response_payload"), "raw_response_payload"))
        if row.get("prompt_hash") != sha256_bytes(prompt):
            raise EnvelopeReplayError("prompt_hash")
        if row.get("payload_hash") != sha256_bytes(raw):
            raise EnvelopeReplayError("payload_hash")
        parsed = parse_control_response(raw.decode("utf-8", "replace"))
        if row.get("parser_name") != PARSER_NAME or row.get("parser_version") != PARSER_VERSION:
            raise EnvelopeReplayError("parser_version")
        if row.get("parsed_object") != parsed["parsed_object"]:
            raise EnvelopeReplayError("parsed_object")
        outcome = dict(row.get("exact_control_outcome") or row.get("exact_validator_outcome") or {})
        expected_outcome = _expected_outcome_from_row(row, parsed)
        if outcome != expected_outcome:
            raise EnvelopeReplayError("exact_control_outcome")
        if row.get("control_kind") != "positive_control" and outcome.get("accepted") is True:
            semantic_false_accepts += 1
        if row.get("control_kind") in {"truncated_control", "stop_control"}:
            stop_controls += 1
            if outcome.get("control_passed") is True:
                stop_controls_passed += 1
        replayed += 1
        previous_hash = str(row["row_hash"])
    return {
        "row_count": len(rows),
        "lossless_replay_rate": 0.0 if not rows else round(replayed / len(rows), 6),
        "stop_control_pass_rate": 0.0
        if stop_controls == 0
        else round(stop_controls_passed / stop_controls, 6),
        "semantic_false_accept_count": semantic_false_accepts,
    }


def write_response_envelope_rows(rows: Sequence[Mapping[str, Any]], path: Path | str) -> None:
    """Write one lossless response-envelope row per JSONL line."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(json.dumps(dict(row), sort_keys=True, ensure_ascii=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def replay_response_envelope_path(path: Path | str) -> JsonDict:
    """Replay a certificate response-envelope JSONL file from disk."""

    rows = [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return replay_response_envelope_rows(rows)


def build_artifact(
    *,
    model_specs: Sequence[Mapping[str, Any]],
    preconditions: Mapping[str, Any],
    control_results: Sequence[Mapping[str, Any]],
    evidence_rows: Sequence[Mapping[str, Any]],
    response_envelope_path: str,
    orphan_process_count: int = 0,
    tests_run: Sequence[Mapping[str, Any]] = (),
) -> JsonDict:
    """Build the terminal Exp5615 runtime certificate artifact."""

    specs = normalize_model_specs(model_specs)
    precondition_blockers = list(preconditions.get("blocked_preconditions", []))
    replay = _safe_replay(evidence_rows)
    offload_layers = offload_layers_by_model(control_results, specs)
    gpu_deltas = gpu_memory_delta_by_model(control_results, specs)
    certified = certified_models(
        model_specs=specs,
        control_results=control_results,
        replay=replay,
        precondition_blockers=precondition_blockers,
        orphan_process_count=orphan_process_count,
    )
    ready = bool(
        len(certified) == len(MANDATED_HEADLINE_IDS)
        and replay["lossless_replay_rate"] == 1.0
        and replay["stop_control_pass_rate"] == 1.0
        and replay["semantic_false_accept_count"] == 0
        and orphan_process_count == 0
        and not precondition_blockers
    )
    score = 1.0 if ready else round(min(len(certified), len(MANDATED_HEADLINE_IDS) - 1) / 3, 6)
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "spec_refs": list(SPEC_REFS),
        "result_path": RESULT_RELATIVE_PATH.as_posix(),
        "field_principles": dict(FIELD_PRINCIPLES),
        "model_specs": specs,
        "MODEL_SPECS": specs,
        "native_binary_receipt": dict(preconditions.get("native_binary_receipt") or {}),
        "cuda_build_capability": dict(preconditions.get("cuda_build_capability") or {}),
        "gpu_device_receipts": dict(preconditions.get("gpu_device_receipts") or {}),
        "offload_layers_by_model": offload_layers,
        "gpu_memory_delta_by_model": gpu_deltas,
        "response_envelope_path": response_envelope_path,
        "lossless_replay_rate": replay["lossless_replay_rate"],
        "stop_control_pass_rate": replay["stop_control_pass_rate"],
        "semantic_false_accept_count": replay["semantic_false_accept_count"],
        "orphan_process_count": int(orphan_process_count),
        "models_certified_count": len(certified),
        "models_certified_denominator": len(MANDATED_HEADLINE_IDS),
        "runtime_certificate_ready_score": score,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
        "honest_verdict": honest_verdict(ready, precondition_blockers),
        "blocked_preconditions": precondition_blockers,
        "control_results": [dict(row) for row in control_results],
        "response_envelope_rows": [dict(row) for row in evidence_rows],
        "repeatable_native_arguments": repeatable_native_arguments(control_results),
        "legacy_smoke_models": [
            {"hf_id": hf_id, "label": "cpu_smoke_only", "certificate_eligible": False}
            for hf_id in LEGACY_SMOKE_IDS
        ],
        "no_task_accuracy_computed": True,
        "solve_verify_accuracy_inferred": False,
        "tests_run": [dict(row) for row in tests_run],
    }
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    validate_artifact(artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate required fields and fail closed on unsupported certificate claims."""

    for field in REQUIRED_ARTIFACT_FIELDS:
        _require(field in artifact, field)
    _require(set(REQUIRED_ARTIFACT_FIELDS).issubset(artifact["field_principles"]), "field_principles")
    _require(artifact.get("inference_substrate") == INFERENCE_SUBSTRATE, "inference_substrate")
    _require(artifact.get("random_seed") == RANDOM_SEED, "random_seed")
    _require(_model_specs_ready(artifact.get("model_specs", [])), "model_specs")
    _require(artifact.get("no_task_accuracy_computed") is True, "no_task_accuracy_computed")
    _require(artifact.get("solve_verify_accuracy_inferred") is False, "solve_verify_accuracy_inferred")
    _require(artifact.get("reproducibility_checksum") == artifact_checksum(artifact), "checksum")
    score = float(artifact.get("runtime_certificate_ready_score", 0.0) or 0.0)
    if score == 1.0:
        _require(artifact.get("models_certified_count") == 3, "models_certified_count")
        _require(artifact.get("lossless_replay_rate") == 1.0, "lossless_replay_rate")
        _require(artifact.get("stop_control_pass_rate") == 1.0, "stop_control_pass_rate")
        _require(artifact.get("semantic_false_accept_count") == 0, "semantic_false_accept_count")
        _require(artifact.get("orphan_process_count") == 0, "orphan_process_count")
        _require(str(artifact.get("honest_verdict", "")).startswith("complete:"), "honest_verdict")
        _require(artifact.get("blocked_preconditions") == [], "blocked_preconditions")
        _require(_commands_use_single_turn(artifact.get("response_envelope_rows", [])), "single_turn")
    else:
        _require(str(artifact.get("honest_verdict", "")).startswith("blocked_"), "honest_verdict")


def run(
    *,
    result_path: Path | str = REPO_ROOT / RESULT_RELATIVE_PATH,
    response_envelope_path: Path | str = REPO_ROOT / RESPONSE_ENVELOPE_RELATIVE_PATH,
    model_specs: Sequence[Mapping[str, Any]] | None = None,
    preconditions: Mapping[str, Any] | None = None,
    native_runner: NativeRunner = default_native_runner,
    tests_run: Sequence[Mapping[str, Any]] = (),
) -> JsonDict:
    """Run Exp5615 or write a terminal blocked certificate."""

    specs = normalize_model_specs(model_specs) if model_specs is not None else resolve_all_headline_model_specs()
    collected = dict(preconditions or collect_preconditions(REPO_ROOT))
    blockers = precondition_blockers(collected, specs)
    collected["blocked_preconditions"] = blockers
    if blockers:
        controls: list[JsonDict] = []
        rows: list[JsonDict] = []
        orphan_count = 0
    else:
        controls = run_native_controls(
            model_specs=specs,
            native_binary_receipt=collected["native_binary_receipt"],
            native_runner=native_runner,
        )
        rows = build_response_envelope_rows(control_results=controls, model_specs=specs)
        orphan_count = orphan_process_count(controls)
    write_response_envelope_rows(rows, response_envelope_path)
    artifact = build_artifact(
        model_specs=specs,
        preconditions=collected,
        control_results=controls,
        evidence_rows=rows,
        response_envelope_path=str(Path(response_envelope_path)),
        orphan_process_count=orphan_count,
        tests_run=tests_run,
    )
    output = Path(result_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(artifact, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    return artifact


def collect_preconditions(root: Path = REPO_ROOT) -> JsonDict:  # pragma: no cover - host dependent.
    """Collect native binary, CUDA build, GPU, and envelope API receipts."""

    binary_path = _first_executable("llama-cli")
    native = native_binary_receipt(binary_path)
    gpu_before = _gpu_snapshot()
    raw_nvidia = _run_command(["nvidia-smi"], timeout_s=10)
    query_nvidia = _run_command(
        [
            "nvidia-smi",
            "--query-gpu=index,name,driver_version,memory.total,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        timeout_s=10,
    )
    cuda_ready = cuda_build_capability(native)
    return {
        "native_binary_receipt": native,
        "cuda_build_capability": cuda_ready,
        "gpu_device_receipts": {
            "before": gpu_before,
            "after": [],
            "nvidia_smi": query_nvidia,
            "raw_nvidia_smi": raw_nvidia,
        },
        "response_envelope_api": {
            "schema": exp5605.ENVELOPE_SCHEMA_VERSION,
            "encode_lossless_payload": callable(exp5605.encode_lossless_payload),
            "decode_lossless_payload": callable(exp5605.decode_lossless_payload),
        },
        "root": str(root),
        "blocked_preconditions": [],
    }


def native_binary_receipt(binary_path: Path | None) -> JsonDict:  # pragma: no cover - host dependent.
    """Authenticate the selected native llama.cpp binary without model loading."""

    if binary_path is None:
        return {
            "kind": "llama-cli",
            "path": None,
            "executable": False,
            "sha256": "",
            "version": {"ok": False, "stdout": "", "stderr": "llama-cli not found"},
            "help": {
                "ok": False,
                "stdout": "",
                "stderr": "llama-cli not found",
                "contains_single_turn": False,
                "contains_gpu_layers": False,
                "contains_json_schema": False,
            },
            "dynamic_libraries": {"ok": False, "stdout": "", "stderr": "llama-cli not found"},
            "list_devices": {"ok": False, "stdout": "", "stderr": "llama-cli not found"},
            "candidate_paths": [str(path) for path in _candidate_binary_paths("llama-cli")],
        }
    version = _run_command([str(binary_path), "--version"], timeout_s=20)
    help_result = _run_command([str(binary_path), "--help"], timeout_s=20)
    ldd = _run_command(["ldd", str(binary_path)], timeout_s=20)
    list_devices = _run_command([str(binary_path), "--list-devices"], timeout_s=20)
    help_text = str(help_result.get("stdout", "")) + str(help_result.get("stderr", ""))
    return {
        "kind": "llama-cli",
        "path": str(binary_path),
        "executable": binary_path.is_file() and os.access(binary_path, os.X_OK),
        "sha256": sha256_file(binary_path),
        "version": version,
        "help": {
            **help_result,
            "stdout": str(help_result.get("stdout", ""))[-12000:],
            "contains_single_turn": "--single-turn" in help_text,
            "contains_gpu_layers": "--gpu-layers" in help_text or "--n-gpu-layers" in help_text,
            "contains_json_schema": "--json-schema" in help_text,
        },
        "dynamic_libraries": {**ldd, "stdout": str(ldd.get("stdout", ""))[-8000:]},
        "list_devices": list_devices,
        "candidate_paths": [str(path) for path in _candidate_binary_paths("llama-cli")],
    }


def cuda_build_capability(native_binary: Mapping[str, Any]) -> JsonDict:
    """Summarize whether the native binary is compiled for CUDA offload."""

    libs = str(native_binary.get("dynamic_libraries", {}).get("stdout", ""))
    devices = str(native_binary.get("list_devices", {}).get("stdout", ""))
    help_row = native_binary.get("help", {})
    cuda_linked = bool(CUDA_LIBRARY_RE.search(libs))
    devices_cuda = "CUDA" in devices
    help_gpu = bool(isinstance(help_row, Mapping) and help_row.get("contains_gpu_layers"))
    missing = []
    if not native_binary.get("executable"):
        missing.append("native_llama_cpp_binary_unavailable")
    if not cuda_linked:
        missing.append("native_llama_cpp_cuda_libraries_absent")
    if not devices_cuda:
        missing.append("native_llama_cpp_cuda_devices_absent")
    if not help_gpu:
        missing.append("native_llama_cpp_gpu_layers_flag_absent")
    return {
        "cuda_backend_linked": cuda_linked,
        "list_devices_reports_cuda": devices_cuda,
        "help_reports_gpu_layers": help_gpu,
        "native_cuda_ready": not missing,
        "missing_preconditions": missing,
    }


def precondition_blockers(
    preconditions: Mapping[str, Any],
    model_specs: Sequence[Mapping[str, Any]],
) -> list[str]:
    """Return exact blockers that must stop model loading."""

    blockers = list(preconditions.get("blocked_preconditions", []))
    native = preconditions.get("native_binary_receipt", {})
    cuda = preconditions.get("cuda_build_capability", {})
    gpu = preconditions.get("gpu_device_receipts", {})
    if not _model_specs_ready(model_specs):
        blockers.append("mandated_gguf_cache_missing_or_unhashed")
    native_executable = isinstance(native, Mapping) and native.get("executable") is True
    if not native_executable:
        blockers.append("native_llama_cpp_binary_unavailable")
    help_row = native.get("help", {}) if isinstance(native, Mapping) else {}
    if native_executable and (
        not isinstance(help_row, Mapping) or help_row.get("contains_single_turn") is not True
    ):
        blockers.append("native_llama_cpp_single_turn_unavailable")
    if not isinstance(cuda, Mapping) or cuda.get("native_cuda_ready") is not True:
        blockers.extend(list(cuda.get("missing_preconditions", [])) if isinstance(cuda, Mapping) else [])
    if not isinstance(gpu, Mapping) or not gpu.get("before"):
        blockers.append("gpu_device_receipt_unavailable")
    return list(dict.fromkeys(blockers))


def offload_layers_by_model(
    control_results: Sequence[Mapping[str, Any]],
    model_specs: Sequence[Mapping[str, Any]],
) -> dict[str, JsonDict]:
    """Return requested and observed offloaded layers for each mandated model."""

    out: dict[str, JsonDict] = {}
    by_model: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in control_results:
        by_model[str(row.get("model_hf_id"))].append(row)
    for spec in model_specs:
        hf_id = str(spec["hf_id"])
        receipts = [dict(row.get("native_process_receipt") or {}) for row in by_model[hf_id]]
        observed = max((int(row.get("observed_offloaded_layers", 0) or 0) for row in receipts), default=0)
        totals = [row.get("observed_total_layers") for row in receipts if row.get("observed_total_layers")]
        out[hf_id] = {
            "requested": "all",
            "observed": observed,
            "observed_total": totals[-1] if totals else None,
            "source": "native_llama_cpp_process_log",
        }
    return out


def gpu_memory_delta_by_model(
    control_results: Sequence[Mapping[str, Any]],
    model_specs: Sequence[Mapping[str, Any]],
) -> dict[str, int]:
    """Return the max nonzero GPU process memory delta per mandated model."""

    out: dict[str, int] = {}
    by_model: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in control_results:
        by_model[str(row.get("model_hf_id"))].append(row)
    for spec in model_specs:
        hf_id = str(spec["hf_id"])
        out[hf_id] = max(
            (
                int(dict(row.get("native_process_receipt") or {}).get("gpu_memory_delta_mb", 0) or 0)
                for row in by_model[hf_id]
            ),
            default=0,
        )
    return out


def certified_models(
    *,
    model_specs: Sequence[Mapping[str, Any]],
    control_results: Sequence[Mapping[str, Any]],
    replay: Mapping[str, Any],
    precondition_blockers: Sequence[str],
    orphan_process_count: int,
) -> list[str]:
    """Return mandated model IDs that passed all native CUDA certificate gates."""

    if precondition_blockers:
        return []
    by_model: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in control_results:
        by_model[str(row.get("model_hf_id"))].append(row)
    certified: list[str] = []
    for spec in model_specs:
        hf_id = str(spec["hf_id"])
        rows = by_model[hf_id]
        kinds = {str(row.get("control_kind")) for row in rows}
        if kinds != {"positive_control", "truncated_control"}:
            continue
        if not all(dict(row.get("exact_control_outcome") or {}).get("control_passed") is True for row in rows):
            continue
        if not all(dict(row.get("native_process_receipt") or {}).get("returncode") == 0 for row in rows):
            continue
        if not all(
            int(dict(row.get("native_process_receipt") or {}).get("observed_offloaded_layers", 0) or 0)
            > 0
            for row in rows
        ):
            continue
        if not all(
            int(dict(row.get("native_process_receipt") or {}).get("gpu_memory_delta_mb", 0) or 0)
            > 0
            for row in rows
        ):
            continue
        if not all("--single-turn" in dict(row.get("native_process_receipt") or {}).get("command", []) for row in rows):
            continue
        certified.append(hf_id)
    if replay.get("lossless_replay_rate") != 1.0 or replay.get("semantic_false_accept_count") != 0:
        return []
    return certified


def repeatable_native_arguments(control_results: Sequence[Mapping[str, Any]]) -> dict[str, list[list[str]]]:
    """Record exact native commands so the certificate method is repeatable."""

    out: dict[str, list[list[str]]] = defaultdict(list)
    for row in control_results:
        receipt = dict(row.get("native_process_receipt") or {})
        out[str(row.get("model_hf_id"))].append(list(receipt.get("command") or []))
    return dict(out)


def orphan_process_count(control_results: Sequence[Mapping[str, Any]]) -> int:
    """Count native process PIDs that are still alive after the runner returns."""

    count = 0
    for row in control_results:
        pid = int(dict(row.get("native_process_receipt") or {}).get("pid", 0) or 0)
        if pid > 0 and _pid_alive(pid):
            count += 1
    return count


def artifact_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash the artifact while blanking the self-referential checksum field."""

    stable = dict(artifact)
    stable["reproducibility_checksum"] = ""
    return sha256_json(stable)


def honest_verdict(ready: bool, blockers: Sequence[str]) -> str:
    """Return the terminal certificate verdict."""

    if ready:
        return "complete: native_llamacpp_cuda_runtime_certificate_ready_all_three_models"
    if blockers:
        return "blocked_native_preconditions_missing:" + ",".join(blockers)
    return "blocked_native_cuda_runtime_certificate_failed_terminal_retirement_evidence"


def row_hash(row: Mapping[str, Any]) -> str:
    """Hash one envelope row while excluding the row hash field itself."""

    stable = dict(row)
    stable["row_hash"] = ""
    return sha256_json(stable)


def _expected_outcome_from_row(row: Mapping[str, Any], parsed: Mapping[str, Any]) -> JsonDict:
    control_kind = str(row.get("control_kind") or "")
    stop_reason = str(row.get("stop_reason") or "")
    truncation_flag = row.get("truncation_flag") is True
    if control_kind == "positive_control":
        control_passed = bool(parsed["accepted"] and stop_reason != "length")
        observed = "positive_json" if parsed["accepted"] else "parse_failed"
    else:
        control_passed = bool(
            not parsed["accepted"] and (truncation_flag or stop_reason in {"length", "timeout"})
        )
        observed = "truncated_or_stopped" if control_passed else "unexpected_accept"
    return {
        "validator": "exp5615_certificate_control_v1",
        "accepted": bool(parsed["accepted"]),
        "expected_control": control_kind,
        "observed_control": observed,
        "parser_ok": bool(parsed["parser_ok"]),
        "parser_error_type": str(parsed["parser_error_type"]),
        "control_passed": control_passed,
    }


def _arguments_from_command(command: Any) -> JsonDict:
    if not isinstance(command, Sequence) or isinstance(command, (str, bytes)):
        return {}
    args = list(command)
    def _after(flag: str, default: Any = None) -> Any:
        try:
            return args[args.index(flag) + 1]
        except (ValueError, IndexError):
            return default

    return {
        "command": args,
        "n_gpu_layers": _after("--gpu-layers"),
        "n_ctx": int(_after("--ctx-size", 0) or 0),
        "n_batch": int(_after("--batch-size", 0) or 0),
        "n_ubatch": int(_after("--ubatch-size", 0) or 0),
        "n_predict": int(_after("--predict", 0) or 0),
        "single_turn": "--single-turn" in args,
        "json_schema_sha256": sha256_text(_after("--json-schema", "")),
    }


def _safe_replay(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    try:
        return replay_response_envelope_rows(rows)
    except EnvelopeReplayError as exc:
        return {
            "row_count": len(rows),
            "lossless_replay_rate": 0.0,
            "stop_control_pass_rate": 0.0,
            "semantic_false_accept_count": 1,
            "replay_error": str(exc),
        }


def _model_specs_ready(model_specs: Any) -> bool:
    if not isinstance(model_specs, Sequence) or isinstance(model_specs, (str, bytes)):
        return False
    rows = [row for row in model_specs if isinstance(row, Mapping)]
    if [str(row.get("hf_id")) for row in rows] != list(MANDATED_HEADLINE_IDS):
        return False
    return all(
        row.get("local_model_present") is True
        and str(row.get("model_path", "")).endswith(".gguf")
        and bool(row.get("model_sha256"))
        and row.get("headline_eligible") is True
        for row in rows
    )


def _commands_use_single_turn(rows: Any) -> bool:
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        return False
    return all(
        "--single-turn" in dict(row.get("native_process_receipt") or {}).get("command", [])
        for row in rows
        if isinstance(row, Mapping)
    )


def _run_command(command: Sequence[str], *, timeout_s: float) -> JsonDict:  # pragma: no cover
    started = time.perf_counter()
    try:
        result = subprocess.run(
            list(command),
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
        return {
            "command": list(command),
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "duration_s": round(time.perf_counter() - started, 6),
            "ok": result.returncode == 0,
        }
    except Exception as exc:
        return {
            "command": list(command),
            "returncode": None,
            "stdout": "",
            "stderr": f"{type(exc).__name__}: {exc}",
            "duration_s": round(time.perf_counter() - started, 6),
            "ok": False,
        }


def _candidate_binary_paths(binary_name: str) -> list[Path]:  # pragma: no cover
    env_specific = os.environ.get(f"CARNOT_{binary_name.upper().replace('-', '_')}")
    candidates: list[Path] = []
    if env_specific:
        candidates.append(Path(env_specific))
    bin_dir = os.environ.get("CARNOT_LLAMA_CPP_BIN_DIR")
    if bin_dir:
        candidates.append(Path(bin_dir) / binary_name)
    which = shutil.which(binary_name)
    if which:
        candidates.append(Path(which))
    candidates.extend(
        [
            Path.home() / ".cache" / "llama.cpp-master" / "build" / "bin" / binary_name,
            Path.home() / "github.com" / "ggml-org" / "llama.cpp" / "build" / "bin" / binary_name,
        ]
    )
    unique: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        text = str(candidate.expanduser())
        if text not in seen:
            seen.add(text)
            unique.append(Path(text))
    return unique


def _first_executable(binary_name: str) -> Path | None:  # pragma: no cover
    for candidate in _candidate_binary_paths(binary_name):
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate
    return None


def _gpu_snapshot() -> list[JsonDict]:  # pragma: no cover - host dependent.
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,driver_version,memory.total,memory.used,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    rows: list[JsonDict] = []
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 7:
            continue
        try:
            rows.append(
                {
                    "index": int(parts[0]),
                    "name": parts[1],
                    "driver_version": parts[2],
                    "memory_total_mb": int(float(parts[3])),
                    "memory_used_mb": int(float(parts[4])),
                    "memory_free_mb": int(float(parts[5])),
                    "utilization_gpu_pct": int(float(parts[6])),
                }
            )
        except ValueError:
            continue
    return rows


def _query_pid_gpu_memory(pid: int) -> float:  # pragma: no cover - host dependent.
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
    except Exception:
        return 0.0
    peak = 0.0
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) >= 2 and parts[0] == str(pid):
            try:
                peak = max(peak, float(parts[1]))
            except ValueError:
                pass
    return peak


def _total_gpu_used(snapshot: Sequence[Mapping[str, Any]]) -> int:
    return sum(int(row.get("memory_used_mb", 0) or 0) for row in snapshot)


def _extract_int(pattern: re.Pattern[str], text: str, default: int) -> int:
    match = pattern.search(text)
    if not match:
        return int(default)
    return int(match.group(1))


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise EnvelopeReplayError(field)
    return value


def _require(condition: bool, field: str) -> None:
    if not condition:
        raise ValueError(field)


if __name__ == "__main__":  # pragma: no cover
    run()
