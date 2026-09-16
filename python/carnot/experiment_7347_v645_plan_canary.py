"""Qualify current local-model transport for the public plan format.

The model sees four public development requests and the two-field plan schema.
It never sees private rules, executor decisions, or evaluator feedback.

Spec refs: REQ-CL-7347 and SCENARIO-CL-7347-*.
"""

from __future__ import annotations

import argparse
import base64
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from datetime import UTC, datetime
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
from urllib import error, request as urlrequest

from carnot import experiment_7160_v631_qwen38_lease_diagnosis as lease_preflight
from carnot import gpu_lease_phase_journal as lease_api
from carnot.experiment_6212_three_family_gguf_runtime_recovery import (
    read_gguf_metadata,
    resolve_native_llama_server,
    snapshot_revision,
)
from carnot.experiment_7150_v628_grounding_preflight import _gpu_snapshot
from carnot.experiment_7181_v633_qwen38_symbolic_traces import (
    _content_addressed_hash,
    _free_port,
    _gpu_provenance,
    _heartbeat,
    _release_lease,
    _server_command,
    _stream_server_log,
    _wait_for_health,
)
from carnot.inference.llama_server_supervisor import (
    NativeLlamaServerSupervisor,
    canonical_json,
    supervisor_contract,
)
from carnot.inference.sota_models import cached_current_model, gguf_tokenizer_loadable
from carnot.reporting.experiment_7303_validation_scope import (
    CommandSpec,
    REQUIRED_CHECK_NAMES,
    run_commands,
    run_scoped_validation,
)


JsonDict = dict[str, Any]
REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_DATE = "20260916"
MILESTONE = "2026.09.645"
EXPERIMENT_ID = "exp7347-plan-canary"
SCHEMA = "carnot.exp7347.v645_plan_canary.v1"
TASK_ID = "experiment_7347_v645_plan_canary"
QWEN_MODEL_ID = "unsloth/Qwen3.8-27B-GGUF"
QUANTIZATION = "Q4_K_M"
MODEL_SPECS: list[JsonDict] = [{"hf_id": QWEN_MODEL_ID, "quantization": QUANTIZATION}]
MODULE_PATH = Path("python/carnot/experiment_7347_v645_plan_canary.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_7347_v645_plan_canary.py")
TEST_PATH = Path("tests/python/test_experiment_7347_v645_plan_canary.py")
SPEC_PATH = Path("openspec/capabilities/continuous-learning/spec.md")
PRODUCER_PATH = Path("results/experiment_7344_v645_executor_fixture.json")
PUBLIC_MANIFEST_PATH = REPO_ROOT / (
    "results/raw/experiment_7344_v645_executor_fixture/public/public_manifest.json"
)
RESULT_PATH = Path("results/experiment_7347_v645_plan_canary.json")
RAW_DIR = Path("results/raw/experiment_7347_v645_plan_canary")
RAW_MANIFEST_PATH = RAW_DIR / "raw_call_manifest.json"
RANDOM_SEED = {"development": 7_347_101, "evaluation": 7_347_201, "resampling": 7_347_301}
MAX_GENERATED_TOKENS = 128
MODEL_LOAD_TIMEOUT_S = 600.0
INFERENCE_WINDOW_TIMEOUT_S = 900.0
REQUEST_TIMEOUT_S = 180.0
TERMINAL_CHECK_NAMES = (
    "independent_reducer",
    "adversarial_verify",
    "verdict_row_consistency_strict",
)

FIELD_PRINCIPLES: JsonDict = {
    "schema": "Version the record and retain ordinary top-level experiment_id and milestone.",
    "status": "Write a terminal result only after actual work and affected checks.",
    "run_date": "Use 20260916; record real UTC timestamps as well.",
    "preconditions_checked": "Record each actual input/resource check before dependent work.",
    "MODEL_SPECS": "List actual intended model identities; LLM tasks include unsloth/Qwen3.8-27B-GGUF.",
    "model_invoked": "True for any attempted current model load or generation, including failures.",
    "invocation_counts": "Separate attempted, completed, failed, cancelled and in-flight operations.",
    "inference_substrate": "Declare actual computation; historical model receipts are not current inference.",
    "inference_substrate_class": "Use the closed duration class matching the actual run.",
    "execution_venue": "Use host; this milestone makes no new board-execution claim.",
    "duration_s": "Measure monotonic time; never wait merely to pass a duration floor.",
    "phase_spans": "Measure disjoint load, generation, evaluation, test and write spans.",
    "random_seed": "Freeze development, evaluation and resampling seeds before outcomes.",
    "reproducibility_checksum": "Bind code, settings, inputs, evaluator identity and raw evidence.",
    "source_artifact_hashes": "Authenticate exact producers and current same-milestone paths.",
    "rows": "Keep every comparative unit, arm, metric, cost, failure and censoring disposition.",
    "sample_size_budget": "Record planned, attempted, completed and censored units and stopping rules.",
    "acceptance_gate_results": "Each gate records expected, observed, passed and its principle.",
    "gate_check_summary": "Every blocked_* names upstream, failed check, exact artifact field, expected and observed value.",
    "verifier_is_oracle": "True when the executor defines correctness; separate code does not remove circularity.",
    "honest_verdict": "Completed work starts complete_ or complete:; external absence starts blocked_ with its failed check.",
    "verdict_class": "Closed enum: positive | circular_positive | null | blocked | disqualified | partial. Only unfinished own work is partial.",
    "flagged_adversarial": "Set false only after current verification; a critical finding sets true and prevents promotion.",
    "validation_receipts": "Retain exact command, scope, exit code, elapsed time and log hash, including failures.",
    "repository_health": "Preserve dated unrelated failures separately from affected required validation.",
    "field_principles": "Explain fields separately; do not wrap numeric gates or ordinary dictionaries.",
    "plan_transport_ready_score": "Qualify transport and identity independently from source correctness.",
    "usable_plan_count": "Keep semantic nulls visible when transport is healthy.",
    "runtime_identity_receipt": "Bind the owned process and exact GGUF bytes to each request.",
    "raw_call_manifest": "Keep four bounded calls, including errors and unparsed outputs.",
}
REQUIRED_ARTIFACT_FIELDS = frozenset(FIELD_PRINCIPLES)


def sha256_text(value: str) -> str:
    """Hash exact text so prompt and reply changes remain detectable."""

    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:  # pragma: no cover - used by live orchestration.
    """Hash exact source bytes without normalizing line endings."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def artifact_checksum(artifact: Mapping[str, Any]) -> str:  # pragma: no cover
    """Bind the terminal evidence without hashing the checksum into itself."""

    value = deepcopy(dict(artifact))
    value.pop("reproducibility_checksum", None)
    return sha256_text(canonical_json(value))


def _utc_now() -> str:  # pragma: no cover
    """Record real wall-clock boundaries while durations use monotonic time."""

    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def progress(phase: str, event: str, **details: Any) -> None:  # pragma: no cover
    """Flush every boundary so a long native call remains observable."""

    suffix = " ".join(f"{key}={value}" for key, value in details.items())
    print(f"[exp7347] phase={phase} event={event} {suffix}".rstrip(), flush=True)


def gate_row(
    check: str,
    upstream: str,
    artifact_field: str,
    expected: Any,
    observed: Any,
    passed: bool,
    principle: str,
) -> JsonDict:  # pragma: no cover
    """Keep both sides of a gate so a blocked result names the real cause."""

    return {
        "check": check,
        "upstream": upstream,
        "artifact_field": artifact_field,
        "expected_value": deepcopy(expected),
        "observed_value": deepcopy(observed),
        "passed": bool(passed),
        "principle": principle,
    }


def gate_check_summary(checks: Sequence[Mapping[str, Any]]) -> JsonDict:  # pragma: no cover
    """Project the first failure without replacing its observed value."""

    failed = next((row for row in checks if row.get("passed") is not True), None)
    if failed is None:
        return {
            "check_count": len(checks),
            "failed_check_count": 0,
            "failed_check": None,
            "upstream": EXPERIMENT_ID,
            "artifact_field": "plan_transport_ready_score",
            "expected_value": 1,
            "observed_value": 1,
            "passed": True,
        }
    return {
        "check_count": len(checks),
        "failed_check_count": sum(row.get("passed") is not True for row in checks),
        "failed_check": failed.get("check"),
        "upstream": failed.get("upstream"),
        "artifact_field": failed.get("artifact_field"),
        "expected_value": deepcopy(failed.get("expected_value")),
        "observed_value": deepcopy(failed.get("observed_value")),
        "passed": False,
    }


def select_development_requests(public_manifest: Mapping[str, Any]) -> list[JsonDict]:
    """Freeze four requests from one development stream before model outcomes."""

    streams = public_manifest.get("development_streams")
    if not isinstance(streams, list) or not streams or not isinstance(streams[0], Mapping):
        raise ValueError("development_streams")
    requests = streams[0].get("requests")
    if not isinstance(requests, list) or len(requests) < 4:
        raise ValueError("development_request_count")
    selected = [deepcopy(dict(row)) for row in requests[:4] if isinstance(row, Mapping)]
    if len(selected) != 4 or len({row.get("request_id") for row in selected}) != 4:
        raise ValueError("development_request_identity")
    return selected


def render_public_prompt(public_request: Mapping[str, Any]) -> str:
    """Show the source request and exact output fields with no private hints."""

    schema = {
        "request_id": str(public_request["request_id"]),
        "assignments": {
            str(activity): "one integer from this activity's allowed_starts"
            for activity in public_request["activities"]
        },
    }
    return (
        "Return exactly one JSON object and no other text. "
        "Use exactly the public plan fields shown below.\n"
        f"PUBLIC_PLAN_SCHEMA={canonical_json(schema)}\n"
        f"PUBLIC_REQUEST={canonical_json(dict(public_request))}"
    )


def decode_public_plan(raw_reply: str, public_request: Mapping[str, Any]) -> JsonDict:
    """Parse one exact public plan object without repair or schema guessing."""

    try:
        value = json.loads(raw_reply)
    except (json.JSONDecodeError, TypeError):
        return {"parse_status": "invalid", "parse_errors": ["json_object"], "plan": None}
    if not isinstance(value, dict):
        return {"parse_status": "invalid", "parse_errors": ["json_object"], "plan": None}
    if set(value) != {"request_id", "assignments"}:
        return {
            "parse_status": "invalid",
            "parse_errors": ["top_level_fields"],
            "plan": None,
        }
    if value["request_id"] != public_request.get("request_id"):
        return {"parse_status": "invalid", "parse_errors": ["request_id"], "plan": None}
    assignments = value["assignments"]
    activities = list(public_request.get("activities", []))
    if not isinstance(assignments, dict) or set(assignments) != set(activities):
        return {
            "parse_status": "invalid",
            "parse_errors": ["assignment_fields"],
            "plan": None,
        }
    for activity in activities:
        assigned = assignments[activity]
        if isinstance(assigned, bool) or not isinstance(assigned, int):
            return {
                "parse_status": "invalid",
                "parse_errors": [f"assignment_type:{activity}"],
                "plan": None,
            }
        if assigned not in public_request["allowed_starts"][activity]:
            return {
                "parse_status": "invalid",
                "parse_errors": [f"assignment_domain:{activity}"],
                "plan": None,
            }
    return {
        "parse_status": "valid",
        "parse_errors": [],
        "plan": {"request_id": value["request_id"], "assignments": deepcopy(assignments)},
    }


def cpu_parser_controls(public_request: Mapping[str, Any]) -> list[JsonDict]:
    """Exercise valid and invalid formats without adding model invocations."""

    valid_plan = {
        "request_id": public_request["request_id"],
        "assignments": {
            activity: public_request["allowed_starts"][activity][0]
            for activity in public_request["activities"]
        },
    }
    controls = (("valid", canonical_json(valid_plan)), ("invalid", "```json\n{}\n```"))
    rows: list[JsonDict] = []
    for expected, raw in controls:
        parsed = decode_public_plan(raw, public_request)
        rows.append(
            {
                "control": f"cpu_{expected}_format",
                "expected": expected,
                "observed": parsed["parse_status"],
                "passed": parsed["parse_status"] == expected,
                "inference_substrate": "cpu_parser_control",
                "raw_reply": raw,
            }
        )
    return rows


def cpu_embedded_tokenizer_check(
    model_path: str,
    *,
    probe: Callable[[str | None], tuple[bool, str]] = gguf_tokenizer_loadable,
) -> tuple[bool, str]:
    """Load GGUF vocabulary on CPU so preflight cannot occupy the target GPU."""

    previous_cuda = os.environ.get("CUDA_VISIBLE_DEVICES")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    try:
        return probe(model_path)
    finally:
        if previous_cuda is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = previous_cuda


def build_call_row(
    *,
    call_index: int,
    request: Mapping[str, Any],
    prompt: str,
    response: Mapping[str, Any],
    runtime_identity: Mapping[str, Any],
) -> JsonDict:
    """Retain one response even when public-plan parsing fails."""

    raw_reply = str(response.get("raw_reply") or "")
    parsed = decode_public_plan(raw_reply, request)
    error_text = response.get("error")
    terminal_state = "request_error" if error_text else "response"
    return {
        "call_index": int(call_index),
        "request_id": request.get("request_id"),
        "public_request": deepcopy(dict(request)),
        "prompt": prompt,
        "prompt_sha256": sha256_text(prompt),
        "raw_reply": raw_reply,
        "raw_reply_sha256": sha256_text(raw_reply),
        "raw_response": deepcopy(dict(response.get("raw_response") or {})),
        "parse_status": parsed["parse_status"],
        "parse_errors": deepcopy(parsed["parse_errors"]),
        "decoded_plan": deepcopy(parsed["plan"]),
        "attempted": True,
        "terminal_state": terminal_state,
        "error": error_text,
        "prompt_tokens": int(response.get("prompt_tokens", 0) or 0),
        "completion_tokens": int(response.get("completion_tokens", 0) or 0),
        "max_generated_tokens": MAX_GENERATED_TOKENS,
        "latency_s": float(response.get("latency_s", 0.0) or 0.0),
        "finish_reason": response.get("finish_reason"),
        "runtime_identity_receipt": deepcopy(dict(runtime_identity)),
        "censored": terminal_state != "response",
    }


def _same_model(left: Any, right: Any) -> bool:
    """Compare served model labels without trusting a request-supplied name."""

    if not left or not right:
        return False
    left_text = str(left)
    right_text = str(right)
    return left_text == right_text or Path(left_text).name == Path(right_text).name


def _served_identity_sound(identity: Mapping[str, Any], raw_response: Mapping[str, Any]) -> bool:
    """Bind the served alias to the model path in the owned process command."""

    command_value = identity.get("command")
    command = (
        list(command_value)
        if isinstance(command_value, Sequence) and not isinstance(command_value, (str, bytes))
        else []
    )
    command_model = next(
        (
            command[index + 1]
            for index, argument in enumerate(command[:-1])
            if argument in {"--model", "-m"}
        ),
        None,
    )
    served_model = identity.get("served_model")
    return bool(
        identity.get("model_path")
        and identity.get("model_sha256")
        and _same_model(command_model, served_model)
        and _same_model(served_model, raw_response.get("model"))
    )


def reduce_raw_calls(
    rows: Sequence[Mapping[str, Any]], *, load_receipt: Mapping[str, Any]
) -> JsonDict:
    """Separate transport integrity from whether a reply is a usable plan."""

    attempted_rows = [row for row in rows if row.get("attempted") is True]
    completed_rows = [row for row in attempted_rows if row.get("terminal_state") == "response"]
    failed_rows = [row for row in attempted_rows if row.get("terminal_state") == "request_error"]
    cancelled_rows = [row for row in rows if row.get("terminal_state") == "cancelled"]
    in_flight_rows = [row for row in rows if row.get("terminal_state") == "in_flight"]
    identity_sound = len(rows) == 4
    for row in rows:
        identity = row.get("runtime_identity_receipt")
        raw_response = row.get("raw_response")
        identity_sound = bool(
            identity_sound
            and isinstance(identity, Mapping)
            and identity.get("owned_by_task") is True
            and identity.get("cuda_provenance_ok") is True
            and identity.get("pid") is not None
            and identity.get("start_time_ticks") is not None
            and identity.get("lease_id")
            and identity.get("gpu_uuid")
            and identity.get("model_sha256")
            and isinstance(raw_response, Mapping)
            and _served_identity_sound(identity, raw_response)
        )
    transport_ready = int(
        load_receipt.get("completed") is True
        and identity_sound
        and len(completed_rows) == 4
        and not failed_rows
        and not cancelled_rows
        and not in_flight_rows
    )
    return {
        "plan_transport_ready_score": transport_ready,
        "usable_plan_count": sum(row.get("parse_status") == "valid" for row in rows),
        "invocation_counts": {
            "model_loads_attempted": int(load_receipt.get("attempted") is True),
            "model_loads_completed": int(load_receipt.get("completed") is True),
            "model_loads_failed": int(load_receipt.get("failed") is True),
            "model_loads_cancelled": int(load_receipt.get("cancelled") is True),
            "model_loads_in_flight": int(load_receipt.get("in_flight") is True),
            "generation_calls_attempted": len(attempted_rows),
            "generation_calls_completed": len(completed_rows),
            "generation_calls_failed": len(failed_rows),
            "generation_calls_cancelled": len(cancelled_rows),
            "generation_calls_in_flight": len(in_flight_rows),
        },
    }


def independent_reduce(artifact: Mapping[str, Any]) -> list[str]:
    """Rebuild headline counts only from the retained raw call manifest."""

    errors: list[str] = []
    manifest = artifact.get("raw_call_manifest")
    calls = manifest.get("calls") if isinstance(manifest, Mapping) else None
    rows = artifact.get("rows")
    if not isinstance(calls, list) or not isinstance(rows, list):
        return ["raw_calls_unavailable"]
    if rows != calls:
        errors.append("rows_manifest_mismatch")
    reduced = reduce_raw_calls(calls, load_receipt=dict(artifact.get("load_receipt") or {}))
    if artifact.get("usable_plan_count") != reduced["usable_plan_count"]:
        errors.append("usable_plan_count_mismatch")
    observed_transport = artifact.get(
        "observed_plan_transport_ready_score", artifact.get("plan_transport_ready_score")
    )
    if observed_transport != reduced["plan_transport_ready_score"]:
        errors.append("plan_transport_ready_score_mismatch")
    if artifact.get("invocation_counts") != reduced["invocation_counts"]:
        errors.append("invocation_counts_mismatch")
    return errors


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:  # pragma: no cover
    """Publish one complete object by replacing a same-directory temporary."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _phase_close(
    spans: list[JsonDict], name: str, started: float, units: int, checkpoint: str | None = None
) -> None:  # pragma: no cover
    """Close one measured phase and record its durable checkpoint."""

    ended = time.monotonic()
    spans.append(
        {
            "phase": name,
            "start_elapsed_s": round(started, 6),
            "end_elapsed_s": round(ended, 6),
            "duration_s": round(ended - started, 6),
            "completed_units": units,
            "checkpoint": checkpoint,
        }
    )


def _load_json(path: Path) -> JsonDict:  # pragma: no cover
    """Reject missing or non-object JSON before dependent work starts."""

    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected_object:{path}")
    return value


def _collect_preconditions(
    root: Path, run_date: str
) -> tuple[list[JsonDict], JsonDict]:  # pragma: no cover
    """Authenticate the producer, public bytes, current model, runner, and GPU."""

    progress("preconditions", "start")
    os.environ["CARNOT_FORCE_LIVE"] = "1"
    checks: list[JsonDict] = []
    context: JsonDict = {}
    checks.append(
        gate_row(
            "run_date",
            "execution_contract",
            "run_date",
            RUN_DATE,
            run_date,
            run_date == RUN_DATE,
            "This run is tied to the V645 execution date.",
        )
    )
    required = {
        "producer": root / PRODUCER_PATH,
        "public_manifest": PUBLIC_MANIFEST_PATH,
        "module": root / MODULE_PATH,
        "entrypoint": root / WRAPPER_PATH,
        "tests": root / TEST_PATH,
        "spec": root / SPEC_PATH,
        "validation_runner": root / "python/carnot/reporting/experiment_7303_validation_scope.py",
        "e2e_plan": root / "ops/e2e-test-plan.md",
        "exclusion_manifest": root / "ops/exclusion_manifest.yaml",
        "research_program": root / "research-program.md",
        "research_references": root / "research-references.md",
    }
    observed_paths = {name: path.is_file() for name, path in required.items()}
    observed_paths["spec_has_req"] = bool(
        required["spec"].is_file() and "REQ-CL-7347" in required["spec"].read_text(encoding="utf-8")
    )
    checks.append(
        gate_row(
            "required_source_paths",
            "repository",
            "preconditions_checked",
            {name: True for name in observed_paths},
            observed_paths,
            all(observed_paths.values()),
            "All exact inputs and the driving requirement must exist before inference.",
        )
    )
    producer: JsonDict = {}
    public_manifest: JsonDict = {}
    if required["producer"].is_file() and required["public_manifest"].is_file():
        producer = _load_json(required["producer"])
        public_manifest = _load_json(required["public_manifest"])
    expected_producer = {
        "status": "complete",
        "milestone": MILESTONE,
        "executor_fixture_ready_score": 1,
        "flagged_adversarial": False,
        "verdict_class": "circular_positive",
    }
    observed_producer = {key: producer.get(key) for key in expected_producer}
    quarantined = "quarantine" in str(required["producer"]).lower()
    producer_ok = observed_producer == expected_producer and not quarantined
    checks.append(
        gate_row(
            "same_milestone_executor_fixture",
            "exp7344-executor-fixture",
            "executor_fixture_ready_score",
            {**expected_producer, "quarantined": False},
            {**observed_producer, "quarantined": quarantined},
            producer_ok,
            "A blocked, partial, disqualified, or quarantined producer cannot authorize model work.",
        )
    )
    public_hash = (
        sha256_file(required["public_manifest"]) if required["public_manifest"].is_file() else None
    )
    declared_public_hash = dict(producer.get("source_artifact_hashes") or {}).get(
        str(required["public_manifest"])
    )
    checks.append(
        gate_row(
            "public_manifest_bytes",
            "exp7344-executor-fixture",
            "source_artifact_hashes",
            declared_public_hash,
            public_hash,
            bool(declared_public_hash and public_hash == declared_public_hash),
            "The canary must use the exact public bytes authenticated by its producer.",
        )
    )
    selected: list[JsonDict] = []
    selection_error = None
    try:
        selected = select_development_requests(public_manifest)
    except (TypeError, ValueError, KeyError) as exc:
        selection_error = f"{type(exc).__name__}:{exc}"
    checks.append(
        gate_row(
            "four_development_requests",
            "public_manifest",
            "sample_size_budget",
            {"count": 4, "selection_error": None},
            {"count": len(selected), "selection_error": selection_error},
            len(selected) == 4 and selection_error is None,
            "The fixed budget permits four public requests and no replacement samples.",
        )
    )

    resolved = cached_current_model(gpu_index=0, preferred_quant=QUANTIZATION)
    model_path = Path(str(resolved.get("model_path"))) if resolved else None
    model_exists = bool(model_path and model_path.is_file())
    model_hash = _content_addressed_hash(model_path) if model_exists and model_path else None
    metadata = read_gguf_metadata(model_path) if model_exists and model_path else {}
    tokenizer_ok, tokenizer_detail = cpu_embedded_tokenizer_check(
        str(model_path) if model_path else ""
    )
    resolved_spec = {
        "hf_id": resolved.get("hf_id") if resolved else None,
        "quantization": QUANTIZATION,
        "path": str(model_path.resolve()) if model_exists and model_path else None,
        "revision": snapshot_revision(model_path) if model_exists and model_path else None,
        "bytes": model_path.stat().st_size if model_exists and model_path else None,
        "sha256": model_hash,
        "hash_source": "content_addressed_cache_target" if model_hash else None,
        "embedded_tokenizer_loadable": tokenizer_ok,
        "embedded_tokenizer_detail": tokenizer_detail,
        "embedded_chat_template": metadata.get("chat_template_present") is True,
        "chat_template_sha256": metadata.get("chat_template_sha256"),
        "runtime_settings": {
            "max_generated_tokens": MAX_GENERATED_TOKENS,
            "model_load_timeout_s": MODEL_LOAD_TIMEOUT_S,
            "inference_window_timeout_s": INFERENCE_WINDOW_TIMEOUT_S,
            "temperature": 0.0,
            "top_k": 1,
            "top_p": 1.0,
            "seed": RANDOM_SEED["development"],
            "cuda_gpu_layers": "all",
            "parallel_requests": 1,
        },
    }
    model_ok = bool(
        model_exists
        and resolved_spec["hf_id"] == QWEN_MODEL_ID
        and model_path
        and QUANTIZATION.lower() in model_path.name.lower()
        and model_hash
        and tokenizer_ok
        and resolved_spec["embedded_chat_template"]
    )
    checks.append(
        gate_row(
            "cached_current_model",
            "cached_current_model",
            "MODEL_SPECS",
            {
                "hf_id": QWEN_MODEL_ID,
                "quantization": QUANTIZATION,
                "exists": True,
                "content_addressed": True,
                "embedded_tokenizer_loadable": True,
                "embedded_chat_template": True,
            },
            {
                "hf_id": resolved_spec["hf_id"],
                "quantization": resolved_spec["quantization"],
                "exists": model_exists,
                "content_addressed": bool(model_hash),
                "embedded_tokenizer_loadable": tokenizer_ok,
                "embedded_chat_template": resolved_spec["embedded_chat_template"],
            },
            model_ok,
            "The owned runner must load the current Q4 GGUF and its embedded tokenizer.",
        )
    )

    server = resolve_native_llama_server()
    progress("preconditions", "before_subprocess", operation="native_runner_capabilities")
    runner_rows = lease_preflight.collect_runner_capabilities(server)
    progress("preconditions", "after_subprocess", operation="native_runner_capabilities")
    runner_errors = lease_preflight.runner_capability_errors(runner_rows)
    checks.append(
        gate_row(
            "native_cuda_runner",
            "native_llama_server",
            "runtime_identity_receipt",
            [],
            runner_errors,
            not runner_errors,
            "Static runner receipts must confirm CUDA linkage and bounded generation.",
        )
    )
    progress("preconditions", "before_subprocess", operation="gpu_inventory")
    process_rows, query_receipts = lease_preflight.collect_gpu_process_rows()
    lease_rows = lease_preflight.scan_lease_rows(lease_preflight.LEASE_RUNTIME_DIR, process_rows)
    classified = lease_preflight.classify_process_rows(
        process_rows, lease_rows, current_task_id=TASK_ID
    )
    decision = lease_preflight.readiness_decision(
        classified,
        lease_rows,
        [
            {
                "repository": QWEN_MODEL_ID,
                "filename": model_path.name if model_path else None,
                "path": str(model_path) if model_path else None,
                "real_path": str(model_path.resolve()) if model_exists and model_path else None,
                "revision": resolved_spec["revision"],
                "bytes": resolved_spec["bytes"],
                "sha256": model_hash,
                "hash_source": resolved_spec["hash_source"],
                "weights_opened": False,
                "valid": model_ok,
            }
        ],
        runner_rows,
    )
    progress("preconditions", "after_subprocess", operation="gpu_inventory")
    query_ok = all(row.get("returncode") == 0 for row in query_receipts)
    available = list(decision.get("available_gpu_uuids", []))
    checks.append(
        gate_row(
            "cuda_inventory_and_owned_capacity",
            "nvidia-smi_and_gpu_lease_journal",
            "runtime_identity_receipt",
            {"query_ok": True, "minimum_available_gpu_count": 1, "conflicts": []},
            {
                "query_ok": query_ok,
                "available_gpu_uuids": available,
                "conflicts": decision.get("conflicting_processes", []),
                "conflicting_lease_ids": decision.get("conflicting_lease_ids", []),
            },
            query_ok and bool(available),
            "Actual CUDA inventory, free memory, and lease ownership must permit one runner.",
        )
    )
    checks.append(
        gate_row(
            "force_live_mode",
            "process_environment",
            "inference_substrate",
            "1",
            os.environ.get("CARNOT_FORCE_LIVE"),
            os.environ.get("CARNOT_FORCE_LIVE") == "1",
            "The canary forbids simulated fallback.",
        )
    )
    context.update(
        {
            "producer": producer,
            "public_manifest": public_manifest,
            "selected_requests": selected,
            "model_path": model_path,
            "model_spec": resolved_spec,
            "server_path": server,
            "runner": deepcopy(runner_rows[0]) if runner_rows else {},
            "process_rows": classified,
            "lease_rows": lease_rows,
            "query_receipts": query_receipts,
            "available_gpu_uuids": available,
        }
    )
    progress("preconditions", "complete", passed=all(row["passed"] for row in checks))
    return checks, context


def _selected_device(context: Mapping[str, Any], gpu_uuid: str) -> JsonDict:  # pragma: no cover
    """Return the inventory row for the GPU selected by the lease decision."""

    return next(
        deepcopy(dict(row)) for row in context["process_rows"] if row.get("gpu_uuid") == gpu_uuid
    )


def _served_model(port: int) -> JsonDict:  # pragma: no cover
    """Read the server's own model identity instead of trusting request text."""

    started = time.monotonic()
    with urlrequest.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=10.0) as response:
        body = json.loads(response.read().decode("utf-8"))
    data = body.get("data") if isinstance(body, Mapping) else None
    first = data[0] if isinstance(data, list) and data and isinstance(data[0], Mapping) else {}
    return {
        "raw_response": body,
        "served_model": first.get("id"),
        "latency_s": time.monotonic() - started,
    }


def _chat_request(port: int, prompt: str, timeout_s: float) -> JsonDict:  # pragma: no cover
    """Send one unconstrained public prompt and retain the exact native response."""

    payload = {
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "top_k": 1,
        "top_p": 1.0,
        "seed": RANDOM_SEED["development"],
        "cache_prompt": False,
        "max_tokens": MAX_GENERATED_TOKENS,
    }
    encoded = canonical_json(payload).encode("utf-8")
    native_request = urlrequest.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=encoded,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.monotonic()
    try:
        with urlrequest.urlopen(native_request, timeout=timeout_s) as response:
            body = json.loads(response.read().decode("utf-8"))
        choices = list(body.get("choices") or [{}])
        choice = dict(choices[0])
        message = dict(choice.get("message") or {})
        usage = dict(body.get("usage") or {})
        return {
            "raw_request": payload,
            "raw_request_bytes_b64": base64.b64encode(encoded).decode("ascii"),
            "raw_reply": str(message.get("content") or message.get("reasoning_content") or ""),
            "raw_response": body,
            "prompt_tokens": int(usage.get("prompt_tokens", 0) or 0),
            "completion_tokens": int(usage.get("completion_tokens", 0) or 0),
            "latency_s": time.monotonic() - started,
            "finish_reason": choice.get("finish_reason"),
            "error": None,
        }
    except Exception as exc:  # noqa: BLE001 - the terminal row retains exact failure text.
        return {
            "raw_request": payload,
            "raw_request_bytes_b64": base64.b64encode(encoded).decode("ascii"),
            "raw_reply": "",
            "raw_response": {},
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "latency_s": time.monotonic() - started,
            "finish_reason": None,
            "error": f"{type(exc).__name__}:{exc}",
        }


def _live_capture(context: Mapping[str, Any], raw_dir: Path) -> JsonDict:  # pragma: no cover
    """Run one owned server and four bounded calls, then release only that server."""

    capture_started = time.monotonic()
    spans: list[JsonDict] = []
    rows: list[JsonDict] = []
    gpu_uuid = str(context["available_gpu_uuids"][0])
    device = _selected_device(context, gpu_uuid)
    gpu_index = int(device["gpu_index"])
    model_path = Path(str(context["model_path"]))
    model_spec = deepcopy(dict(context["model_spec"]))
    port = _free_port()
    command = _server_command(Path(context["server_path"]), model_path, port)
    contract = supervisor_contract(
        outer_deadline_s=INFERENCE_WINDOW_TIMEOUT_S,
        health_timeout_s=MODEL_LOAD_TIMEOUT_S,
        token_timeout_s=REQUEST_TIMEOUT_S,
        cleanup_grace_s=30.0,
        kill_after_cleanup_timeout_s=10.0,
        retry_budget=0,
        endurance_interval_s=0.0,
        endurance_sample_count=1,
    )
    supervisor = NativeLlamaServerSupervisor(command, raw_dir, contract)
    lease: lease_api.GpuLease | None = None
    identity: JsonDict = {}
    runtime_identity: JsonDict = {}
    served: JsonDict = {}
    provenance: JsonDict = {"provenance_ok": False}
    snapshots: list[JsonDict] = []
    cleanup: JsonDict = {"action": "not_started", "leak_free": True}
    lease_release: JsonDict = {"released": False}
    load_receipt: JsonDict = {
        "attempted": False,
        "completed": False,
        "failed": False,
        "cancelled": False,
        "in_flight": False,
        "error": None,
    }
    runtime_error: str | None = None
    previous_cuda = os.environ.get("CUDA_VISIBLE_DEVICES")
    try:
        load_started = time.monotonic()
        progress("load", "start", model=QWEN_MODEL_ID, gpu_uuid=gpu_uuid)
        snapshots.append(_gpu_snapshot("before_model_load", phase=1))
        lease = lease_api.GpuLease.acquire(
            runtime_dir=lease_preflight.LEASE_RUNTIME_DIR,
            task_id=TASK_ID,
            device_uuid=gpu_uuid,
            expected_model=str(model_path),
            vram_before_mb=int(device.get("gpu_memory_used_mb", 0) or 0),
            ttl_s=INFERENCE_WINDOW_TIMEOUT_S,
        )
        lease.transition("admitted")
        lease.transition("loading")
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
        load_receipt["attempted"] = True
        progress("load", "before_model_load", command=canonical_json(command))
        identity = supervisor.launch()
        with _stream_server_log(supervisor), _heartbeat(1, "model_load", lambda: 0, 1):
            health = _wait_for_health(supervisor, port, MODEL_LOAD_TIMEOUT_S)
        progress("load", "after_model_load", healthy=health.get("ok"))
        if health.get("ok") is not True:
            raise RuntimeError(f"model_health_failed:{health.get('classification')}")
        served = _served_model(port)
        resident = _gpu_snapshot("model_resident", phase=1)
        snapshots.append(resident)
        provenance = _gpu_provenance(resident, identity, gpu_uuid, supervisor.stderr_tail())
        if provenance.get("provenance_ok") is not True:
            raise RuntimeError("cuda_placement_or_owned_vram_unconfirmed")
        if not _same_model(model_path, served.get("served_model")):
            raise RuntimeError(f"served_model_identity_mismatch:{served.get('served_model')}")
        load_receipt["completed"] = True
        lease.transition("resident", vram_mb=int(provenance["task_owned_vram_mb"]))
        lease.transition("inferencing")
        runtime_identity = {
            **deepcopy(identity),
            "owned_by_task": identity.get("owned_by_task") is True,
            "model_path": str(model_path.resolve()),
            "model_sha256": model_spec["sha256"],
            "model_bytes": model_spec["bytes"],
            "model_revision": model_spec["revision"],
            "quantization": QUANTIZATION,
            "served_model": served.get("served_model"),
            "served_model_response": deepcopy(served.get("raw_response")),
            "gpu_uuid": gpu_uuid,
            "gpu_index": gpu_index,
            "lease_id": lease.lease_id,
            "cuda_provenance_ok": provenance.get("provenance_ok") is True,
            "task_owned_vram_mb": provenance.get("task_owned_vram_mb"),
        }
        _phase_close(spans, "model_load", load_started, 1, str(supervisor.log_path))

        generation_started = time.monotonic()
        progress("generation", "start", planned_calls=4)
        for index, public_request in enumerate(context["selected_requests"]):
            elapsed = time.monotonic() - capture_started
            if elapsed >= INFERENCE_WINDOW_TIMEOUT_S:
                rows.append(
                    {
                        "call_index": index,
                        "request_id": public_request.get("request_id"),
                        "public_request": deepcopy(dict(public_request)),
                        "prompt": render_public_prompt(public_request),
                        "prompt_sha256": sha256_text(render_public_prompt(public_request)),
                        "raw_reply": "",
                        "raw_reply_sha256": sha256_text(""),
                        "raw_response": {},
                        "parse_status": "invalid",
                        "parse_errors": ["cancelled_before_attempt"],
                        "decoded_plan": None,
                        "attempted": False,
                        "terminal_state": "cancelled",
                        "error": "inference_window_timeout",
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "max_generated_tokens": MAX_GENERATED_TOKENS,
                        "latency_s": 0.0,
                        "finish_reason": None,
                        "runtime_identity_receipt": deepcopy(runtime_identity),
                        "censored": True,
                    }
                )
                continue
            prompt = render_public_prompt(public_request)
            remaining = max(0.25, INFERENCE_WINDOW_TIMEOUT_S - elapsed)
            progress("generation", "before_call", completed=index, total=4)
            with _heartbeat(2, "plan_generation", lambda: index, 4):
                response = _chat_request(port, prompt, min(REQUEST_TIMEOUT_S, remaining))
            row = build_call_row(
                call_index=index,
                request=public_request,
                prompt=prompt,
                response=response,
                runtime_identity=runtime_identity,
            )
            rows.append(row)
            _atomic_json(raw_dir / f"call_{index:02d}.json", row)
            progress(
                "generation",
                "after_call",
                completed=index + 1,
                total=4,
                terminal_state=row["terminal_state"],
                parse_status=row["parse_status"],
            )
            lease.heartbeat()
        _phase_close(spans, "generation", generation_started, len(rows), str(raw_dir))
        progress("generation", "complete", completed=len(rows), total=4)
    except Exception as exc:  # noqa: BLE001 - exact runtime failure is terminal evidence.
        runtime_error = f"{type(exc).__name__}:{exc}"
        load_receipt["failed"] = load_receipt["attempted"] and not load_receipt["completed"]
        load_receipt["error"] = runtime_error
        progress("runtime", "error", error=runtime_error, completed_calls=len(rows))
    finally:
        if previous_cuda is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = previous_cuda
        cleanup_started = time.monotonic()
        progress("cleanup", "start", pid=identity.get("pid"))
        cleanup = supervisor.cleanup()
        if supervisor.proc is not None:
            try:
                supervisor.proc.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                pass
        after = _gpu_snapshot("after_cleanup", phase=3)
        snapshots.append(after)
        if lease is not None:
            try:
                lease_release = _release_lease(
                    lease,
                    cleanup,
                    after,
                    identity,
                    len(rows) == 4 and provenance.get("provenance_ok") is True,
                )
            except lease_api.LeaseError as exc:
                lease_release = {"released": False, "error": f"{type(exc).__name__}:{exc}"}
                lease.close()
        _phase_close(spans, "cleanup", cleanup_started, 1)
        progress(
            "cleanup",
            "complete",
            process_released=cleanup.get("leak_free"),
            lease_released=lease_release.get("released"),
        )
    while len(rows) < 4:
        index = len(rows)
        public_request = context["selected_requests"][index]
        prompt = render_public_prompt(public_request)
        rows.append(
            {
                "call_index": index,
                "request_id": public_request.get("request_id"),
                "public_request": deepcopy(dict(public_request)),
                "prompt": prompt,
                "prompt_sha256": sha256_text(prompt),
                "raw_reply": "",
                "raw_reply_sha256": sha256_text(""),
                "raw_response": {},
                "parse_status": "invalid",
                "parse_errors": ["cancelled_after_runtime_failure"],
                "decoded_plan": None,
                "attempted": False,
                "terminal_state": "cancelled",
                "error": runtime_error,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "max_generated_tokens": MAX_GENERATED_TOKENS,
                "latency_s": 0.0,
                "finish_reason": None,
                "runtime_identity_receipt": deepcopy(runtime_identity),
                "censored": True,
            }
        )
    return {
        "rows": rows,
        "load_receipt": load_receipt,
        "runtime_identity": runtime_identity,
        "served_model_receipt": served,
        "gpu_receipts": {
            "snapshots": snapshots,
            "provenance": provenance,
            "cleanup": cleanup,
            "lease_release": lease_release,
        },
        "phase_spans": spans,
        "runtime_error": runtime_error,
    }


def _source_hashes(
    root: Path, context: Mapping[str, Any], raw_manifest: Path
) -> JsonDict:  # pragma: no cover
    """Authenticate the producer, code, contract, model, and raw evidence."""

    paths = (
        Path("AGENTS.md"),
        Path("CLAUDE.md"),
        Path("CODEX.md"),
        Path("research-program.md"),
        Path("research-references.md"),
        Path("ops/exclusion_manifest.yaml"),
        Path("ops/e2e-test-plan.md"),
        SPEC_PATH,
        MODULE_PATH,
        WRAPPER_PATH,
        TEST_PATH,
        PRODUCER_PATH,
        Path("python/carnot/reporting/experiment_7303_validation_scope.py"),
        Path("scripts/adversarial_verify.py"),
        Path("scripts/verdict_row_consistency_lint.py"),
    )
    hashes = {path.as_posix(): sha256_file(root / path) for path in paths}
    hashes[str(PUBLIC_MANIFEST_PATH)] = sha256_file(PUBLIC_MANIFEST_PATH)
    hashes[str(raw_manifest)] = sha256_file(raw_manifest)
    model_spec = dict(context.get("model_spec") or {})
    if model_spec.get("path") and model_spec.get("sha256"):
        hashes[str(model_spec["path"])] = str(model_spec["sha256"])
    return hashes


def _acceptance_gates(
    preconditions: Sequence[Mapping[str, Any]],
    controls: Sequence[Mapping[str, Any]],
    reduced: Mapping[str, Any],
    validation_ok: bool,
    terminal_ok: bool,
    flagged_adversarial: bool,
) -> JsonDict:  # pragma: no cover
    """Expose resource, transport, parser, and validation gates separately."""

    return {
        "preconditions": {
            "expected": True,
            "observed": all(row.get("passed") is True for row in preconditions),
            "passed": all(row.get("passed") is True for row in preconditions),
            "principle": "Every exact producer and runtime check precedes dependent work.",
        },
        "cpu_parser_controls": {
            "expected": True,
            "observed": all(row.get("passed") is True for row in controls),
            "passed": all(row.get("passed") is True for row in controls),
            "principle": "Valid and invalid public formats must exercise the production parser.",
        },
        "plan_transport": {
            "expected": 1,
            "observed": reduced.get("plan_transport_ready_score"),
            "passed": reduced.get("plan_transport_ready_score") == 1,
            "principle": "Ownership, identity, CUDA evidence, and four captures qualify transport.",
        },
        "affected_validation": {
            "expected": True,
            "observed": validation_ok,
            "passed": validation_ok,
            "principle": "Changed code and its exact tests must pass current scoped checks.",
        },
        "terminal_validation": {
            "expected": True,
            "observed": terminal_ok,
            "passed": terminal_ok,
            "principle": "Independent reduction and both strict readers must pass.",
        },
        "adversarial_clear": {
            "expected": False,
            "observed": flagged_adversarial,
            "passed": not flagged_adversarial,
            "principle": "A critical finding prevents readiness or promotion.",
        },
    }


def _base_artifact(run_date: str, started_utc: str) -> JsonDict:  # pragma: no cover
    """Create every required field before a fallible external resource check."""

    return {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "status": "blocked",
        "run_date": run_date,
        "started_at_utc": started_utc,
        "completed_at_utc": None,
        "preconditions_checked": [],
        "MODEL_SPECS": deepcopy(MODEL_SPECS),
        "model_specs": [],
        "model_invoked": False,
        "invocation_counts": {
            "model_loads_attempted": 0,
            "model_loads_completed": 0,
            "model_loads_failed": 0,
            "model_loads_cancelled": 0,
            "model_loads_in_flight": 0,
            "generation_calls_attempted": 0,
            "generation_calls_completed": 0,
            "generation_calls_failed": 0,
            "generation_calls_cancelled": 0,
            "generation_calls_in_flight": 0,
        },
        "inference_substrate": "blocked_no_run",
        "inference_substrate_class": "blocked_no_run",
        "execution_venue": "host",
        "duration_s": 0.0,
        "phase_spans": [],
        "random_seed": deepcopy(RANDOM_SEED),
        "reproducibility_checksum": None,
        "source_artifact_hashes": {},
        "rows": [],
        "sample_size_budget": {
            "planned_units": 4,
            "attempted_units": 0,
            "completed_units": 0,
            "censored_units": 4,
            "max_generated_tokens_per_unit": MAX_GENERATED_TOKENS,
            "model_load_timeout_s": MODEL_LOAD_TIMEOUT_S,
            "inference_window_timeout_s": INFERENCE_WINDOW_TIMEOUT_S,
            "stopping_rule": "four terminal calls or the fixed inference deadline; no replacement",
        },
        "acceptance_gate_results": {},
        "gate_check_summary": {},
        "verifier_is_oracle": False,
        "honest_verdict": "blocked_not_started",
        "verdict_class": "blocked",
        "flagged_adversarial": True,
        "validation_receipts": [],
        "repository_health": {
            "status": "not_checked",
            "historical_failures": [],
            "affects_required_checks": False,
        },
        "field_principles": deepcopy(FIELD_PRINCIPLES),
        "plan_transport_ready_score": 0,
        "observed_plan_transport_ready_score": 0,
        "usable_plan_count": 0,
        "value_ready_score": 0,
        "promotion_ready_score": 0,
        "runtime_identity_receipt": {},
        "raw_call_manifest": {"schema": "carnot.exp7347.raw_calls.v1", "calls": []},
        "load_receipt": {
            "attempted": False,
            "completed": False,
            "failed": False,
            "cancelled": False,
            "in_flight": False,
            "error": None,
        },
        "cpu_parser_controls": [],
        "gpu_receipts": {},
    }


def _blocked_artifact(
    root: Path,
    artifact: JsonDict,
    checks: Sequence[Mapping[str, Any]],
    started: float,
    output_path: Path,
) -> JsonDict:  # pragma: no cover
    """Write one honest external block without success-shaped model evidence."""

    summary = gate_check_summary(checks)
    failed = str(summary.get("failed_check") or "unknown_precondition")
    artifact.update(
        {
            "preconditions_checked": [deepcopy(dict(row)) for row in checks],
            "completed_at_utc": _utc_now(),
            "duration_s": round(time.monotonic() - started, 6),
            "gate_check_summary": summary,
            "honest_verdict": f"blocked_{failed}",
            "flagged_adversarial": True,
        }
    )
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    _atomic_json(output_path, artifact)
    return artifact


def run_affected_validation(root: Path, raw_dir: Path) -> JsonDict:  # pragma: no cover
    """Run the shipped fixed validation set with explicit task-owned paths."""

    basetemp = Path("/tmp/carnot-exp7347-v645-scoped")
    coverage_file = Path("/tmp/carnot-exp7347-v645-coverage/.coverage")
    basetemp.mkdir(parents=True, exist_ok=True)
    coverage_file.parent.mkdir(parents=True, exist_ok=True)
    return run_scoped_validation(
        root,
        test_paths=[str(TEST_PATH)],
        changed_modules=[str(MODULE_PATH)],
        static_paths=[str(WRAPPER_PATH)],
        basetemp=basetemp,
        coverage_file=coverage_file,
        log_dir=raw_dir / "validation/scoped",
        historical_failures=[],
    )


def run_terminal_validation(
    root: Path, candidate: Path, raw_dir: Path
) -> list[JsonDict]:  # pragma: no cover
    """Cold-reduce the candidate and run both strict artifact readers."""

    python = str(root / ".venv/bin/python")
    reducer = (
        "import json,pathlib;"
        "from carnot.experiment_7347_v645_plan_canary import independent_reduce;"
        f"v=json.loads(pathlib.Path({str(candidate)!r}).read_text());"
        "e=independent_reduce(v);print(e,flush=True);raise SystemExit(bool(e))"
    )
    commands = [
        CommandSpec("independent_reducer", (python, "-u", "-c", reducer), "candidate"),
        CommandSpec(
            "adversarial_verify",
            (python, "-u", "scripts/adversarial_verify.py", str(candidate)),
            "candidate",
        ),
        CommandSpec(
            "verdict_row_consistency_strict",
            (python, "-u", "scripts/verdict_row_consistency_lint.py", "--strict", str(candidate)),
            "candidate",
        ),
    ]
    return run_commands(root, commands, log_dir=raw_dir / "validation/terminal")


def _all_receipts_pass(
    receipts: Sequence[Mapping[str, Any]], names: Sequence[str]
) -> bool:  # pragma: no cover
    """Require one successful receipt for every named current check."""

    return all(
        sum(
            row.get("name") == name and row.get("passed") is True and row.get("exit_code") == 0
            for row in receipts
        )
        == 1
        for name in names
    )


def validate_artifact(artifact: object) -> list[str]:  # pragma: no cover
    """Reject schema drift, forged counts, and nonterminal result shapes."""

    if not isinstance(artifact, Mapping):
        return ["artifact_not_object"]
    errors: list[str] = []
    missing = sorted(REQUIRED_ARTIFACT_FIELDS - set(artifact))
    errors.extend(f"missing_required_field:{field}" for field in missing)
    if errors:
        return errors
    if artifact.get("schema") != SCHEMA or artifact.get("milestone") != MILESTONE:
        errors.append("identity_invalid")
    if artifact.get("run_date") != RUN_DATE or artifact.get("execution_venue") != "host":
        errors.append("lifecycle_invalid")
    if artifact.get("verdict_class") not in {
        "positive",
        "circular_positive",
        "null",
        "blocked",
        "disqualified",
        "partial",
    }:
        errors.append("verdict_class_invalid")
    if artifact.get("status") not in {"complete", "blocked"}:
        errors.append("status_invalid")
    errors.extend(independent_reduce(artifact) if artifact.get("model_invoked") else [])
    counts = dict(artifact.get("invocation_counts") or {})
    if artifact.get("model_invoked") is not (counts.get("model_loads_attempted", 0) > 0):
        errors.append("model_invoked_mismatch")
    if artifact.get("status") == "blocked":
        if artifact.get("verdict_class") != "blocked" or not str(
            artifact.get("honest_verdict", "")
        ).startswith("blocked_"):
            errors.append("blocked_terminal_invalid")
        if artifact.get("inference_substrate_class") != "blocked_no_run":
            errors.append("blocked_substrate_invalid")
    else:
        if not str(artifact.get("honest_verdict", "")).startswith("complete_"):
            errors.append("complete_verdict_invalid")
        if artifact.get("inference_substrate_class") not in {
            "model_bounded_generation",
            "model_load_no_generation",
        }:
            errors.append("complete_substrate_invalid")
    if artifact.get("verdict_class") in {"blocked", "disqualified"} and any(
        artifact.get(field) != 0
        for field in ("plan_transport_ready_score", "value_ready_score", "promotion_ready_score")
    ):
        errors.append("failed_scores_invalid")
    if artifact.get("reproducibility_checksum") != artifact_checksum(artifact):
        errors.append("reproducibility_checksum_mismatch")
    return list(dict.fromkeys(errors))


def run_experiment(
    *, root: Path = REPO_ROOT, run_date: str = RUN_DATE, output_path: Path | None = None
) -> JsonDict:  # pragma: no cover
    """Authenticate, infer, validate, independently reduce, and publish once."""

    started = time.monotonic()
    started_utc = _utc_now()
    output = output_path or root / RESULT_PATH
    raw_dir = root / RAW_DIR
    raw_dir.mkdir(parents=True, exist_ok=True)
    artifact = _base_artifact(run_date, started_utc)
    checks, context = _collect_preconditions(root, run_date)
    if any(row.get("passed") is not True for row in checks):
        progress("write", "blocked_start")
        result = _blocked_artifact(root, artifact, checks, started, output)
        progress("write", "blocked_complete", artifact=output)
        return result

    progress("inference", "start")
    capture = _live_capture(context, raw_dir)
    progress("inference", "complete", calls=len(capture["rows"]))
    evaluation_started = time.monotonic()
    controls = cpu_parser_controls(context["selected_requests"][0])
    reduced = reduce_raw_calls(capture["rows"], load_receipt=capture["load_receipt"])
    raw_manifest = {
        "schema": "carnot.exp7347.raw_calls.v1",
        "producer": EXPERIMENT_ID,
        "calls": deepcopy(capture["rows"]),
    }
    _atomic_json(root / RAW_MANIFEST_PATH, raw_manifest)
    _phase_close(
        capture["phase_spans"], "evaluation", evaluation_started, 4, str(root / RAW_MANIFEST_PATH)
    )

    test_started = time.monotonic()
    progress("tests", "before_scoped_validation")
    validation = run_affected_validation(root, raw_dir)
    progress("tests", "after_scoped_validation", passed=validation.get("required_checks_passed"))
    scoped_receipts = list(validation.get("validation_receipts", []))
    scoped_ok = bool(
        validation.get("required_checks_passed")
        and _all_receipts_pass(scoped_receipts, REQUIRED_CHECK_NAMES)
    )
    _phase_close(capture["phase_spans"], "tests", test_started, len(scoped_receipts))

    generation_attempted = reduced["invocation_counts"]["generation_calls_attempted"] > 0
    authentic_generation = bool(
        generation_attempted
        and capture["gpu_receipts"].get("provenance", {}).get("provenance_ok") is True
    )
    if authentic_generation:
        substrate = "live_llm_inference"
        substrate_class = "model_bounded_generation"
    else:
        substrate = "live_llm_inference"
        substrate_class = "model_load_no_generation"
    repository_health = deepcopy(validation.get("repository_health") or {})
    artifact.update(
        {
            "status": "complete",
            "preconditions_checked": [deepcopy(dict(row)) for row in checks],
            "MODEL_SPECS": deepcopy(MODEL_SPECS),
            "model_specs": [deepcopy(context["model_spec"])],
            "model_invoked": capture["load_receipt"].get("attempted") is True,
            "invocation_counts": deepcopy(reduced["invocation_counts"]),
            "inference_substrate": substrate,
            "inference_substrate_class": substrate_class,
            "phase_spans": deepcopy(capture["phase_spans"]),
            "rows": deepcopy(capture["rows"]),
            "raw_call_manifest": raw_manifest,
            "load_receipt": deepcopy(capture["load_receipt"]),
            "runtime_identity_receipt": deepcopy(capture["runtime_identity"]),
            "gpu_receipts": deepcopy(capture["gpu_receipts"]),
            "cpu_parser_controls": controls,
            "plan_transport_ready_score": int(reduced["plan_transport_ready_score"]),
            "observed_plan_transport_ready_score": int(reduced["plan_transport_ready_score"]),
            "usable_plan_count": int(reduced["usable_plan_count"]),
            "validation_receipts": scoped_receipts,
            "repository_health": repository_health,
            "sample_size_budget": {
                **artifact["sample_size_budget"],
                "attempted_units": reduced["invocation_counts"]["generation_calls_attempted"],
                "completed_units": reduced["invocation_counts"]["generation_calls_completed"],
                "censored_units": sum(row.get("censored") is True for row in capture["rows"]),
            },
        }
    )
    candidate = raw_dir / "measured-terminal-candidate.json"
    artifact["completed_at_utc"] = _utc_now()
    artifact["duration_s"] = round(time.monotonic() - started, 6)
    artifact["flagged_adversarial"] = False
    artifact["acceptance_gate_results"] = _acceptance_gates(
        checks, controls, reduced, scoped_ok, False, False
    )
    artifact["gate_check_summary"] = gate_check_summary(checks)
    artifact["source_artifact_hashes"] = _source_hashes(root, context, root / RAW_MANIFEST_PATH)
    artifact["honest_verdict"] = (
        f"complete_positive_plan_transport_ready_{artifact['usable_plan_count']}_of_4_usable"
        if artifact["plan_transport_ready_score"] == 1 and scoped_ok
        else f"complete_null_plan_transport_not_ready_{artifact['usable_plan_count']}_of_4_usable"
    )
    artifact["verdict_class"] = (
        "positive" if artifact["plan_transport_ready_score"] == 1 and scoped_ok else "null"
    )
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    _atomic_json(candidate, artifact)

    terminal_started = time.monotonic()
    progress("tests", "before_terminal_validation", candidate=candidate)
    terminal_receipts = run_terminal_validation(root, candidate, raw_dir)
    progress("tests", "after_terminal_validation")
    terminal_ok = _all_receipts_pass(terminal_receipts, TERMINAL_CHECK_NAMES)
    artifact["validation_receipts"] = [*scoped_receipts, *terminal_receipts]
    _phase_close(
        artifact["phase_spans"], "terminal_validation", terminal_started, len(terminal_receipts)
    )
    independent_ok = not independent_reduce(artifact)
    adversarial_receipt = next(
        (row for row in terminal_receipts if row.get("name") == "adversarial_verify"), {}
    )
    flagged = not bool(adversarial_receipt.get("passed"))
    disqualified = not (scoped_ok and terminal_ok and independent_ok and not flagged)
    final_transport = int(reduced["plan_transport_ready_score"])
    if disqualified:
        artifact.update(
            {
                "verdict_class": "disqualified",
                "honest_verdict": "complete_disqualified_plan_canary_required_check_failed",
                "plan_transport_ready_score": 0,
                "value_ready_score": 0,
                "promotion_ready_score": 0,
            }
        )
    else:
        artifact.update(
            {
                "verdict_class": "positive" if final_transport == 1 else "null",
                "honest_verdict": (
                    f"complete_positive_plan_transport_ready_{artifact['usable_plan_count']}_of_4_usable"
                    if final_transport == 1
                    else f"complete_null_plan_transport_not_ready_{artifact['usable_plan_count']}_of_4_usable"
                ),
                "plan_transport_ready_score": final_transport,
                "value_ready_score": 0,
                "promotion_ready_score": 0,
            }
        )
    artifact["flagged_adversarial"] = flagged
    artifact["acceptance_gate_results"] = _acceptance_gates(
        checks, controls, reduced, scoped_ok, terminal_ok and independent_ok, flagged
    )
    failed_gate_rows = [
        gate_row(
            name,
            EXPERIMENT_ID,
            "acceptance_gate_results",
            value["expected"],
            value["observed"],
            value["passed"],
            value["principle"],
        )
        for name, value in artifact["acceptance_gate_results"].items()
    ]
    artifact["gate_check_summary"] = gate_check_summary([*checks, *failed_gate_rows])
    write_started = time.monotonic()
    artifact["completed_at_utc"] = _utc_now()
    artifact["duration_s"] = round(time.monotonic() - started, 6)
    _phase_close(artifact["phase_spans"], "write", write_started, 1, str(output))
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    validation_errors = validate_artifact(artifact)
    if validation_errors:
        artifact.update(
            {
                "verdict_class": "disqualified",
                "honest_verdict": "complete_disqualified_internal_artifact_validation_failed",
                "flagged_adversarial": True,
                "plan_transport_ready_score": 0,
                "value_ready_score": 0,
                "promotion_ready_score": 0,
                "internal_validation_errors": validation_errors,
            }
        )
        artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    progress("write", "start", artifact=output)
    _atomic_json(output, artifact)
    progress("write", "complete", artifact=output, verdict=artifact["honest_verdict"])
    return artifact


def _date_argument(value: str) -> str:  # pragma: no cover
    """Reject accidental execution outside the fixed milestone date."""

    if value != RUN_DATE:
        raise argparse.ArgumentTypeError(f"date must be {RUN_DATE}")
    return value


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover
    """Run the bounded canary or cold-check one task-owned candidate."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", type=_date_argument, default=RUN_DATE)
    parser.add_argument("--validate", type=Path)
    args = parser.parse_args(argv)
    progress("entrypoint", "start", date=args.date)
    if args.validate is not None:
        value = _load_json(args.validate)
        errors = validate_artifact(value)
        print(canonical_json({"errors": errors}), flush=True)
        return int(bool(errors))
    result = run_experiment(root=REPO_ROOT, run_date=args.date)
    print(
        canonical_json(
            {
                "artifact": str(REPO_ROOT / RESULT_PATH),
                "status": result["status"],
                "honest_verdict": result["honest_verdict"],
                "plan_transport_ready_score": result["plan_transport_ready_score"],
                "usable_plan_count": result["usable_plan_count"],
            }
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
