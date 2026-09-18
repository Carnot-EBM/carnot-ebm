"""Run four fresh bounded Qwen assignment calls on the qualified reducer.

The experiment measures transport only. It records satisfiability separately
and does not update a model, a generator weight, or a production default.

Spec refs: REQ-REPORT-7400 and SCENARIO-REPORT-7400-*.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from copy import deepcopy
from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any

from carnot import experiment_7347_v645_plan_canary as runtime
from carnot import experiment_7372_v647_qwen_canary as prior_canary
from carnot import experiment_7383_v648_canary_reducer as independent_reducer
from carnot.experiment_6212_three_family_gguf_runtime_recovery import (
    read_gguf_metadata,
    resolve_native_llama_server,
    snapshot_revision,
)
from carnot.experiment_7358_v646_validation_contract import (
    AffectedManifest,
    PlannedCommand,
    build_command_plan,
    reduce_affected_receipts,
    run_categorized_commands,
    validate_command_plan,
)
from carnot.inference.llama_server_supervisor import canonical_json
from carnot.inference.sota_models import cached_current_model
from carnot.reporting import current_work_receipt
from carnot.reporting.experiment_7303_validation_scope import (
    CommandSpec,
    REQUIRED_CHECK_NAMES,
)


JsonDict = dict[str, Any]
REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_DATE = "20260918"
MILESTONE = "2026.09.649"
PHASE = 2
EXPERIMENT_ID = "exp7400-v649-assignment-canary"
TASK_ID = "experiment_7400_v649_assignment_canary"
SCHEMA = "carnot.exp7400.v649.assignment_canary.v1"
MODEL_ID = "unsloth/Qwen3.8-27B-GGUF"
MODEL_SPECS = [MODEL_ID]
QUANTIZATION = "Q4_K_M"
MAX_GENERATED_TOKENS = 128
MODEL_LOAD_TIMEOUT_S = prior_canary.MODEL_LOAD_TIMEOUT_S
INFERENCE_WINDOW_TIMEOUT_S = prior_canary.INFERENCE_WINDOW_TIMEOUT_S
REQUEST_TIMEOUT_S = prior_canary.REQUEST_TIMEOUT_S
RANDOM_SEED = deepcopy(prior_canary.RANDOM_SEED)

MODULE_PATH = Path("python/carnot/experiment_7400_v649_assignment_canary.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_7400_v649_assignment_canary.py")
TEST_PATH = Path("tests/python/test_experiment_7400_v649_assignment_canary.py")
SPEC_PATH = Path("openspec/capabilities/research-reporting/spec.md")
RESULT_PATH = Path("results/experiment_7400_v649_assignment_canary.json")
RAW_DIR = Path("results/raw/experiment_7400_v649_assignment_canary")
PRODUCER_PATH = Path("results/experiment_7395_v649_receipt_protocol.json")
DEVELOPMENT_FORMULAS_PATH = Path(
    "results/raw/experiment_7371_v647_proof_boundary/development_formulas.json"
)
QUALIFIED_CODE_PATHS = (
    Path("python/carnot/experiment_7395_v649_receipt_protocol.py"),
    Path("python/carnot/reporting/current_work_receipt.py"),
    Path("python/carnot/experiment_7383_v648_canary_reducer.py"),
)
TERMINAL_CHECK_NAMES = (
    "declared_entrypoint_cold_replay",
    "independent_cold_reducer",
    "adversarial_verify",
    "verdict_row_consistency_strict",
)

VALIDATION_MANIFEST = AffectedManifest(
    experiment_id=EXPERIMENT_ID,
    test_paths=(TEST_PATH.as_posix(),),
    changed_modules=(MODULE_PATH.as_posix(),),
    static_paths=(WRAPPER_PATH.as_posix(),),
)

FIELD_PRINCIPLES: dict[str, str] = {
    "schema": "Use a versioned schema and ordinary identity fields; publish terminal state only after measured work and checks.",
    "run_date": "Use 20260918 and preserve actual UTC start and completion timestamps.",
    "preconditions_checked": "Authenticate exact inputs, eligibility, cache, runner, device, and entrypoint before dependent work.",
    "MODEL_SPECS": "Name the current Qwen GGUF for fresh LLM work; keep small Gibbs training separate.",
    "model_invoked": "Set true for an attempted current real model load or generation, including failure.",
    "invocation_counts": "Derive current owned load and generation dispositions from events, never historical evidence.",
    "inference_substrate": "Describe current work with a string and keep device and software facts in details.",
    "inference_substrate_class": "Use model_bounded_generation for these fixed short outputs and never pad runtime.",
    "execution_venue": "Use the closed host string instead of a hostname or device mapping.",
    "duration_s": "Measure current task time monotonically and record scientific and validation spans separately.",
    "phase_spans": "Retain actual phase boundaries, heartbeat checkpoints, and completed units.",
    "random_seed": "Freeze experiment and resampling seeds before observing outputs.",
    "reproducibility_checksum": "Bind current code, settings, inputs, protocol, events, and raw rows.",
    "source_artifact_hashes": "Hash exact producer, code, model, runner, sidecar, and raw evidence bytes.",
    "rows": "Keep every call, condition, metric contribution, cost, failure, and censoring disposition.",
    "sample_size_budget": "Predeclare planned, attempted, completed, censored, unstarted, limits, stop rule, and independent groups.",
    "acceptance_gate_results": "Keep validation, safety, completion, and efficacy gates separate with both operands.",
    "gate_check_summary": "Every blocked result names upstream path, check, field, expected value, and observed value.",
    "verifier_is_oracle": "Mark true because exact SAT evaluation defines correctness; transport remains a separate metric.",
    "honest_verdict": "Use complete_ for finished work and blocked_ for an unchanged missing prerequisite.",
    "verdict_class": "Use only positive, circular_positive, null, blocked, disqualified, or partial.",
    "flagged_adversarial": "Preserve critical verifier findings and never use invalid science for readiness.",
    "validation_receipts": "Retain exact argv, scoped environment, check name, exit, duration, and hashed log, including failures.",
    "repository_health": "Keep unrelated broad-suite observations separate while affected failures remain disqualifying.",
    "field_principles": "Explain fields without wrapping numeric gates or ordinary dictionaries.",
    "promotion_score": "Keep zero; the canary cannot roll out, update weights, publish, or submit externally.",
    "qwen_assignment_transport_ready_score": "Require at least three usable fresh proposals plus sound evidence and validation.",
    "transport_rows": "Retain all four request hashes, budgets, source literals, parse, fidelity, extendibility, and cost fields.",
    "owned_runtime_receipt": "Bind current PID, start tick, GGUF hash, CUDA offload, and resource ownership evidence.",
}
REQUIRED_FIELDS = frozenset(FIELD_PRINCIPLES)


def utc_now() -> str:  # pragma: no cover - real wall-clock boundary.
    """Return one actual UTC boundary while elapsed time stays monotonic."""

    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def progress(started: float, phase: str, event: str, **details: Any) -> None:  # pragma: no cover
    """Flush each boundary so model and subprocess work remains observable."""

    suffix = " ".join(f"{key}={value}" for key, value in sorted(details.items()))
    print(
        f"[exp7400] phase={phase} event={event} elapsed_s={time.monotonic() - started:.3f}"
        + (f" {suffix}" if suffix else ""),
        flush=True,
    )


def sha256_file(path: Path) -> str:
    """Hash exact bytes so an input or raw row cannot drift silently."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def canonical_hash(value: Any) -> str:
    """Hash stable JSON bytes for events and independently reduced evidence."""

    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    """Write one complete JSON value with fsync and a local atomic rename."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def load_object(path: Path) -> JsonDict:
    """Return one JSON object, or an empty object for absent or malformed bytes."""

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def artifact_checksum(artifact: Mapping[str, Any]) -> str:
    """Bind the complete artifact without recursively hashing its checksum."""

    value = deepcopy(dict(artifact))
    value.pop("reproducibility_checksum", None)
    return canonical_hash(value)


def compare(operator: str, observed: Any, expected: Any) -> bool:
    """Apply one explicit gate operator instead of guessing from prose."""

    if operator == "==":
        return observed == expected
    if operator == "in":
        return observed in expected
    if operator == ">=":
        return observed >= expected
    raise ValueError(f"unsupported_operator:{operator}")


def gate_row(
    check: str,
    upstream: str,
    artifact_field: str,
    operator: str,
    expected: Any,
    observed: Any,
    *,
    category: str = "precondition",
) -> JsonDict:
    """Keep the exact gate operands beside its independently computed result."""

    try:
        passed = compare(operator, observed, expected)
    except (TypeError, ValueError):
        passed = False
    return {
        "category": category,
        "check": check,
        "upstream": upstream,
        "artifact_field": artifact_field,
        "operator": operator,
        "expected_value": deepcopy(expected),
        "observed_value": deepcopy(observed),
        "passed": passed,
    }


def gate_check_summary(checks: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Name the first exact failure while retaining every failed check name."""

    failures = [dict(row) for row in checks if row.get("passed") is not True]
    first = failures[0] if failures else None
    return {
        "all_passed": not failures,
        "failed_count": len(failures),
        "failed_checks": [row.get("check") for row in failures],
        "upstream": first.get("upstream") if first else EXPERIMENT_ID,
        "path": first.get("upstream") if first else RESULT_PATH.as_posix(),
        "check": first.get("check") if first else "all_required_checks",
        "artifact_field": first.get("artifact_field") if first else "gate_check_summary",
        "operator": first.get("operator") if first else "==",
        "expected_value": deepcopy(first.get("expected_value")) if first else True,
        "observed_value": deepcopy(first.get("observed_value")) if first else True,
        "passed": not failures,
    }


def producer_gate_rows(producer: Mapping[str, Any], *, root: Path) -> list[JsonDict]:
    """Authenticate the eligible Exp7395 fields and its exact qualified code."""

    fields: tuple[tuple[str, str, Any, Any], ...] = (
        ("producer_experiment_id", "experiment_id", "==", "exp7395-receipt-protocol"),
        ("producer_milestone", "milestone", "==", MILESTONE),
        ("assignment_reducer_ready", "assignment_reducer_ready_score", "==", 1),
        (
            "producer_verdict_class",
            "verdict_class",
            "in",
            ["positive", "circular_positive", "null"],
        ),
        ("producer_adversarial_flag", "flagged_adversarial", "==", False),
    )
    rows = [
        gate_row(
            check,
            PRODUCER_PATH.as_posix(),
            field,
            operator,
            expected,
            producer.get(field),
        )
        for check, field, operator, expected in fields
    ]
    declared = dict(producer.get("source_artifact_hashes") or {})
    for relative in QUALIFIED_CODE_PATHS:
        path = root / relative
        observed = sha256_file(path) if path.is_file() else None
        rows.append(
            gate_row(
                f"qualified_code_hash:{relative.as_posix()}",
                PRODUCER_PATH.as_posix(),
                f"source_artifact_hashes.{relative.as_posix()}",
                "==",
                declared.get(relative.as_posix()),
                observed,
            )
        )
    return rows


def build_schedule(fixture: Mapping[str, Any]) -> list[JsonDict]:
    """Reuse the four frozen Exp7372 prompts without observing new outcomes."""

    return prior_canary.build_canary_schedule(fixture)


def collect_preconditions(root: Path) -> tuple[list[JsonDict], JsonDict]:
    """Check source identity and producer eligibility before any runtime work."""

    required = (
        Path("AGENTS.md"),
        Path("CLAUDE.md"),
        Path("CODEX.md"),
        Path("research-program.md"),
        Path("ops/exclusion_manifest.yaml"),
        Path("ops/e2e-test-plan.md"),
        Path("scripts/experiment_template.py"),
        Path("python/carnot/reporting/experiment_7303_validation_scope.py"),
        Path("python/carnot/experiment_7358_v646_validation_contract.py"),
        Path("python/carnot/experiment_7372_v647_qwen_canary.py"),
        Path("python/carnot/experiment_7383_v648_canary_reducer.py"),
        Path("python/carnot/experiment_7371_v647_proof_boundary.py"),
        Path("python/carnot/experiment_7209_v635_span_canary.py"),
        Path("python/carnot/inference/sota_models.py"),
        Path("data/v647_implication_stream_manifest.json"),
        Path("openspec/capabilities/constraint-verification/spec.md"),
        SPEC_PATH,
        MODULE_PATH,
        WRAPPER_PATH,
        TEST_PATH,
        PRODUCER_PATH,
        DEVELOPMENT_FORMULAS_PATH,
    )
    checks: list[JsonDict] = []
    for relative in required:
        path = root / relative
        observed = "readable_nonempty_bytes" if path.is_file() and path.stat().st_size else None
        checks.append(
            gate_row(
                f"source_bytes:{relative.as_posix()}",
                relative.as_posix(),
                "bytes",
                "==",
                "readable_nonempty_bytes",
                observed,
            )
        )
    spec = (root / SPEC_PATH).read_text(encoding="utf-8") if (root / SPEC_PATH).is_file() else ""
    checks.append(
        gate_row(
            "driving_requirement",
            SPEC_PATH.as_posix(),
            "REQ-*",
            "==",
            "REQ-REPORT-7400",
            "REQ-REPORT-7400" if "REQ-REPORT-7400" in spec else None,
        )
    )
    producer = load_object(root / PRODUCER_PATH)
    checks.extend(producer_gate_rows(producer, root=root))
    exclusion = (
        (root / "ops/exclusion_manifest.yaml").read_text(encoding="utf-8")
        if (root / "ops/exclusion_manifest.yaml").is_file()
        else ""
    )
    excluded = "experiment_id: 7400" in exclusion or EXPERIMENT_ID in exclusion
    checks.append(
        gate_row(
            "not_excluded",
            "ops/exclusion_manifest.yaml",
            "experiment_id",
            "==",
            False,
            excluded,
        )
    )
    fixture = load_object(root / DEVELOPMENT_FORMULAS_PATH)
    schedule: list[JsonDict] = []
    selection_error: str | None = None
    try:
        schedule = build_schedule(fixture)
    except (KeyError, TypeError, ValueError) as exc:
        selection_error = f"{type(exc).__name__}:{exc}"
    checks.append(
        gate_row(
            "four_frozen_development_prompts",
            DEVELOPMENT_FORMULAS_PATH.as_posix(),
            "development_formulas",
            "==",
            {"count": 4, "selection_error": None},
            {"count": len(schedule), "selection_error": selection_error},
        )
    )
    return checks, {
        "producer": producer,
        "producer_sha256": sha256_file(root / PRODUCER_PATH)
        if (root / PRODUCER_PATH).is_file()
        else None,
        "fixture": fixture,
        "schedule": schedule,
    }


def _runtime_preconditions(
    root: Path, context: JsonDict, started: float
) -> list[JsonDict]:  # pragma: no cover
    """Resolve cached GGUF, embedded metadata, native runner, and one RTX 3090."""

    checks = [
        gate_row(
            "force_live_mode",
            "process_environment",
            "CARNOT_FORCE_LIVE",
            "==",
            "1",
            os.environ.get("CARNOT_FORCE_LIVE"),
        )
    ]
    resolved = cached_current_model(gpu_index=0, preferred_quant=QUANTIZATION)
    model_path = Path(str(resolved.get("model_path"))) if resolved else None
    exists = bool(model_path and model_path.is_file())
    model_hash = runtime._content_addressed_hash(model_path) if exists and model_path else None
    metadata = read_gguf_metadata(model_path) if exists and model_path else {}
    tokenizer_ok, tokenizer_detail = runtime.cpu_embedded_tokenizer_check(
        str(model_path) if model_path else ""
    )
    model_block_count = 0
    if exists and model_path:
        from llama_cpp import Llama

        vocab = Llama(model_path=str(model_path), vocab_only=True, verbose=False)
        model_block_count = int(vocab.metadata.get("qwen35.block_count", 0) or 0)
        del vocab
    model_spec = {
        "hf_id": resolved.get("hf_id") if resolved else None,
        "quantization": QUANTIZATION,
        "path": str(model_path.resolve()) if exists and model_path else None,
        "revision": snapshot_revision(model_path) if exists and model_path else None,
        "bytes": model_path.stat().st_size if exists and model_path else None,
        "sha256": model_hash,
        "native_tokenizer": "embedded_gguf",
        "embedded_tokenizer_loadable": tokenizer_ok,
        "embedded_tokenizer_detail": tokenizer_detail,
        "native_chat_template": metadata.get("chat_template_present") is True,
        "chat_template_sha256": metadata.get("chat_template_sha256"),
        "model_block_count": model_block_count,
        "decoding": {
            "max_new_tokens": MAX_GENERATED_TOKENS,
            "temperature": 0.0,
            "top_k": 1,
            "top_p": 1.0,
            "seed": RANDOM_SEED["experiment"],
            "repair_attempts": 0,
            "retry_budget": 0,
        },
    }
    model_ok = bool(
        exists
        and resolved
        and resolved.get("hf_id") == MODEL_ID
        and model_path
        and QUANTIZATION.lower() in model_path.name.lower()
        and model_hash
        and tokenizer_ok
        and model_spec["native_chat_template"]
        and model_block_count > 0
    )
    checks.append(
        gate_row(
            "cached_current_qwen_gguf",
            "cached_current_model",
            "MODEL_SPECS",
            "==",
            {
                "hf_id": MODEL_ID,
                "exists": True,
                "tokenizer": True,
                "chat_template": True,
                "model_block_count_positive": True,
            },
            {
                "hf_id": model_spec["hf_id"],
                "exists": exists,
                "tokenizer": tokenizer_ok,
                "chat_template": model_spec["native_chat_template"],
                "model_block_count_positive": model_block_count > 0,
            },
        )
    )
    server = Path(resolve_native_llama_server())
    server_hash = sha256_file(server) if server.is_file() else None
    progress(started, "preconditions", "before_subprocess", operation="native_runner_capabilities")
    runner_rows = runtime.lease_preflight.collect_runner_capabilities(server)
    progress(started, "preconditions", "after_subprocess", operation="native_runner_capabilities")
    runner_errors = runtime.lease_preflight.runner_capability_errors(runner_rows)
    checks.append(
        gate_row(
            "native_cuda_runner",
            server.as_posix(),
            "runner_capabilities",
            "==",
            {"errors": [], "server_hash_present": True},
            {"errors": runner_errors, "server_hash_present": bool(server_hash)},
        )
    )
    progress(started, "preconditions", "before_subprocess", operation="gpu_inventory")
    process_rows, query_receipts = runtime.lease_preflight.collect_gpu_process_rows()
    lease_rows = runtime.lease_preflight.scan_lease_rows(
        runtime.lease_preflight.LEASE_RUNTIME_DIR, process_rows
    )
    classified = runtime.lease_preflight.classify_process_rows(
        process_rows, lease_rows, current_task_id=TASK_ID
    )
    decision = runtime.lease_preflight.readiness_decision(
        classified,
        lease_rows,
        [
            {
                "repository": MODEL_ID,
                "filename": model_path.name if model_path else None,
                "path": str(model_path) if model_path else None,
                "real_path": str(model_path.resolve()) if exists and model_path else None,
                "revision": model_spec["revision"],
                "bytes": model_spec["bytes"],
                "sha256": model_hash,
                "hash_source": "content_addressed_cache_target" if model_hash else None,
                "weights_opened": False,
                "valid": model_ok,
            }
        ],
        runner_rows,
    )
    available_all = list(decision.get("available_gpu_uuids", []))
    rtx3090 = {
        str(row.get("gpu_uuid"))
        for row in classified
        if row.get("gpu_name") == "NVIDIA GeForce RTX 3090"
    }
    available = [uuid for uuid in available_all if uuid in rtx3090]
    query_ok = all(row.get("returncode") == 0 for row in query_receipts)
    progress(started, "preconditions", "after_subprocess", operation="gpu_inventory")
    checks.append(
        gate_row(
            "one_owned_rtx3090_slot",
            "nvidia-smi_and_gpu_lease_journal",
            "owned_runtime_receipt",
            "==",
            {"query_ok": True, "minimum_rtx3090_slots": 1},
            {"query_ok": query_ok, "minimum_rtx3090_slots": len(available)},
        )
    )
    context.update(
        {
            "model_path": model_path,
            "model_spec": model_spec,
            "server_path": server,
            "server_sha256": server_hash,
            "runner_capabilities": runner_rows,
            "process_rows": classified,
            "lease_rows": lease_rows,
            "gpu_query_receipts": query_receipts,
            "available_gpu_uuids": available,
        }
    )
    return checks


class InvocationEventRecorder:  # pragma: no cover - records live native callbacks.
    """Translate native progress boundaries into the qualified current ledger."""

    def __init__(self, run_id: str, owner_pid: int, started: float) -> None:
        self.run_id = run_id
        self.owner_pid = owner_pid
        self.started = started
        self.events: list[JsonDict] = []

    def _append(self, call_id: str, operation: str, state: str) -> None:
        self.events.append(
            {
                "scope": "current",
                "transport": "owned_runtime",
                "run_id": self.run_id,
                "owner_pid": self.owner_pid,
                "call_id": call_id,
                "operation": operation,
                "state": state,
                "monotonic_ns": time.monotonic_ns(),
            }
        )

    def __call__(self, phase: str, event: str, **details: Any) -> None:
        progress(self.started, phase, event, **details)
        if (phase, event) == ("load", "before_model_load"):
            self._append("model-load", "model_load", "attempted")
        elif (phase, event) == ("load", "after_model_load") and details.get("healthy") is True:
            self._append("model-load", "model_load", "completed")
        elif (phase, event) == ("generation", "before_call"):
            self._append(f"generation-{int(details['completed'])}", "generation", "attempted")
        elif (phase, event) == ("generation", "after_call"):
            state = "completed" if details.get("terminal_state") == "response" else "failed"
            self._append(f"generation-{int(details['completed']) - 1}", "generation", state)

    def close(self, capture: Mapping[str, Any]) -> None:
        """Close an attempted load if the native runtime returned before health."""

        load = dict(capture.get("load_receipt") or {})
        load_events = [row for row in self.events if row["operation"] == "model_load"]
        if load.get("attempted") and len(load_events) == 1:
            self._append("model-load", "model_load", "failed")


def _capture_current(
    context: Mapping[str, Any], raw_dir: Path, started: float
) -> JsonDict:  # pragma: no cover
    """Reuse the shipped native capture with this task's lease and event owner."""

    recorder = InvocationEventRecorder(
        f"{EXPERIMENT_ID}:{os.getpid()}:{time.monotonic_ns()}", os.getpid(), started
    )
    old_task = prior_canary.TASK_ID
    old_progress = prior_canary.progress
    prior_canary.TASK_ID = TASK_ID
    prior_canary.progress = recorder
    try:
        capture = prior_canary._capture(context, raw_dir)
    finally:
        prior_canary.TASK_ID = old_task
        prior_canary.progress = old_progress
    recorder.close(capture)
    capture["current_run_id"] = recorder.run_id
    capture["current_owner_pid"] = recorder.owner_pid
    capture["current_invocation_events"] = recorder.events
    return capture


def reduce_current_events(
    events: Sequence[Mapping[str, Any]], *, run_id: str, owner_pid: int
) -> JsonDict:
    """Use the exact Exp7395-qualified reducer for current invocation counts."""

    counts, errors = current_work_receipt._reduce_events(events, run_id=run_id, owner_pid=owner_pid)
    if errors:
        raise ValueError("invalid_current_event_ledger:" + ",".join(errors))
    return {
        "model_invoked": bool(
            counts["model_loads_attempted"] + counts["generation_calls_attempted"]
        ),
        "invocation_counts": counts,
        "event_count": len(events),
        "event_sha256": canonical_hash(events),
    }


def build_offload_receipt(
    identity: Mapping[str, Any], provenance: Mapping[str, Any], model_block_count: int
) -> JsonDict:
    """Bind the all-layer request to owned CUDA residency and model structure."""

    command = list(identity.get("command") or [])
    try:
        option = command.index("--n-gpu-layers")
        requested = command[option + 1]
    except (ValueError, IndexError):
        requested = None
    confirmed = bool(
        requested == "all"
        and model_block_count > 0
        and identity.get("owned_by_task") is True
        and identity.get("cuda_provenance_ok") is True
        and provenance.get("provenance_ok") is True
        and provenance.get("cuda_log_evidence") is True
        and int(provenance.get("task_owned_vram_mb", 0) or 0) > 0
    )
    return {
        "requested_gpu_layers": requested,
        "actual_offloaded_layers": model_block_count if confirmed else 0,
        "total_model_layers": model_block_count,
        "all_layers_offloaded": confirmed,
        "evidence_method": "all-layer native command plus owned PID CUDA log and resident VRAM",
        "owned_vram_mb": int(provenance.get("task_owned_vram_mb", 0) or 0),
    }


def reduce_transport(
    rows: Sequence[Mapping[str, Any]], formulas: Sequence[Mapping[str, Any]]
) -> JsonDict:
    """Replay the independent reducer while treating one bad proposal as yield loss."""

    reduced = independent_reducer.reduce_assignment_receipts(rows, formulas)
    proposal_failures = ("variable_reference_invalid:", "response_fidelity_invalid:")
    integrity_errors = [
        error for error in reduced["errors"] if not str(error).startswith(proposal_failures)
    ]
    transport_rows: list[JsonDict] = []
    for row in reduced["rows"]:
        current = deepcopy(dict(row))
        current.update(
            {
                "historical": False,
                "current_model_invocation": True,
                "metric": "current_assignment_transport_usable",
            }
        )
        transport_rows.append(current)
    ready = int(
        len(transport_rows) == 4
        and reduced["usable_proposal_count"] >= 3
        and reduced["response_fidelity_count"] >= 3
        and reduced["complete_response_count"] == 4
        and reduced["runtime_identity_consistent"] is True
        and not integrity_errors
    )
    return {
        "transport_rows": transport_rows,
        "errors": deepcopy(reduced["errors"]),
        "integrity_errors": integrity_errors,
        "usable_proposal_count": int(reduced["usable_proposal_count"]),
        "source_literal_fidelity_count": int(reduced["response_fidelity_count"]),
        "sat_extendible_count": int(reduced["sat_extendible_count"]),
        "complete_response_count": int(reduced["complete_response_count"]),
        "runtime_identity_consistent": bool(reduced["runtime_identity_consistent"]),
        "qwen_assignment_transport_ready_score": ready,
    }


def build_validation_plan(root: Path, private_root: Path) -> list[CommandSpec]:
    """Freeze the exact Exp7358 affected list before subprocess execution."""

    return build_command_plan(root, VALIDATION_MANIFEST, private_root)


def validate_validation_plan(root: Path, commands: Sequence[CommandSpec]) -> list[str]:
    """Reject broad, duplicate, missing, or path-invalid affected commands."""

    return validate_command_plan(root, VALIDATION_MANIFEST, commands)


def receipts_pass(receipts: Sequence[Mapping[str, Any]], names: Sequence[str]) -> bool:
    """Require exactly one successful terminal receipt for every declared name."""

    return all(
        len(matched := [row for row in receipts if row.get("name") == name]) == 1
        and matched[0].get("passed") is True
        and matched[0].get("exit_code") == 0
        and matched[0].get("timed_out") is False
        for name in names
    )


def zero_counts() -> JsonDict:
    """Return explicit zero current load and generation dispositions."""

    return deepcopy(current_work_receipt.ZERO_INVOCATION_COUNTS)


def _acceptance_gates(
    *,
    preconditions_ok: bool,
    reduced: Mapping[str, Any],
    events_ok: bool,
    offload_ok: bool,
    affected_ok: bool,
    terminal_state: str,
    flagged: bool,
) -> list[JsonDict]:
    """Keep transport, evidence, validation, safety, and promotion gates separate."""

    values = (
        ("prerequisites", "completion", "==", True, preconditions_ok),
        ("four_terminal_calls", "completion", "==", 4, reduced.get("complete_response_count")),
        (
            "usable_source_faithful_assignments",
            "efficacy",
            ">=",
            3,
            reduced.get("usable_proposal_count"),
        ),
        ("qualified_event_reduction", "evidence", "==", True, events_ok),
        ("owned_all_layer_cuda_runtime", "evidence", "==", True, offload_ok),
        ("affected_validation", "validation", "==", True, affected_ok),
        (
            "terminal_readers",
            "validation",
            "in",
            ["not_yet_executed", "passed"],
            terminal_state,
        ),
        ("adversarial_clean", "safety", "==", False, flagged),
        ("promotion_disabled", "safety", "==", 0, 0),
    )
    return [
        gate_row(
            check,
            EXPERIMENT_ID,
            check,
            operator,
            expected,
            observed,
            category=category,
        )
        for check, category, operator, expected, observed in values
    ]


def _base_artifact() -> JsonDict:
    """Create one schema-complete no-run shape for blocked publication."""

    return {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "phase": PHASE,
        "status": "blocked_not_started",
        "run_date": RUN_DATE,
        "started_at_utc": None,
        "completed_at_utc": None,
        "preconditions_checked": [],
        "MODEL_SPECS": deepcopy(MODEL_SPECS),
        "model_settings": {},
        "model_invoked": False,
        "invocation_counts": zero_counts(),
        "current_invocation_events": [],
        "current_run_id": None,
        "current_owner_pid": None,
        "event_count": 0,
        "event_sha256": canonical_hash([]),
        "inference_mode": "live_gpu",
        "inference_substrate": "blocked_no_run",
        "inference_substrate_details": {},
        "inference_substrate_class": "blocked_no_run",
        "execution_venue": "host",
        "duration_s": 0.0,
        "scientific_duration_s": 0.0,
        "validation_duration_s": 0.0,
        "phase_spans": [],
        "random_seed": deepcopy(RANDOM_SEED),
        "reproducibility_checksum": None,
        "source_artifact_hashes": {},
        "historical_receipt_sidecars": [],
        "rows": [],
        "transport_rows": [],
        "raw_call_rows": [],
        "sample_size_budget": {
            "planned": 4,
            "attempted": 0,
            "completed": 0,
            "censored": 0,
            "unstarted": 4,
            "max_new_tokens_per_unit": MAX_GENERATED_TOKENS,
            "model_load_timeout_s": MODEL_LOAD_TIMEOUT_S,
            "request_timeout_s": REQUEST_TIMEOUT_S,
            "stop_rule": "attempt four frozen prompts once; no retries, repair, replacement, or padding",
            "effective_independent_group_count": 4,
        },
        "acceptance_gate_results": [],
        "gate_check_summary": {},
        "verifier_is_oracle": True,
        "honest_verdict": "blocked_not_started",
        "verdict_class": "blocked",
        "flagged_adversarial": False,
        "validation_receipts": [],
        "repository_health": {
            "status": "not_checked",
            "unrelated_broad_suite_observations": [],
            "affects_required_checks": False,
        },
        "field_principles": deepcopy(FIELD_PRINCIPLES),
        "promotion_score": 0,
        "qwen_assignment_transport_ready_score": 0,
        "observed_qwen_assignment_transport_ready_score": 0,
        "usable_proposal_count": 0,
        "source_literal_fidelity_count": 0,
        "sat_extendible_count": 0,
        "complete_response_count": 0,
        "owned_runtime_receipt": {},
        "numbered_e2e_applicable": False,
        "capability_e2e_checks": ["declared_entrypoint", "fresh_process_cold_replay"],
        "learning_claim": False,
        "production_defaults_changed": False,
        "generator_weights_changed": False,
        "research_conductor_changed": False,
    }


def build_blocked_artifact(
    checks: Sequence[Mapping[str, Any]],
    *,
    started_at: str | None = None,
    duration_s: float = 0.0,
) -> JsonDict:
    """Publish missing unchanged inputs as blocked with zero current attempts."""

    artifact = _base_artifact()
    summary = gate_check_summary(checks)
    artifact.update(
        {
            "status": "blocked_precondition_failed",
            "started_at_utc": started_at,
            "completed_at_utc": started_at,
            "preconditions_checked": [deepcopy(dict(row)) for row in checks],
            "duration_s": duration_s,
            "gate_check_summary": summary,
            "honest_verdict": f"blocked_{summary['check']}",
        }
    )
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return artifact


def build_artifact_for_test(
    *,
    rows: Sequence[Mapping[str, Any]],
    raw_rows: Sequence[Mapping[str, Any]],
    events: Sequence[Mapping[str, Any]],
    receipts: Sequence[Mapping[str, Any]],
    reduced: Mapping[str, Any],
) -> JsonDict:
    """Build a deterministic terminal fixture through the production reducers."""

    current = reduce_current_events(events, run_id="run-test", owner_pid=7400)
    offload = build_offload_receipt(
        {
            "owned_by_task": True,
            "cuda_provenance_ok": True,
            "command": ["llama-server", "--n-gpu-layers", "all"],
        },
        {"provenance_ok": True, "cuda_log_evidence": True, "task_owned_vram_mb": 16340},
        65,
    )
    gates = _acceptance_gates(
        preconditions_ok=True,
        reduced=reduced,
        events_ok=True,
        offload_ok=True,
        affected_ok=True,
        terminal_state="passed",
        flagged=False,
    )
    artifact = _base_artifact()
    artifact.update(
        {
            "status": "complete_assignment_transport_ready",
            "started_at_utc": "2026-09-18T00:00:00Z",
            "completed_at_utc": "2026-09-18T00:00:12Z",
            "preconditions_checked": [
                gate_row("fixture", "unit_test", "fixture", "==", True, True)
            ],
            "model_settings": {"max_new_tokens": 128, "seed": RANDOM_SEED["experiment"]},
            **current,
            "current_invocation_events": [deepcopy(dict(row)) for row in events],
            "current_run_id": "run-test",
            "current_owner_pid": 7400,
            "inference_substrate": "owned_native_cuda_llama_cpp_bounded_generation",
            "inference_substrate_details": {"gpu": "RTX 3090"},
            "inference_substrate_class": "model_bounded_generation",
            "duration_s": 12.0,
            "scientific_duration_s": 10.5,
            "validation_duration_s": 1.5,
            "phase_spans": [
                {
                    "phase": "generation",
                    "start_s": 0.0,
                    "end_s": 10.5,
                    "heartbeat_times_s": [],
                    "checkpoint_times_s": [10.5],
                }
            ],
            "rows": [deepcopy(dict(row)) for row in rows],
            "transport_rows": deepcopy(reduced["transport_rows"]),
            "raw_call_rows": [deepcopy(dict(row)) for row in raw_rows],
            "sample_size_budget": {
                **artifact["sample_size_budget"],
                "attempted": 4,
                "completed": 4,
                "censored": 0,
                "unstarted": 0,
            },
            "acceptance_gate_results": gates,
            "gate_check_summary": gate_check_summary(gates),
            "honest_verdict": "complete_circular_positive_assignment_transport_ready_4_of_4",
            "verdict_class": "circular_positive",
            "validation_receipts": [deepcopy(dict(row)) for row in receipts],
            "repository_health": {
                "status": "healthy",
                "unrelated_broad_suite_observations": [],
                "affects_required_checks": False,
            },
            "qwen_assignment_transport_ready_score": 1,
            "observed_qwen_assignment_transport_ready_score": 1,
            "usable_proposal_count": reduced["usable_proposal_count"],
            "source_literal_fidelity_count": reduced["source_literal_fidelity_count"],
            "sat_extendible_count": reduced["sat_extendible_count"],
            "complete_response_count": reduced["complete_response_count"],
            "owned_runtime_receipt": {"offload": offload},
        }
    )
    return artifact


def _raw_rows(artifact: Mapping[str, Any], *, root: Path) -> tuple[list[JsonDict], list[str]]:
    """Reload each exact raw row and report path, hash, or embedded-row drift."""

    manifests = artifact.get("raw_call_rows")
    embedded = artifact.get("rows")
    if not isinstance(manifests, list) or not isinstance(embedded, list) or not manifests:
        return [], ["raw_evidence_unavailable"]
    rows: list[JsonDict] = []
    errors: list[str] = []
    for index, manifest in enumerate(manifests):
        if not isinstance(manifest, Mapping):
            errors.append(f"raw_call_manifest_invalid:{index}")
            continue
        path = Path(str(manifest.get("path") or ""))
        resolved = path if path.is_absolute() else root / path
        if not resolved.is_file():
            errors.append(f"raw_call_path_missing:{index}")
            continue
        if sha256_file(resolved) != manifest.get("sha256"):
            errors.append(f"raw_call_hash_mismatch:{index}")
            continue
        row = load_object(resolved)
        rows.append(row)
        if index >= len(embedded) or row != embedded[index]:
            errors.append(f"raw_call_row_mismatch:{index}")
    return rows, errors


def independent_reduce_artifact(
    artifact: Mapping[str, Any], *, root: Path = REPO_ROOT
) -> list[str]:
    """Reload raw rows and recompute transport, current counts, and readiness."""

    rows, errors = _raw_rows(artifact, root=root)
    if errors:
        return errors
    fixture = load_object(root / DEVELOPMENT_FORMULAS_PATH)
    formulas = fixture.get("development_formulas")
    if not isinstance(formulas, list):
        return ["development_formulas_unavailable"]
    reduced = reduce_transport(rows, formulas)
    events = artifact.get("current_invocation_events")
    if not isinstance(events, list):
        return ["current_invocation_events_invalid"]
    try:
        current = reduce_current_events(
            events,
            run_id=str(artifact.get("current_run_id")),
            owner_pid=int(artifact.get("current_owner_pid")),
        )
    except (TypeError, ValueError) as exc:
        return [f"current_event_reduction_failed:{exc}"]
    comparisons = {
        "usable_proposal_count": reduced["usable_proposal_count"],
        "source_literal_fidelity_count": reduced["source_literal_fidelity_count"],
        "sat_extendible_count": reduced["sat_extendible_count"],
        "complete_response_count": reduced["complete_response_count"],
        "observed_qwen_assignment_transport_ready_score": reduced[
            "qwen_assignment_transport_ready_score"
        ],
        "invocation_counts": current["invocation_counts"],
        "model_invoked": current["model_invoked"],
        "event_count": current["event_count"],
        "event_sha256": current["event_sha256"],
        "transport_rows": reduced["transport_rows"],
    }
    for field, expected in comparisons.items():
        if artifact.get(field) != expected:
            errors.append(f"{field}_mismatch")
    expected_score = (
        0
        if artifact.get("verdict_class") in {"blocked", "disqualified"}
        else reduced["qwen_assignment_transport_ready_score"]
    )
    if artifact.get("qwen_assignment_transport_ready_score") != expected_score:
        errors.append("qwen_assignment_transport_ready_score_mismatch")
    return list(dict.fromkeys(errors))


def validate_artifact(
    value: object, *, root: Path = REPO_ROOT, require_terminal: bool = True
) -> list[str]:
    """Cold-check shape, exact reductions, closed declarations, and checksum."""

    if not isinstance(value, Mapping):
        return ["artifact_not_object"]
    artifact = dict(value)
    missing = sorted(REQUIRED_FIELDS - set(artifact))
    if missing:
        return [f"missing_required_field:{field}" for field in missing]
    errors: list[str] = []
    if (
        artifact.get("schema"),
        artifact.get("experiment_id"),
        artifact.get("milestone"),
        artifact.get("run_date"),
    ) != (SCHEMA, EXPERIMENT_ID, MILESTONE, RUN_DATE):
        errors.append("identity_invalid")
    if artifact.get("MODEL_SPECS") != MODEL_SPECS or artifact.get("promotion_score") != 0:
        errors.append("model_contract_invalid")
    if artifact.get("execution_venue") != "host":
        errors.append("execution_venue_invalid")
    if artifact.get("verdict_class") not in {
        "positive",
        "circular_positive",
        "null",
        "blocked",
        "disqualified",
        "partial",
    }:
        errors.append("verdict_class_invalid")
    if artifact.get("model_invoked"):
        if artifact.get("inference_substrate_class") != "model_bounded_generation":
            errors.append("substrate_class_invalid")
        if float(artifact.get("duration_s", 0.0) or 0.0) < 10.0:
            errors.append("bounded_generation_duration_floor")
        errors.extend(independent_reduce_artifact(artifact, root=root))
    elif artifact.get("verdict_class") == "blocked":
        if artifact.get("inference_substrate_class") != "blocked_no_run":
            errors.append("blocked_substrate_invalid")
    else:
        errors.append("completed_artifact_without_model_attempt")
    if require_terminal and not receipts_pass(
        list(artifact.get("validation_receipts") or []),
        (*REQUIRED_CHECK_NAMES, *TERMINAL_CHECK_NAMES),
    ):
        errors.append("required_validation_receipts_invalid")
    if artifact.get("verdict_class") in {"blocked", "disqualified"}:
        if artifact.get("qwen_assignment_transport_ready_score") != 0:
            errors.append("failed_readiness_invalid")
    elif artifact.get("verdict_class") == "circular_positive":
        if artifact.get("qwen_assignment_transport_ready_score") != 1:
            errors.append("positive_readiness_invalid")
    if artifact.get("reproducibility_checksum") != artifact_checksum(artifact):
        errors.append("reproducibility_checksum_mismatch")
    return list(dict.fromkeys(errors))


def _source_hashes(
    root: Path,
    context: Mapping[str, Any],
    raw_rows: Sequence[Mapping[str, Any]],
    historical_sidecar: Mapping[str, Any],
) -> JsonDict:  # pragma: no cover
    """Bind current sources, prerequisite bytes, model, runner, and raw rows."""

    paths = (
        Path("AGENTS.md"),
        Path("CLAUDE.md"),
        Path("CODEX.md"),
        Path("research-program.md"),
        Path("ops/exclusion_manifest.yaml"),
        Path("ops/e2e-test-plan.md"),
        SPEC_PATH,
        MODULE_PATH,
        WRAPPER_PATH,
        TEST_PATH,
        PRODUCER_PATH,
        DEVELOPMENT_FORMULAS_PATH,
        *QUALIFIED_CODE_PATHS,
        Path("python/carnot/experiment_7372_v647_qwen_canary.py"),
        Path("python/carnot/experiment_7358_v646_validation_contract.py"),
        Path("python/carnot/reporting/experiment_7303_validation_scope.py"),
        Path("scripts/adversarial_verify.py"),
        Path("scripts/verdict_row_consistency_lint.py"),
    )
    hashes = {path.as_posix(): sha256_file(root / path) for path in paths}
    model_spec = dict(context.get("model_spec") or {})
    if model_spec.get("path") and model_spec.get("sha256"):
        hashes[str(model_spec["path"])] = str(model_spec["sha256"])
    if context.get("server_path") and context.get("server_sha256"):
        hashes[str(context["server_path"])] = str(context["server_sha256"])
    for row in raw_rows:
        hashes[str(row["path"])] = str(row["sha256"])
    hashes[str(historical_sidecar["path"])] = str(historical_sidecar["sha256"])
    return hashes


def _span(
    spans: list[JsonDict],
    phase: str,
    phase_started: float,
    run_started: float,
    completed_units: int,
) -> None:  # pragma: no cover
    """Close one actual phase with its checkpoint and monotonic boundaries."""

    ended = time.monotonic()
    spans.append(
        {
            "phase": phase,
            "start_s": round(phase_started - run_started, 6),
            "end_s": round(ended - run_started, 6),
            "duration_s": round(ended - phase_started, 6),
            "completed_units": completed_units,
            "heartbeat_times_s": [],
            "checkpoint_times_s": [round(ended - run_started, 6)],
            "checkpoint_at_utc": utc_now(),
        }
    )


def _run_affected_validation(
    root: Path, raw_dir: Path, started: float
) -> tuple[list[JsonDict], JsonDict]:  # pragma: no cover
    """Execute the frozen Exp7358 plan through the streaming Exp7303 runner."""

    private = Path(tempfile.mkdtemp(prefix="exp7400-validation-", dir="/tmp"))
    commands = build_validation_plan(root, private)
    plan_errors = validate_validation_plan(root, commands)
    if plan_errors:
        return [], {"passed": False, "plan_errors": plan_errors}
    planned = [
        PlannedCommand(spec=command, category="required_affected_validation", required=True)
        for command in commands
    ]
    progress(started, "validation", "before_affected_commands", count=len(planned))
    receipts = run_categorized_commands(
        root, planned, log_dir=raw_dir / "validation/affected", heartbeat_s=60.0
    )
    progress(started, "validation", "after_affected_commands", count=len(receipts))
    reduced = reduce_affected_receipts(root, VALIDATION_MANIFEST, receipts)
    return receipts, {**reduced, "plan_errors": []}


def _terminal_commands(root: Path, candidate: Path) -> list[CommandSpec]:  # pragma: no cover
    """Build fresh-process replay and unchanged strict-reader commands."""

    python = str(root / ".venv/bin/python")
    reducer = (
        "import json,pathlib,sys;"
        "from carnot.experiment_7400_v649_assignment_canary import independent_reduce_artifact;"
        "value=json.loads(pathlib.Path(sys.argv[1]).read_text());"
        "errors=independent_reduce_artifact(value);"
        "print(json.dumps({'errors':errors},sort_keys=True),flush=True);"
        "raise SystemExit(bool(errors))"
    )
    return [
        CommandSpec(
            "declared_entrypoint_cold_replay",
            (
                python,
                "-u",
                WRAPPER_PATH.as_posix(),
                "--date",
                RUN_DATE,
                "--validate",
                str(candidate),
            ),
            "measured_candidate",
        ),
        CommandSpec(
            "independent_cold_reducer",
            (python, "-u", "-c", reducer, str(candidate)),
            "measured_candidate",
        ),
        CommandSpec(
            "adversarial_verify",
            (python, "-u", "scripts/adversarial_verify.py", str(candidate)),
            "measured_candidate",
        ),
        CommandSpec(
            "verdict_row_consistency_strict",
            (
                python,
                "-u",
                "scripts/verdict_row_consistency_lint.py",
                "--strict",
                str(candidate),
            ),
            "measured_candidate",
        ),
    ]


def run_experiment(
    *, root: Path = REPO_ROOT, run_date: str = RUN_DATE, output_path: Path | None = None
) -> JsonDict:  # pragma: no cover - live bounded orchestration.
    """Authenticate, generate four rows, validate, and publish atomically."""

    started = time.monotonic()
    started_at = utc_now()
    output = output_path or root / RESULT_PATH
    raw_dir = root / RAW_DIR
    raw_dir.mkdir(parents=True, exist_ok=True)
    spans: list[JsonDict] = []

    phase_started = time.monotonic()
    progress(started, "preconditions", "start")
    checks, context = collect_preconditions(root)
    checks.insert(
        0,
        gate_row("run_date", "execution_contract", "run_date", "==", RUN_DATE, run_date),
    )
    _span(spans, "preconditions_static", phase_started, started, len(checks))
    if not all(row["passed"] for row in checks):
        artifact = build_blocked_artifact(
            checks, started_at=started_at, duration_s=round(time.monotonic() - started, 6)
        )
        artifact["phase_spans"] = spans
        artifact["completed_at_utc"] = utc_now()
        artifact["reproducibility_checksum"] = artifact_checksum(artifact)
        atomic_json(output, artifact)
        return artifact

    phase_started = time.monotonic()
    progress(started, "preconditions", "runtime_start")
    runtime_checks = _runtime_preconditions(root, context, started)
    checks.extend(runtime_checks)
    progress(
        started,
        "preconditions",
        "runtime_complete",
        passed=all(row["passed"] for row in checks),
    )
    _span(spans, "preconditions_runtime", phase_started, started, len(runtime_checks))
    if not all(row["passed"] for row in checks):
        artifact = build_blocked_artifact(
            checks, started_at=started_at, duration_s=round(time.monotonic() - started, 6)
        )
        artifact["phase_spans"] = spans
        artifact["model_settings"] = deepcopy(context.get("model_spec") or {})
        artifact["completed_at_utc"] = utc_now()
        artifact["reproducibility_checksum"] = artifact_checksum(artifact)
        atomic_json(output, artifact)
        return artifact

    sidecar_path = raw_dir / "historical_exp7395_receipt.json"
    historical_sidecar = current_work_receipt.write_immutable_sidecar(
        sidecar_path,
        scope="historical_model_receipts",
        payload={
            "artifact_path": PRODUCER_PATH.as_posix(),
            "artifact_sha256": context["producer_sha256"],
            "original_verdict_class": context["producer"].get("verdict_class"),
            "original_flagged_adversarial": context["producer"].get("flagged_adversarial"),
            "assignment_reducer_ready_score": context["producer"].get(
                "assignment_reducer_ready_score"
            ),
            "authorizes_current_inference_counts": False,
        },
        root=root,
    )

    phase_started = time.monotonic()
    scientific_started = phase_started
    progress(started, "generation", "before_model_load_and_generation", planned_calls=4)
    capture = _capture_current(context, raw_dir / "owned_runtime", started)
    progress(
        started,
        "generation",
        "after_model_load_and_generation",
        completed_calls=len(capture.get("rows") or []),
    )
    _span(spans, "model_load_and_generation", phase_started, started, len(capture["rows"]))
    scientific_duration = time.monotonic() - scientific_started

    phase_started = time.monotonic()
    rows = [deepcopy(dict(row)) for row in capture["rows"]]
    raw_rows: list[JsonDict] = []
    for index, row in enumerate(rows):
        path = raw_dir / f"call_{index:02d}.json"
        atomic_json(path, row)
        raw_rows.append({"path": path.relative_to(root).as_posix(), "sha256": sha256_file(path)})
        progress(started, "reduction", "checkpoint", completed=index + 1, total=4)
    reduced = reduce_transport(rows, context["fixture"]["development_formulas"])
    current = reduce_current_events(
        capture["current_invocation_events"],
        run_id=capture["current_run_id"],
        owner_pid=capture["current_owner_pid"],
    )
    provenance = dict(capture.get("gpu_receipts") or {}).get("provenance") or {}
    offload = build_offload_receipt(
        dict(capture.get("runtime_identity") or {}),
        provenance,
        int(context["model_spec"]["model_block_count"]),
    )
    _span(spans, "reduction", phase_started, started, len(rows))

    validation_started = time.monotonic()
    affected_receipts, affected = _run_affected_validation(root, raw_dir, started)
    affected_ok = bool(affected.get("passed"))
    observed_ready = int(
        reduced["qwen_assignment_transport_ready_score"] == 1
        and offload["all_layers_offloaded"] is True
        and current["model_invoked"] is True
        and affected_ok
    )
    gates = _acceptance_gates(
        preconditions_ok=True,
        reduced=reduced,
        events_ok=True,
        offload_ok=offload["all_layers_offloaded"] is True,
        affected_ok=affected_ok,
        terminal_state="not_yet_executed",
        flagged=False,
    )
    artifact = _base_artifact()
    artifact.update(
        {
            "status": "complete_candidate_awaiting_terminal_readers",
            "started_at_utc": started_at,
            "completed_at_utc": utc_now(),
            "preconditions_checked": [deepcopy(dict(row)) for row in checks],
            "model_settings": deepcopy(context["model_spec"]),
            **current,
            "current_invocation_events": deepcopy(capture["current_invocation_events"]),
            "current_run_id": capture["current_run_id"],
            "current_owner_pid": capture["current_owner_pid"],
            "inference_substrate": "owned_native_cuda_llama_cpp_bounded_generation",
            "inference_substrate_details": {
                "model": MODEL_ID,
                "quantization": QUANTIZATION,
                "runner": str(context["server_path"]),
                "gpu_uuid": dict(capture.get("runtime_identity") or {}).get("gpu_uuid"),
                "gpu_name": "NVIDIA GeForce RTX 3090",
                "embedded_tokenizer": True,
                "embedded_chat_template": True,
            },
            "inference_substrate_class": "model_bounded_generation",
            "scientific_duration_s": round(scientific_duration, 6),
            "phase_spans": spans,
            "historical_receipt_sidecars": [historical_sidecar],
            "rows": rows,
            "transport_rows": reduced["transport_rows"],
            "raw_call_rows": raw_rows,
            "sample_size_budget": {
                **artifact["sample_size_budget"],
                "attempted": current["invocation_counts"]["generation_calls_attempted"],
                "completed": current["invocation_counts"]["generation_calls_completed"],
                "censored": sum(row.get("censored") is True for row in rows),
                "unstarted": 4 - current["invocation_counts"]["generation_calls_attempted"],
            },
            "acceptance_gate_results": gates,
            "gate_check_summary": gate_check_summary([*checks, *gates]),
            "honest_verdict": (
                f"complete_circular_positive_assignment_transport_ready_{reduced['usable_proposal_count']}_of_4"
                if observed_ready
                else f"complete_null_assignment_transport_low_yield_{reduced['usable_proposal_count']}_of_4"
            ),
            "verdict_class": "circular_positive" if observed_ready else "null",
            "validation_receipts": affected_receipts,
            "repository_health": {
                "status": "healthy" if affected_ok else "required_checks_failed",
                "unrelated_broad_suite_observations": [],
                "affects_required_checks": not affected_ok,
                "affected_reduction": affected,
            },
            "qwen_assignment_transport_ready_score": observed_ready,
            "observed_qwen_assignment_transport_ready_score": int(
                reduced["qwen_assignment_transport_ready_score"]
            ),
            "usable_proposal_count": reduced["usable_proposal_count"],
            "source_literal_fidelity_count": reduced["source_literal_fidelity_count"],
            "sat_extendible_count": reduced["sat_extendible_count"],
            "complete_response_count": reduced["complete_response_count"],
            "owned_runtime_receipt": {
                **deepcopy(dict(capture.get("runtime_identity") or {})),
                "model_load": deepcopy(capture.get("load_receipt") or {}),
                "gpu_provenance": deepcopy(provenance),
                "offload": offload,
                "cleanup": deepcopy(dict(capture.get("gpu_receipts") or {}).get("cleanup") or {}),
                "lease_release": deepcopy(
                    dict(capture.get("gpu_receipts") or {}).get("lease_release") or {}
                ),
            },
        }
    )
    artifact["source_artifact_hashes"] = _source_hashes(root, context, raw_rows, historical_sidecar)
    artifact["duration_s"] = round(time.monotonic() - started, 6)
    artifact["validation_duration_s"] = round(time.monotonic() - validation_started, 6)
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    candidate = raw_dir / "measured_terminal_candidate.json"
    atomic_json(candidate, artifact)

    progress(started, "validation", "before_terminal_commands", candidate=candidate)
    terminal_receipts = run_categorized_commands(
        root,
        [
            PlannedCommand(
                spec=spec,
                category="safety" if spec.name == "adversarial_verify" else "completion",
                required=True,
            )
            for spec in _terminal_commands(root, candidate)
        ],
        log_dir=raw_dir / "validation/terminal",
        heartbeat_s=60.0,
    )
    progress(started, "validation", "after_terminal_commands", count=len(terminal_receipts))
    terminal_ok = receipts_pass(terminal_receipts, TERMINAL_CHECK_NAMES)
    independent_ok = not independent_reduce_artifact(artifact, root=root)
    adversarial = next(
        (row for row in terminal_receipts if row.get("name") == "adversarial_verify"), {}
    )
    flagged = adversarial.get("passed") is not True
    _span(
        spans,
        "validation",
        validation_started,
        started,
        len(affected_receipts) + len(terminal_receipts),
    )
    final_ready = int(observed_ready and terminal_ok and independent_ok and not flagged)
    disqualified = not affected_ok or not terminal_ok or not independent_ok or flagged
    if disqualified:
        artifact.update(
            {
                "status": "complete_required_validation_failed",
                "verdict_class": "disqualified",
                "honest_verdict": "complete_disqualified_assignment_canary_required_check_failed",
                "qwen_assignment_transport_ready_score": 0,
            }
        )
    elif final_ready:
        artifact["status"] = "complete_assignment_transport_ready"
        artifact["verdict_class"] = "circular_positive"
    else:
        artifact["status"] = "complete_assignment_transport_null"
        artifact["verdict_class"] = "null"
    artifact["flagged_adversarial"] = flagged
    artifact["qwen_assignment_transport_ready_score"] = final_ready
    artifact["validation_receipts"] = [*affected_receipts, *terminal_receipts]
    artifact["acceptance_gate_results"] = _acceptance_gates(
        preconditions_ok=True,
        reduced=reduced,
        events_ok=independent_ok,
        offload_ok=offload["all_layers_offloaded"] is True,
        affected_ok=affected_ok,
        terminal_state="passed" if terminal_ok else "failed",
        flagged=flagged,
    )
    artifact["gate_check_summary"] = gate_check_summary(
        [*checks, *artifact["acceptance_gate_results"]]
    )
    artifact["phase_spans"] = spans
    artifact["completed_at_utc"] = utc_now()
    artifact["duration_s"] = round(time.monotonic() - started, 6)
    artifact["validation_duration_s"] = round(time.monotonic() - validation_started, 6)
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    errors = validate_artifact(artifact, root=root, require_terminal=True)
    if errors:
        artifact.update(
            {
                "status": "complete_internal_validation_failed",
                "verdict_class": "disqualified",
                "honest_verdict": "complete_disqualified_internal_artifact_validation_failed",
                "flagged_adversarial": True,
                "qwen_assignment_transport_ready_score": 0,
                "internal_validation_errors": errors,
            }
        )
        artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    progress(started, "write", "before_atomic_publish", artifact=output)
    atomic_json(output, artifact)
    progress(
        started,
        "write",
        "after_atomic_publish",
        artifact=output,
        verdict=artifact["honest_verdict"],
    )
    return artifact


def _date_argument(value: str) -> str:
    """Reject execution outside the fixed V649 date."""

    if value != RUN_DATE:
        raise ValueError(f"date must be {RUN_DATE}")
    return value


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - thin CLI.
    """Run live capture or cold-validate one measured candidate."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", type=_date_argument, default=RUN_DATE)
    parser.add_argument("--validate", type=Path)
    args = parser.parse_args(argv)
    started = time.monotonic()
    progress(started, "entrypoint", "start", date=args.date)
    if args.validate is not None:
        errors = validate_artifact(
            load_object(args.validate), root=REPO_ROOT, require_terminal=False
        )
        print(canonical_json({"errors": errors}), flush=True)
        return int(bool(errors))
    result = run_experiment(root=REPO_ROOT, run_date=args.date)
    print(
        canonical_json(
            {
                "artifact": str(REPO_ROOT / RESULT_PATH),
                "status": result["status"],
                "honest_verdict": result["honest_verdict"],
                "qwen_assignment_transport_ready_score": result[
                    "qwen_assignment_transport_ready_score"
                ],
                "usable_proposal_count": result["usable_proposal_count"],
            }
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
