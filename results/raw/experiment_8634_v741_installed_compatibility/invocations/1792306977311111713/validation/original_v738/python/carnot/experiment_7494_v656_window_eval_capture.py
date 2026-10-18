"""Capture sealed V656 held-out whole-response and focus-window logits.

The capture changes only the sealed granularity. It keeps every source byte,
model outcome, and failure recheckable while test and online labels stay closed.

Spec refs: REQ-VERIFY-7494 and SCENARIO-VERIFY-7494-*.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any

from carnot import experiment_7462_v654_option_protocol as option_protocol
from carnot import experiment_7463_v654_semif_e0_logprob_parity as parity
from carnot import experiment_7491_v656_window_protocol as window_protocol
from carnot import experiment_7492_v656_window_pilot as window_pilot
from carnot import experiment_7493_v656_window_fit_capture as fit_capture
from carnot.experiment_7358_v646_validation_contract import AffectedManifest
from carnot.inference.sota_models import cached_current_model, current_model
from carnot.reporting import experiment_7303_validation_scope as validation_scope
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file


JsonDict = dict[str, Any]
RUN_DATE = "20260921"
MILESTONE = "2026.09.656"
EXPERIMENT_ID = "exp7494-v656-window-eval-capture"
SCHEMA = "carnot.exp7494.v656.window_eval_capture.v1"
MODEL_HF_ID = "unsloth/Qwen3.8-27B-GGUF"
MODEL_SPECS = [MODEL_HF_ID]
model_specs = [MODEL_HF_ID]
INFERENCE_SUBSTRATE = "live_native_llama_cpp_lossless_window_evaluation_raw_logit_capture"
INFERENCE_SUBSTRATE_CLASS = "model_load_no_generation"
EXECUTION_VENUE = "host"
FORWARD_BUDGET = 2_192
ELIGIBLE_FORWARD_BUDGET = 2_152
FORWARD_CEILING = 2_240
MAX_LIVE_SECONDS = 3_300.0
TOKEN_CEILING = 2_048
MAX_ARTIFACT_BYTES = fit_capture.MAX_ARTIFACT_BYTES
ROLE_MINIMUMS = {"test": 100, "online": 120}
ROLE_PLANNED = {"test": 120, "online": 160}
OPTION_IDS = option_protocol.OPTION_IDS
FORBIDDEN_CAPTURE_FIELDS = fit_capture.FORBIDDEN_CAPTURE_FIELDS | {"outcome"}

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_PATH = Path("results/experiment_7494_v656_window_eval_capture.json")
RAW_DIR = Path("results/raw/experiment_7494_v656_window_eval_capture")
MODULE_PATH = Path("python/carnot/experiment_7494_v656_window_eval_capture.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_7494_v656_window_eval_capture.py")
TEST_PATH = Path("tests/python/test_experiment_7494_v656_window_eval_capture.py")
SPEC_PATH = REPO_ROOT / "openspec/capabilities/verification/spec.md"
PROTOCOL_PATH = REPO_ROOT / "results/experiment_7491_v656_window_protocol.json"
PILOT_PATH = REPO_ROOT / "results/experiment_7492_v656_window_pilot.json"
PROTOCOL_RAW_DIR = Path("results/raw/experiment_7491_v656_window_protocol")

VALIDATION_MANIFEST = AffectedManifest(
    experiment_id=EXPERIMENT_ID,
    test_paths=(TEST_PATH.as_posix(),),
    changed_modules=(MODULE_PATH.as_posix(),),
    static_paths=(WRAPPER_PATH.as_posix(),),
)
AFFECTED_CHECK_NAMES = validation_scope.REQUIRED_CHECK_NAMES
TERMINAL_CHECK_NAMES = fit_capture.TERMINAL_CHECK_NAMES

WindowCaptureError = fit_capture.WindowCaptureError
_FIT_REDUCE_CAPTURE = fit_capture.reduce_capture
_FIT_CALL_WITH_HEARTBEATS = fit_capture._call_with_heartbeats
reduce_upstream_gates = fit_capture.reduce_upstream_gates
write_raw_shards = fit_capture.write_raw_shards
reload_raw_shards = fit_capture.reload_raw_shards
fixture_invocation_events = fit_capture.fixture_invocation_events
reduce_invocation_events = fit_capture.reduce_invocation_events
invocation_counts_balanced = fit_capture.invocation_counts_balanced
build_acceptance_gates = fit_capture.build_acceptance_gates
gate_check_summary = fit_capture.gate_check_summary
artifact_checksum = fit_capture.artifact_checksum
validation_names_passed = fit_capture.validation_names_passed


def _window_key(row: Mapping[str, Any]) -> tuple[str, int]:
    return str(row["group_id"]), int(row["window_index"])


def build_capture_plan(
    predictors: Sequence[Mapping[str, Any]],
    group_rows: Sequence[Mapping[str, Any]],
    window_rows: Sequence[Mapping[str, Any]],
    request_rows: Sequence[Mapping[str, Any]],
) -> list[JsonDict]:
    """Join only the sealed label-free test and online requests in stored order."""

    evaluation_groups = [row for row in group_rows if row.get("role") in ROLE_PLANNED]
    group_counts = {
        role: sum(row.get("role") == role for row in evaluation_groups) for role in ROLE_PLANNED
    }
    if group_counts != ROLE_PLANNED:
        raise WindowCaptureError("evaluation_role_counts_invalid")
    evaluation_requests = [row for row in request_rows if row.get("role") in ROLE_PLANNED]
    if len(evaluation_requests) != FORWARD_BUDGET or len(evaluation_requests) > FORWARD_CEILING:
        raise WindowCaptureError("evaluation_request_count_invalid")

    predictor_by_group = {str(row["group_id"]): row for row in predictors}
    group_by_id = {str(row["group_id"]): row for row in evaluation_groups}
    windows = {_window_key(row): row for row in window_rows}
    plan: list[JsonDict] = []
    for request in evaluation_requests:
        group_id = str(request["group_id"])
        source_group_id = str(request["source_group_id"])
        group = group_by_id.get(group_id)
        predictor = predictor_by_group.get(group_id)
        source_predictor = predictor_by_group.get(source_group_id)
        if group is None or predictor is None or source_predictor is None:
            raise WindowCaptureError(f"sealed_group_missing:{group_id}:{source_group_id}")
        if request.get("role") != group.get("role"):
            raise WindowCaptureError(f"sealed_role_mismatch:{request.get('request_id')}")
        window_index = request.get("window_index")
        window = None if window_index is None else windows.get((group_id, int(window_index)))
        source = str(source_predictor["source_text"])
        response = str(predictor["response_text"])
        prompt, marked_response = fit_capture._build_prompt(
            request,
            source=source,
            response=response,
            window=window,
        )
        prompt_hash = window_protocol.sha256_text(prompt)
        if prompt_hash != request.get("prompt_sha256"):
            raise WindowCaptureError(f"sealed_prompt_hash_mismatch:{request.get('request_id')}")
        eligible = request.get("eligible") is True
        token_count = int(request["prompt_token_count"])
        if eligible != (token_count <= TOKEN_CEILING):
            raise WindowCaptureError(f"sealed_eligibility_mismatch:{request.get('request_id')}")
        row = {
            "call_id": str(request["request_id"]),
            "request_id": str(request["request_id"]),
            "group_id": group_id,
            "source_group_id": source_group_id,
            "role": str(request["role"]),
            "arm": str(request["arm"]),
            "window_index": None if window_index is None else int(window_index),
            "option_order": [str(item) for item in request["option_order"]],
            "prompt": prompt,
            "prompt_sha256": prompt_hash,
            "sealed_prompt_sha256": str(request["prompt_sha256"]),
            "sealed_prompt_token_count": token_count,
            "prompt_token_count": token_count,
            "source_text": source,
            "response_text": response,
            "marked_response": marked_response,
            "source_sha256": window_protocol.sha256_text(source),
            "response_sha256": window_protocol.sha256_text(response),
            "source_version": {
                "corpus": source_predictor.get("corpus"),
                "release_revision": source_predictor.get("release_revision"),
                "license": source_predictor.get("license"),
            },
            "byte_start": None if window is None else int(window["byte_start"]),
            "byte_end": None if window is None else int(window["byte_end"]),
            "sentence_count": None if window is None else int(window["sentence_count"]),
            "sentence_version": None if window is None else window["sentence_version"],
            "window_version": None if window is None else window["window_version"],
            "window_sha256": None if window is None else window["window_sha256"],
            "eligible": eligible,
            "disposition": "unstarted" if eligible else "excluded",
            "attempted": False,
            "error": None if eligible else "complete_prompt_over_2048_tokens",
            "gold_label": None,
            "archived_pilot_reuse": False,
        }
        if set(row) & FORBIDDEN_CAPTURE_FIELDS:
            raise WindowCaptureError(f"forbidden_capture_field:{request.get('request_id')}")
        plan.append(row)
    if sum(row["eligible"] is True for row in plan) != ELIGIBLE_FORWARD_BUDGET:
        raise WindowCaptureError("eligible_forward_count_invalid")
    return plan


def _translate_roles(
    rows: Sequence[Mapping[str, Any]], mapping: Mapping[str, str]
) -> list[JsonDict]:
    translated: list[JsonDict] = []
    for source in rows:
        row = deepcopy(dict(source))
        role = str(row.get("role"))
        if role in mapping:
            row["role"] = mapping[role]
        translated.append(row)
    return translated


def controlled_plan_fixture() -> list[JsonDict]:
    """Reuse the qualified fixture with held-out role names."""

    return _translate_roles(
        fit_capture.controlled_plan_fixture(),
        {"training": "test", "calibration_tuning": "online"},
    )


def controlled_call_fixture(plan: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Create deterministic finite native rows through the qualified helper."""

    return fit_capture.controlled_call_fixture(plan)


def reduce_capture(
    plan: Sequence[Mapping[str, Any]],
    observed_rows: Sequence[Mapping[str, Any]],
    *,
    minimums: Mapping[str, int] = ROLE_MINIMUMS,
) -> JsonDict:
    """Adapt held-out roles to the qualified lossless capture reducer."""

    to_fit = {"test": "training", "online": "calibration_tuning"}
    from_fit = {value: key for key, value in to_fit.items()}
    reduction = _FIT_REDUCE_CAPTURE(
        _translate_roles(plan, to_fit),
        _translate_roles(observed_rows, to_fit),
        minimums={to_fit[role]: int(value) for role, value in minimums.items()},
    )
    reduction["role_counts"] = {
        from_fit[role]: deepcopy(value) for role, value in reduction["role_counts"].items()
    }
    reduction["group_rows"] = _translate_roles(reduction["group_rows"], from_fit)
    reduction["reconciled_rows"] = _translate_roles(reduction["reconciled_rows"], from_fit)
    return reduction


def checkpoint_binding(
    plan: Sequence[Mapping[str, Any]],
    *,
    model_sha256: str,
    tokenizer_identity: Mapping[str, Any],
    request_manifest_sha256: str,
) -> JsonDict:
    """Bind resume to exact held-out requests, roles, model, and tokenizer."""

    stable_plan = [{key: value for key, value in row.items() if key != "prompt"} for row in plan]
    return {
        "schema": SCHEMA,
        "schedule_sha256": canonical_hash(stable_plan),
        "model_sha256": model_sha256,
        "tokenizer_identity_sha256": canonical_hash(tokenizer_identity),
        "request_manifest_sha256": request_manifest_sha256,
        "role_counts": deepcopy(ROLE_PLANNED),
    }


def _load_json_object(path: Path) -> JsonDict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(value) if isinstance(value, Mapping) else {}


def write_group_checkpoint(
    root: Path,
    *,
    binding: Mapping[str, Any],
    group_id: str,
    rows: Sequence[Mapping[str, Any]],
) -> None:
    """Write one evaluation group and update its hash-bound index."""

    root.mkdir(parents=True, exist_ok=True)
    index_path = root / "checkpoint-index.json"
    index = _load_json_object(index_path) if index_path.is_file() else {}
    if index and index.get("binding") != dict(binding):
        raise WindowCaptureError("checkpoint_binding_mismatch")
    digest = hashlib.sha256(group_id.encode()).hexdigest()[:24]
    group_path = root / f"group-{digest}.json"
    values = [deepcopy(dict(row)) for row in rows]
    atomic_json(
        group_path,
        {
            "schema": "carnot.exp7494.group_checkpoint.v1",
            "binding": dict(binding),
            "group_id": group_id,
            "rows": values,
            "rows_sha256": canonical_hash(values),
        },
    )
    groups = dict(index.get("groups") or {})
    groups[group_id] = {
        "path": group_path.name,
        "sha256": sha256_file(group_path),
        "rows": len(values),
    }
    atomic_json(
        index_path,
        {
            "schema": "carnot.exp7494.checkpoint_index.v1",
            "binding": dict(binding),
            "groups": groups,
        },
    )


def load_checkpoint_rows(root: Path, *, expected_binding: Mapping[str, Any]) -> list[JsonDict]:
    """Load only checkpoint groups whose binding and bytes still match."""

    index_path = root / "checkpoint-index.json"
    if not index_path.is_file():
        raise WindowCaptureError("checkpoint_index_missing")
    index = _load_json_object(index_path)
    if index.get("binding") != dict(expected_binding):
        raise WindowCaptureError("checkpoint_binding_mismatch")
    groups = index.get("groups")
    if not isinstance(groups, Mapping):
        raise WindowCaptureError("checkpoint_groups_invalid")
    rows: list[JsonDict] = []
    for group_id, receipt in groups.items():
        if not isinstance(receipt, Mapping):
            raise WindowCaptureError("checkpoint_group_receipt_invalid")
        path = root / str(receipt.get("path"))
        if not path.is_file() or sha256_file(path) != receipt.get("sha256"):
            raise WindowCaptureError("checkpoint_group_hash_mismatch")
        value = _load_json_object(path)
        values = value.get("rows")
        if (
            value.get("binding") != dict(expected_binding)
            or value.get("group_id") != group_id
            or not isinstance(values, list)
            or value.get("rows_sha256") != canonical_hash(values)
            or len(values) != receipt.get("rows")
        ):
            raise WindowCaptureError("checkpoint_group_invalid")
        rows.extend(dict(row) for row in values if isinstance(row, Mapping))
    return rows


_FIELD_PRINCIPLES = {
    "schema": "Versioned schema, exact experiment_id, milestone and terminal status prevent reader drift.",
    "run_date": "The fixed 20260921 date and measured clocks prevent evidence from moving between runs.",
    "preconditions_checked": "Exact paths, observed values, ownership and input validity prevent guessed prerequisites.",
    "MODEL_SPECS": "The Qwen model declaration plus resolved identity prevent a small substitute from becoming headline evidence.",
    "model_specs": "The lowercase model declaration prevents schema-reader model identity drift.",
    "model_invoked": "The current invocation flag prevents archived events from becoming live work.",
    "invocation_counts": "Balanced attempts and outcomes prevent failed or in-flight model work from disappearing.",
    "inference_substrate": "The native readout name prevents aggregation or generation from being implied.",
    "inference_substrate_class": "The model-load/no-generation class prevents padded durations and generated-token claims.",
    "execution_venue": "Host CPU and owned CUDA identities prevent archived board evidence from replacing current execution.",
    "duration_s": "Measured elapsed time and components prevent a synthetic duration floor.",
    "phase_spans": "Flushed boundaries and checkpoints prevent unfinished work or stalls from disappearing.",
    "random_seed": "Frozen role, optimizer, audit, order and interval choices prevent outcome-guided retries.",
    "reproducibility_checksum": "The checksum binds code, model, prompts, roles, shards and scope to prevent silent drift.",
    "source_artifact_hashes": "Original upstream bytes, verdicts and flags prevent evidence laundering.",
    "rows": "Per-group failures and censoring prevent difficult independent units from disappearing.",
    "sample_size_budget": "Separate planned and terminal call/group counts prevent silent roster shrinkage.",
    "acceptance_gate_results": "Typed validity, readiness and benefit operands prevent one favorable metric bypassing another.",
    "gate_check_summary": "Exact failed checks and paths prevent a blocked operand from being hidden.",
    "honest_verdict": "A complete terminal finding prevents finished capture work from looking retryable or positive.",
    "verdict_class": "The closed class prevents a null, blocked, disqualified, or partial result from changing meaning.",
    "verifier_is_oracle": "The false oracle flag prevents analytic fixtures from becoming positive evidence.",
    "flagged_adversarial": "Retained reader flags prevent deletion of an adverse finding from opening a gate.",
    "validation_receipts": "Exact commands, scopes, exits and log hashes prevent validation laundering.",
    "field_principles": "A failure-prevention purpose for every field prevents unexplained evidence from becoming authoritative.",
    "window_evaluation_ready_score": "Exact valid completion and role support prevent efficacy from defining capture readiness.",
    "capture_complete_score": "Explicit dispositions for every manifest cell prevent missing values from becoming zeros.",
    "role_support_score": "Eligible held-out group minima prevent windows, calls or seeds from inflating support.",
    "raw_logit_shards": "Hash-bound per-call scores prevent summary-only native evidence claims.",
    "role_counts": "Distinct planned and actual groups prevent windows or seeds from replacing independent units.",
}


def _field_principles(keys: Sequence[str]) -> dict[str, str]:
    return {
        key: _FIELD_PRINCIPLES.get(
            key,
            f"The {key} field preserves measured evidence to prevent silent record drift.",
        )
        for key in keys
    }


def _public_reduction(reduction: Mapping[str, Any]) -> JsonDict:
    return {
        key: deepcopy(value)
        for key, value in reduction.items()
        if key not in {"group_rows", "reconciled_rows"}
    }


def _build_artifact(
    *,
    plan: Sequence[Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
    raw_logit_root: str,
    raw_logit_shards: Sequence[Mapping[str, Any]],
    events: Sequence[Mapping[str, Any]],
    preconditions: Sequence[Mapping[str, Any]],
    source_hashes: Sequence[Mapping[str, Any]],
    model_identity: Mapping[str, Any],
    device_identity: Mapping[str, Any],
    validation_receipts: Sequence[Mapping[str, Any]],
    validation_passed: bool,
    phase_spans: Sequence[Mapping[str, Any]],
    duration_breakdown: Mapping[str, float],
    started_at: str,
    ended_at: str,
    started_ns: int,
    ended_ns: int,
    role_minimums: Mapping[str, int] = ROLE_MINIMUMS,
) -> JsonDict:
    reduction = reduce_capture(plan, rows, minimums=role_minimums)
    counts = reduce_invocation_events(events)
    call_budget = reduction["sample_size_budget"]["calls"]
    identity_valid = (
        all(row.get("passed") is True for row in preconditions)
        and reduction["source_transport_identity_valid"] is True
        and invocation_counts_balanced(counts)
        and counts["model_loads"]["attempted"] == 1
        and counts["model_loads"]["completed"] == 1
        and counts["forward_calls"]["attempted"] == call_budget["attempted"]
        and counts["forward_calls"]["completed"] == call_budget["complete"]
        and counts["forward_calls"]["failed"] == call_budget["failed"]
        and model_identity.get("identity_authenticated") is True
    )
    gates = build_acceptance_gates(
        validity_passed=identity_valid,
        capture_complete=reduction["capture_complete_score"] == 1,
        role_support=reduction["role_support_score"] == 1,
        eligible_calls_complete=reduction["all_eligible_calls_valid"] is True,
        validation_passed=validation_passed,
    )
    ready = int(all(row["passed"] for row in gates))
    if not identity_valid or not validation_passed:
        verdict = "complete_disqualified_window_evaluation_capture_invalid_evidence"
        verdict_class = "disqualified"
    elif not reduction["all_eligible_calls_valid"]:
        verdict = "complete_partial_window_evaluation_capture_incomplete"
        verdict_class = "partial"
    elif ready:
        verdict = "complete_null_window_evaluation_capture_ready_predictive_benefit_not_tested"
        verdict_class = "null"
    else:
        verdict = "complete_disqualified_window_evaluation_capture_not_ready"
        verdict_class = "disqualified"

    resolved_model = {
        "model_id": MODEL_HF_ID,
        "resolved_file": model_identity.get("model_path"),
        "sha256": model_identity.get("model_sha256"),
        "quantization": model_identity.get("quantization"),
        "runtime": model_identity.get("llama_cpp_build"),
    }
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "terminal_status": "complete",
        "run_date": RUN_DATE,
        "started_at_utc": started_at,
        "ended_at_utc": ended_at,
        "process_identity": {
            "pid": int(model_identity.get("owned_pid", os.getpid())),
            "started_monotonic_ns": started_ns,
            "ended_monotonic_ns": ended_ns,
            "clock": "time.monotonic_ns",
        },
        "preconditions_checked": [deepcopy(dict(row)) for row in preconditions],
        "MODEL_SPECS": deepcopy(MODEL_SPECS),
        "model_specs": deepcopy(model_specs),
        "resolved_model_specs": [resolved_model],
        "model_invoked": counts["model_loads"]["attempted"] > 0,
        "invocation_counts": counts,
        "current_invocation_events": [deepcopy(dict(row)) for row in events],
        "inference_substrate": INFERENCE_SUBSTRATE,
        "inference_substrate_class": INFERENCE_SUBSTRATE_CLASS,
        "execution_venue": EXECUTION_VENUE,
        "device_identity": deepcopy(dict(device_identity)),
        "model_identity": deepcopy(dict(model_identity)),
        "duration_s": (ended_ns - started_ns) / 1_000_000_000,
        "duration_breakdown_s": deepcopy(dict(duration_breakdown)),
        "phase_spans": [deepcopy(dict(row)) for row in phase_spans],
        "random_seed": {
            "role_seed": 656001,
            "optimizer_seed": None,
            "audit_seed": 656094,
            "order_seed": "both_frozen_orders",
            "interval_seed": None,
            "deterministic_null_seed_reason": (
                "Capture performs no fit, interval, label access, or outcome-guided retry."
            ),
        },
        "source_artifact_hashes": [deepcopy(dict(row)) for row in source_hashes],
        "request_manifest_identity": {
            "planned_calls": len(plan),
            "eligible_calls": sum(row.get("eligible") is True for row in plan),
            "excluded_calls": sum(row.get("eligible") is False for row in plan),
            "schedule_sha256": canonical_hash(
                [{key: value for key, value in row.items() if key != "prompt"} for row in plan]
            ),
            "protocol_request_shard_sha256": next(
                (
                    row.get("sha256")
                    for row in source_hashes
                    if str(row.get("path")).endswith("requests.jsonl")
                ),
                None,
            ),
            "forward_ceiling": FORWARD_CEILING,
            "live_work_cap_s": MAX_LIVE_SECONDS,
            "pilot_rows_reused_as_current_inference": 0,
        },
        "rows": deepcopy(reduction["group_rows"]),
        "sample_size_budget": deepcopy(reduction["sample_size_budget"]),
        "acceptance_gate_results": gates,
        "gate_check_summary": gate_check_summary(gates),
        "honest_verdict": verdict,
        "verdict_class": verdict_class,
        "verifier_is_oracle": False,
        "flagged_adversarial": False,
        "validation_receipts": [deepcopy(dict(row)) for row in validation_receipts],
        "window_evaluation_ready_score": ready,
        "capture_complete_score": reduction["capture_complete_score"],
        "role_support_score": reduction["role_support_score"],
        "role_minimums": {key: int(value) for key, value in role_minimums.items()},
        "role_counts": deepcopy(reduction["role_counts"]),
        "raw_logit_root": raw_logit_root,
        "raw_logit_shards": [deepcopy(dict(row)) for row in raw_logit_shards],
        "capture_reduction": _public_reduction(reduction),
        "fresh_evaluation_labels_opened": False,
        "predictive_benefit_claimed": False,
        "production_defaults_changed": False,
        "generator_weights_changed": False,
        "external_publication_authorized": False,
        "affected_validation_manifest": {
            "experiment_id": VALIDATION_MANIFEST.experiment_id,
            "test_paths": list(VALIDATION_MANIFEST.test_paths),
            "changed_modules": list(VALIDATION_MANIFEST.changed_modules),
            "static_paths": list(VALIDATION_MANIFEST.static_paths),
        },
        "capability_e2e": {
            "declared_entrypoint": WRAPPER_PATH.as_posix(),
            "fresh_process_cold_replay": "passed" if validation_passed else "pending",
            "independent_raw_reduction": "passed" if validation_passed else "pending",
            "numbered_runtime_e2e": "not_applicable_isolated_capture_no_shared_runtime_change",
        },
        "methodology": (
            "One owned native Qwen load scores both option orders for every eligible sealed "
            "test and online whole-response, focus-window, and derangement request. Labels "
            "remain sealed and complete context is never truncated."
        ),
        "field_principles": {},
        "reproducibility_checksum": "",
    }
    artifact["field_principles"] = _field_principles(tuple(artifact))
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return artifact


def _fixture_receipts() -> list[JsonDict]:
    return [
        {
            "name": name,
            "passed": True,
            "exit_code": 0,
            "timed_out": False,
            "command": ["fixture", name],
            "log_sha256": f"sha256:{name}",
        }
        for name in (*AFFECTED_CHECK_NAMES, *TERMINAL_CHECK_NAMES)
    ]


def build_artifact_for_test(
    tmp_path: Path,
    *,
    failed_call: bool = False,
    validation_passed: bool = True,
    role_minimums: Mapping[str, int] | None = None,
) -> JsonDict:
    """Build a complete circular fixture through production reducers."""

    plan = controlled_plan_fixture()
    rows = controlled_call_fixture(plan)
    events = fixture_invocation_events(forwards=len(rows))
    if failed_call:
        rows[0].update({"disposition": "failed", "error": "RuntimeError:fixture"})
        next(
            event
            for event in events
            if event["operation"] == "forward_calls"
            and event["call_id"] == "forward-0"
            and event["state"] == "completed"
        )["state"] = "failed"
    raw_root = tmp_path / "raw"
    shards = write_raw_shards(raw_root, plan=plan, rows=rows)
    return _build_artifact(
        plan=plan,
        rows=rows,
        raw_logit_root="raw",
        raw_logit_shards=shards,
        events=events,
        preconditions=[
            {
                "check": "fixture",
                "expected": True,
                "observed": True,
                "passed": True,
                "principle": "The bounded fixture prevents simulated evidence from escaping tests.",
            }
        ],
        source_hashes=[{"path": "fixture", "sha256": "sha256:fixture", "bytes": 1}],
        model_identity={
            "identity_authenticated": True,
            "model_id": MODEL_HF_ID,
            "model_path": "/fixture/Qwen3.8-27B-Q4_K_M.gguf",
            "model_sha256": "sha256:model",
            "quantization": "Q4_K_M",
            "owned_pid": os.getpid(),
            "gpu_uuid": "GPU-fixture",
            "peak_owned_vram_mb": 16_000,
            "actual_layer_placement": {"placement_authenticated": True},
            "llama_cpp_build": {"version": "fixture", "module_sha256": "sha256:fixture"},
        },
        device_identity={"cpu": "fixture", "selected_cuda": {"uuid": "GPU-fixture"}},
        validation_receipts=_fixture_receipts(),
        validation_passed=validation_passed,
        phase_spans=[
            {"phase": "fixture", "start_s": 0.0, "end_s": 3.0, "completed_units": len(rows)}
        ],
        duration_breakdown={
            "model_load": 1.0,
            "tokenize": 0.1,
            "prefill": 1.0,
            "readout": 0.01,
            "generation": 0.0,
            "optimization": 0.0,
            "reduction": 0.1,
            "validation": 0.1,
        },
        started_at="2026-09-21T00:00:00Z",
        ended_at="2026-09-21T00:00:03Z",
        started_ns=1_000_000_000,
        ended_ns=4_000_000_000,
        role_minimums=role_minimums or {"test": 2, "online": 1},
    )


def independent_reduce(
    artifact: Mapping[str, Any], *, root: Path, require_terminal: bool
) -> JsonDict:
    """Reload raw shards and recompute every public capture reduction."""

    manifest = artifact.get("raw_logit_shards")
    if not isinstance(manifest, list):
        return {"passed": False, "errors": ["raw_logit_shards_invalid"]}
    raw_root = root / str(artifact.get("raw_logit_root", RAW_DIR.as_posix()))
    try:
        raw = reload_raw_shards(raw_root, manifest)
        minimums = artifact.get("role_minimums")
        if not isinstance(minimums, Mapping):
            raise WindowCaptureError("role_minimums_invalid")
        reduction = reduce_capture(raw["plan"], raw["rows"], minimums=minimums)
    except WindowCaptureError as exc:
        return {"passed": False, "errors": [str(exc)]}
    errors: list[str] = []
    if _public_reduction(reduction) != artifact.get("capture_reduction"):
        errors.append("capture_reduction_mismatch")
    if reduction["group_rows"] != artifact.get("rows"):
        errors.append("group_rows_mismatch")
    if reduction["sample_size_budget"] != artifact.get("sample_size_budget"):
        errors.append("sample_size_budget_mismatch")
    if reduction["role_counts"] != artifact.get("role_counts"):
        errors.append("role_counts_mismatch")
    if require_terminal and not validation_names_passed(
        artifact.get("validation_receipts"), (*AFFECTED_CHECK_NAMES, *TERMINAL_CHECK_NAMES)
    ):
        errors.append("required_validation_failed")
    return {"passed": not errors, "errors": errors, "reduction": reduction}


def independent_cli_summary(reduced: Mapping[str, Any]) -> JsonDict:
    """Project an independent reduction to bounded headline CLI evidence."""

    reduction = reduced.get("reduction")
    headline = reduction if isinstance(reduction, Mapping) else {}
    return {
        "passed": reduced.get("passed") is True,
        "errors": deepcopy(reduced.get("errors", [])),
        "capture_complete_score": headline.get("capture_complete_score"),
        "role_support_score": headline.get("role_support_score"),
        "all_eligible_calls_valid": headline.get("all_eligible_calls_valid"),
        "source_transport_identity_valid": headline.get("source_transport_identity_valid"),
        "sample_size_budget": deepcopy(headline.get("sample_size_budget")),
        "role_counts": deepcopy(headline.get("role_counts")),
    }


def validate_artifact(
    value: object,
    *,
    root: Path = REPO_ROOT,
    require_validation: bool = True,
) -> list[str]:
    """Cold-check identities, raw calls, counters, gates, scores, and checksum."""

    if not isinstance(value, Mapping):
        return ["artifact_not_object"]
    artifact = dict(value)
    errors: list[str] = []
    expected = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "terminal_status": "complete",
        "run_date": RUN_DATE,
        "MODEL_SPECS": MODEL_SPECS,
        "model_specs": model_specs,
        "model_invoked": True,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "inference_substrate_class": INFERENCE_SUBSTRATE_CLASS,
        "execution_venue": EXECUTION_VENUE,
        "verifier_is_oracle": False,
        "fresh_evaluation_labels_opened": False,
        "predictive_benefit_claimed": False,
    }
    errors.extend(
        f"identity_mismatch:{field}"
        for field, wanted in expected.items()
        if artifact.get(field) != wanted
    )
    independent = independent_reduce(artifact, root=root, require_terminal=require_validation)
    if independent.get("passed") is not True:
        errors.extend(str(error) for error in independent.get("errors", []))
    reduction = independent.get("reduction")
    counts = artifact.get("invocation_counts")
    events = artifact.get("current_invocation_events")
    counts_balanced = False
    if not isinstance(counts, Mapping) or not isinstance(events, list):
        errors.append("invocation_evidence_invalid")
    else:
        recomputed = reduce_invocation_events([row for row in events if isinstance(row, Mapping)])
        if recomputed != counts:
            errors.append("invocation_counts_mismatch")
        counts_balanced = invocation_counts_balanced(counts)
        if not counts_balanced:
            errors.append("invocation_counts_unbalanced")

    validation_passed = True
    if require_validation:
        validation_passed = validation_names_passed(
            artifact.get("validation_receipts"), (*AFFECTED_CHECK_NAMES, *TERMINAL_CHECK_NAMES)
        )
        if not validation_passed:
            errors.append("required_validation_failed")
    if isinstance(reduction, Mapping) and isinstance(counts, Mapping):
        budget = reduction["sample_size_budget"]["calls"]
        identity_valid = (
            all(row.get("passed") is True for row in artifact.get("preconditions_checked") or [])
            and reduction["source_transport_identity_valid"] is True
            and counts_balanced
            and counts["model_loads"]["attempted"] == 1
            and counts["model_loads"]["completed"] == 1
            and counts["forward_calls"]["attempted"] == budget["attempted"]
            and counts["forward_calls"]["completed"] == budget["complete"]
            and counts["forward_calls"]["failed"] == budget["failed"]
            and (artifact.get("model_identity") or {}).get("identity_authenticated") is True
        )
        expected_ready = int(
            identity_valid
            and reduction["capture_complete_score"] == 1
            and reduction["role_support_score"] == 1
            and reduction["all_eligible_calls_valid"] is True
            and validation_passed
            and artifact.get("flagged_adversarial") is False
        )
        if artifact.get("window_evaluation_ready_score") != expected_ready:
            errors.append("window_evaluation_ready_score_mismatch")
        if artifact.get("capture_complete_score") != reduction["capture_complete_score"]:
            errors.append("capture_complete_score_mismatch")
        if artifact.get("role_support_score") != reduction["role_support_score"]:
            errors.append("role_support_score_mismatch")
        expected_gates = build_acceptance_gates(
            validity_passed=identity_valid,
            capture_complete=reduction["capture_complete_score"] == 1,
            role_support=reduction["role_support_score"] == 1,
            eligible_calls_complete=reduction["all_eligible_calls_valid"] is True,
            validation_passed=validation_passed,
        )
        if artifact.get("acceptance_gate_results") != expected_gates:
            errors.append("acceptance_gates_mismatch")
    gates = artifact.get("acceptance_gate_results")
    if not isinstance(gates, list) or any(
        not isinstance(row, Mapping) or "prevent" not in str(row.get("principle", ""))
        for row in gates or []
    ):
        errors.append("acceptance_gates_invalid")
    elif artifact.get("gate_check_summary") != gate_check_summary(gates):
        errors.append("gate_summary_mismatch")
    principles = artifact.get("field_principles")
    if not isinstance(principles, Mapping) or set(principles) != set(artifact):
        errors.append("field_principles_incomplete")
    elif any("prevent" not in str(principle) for principle in principles.values()):
        errors.append("field_principles_missing_failure_mode")
    if artifact.get("reproducibility_checksum") != artifact_checksum(artifact):
        errors.append("reproducibility_checksum_mismatch")
    return list(dict.fromkeys(errors))


# The remaining functions are live capability boundaries. Unit tests cover the
# reducers above; the entrypoint itself supplies the required hardware E2E.
def utc_now() -> str:  # pragma: no cover
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def progress(started: float, phase: str, event: str, **details: Any) -> None:  # pragma: no cover
    suffix = " ".join(f"{key}={value}" for key, value in sorted(details.items()))
    print(
        f"[exp7494] phase={phase} event={event} elapsed_s={time.monotonic() - started:.3f}"
        + (f" {suffix}" if suffix else ""),
        flush=True,
    )


def _source_receipt(path: Path, root: Path) -> JsonDict:  # pragma: no cover
    label = path.relative_to(root).as_posix() if path.is_relative_to(root) else str(path)
    receipt: JsonDict = {"path": label, "sha256": sha256_file(path), "bytes": path.stat().st_size}
    if path.suffix == ".json":
        value = _load_json_object(path)
        receipt.update(
            {
                "original_honest_verdict": value.get("honest_verdict"),
                "original_verdict_class": value.get("verdict_class"),
                "original_flagged_adversarial": value.get("flagged_adversarial"),
            }
        )
    return receipt


def _call_with_heartbeats(
    operation: Callable[[], Any], *, started: float, phase: str, operation_name: str
) -> Any:  # pragma: no cover
    return _FIT_CALL_WITH_HEARTBEATS(
        operation,
        started=started,
        phase=phase,
        operation_name=operation_name,
    )


def collect_preconditions(
    root: Path, *, started: float
) -> tuple[list[JsonDict], JsonDict, list[JsonDict]]:  # pragma: no cover
    """Authenticate exact producers, model bytes, runtime, and idle CUDA."""

    paths = (
        Path("AGENTS.md"),
        Path("CLAUDE.md"),
        Path("CODEX.md"),
        Path("research-program.md"),
        Path("ops/exclusion_manifest.yaml"),
        Path("ops/e2e-test-plan.md"),
        Path("scripts/experiment_template.py"),
        Path("python/carnot/reporting/current_work_receipt.py"),
        Path("python/carnot/experiment_7358_v646_validation_contract.py"),
        Path("python/carnot/reporting/experiment_7303_validation_scope.py"),
        Path("python/carnot/experiment_7477_v655_native_readout_pilot.py"),
        Path("python/carnot/experiment_7462_v654_option_protocol.py"),
        Path("python/carnot/experiment_7449_v653_source_protocol.py"),
        Path("python/carnot/experiment_7480_v655_source_eval_capture.py"),
        Path("python/carnot/experiment_7491_v656_window_protocol.py"),
        Path("python/carnot/experiment_7492_v656_window_pilot.py"),
        Path("openspec/capabilities/verification/spec.md"),
        Path("results/experiment_7491_v656_window_protocol.json"),
        Path("results/experiment_7492_v656_window_pilot.json"),
        PROTOCOL_RAW_DIR / "predictors.jsonl",
        PROTOCOL_RAW_DIR / "groups.jsonl",
        PROTOCOL_RAW_DIR / "windows.jsonl",
        PROTOCOL_RAW_DIR / "requests.jsonl",
        MODULE_PATH,
        WRAPPER_PATH,
        TEST_PATH,
    )
    checks: list[JsonDict] = []
    sources: list[JsonDict] = []
    for relative in paths:
        path = root / relative
        available = path.is_file() and path.stat().st_size > 0
        checks.append(
            fit_capture._check_row(
                f"resource_exists:{relative.as_posix()}",
                True,
                available,
                upstream=str(path),
                field_path="readable_nonempty_bytes",
            )
        )
        if available:
            sources.append(_source_receipt(path, root))

    protocol = _load_json_object(PROTOCOL_PATH)
    pilot = _load_json_object(PILOT_PATH)
    protocol_errors = (
        window_protocol.validate_artifact(protocol, root=root, require_terminal=True)
        if protocol
        else ["artifact_unreadable"]
    )
    pilot_errors = (
        window_pilot.validate_artifact(pilot, root=root, require_validation=True)
        if pilot
        else ["artifact_unreadable"]
    )
    upstream = reduce_upstream_gates(
        protocol,
        pilot,
        protocol_errors=protocol_errors,
        pilot_errors=pilot_errors,
    )
    checks.extend(upstream["checks"])

    cached = cached_current_model(gpu_index=0)
    model_path = Path(str(cached.get("model_path"))) if cached else Path("/absent-model")
    mandate = current_model()
    pilot_identity = dict(pilot.get("model_identity") or {})
    gpus = parity._gpu_inventory()
    idle = [
        row
        for row in gpus
        if row["memory_free_mb"] >= 20_000
        and row["memory_used_mb"] <= 1_024
        and row["utilization_pct"] <= 5
    ]
    selected = deepcopy(idle[0]) if idle else {}
    checks.extend(
        (
            fit_capture._check_row(
                "force_live",
                "1",
                os.environ.get("CARNOT_FORCE_LIVE"),
                upstream="environment",
                field_path="CARNOT_FORCE_LIVE",
            ),
            fit_capture._check_row(
                "model_hf_id",
                MODEL_HF_ID,
                cached.get("hf_id") if cached else None,
                upstream=str(model_path),
                field_path="hf_id",
            ),
            fit_capture._check_row(
                "model_path_matches_pilot",
                pilot_identity.get("model_path"),
                str(model_path),
                upstream=str(model_path),
                field_path="model_path",
            ),
            fit_capture._check_row(
                "model_quantization",
                "Q4_K_M",
                mandate.get("quantization"),
                upstream="carnot.inference.sota_models.current_model",
                field_path="quantization",
            ),
            fit_capture._check_row(
                "cached_gguf",
                True,
                model_path.is_file(),
                upstream=str(model_path),
                field_path="is_file",
            ),
            fit_capture._check_row(
                "idle_cuda_device",
                True,
                bool(idle),
                upstream="nvidia-smi",
                field_path="idle_candidate",
            ),
            fit_capture._check_row(
                "current_task_not_quarantined",
                False,
                "7494" in (root / "ops/exclusion_manifest.yaml").read_text(encoding="utf-8"),
                upstream="ops/exclusion_manifest.yaml",
                field_path=EXPERIMENT_ID,
            ),
            fit_capture._check_row(
                "driving_requirement",
                "REQ-VERIFY-7494",
                (
                    "REQ-VERIFY-7494"
                    if "REQ-VERIFY-7494" in SPEC_PATH.read_text(encoding="utf-8")
                    else None
                ),
                upstream="openspec/capabilities/verification/spec.md",
                field_path="REQ-*",
            ),
            fit_capture._check_row(
                "planned_forwards_within_ceiling",
                True,
                FORWARD_BUDGET <= FORWARD_CEILING,
                upstream="REQ-VERIFY-7494",
                field_path="forward_ceiling",
            ),
        )
    )
    model_sha256 = None
    if model_path.is_file():
        progress(started, "preconditions", "before_model_hash", path=model_path)
        model_sha256 = _call_with_heartbeats(
            lambda: sha256_file(model_path),
            started=started,
            phase="preconditions",
            operation_name="model_sha256",
        )
        progress(started, "preconditions", "after_model_hash", sha256=model_sha256)
        checks.append(
            fit_capture._check_row(
                "model_hash_matches_pilot",
                pilot_identity.get("model_sha256"),
                model_sha256,
                upstream=str(model_path),
                field_path="model_sha256",
            )
        )
        sources.append(
            {
                "path": str(model_path),
                "sha256": model_sha256,
                "bytes": model_path.stat().st_size,
                "quantization": mandate.get("quantization"),
            }
        )
    return (
        checks,
        {
            "protocol": protocol,
            "pilot": pilot,
            "pilot_model_identity": pilot_identity,
            "model_path": model_path,
            "model_sha256": model_sha256,
            "quantization": mandate.get("quantization"),
            "gpu_inventory": gpus,
            "selected_gpu": selected,
        },
        sources,
    )


def run_experiment(
    root: Path, run_date: str, *, output: Path = RESULT_PATH
) -> JsonDict:  # pragma: no cover
    """Configure and run the qualified one-load capture lifecycle."""

    if run_date != RUN_DATE:
        raise WindowCaptureError(f"run_date_mismatch:{run_date}")
    overrides: dict[str, Any] = {
        "EXPERIMENT_ID": EXPERIMENT_ID,
        "SCHEMA": SCHEMA,
        "MODEL_HF_ID": MODEL_HF_ID,
        "MODEL_SPECS": MODEL_SPECS,
        "model_specs": model_specs,
        "INFERENCE_SUBSTRATE": INFERENCE_SUBSTRATE,
        "INFERENCE_SUBSTRATE_CLASS": INFERENCE_SUBSTRATE_CLASS,
        "EXECUTION_VENUE": EXECUTION_VENUE,
        "FORWARD_BUDGET": FORWARD_BUDGET,
        "ELIGIBLE_FORWARD_BUDGET": ELIGIBLE_FORWARD_BUDGET,
        "MAX_LIVE_SECONDS": MAX_LIVE_SECONDS,
        "RESULT_PATH": RESULT_PATH,
        "RAW_DIR": RAW_DIR,
        "MODULE_PATH": MODULE_PATH,
        "WRAPPER_PATH": WRAPPER_PATH,
        "TEST_PATH": TEST_PATH,
        "VALIDATION_MANIFEST": VALIDATION_MANIFEST,
        "build_capture_plan": build_capture_plan,
        "reduce_capture": reduce_capture,
        "checkpoint_binding": checkpoint_binding,
        "write_group_checkpoint": write_group_checkpoint,
        "load_checkpoint_rows": load_checkpoint_rows,
        "collect_preconditions": collect_preconditions,
        "_build_artifact": _build_artifact,
        "validate_artifact": validate_artifact,
        "progress": progress,
        "_call_with_heartbeats": _call_with_heartbeats,
    }
    originals = {name: getattr(fit_capture, name) for name in overrides}
    try:
        for name, value in overrides.items():
            setattr(fit_capture, name, value)
        return fit_capture.run_experiment(root, run_date, output=output)
    finally:
        for name, value in originals.items():
            setattr(fit_capture, name, value)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:  # pragma: no cover
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", required=True)
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--output", type=Path, default=RESULT_PATH)
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--independent-reduce", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover
    """Run live capture or one read-only fresh-process capability replay."""

    args = parse_args(argv)
    if args.date != RUN_DATE:
        raise SystemExit(f"--date must be {RUN_DATE}")
    root = args.root.resolve()
    if args.cold_replay is not None:
        path = args.cold_replay if args.cold_replay.is_absolute() else root / args.cold_replay
        artifact = _load_json_object(path)
        errors = (
            validate_artifact(artifact, root=root, require_validation=False)
            if artifact
            else ["artifact_unreadable"]
        )
        print(
            json.dumps(
                {"mode": "cold_replay", "passed": not errors, "errors": errors}, sort_keys=True
            ),
            flush=True,
        )
        return int(bool(errors))
    if args.independent_reduce is not None:
        path = (
            args.independent_reduce
            if args.independent_reduce.is_absolute()
            else root / args.independent_reduce
        )
        artifact = _load_json_object(path)
        reduced = (
            independent_reduce(artifact, root=root, require_terminal=False)
            if artifact
            else {"passed": False, "errors": ["artifact_unreadable"]}
        )
        print(
            json.dumps(
                {"mode": "independent_reduce", **independent_cli_summary(reduced)}, sort_keys=True
            ),
            flush=True,
        )
        return int(reduced.get("passed") is not True)
    result = run_experiment(root, args.date, output=args.output)
    print(
        json.dumps(
            {
                "artifact": str(root / args.output),
                "honest_verdict": result["honest_verdict"],
                "window_evaluation_ready_score": result["window_evaluation_ready_score"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
