"""Capture sealed V656 whole-response and focus-window native logits.

The capture keeps model outcomes separate from labels. It records every sealed
fit request, including exclusions and failures, so later policy fitting can be
rechecked from raw evidence.

Spec refs: REQ-VERIFY-7493 and SCENARIO-VERIFY-7493-*.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from datetime import UTC, datetime
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import tempfile
import threading
import time
from typing import Any

from carnot import experiment_7462_v654_option_protocol as option_protocol
from carnot import experiment_7463_v654_semif_e0_logprob_parity as parity
from carnot import experiment_7477_v655_native_readout_pilot as native_pilot
from carnot import experiment_7491_v656_window_protocol as window_protocol
from carnot import experiment_7492_v656_window_pilot as window_pilot
from carnot import gpu_lease_phase_journal as lease_api
from carnot.experiment_7358_v646_validation_contract import (
    AffectedManifest,
    PlannedCommand,
    build_command_plan,
    reduce_affected_receipts,
    run_categorized_commands,
    validate_command_plan,
)
from carnot.inference.sota_models import cached_current_model, current_model
from carnot.reporting import experiment_7303_validation_scope as validation_scope
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file


JsonDict = dict[str, Any]
RUN_DATE = "20260921"
MILESTONE = "2026.09.656"
EXPERIMENT_ID = "exp7493-v656-window-fit-capture"
SCHEMA = "carnot.exp7493.v656.window_fit_capture.v1"
MODEL_HF_ID = "unsloth/Qwen3.8-27B-GGUF"
MODEL_SPECS = [MODEL_HF_ID]
model_specs = [MODEL_HF_ID]
INFERENCE_SUBSTRATE = "live_native_llama_cpp_lossless_window_fit_raw_logit_capture"
INFERENCE_SUBSTRATE_CLASS = "model_load_no_generation"
EXECUTION_VENUE = "host"
FORWARD_BUDGET = 1_936
ELIGIBLE_FORWARD_BUDGET = 1_904
MAX_LIVE_SECONDS = 3_300.0
TOKEN_CEILING = 2_048
MAX_ARTIFACT_BYTES = 20 * 1024 * 1024
RAW_SHARD_TARGET_BYTES = 12 * 1024 * 1024
ROLE_MINIMUMS = {"training": 150, "calibration_tuning": 40}
ROLE_PLANNED = {"training": 180, "calibration_tuning": 60}
OPTION_IDS = option_protocol.OPTION_IDS
FORBIDDEN_CAPTURE_FIELDS = {
    "label",
    "annotation",
    "annotations",
    "annotation_text",
    "annotation_spans",
    "response_generator",
    "response_generator_identity",
}

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_PATH = Path("results/experiment_7493_v656_window_fit_capture.json")
RAW_DIR = Path("results/raw/experiment_7493_v656_window_fit_capture")
MODULE_PATH = Path("python/carnot/experiment_7493_v656_window_fit_capture.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_7493_v656_window_fit_capture.py")
TEST_PATH = Path("tests/python/test_experiment_7493_v656_window_fit_capture.py")
SPEC_PATH = REPO_ROOT / "openspec/capabilities/verification/spec.md"
PROTOCOL_PATH = REPO_ROOT / "results/experiment_7491_v656_window_protocol.json"
PILOT_PATH = REPO_ROOT / "results/experiment_7492_v656_window_pilot.json"
PROTOCOL_RAW_DIR = Path("results/raw/experiment_7491_v656_window_protocol")
LEASE_RUNTIME_DIR = Path(os.environ.get("CARNOT_GPU_LEASE_RUNTIME_DIR", "/tmp/carnot-gpu-leases"))

VALIDATION_MANIFEST = AffectedManifest(
    experiment_id=EXPERIMENT_ID,
    test_paths=(TEST_PATH.as_posix(),),
    changed_modules=(MODULE_PATH.as_posix(),),
    static_paths=(WRAPPER_PATH.as_posix(),),
)
AFFECTED_CHECK_NAMES = validation_scope.REQUIRED_CHECK_NAMES
TERMINAL_CHECK_NAMES = (
    "declared_entrypoint_cold_replay",
    "independent_raw_reduction",
    "adversarial_verify",
    "verdict_row_consistency_strict",
)

PROTOCOL_EXPECTED = {
    "schema": "carnot.exp7491.v656.window_protocol.v1",
    "experiment_id": "exp7491-v656-window-protocol",
    "milestone": MILESTONE,
    "terminal_status": "complete",
    "honest_verdict": "complete_null_structural_window_protocol_ready",
    "verdict_class": "null",
    "flagged_adversarial": False,
    "window_protocol_ready_score": 1,
}
PILOT_EXPECTED = {
    "schema": "carnot.exp7492.v656.window_pilot.v1",
    "experiment_id": "exp7492-v656-window-pilot",
    "milestone": MILESTONE,
    "terminal_status": "complete",
    "honest_verdict": "complete_null_window_native_transport_and_capture_forecasts_ready",
    "verdict_class": "null",
    "flagged_adversarial": False,
    "window_native_ready_score": 1,
    "pilot_complete_score": 1,
}


class WindowCaptureError(ValueError):
    """Reject evidence that can silently change the sealed capture meaning."""


def _check_row(
    check: str,
    expected: Any,
    observed: Any,
    *,
    upstream: str,
    field_path: str,
) -> JsonDict:
    """Record exact operands so a changed producer cannot pass by summary."""

    return {
        "check": check,
        "upstream": upstream,
        "field_path": field_path,
        "expected": expected,
        "observed": observed,
        "op": "eq",
        "passed": observed == expected,
        "principle": "Exact producer checks prevent guessed or laundered prerequisite evidence.",
    }


def reduce_upstream_gates(
    protocol: Mapping[str, Any],
    pilot: Mapping[str, Any],
    *,
    protocol_errors: Sequence[str],
    pilot_errors: Sequence[str],
) -> JsonDict:
    """Require both V656 producer contracts before any current model work."""

    checks = [
        _check_row(
            f"protocol_{field}",
            expected,
            protocol.get(field),
            upstream=PROTOCOL_PATH.relative_to(REPO_ROOT).as_posix(),
            field_path=field,
        )
        for field, expected in PROTOCOL_EXPECTED.items()
    ]
    checks.extend(
        _check_row(
            f"pilot_{field}",
            expected,
            pilot.get(field),
            upstream=PILOT_PATH.relative_to(REPO_ROOT).as_posix(),
            field_path=field,
        )
        for field, expected in PILOT_EXPECTED.items()
    )
    pilot_identity = pilot.get("model_identity")
    identity = pilot_identity if isinstance(pilot_identity, Mapping) else {}
    for field, expected in (
        ("model_id", MODEL_HF_ID),
        ("quantization", "Q4_K_M"),
        ("identity_authenticated", True),
    ):
        checks.append(
            _check_row(
                f"pilot_model_{field}",
                expected,
                identity.get(field),
                upstream=PILOT_PATH.relative_to(REPO_ROOT).as_posix(),
                field_path=f"model_identity.{field}",
            )
        )
    checks.extend(
        (
            _check_row(
                "protocol_cold_validator",
                [],
                list(protocol_errors),
                upstream=PROTOCOL_PATH.relative_to(REPO_ROOT).as_posix(),
                field_path="cold_validator_errors",
            ),
            _check_row(
                "pilot_cold_validator",
                [],
                list(pilot_errors),
                upstream=PILOT_PATH.relative_to(REPO_ROOT).as_posix(),
                field_path="cold_validator_errors",
            ),
        )
    )
    return {"passed": all(row["passed"] for row in checks), "checks": checks}


def _window_key(row: Mapping[str, Any]) -> tuple[str, int]:
    return str(row["group_id"]), int(row["window_index"])


def _build_prompt(
    request: Mapping[str, Any],
    *,
    source: str,
    response: str,
    window: Mapping[str, Any] | None,
) -> tuple[str, str | None]:
    order = tuple(str(item) for item in request["option_order"])
    arm = str(request["arm"])
    if arm in {"whole_response", "source_derangement_control"}:
        return window_protocol.build_whole_prompt(source, response, order), None
    if arm != "focused_window" or window is None:
        raise WindowCaptureError(f"window_request_invalid:{request.get('request_id')}")
    focused = window_protocol.build_focused_prompt(source, response, order, window)
    return str(focused["prompt"]), str(focused["marked_response"])


def build_capture_plan(
    predictors: Sequence[Mapping[str, Any]],
    group_rows: Sequence[Mapping[str, Any]],
    window_rows: Sequence[Mapping[str, Any]],
    request_rows: Sequence[Mapping[str, Any]],
) -> list[JsonDict]:
    """Join the sealed label-free fit requests without changing their order."""

    fit_groups = [
        row for row in group_rows if row.get("role") in {"training", "calibration_tuning"}
    ]
    counts = Counter(str(row.get("role")) for row in fit_groups)
    if {role: counts[role] for role in ROLE_PLANNED} != ROLE_PLANNED:
        raise WindowCaptureError("fit_role_counts_invalid")
    fit_requests = [
        row for row in request_rows if row.get("role") in {"training", "calibration_tuning"}
    ]
    if len(fit_requests) != FORWARD_BUDGET:
        raise WindowCaptureError("fit_request_count_invalid")

    predictor_by_group = {str(row["group_id"]): row for row in predictors}
    group_by_id = {str(row["group_id"]): row for row in fit_groups}
    windows = {_window_key(row): row for row in window_rows}
    plan: list[JsonDict] = []
    for request in fit_requests:
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
        prompt, marked_response = _build_prompt(
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


def controlled_plan_fixture() -> list[JsonDict]:
    """Build three label-free groups with whole and focused order pairs."""

    plan: list[JsonDict] = []
    roles = ("training", "training", "calibration_tuning")
    for group_index, role in enumerate(roles):
        group_id = f"{role}-{group_index}"
        source = f"Source text {group_index}."
        response = f"Response text {group_index}."
        for arm, window_index in (("whole_response", None), ("focused_window", 0)):
            for order_index, order in enumerate((OPTION_IDS, tuple(reversed(OPTION_IDS)))):
                request_id = f"request-{group_index}-{arm}-{order_index}"
                plan.append(
                    {
                        "call_id": request_id,
                        "request_id": request_id,
                        "group_id": group_id,
                        "source_group_id": group_id,
                        "role": role,
                        "arm": arm,
                        "window_index": window_index,
                        "option_order": list(order),
                        "prompt": f"fixture prompt {request_id}",
                        "prompt_sha256": f"sha256:{request_id}",
                        "sealed_prompt_sha256": f"sha256:{request_id}",
                        "sealed_prompt_token_count": 3,
                        "prompt_token_count": 3,
                        "source_text": source,
                        "response_text": response,
                        "marked_response": response if arm == "focused_window" else None,
                        "source_sha256": canonical_hash(source),
                        "response_sha256": canonical_hash(response),
                        "source_version": {
                            "corpus": "fixture",
                            "release_revision": "fixture-v1",
                            "license": "fixture",
                        },
                        "byte_start": 0 if arm == "focused_window" else None,
                        "byte_end": len(response.encode()) if arm == "focused_window" else None,
                        "sentence_count": 1 if arm == "focused_window" else None,
                        "sentence_version": (
                            window_protocol.SENTENCE_VERSION if arm == "focused_window" else None
                        ),
                        "window_version": (
                            window_protocol.WINDOW_VERSION if arm == "focused_window" else None
                        ),
                        "window_sha256": (
                            canonical_hash(response) if arm == "focused_window" else None
                        ),
                        "eligible": True,
                        "disposition": "unstarted",
                        "attempted": False,
                        "error": None,
                        "gold_label": None,
                        "archived_pilot_reuse": False,
                    }
                )
    return plan


def controlled_call_fixture(plan: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Create finite native rows for deterministic reduction tests."""

    rows: list[JsonDict] = []
    for index, cell in enumerate(plan):
        order = tuple(str(item) for item in cell["option_order"])
        mapping = dict(zip(option_protocol.DISPLAY_LABELS, order, strict=True))
        logits = {
            "supported": 2.0 + index / 100.0,
            "contains_unsupported": 0.5 + index / 200.0,
        }
        probabilities = window_pilot._softmax_pair(logits)
        display = {label: logits[option_id] for label, option_id in mapping.items()}
        rows.append(
            {
                **{key: deepcopy(value) for key, value in cell.items() if key != "prompt"},
                "disposition": "complete",
                "attempted": True,
                "prompt_token_ids": [1, 2, 10 + index],
                "prompt_token_count": 3,
                "native_option_token_ids": {" A": 11, " B": 17},
                "label_to_option_id": mapping,
                "requested_score_position": 2,
                "actual_last_evaluated_position": 2,
                "token_boundary_receipts": {" A": True, " B": True},
                "raw_logits_by_display_label": display,
                "raw_logits_by_option_id": logits,
                "probabilities_by_option_id": probabilities,
                "log_odds_contains_unsupported": math.log(
                    probabilities["contains_unsupported"] / probabilities["supported"]
                ),
                "normalization": "two_option_softmax_float64",
                "order_remapping": mapping,
                "state_reset": True,
                "kv_state_id": f"fixture-reset-{index}",
                "generated_tokens": 0,
                "tokenize_s": 0.001,
                "prefill_s": 0.01,
                "readout_s": 0.0001,
                "model_sha256": "sha256:model",
                "tokenizer_identity": {"representation": "embedded_gguf"},
                "error": None,
            }
        )
    return rows


def _valid_complete_call(row: Mapping[str, Any], planned: Mapping[str, Any]) -> bool:
    """Require every field needed to reproduce one native option readout."""

    raw = row.get("raw_logits_by_option_id")
    display = row.get("raw_logits_by_display_label")
    probabilities = row.get("probabilities_by_option_id")
    order = tuple(str(item) for item in row.get("option_order") or ())
    mapping = dict(zip(option_protocol.DISPLAY_LABELS, order, strict=False))
    if not (
        isinstance(raw, Mapping)
        and isinstance(display, Mapping)
        and isinstance(probabilities, Mapping)
        and set(raw) == set(OPTION_IDS)
        and set(order) == set(OPTION_IDS)
        and row.get("label_to_option_id") == mapping
    ):
        return False
    try:
        logits = {key: float(raw[key]) for key in OPTION_IDS}
        expected = window_pilot._softmax_pair(logits)
        remapped = {option_id: float(display[label]) for label, option_id in mapping.items()}
    except (KeyError, TypeError, ValueError):
        return False
    count = row.get("prompt_token_count")
    prompt_ids = row.get("prompt_token_ids")
    native_ids = row.get("native_option_token_ids")
    boundaries = row.get("token_boundary_receipts")
    timings_valid = all(
        isinstance(row.get(field), (int, float))
        and math.isfinite(float(row[field]))
        and float(row[field]) >= 0
        for field in ("tokenize_s", "prefill_s", "readout_s")
    )
    identity_fields = (
        "request_id",
        "group_id",
        "source_group_id",
        "role",
        "arm",
        "window_index",
        "option_order",
        "prompt_sha256",
        "source_sha256",
        "response_sha256",
        "byte_start",
        "byte_end",
        "sentence_version",
        "window_version",
    )
    return (
        row.get("disposition") == "complete"
        and row.get("attempted") is True
        and all(row.get(field) == planned.get(field) for field in identity_fields)
        and isinstance(row.get("source_text"), str)
        and isinstance(row.get("response_text"), str)
        and isinstance(prompt_ids, list)
        and isinstance(count, int)
        and len(prompt_ids) == count
        and row.get("requested_score_position") == count - 1
        and row.get("actual_last_evaluated_position") == count - 1
        and isinstance(native_ids, Mapping)
        and set(native_ids) == set(option_protocol.DISPLAY_LABELS)
        and len(set(native_ids.values())) == 2
        and isinstance(boundaries, Mapping)
        and all(boundaries.get(label) is True for label in option_protocol.DISPLAY_LABELS)
        and all(math.isfinite(value) for value in logits.values())
        and all(math.isclose(logits[key], remapped[key], abs_tol=1e-12) for key in OPTION_IDS)
        and all(
            math.isclose(float(probabilities.get(key, math.nan)), expected[key], abs_tol=1e-12)
            for key in OPTION_IDS
        )
        and row.get("state_reset") is True
        and isinstance(row.get("kv_state_id"), str)
        and row.get("generated_tokens") == 0
        and timings_valid
        and isinstance(row.get("model_sha256"), str)
        and isinstance(row.get("tokenizer_identity"), Mapping)
        and row.get("error") is None
    )


def reduce_capture(
    plan: Sequence[Mapping[str, Any]],
    observed_rows: Sequence[Mapping[str, Any]],
    *,
    minimums: Mapping[str, int] = ROLE_MINIMUMS,
) -> JsonDict:
    """Reconcile calls and groups while keeping readiness separate from benefit."""

    expected = {str(row["request_id"]): row for row in plan}
    observed = {str(row.get("request_id")): row for row in observed_rows}
    duplicate_rows = len(observed) != len(observed_rows)
    extra_rows = sorted(set(observed) - set(expected))
    reconciled: list[JsonDict] = []
    source_transport_identity_valid = not duplicate_rows and not extra_rows
    for planned in plan:
        request_id = str(planned["request_id"])
        row = observed.get(request_id)
        if row is None:
            row = {
                **{key: deepcopy(value) for key, value in planned.items() if key != "prompt"},
                "disposition": "unstarted",
                "attempted": False,
                "error": "missing_observed_disposition",
            }
        identity_fields = (
            "request_id",
            "group_id",
            "source_group_id",
            "role",
            "arm",
            "window_index",
            "option_order",
            "prompt_sha256",
            "source_sha256",
            "response_sha256",
        )
        source_transport_identity_valid &= all(
            row.get(field) == planned.get(field) for field in identity_fields
        )
        reconciled.append(deepcopy(dict(row)))

    terminal = {"complete", "failed", "excluded", "censored", "unstarted"}
    capture_complete = (
        not duplicate_rows
        and not extra_rows
        and len(observed_rows) == len(plan)
        and all(row.get("disposition") in terminal for row in observed_rows)
    )
    all_eligible_valid = source_transport_identity_valid and all(
        _valid_complete_call(row, expected[str(row["request_id"])])
        for row in reconciled
        if expected[str(row["request_id"])].get("eligible") is True
    )
    excluded_valid = all(
        row.get("disposition") == "excluded" and row.get("attempted") is False
        for row in reconciled
        if expected[str(row["request_id"])].get("eligible") is False
    )
    all_eligible_valid &= excluded_valid

    group_order = list(dict.fromkeys(str(row["group_id"]) for row in plan))
    group_rows: list[JsonDict] = []
    role_counts: JsonDict = {}
    for group_id in group_order:
        planned_calls = [row for row in plan if str(row["group_id"]) == group_id]
        calls = [row for row in reconciled if str(row["group_id"]) == group_id]
        eligible_calls = [row for row in planned_calls if row.get("eligible") is True]
        eligible = bool(eligible_calls)
        complete = eligible and all(
            _valid_complete_call(
                next(row for row in calls if row["request_id"] == planned["request_id"]),
                planned,
            )
            for planned in eligible_calls
        )
        dispositions = Counter(str(row.get("disposition")) for row in calls)
        if not eligible and dispositions["excluded"] == len(calls):
            status = "excluded"
        elif complete:
            status = "complete"
        elif dispositions["failed"]:
            status = "failed"
        elif dispositions["censored"]:
            status = "censored"
        else:
            status = "unstarted"
        first = planned_calls[0]
        group_rows.append(
            {
                "unit_id": group_id,
                "group_id": group_id,
                "role": first["role"],
                "source_sha256": first["source_sha256"],
                "response_sha256": first["response_sha256"],
                "eligible": eligible,
                "status": status,
                "planned_calls": len(planned_calls),
                "attempted_calls": sum(row.get("attempted") is True for row in calls),
                "complete_calls": dispositions["complete"],
                "failed_calls": dispositions["failed"],
                "excluded_calls": dispositions["excluded"],
                "censored_calls": dispositions["censored"],
                "unstarted_calls": dispositions["unstarted"],
                "error": next((row.get("error") for row in calls if row.get("error")), None),
            }
        )

    for role in ROLE_PLANNED:
        groups = [row for row in group_rows if row["role"] == role]
        role_counts[role] = {
            "planned": len(groups),
            "eligible": sum(row["eligible"] is True for row in groups),
            "complete": sum(row["status"] == "complete" for row in groups),
            "failed": sum(row["status"] == "failed" for row in groups),
            "excluded": sum(row["status"] == "excluded" for row in groups),
            "censored": sum(row["status"] == "censored" for row in groups),
            "unstarted": sum(row["status"] == "unstarted" for row in groups),
            "minimum": int(minimums.get(role, 0)),
        }
    role_support = all(
        role_counts[role]["eligible"] >= int(minimums.get(role, 0)) for role in role_counts
    )

    call_dispositions = Counter(str(row.get("disposition")) for row in reconciled)
    call_budget = {
        "planned": len(plan),
        "attempted": sum(row.get("attempted") is True for row in reconciled),
        "complete": call_dispositions["complete"],
        "failed": call_dispositions["failed"],
        "excluded": call_dispositions["excluded"],
        "censored": call_dispositions["censored"],
        "unstarted": call_dispositions["unstarted"],
    }
    group_statuses = Counter(str(row["status"]) for row in group_rows)
    group_budget = {
        "planned": len(group_rows),
        "attempted": sum(row["attempted_calls"] > 0 for row in group_rows),
        "complete": group_statuses["complete"],
        "failed": group_statuses["failed"],
        "excluded": group_statuses["excluded"],
        "censored": group_statuses["censored"],
        "unstarted": group_statuses["unstarted"],
    }
    return {
        "capture_complete_score": int(capture_complete),
        "role_support_score": int(role_support),
        "all_eligible_calls_valid": bool(all_eligible_valid),
        "source_transport_identity_valid": bool(source_transport_identity_valid),
        "duplicate_rows": duplicate_rows,
        "extra_rows": extra_rows,
        "sample_size_budget": {"calls": call_budget, "groups": group_budget},
        "role_counts": role_counts,
        "group_rows": group_rows,
        "reconciled_rows": reconciled,
    }


def checkpoint_binding(
    plan: Sequence[Mapping[str, Any]],
    *,
    model_sha256: str,
    tokenizer_identity: Mapping[str, Any],
    request_manifest_sha256: str,
) -> JsonDict:
    """Bind resume evidence to every request and model identity input."""

    stable_plan = [{key: value for key, value in row.items() if key != "prompt"} for row in plan]
    return {
        "schema": SCHEMA,
        "schedule_sha256": canonical_hash(stable_plan),
        "model_sha256": model_sha256,
        "tokenizer_identity_sha256": canonical_hash(tokenizer_identity),
        "request_manifest_sha256": request_manifest_sha256,
        "role_counts": deepcopy(ROLE_PLANNED),
    }


def write_group_checkpoint(
    root: Path,
    *,
    binding: Mapping[str, Any],
    group_id: str,
    rows: Sequence[Mapping[str, Any]],
) -> None:
    """Save one source group and update a hash-bound checkpoint index."""

    root.mkdir(parents=True, exist_ok=True)
    index_path = root / "checkpoint-index.json"
    index = _load_json_object(index_path) if index_path.is_file() else {}
    if index and index.get("binding") != dict(binding):
        raise WindowCaptureError("checkpoint_binding_mismatch")
    digest = hashlib.sha256(group_id.encode()).hexdigest()[:24]
    group_path = root / f"group-{digest}.json"
    row_values = [deepcopy(dict(row)) for row in rows]
    payload = {
        "schema": "carnot.exp7493.group_checkpoint.v1",
        "binding": dict(binding),
        "group_id": group_id,
        "rows": row_values,
        "rows_sha256": canonical_hash(row_values),
    }
    atomic_json(group_path, payload)
    groups = dict(index.get("groups") or {})
    groups[group_id] = {
        "path": group_path.name,
        "sha256": sha256_file(group_path),
        "rows": len(row_values),
    }
    atomic_json(
        index_path,
        {
            "schema": "carnot.exp7493.checkpoint_index.v1",
            "binding": dict(binding),
            "groups": groups,
        },
    )


def load_checkpoint_rows(root: Path, *, expected_binding: Mapping[str, Any]) -> list[JsonDict]:
    """Load only checkpoint groups with the exact current binding and bytes."""

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


def _jsonl_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(
        (json.dumps(dict(row), sort_keys=True, separators=(",", ":")) + "\n").encode()
        for row in rows
    )


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def _split_rows(rows: Sequence[Mapping[str, Any]]) -> list[list[Mapping[str, Any]]]:
    chunks: list[list[Mapping[str, Any]]] = []
    current: list[Mapping[str, Any]] = []
    current_size = 0
    for row in rows:
        row_size = len(_jsonl_bytes([row]))
        if row_size >= MAX_ARTIFACT_BYTES:
            raise WindowCaptureError("single_raw_row_exceeds_20_mib")
        if current and current_size + row_size > RAW_SHARD_TARGET_BYTES:
            chunks.append(current)
            current = []
            current_size = 0
        current.append(row)
        current_size += row_size
    if current:
        chunks.append(current)
    return chunks


def write_raw_shards(
    raw_dir: Path,
    *,
    plan: Sequence[Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
) -> list[JsonDict]:
    """Write bounded plan and logit shards with exact byte hashes."""

    stable_plan = [deepcopy(dict(row)) for row in plan]
    manifest: list[JsonDict] = []
    for kind, values in (("plan", stable_plan), ("raw_logits", list(rows))):
        for index, chunk in enumerate(_split_rows(values)):
            name = f"{kind}-{index:03d}.jsonl"
            payload = _jsonl_bytes(chunk)
            if len(payload) >= MAX_ARTIFACT_BYTES:
                raise WindowCaptureError(f"raw_shard_exceeds_20_mib:{name}")
            _atomic_bytes(raw_dir / name, payload)
            manifest.append(
                {
                    "path": name,
                    "kind": kind,
                    "rows": len(chunk),
                    "size_bytes": len(payload),
                    "sha256": "sha256:" + hashlib.sha256(payload).hexdigest(),
                }
            )
    return manifest


def reload_raw_shards(raw_dir: Path, manifest: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Rehash every shard and restore the exact ordered plan and call rows."""

    if not manifest or {row.get("kind") for row in manifest} != {"plan", "raw_logits"}:
        raise WindowCaptureError("raw_manifest_invalid")
    loaded: JsonDict = {"plan": [], "rows": []}
    for receipt in manifest:
        path = raw_dir / str(receipt.get("path"))
        if not path.is_file() or sha256_file(path) != receipt.get("sha256"):
            raise WindowCaptureError(f"raw_shard_hash_mismatch:{receipt.get('path')}")
        if path.stat().st_size >= MAX_ARTIFACT_BYTES:
            raise WindowCaptureError(f"raw_shard_exceeds_20_mib:{receipt.get('path')}")
        values = _load_jsonl(path)
        if len(values) != receipt.get("rows"):
            raise WindowCaptureError(f"raw_shard_row_count_mismatch:{receipt.get('path')}")
        target = "plan" if receipt.get("kind") == "plan" else "rows"
        loaded[target].extend(values)
    return loaded


fixture_invocation_events = native_pilot.fixture_invocation_events
reduce_invocation_events = native_pilot.reduce_invocation_events
invocation_counts_balanced = native_pilot.invocation_counts_balanced


def _gate(
    check: str,
    category: str,
    expected: Any,
    observed: Any,
    *,
    upstream: str,
    field_path: str,
    principle: str,
) -> JsonDict:
    return {
        "check": check,
        "category": category,
        "expected": expected,
        "observed": observed,
        "op": "eq",
        "passed": observed == expected,
        "upstream": upstream,
        "field_path": field_path,
        "principle": principle,
    }


def build_acceptance_gates(
    *,
    validity_passed: bool,
    capture_complete: bool,
    role_support: bool,
    eligible_calls_complete: bool,
    validation_passed: bool,
) -> list[JsonDict]:
    """Keep evidence validity, capture readiness, and unmeasured benefit separate."""

    validity = "A favorable metric cannot excuse invalid evidence; this check prevents invalid evidence from being promoted."
    readiness = "A valid scientific null must not block independent measurements; this check prevents efficacy from redefining capture readiness."
    benefit = "A favorable seed, fixture or low-support result cannot replace held-out value; this check prevents capture data from becoming a benefit claim."
    return [
        _gate(
            "authenticated_identity_and_accounting",
            "validity",
            True,
            validity_passed,
            upstream="preconditions_checked",
            field_path="all",
            principle=validity,
        ),
        _gate(
            "required_validation",
            "validity",
            True,
            validation_passed,
            upstream="validation_receipts",
            field_path="required",
            principle=validity,
        ),
        _gate(
            "capture_complete",
            "readiness",
            True,
            capture_complete,
            upstream="capture_reduction",
            field_path="capture_complete_score",
            principle=readiness,
        ),
        _gate(
            "role_support",
            "readiness",
            True,
            role_support,
            upstream="capture_reduction",
            field_path="role_support_score",
            principle=readiness,
        ),
        _gate(
            "eligible_calls_complete",
            "readiness",
            True,
            eligible_calls_complete,
            upstream="capture_reduction",
            field_path="all_eligible_calls_valid",
            principle=readiness,
        ),
        _gate(
            "predictive_benefit_not_measured",
            "benefit",
            False,
            False,
            upstream="honest_verdict",
            field_path="predictive_benefit_claimed",
            principle=benefit,
        ),
    ]


def gate_check_summary(gates: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Name every failed operand so a blocked check cannot disappear."""

    failed = [
        {
            key: row.get(key)
            for key in ("check", "category", "upstream", "field_path", "expected", "observed")
        }
        for row in gates
        if row.get("passed") is not True
    ]
    return {
        "passed": not failed,
        "failed_checks": failed,
        "first_failure": failed[0] if failed else None,
    }


def artifact_checksum(value: Mapping[str, Any]) -> str:
    """Bind every artifact field except the checksum to stable JSON bytes."""

    payload = deepcopy(dict(value))
    payload.pop("reproducibility_checksum", None)
    return canonical_hash(payload)


_FIELD_PRINCIPLES = {
    "schema": "Versioned identity prevents reader drift across experiments.",
    "run_date": "The fixed run date prevents evidence from being assigned to another capture.",
    "preconditions_checked": "Exact resources and producer flags prevent fabricated prerequisites.",
    "MODEL_SPECS": "The mandated model name prevents a small substitute from becoming headline evidence.",
    "model_specs": "The lowercase model declaration prevents schema readers from missing the executed model.",
    "model_invoked": "The current invocation flag prevents archived model work from becoming current inference.",
    "invocation_counts": "Balanced counters prevent failed or unfinished model work from disappearing.",
    "inference_substrate": "The native readout name prevents aggregation or generation from being implied.",
    "inference_substrate_class": "The no-generation class prevents padded duration and token-generation claims.",
    "execution_venue": "The host venue prevents archived board evidence from replacing current CUDA work.",
    "duration_s": "Measured elapsed time prevents a synthetic duration floor.",
    "phase_spans": "Measured checkpoints prevent unfinished phases and stalls from disappearing.",
    "random_seed": "Frozen role, order, audit, and interval choices prevent outcome-guided retries.",
    "reproducibility_checksum": "The checksum prevents silent code, model, prompt, shard, or scope drift.",
    "source_artifact_hashes": "Original bytes and flags prevent upstream evidence laundering.",
    "rows": "Per-group dispositions prevent failed or excluded units from disappearing.",
    "sample_size_budget": "Separate call and group budgets prevent silent roster shrinkage.",
    "acceptance_gate_results": "Typed operands prevent one favorable metric from bypassing another gate.",
    "gate_check_summary": "Exact failed paths prevent blocked evidence from being hidden.",
    "honest_verdict": "A terminal finding prevents completed capture work from looking unfinished.",
    "verdict_class": "The closed class prevents retryable work from masquerading as a scientific null.",
    "verifier_is_oracle": "The oracle declaration prevents analytic controls from becoming positive evidence.",
    "flagged_adversarial": "Actual reader flags remain visible to prevent deletion from opening a gate.",
    "validation_receipts": "Exact commands, exits, and hashes prevent validation scope drift.",
    "field_principles": "A purpose for every field prevents unexplained evidence from becoming authoritative.",
    "window_fit_ready_score": "Exact valid completion and support prevent efficacy from defining capture readiness.",
    "capture_complete_score": "Explicit dispositions prevent missing calls from becoming silent zeros.",
    "role_support_score": "Independent group minima prevent windows or seeds from inflating support.",
    "raw_logit_shards": "Hash-bound native rows prevent summary-only evidence claims.",
    "role_counts": "Planned and actual groups prevent calls or windows from replacing independent units.",
}


def _field_principles(keys: Sequence[str]) -> dict[str, str]:
    return {
        key: _FIELD_PRINCIPLES.get(
            key,
            f"The {key} field preserves measured evidence to prevent silent record drift.",
        )
        for key in keys
    }


def validation_names_passed(receipts: object, names: Sequence[str]) -> bool:
    if not isinstance(receipts, list):
        return False
    rows = [row for row in receipts if isinstance(row, Mapping)]
    return all(
        sum(
            row.get("name") == name
            and row.get("passed") is True
            and row.get("exit_code") == 0
            and row.get("timed_out") is not True
            for row in rows
        )
        == 1
        for name in names
    )


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
        verdict = "complete_disqualified_window_fit_capture_invalid_evidence"
        verdict_class = "disqualified"
    elif not reduction["all_eligible_calls_valid"]:
        verdict = "complete_partial_window_fit_capture_incomplete"
        verdict_class = "partial"
    elif ready:
        verdict = "complete_null_window_fit_capture_ready_predictive_benefit_not_tested"
        verdict_class = "null"
    else:
        verdict = "complete_disqualified_window_fit_capture_not_ready"
        verdict_class = "disqualified"
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
            "audit_seed": 656093,
            "order_seed": "both_frozen_orders",
            "interval_seed": None,
            "deterministic_null_seed_reason": "Capture performs no fit, interval, or outcome-guided retry.",
        },
        "source_artifact_hashes": [deepcopy(dict(row)) for row in source_hashes],
        "request_manifest_identity": {
            "planned_calls": len(plan),
            "eligible_calls": sum(row.get("eligible") is True for row in plan),
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
            "forward_ceiling": FORWARD_BUDGET,
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
        "window_fit_ready_score": ready,
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
            "whole-response, focus-window, and derangement request. No labels enter prompts."
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
            "reduction": 0.1,
            "validation": 0.1,
        },
        started_at="2026-09-21T00:00:00Z",
        ended_at="2026-09-21T00:00:03Z",
        started_ns=1_000_000_000,
        ended_ns=4_000_000_000,
        role_minimums=role_minimums or {"training": 2, "calibration_tuning": 1},
    )


def independent_reduce(
    artifact: Mapping[str, Any], *, root: Path, require_terminal: bool
) -> JsonDict:
    """Reload raw shards and recompute all capture scores independently."""

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


def validate_artifact(
    value: object,
    *,
    root: Path = REPO_ROOT,
    require_validation: bool = True,
) -> list[str]:
    """Cold-check identities, raw rows, counters, gates, scores, and checksum."""

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
        if artifact.get("window_fit_ready_score") != expected_ready:
            errors.append("window_fit_ready_score_mismatch")
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


def _load_json_object(path: Path) -> JsonDict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(value) if isinstance(value, Mapping) else {}


def _load_jsonl(path: Path) -> list[JsonDict]:
    try:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    except (OSError, json.JSONDecodeError) as exc:
        raise WindowCaptureError(f"jsonl_unreadable:{path}") from exc
    if any(not isinstance(row, dict) for row in rows):
        raise WindowCaptureError(f"jsonl_object_required:{path}")
    return rows


# The remaining functions are live capability boundaries. Pure behavior is
# covered above; these functions retain real clocks, subprocesses, CUDA, and I/O.
def utc_now() -> str:  # pragma: no cover
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def progress(started: float, phase: str, event: str, **details: Any) -> None:  # pragma: no cover
    suffix = " ".join(f"{key}={value}" for key, value in sorted(details.items()))
    print(
        f"[exp7493] phase={phase} event={event} elapsed_s={time.monotonic() - started:.3f}"
        + (f" {suffix}" if suffix else ""),
        flush=True,
    )


def _call_with_heartbeats(
    operation: Callable[[], Any], *, started: float, phase: str, operation_name: str
) -> Any:  # pragma: no cover
    result: list[Any] = []
    errors: list[BaseException] = []

    def target() -> None:
        try:
            result.append(operation())
        except BaseException as exc:  # noqa: BLE001 - owner thread re-raises.
            errors.append(exc)

    worker = threading.Thread(target=target, name=f"{EXPERIMENT_ID}-{phase}", daemon=True)
    worker.start()
    while worker.is_alive():
        worker.join(timeout=60.0)
        if worker.is_alive():
            progress(started, phase, "pending", operation=operation_name)
    if errors:
        raise errors[0]
    return result[0] if result else None


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


def _gpu_process_rows(pid: int) -> list[JsonDict]:  # pragma: no cover
    completed = subprocess.run(  # noqa: S603 - fixed local diagnostic.
        (
            "nvidia-smi",
            "--query-compute-apps=pid,gpu_uuid,used_memory",
            "--format=csv,noheader,nounits",
        ),
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    rows: list[JsonDict] = []
    for line in completed.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) == 3 and parts[0].isdigit() and int(parts[0]) == pid:
            rows.append(
                {"pid": int(parts[0]), "gpu_uuid": parts[1], "used_memory_mb": int(parts[2])}
            )
    return rows


def collect_preconditions(
    root: Path, *, started: float
) -> tuple[list[JsonDict], JsonDict, list[JsonDict]]:  # pragma: no cover
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
        Path("python/carnot/experiment_7479_v655_source_fit_capture.py"),
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
            _check_row(
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
            _check_row(
                "force_live",
                "1",
                os.environ.get("CARNOT_FORCE_LIVE"),
                upstream="environment",
                field_path="CARNOT_FORCE_LIVE",
            ),
            _check_row(
                "model_hf_id",
                MODEL_HF_ID,
                cached.get("hf_id") if cached else None,
                upstream=str(model_path),
                field_path="hf_id",
            ),
            _check_row(
                "model_path_matches_pilot",
                pilot_identity.get("model_path"),
                str(model_path),
                upstream=str(model_path),
                field_path="model_path",
            ),
            _check_row(
                "model_quantization",
                "Q4_K_M",
                mandate.get("quantization"),
                upstream="carnot.inference.sota_models.current_model",
                field_path="quantization",
            ),
            _check_row(
                "cached_gguf",
                True,
                model_path.is_file(),
                upstream=str(model_path),
                field_path="is_file",
            ),
            _check_row(
                "idle_cuda_device",
                True,
                bool(idle),
                upstream="nvidia-smi",
                field_path="idle_candidate",
            ),
            _check_row(
                "current_task_not_quarantined",
                False,
                "7493" in (root / "ops/exclusion_manifest.yaml").read_text(encoding="utf-8"),
                upstream="ops/exclusion_manifest.yaml",
                field_path=EXPERIMENT_ID,
            ),
            _check_row(
                "driving_requirement",
                "REQ-VERIFY-7493",
                "REQ-VERIFY-7493"
                if "REQ-VERIFY-7493" in SPEC_PATH.read_text(encoding="utf-8")
                else None,
                upstream="openspec/capabilities/verification/spec.md",
                field_path="REQ-*",
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
            _check_row(
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


class WindowCaptureNativeScorer(parity.NativeScorer):  # pragma: no cover
    """Load a 4,096-token native context for all sealed eligible prompts."""

    def load(self) -> None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.gpu_index)
        import llama_cpp  # noqa: PLC0415

        self.backend = llama_cpp
        self.llm = llama_cpp.Llama(
            model_path=str(self.model_path),
            n_ctx=4096,
            n_batch=512,
            n_gpu_layers=-1,
            split_mode=0,
            main_gpu=0,
            logits_all=True,
            verbose=False,
        )
        for label in ("A", "B"):
            ids = self.tokenize(f" {label}", add_bos=False)
            if len(ids) != 1:
                raise RuntimeError(f"option_label_not_single_token:{label}:{ids}")
            self.option_token_ids[label] = ids[0]


class WindowCaptureRunner(native_pilot.NativeReadoutRunner):  # pragma: no cover
    """Reuse qualified native receipts with a larger sealed prompt context."""

    def __init__(self, model_path: Path, gpu_index: int) -> None:
        self.scorer = WindowCaptureNativeScorer(model_path, gpu_index)
        self.reset_epoch = 0
        self.identity: JsonDict = {}


def _event(call_id: str, operation: str, state: str) -> JsonDict:  # pragma: no cover
    return {
        "call_id": call_id,
        "operation": operation,
        "state": state,
        "monotonic_ns": time.monotonic_ns(),
        "scope": "current_owned_process",
    }


def _phase_span(
    phase: str, phase_started: float, run_started: float, units: int, checkpoint: str
) -> JsonDict:  # pragma: no cover
    return {
        "phase": phase,
        "start_s": phase_started - run_started,
        "end_s": time.monotonic() - run_started,
        "completed_units": units,
        "checkpoint": checkpoint,
    }


def _terminal_commands(root: Path, candidate: Path) -> list[PlannedCommand]:  # pragma: no cover
    relative = candidate.relative_to(root).as_posix()
    specs = (
        validation_scope.CommandSpec(
            "declared_entrypoint_cold_replay",
            (
                ".venv/bin/python",
                "-u",
                WRAPPER_PATH.as_posix(),
                "--date",
                RUN_DATE,
                "--cold-replay",
                relative,
            ),
            "completion",
            180,
        ),
        validation_scope.CommandSpec(
            "independent_raw_reduction",
            (
                ".venv/bin/python",
                "-u",
                WRAPPER_PATH.as_posix(),
                "--date",
                RUN_DATE,
                "--independent-reduce",
                relative,
            ),
            "completion",
            180,
        ),
        validation_scope.CommandSpec(
            "adversarial_verify",
            (".venv/bin/python", "-u", "scripts/adversarial_verify.py", relative),
            "safety",
            180,
        ),
        validation_scope.CommandSpec(
            "verdict_row_consistency_strict",
            (
                ".venv/bin/python",
                "-u",
                "scripts/verdict_row_consistency_lint.py",
                "--strict",
                relative,
            ),
            "completion",
            180,
        ),
    )
    return [
        PlannedCommand(spec, "safety" if spec.name == "adversarial_verify" else "completion", True)
        for spec in specs
    ]


def run_experiment(
    root: Path, run_date: str, *, output: Path = RESULT_PATH
) -> JsonDict:  # pragma: no cover
    """Run one owned Qwen load, sealed fit forwards, and scoped validation."""

    if run_date != RUN_DATE:
        raise WindowCaptureError(f"run_date_mismatch:{run_date}")
    run_started = time.monotonic()
    started_ns = time.monotonic_ns()
    started_at = utc_now()
    spans: list[JsonDict] = []
    events: list[JsonDict] = []
    rows: list[JsonDict] = []
    durations: JsonDict = {
        "model_load": 0.0,
        "tokenize": 0.0,
        "prefill": 0.0,
        "readout": 0.0,
        "generation": 0.0,
        "reduction": 0.0,
        "validation": 0.0,
    }

    progress(run_started, "preconditions", "start")
    phase_started = time.monotonic()
    preconditions, context, source_hashes = collect_preconditions(root, started=run_started)
    if not all(row.get("passed") is True for row in preconditions):
        failed = next(row for row in preconditions if row.get("passed") is not True)
        raise WindowCaptureError(f"precondition_failed:{failed['check']}:{failed.get('observed')}")
    spans.append(
        _phase_span(
            "preconditions", phase_started, run_started, len(preconditions), "authenticated"
        )
    )
    progress(run_started, "preconditions", "complete", checked=len(preconditions))

    progress(run_started, "schedule", "start")
    phase_started = time.monotonic()
    upstream_raw = root / PROTOCOL_RAW_DIR
    plan = build_capture_plan(
        _load_jsonl(upstream_raw / "predictors.jsonl"),
        _load_jsonl(upstream_raw / "groups.jsonl"),
        _load_jsonl(upstream_raw / "windows.jsonl"),
        _load_jsonl(upstream_raw / "requests.jsonl"),
    )
    manifest_path = root / RAW_DIR / "affected_validation_manifest.json"
    atomic_json(
        manifest_path,
        {
            "experiment_id": VALIDATION_MANIFEST.experiment_id,
            "test_paths": list(VALIDATION_MANIFEST.test_paths),
            "changed_modules": list(VALIDATION_MANIFEST.changed_modules),
            "static_paths": list(VALIDATION_MANIFEST.static_paths),
        },
    )
    source_hashes.append(_source_receipt(manifest_path, root))
    spans.append(_phase_span("schedule", phase_started, run_started, len(plan), "frozen"))
    progress(run_started, "schedule", "complete", planned=len(plan))

    gpu = dict(context["selected_gpu"])
    model_path = Path(context["model_path"])
    lease = lease_api.GpuLease.acquire(
        runtime_dir=LEASE_RUNTIME_DIR,
        task_id=EXPERIMENT_ID,
        device_uuid=str(gpu["uuid"]),
        expected_model=str(model_path),
        vram_before_mb=int(gpu["memory_used_mb"]),
        ttl_s=MAX_LIVE_SECONDS + 300.0,
    )
    owner = lease.owner_receipt()
    runner = WindowCaptureRunner(model_path, int(gpu["index"]))
    model_identity: JsonDict = {}
    release: JsonDict = {}
    inference_ok = False
    live_started = time.monotonic()
    try:
        lease.transition("admitted")
        lease.transition("loading")
        phase_started = time.monotonic()
        progress(run_started, "model_load", "before_model_load", gpu_uuid=gpu["uuid"])
        events.append(_event("native-load", "model_loads", "attempted"))
        load_started = time.monotonic()
        try:
            model_identity = _call_with_heartbeats(
                lambda: runner.load(weight_sha256=str(context["model_sha256"])),
                started=run_started,
                phase="model_load",
                operation_name="native_qwen_load",
            )
        except BaseException:
            events.append(_event("native-load", "model_loads", "failed"))
            raise
        events.append(_event("native-load", "model_loads", "completed"))
        durations["model_load"] = time.monotonic() - load_started
        process_rows = _gpu_process_rows(os.getpid())
        selected_rows = [row for row in process_rows if row["gpu_uuid"] == gpu["uuid"]]
        peak_owned_vram_mb = max((row["used_memory_mb"] for row in selected_rows), default=0)
        resident_mb = parity._gpu_memory(int(gpu["index"]))
        metadata = dict(getattr(runner.scorer.llm, "metadata", {}) or {})
        block_counts = {
            key: value for key, value in metadata.items() if str(key).endswith(".block_count")
        }
        placement_authenticated = bool(selected_rows) and peak_owned_vram_mb > 1_000
        pilot_identity = context["pilot_model_identity"]
        runtime_match = (
            model_identity.get("model_path") == pilot_identity.get("model_path")
            and model_identity.get("model_sha256") == pilot_identity.get("model_sha256")
            and model_identity.get("tokenizer_identity") == pilot_identity.get("tokenizer_identity")
            and model_identity.get("llama_cpp_build") == pilot_identity.get("llama_cpp_build")
        )
        model_identity.update(
            {
                "quantization": context["quantization"],
                "owned_pid": os.getpid(),
                "gpu_uuid": gpu["uuid"],
                "peak_owned_vram_mb": peak_owned_vram_mb,
                "owned_gpu_process_rows": process_rows,
                "actual_layer_placement": {
                    "requested_n_gpu_layers": -1,
                    "model_block_counts": block_counts,
                    "selected_gpu_uuid": gpu["uuid"],
                    "owned_pid_resident_vram_mb": peak_owned_vram_mb,
                    "placement_evidence": "nvidia_smi_compute_app_pid_gpu_uuid_and_memory",
                    "placement_authenticated": placement_authenticated,
                },
                "observed_offload": {
                    "baseline_vram_mb": gpu["memory_used_mb"],
                    "resident_vram_mb": resident_mb,
                    "cuda_offload": placement_authenticated,
                },
                "qualified_pilot_identity_match": runtime_match,
                "identity_authenticated": runtime_match and placement_authenticated,
                "lease_id": owner["lease_id"],
            }
        )
        if model_identity["identity_authenticated"] is not True:
            raise WindowCaptureError("loaded_model_identity_mismatch")
        lease.transition("resident", vram_mb=resident_mb)
        lease.transition("inferencing")
        spans.append(_phase_span("model_load", phase_started, run_started, 1, "resident"))
        progress(run_started, "model_load", "after_model_load", owned_vram_mb=peak_owned_vram_mb)

        binding = checkpoint_binding(
            plan,
            model_sha256=str(context["model_sha256"]),
            tokenizer_identity=model_identity["tokenizer_identity"],
            request_manifest_sha256=str(
                next(
                    row["sha256"]
                    for row in source_hashes
                    if str(row["path"]).endswith("requests.jsonl")
                )
            ),
        )
        checkpoint_root = root / RAW_DIR / "checkpoints"
        existing: dict[str, JsonDict] = {}
        if (checkpoint_root / "checkpoint-index.json").is_file():
            existing = {
                str(row["request_id"]): row
                for row in load_checkpoint_rows(checkpoint_root, expected_binding=binding)
            }
        plan_by_group: dict[str, list[JsonDict]] = {}
        for cell in plan:
            plan_by_group.setdefault(str(cell["group_id"]), []).append(cell)
        group_order = list(plan_by_group)
        phase_started = time.monotonic()
        progress(run_started, "native_capture", "before_forward_loop", planned=len(plan))
        for group_index, group_id in enumerate(group_order, start=1):
            group_rows: list[JsonDict] = []
            for cell in plan_by_group[group_id]:
                request_id = str(cell["request_id"])
                if request_id in existing:
                    group_rows.append(existing[request_id])
                    continue
                stored = {key: deepcopy(value) for key, value in cell.items() if key != "prompt"}
                if cell["eligible"] is False:
                    group_rows.append(stored)
                    existing[request_id] = stored
                    continue
                if time.monotonic() - live_started >= MAX_LIVE_SECONDS:
                    stored.update(
                        {
                            "disposition": "unstarted",
                            "attempted": False,
                            "error": "live_work_cap_reached",
                            "generated_tokens": 0,
                        }
                    )
                    group_rows.append(stored)
                    existing[request_id] = stored
                    continue
                events.append(_event(request_id, "forward_calls", "attempted"))
                try:
                    scored = window_pilot._score_native_prompt(runner, cell)
                except BaseException as exc:  # noqa: BLE001 - preserve the failed call.
                    events.append(_event(request_id, "forward_calls", "failed"))
                    stored.update(
                        {
                            "disposition": "failed",
                            "attempted": True,
                            "error": f"{type(exc).__name__}:{exc}",
                            "generated_tokens": 0,
                        }
                    )
                    row = stored
                else:
                    events.append(_event(request_id, "forward_calls", "completed"))
                    row = {**scored, "attempted": True, "error": None}
                    durations["tokenize"] += float(row["tokenize_s"])
                    durations["prefill"] += float(row["prefill_s"])
                    durations["readout"] += float(row["readout_s"])
                group_rows.append(row)
                existing[request_id] = row
            write_group_checkpoint(
                checkpoint_root,
                binding=binding,
                group_id=group_id,
                rows=group_rows,
            )
            progress(
                run_started,
                "native_capture",
                "group_complete",
                completed=group_index,
                planned=len(group_order),
                calls=len(existing),
            )
        rows = [
            existing.get(
                str(cell["request_id"]),
                {
                    **{key: deepcopy(value) for key, value in cell.items() if key != "prompt"},
                    "disposition": "unstarted",
                    "attempted": False,
                    "error": "missing_after_loop",
                },
            )
            for cell in plan
        ]
        reduction = reduce_capture(plan, rows)
        inference_ok = reduction["all_eligible_calls_valid"] is True
        spans.append(
            _phase_span(
                "native_capture", phase_started, run_started, len(rows), "groups_checkpointed"
            )
        )
        progress(
            run_started,
            "native_capture",
            "after_forward_loop",
            complete=reduction["sample_size_budget"]["calls"]["complete"],
            failed=reduction["sample_size_budget"]["calls"]["failed"],
        )
    finally:
        progress(run_started, "model_unload", "before_model_unload")
        runner.close()
        gc.collect()
        progress(run_started, "model_unload", "after_model_unload")
        phase = lease.document.get("phase")
        if phase in {"resident", "inferencing"}:
            lease.transition("unloading")
            after_mb = parity._gpu_memory(int(gpu["index"]))
            unload_observed = after_mb <= int(gpu["memory_used_mb"]) + 1_024
            lease.transition(
                "validating",
                vram_mb=after_mb,
                exit_code=0 if inference_ok else 1,
                unload_observed=unload_observed,
            )
            lease.transition(
                "terminal_complete" if inference_ok and unload_observed else "terminal_blocked"
            )
        elif phase in {"preflight", "admitted", "loading"}:
            lease.transition("terminal_blocked")
        release = lease.release()
    model_identity["gpu_lease"] = {**owner, "release": release}

    phase_started = time.monotonic()
    reduction_started = time.monotonic()
    progress(run_started, "reduction", "before_benchmark", rows=len(rows))
    raw_dir = root / RAW_DIR
    shards = write_raw_shards(raw_dir, plan=plan, rows=rows)
    reduction = reduce_capture(plan, rows)
    durations["reduction"] = time.monotonic() - reduction_started
    spans.append(
        _phase_span("reduction", phase_started, run_started, len(rows), "raw_shards_sealed")
    )
    progress(
        run_started, "reduction", "after_benchmark", complete=reduction["capture_complete_score"]
    )

    private_root = Path(tempfile.mkdtemp(prefix="carnot-exp7493-", dir="/tmp"))
    commands = build_command_plan(root, VALIDATION_MANIFEST, private_root)
    plan_errors = validate_command_plan(root, VALIDATION_MANIFEST, commands)
    if plan_errors:
        raise WindowCaptureError("validation_plan_invalid:" + ",".join(plan_errors))
    validation_started = time.monotonic()
    phase_started = time.monotonic()
    progress(run_started, "affected_validation", "before_subprocesses", planned=len(commands))
    affected = run_categorized_commands(
        root,
        [PlannedCommand(command, "validity", True) for command in commands],
        log_dir=raw_dir / "validation/affected",
        heartbeat_s=60,
    )
    affected_reduction = reduce_affected_receipts(root, VALIDATION_MANIFEST, affected)
    spans.append(
        _phase_span(
            "affected_validation", phase_started, run_started, len(affected), "checks_terminal"
        )
    )
    progress(
        run_started,
        "affected_validation",
        "after_subprocesses",
        passed=affected_reduction["passed"],
    )
    if not affected_reduction["passed"]:
        raise WindowCaptureError("affected_validation_failed")

    device_identity = {
        "cpu": platform.processor() or platform.machine(),
        "platform": platform.platform(),
        "cuda_inventory": context["gpu_inventory"],
        "selected_cuda": gpu,
        "owned_pid": os.getpid(),
    }
    candidate = _build_artifact(
        plan=plan,
        rows=rows,
        raw_logit_root=RAW_DIR.as_posix(),
        raw_logit_shards=shards,
        events=events,
        preconditions=preconditions,
        source_hashes=source_hashes,
        model_identity=model_identity,
        device_identity=device_identity,
        validation_receipts=affected,
        validation_passed=True,
        phase_spans=spans,
        duration_breakdown=durations,
        started_at=started_at,
        ended_at=utc_now(),
        started_ns=started_ns,
        ended_ns=time.monotonic_ns(),
    )
    candidate_path = raw_dir / "measured_terminal_candidate.json"
    atomic_json(candidate_path, candidate)

    terminal_plan = _terminal_commands(root, candidate_path)
    phase_started = time.monotonic()
    progress(run_started, "terminal_validation", "before_subprocesses", planned=len(terminal_plan))
    terminal = run_categorized_commands(
        root,
        terminal_plan,
        log_dir=raw_dir / "validation/terminal",
        heartbeat_s=60,
    )
    spans.append(
        _phase_span(
            "terminal_validation", phase_started, run_started, len(terminal), "readers_terminal"
        )
    )
    terminal_passed = all(row.get("passed") is True for row in terminal)
    critical = any("CRITICAL" in str(row.get("output_tail") or "") for row in terminal)
    progress(
        run_started,
        "terminal_validation",
        "after_subprocesses",
        passed=terminal_passed,
        critical=critical,
    )
    if not terminal_passed or critical:
        raise WindowCaptureError("terminal_validation_failed")
    durations["validation"] = time.monotonic() - validation_started

    final = _build_artifact(
        plan=plan,
        rows=rows,
        raw_logit_root=RAW_DIR.as_posix(),
        raw_logit_shards=shards,
        events=events,
        preconditions=preconditions,
        source_hashes=source_hashes,
        model_identity=model_identity,
        device_identity=device_identity,
        validation_receipts=[*affected, *terminal],
        validation_passed=True,
        phase_spans=spans,
        duration_breakdown=durations,
        started_at=started_at,
        ended_at=utc_now(),
        started_ns=started_ns,
        ended_ns=time.monotonic_ns(),
    )
    errors = validate_artifact(final, root=root, require_validation=True)
    if errors:
        raise WindowCaptureError("terminal_artifact_invalid:" + ",".join(errors))
    progress(run_started, "publish", "before_atomic_publish", path=output)
    atomic_json(root / output, final)
    progress(run_started, "publish", "after_atomic_publish", verdict=final["honest_verdict"])
    return final


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:  # pragma: no cover
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", required=True)
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--output", type=Path, default=RESULT_PATH)
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--independent-reduce", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover
    """Run the live capture or one read-only fresh-process reader."""

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
        print(json.dumps({"mode": "independent_reduce", **reduced}, sort_keys=True), flush=True)
        return int(reduced.get("passed") is not True)
    result = run_experiment(root, args.date, output=args.output)
    print(
        json.dumps(
            {
                "artifact": str(root / args.output),
                "honest_verdict": result["honest_verdict"],
                "window_fit_ready_score": result["window_fit_ready_score"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0
