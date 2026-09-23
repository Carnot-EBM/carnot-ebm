"""Qualify a resumable native option-logit capture runner without loading a model.

The task freezes complete Exp7533 prompts and tests collection with a private
scripted transport. Later experiments can pass Exp7535 ``NativeOptionRunner``
through the same small interface after they acquire their own GPU lease.

Spec refs: REQ-VERIFY-7548 and SCENARIO-VERIFY-7548-*.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from copy import deepcopy
from datetime import UTC, datetime
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import time
from typing import Any, Protocol

from carnot import experiment_7160_v631_qwen38_lease_diagnosis as gpu_diagnosis
from carnot import experiment_7533_v659_tool_protocol as tool_protocol
from carnot import experiment_7535_v659_native_pilot as native_pilot
from carnot.experiment_7358_v646_validation_contract import (
    AffectedManifest,
    PlannedCommand,
    build_command_plan,
    reduce_affected_receipts,
    run_categorized_commands,
    validate_command_plan,
)
from carnot.reporting import experiment_7303_validation_scope as validation_scope
from carnot.reporting.current_work_receipt import atomic_json


JsonDict = dict[str, Any]
RUN_DATE = "20260923"
MILESTONE = "2026.09.660"
VERSION = "v660"
EXPERIMENT_ID = "exp7548-capture-runner"
SCHEMA = "carnot.exp7548.v660.capture_runner.v1"
MODEL_SPECS: list[JsonDict] = []
N_CTX = 4096
VALIDATION_RESERVE_S = 900.0
CAPTURE_LIMIT_S = 4200.0
MAX_GPU_OBSERVATION_WAIT_S = 300.0
GPU_IDLE_MAX_USED_MB = native_pilot.GPU_IDLE_MAX_USED_MB
CONDITIONS = ("original", "absent", "mismatched")
OPTION_ORDERS = tool_protocol.OPTION_ORDERS
CAPTURE_ROLE_COUNTS = {"fit": 160, "tune": 40, "policy": 40, "test": 80}
ROLE_COUNTS = {**CAPTURE_ROLE_COUNTS, "development": 12}
UPSTREAM_ROLE_COUNTS = {**CAPTURE_ROLE_COUNTS, "online": 160}
TOTAL_GROUPS = sum(ROLE_COUNTS.values())
TOTAL_FORWARDS = TOTAL_GROUPS * 6
ZERO_INVOCATION_COUNTS = {
    "model_loads_attempted": 0,
    "model_loads_completed": 0,
    "model_loads_failed": 0,
    "model_loads_cancelled": 0,
    "model_loads_in_flight": 0,
    "forward_calls_attempted": 0,
    "forward_calls_completed": 0,
    "forward_calls_failed": 0,
    "forward_calls_cancelled": 0,
    "forward_calls_in_flight": 0,
    "generation_calls_attempted": 0,
    "generation_calls_completed": 0,
    "generation_calls_failed": 0,
    "generation_calls_cancelled": 0,
    "generation_calls_in_flight": 0,
}

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_PATH = Path("results/experiment_7548_v660_capture_runner.json")
RAW_DIR = Path("results/raw/experiment_7548_v660_capture_runner")
CANDIDATE_PATH = RAW_DIR / "measured_terminal_candidate.json"
SCHEDULE_PATH = RAW_DIR / "capture_schedule.jsonl"
MODULE_PATH = Path("python/carnot/experiment_7548_v660_capture_runner.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_7548_v660_capture_runner.py")
TEST_PATH = Path("tests/python/test_experiment_7548_v660_capture_runner.py")
SPEC_PATH = Path("openspec/capabilities/verification/spec.md")
UPSTREAM_PROTOCOL_PATH = Path("results/experiment_7533_v659_tool_protocol.json")
UPSTREAM_PILOT_PATH = Path("results/experiment_7535_v659_native_pilot.json")
LEASE_RUNTIME_DIR = gpu_diagnosis.LEASE_RUNTIME_DIR

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


class CaptureRunnerError(ValueError):
    """Reject schedule, checkpoint, transport, or artifact drift."""


class NativeTransport(Protocol):
    """The subset of Exp7535 NativeOptionRunner used by capture tasks."""

    def score(self, request: JsonDict, owner: JsonDict) -> JsonDict:
        """Return one final-token native option-logit row."""


def utc_now() -> str:
    """Return a timezone-aware boundary for an observation or publication."""

    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def progress(started: float, phase: str, event: str, **details: object) -> None:
    """Flush one truthful phase or operation boundary with monotonic elapsed time."""

    suffix = " ".join(f"{key}={value}" for key, value in sorted(details.items()))
    print(
        f"[exp7548] phase={phase} event={event} elapsed_s={time.monotonic() - started:.3f}"
        + (f" {suffix}" if suffix else ""),
        flush=True,
    )


def sha256_text(value: str) -> str:
    """Hash exact UTF-8 bytes so prompt or identity edits remain detectable."""

    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def canonical_hash(value: object) -> str:
    """Hash stable JSON bytes so reordered or removed evidence changes identity."""

    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return sha256_text(payload)


def sha256_file(path: Path) -> str:
    """Hash exact file bytes without loading a large sidecar into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _path_label(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _jsonl_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(
        json.dumps(dict(row), sort_keys=True, separators=(",", ":"), default=str).encode() + b"\n"
        for row in rows
    )


def write_jsonl_sidecar(
    path: Path, rows: Sequence[Mapping[str, Any]], *, root: Path = REPO_ROOT
) -> JsonDict:
    """Reuse Exp7533 atomic bytes and return a compact content receipt."""

    payload = _jsonl_bytes(rows)
    tool_protocol._atomic_bytes(path, payload)
    return {
        "path": _path_label(path, root),
        "sha256": "sha256:" + hashlib.sha256(payload).hexdigest(),
        "bytes": len(payload),
        "rows": len(rows),
    }


def _receipt_path(receipt: Mapping[str, Any], root: Path) -> Path:
    path = Path(str(receipt.get("path") or ""))
    return path if path.is_absolute() else root / path


def _receipt_valid(receipt: Mapping[str, Any], root: Path) -> bool:
    path = _receipt_path(receipt, root)
    return bool(
        path.is_file()
        and path.stat().st_size == receipt.get("bytes")
        and sha256_file(path) == receipt.get("sha256")
        and path.stat().st_size < 20 * 1024 * 1024
    )


def _read_jsonl_receipt(receipt: Mapping[str, Any], root: Path) -> list[JsonDict]:
    if not _receipt_valid(receipt, root):
        raise CaptureRunnerError("sidecar_receipt_invalid")
    rows: list[JsonDict] = []
    with _receipt_path(receipt, root).open(encoding="utf-8") as handle:
        for line in handle:
            value = json.loads(line)
            if not isinstance(value, Mapping):
                raise CaptureRunnerError("sidecar_row_not_object")
            rows.append(dict(value))
    if len(rows) != receipt.get("rows"):
        raise CaptureRunnerError("sidecar_row_count_mismatch")
    return rows


def gate(
    check: str,
    category: str,
    expected: object,
    observed: object,
    op: str,
    passed: bool,
    principle: str,
    *,
    upstream: str,
    field: str,
    path: str | None = None,
) -> JsonDict:
    """Retain exact gate operands and the failure that the gate prevents."""

    row: JsonDict = {
        "check": check,
        "category": category,
        "condition": f"{field} {op} expected",
        "expected": deepcopy(expected),
        "observed": deepcopy(observed),
        "op": op,
        "passed": bool(passed),
        "principle": principle,
        "upstream": upstream,
        "field": field,
    }
    if path is not None:
        row["path"] = path
    return row


def gate_check_summary(gates: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Expose every failed gate and keep its first exact routing operand."""

    failed = [deepcopy(dict(row)) for row in gates if row.get("passed") is not True]
    return {
        "passed": not failed,
        "failed_count": len(failed),
        "failed_checks": [str(row.get("check")) for row in failed],
        "first_failure": failed[0] if failed else None,
    }


def _group_identity(row: Mapping[str, Any]) -> JsonDict:
    return {
        "role": str(row.get("role")),
        "component_hash": str(row.get("component_hash")),
        "donor_component_hash": str(row.get("donor_component_hash")),
        "tool_type": str(row.get("tool_type")),
    }


def _group_hash(row: Mapping[str, Any]) -> str:
    return canonical_hash(_group_identity(row))


def _request_id(row: Mapping[str, Any], condition: str, order_index: int) -> str:
    return sha256_text(f"{row.get('role')}:{row.get('component_hash')}:{condition}:{order_index}")


def compile_capture_schedule(
    groups: Sequence[Mapping[str, Any]],
    *,
    expected_role_counts: Mapping[str, int] = ROLE_COUNTS,
) -> list[JsonDict]:
    """Expand each sealed group into six stable zero-generation requests."""

    schedule: list[JsonDict] = []
    role_indexes: Counter[str] = Counter()
    for source_group in groups:
        group = deepcopy(dict(source_group))
        role = str(group.get("role"))
        group_index = role_indexes[role]
        role_indexes[role] += 1
        requests = group.get("requests")
        if not isinstance(requests, list):
            raise CaptureRunnerError("group_requests_missing")
        for request in requests:
            if not isinstance(request, Mapping):
                raise CaptureRunnerError("request_not_object")
            condition = str(request.get("condition"))
            order = list(request.get("option_order") or [])
            if condition not in CONDITIONS or tuple(order) not in OPTION_ORDERS:
                raise CaptureRunnerError("request_semantics_invalid")
            order_index = OPTION_ORDERS.index(tuple(order))
            prompt = str(request.get("prompt") or "")
            row = {
                **deepcopy(dict(request)),
                "schedule_index": len(schedule),
                "role_group_index": group_index,
                "role": role,
                "component_hash": group.get("component_hash"),
                "donor_component_hash": group.get("donor_component_hash"),
                "tool_type": group.get("tool_type"),
                "label_scope": group.get("label_scope", "original_source_only"),
                "condition": condition,
                "option_order": order,
                "prompt": prompt,
                "prompt_sha256": sha256_text(prompt),
                "prompt_utf8_bytes": len(prompt.encode("utf-8")),
                "n_ctx": N_CTX,
                "readout_kind": "option_logits",
                "generated_token_budget": 0,
                "group_hash": _group_hash(group),
                "request_id": _request_id(group, condition, order_index),
            }
            schedule.append(row)
    reduction = validate_capture_schedule(schedule, expected_role_counts=expected_role_counts)
    if reduction["passed"] is not True:
        raise CaptureRunnerError("capture_schedule_invalid:" + ",".join(reduction["failed_checks"]))
    return schedule


def validate_capture_schedule(
    schedule: Sequence[Mapping[str, Any]],
    *,
    expected_role_counts: Mapping[str, int] = ROLE_COUNTS,
) -> JsonDict:
    """Recompute exact identities, cell products, counts, and role hashes."""

    failures: set[str] = set()
    by_group: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    request_ids: set[str] = set()
    components: set[str] = set()
    schedule_indices: list[int] = []
    for row in schedule:
        group_hash = str(row.get("group_hash") or "")
        by_group[group_hash].append(row)
        component = str(row.get("component_hash") or "")
        components.add(component)
        donor = str(row.get("donor_component_hash") or "")
        if not component.startswith("sha256:") or not donor.startswith("sha256:"):
            failures.add("donor_identity")
        if component == donor:
            failures.add("donor_identity")
        if group_hash != _group_hash(row):
            failures.add("group_identity")
        condition = str(row.get("condition"))
        order_value = row.get("option_order")
        order = list(order_value) if isinstance(order_value, list) else []
        order_tuple = tuple(order)
        if condition not in CONDITIONS or order_tuple not in OPTION_ORDERS:
            failures.add("request_identity")
            order_index = -1
        else:
            order_index = OPTION_ORDERS.index(order_tuple)
        if row.get("request_id") != _request_id(row, condition, order_index):
            failures.add("request_identity")
        request_id = str(row.get("request_id") or "")
        if request_id in request_ids:
            failures.add("request_identity")
        request_ids.add(request_id)
        schedule_index = row.get("schedule_index")
        if not isinstance(schedule_index, int) or isinstance(schedule_index, bool):
            failures.add("request_identity")
        else:
            schedule_indices.append(schedule_index)
        prompt = row.get("prompt")
        token_count = row.get("prompt_token_count")
        if (
            not isinstance(prompt, str)
            or not isinstance(token_count, int)
            or isinstance(token_count, bool)
            or token_count <= 0
            or token_count > N_CTX
            or row.get("prompt_sha256") != sha256_text(prompt)
            or row.get("prompt_utf8_bytes") != len(prompt.encode("utf-8"))
        ):
            failures.add("prompt_token_custody")
        if (
            row.get("n_ctx") != N_CTX
            or row.get("readout_kind") != "option_logits"
            or row.get("generated_token_budget") != 0
        ):
            failures.add("request_identity")

    if len(schedule_indices) != len(set(schedule_indices)) or schedule_indices != sorted(
        schedule_indices
    ):
        failures.add("request_identity")

    role_groups: dict[str, list[JsonDict]] = defaultdict(list)
    for group_hash, rows in by_group.items():
        cells = {(str(row.get("condition")), tuple(row.get("option_order") or [])) for row in rows}
        expected_cells = {(condition, order) for condition in CONDITIONS for order in OPTION_ORDERS}
        if len(rows) != 6 or cells != expected_cells:
            failures.add("group_cell_product")
        identities = {_group_hash(row) for row in rows}
        if identities != {group_hash}:
            failures.add("group_identity")
        sample = rows[0]
        role_groups[str(sample.get("role"))].append(
            {**_group_identity(sample), "group_hash": group_hash}
        )
    group_counts = {role: len(role_groups.get(role, [])) for role in expected_role_counts}
    if group_counts != dict(expected_role_counts):
        failures.add("role_group_counts")
    if len(components) != len(by_group):
        failures.add("component_role_overlap")
    role_hashes = {
        role: canonical_hash(sorted(groups, key=lambda row: str(row["component_hash"])))
        for role, groups in sorted(role_groups.items())
    }
    return {
        "passed": not failures,
        "failed_checks": sorted(failures),
        "group_counts": group_counts,
        "complete_group_count": len(by_group),
        "forward_count": len(schedule),
        "role_schedule_hashes": role_hashes,
    }


def _row_identity_failures(row: Mapping[str, Any], expected: Mapping[str, Any]) -> set[str]:
    failures: set[str] = set()
    if row.get("donor_component_hash") != expected.get("donor_component_hash"):
        failures.add("donor_identity")
    if row.get("group_hash") != expected.get("group_hash"):
        failures.add("group_identity")
    request_fields = (
        "request_id",
        "schedule_index",
        "role",
        "component_hash",
        "condition",
        "option_order",
        "prompt",
        "prompt_sha256",
    )
    if any(row.get(field) != expected.get(field) for field in request_fields):
        failures.add("request_identity")
    prompt = row.get("prompt")
    ids = row.get("prompt_token_ids")
    count = row.get("prompt_token_count")
    if (
        not isinstance(prompt, str)
        or not isinstance(ids, list)
        or not isinstance(count, int)
        or isinstance(count, bool)
        or count != expected.get("prompt_token_count")
        or len(ids) != count
        or count > N_CTX
        or row.get("prompt_sha256") != sha256_text(prompt)
        or row.get("prompt_utf8_bytes") != len(prompt.encode("utf-8"))
    ):
        failures.add("prompt_token_custody")
    native_failures = native_pilot._row_checks(row, expected)
    for failure in native_failures:
        failures.add("request_identity" if failure == "schedule_identity" else failure)
    logits = row.get("full_logits_by_option_id")
    if not isinstance(logits, Mapping):
        failures.add("finite_logits")
    else:
        try:
            values = [float(value) for value in logits.values()]
            if len(values) != 2 or any(not math.isfinite(value) for value in values):
                failures.add("finite_logits")
        except (TypeError, ValueError):
            failures.add("finite_logits")
    return failures


def reduce_capture_rows(
    rows: Sequence[Mapping[str, Any]],
    schedule: Sequence[Mapping[str, Any]],
    *,
    expected_role_counts: Mapping[str, int] = ROLE_COUNTS,
) -> JsonDict:
    """Independently reduce native rows against every sealed request identity."""

    schedule_reduction = validate_capture_schedule(
        schedule, expected_role_counts=expected_role_counts
    )
    failures = set(schedule_reduction["failed_checks"])
    expected_by_id = {str(row.get("request_id")): row for row in schedule}
    observed_by_id: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        request_id = str(row.get("request_id") or "")
        if request_id in observed_by_id:
            failures.add("request_identity")
        observed_by_id[request_id] = row
    if set(observed_by_id) != set(expected_by_id):
        failures.add("request_identity")
        failures.add("partial_group")

    condition_values: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    complete = 0
    for request_id, expected in expected_by_id.items():
        row = observed_by_id.get(request_id)
        if row is None:
            continue
        failures.update(_row_identity_failures(row, expected))
        if row.get("disposition") != "complete" or row.get("error") is not None:
            failures.add("complete_forward_custody")
            continue
        complete += 1
        probabilities = row.get("probabilities_by_option_id")
        try:
            probability = float(dict(probabilities or {})["contains_unsupported"])
            if not math.isfinite(probability):
                raise ValueError
            condition_values[str(expected["group_hash"])][str(expected["condition"])].append(
                probability
            )
        except (KeyError, TypeError, ValueError):
            failures.add("finite_probabilities")

    comparative_rows: list[JsonDict] = []
    for group_hash, values in sorted(condition_values.items()):
        if set(values) != set(CONDITIONS) or any(len(cells) != 2 for cells in values.values()):
            failures.add("partial_group")
            continue
        absolute = {
            condition: sum(cells) / len(cells) for condition, cells in sorted(values.items())
        }
        comparative_rows.append(
            {
                "group_hash": group_hash,
                "absolute_condition_probabilities": absolute,
                "original_minus_absent": absolute["original"] - absolute["absent"],
                "original_minus_mismatched": absolute["original"] - absolute["mismatched"],
                "disposition": "complete_transport_qualification",
                "positive_claim": False,
            }
        )
    if len(comparative_rows) != schedule_reduction["complete_group_count"]:
        failures.add("partial_group")
    return {
        "passed": not failures,
        "failed_checks": sorted(failures),
        "planned_forward_count": len(schedule),
        "attempted_forward_count": len(rows),
        "complete_forward_count": complete,
        "complete_group_count": len(comparative_rows),
        "generated_token_count": sum(int(row.get("generated_tokens") or 0) for row in rows),
        "comparative_rows": comparative_rows,
        "role_schedule_hashes": schedule_reduction["role_schedule_hashes"],
        "benefit_gate_applied": False,
    }


def independent_reduce(value: Mapping[str, Any], *, root: Path = REPO_ROOT) -> JsonDict:
    """Reload schedule and scripted rows without trusting artifact headlines."""

    if (
        value.get("verdict_class") == "blocked"
        and value.get("capture_runner_ready_score") == 0
        and value.get("capture_schedule") is None
    ):
        budget = value.get("sample_size_budget")
        passed = bool(
            value.get("rows") == []
            and value.get("model_invoked") is False
            and isinstance(budget, Mapping)
            and budget.get("unstarted") == TOTAL_GROUPS
        )
        return {"passed": passed, "mode": "blocked_no_measurement", "row_count": 0}
    try:
        capture_receipt = value.get("capture_schedule")
        rows_receipt = value.get("qualification_rows")
        qualification_receipt = value.get("qualification_schedule") or capture_receipt
        if not all(
            isinstance(receipt, Mapping)
            for receipt in (capture_receipt, rows_receipt, qualification_receipt)
        ):
            raise CaptureRunnerError("sidecar_receipt_invalid")
        capture = _read_jsonl_receipt(dict(capture_receipt), root)
        qualification = _read_jsonl_receipt(dict(qualification_receipt), root)
        rows = _read_jsonl_receipt(dict(rows_receipt), root)
        capture_counts = value.get("role_counts") or value.get("qualification_role_counts")
        qualification_counts = value.get("qualification_role_counts") or capture_counts
        if not isinstance(capture_counts, Mapping) or not isinstance(qualification_counts, Mapping):
            raise CaptureRunnerError("role_counts_missing")
        capture_reduction = validate_capture_schedule(
            capture, expected_role_counts=dict(capture_counts)
        )
        row_reduction = reduce_capture_rows(
            rows, qualification, expected_role_counts=dict(qualification_counts)
        )
    except (CaptureRunnerError, OSError, json.JSONDecodeError):
        return {"passed": False, "error": "sidecar_receipt_invalid"}
    return {
        "passed": capture_reduction["passed"] is True and row_reduction["passed"] is True,
        "schedule_count": len(capture),
        "qualification_row_count": len(rows),
        "complete_group_count": row_reduction["complete_group_count"],
        "capture_schedule_sha256": canonical_hash(capture),
        "qualification_reduction_sha256": canonical_hash(row_reduction),
    }


class ResumableCaptureRunner:
    """Checkpoint only six-row groups produced by one caller-owned transport."""

    def __init__(
        self,
        *,
        schedule: Sequence[Mapping[str, Any]],
        transport: NativeTransport,
        checkpoint_path: Path,
        roles: Sequence[str],
        validation_reserve_s: float = VALIDATION_RESERVE_S,
        hard_limit_s: float = CAPTURE_LIMIT_S,
        owner: Mapping[str, Any],
    ) -> None:
        self.schedule = [deepcopy(dict(row)) for row in schedule]
        self.transport = transport
        self.checkpoint_path = checkpoint_path
        self.roles = tuple(str(role) for role in roles)
        self.validation_reserve_s = float(validation_reserve_s)
        self.hard_limit_s = float(hard_limit_s)
        self.owner = deepcopy(dict(owner))
        if self.validation_reserve_s < 0 or self.hard_limit_s <= self.validation_reserve_s:
            raise CaptureRunnerError("invalid_capture_deadline")
        available_roles = {str(row.get("role")) for row in self.schedule}
        if not self.roles or not set(self.roles) <= available_roles:
            raise CaptureRunnerError("bounded_roles_invalid")

    @property
    def selected_schedule(self) -> list[JsonDict]:
        return [row for row in self.schedule if row.get("role") in self.roles]

    def _load_checkpoint(self) -> list[JsonDict]:
        if not self.checkpoint_path.exists():
            return []
        try:
            value = json.loads(self.checkpoint_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise CaptureRunnerError("checkpoint_invalid") from exc
        if not isinstance(value, Mapping):
            raise CaptureRunnerError("checkpoint_invalid")
        if value.get("schedule_sha256") != canonical_hash(self.selected_schedule):
            raise CaptureRunnerError("checkpoint_schedule_mismatch")
        rows = value.get("rows")
        if not isinstance(rows, list) or any(not isinstance(row, Mapping) for row in rows):
            raise CaptureRunnerError("checkpoint_rows_invalid")
        retained = [deepcopy(dict(row)) for row in rows]
        if not retained:
            return []
        group_hashes = {str(row.get("group_hash")) for row in retained}
        expected = [
            row for row in self.selected_schedule if str(row.get("group_hash")) in group_hashes
        ]
        counts = Counter(str(row.get("role")) for row in expected[::6])
        reduced = reduce_capture_rows(retained, expected, expected_role_counts=dict(counts))
        if reduced["passed"] is not True:
            raise CaptureRunnerError("checkpoint_partial_or_changed")
        return retained

    def _save_checkpoint(self, rows: Sequence[Mapping[str, Any]]) -> None:
        atomic_json(
            self.checkpoint_path,
            {
                "schema": "carnot.exp7548.capture_checkpoint.v1",
                "schedule_sha256": canonical_hash(self.selected_schedule),
                "completed_group_hashes": sorted({str(row.get("group_hash")) for row in rows}),
                "rows": [deepcopy(dict(row)) for row in rows],
            },
        )

    def run(self) -> JsonDict:
        """Run whole groups until completion, an error, or the validation reserve."""

        started = time.monotonic()
        retained = self._load_checkpoint()
        resumed_groups = len({str(row["group_hash"]) for row in retained})
        complete_hashes = {str(row["group_hash"]) for row in retained}
        groups: dict[str, list[JsonDict]] = defaultdict(list)
        for row in self.selected_schedule:
            groups[str(row["group_hash"])].append(row)
        partial_discarded = 0
        error: str | None = None
        for group_index, (group_hash, requests) in enumerate(groups.items()):
            if group_hash in complete_hashes:
                continue
            elapsed = time.monotonic() - started
            if elapsed + self.validation_reserve_s >= self.hard_limit_s:
                error = "validation_reserve_reached"
                break
            group_rows: list[JsonDict] = []
            for request in requests:
                progress(
                    started,
                    "scripted_or_owned_capture",
                    "before_forward",
                    group=group_index + 1,
                    request=request["request_id"],
                )
                try:
                    row = native_pilot._call_with_heartbeats(
                        lambda request=request: self.transport.score(request, self.owner),
                        started=started,
                        phase="scripted_or_owned_capture",
                        pending=str(request["request_id"]),
                    )
                except BaseException as exc:  # noqa: BLE001 - retain exact owned failure.
                    error = f"{type(exc).__name__}:{exc}"
                    partial_discarded += int(bool(group_rows))
                    group_rows = []
                    break
                progress(
                    started,
                    "scripted_or_owned_capture",
                    "after_forward",
                    completed_units=len(retained) + len(group_rows) + 1,
                )
                group_rows.append(deepcopy(dict(row)))
            if error is not None:
                break
            group_counts = {str(requests[0]["role"]): 1}
            reduction = reduce_capture_rows(group_rows, requests, expected_role_counts=group_counts)
            if reduction["passed"] is not True:
                error = "group_custody_failed:" + ",".join(reduction["failed_checks"])
                partial_discarded += 1
                break
            retained.extend(group_rows)
            complete_hashes.add(group_hash)
            self._save_checkpoint(retained)
            progress(
                started,
                "scripted_or_owned_capture",
                "group_checkpointed",
                completed_groups=len(complete_hashes),
            )
        status = "complete" if len(complete_hashes) == len(groups) else "partial"
        return {
            "status": status,
            "error": error,
            "rows": retained,
            "resumed_group_count": resumed_groups,
            "completed_group_count": len(complete_hashes),
            "planned_group_count": len(groups),
            "partial_group_discarded": partial_discarded,
            "validation_reserve_s": self.validation_reserve_s,
            "hard_limit_s": self.hard_limit_s,
            "duration_s": time.monotonic() - started,
        }


def _fixture_group(role: str, index: int) -> JsonDict:
    component = sha256_text(f"fixture-component:{role}:{index}")
    donor = sha256_text(f"fixture-donor:{role}:{index}")
    requests: list[JsonDict] = []
    for condition in CONDITIONS:
        for order in OPTION_ORDERS:
            prompt = f"fixture {role} {index} {condition} {'/'.join(order)}"
            requests.append(
                {
                    "condition": condition,
                    "option_order": list(order),
                    "prompt": prompt,
                    "prompt_sha256": sha256_text(prompt),
                    "prompt_token_count": len(prompt.split()) + 1,
                    "readout_kind": "option_logits",
                    "generated_token_budget": 0,
                }
            )
    return {
        "component_hash": component,
        "donor_component_hash": donor,
        "role": role,
        "tool_type": "grep",
        "label_scope": "original_source_only",
        "requests": requests,
    }


def _scripted_native_row(request: Mapping[str, Any], index: int) -> JsonDict:
    order = list(request["option_order"])
    logits = {order[0]: 1.5 + index / 100.0, order[1]: -0.5}
    probabilities = native_pilot._softmax_pair(logits, order)
    count = int(request["prompt_token_count"])
    return {
        **deepcopy(dict(request)),
        "call_id": f"private-scripted-{index}",
        "disposition": "complete",
        "prompt_utf8_bytes": len(str(request["prompt"]).encode("utf-8")),
        "prompt_token_ids": list(range(count)),
        "requested_score_position": count - 1,
        "actual_last_evaluated_position": count - 1,
        "display_labels": [" A", " B"],
        "label_token_ids": [101, 102],
        "label_to_option_id": dict(zip((" A", " B"), order, strict=True)),
        "full_logits_by_option_id": logits,
        "probabilities_by_option_id": probabilities,
        "generated_tokens": 0,
        "forward_seconds": 0.001,
        "state_reset": True,
        "kv_state_id": f"scripted-reset-{index}",
        "server_receipt": {
            "transport": "private_scripted_transport",
            "pid": 123,
            "process_start_ticks": 456,
            "gpu_uuid": "GPU-fixture",
            "n_ctx": N_CTX,
            "generated_tokens": 0,
        },
        "error": None,
    }


class _PrivateScriptedTransport:
    def __init__(self, fail_at: int | None = None) -> None:
        self.fail_at = fail_at
        self.calls = 0

    def score(self, request: JsonDict, owner: JsonDict) -> JsonDict:
        del owner
        index = self.calls
        self.calls += 1
        if index == self.fail_at:
            raise RuntimeError("private scripted interruption")
        return _scripted_native_row(request, index)


def run_qualification_controls(output_dir: Path) -> JsonDict:
    """Exercise interruption, resume, reduction, and one-field mutations."""

    output_dir.mkdir(parents=True, exist_ok=True)
    role_counts = {"fit": 2, "development": 1}
    groups = [
        _fixture_group("fit", 0),
        _fixture_group("fit", 1),
        _fixture_group("development", 0),
    ]
    schedule = compile_capture_schedule(groups, expected_role_counts=role_counts)
    private = Path(tempfile.mkdtemp(prefix="resume-", dir=output_dir))
    checkpoint = private / "checkpoint.json"
    first_transport = _PrivateScriptedTransport(fail_at=8)
    first = ResumableCaptureRunner(
        schedule=schedule,
        transport=first_transport,
        checkpoint_path=checkpoint,
        roles=("fit", "development"),
        owner={"pid": 123, "pid_start_ticks": 456},
    ).run()
    second_transport = _PrivateScriptedTransport()
    second = ResumableCaptureRunner(
        schedule=schedule,
        transport=second_transport,
        checkpoint_path=checkpoint,
        roles=("fit", "development"),
        owner={"pid": 123, "pid_start_ticks": 456},
    ).run()
    reduction = reduce_capture_rows(second["rows"], schedule, expected_role_counts=role_counts)

    mutations: dict[str, object] = {
        "donor_component_hash": "sha256:changed",
        "option_order": ["contains_unsupported", "supported"],
        "request_id": "changed-request",
        "prompt_token_count": 999,
        "group_hash": "sha256:changed",
    }
    mutation_failures: dict[str, bool] = {}
    for field, changed in mutations.items():
        rows = deepcopy(second["rows"])
        rows[0][field] = changed
        mutated = reduce_capture_rows(rows, schedule, expected_role_counts=role_counts)
        mutation_failures[field] = mutated["passed"] is False

    schedule_receipt = write_jsonl_sidecar(
        output_dir / "qualification_schedule.jsonl", schedule, root=REPO_ROOT
    )
    rows_receipt = write_jsonl_sidecar(
        output_dir / "qualification_rows.jsonl", second["rows"], root=REPO_ROOT
    )
    resume_parity = bool(
        first["status"] == "partial"
        and first["completed_group_count"] == 1
        and second["status"] == "complete"
        and second["resumed_group_count"] == 1
        and reduction["passed"] is True
    )
    return {
        "passed": resume_parity and all(mutation_failures.values()),
        "resume_parity": resume_parity,
        "first_run": {key: value for key, value in first.items() if key != "rows"},
        "second_run": {key: value for key, value in second.items() if key != "rows"},
        "reduction": reduction,
        "mutation_failures": mutation_failures,
        "qualification_role_counts": role_counts,
        "qualification_schedule": schedule_receipt,
        "qualification_rows": rows_receipt,
        "scripted_transport_calls": first_transport.calls + second_transport.calls,
        "counted_as_current_model_invocations": False,
    }


def _load_object(path: Path) -> JsonDict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(value) if isinstance(value, Mapping) else {}


def _sealed_path(receipt: Mapping[str, Any], root: Path) -> Path:
    path = Path(str(receipt.get("path") or ""))
    return path if path.is_absolute() else root / path


def authenticate_protocol(root: Path) -> tuple[list[JsonDict], JsonDict]:
    """Authenticate Exp7533 manifests and bytes without reading label rows."""

    path = root / UPSTREAM_PROTOCOL_PATH
    artifact = _load_object(path)
    checks: list[JsonDict] = []

    def add(
        check: str,
        expected: object,
        observed: object,
        *,
        field: str,
        passed: bool | None = None,
        path_value: str | None = None,
    ) -> None:
        checks.append(
            gate(
                check,
                "external_precondition",
                expected,
                observed,
                "==",
                observed == expected if passed is None else passed,
                "Changed or missing sealed protocol evidence cannot authorize collection.",
                upstream="exp7533-v659-tool-protocol",
                field=field,
                path=path_value,
            )
        )

    add(
        "protocol_terminal_identity",
        {
            "experiment_id": "exp7533-v659-tool-protocol",
            "tool_protocol_ready_score": 1,
            "verdict_class": "null",
            "flagged_adversarial": False,
        },
        {
            "experiment_id": artifact.get("experiment_id"),
            "tool_protocol_ready_score": artifact.get("tool_protocol_ready_score"),
            "verdict_class": artifact.get("verdict_class"),
            "flagged_adversarial": artifact.get("flagged_adversarial"),
        },
        field="terminal_identity_and_readiness",
        path_value=UPSTREAM_PROTOCOL_PATH.as_posix(),
    )
    role_manifest = artifact.get("role_manifest")
    role_groups = (
        list(role_manifest.get("groups") or []) if isinstance(role_manifest, Mapping) else []
    )
    observed_counts = dict(Counter(str(row.get("role")) for row in role_groups))
    add(
        "exact_upstream_roles",
        UPSTREAM_ROLE_COUNTS,
        observed_counts,
        field="role_manifest.groups.role_counts",
    )
    donor_manifest = artifact.get("donor_manifest")
    donor_groups = (
        list(donor_manifest.get("groups") or []) if isinstance(donor_manifest, Mapping) else []
    )
    role_donors = {
        (str(row.get("role")), str(row.get("component_hash"))): str(row.get("donor_component_hash"))
        for row in donor_groups
    }
    donor_ok = bool(
        len(role_donors) == sum(UPSTREAM_ROLE_COUNTS.values())
        and all(
            role_donors.get((str(row.get("role")), str(row.get("component_hash"))))
            not in {None, str(row.get("component_hash"))}
            for row in role_groups
        )
        and isinstance(donor_manifest, Mapping)
        and donor_manifest.get("altered_sources_have_labels") is False
    )
    add(
        "sealed_donor_manifest",
        True,
        donor_ok,
        field="donor_manifest.distinct_same_role_and_unlabelled",
    )
    tokenizer = artifact.get("tokenizer_identity")
    tokenizer_ok = bool(
        isinstance(tokenizer, Mapping)
        and tokenizer.get("repository") == "unsloth/Qwen3.8-27B-GGUF"
        and tokenizer.get("quantization") == "Q4_K_M"
        and tokenizer.get("n_ctx") == N_CTX
        and tokenizer.get("vocab_only") is True
        and tokenizer.get("model_weights_read") is False
        and str(tokenizer.get("content_sha256", "")).startswith("sha256:")
        and Path(str(tokenizer.get("path") or "")).is_file()
    )
    add(
        "embedded_tokenizer_manifest",
        True,
        tokenizer_ok,
        field="tokenizer_identity",
        path_value=str(dict(tokenizer or {}).get("path")),
    )
    readers = artifact.get("reader_access_contract")
    reader_ok = bool(
        isinstance(readers, Mapping)
        and readers.get("capture") == "predictors_only_no_labels"
        and readers.get("actual_capture_label_access") == []
        and readers.get("actual_fit_reader_access") == []
    )
    add(
        "sealed_label_access",
        True,
        reader_ok,
        field="reader_access_contract.capture",
    )

    sealed = artifact.get("sealed_shards")
    sealed_rows = dict(sealed) if isinstance(sealed, Mapping) else {}
    shard_errors: list[str] = []
    for name, receipt in sorted(sealed_rows.items()):
        if not isinstance(receipt, Mapping):
            shard_errors.append(f"{name}:receipt_invalid")
            continue
        shard_path = _sealed_path(receipt, root)
        if (
            not shard_path.is_file()
            or shard_path.stat().st_size != receipt.get("bytes")
            or sha256_file(shard_path) != receipt.get("sha256")
            or shard_path.stat().st_size >= 20 * 1024 * 1024
        ):
            shard_errors.append(f"{name}:bytes_changed")
    required_shards = {
        "predictor",
        "fit_labels",
        "tune_labels",
        "policy_labels",
        "test_labels",
        "online_release",
        "interventions_000",
        "interventions_001",
    }
    if not required_shards <= set(sealed_rows):
        shard_errors.append("required_shards_missing")
    add(
        "sealed_shard_hashes",
        [],
        shard_errors,
        field="sealed_shards.path_bytes_sha256",
    )
    required_terminal = {
        "declared_entrypoint_cold_replay",
        "independent_reduction",
        "adversarial_verify",
        "verdict_row_consistency_strict",
    }
    passed_terminal = {
        str(row.get("name"))
        for row in artifact.get("validation_receipts") or []
        if isinstance(row, Mapping) and row.get("passed") is True and row.get("exit_code") == 0
    }
    add(
        "qualified_terminal_checks",
        sorted(required_terminal),
        sorted(required_terminal & passed_terminal),
        field="validation_receipts.required_terminal",
    )
    interventions = {
        name: deepcopy(dict(sealed_rows[name]))
        for name in ("interventions_000", "interventions_001")
        if name in sealed_rows and isinstance(sealed_rows[name], Mapping)
    }
    return checks, {
        "artifact": artifact,
        "artifact_sha256": sha256_file(path) if path.is_file() else None,
        "role_counts": observed_counts,
        "excluded_role_hashes": {str(row.get("component_hash")) for row in role_groups},
        "intervention_receipts": interventions,
        "actual_label_rows_read": 0,
    }


INPUT_PATHS = (
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
    Path("python/carnot/experiment_7533_v659_tool_protocol.py"),
    Path("python/carnot/experiment_7535_v659_native_pilot.py"),
    Path("python/carnot/experiment_7160_v631_qwen38_lease_diagnosis.py"),
    Path("python/carnot/gpu_lease_phase_journal.py"),
    Path("openspec/change-proposals/research-roadmap-vNEXT.md"),
    UPSTREAM_PROTOCOL_PATH,
    UPSTREAM_PILOT_PATH,
    SPEC_PATH,
    MODULE_PATH,
    WRAPPER_PATH,
    TEST_PATH,
)


def collect_preconditions(root: Path) -> tuple[list[JsonDict], JsonDict]:
    """Authenticate named inputs, the requirement, and historical disposition."""

    checks: list[JsonDict] = []
    source_hashes: dict[str, str] = {}
    for relative in INPUT_PATHS:
        path = root / relative
        observed = "readable_nonempty_bytes" if path.is_file() and path.stat().st_size else None
        checks.append(
            gate(
                f"required_input:{relative.as_posix()}",
                "external_precondition",
                "readable_nonempty_bytes",
                observed,
                "==",
                observed == "readable_nonempty_bytes",
                "A missing named input must block before schedule qualification.",
                upstream=relative.as_posix(),
                field="path.bytes",
                path=relative.as_posix(),
            )
        )
        if observed is not None:
            source_hashes[relative.as_posix()] = sha256_file(path)
    spec_path = root / SPEC_PATH
    spec_text = spec_path.read_text(encoding="utf-8") if spec_path.is_file() else ""
    checks.append(
        gate(
            "driving_requirement",
            "external_precondition",
            "REQ-VERIFY-7548",
            "REQ-VERIFY-7548" if "REQ-VERIFY-7548" in spec_text else None,
            "==",
            "REQ-VERIFY-7548" in spec_text,
            "Implementation without its requirement cannot become accepted evidence.",
            upstream=SPEC_PATH.as_posix(),
            field="REQ-*",
            path=SPEC_PATH.as_posix(),
        )
    )
    pilot = _load_object(root / UPSTREAM_PILOT_PATH)
    checks.append(
        gate(
            "historical_pilot_disposition",
            "external_precondition",
            "complete_blocked_owned_gpu_available",
            pilot.get("honest_verdict"),
            "==",
            pilot.get("honest_verdict") == "complete_blocked_owned_gpu_available",
            "The changed-capacity observation must retain the literal prior failure.",
            upstream="exp7535-v659-native-pilot",
            field="honest_verdict",
            path=UPSTREAM_PILOT_PATH.as_posix(),
        )
    )
    protocol_checks, protocol_context = authenticate_protocol(root)
    checks.extend(protocol_checks)
    return checks, {
        "source_artifact_hashes": source_hashes,
        "protocol": protocol_context,
        "historical_pilot": pilot,
    }


def _development_groups(
    excluded_hashes: set[str], *, started: float
) -> tuple[list[JsonDict], JsonDict]:  # pragma: no cover - pinned parquet/tokenizer boundary.
    progress(started, "schedule", "before_vocab_only_development_selection")
    panel, native_schedule, tokenizer = native_pilot._load_panel_and_schedule(
        excluded_role_hashes=excluded_hashes,
        started=started,
    )
    progress(
        started,
        "schedule",
        "after_vocab_only_development_selection",
        completed_units=len(panel["groups"]),
    )
    requests_by_component: dict[str, list[JsonDict]] = defaultdict(list)
    for source in native_schedule:
        request = deepcopy(dict(source))
        if request.get("condition") == "donor":
            request["condition"] = "mismatched"
        requests_by_component[str(request["component_hash"])].append(request)
    groups: list[JsonDict] = []
    for source in panel["groups"]:
        component = str(source["component_hash"])
        groups.append(
            {
                "component_hash": component,
                "donor_component_hash": source["donor_component_hash"],
                "role": "development",
                "tool_type": source["tool_type"],
                "label_scope": "development_excluded_no_labels",
                "requests": requests_by_component[component],
            }
        )
    return groups, {
        "panel_sha256": panel["panel_sha256"],
        "group_count": len(groups),
        "labels_read": panel["labels_read"],
        "excluded_exp7533_role_count": panel["excluded_exp7533_role_count"],
        "tokenizer": tokenizer,
    }


def freeze_capture_schedule(
    context: Mapping[str, Any], *, started: float
) -> tuple[list[JsonDict], JsonDict]:  # pragma: no cover - authenticated sidecar boundary.
    """Read exact intervention bytes, omit online groups, and add development."""

    protocol_context = dict(context["protocol"])
    groups: list[JsonDict] = []
    online_excluded = 0
    for name, receipt in sorted(dict(protocol_context["intervention_receipts"]).items()):
        progress(started, "schedule", "before_intervention_read", shard=name)
        shard_rows = _read_jsonl_receipt(dict(receipt), REPO_ROOT)
        progress(started, "schedule", "after_intervention_read", shard=name, rows=len(shard_rows))
        for source in shard_rows:
            role = str(source.get("role"))
            if role == "online":
                online_excluded += 1
                continue
            if role not in CAPTURE_ROLE_COUNTS:
                continue
            groups.append(
                {
                    "component_hash": source.get("component_hash"),
                    "donor_component_hash": source.get("donor_component_hash"),
                    "role": role,
                    "tool_type": source.get("tool_type"),
                    "label_scope": "original_source_only",
                    "requests": deepcopy(list(source.get("requests") or [])),
                }
            )
    development, development_manifest = _development_groups(
        set(protocol_context["excluded_role_hashes"]), started=started
    )
    groups.extend(development)
    schedule = compile_capture_schedule(groups)
    reduction = validate_capture_schedule(schedule)
    if online_excluded != 160 or reduction["passed"] is not True:
        raise CaptureRunnerError("frozen_schedule_boundary_invalid")
    return schedule, {
        "role_counts": reduction["group_counts"],
        "role_schedule_hashes": reduction["role_schedule_hashes"],
        "online_groups_excluded": online_excluded,
        "labels_read": 0,
        "development": development_manifest,
        "forward_count": reduction["forward_count"],
        "n_ctx": N_CTX,
        "conditions": list(CONDITIONS),
        "option_orders": [list(order) for order in OPTION_ORDERS],
        "validation_reserve_s": VALIDATION_RESERVE_S,
    }


def _process_ancestry(pid: int) -> list[JsonDict]:  # pragma: no cover - live procfs boundary.
    """Read a bounded parent chain without sending signals or opening endpoints."""

    rows: list[JsonDict] = []
    current = pid
    seen: set[int] = set()
    for _ in range(8):
        if current <= 0 or current in seen:
            break
        seen.add(current)
        directory = Path("/proc") / str(current)
        try:
            stat = gpu_diagnosis.parse_proc_stat((directory / "stat").read_text(encoding="utf-8"))
            argv = [
                part.decode("utf-8", "replace")
                for part in (directory / "cmdline").read_bytes().split(b"\0")
                if part
            ]
            parent = int(stat["ppid"])
            rows.append(
                {
                    "pid": current,
                    "ppid": parent,
                    "start_time_ticks": int(stat["start_time_ticks"]),
                    "command_sha256": canonical_hash(argv),
                }
            )
            current = parent
        except (OSError, ValueError, KeyError):
            break
    return rows


def reduce_gpu_capacity(
    process_rows: Sequence[Mapping[str, Any]],
    lease_rows: Sequence[Mapping[str, Any]],
    *,
    observed_at: str,
) -> JsonDict:
    """Apply Exp7535 idle admission while retaining independent lease evidence."""

    by_uuid: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in process_rows:
        uuid = str(row.get("gpu_uuid") or "")
        if uuid:
            by_uuid[uuid].append(row)
    devices: list[JsonDict] = []
    idle: list[str] = []
    for uuid, rows in sorted(by_uuid.items()):
        sample = rows[0]
        processes = [row for row in rows if row.get("pid") is not None]
        used = int(sample.get("gpu_memory_used_mb") or 0)
        supported = "RTX 3090" in str(sample.get("gpu_name") or "")
        admissible = supported and not processes and used < GPU_IDLE_MAX_USED_MB
        if admissible:
            idle.append(uuid)
        devices.append(
            {
                "gpu_uuid": uuid,
                "gpu_name": sample.get("gpu_name"),
                "gpu_memory_total_mb": sample.get("gpu_memory_total_mb"),
                "gpu_memory_used_mb": used,
                "gpu_memory_free_mb": sample.get("gpu_memory_free_mb"),
                "gpu_utilization_pct": sample.get("gpu_utilization_pct"),
                "supported": supported,
                "admissible_under_exp7535_rule": admissible,
                "process_count": len(processes),
                "processes": [deepcopy(dict(row)) for row in processes],
            }
        )
    return {
        "observed_at_utc": observed_at,
        "gpu_capacity_observed_score": int(bool(idle)),
        "idle_supported_gpu_uuids": idle,
        "devices": devices,
        "lease_rows": [deepcopy(dict(row)) for row in lease_rows],
        "admission_rule": {
            "source": "experiment_7535_v659_native_pilot._admissible_gpu",
            "supported_device": "NVIDIA GeForce RTX 3090",
            "foreign_compute_processes": 0,
            "residual_used_memory_mb_lt": GPU_IDLE_MAX_USED_MB,
        },
        "snapshot_is_lease": False,
        "fresh_exclusive_lease_required_downstream": True,
        "signals_sent": [],
        "leases_acquired": [],
        "foreign_endpoints_adopted": [],
    }


def observe_gpu_resources(started: float) -> JsonDict:  # pragma: no cover - live host boundary.
    """Observe for at most 300 seconds and never mutate a process or lease."""

    configured = float(os.environ.get("CARNOT_EXP7548_OBSERVATION_WAIT_S", "0"))
    wait_limit = min(MAX_GPU_OBSERVATION_WAIT_S, max(0.0, configured))
    observation_started = time.monotonic()
    observation_count = 0
    last: JsonDict | None = None
    query_receipts: list[JsonDict] = []
    while True:
        progress(started, "resource_observation", "before_gpu_queries")
        process_rows, receipts = gpu_diagnosis.collect_gpu_process_rows()
        progress(
            started,
            "resource_observation",
            "after_gpu_queries",
            completed_units=len(process_rows),
        )
        for row in process_rows:
            pid = row.get("pid")
            row["process_ancestry"] = _process_ancestry(int(pid)) if isinstance(pid, int) else []
        progress(started, "resource_observation", "before_lease_snapshot")
        leases = gpu_diagnosis.scan_lease_rows(LEASE_RUNTIME_DIR, process_rows)
        progress(
            started,
            "resource_observation",
            "after_lease_snapshot",
            completed_units=len(leases),
        )
        last = reduce_gpu_capacity(process_rows, leases, observed_at=utc_now())
        query_receipts.extend(deepcopy(receipts))
        observation_count += 1
        elapsed = time.monotonic() - observation_started
        if last["gpu_capacity_observed_score"] == 1 or elapsed >= wait_limit:
            break
        progress(
            started,
            "resource_observation",
            "pending",
            pending_operation="read_only_capacity_wait",
            observation_count=observation_count,
        )
        time.sleep(min(5.0, wait_limit - elapsed))
    assert last is not None
    last.update(
        {
            "wait_limit_s": wait_limit,
            "waited_s": time.monotonic() - observation_started,
            "observation_count": observation_count,
            "query_receipts": query_receipts,
        }
    )
    return last


def artifact_checksum(value: Mapping[str, Any]) -> str:
    """Bind code, manifests, schedule receipts, resource rows, and validation."""

    payload = deepcopy(dict(value))
    payload.pop("reproducibility_checksum", None)
    return canonical_hash(payload)


def _field_principles(fields: Sequence[str]) -> dict[str, str]:
    specific = {
        "experiment_id": "The exact task identity prevents evidence from moving between milestones.",
        "preconditions_checked": "Exact input observations prevent a missing resource from becoming fabricated evidence.",
        "MODEL_SPECS": "An empty list proves this qualification task planned no model load.",
        "model_specs": "An empty resolved list prevents historical Qwen identity from becoming a current invocation.",
        "model_invoked": "A false current flag separates scripted transport from real model work.",
        "invocation_counts": "Typed zero counters expose any accidental load, forward, or generation claim.",
        "inference_substrate_class": "The no-load class prevents padding time or applying a model duration floor.",
        "inference_substrate": "Aggregation names the actual use of sealed upstream bytes.",
        "execution_venue": "The legal host enum prevents the invalid host_cpu venue from recurring.",
        "duration_s": "Measured monotonic duration separates current work from history and authoring.",
        "random_seed": "Frozen roster and ordering seeds prevent selection after outcomes.",
        "reproducibility_checksum": "One digest binds code, protocol, schedule, raw controls, and observations.",
        "rows": "Absolute role and resource rows keep missing evidence distinct from zero.",
        "sample_size_budget": "Planned, complete, failed, censored, and unstarted groups cannot be silently dropped.",
        "acceptance_gate_results": "Validity, readiness, resource capacity, and benefit remain separate decisions.",
        "gate_check_summary": "A block retains exact upstream, field, expected, and observed operands.",
        "honest_verdict": "A complete terminal prefix distinguishes external blockage from unfinished own work.",
        "verdict_class": "The closed enum prevents busy hardware from becoming a partial scientific result.",
        "verifier_is_oracle": "Scripted fixtures cannot establish empirical predictive benefit.",
        "flagged_adversarial": "Actual verifier findings cannot be cleared to open a downstream gate.",
        "validation_receipts": "Command scopes, exits, hashes, and cold replay make validation auditable.",
        "capture_runner_ready_score": "Runner readiness depends on code and sealed roles, not current GPU access.",
        "gpu_capacity_observed_score": "A dated idle snapshot records capacity but grants no load authority.",
        "resource_observations": "UUID, process ancestry, and leases expose every observed conflict.",
        "role_schedule_hashes": "Each arm stays bound to the same complete source groups.",
        "field_principles": "Every field states the failure that it prevents.",
    }
    return {
        field: specific.get(field, f"The {field} field preserves one auditable contract operand.")
        for field in fields
    }


def _base_artifact(
    *,
    preconditions: Sequence[Mapping[str, Any]],
    duration_s: float,
    phase_spans: Sequence[Mapping[str, Any]],
    source_hashes: Mapping[str, Any] | None = None,
) -> JsonDict:
    return {
        "schema": SCHEMA,
        "schema_binding": {
            "experiment_id": EXPERIMENT_ID,
            "milestone": MILESTONE,
            "version": VERSION,
            "run_date": RUN_DATE,
        },
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "version": VERSION,
        "run_date": RUN_DATE,
        "title": "V660 resumable source capture runner qualification",
        "preconditions_checked": [deepcopy(dict(row)) for row in preconditions],
        "source_artifact_hashes": deepcopy(dict(source_hashes or {})),
        "MODEL_SPECS": [],
        "model_specs": [],
        "model_invoked": False,
        "invocation_counts": deepcopy(ZERO_INVOCATION_COUNTS),
        "historical_invocation_counts": {
            "exp7533": deepcopy(ZERO_INVOCATION_COUNTS),
            "exp7535_literal_verdict": "complete_blocked_owned_gpu_available",
            "counted_as_current": False,
        },
        "inference_substrate_class": "no_model_load",
        "inference_substrate": "aggregation_from_upstream_artifacts",
        "inference_substrate_details": {
            "sealed_protocol": "exp7533-v659-tool-protocol",
            "transport_interface": "experiment_7535_v659_native_pilot.NativeOptionRunner",
            "current_model_loads": 0,
            "current_native_forwards": 0,
            "current_generated_tokens": 0,
            "scripted_fixture_calls_are_current_model_calls": False,
        },
        "planned_inference_substrate_class": "no_model_load",
        "execution_venue": "host",
        "execution_device": "host_cpu_plus_read_only_cuda_inventory",
        "duration_s": max(0.000001, float(duration_s)),
        "duration_breakdown_s": {
            "current_cpu_qualification_and_validation": max(0.000001, float(duration_s)),
            "authoring": 0.0,
            "historical": 0.0,
        },
        "phase_spans": [deepcopy(dict(row)) for row in phase_spans],
        "process_identity": {
            "pid": os.getpid(),
            "pid_start_ticks": gpu_diagnosis.lease_api.proc_start_ticks(os.getpid()),
            "python": os.path.realpath(os.sys.executable),
            "clock": "time.monotonic",
        },
        "random_seed": {
            "upstream_role_allocation": tool_protocol.SELECTION_SEED,
            "development_selection": native_pilot.SELECTION_SEED,
            "ordering": 6600481,
            "sampling": 6600482,
            "fitting": 6600483,
            "bootstrap": 6600484,
        },
        "verifier_is_oracle": False,
        "flagged_adversarial": False,
        "adversarial_corrections": [],
        "positive_claim": False,
        "production_defaults_changed": False,
        "generator_weights_changed": False,
        "external_publication_authorized": False,
        "applicable_numbered_e2e": [],
        "private_llm_off_real_environment_smoke": "not_applicable_capture_transport_qualification",
        "source_swap_label_semantics": "features_only_original_source_label_applies",
        "validation_reserve_s": VALIDATION_RESERVE_S,
    }


def _role_rows(role_counts: Mapping[str, int], gpu_score: int) -> list[JsonDict]:
    rows = [
        {
            "role": role,
            "absolute_metrics": {
                "planned_groups": count,
                "complete_schedule_groups": count,
                "forwards_per_group": 6,
                "complete_prompt_forwards": count * 6,
            },
            "planned_count": count,
            "complete_count": count,
            "failed_count": 0,
            "censored_count": 0,
            "excluded_count": 0,
            "unstarted_count": 0,
            "disposition": "schedule_frozen_no_model_call",
        }
        for role, count in role_counts.items()
    ]
    rows.append(
        {
            "role": "gpu_capacity_observation",
            "absolute_metrics": {"idle_supported_device_count_positive": gpu_score},
            "planned_count": 1,
            "complete_count": 1,
            "failed_count": 0,
            "censored_count": 0,
            "excluded_count": 0,
            "unstarted_count": 0,
            "disposition": "idle_capacity_observed" if gpu_score else "gpu_busy_observed",
        }
    )
    return rows


def _required_validation_passed(
    receipts: Sequence[Mapping[str, Any]], *, include_terminal: bool
) -> bool:
    required = set(AFFECTED_CHECK_NAMES)
    if include_terminal:
        required.update(TERMINAL_CHECK_NAMES)
    passed = {
        str(row.get("name"))
        for row in receipts
        if row.get("passed") is True and row.get("exit_code") == 0
    }
    return required <= passed


def _finish_artifact(artifact: JsonDict) -> JsonDict:
    fields = (*artifact.keys(), "field_principles", "reproducibility_checksum")
    artifact["field_principles"] = _field_principles(fields)
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return artifact


def build_complete_artifact(
    *,
    preconditions: Sequence[Mapping[str, Any]],
    source_hashes: Mapping[str, Any],
    schedule_receipt: Mapping[str, Any],
    schedule_manifest: Mapping[str, Any],
    controls: Mapping[str, Any],
    resource: Mapping[str, Any],
    validation_receipts: Sequence[Mapping[str, Any]],
    duration_s: float,
    phase_spans: Sequence[Mapping[str, Any]],
    include_terminal_validation: bool,
) -> JsonDict:
    """Build one null, blocked, or disqualified terminal candidate."""

    all_preconditions = all(row.get("passed") is True for row in preconditions)
    schedule_ready = (
        schedule_manifest.get("role_counts") == ROLE_COUNTS
        and schedule_manifest.get("forward_count") == TOTAL_FORWARDS
        and schedule_manifest.get("online_groups_excluded") == 160
        and schedule_manifest.get("labels_read") == 0
    )
    controls_ready = controls.get("passed") is True
    validation_ready = _required_validation_passed(
        validation_receipts, include_terminal=include_terminal_validation
    )
    capture_ready = all_preconditions and schedule_ready and controls_ready and validation_ready
    gpu_score = int(resource.get("gpu_capacity_observed_score") == 1)
    reduction_input = {
        "capture_schedule": deepcopy(dict(schedule_receipt)),
        "qualification_schedule": deepcopy(dict(controls["qualification_schedule"])),
        "qualification_rows": deepcopy(dict(controls["qualification_rows"])),
        "role_counts": ROLE_COUNTS,
        "qualification_role_counts": deepcopy(dict(controls["qualification_role_counts"])),
    }
    raw_reduction = independent_reduce(reduction_input)
    raw_ready = raw_reduction.get("passed") is True
    capture_ready = capture_ready and raw_ready
    gates = [
        gate(
            "authenticated_protocol",
            "validity",
            True,
            all_preconditions,
            "==",
            all_preconditions,
            "Invalid or changed upstream evidence cannot support collection.",
            upstream="exp7533-v659-tool-protocol",
            field="all_preconditions_passed",
        ),
        gate(
            "exact_capture_schedule",
            "readiness",
            {"groups": TOTAL_GROUPS, "forwards": TOTAL_FORWARDS},
            {
                "groups": sum(dict(schedule_manifest.get("role_counts") or {}).values()),
                "forwards": schedule_manifest.get("forward_count"),
            },
            "==",
            schedule_ready,
            "A valid null remains auditable only with every registered group and arm.",
            upstream="capture_schedule",
            field="role_counts_and_forward_count",
        ),
        gate(
            "scripted_transport_qualification",
            "validity",
            True,
            controls_ready,
            "==",
            controls_ready,
            "Checkpoint, resume, reset, and mutation failures must pass before model work.",
            upstream="private_scripted_transport",
            field="qualification_controls.passed",
        ),
        gate(
            "independent_raw_reduction",
            "validity",
            True,
            raw_ready,
            "==",
            raw_ready,
            "Hash-bound rows must reduce independently before readiness can open.",
            upstream="qualification_sidecars",
            field="independent_reduce.passed",
        ),
        gate(
            "required_validation",
            "validity",
            True,
            validation_ready,
            "==",
            validation_ready,
            "Scoped checks and cold replay must pass before publication.",
            upstream="validation_receipts",
            field="all_required_checks_passed",
        ),
        gate(
            "gpu_capacity_available",
            "external_precondition",
            1,
            gpu_score,
            "==",
            gpu_score == 1,
            "A busy device cannot authorize the later native pilot.",
            upstream="read_only_gpu_process_and_lease_snapshot",
            field="gpu_capacity_observed_score",
            path="nvidia-smi_compute_process_inventory",
        ),
        gate(
            "positive_claim",
            "benefit",
            False,
            False,
            "==",
            True,
            "Transport qualification and idle capacity do not measure predictive benefit.",
            upstream="runner_qualification_only",
            field="positive_claim",
        ),
    ]
    if not capture_ready:
        verdict_class = "disqualified"
        verdict = "complete_disqualified_capture_runner_validation"
    elif not gpu_score:
        verdict_class = "blocked"
        verdict = "complete_blocked_gpu_capacity_unavailable"
    else:
        verdict_class = "null"
        verdict = "complete_null_capture_runner_ready_benefit_unmeasured"
    artifact = _base_artifact(
        preconditions=preconditions,
        duration_s=duration_s,
        phase_spans=phase_spans,
        source_hashes=source_hashes,
    )
    artifact.update(
        {
            **reduction_input,
            "capture_schedule_manifest": deepcopy(dict(schedule_manifest)),
            "role_counts": deepcopy(ROLE_COUNTS),
            "role_schedule_hashes": deepcopy(dict(schedule_manifest["role_schedule_hashes"])),
            "qualification_controls": {
                key: deepcopy(value)
                for key, value in controls.items()
                if key not in {"qualification_schedule", "qualification_rows"}
            },
            "independent_raw_reduction": raw_reduction,
            "resource_observations": deepcopy(dict(resource)),
            "capture_runner_ready_score": int(capture_ready),
            "gpu_capacity_observed_score": gpu_score,
            "rows": _role_rows(ROLE_COUNTS, gpu_score),
            "sample_size_budget": {
                "planned": TOTAL_GROUPS,
                "attempted": TOTAL_GROUPS,
                "completed": TOTAL_GROUPS if schedule_ready else 0,
                "excluded": 160,
                "failed": 0 if schedule_ready else TOTAL_GROUPS,
                "censored": 0,
                "unstarted": 0 if schedule_ready else TOTAL_GROUPS,
                "planned_forwards": TOTAL_FORWARDS,
                "current_model_forwards": 0,
            },
            "acceptance_gate_results": gates,
            "gate_check_summary": gate_check_summary(gates),
            "validation_receipts": [deepcopy(dict(row)) for row in validation_receipts],
            "honest_verdict": verdict,
            "verdict_class": verdict_class,
            "complete": 1,
            "ready": int(capture_ready and gpu_score == 1),
            "benefit_measured": False,
            "honest_no_headroom_annotation": "runner_qualification_only_benefit_unmeasured",
        }
    )
    return _finish_artifact(artifact)


def build_blocked_artifact(
    *,
    failed_gate: Mapping[str, Any],
    preconditions: Sequence[Mapping[str, Any]],
    duration_s: float,
    phase_spans: Sequence[Mapping[str, Any]],
    source_hashes: Mapping[str, Any] | None = None,
    resource: Mapping[str, Any] | None = None,
    validation_receipts: Sequence[Mapping[str, Any]] = (),
) -> JsonDict:
    """Publish exact external absence without inventing a fallback schedule."""

    reason = str(failed_gate.get("check") or "external_precondition")
    gates = [deepcopy(dict(failed_gate))]
    observations = deepcopy(
        dict(
            resource
            or {
                "observed_at_utc": utc_now(),
                "gpu_capacity_observed_score": 0,
                "devices": [],
                "lease_rows": [],
                "snapshot_is_lease": False,
                "signals_sent": [],
                "leases_acquired": [],
            }
        )
    )
    artifact = _base_artifact(
        preconditions=preconditions,
        duration_s=duration_s,
        phase_spans=phase_spans,
        source_hashes=source_hashes,
    )
    artifact.update(
        {
            "capture_schedule": None,
            "qualification_schedule": None,
            "qualification_rows": None,
            "capture_schedule_manifest": None,
            "qualification_controls": None,
            "independent_raw_reduction": None,
            "role_counts": deepcopy(ROLE_COUNTS),
            "role_schedule_hashes": {},
            "resource_observations": observations,
            "capture_runner_ready_score": 0,
            "gpu_capacity_observed_score": int(
                observations.get("gpu_capacity_observed_score") == 1
            ),
            "rows": [],
            "sample_size_budget": {
                "planned": TOTAL_GROUPS,
                "attempted": 0,
                "completed": 0,
                "excluded": 160,
                "failed": 0,
                "censored": 0,
                "unstarted": TOTAL_GROUPS,
                "planned_forwards": TOTAL_FORWARDS,
                "current_model_forwards": 0,
            },
            "acceptance_gate_results": gates,
            "gate_check_summary": gate_check_summary(gates),
            "validation_receipts": [deepcopy(dict(row)) for row in validation_receipts],
            "honest_verdict": f"complete_blocked_{reason}",
            "verdict_class": "blocked",
            "complete": 1,
            "ready": 0,
            "benefit_measured": False,
            "honest_no_headroom_annotation": "external_precondition_absent_benefit_unmeasured",
        }
    )
    return _finish_artifact(artifact)


def build_artifact_for_test(*, gpu_capacity_observed_score: int) -> JsonDict:
    """Build a production-shaped artifact without writing repository state."""

    precondition = gate(
        "fixture_protocol",
        "validity",
        True,
        True,
        "==",
        True,
        "The fixture must enter through the same gate shape as production.",
        upstream="private_fixture",
        field="authenticated",
    )
    resource = {
        "observed_at_utc": "2026-09-23T00:00:00Z",
        "gpu_capacity_observed_score": gpu_capacity_observed_score,
        "idle_supported_gpu_uuids": ["GPU-fixture"] if gpu_capacity_observed_score else [],
        "devices": [],
        "lease_rows": [],
        "snapshot_is_lease": False,
        "fresh_exclusive_lease_required_downstream": True,
        "signals_sent": [],
        "leases_acquired": [],
    }
    gates = [
        gate(
            "capture_runner_qualification",
            "readiness",
            1,
            1,
            "==",
            True,
            "A ready fixture must satisfy the real runner boundary.",
            upstream="private_fixture",
            field="capture_runner_ready_score",
        ),
        gate(
            "gpu_capacity_available",
            "external_precondition",
            1,
            gpu_capacity_observed_score,
            "==",
            gpu_capacity_observed_score == 1,
            "A busy device remains an external block.",
            upstream="private_fixture",
            field="gpu_capacity_observed_score",
        ),
        gate(
            "positive_claim",
            "benefit",
            False,
            False,
            "==",
            True,
            "A fixture cannot establish empirical benefit.",
            upstream="private_fixture",
            field="positive_claim",
        ),
    ]
    artifact = _base_artifact(preconditions=[precondition], duration_s=1.0, phase_spans=[])
    artifact.update(
        {
            "role_counts": deepcopy(ROLE_COUNTS),
            "role_schedule_hashes": {
                role: canonical_hash({"role": role, "count": count})
                for role, count in ROLE_COUNTS.items()
            },
            "resource_observations": resource,
            "capture_runner_ready_score": 1,
            "gpu_capacity_observed_score": gpu_capacity_observed_score,
            "rows": _role_rows(ROLE_COUNTS, gpu_capacity_observed_score),
            "sample_size_budget": {
                "planned": TOTAL_GROUPS,
                "attempted": TOTAL_GROUPS,
                "completed": TOTAL_GROUPS,
                "excluded": 160,
                "failed": 0,
                "censored": 0,
                "unstarted": 0,
                "planned_forwards": TOTAL_FORWARDS,
                "current_model_forwards": 0,
            },
            "acceptance_gate_results": gates,
            "gate_check_summary": gate_check_summary(gates),
            "validation_receipts": [],
            "honest_verdict": (
                "complete_null_capture_runner_ready_benefit_unmeasured"
                if gpu_capacity_observed_score
                else "complete_blocked_gpu_capacity_unavailable"
            ),
            "verdict_class": "null" if gpu_capacity_observed_score else "blocked",
            "complete": 1,
            "ready": gpu_capacity_observed_score,
            "benefit_measured": False,
            "honest_no_headroom_annotation": "runner_qualification_only_benefit_unmeasured",
        }
    )
    return _finish_artifact(artifact)


def validate_artifact(
    value: object,
    *,
    require_validation: bool = True,
    root: Path = REPO_ROOT,
) -> list[str]:
    """Cold-check identity, zero calls, bare scores, raw custody, and claims."""

    if not isinstance(value, Mapping):
        return ["artifact_not_object"]
    artifact = dict(value)
    errors: list[str] = []
    expected = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "version": VERSION,
        "run_date": RUN_DATE,
        "MODEL_SPECS": [],
        "model_specs": [],
        "inference_substrate_class": "no_model_load",
        "inference_substrate": "aggregation_from_upstream_artifacts",
        "execution_venue": "host",
        "verifier_is_oracle": False,
        "positive_claim": False,
        "flagged_adversarial": False,
    }
    for field, wanted in expected.items():
        if artifact.get(field) != wanted:
            errors.append(f"{field}_mismatch")
    if artifact.get("model_invoked") is not False:
        errors.append("model_invoked_mismatch")
    if artifact.get("invocation_counts") != ZERO_INVOCATION_COUNTS:
        errors.append("invocation_counts_mismatch")
    for field in ("capture_runner_ready_score", "gpu_capacity_observed_score"):
        score = artifact.get(field)
        if not isinstance(score, int) or isinstance(score, bool) or score not in {0, 1}:
            errors.append(f"{field}_not_bare_int")
    capture_score = artifact.get("capture_runner_ready_score")
    gpu_score = artifact.get("gpu_capacity_observed_score")
    verdict_class = artifact.get("verdict_class")
    if verdict_class not in {
        "positive",
        "circular_positive",
        "null",
        "blocked",
        "disqualified",
        "partial",
    }:
        errors.append("verdict_class_invalid")
    verdict = str(artifact.get("honest_verdict") or "")
    if not verdict.startswith("complete_"):
        errors.append("honest_verdict_terminal_prefix_missing")
    if capture_score == 1 and gpu_score == 0 and verdict_class != "blocked":
        errors.append("busy_gpu_verdict_mismatch")
    if capture_score == 1 and gpu_score == 1 and verdict_class != "null":
        errors.append("ready_verdict_mismatch")
    if capture_score == 0 and verdict_class not in {"blocked", "disqualified"}:
        errors.append("unready_verdict_mismatch")
    resource = artifact.get("resource_observations")
    if not isinstance(resource, Mapping):
        errors.append("resource_observations_invalid")
    elif resource.get("gpu_capacity_observed_score") != gpu_score:
        errors.append("resource_score_mismatch")
    budget = artifact.get("sample_size_budget")
    budget_fields = {
        "planned",
        "attempted",
        "completed",
        "excluded",
        "failed",
        "censored",
        "unstarted",
    }
    if not isinstance(budget, Mapping) or not budget_fields <= set(budget):
        errors.append("sample_size_budget_invalid")
    gates = artifact.get("acceptance_gate_results")
    if not isinstance(gates, list) or any(
        not isinstance(row, Mapping)
        or not str(row.get("principle") or "").strip()
        or not {"expected", "observed", "op", "passed"} <= set(row)
        for row in gates or []
    ):
        errors.append("acceptance_gate_results_invalid")
    elif artifact.get("gate_check_summary") != gate_check_summary(gates):
        errors.append("gate_check_summary_mismatch")
    principles = artifact.get("field_principles")
    if not isinstance(principles, Mapping) or not set(artifact) <= set(principles):
        errors.append("field_principles_incomplete")
    elif any(not str(value).strip() for value in principles.values()):
        errors.append("field_principles_empty")
    if artifact.get("reproducibility_checksum") != artifact_checksum(artifact):
        errors.append("reproducibility_checksum_mismatch")
    duration = artifact.get("duration_s")
    if not isinstance(duration, (int, float)) or isinstance(duration, bool) or duration <= 0:
        errors.append("duration_invalid")
    if not isinstance(artifact.get("preconditions_checked"), list):
        errors.append("preconditions_checked_invalid")
    if capture_score == 1:
        role_hashes = artifact.get("role_schedule_hashes")
        if not isinstance(role_hashes, Mapping) or set(role_hashes) != set(ROLE_COUNTS):
            errors.append("role_schedule_hashes_invalid")
        if isinstance(artifact.get("capture_schedule"), Mapping):
            raw = independent_reduce(artifact, root=root)
            if raw.get("passed") is not True:
                errors.append("independent_raw_reduction_failed")
    if require_validation:
        receipts = artifact.get("validation_receipts")
        if not isinstance(receipts, list) or not _required_validation_passed(
            receipts, include_terminal=True
        ):
            errors.append("required_validation_missing_or_failed")
    return list(dict.fromkeys(errors))


def _phase_span(
    phase: str,
    phase_started: float,
    run_started: float,
    completed_units: int,
    checkpoint: str,
) -> JsonDict:
    return {
        "phase": phase,
        "start_s": phase_started - run_started,
        "end_s": time.monotonic() - run_started,
        "completed_units": completed_units,
        "checkpoint": checkpoint,
        "ended_at_utc": utc_now(),
    }


def _run_affected_validation(
    root: Path, started: float
) -> tuple[list[JsonDict], bool]:  # pragma: no cover - subprocess boundary.
    private = Path(tempfile.mkdtemp(prefix="carnot-exp7548-validation-", dir="/tmp"))
    commands = build_command_plan(root, VALIDATION_MANIFEST, private)
    plan_errors = validate_command_plan(root, VALIDATION_MANIFEST, commands)
    if plan_errors:
        raise CaptureRunnerError("validation_plan_invalid:" + ",".join(plan_errors))
    progress(started, "affected_validation", "before_subprocesses", planned=len(commands))
    receipts = run_categorized_commands(
        root,
        [PlannedCommand(command, "required_validation", True) for command in commands],
        log_dir=root / RAW_DIR / "validation/affected",
        heartbeat_s=60.0,
    )
    progress(
        started,
        "affected_validation",
        "after_subprocesses",
        completed_units=len(receipts),
    )
    reduction = reduce_affected_receipts(root, VALIDATION_MANIFEST, receipts)
    return [dict(row) for row in receipts], reduction.get("passed") is True


def _terminal_commands(candidate: Path) -> list[PlannedCommand]:  # pragma: no cover
    relative = candidate.relative_to(REPO_ROOT).as_posix()
    python = ".venv/bin/python"
    specs = (
        validation_scope.CommandSpec(
            "declared_entrypoint_cold_replay",
            (python, "-u", WRAPPER_PATH.as_posix(), "--date", RUN_DATE, "--cold-replay", relative),
            "completion",
            180,
        ),
        validation_scope.CommandSpec(
            "independent_raw_reduction",
            (
                python,
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
            (python, "-u", "scripts/adversarial_verify.py", relative),
            "safety",
            180,
        ),
        validation_scope.CommandSpec(
            "verdict_row_consistency_strict",
            (python, "-u", "scripts/verdict_row_consistency_lint.py", "--strict", relative),
            "completion",
            180,
        ),
    )
    return [PlannedCommand(spec, spec.scope, True) for spec in specs]


def _run_terminal_validation(
    root: Path,
    artifact: Mapping[str, Any],
    affected_receipts: Sequence[Mapping[str, Any]],
    spans: list[JsonDict],
    started: float,
) -> tuple[list[JsonDict], bool]:  # pragma: no cover - subprocess boundary.
    candidate = root / CANDIDATE_PATH
    atomic_json(candidate, artifact)
    commands = _terminal_commands(candidate)
    phase_started = time.monotonic()
    progress(started, "terminal_validation", "before_subprocesses", planned=len(commands))
    terminal = run_categorized_commands(
        root,
        commands,
        log_dir=root / RAW_DIR / "validation/terminal",
        heartbeat_s=60.0,
    )
    progress(
        started,
        "terminal_validation",
        "after_subprocesses",
        completed_units=len(terminal),
    )
    spans.append(
        _phase_span(
            "terminal_validation",
            phase_started,
            started,
            len(terminal),
            CANDIDATE_PATH.as_posix(),
        )
    )
    receipts = [
        *[deepcopy(dict(row)) for row in affected_receipts],
        *[deepcopy(dict(row)) for row in terminal],
    ]
    return receipts, all(row.get("passed") is True for row in terminal)


def run_experiment(root: Path, run_date: str) -> int:  # pragma: no cover - orchestration.
    """Authenticate, freeze, qualify, observe, validate, and publish atomically."""

    if run_date != RUN_DATE:
        raise CaptureRunnerError(f"run_date_mismatch:{run_date}")
    started = time.monotonic()
    spans: list[JsonDict] = []
    progress(started, "preconditions", "start")
    phase_started = time.monotonic()
    progress(started, "preconditions", "before_named_inputs")
    preconditions, context = collect_preconditions(root)
    progress(
        started,
        "preconditions",
        "after_named_inputs",
        completed_units=len(preconditions),
    )
    spans.append(
        _phase_span(
            "preconditions", phase_started, started, len(preconditions), "inputs_authenticated"
        )
    )
    failed_gate = next((row for row in preconditions if row.get("passed") is not True), None)

    schedule_receipt: JsonDict | None = None
    schedule_manifest: JsonDict | None = None
    controls: JsonDict | None = None
    if failed_gate is None:
        phase_started = time.monotonic()
        progress(started, "schedule", "start")
        try:
            schedule, schedule_manifest = freeze_capture_schedule(context, started=started)
            schedule_receipt = write_jsonl_sidecar(root / SCHEDULE_PATH, schedule, root=root)
        except (CaptureRunnerError, OSError, ValueError) as exc:
            failed_gate = gate(
                "sealed_schedule_unavailable",
                "external_precondition",
                "complete_332_group_schedule",
                f"{type(exc).__name__}:{exc}",
                "==",
                False,
                "A missing sealed roster cannot be replaced with fabricated prompts.",
                upstream="exp7533_and_exp7535_public_inputs",
                field="capture_schedule",
                path=SCHEDULE_PATH.as_posix(),
            )
            preconditions.append(failed_gate)
        spans.append(
            _phase_span(
                "schedule",
                phase_started,
                started,
                int(schedule_manifest is not None),
                SCHEDULE_PATH.as_posix(),
            )
        )
    if failed_gate is None:
        phase_started = time.monotonic()
        progress(started, "scripted_qualification", "start")
        controls = run_qualification_controls(root / RAW_DIR / "qualification")
        progress(
            started,
            "scripted_qualification",
            "end",
            completed_units=controls["reduction"]["complete_group_count"],
        )
        spans.append(
            _phase_span(
                "scripted_qualification",
                phase_started,
                started,
                int(controls["passed"]),
                str(controls["qualification_rows"]["path"]),
            )
        )

    phase_started = time.monotonic()
    affected, affected_passed = _run_affected_validation(root, started)
    spans.append(
        _phase_span(
            "affected_validation",
            phase_started,
            started,
            len(affected),
            "scoped_commands_complete",
        )
    )
    if not affected_passed and failed_gate is None:
        progress(started, "affected_validation", "required_check_failed")

    phase_started = time.monotonic()
    progress(started, "resource_observation", "start")
    resource = observe_gpu_resources(started)
    progress(
        started,
        "resource_observation",
        "end",
        capacity=resource["gpu_capacity_observed_score"],
    )
    spans.append(
        _phase_span(
            "resource_observation",
            phase_started,
            started,
            int(resource["observation_count"]),
            "read_only_snapshot",
        )
    )

    if failed_gate is not None:
        artifact = build_blocked_artifact(
            failed_gate=failed_gate,
            preconditions=preconditions,
            duration_s=time.monotonic() - started,
            phase_spans=spans,
            source_hashes=context["source_artifact_hashes"],
            resource=resource,
            validation_receipts=affected,
        )
    else:
        assert (
            schedule_receipt is not None and schedule_manifest is not None and controls is not None
        )
        artifact = build_complete_artifact(
            preconditions=preconditions,
            source_hashes=context["source_artifact_hashes"],
            schedule_receipt=schedule_receipt,
            schedule_manifest=schedule_manifest,
            controls=controls,
            resource=resource,
            validation_receipts=affected,
            duration_s=time.monotonic() - started,
            phase_spans=spans,
            include_terminal_validation=False,
        )

    receipts, terminal_passed = _run_terminal_validation(root, artifact, affected, spans, started)
    if not terminal_passed:
        progress(started, "terminal_validation", "required_check_failed")
    if failed_gate is not None:
        final = build_blocked_artifact(
            failed_gate=failed_gate,
            preconditions=preconditions,
            duration_s=time.monotonic() - started,
            phase_spans=spans,
            source_hashes=context["source_artifact_hashes"],
            resource=resource,
            validation_receipts=receipts,
        )
    else:
        assert (
            schedule_receipt is not None and schedule_manifest is not None and controls is not None
        )
        final = build_complete_artifact(
            preconditions=preconditions,
            source_hashes=context["source_artifact_hashes"],
            schedule_receipt=schedule_receipt,
            schedule_manifest=schedule_manifest,
            controls=controls,
            resource=resource,
            validation_receipts=receipts,
            duration_s=time.monotonic() - started,
            phase_spans=spans,
            include_terminal_validation=True,
        )
    errors = validate_artifact(final, require_validation=True, root=root)
    if errors:
        raise CaptureRunnerError("terminal_artifact_invalid:" + ",".join(errors))
    progress(started, "publish", "before_atomic_write", path=RESULT_PATH.as_posix())
    atomic_json(root / RESULT_PATH, final)
    progress(
        started,
        "publish",
        "after_atomic_write",
        bytes=(root / RESULT_PATH).stat().st_size,
    )
    return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the fixed run date and fresh-process reader modes."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", required=True, choices=[RUN_DATE])
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--cold-replay", type=Path)
    modes.add_argument("--independent-reduce", type=Path)
    return parser.parse_args(argv)


def _argument_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def main(argv: Sequence[str] | None = None) -> int:
    """Run CPU qualification or one read-only exact-candidate check."""

    args = parse_args(argv)
    if args.cold_replay is not None:
        candidate_path = _argument_path(args.cold_replay)
        value = _load_object(candidate_path)
        errors = validate_artifact(value, require_validation=False, root=REPO_ROOT)
        print(
            json.dumps({"cold_replay_passed": not errors, "errors": errors}, sort_keys=True),
            flush=True,
        )
        return int(bool(errors))
    if args.independent_reduce is not None:
        candidate_path = _argument_path(args.independent_reduce)
        result = independent_reduce(_load_object(candidate_path), root=REPO_ROOT)
        print(json.dumps(result, sort_keys=True), flush=True)
        return int(result.get("passed") is not True)
    return run_experiment(REPO_ROOT, args.date)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
