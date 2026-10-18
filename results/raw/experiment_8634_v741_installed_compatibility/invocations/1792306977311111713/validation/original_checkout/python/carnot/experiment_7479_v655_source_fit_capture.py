"""Capture source-conditioned native option logits for fit roles.

The capture keeps training evidence separate from held-out evaluation. It uses
the complete sealed source and response, and it never puts human labels in a
model prompt.

Spec refs: REQ-VERIFY-7479 and SCENARIO-VERIFY-7479-*.
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
import tempfile
import time
from typing import Any

from carnot import experiment_7462_v654_option_protocol as option_protocol
from carnot import experiment_7463_v654_semif_e0_logprob_parity as parity
from carnot import experiment_7477_v655_native_readout_pilot as pilot
from carnot import gpu_lease_phase_journal as lease_api
from carnot.experiment_7358_v646_validation_contract import (
    AffectedManifest,
    PlannedCommand,
    build_command_plan,
    reduce_affected_receipts,
    run_categorized_commands,
    validate_command_plan,
)
from carnot.inference.sota_models import cached_current_model
from carnot.reporting import experiment_7303_validation_scope as validation_scope
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file


JsonDict = dict[str, Any]
Tokenizer = Callable[..., list[int]]

RUN_DATE = "20260921"
MILESTONE = "2026.09.655"
EXPERIMENT_ID = "exp7479-source-fit-capture"
SCHEMA = "carnot.exp7479.v655.source_fit_capture.v1"
MODEL_HF_ID = "unsloth/Qwen3.8-27B-GGUF"
MODEL_SPECS = [MODEL_HF_ID]
INFERENCE_SUBSTRATE = "live_native_llama_cpp_source_fit_raw_logit_capture"
INFERENCE_SUBSTRATE_CLASS = "model_load_no_generation"
EXECUTION_VENUE = "host"
TOKEN_CEILING = 2_048
FORWARD_BUDGET = 520
MAIN_FORWARD_BUDGET = 480
CONTROL_FORWARD_BUDGET = 40
CONTROL_GROUP_COUNT = 20
MAX_LIVE_SECONDS = 3_300.0
MAX_ARTIFACT_BYTES = 20 * 1024 * 1024
ROLE_MINIMUMS = {"training": 150, "calibration_tuning": 40}
OPTION_IDS = option_protocol.OPTION_IDS
OPTION_ORDERS = (OPTION_IDS, tuple(reversed(OPTION_IDS)))

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_PATH = Path("results/experiment_7479_v655_source_fit_capture.json")
RAW_DIR = Path("results/raw/experiment_7479_v655_source_fit_capture")
MODULE_PATH = Path("python/carnot/experiment_7479_v655_source_fit_capture.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_7479_v655_source_fit_capture.py")
TEST_PATH = Path("tests/python/test_experiment_7479_v655_source_fit_capture.py")
SPEC_PATH = REPO_ROOT / "openspec/capabilities/verification/spec.md"
V654_RAW_DIR = Path("results/raw/experiment_7462_v654_option_protocol")
PREDICTOR_PATH = V654_RAW_DIR / "cohort_predictors.jsonl"
EVALUATOR_PATH = V654_RAW_DIR / "cohort_evaluators.jsonl"
GROUP_PATH = V654_RAW_DIR / "cohort_groups.jsonl"
V654_RESULT_PATH = Path("results/experiment_7462_v654_option_protocol.json")
V655_QUALIFICATION_PATH = Path("results/experiment_7476_v655_option_qualification.json")
V655_PILOT_PATH = Path("results/experiment_7477_v655_native_readout_pilot.json")
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

REQUIRED_FIELDS = (
    "schema",
    "run_date",
    "preconditions_checked",
    "MODEL_SPECS",
    "model_specs",
    "model_invoked",
    "invocation_counts",
    "inference_substrate",
    "inference_substrate_class",
    "execution_venue",
    "duration_s",
    "phase_spans",
    "random_seed",
    "reproducibility_checksum",
    "source_artifact_hashes",
    "rows",
    "sample_size_budget",
    "acceptance_gate_results",
    "gate_check_summary",
    "honest_verdict",
    "verdict_class",
    "verifier_is_oracle",
    "flagged_adversarial",
    "validation_receipts",
    "field_principles",
    "fit_capture_ready_score",
    "role_counts",
    "raw_logit_shards",
    "capture_complete_score",
    "confirmatory_support_score",
)

FIELD_PRINCIPLES = {
    "schema": "Versioned schema with exact roadmap experiment_id, milestone and terminal status prevents silent reader drift.",
    "run_date": "Use 20260921; retain measured UTC and monotonic times with clock/process identity.",
    "preconditions_checked": "Name resources, exact paths, ownership and observed prerequisite values before dependent work.",
    "MODEL_SPECS": "Use unsloth/Qwen3.8-27B-GGUF for current model tasks, and also emit lowercase model_specs.",
    "model_specs": "The lowercase model list prevents reader-specific model identity drift.",
    "model_invoked": "Any attempted current model call differs from archived or scripted events.",
    "invocation_counts": "Balance attempted, complete, failed, cancelled and in-flight loads, forwards and generations.",
    "inference_substrate": "Name the actual native readout without claiming generation or numeric fitting.",
    "inference_substrate_class": "Use model_load_no_generation because one model loads and zero tokens are generated.",
    "execution_venue": "Use host and record actual CPU and CUDA identities; historical board evidence is separate.",
    "duration_s": "Measure current work without padding and separate load, forward, generation, reduction and validation.",
    "phase_spans": "Timestamped flushed progress and checkpoints expose long silent or unfinished operations.",
    "random_seed": "Freeze ordering, fitting, audit and bootstrap seeds; deterministic reducers explain any null seed.",
    "reproducibility_checksum": "Bind code, protocol, data roles, model identity, raw shards and validation scope.",
    "source_artifact_hashes": "Preserve exact upstream bytes and their original flags and classes.",
    "rows": "One row per independent source group retains failures and exclusions without duplicating its two orders.",
    "sample_size_budget": "Separate planned, attempted, complete, failed, censored, excluded and unstarted cells.",
    "acceptance_gate_results": "Each typed check includes operands, result and the failure mode it prevents.",
    "gate_check_summary": "Every blocked verdict names the failed check, upstream field, expected and observed value.",
    "honest_verdict": "Use a complete terminal finding and preserve a null without claiming held-out benefit.",
    "verdict_class": "Use the closed verdict enum so null, blocked, disqualified and partial remain distinct.",
    "verifier_is_oracle": "False prevents validation machinery from becoming positive scientific evidence.",
    "flagged_adversarial": "Keep real reader flags and never clear a flag to open a gate.",
    "validation_receipts": "Exact commands, exits, log hashes and required status establish validation scope.",
    "field_principles": "Explain why each field and gate exists so evidence is independently understandable.",
    "fit_capture_ready_score": "A bare 0 or 1 follows transport, accounting, validation and role support, not benefit.",
    "role_counts": "Track independent eligible groups and every exclusion inside each sealed role.",
    "raw_logit_shards": "Hash-bound per-cell logits permit downstream fitting without another model run.",
    "capture_complete_score": "Account for every planned cell, including failures, exclusions and unstarted cells.",
    "confirmatory_support_score": "Keep role-specific sample floors separate from transport validity.",
}


class CaptureError(ValueError):
    """Reject evidence that could silently change the capture meaning."""


def load_json_object(path: Path) -> JsonDict:
    """Read one JSON object and fail closed to an empty object."""

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(value) if isinstance(value, Mapping) else {}


def _load_jsonl(path: Path) -> list[JsonDict]:
    """Read object rows from one bounded JSONL input."""

    try:
        values = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    except (OSError, json.JSONDecodeError) as exc:  # pragma: no cover - authenticated live input.
        raise CaptureError(f"jsonl_unreadable:{path}") from exc
    if any(
        not isinstance(row, dict) for row in values
    ):  # pragma: no cover - authenticated live input.
        raise CaptureError(f"jsonl_object_required:{path}")
    return values


def role_counts(groups: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    """Count the two roles that are allowed to enter this shard."""

    counts = Counter(str(row.get("role")) for row in groups)
    return {role: counts.get(role, 0) for role in ("training", "calibration_tuning")}


def _stable_text_hash(value: str) -> str:
    """Match the outcome-blind text hash that preregistered V654 controls."""

    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def load_sealed_fit_groups(root: Path) -> list[JsonDict]:
    """Join sealed predictor and evaluator rows without moving labels into prompts."""

    try:
        predictors = _load_jsonl(root / PREDICTOR_PATH)
        evaluators = _load_jsonl(root / EVALUATOR_PATH)
        groups = _load_jsonl(root / GROUP_PATH)
    except CaptureError:
        raise CaptureError("fit_role_counts_invalid") from None
    predictors_by_id = {str(row["group_id"]): row for row in predictors}
    evaluators_by_id = {str(row["group_id"]): row for row in evaluators}
    selected = [row for row in groups if row.get("role") in {"training", "calibration_tuning"}]
    if role_counts(selected) != {"training": 180, "calibration_tuning": 60}:
        raise CaptureError("fit_role_counts_invalid")
    joined: list[JsonDict] = []
    for position, group in enumerate(selected):
        group_id = str(group["group_id"])
        predictor = predictors_by_id.get(group_id)
        evaluator = evaluators_by_id.get(group_id)
        if predictor is None or evaluator is None:  # pragma: no cover - authenticated sealed input.
            raise CaptureError(f"sealed_join_missing:{group_id}")
        joined.append(
            {
                "roster_position": position,
                "group_id": group_id,
                "group_hash": group["group_hash"],
                "source_hash": group["source_hash"],
                "response_hash": group["response_hash"],
                "source_text": predictor["source_text"],
                "response_text": predictor["response_text"],
                "role": group["role"],
                "corpus": group["corpus"],
                "annotation_disposition": evaluator["annotation_disposition"],
                "label": evaluator["label"],
                "annotation_provenance": evaluator["annotation_provenance"],
            }
        )
    return joined


def _cell(
    group: Mapping[str, Any],
    *,
    arm: str,
    order_index: int,
    source_group: Mapping[str, Any],
) -> JsonDict:
    """Build one label-free native request and bind its complete text by hash."""

    order = OPTION_ORDERS[order_index]
    prompt = option_protocol.build_option_prompt(
        str(source_group["source_text"]), str(group["response_text"]), order
    )
    identity = {
        "group_id": group["group_id"],
        "group_hash": group["group_hash"],
        "source_group_id": source_group["group_id"],
        "source_hash": source_group["source_hash"],
        "response_hash": group["response_hash"],
        "arm": arm,
        "option_order": list(order),
        "prompt_sha256": canonical_hash(prompt),
    }
    return {
        "cell_id": canonical_hash(identity),
        "request_hash": canonical_hash(identity),
        "roster_position": group["roster_position"],
        "role": group["role"],
        **identity,
        "source_text": source_group["source_text"],
        "response_text": group["response_text"],
        "gold_label": group["label"] if arm == "full_source_response" else None,
        "eligible": None,
        "prompt_token_count": None,
        "disposition": "unstarted",
        "attempted": False,
        "error": None,
    }


def build_capture_plan(
    groups: Sequence[Mapping[str, Any]], *, control_group_count: int = CONTROL_GROUP_COUNT
) -> list[JsonDict]:
    """Freeze two main orders and one two-order source derangement panel."""

    calibration = [row for row in groups if row.get("role") == "calibration_tuning"]
    if control_group_count < 0 or control_group_count > len(calibration):
        raise CaptureError("control_group_count_invalid")
    registered = sorted(
        calibration,
        key=lambda row: _stable_text_hash(f"v654-control:{row['group_hash']}"),
    )[:control_group_count]
    donor_by_id: dict[str, Mapping[str, Any]] = {}
    if registered:
        for index, group in enumerate(registered):
            donor_by_id[str(group["group_id"])] = registered[(index + 1) % len(registered)]
        if control_group_count == 1:
            donor_pool = [row for row in groups if row["group_id"] != registered[0]["group_id"]]
            if not donor_pool:  # pragma: no cover - only malformed one-group fixtures reach this.
                raise CaptureError("control_derangement_impossible")
            donor_by_id[str(registered[0]["group_id"])] = donor_pool[0]
    plan: list[JsonDict] = []
    for group in groups:
        for order_index in range(2):
            plan.append(
                _cell(
                    group, arm="full_source_response", order_index=order_index, source_group=group
                )
            )
    for group in registered:
        donor = donor_by_id[str(group["group_id"])]
        for order_index in range(2):
            plan.append(
                _cell(group, arm="shuffled_source", order_index=order_index, source_group=donor)
            )
    return plan


def apply_token_ceiling(
    groups: Sequence[Mapping[str, Any]],
    plan: Sequence[Mapping[str, Any]],
    *,
    tokenize: Tokenizer,
    ceiling: int = TOKEN_CEILING,
) -> tuple[list[JsonDict], list[JsonDict]]:
    """Tokenize complete prompts and exclude an overlength order pair as one unit."""

    cells = deepcopy([dict(row) for row in plan])
    for cell in cells:
        prompt = option_protocol.build_option_prompt(
            str(cell["source_text"]), str(cell["response_text"]), cell["option_order"]
        )
        token_ids = list(tokenize(prompt, add_bos=True))
        cell["prompt_token_count"] = len(token_ids)
        cell["prompt_sha256"] = canonical_hash(prompt)
    overlength_pairs = {
        (str(cell["group_id"]), str(cell["arm"]))
        for cell in cells
        if int(cell["prompt_token_count"]) > ceiling
    }
    for cell in cells:
        excluded = (str(cell["group_id"]), str(cell["arm"])) in overlength_pairs
        cell["eligible"] = not excluded
        if excluded:
            cell["disposition"] = "excluded"
            cell["error"] = f"complete_prompt_over_{ceiling}_tokens"
    main_eligibility = {
        str(cell["group_id"]): bool(cell["eligible"])
        for cell in cells
        if cell["arm"] == "full_source_response"
    }
    qualified = []
    for group in groups:
        row = deepcopy(dict(group))
        row["eligible"] = main_eligibility[str(group["group_id"])]
        row["exclusion_reason"] = (
            None if row["eligible"] else f"complete_prompt_over_{ceiling}_tokens"
        )
        qualified.append(row)
    return qualified, cells


def softmax_by_option(logits: Mapping[str, Any]) -> dict[str, float]:
    """Map finite raw logits to stable option IDs with a stable softmax."""

    if set(logits) != set(OPTION_IDS):
        raise CaptureError("probability_logits_invalid")
    values = {key: float(logits[key]) for key in OPTION_IDS}
    if any(not math.isfinite(value) for value in values.values()):
        raise CaptureError("probability_logits_invalid")
    maximum = max(values.values())
    weights = {key: math.exp(value - maximum) for key, value in values.items()}
    total = sum(weights.values())
    return {key: weights[key] / total for key in OPTION_IDS}


def average_order_rows(original: Mapping[str, Any], reversed_row: Mapping[str, Any]) -> JsonDict:
    """Average only after each display order maps back to stable option IDs."""

    if (
        tuple(original.get("option_order") or ()) != OPTION_ORDERS[0]
        or tuple(reversed_row.get("option_order") or ()) != OPTION_ORDERS[1]
    ):
        raise CaptureError("order_pair_invalid")
    mapped = [
        softmax_by_option(original.get("raw_logits_by_option_id") or {}),
        softmax_by_option(reversed_row.get("raw_logits_by_option_id") or {}),
    ]
    return {
        "mapped_probabilities_by_order": mapped,
        "mean_probabilities_by_option_id": {
            option: (mapped[0][option] + mapped[1][option]) / 2 for option in OPTION_IDS
        },
    }


def _valid_complete_cell(row: Mapping[str, Any]) -> bool:
    """Require every raw field needed for an independently reusable model readout."""

    try:
        probabilities = softmax_by_option(row.get("raw_logits_by_option_id") or {})
    except (CaptureError, TypeError, ValueError):
        return False
    observed = row.get("probabilities_by_option_id")
    if not isinstance(observed, Mapping) or any(
        not math.isclose(float(observed.get(key, math.nan)), probabilities[key], abs_tol=1e-12)
        for key in OPTION_IDS
    ):
        return False
    prompt_ids = row.get("prompt_token_ids")
    label_ids = row.get("label_token_ids")
    prompt_count = row.get("prompt_token_count")
    hashes = (row.get("source_hash"), row.get("response_hash"), row.get("prompt_sha256"))
    return (
        row.get("disposition") == "complete"
        and row.get("attempted") is True
        and isinstance(prompt_ids, list)
        and bool(prompt_ids)
        and prompt_count == len(prompt_ids)
        and isinstance(label_ids, list)
        and len(label_ids) == len(set(label_ids)) == 2
        and row.get("requested_score_position") == prompt_count - 1
        and row.get("actual_last_evaluated_position") == prompt_count - 1
        and all(isinstance(value, str) and value.startswith("sha256:") for value in hashes)
        and isinstance(row.get("model_receipt_hash"), str)
        and str(row["model_receipt_hash"]).startswith("sha256:")
        and isinstance(row.get("prefill_s"), (int, float))
        and math.isfinite(float(row["prefill_s"]))
        and float(row["prefill_s"]) >= 0
        and row.get("generated_tokens") == 0
        and row.get("error") is None
    )


def reduce_capture(
    plan: Sequence[Mapping[str, Any]],
    observed_rows: Sequence[Mapping[str, Any]],
    *,
    minimums: Mapping[str, int] = ROLE_MINIMUMS,
) -> JsonDict:
    """Reconcile every planned cell while keeping completeness and support separate."""

    planned = [dict(row) for row in plan]
    by_cell = {str(row.get("cell_id")): dict(row) for row in observed_rows}
    duplicate_cells = len(by_cell) != len(observed_rows)
    reconciled: list[JsonDict] = []
    request_identity_valid = True
    for planned_row in planned:
        cell_id = str(planned_row["cell_id"])
        row = by_cell.get(cell_id, deepcopy(planned_row))
        request_identity_valid &= row.get("request_hash") == planned_row.get("request_hash")
        reconciled.append(row)
    extra_cells = sorted(set(by_cell) - {str(row["cell_id"]) for row in planned})
    dispositions = Counter(str(row.get("disposition")) for row in reconciled)
    terminal = {"complete", "failed", "censored", "excluded"}
    capture_complete = (
        not duplicate_cells
        and not extra_cells
        and len(reconciled) == len(planned)
        and all(row.get("disposition") in terminal for row in reconciled)
    )
    eligible = [row for row in reconciled if row.get("eligible") is not False]
    all_valid = request_identity_valid and all(_valid_complete_cell(row) for row in eligible)
    group_rows: list[JsonDict] = []
    role_summary: JsonDict = {}
    for role in ("training", "calibration_tuning"):
        role_plan = [
            row
            for row in planned
            if row.get("role") == role and row.get("arm") == "full_source_response"
        ]
        group_ids = list(dict.fromkeys(str(row["group_id"]) for row in role_plan))
        excluded_ids = {
            str(row["group_id"])
            for row in reconciled
            if row.get("role") == role
            and row.get("arm") == "full_source_response"
            and row.get("disposition") == "excluded"
        }
        completed_ids = {
            group_id
            for group_id in group_ids
            if len(
                [
                    row
                    for row in reconciled
                    if row.get("group_id") == group_id
                    and row.get("arm") == "full_source_response"
                    and _valid_complete_cell(row)
                ]
            )
            == 2
        }
        role_summary[role] = {
            "planned": len(group_ids),
            "eligible": len(group_ids) - len(excluded_ids),
            "excluded": len(excluded_ids),
            "completed": len(completed_ids),
            "minimum": int(minimums.get(role, 0)),
        }
    support = all(
        role_summary[role]["eligible"] >= int(minimums.get(role, 0)) for role in role_summary
    )
    for group_id in dict.fromkeys(
        str(row["group_id"]) for row in planned if row["arm"] == "full_source_response"
    ):
        pair = [
            row
            for row in reconciled
            if row.get("group_id") == group_id and row.get("arm") == "full_source_response"
        ]
        base = pair[0]
        summary: JsonDict = {
            "unit_id": group_id,
            "group_id": group_id,
            "roster_position": base["roster_position"],
            "role": base["role"],
            "source_hash": base["source_hash"],
            "response_hash": base["response_hash"],
            "eligible": all(row.get("disposition") != "excluded" for row in pair),
            "status": "complete"
            if all(_valid_complete_cell(row) for row in pair)
            else str(pair[0].get("disposition")),
            "error": next((row.get("error") for row in pair if row.get("error")), None),
        }
        if len(pair) == 2 and all(_valid_complete_cell(row) for row in pair):
            ordered = sorted(pair, key=lambda row: tuple(row["option_order"]) != OPTION_ORDERS[0])
            summary.update(average_order_rows(ordered[0], ordered[1]))
            summary["gold_label"] = ordered[0].get("gold_label")
        group_rows.append(summary)
    sample_size_budget = {
        "planned": len(planned),
        "attempted": sum(row.get("attempted") is True for row in reconciled),
        "complete": dispositions["complete"],
        "failed": dispositions["failed"],
        "censored": dispositions["censored"],
        "excluded": dispositions["excluded"],
        "unstarted": dispositions["unstarted"],
    }
    return {
        "capture_complete_score": int(capture_complete),
        "confirmatory_support_score": int(support),
        "all_eligible_calls_valid": all_valid,
        "request_identity_valid": request_identity_valid,
        "duplicate_cells": duplicate_cells,
        "extra_cells": extra_cells,
        "sample_size_budget": sample_size_budget,
        "role_counts": role_summary,
        "group_rows": group_rows,
        "reconciled_rows": reconciled,
    }


def checkpoint_binding(
    plan: Sequence[Mapping[str, Any]], model_sha256: str, tokenizer_sha256: str
) -> JsonDict:
    """Bind resume data to the exact model, tokenizer, and request schedule."""

    return {
        "schedule_sha256": canonical_hash(plan),
        "model_sha256": model_sha256,
        "tokenizer_sha256": tokenizer_sha256,
    }


def write_checkpoint(
    path: Path, *, binding: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]
) -> None:
    """Atomically save all completed dispositions after one group boundary."""

    row_values = deepcopy([dict(row) for row in rows])
    atomic_json(
        path,
        {
            "schema": "carnot.exp7479.checkpoint.v1",
            "binding": dict(binding),
            "rows": row_values,
            "rows_sha256": canonical_hash(row_values),
        },
    )


def load_checkpoint(path: Path, *, expected_binding: Mapping[str, Any]) -> list[JsonDict]:
    """Resume only rows whose complete request binding is unchanged."""

    value = load_json_object(path)
    if value.get("schema") != "carnot.exp7479.checkpoint.v1":
        raise CaptureError("checkpoint_schema_invalid")
    if value.get("binding") != dict(expected_binding):
        raise CaptureError("checkpoint_binding_mismatch")
    rows = value.get("rows")
    if not isinstance(rows, list) or value.get("rows_sha256") != canonical_hash(
        rows
    ):  # pragma: no cover - corruption branch.
        raise CaptureError("checkpoint_rows_invalid")
    return [dict(row) for row in rows if isinstance(row, Mapping)]


def _jsonl_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    """Serialize JSONL with stable bytes for independent hashing."""

    return b"".join(
        (json.dumps(dict(row), sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
        for row in rows
    )


def _atomic_bytes(path: Path, payload: bytes) -> None:
    """Replace one task-owned shard only after all bytes reach disk."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def write_raw_logit_shards(
    raw_dir: Path,
    *,
    plan: Sequence[Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
) -> list[JsonDict]:
    """Write a plan shard and bounded role shards for all raw cell evidence."""

    payloads = [("capture-plan.jsonl", "plan", list(plan))]
    for role in ("training", "calibration_tuning"):
        payloads.append(
            (
                f"raw-logits-{role}.jsonl",
                "raw_logits",
                [row for row in rows if row.get("role") == role],
            )
        )
    manifest: list[JsonDict] = []
    for name, kind, values in payloads:
        payload = _jsonl_bytes(values)
        if len(payload) >= MAX_ARTIFACT_BYTES:  # pragma: no cover - fixed roster guard.
            raise CaptureError(f"raw_shard_too_large:{name}")
        _atomic_bytes(raw_dir / name, payload)
        manifest.append(
            {
                "path": name,
                "kind": kind,
                "rows": len(values),
                "size_bytes": len(payload),
                "sha256": "sha256:" + hashlib.sha256(payload).hexdigest(),
            }
        )
    return manifest


def reload_raw_logit_shards(raw_dir: Path, manifest: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Rehash each shard and recover plan and rows without trusting headlines."""

    plan: list[JsonDict] = []
    rows: list[JsonDict] = []
    for shard in manifest:
        path = raw_dir / str(shard["path"])
        if not path.is_file() or sha256_file(path) != shard.get("sha256"):
            raise CaptureError(f"raw_shard_hash_mismatch:{shard.get('path')}")
        values = _load_jsonl(path)
        if len(values) != shard.get("rows"):  # pragma: no cover - hash already catches edits.
            raise CaptureError(f"raw_shard_count_mismatch:{shard.get('path')}")
        if shard.get("kind") == "plan":
            plan.extend(values)
        else:
            rows.extend(values)
    return {"plan_hash": canonical_hash(plan), "plan": plan, "rows": rows}


def build_acceptance_gates(
    *, required_validity: bool, capture_complete: bool, support: bool, benefit: bool
) -> list[JsonDict]:
    """Keep evidence validity, readiness, support, and benefit independent."""

    values = (
        (
            "authenticated_capture_evidence",
            "required_validity",
            True,
            required_validity,
            "A positive scientific metric cannot excuse invalid evidence.",
        ),
        (
            "capture_complete",
            "readiness",
            True,
            capture_complete,
            "A valid null must not suppress an independent measurement.",
        ),
        (
            "confirmatory_role_support",
            "readiness",
            True,
            support,
            "A valid null must not suppress an independent measurement.",
        ),
        (
            "scientific_benefit_not_claimed",
            "scientific_benefit",
            False,
            benefit,
            "A small sample, a favorable seed or an analytic fixture cannot substitute for held-out value.",
        ),
    )
    return [
        {
            "check": check,
            "category": category,
            "expected": expected,
            "observed": observed,
            "op": "==",
            "passed": observed == expected,
            "principle": principle,
        }
        for check, category, expected, observed, principle in values
    ]


def gate_check_summary(gates: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Name the first failed check without hiding later failures."""

    failed = [row for row in gates if row.get("passed") is not True]
    if not failed:
        return {"passed": True, "failed_check": None, "failed_count": 0}
    first = failed[0]
    return {
        "passed": False,
        "failed_check": first.get("check"),
        "upstream": "current_exp7479_capture",
        "path": RAW_DIR.as_posix(),
        "field": first.get("check"),
        "expected": first.get("expected"),
        "observed": first.get("observed"),
        "op": first.get("op"),
        "failed_count": len(failed),
    }


def artifact_checksum(value: Mapping[str, Any]) -> str:
    """Hash the complete artifact without its self-referential checksum."""

    copied = deepcopy(dict(value))
    copied.pop("reproducibility_checksum", None)
    return canonical_hash(copied)


def validation_names_passed(receipts: object, names: Sequence[str]) -> bool:
    """Require exactly one passing receipt for every frozen command name."""

    if not isinstance(receipts, list):
        return False
    rows = [row for row in receipts if isinstance(row, Mapping)]
    return all(
        sum(row.get("name") == name and row.get("passed") is True for row in rows) == 1
        for name in names
    )


def _principles_for_artifact(artifact: Mapping[str, Any]) -> dict[str, str]:
    """Give every terminal field a short audit purpose."""

    return {
        key: FIELD_PRINCIPLES.get(
            key, "This field preserves one required part of the terminal capture audit."
        )
        for key in artifact
        if key != "field_principles"
    } | {"field_principles": FIELD_PRINCIPLES["field_principles"]}


def _fixture_receipts() -> list[JsonDict]:
    """Build named validation receipts used only by the deterministic fixture."""

    return [
        {"name": name, "passed": True, "exit_code": 0, "timed_out": False}
        for name in (*AFFECTED_CHECK_NAMES, *TERMINAL_CHECK_NAMES)
    ]


def _artifact_from_evidence(
    *,
    groups: Sequence[Mapping[str, Any]],
    reduction: Mapping[str, Any],
    shards: Sequence[Mapping[str, Any]],
    events: Sequence[Mapping[str, Any]],
    preconditions: Sequence[Mapping[str, Any]],
    source_hashes: Mapping[str, Any],
    model_receipt: Mapping[str, Any],
    device_identity: Mapping[str, Any],
    gpu_lease: Mapping[str, Any],
    phase_spans: Sequence[Mapping[str, Any]],
    durations: Mapping[str, Any],
    validation_receipts: Sequence[Mapping[str, Any]],
    started_at: str,
    started_ns: int,
    ended_ns: int,
    require_terminal: bool,
) -> JsonDict:
    """Assemble one schema-complete result from independently reducible evidence."""

    counts = pilot.reduce_invocation_events(events)
    required_names: Sequence[str] = AFFECTED_CHECK_NAMES
    if require_terminal:
        required_names = (*AFFECTED_CHECK_NAMES, *TERMINAL_CHECK_NAMES)
    validation_ok = validation_names_passed(list(validation_receipts), required_names)
    required_validity = (
        all(row.get("passed") is True for row in preconditions)
        and reduction["all_eligible_calls_valid"] is True
        and reduction["request_identity_valid"] is True
        and pilot.invocation_counts_balanced(counts)
        and model_receipt.get("observed_offload", {}).get("cuda_offload") is True
        and gpu_lease.get("release", {}).get("released") is True
        and validation_ok
    )
    gates = build_acceptance_gates(
        required_validity=required_validity,
        capture_complete=reduction["capture_complete_score"] == 1,
        support=reduction["confirmatory_support_score"] == 1,
        benefit=False,
    )
    ready = int(all(row["passed"] for row in gates))
    verdict = (
        "complete_null_fit_capture_ready_predictive_benefit_not_tested"
        if ready
        else "complete_disqualified_fit_capture_not_ready"
    )
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "status": verdict,
        "run_date": RUN_DATE,
        "started_at_utc": started_at,
        "ended_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "started_monotonic_ns": started_ns,
        "ended_monotonic_ns": ended_ns,
        "clock_identity": {
            "wall": "datetime.now(UTC)",
            "monotonic": "time.monotonic_ns",
            "pid": os.getpid(),
        },
        "preconditions_checked": deepcopy(list(preconditions)),
        "MODEL_SPECS": deepcopy(MODEL_SPECS),
        "model_specs": deepcopy(MODEL_SPECS),
        "model_invoked": counts["model_loads"]["attempted"] > 0
        or counts["forward_calls"]["attempted"] > 0,
        "invocation_counts": counts,
        "current_invocation_events": deepcopy(list(events)),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "inference_substrate_class": INFERENCE_SUBSTRATE_CLASS,
        "execution_venue": EXECUTION_VENUE,
        "device_identity": deepcopy(dict(device_identity)),
        "duration_s": (ended_ns - started_ns) / 1_000_000_000,
        "duration_components_s": deepcopy(dict(durations)),
        "phase_spans": deepcopy(list(phase_spans)),
        "random_seed": {
            "ordering": 65_500,
            "fit": [65_401, 65_402, 65_403, 65_404, 65_405],
            "audit": 65_509,
            "bootstrap": 65_510,
        },
        "source_artifact_hashes": deepcopy(dict(source_hashes)),
        "rows": deepcopy(reduction["group_rows"]),
        "sample_size_budget": deepcopy(reduction["sample_size_budget"]),
        "acceptance_gate_results": gates,
        "gate_check_summary": gate_check_summary(gates),
        "honest_verdict": verdict,
        "verdict_class": "null" if ready else "disqualified",
        "verifier_is_oracle": False,
        "flagged_adversarial": not ready,
        "validation_receipts": deepcopy(list(validation_receipts)),
        "fit_capture_ready_score": ready,
        "role_counts": deepcopy(reduction["role_counts"]),
        "raw_logit_shards": deepcopy(list(shards)),
        "capture_complete_score": reduction["capture_complete_score"],
        "confirmatory_support_score": reduction["confirmatory_support_score"],
        "capture_reduction": {
            key: deepcopy(value)
            for key, value in reduction.items()
            if key not in {"reconciled_rows", "group_rows"}
        },
        "model_receipt": deepcopy(dict(model_receipt)),
        "gpu_lease": deepcopy(dict(gpu_lease)),
        "comparison_plan": {
            "primary_readout": "mean_after_stable_option_id_remap",
            "primary_outcome": "paired_multiclass_brier",
            "bootstrap_draws": 10_000,
            "familywise_alpha": 0.05,
            "threshold_fit_role": "calibration_tuning_only",
            "decision_costs": {"false_accept": 10.0, "false_reject": 1.0, "escalate": 0.2},
            "benefit_measured": False,
        },
        "control_design": {
            "kind": "label_independent_source_derangement",
            "groups": sum(
                row.get("arm") == "shuffled_source" for row in reduction["reconciled_rows"]
            )
            // 2,
            "gold_labels_assigned": 0,
            "use": "source_sensitivity_only",
        },
        "capability_e2e": {
            "declared_entrypoint": WRAPPER_PATH.as_posix(),
            "fresh_process_cold_replay": "passed" if require_terminal else "pending",
            "independent_raw_reduction": "passed" if require_terminal else "pending",
            "numbered_runtime_e2e": "not_applicable_isolated_capture_experiment",
        },
        "validation_manifest": {
            "test_paths": list(VALIDATION_MANIFEST.test_paths),
            "changed_modules": list(VALIDATION_MANIFEST.changed_modules),
            "static_paths": list(VALIDATION_MANIFEST.static_paths),
        },
        "production_defaults_changed": False,
        "generator_weights_changed": False,
        "external_publication_authorized": False,
        "promotion_score": 0,
    }
    artifact["field_principles"] = _principles_for_artifact(artifact)
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return artifact


def build_artifact_for_test(root: Path) -> JsonDict:
    """Build a deterministic terminal fixture through production reducers."""

    groups = [
        {
            "roster_position": index,
            "group_id": f"group-{index}",
            "group_hash": canonical_hash(f"group-{index}"),
            "source_hash": canonical_hash(f"source-{index}"),
            "response_hash": canonical_hash(f"response-{index}"),
            "source_text": f"Source {index}",
            "response_text": f"Response {index}",
            "role": role,
            "annotation_disposition": "supported",
            "label": 1,
            "annotation_provenance": "pinned human annotations",
        }
        for index, role in enumerate(("training", "training", "calibration_tuning"))
    ]
    plan = build_capture_plan(groups, control_group_count=1)
    rows: list[JsonDict] = []
    for index, cell in enumerate(plan):
        logits = {cell["option_order"][0]: 3.0 + index / 100.0, cell["option_order"][1]: 1.0}
        rows.append(
            {
                **deepcopy(cell),
                "eligible": True,
                "disposition": "complete",
                "attempted": True,
                "prompt_token_ids": [1, 20 + index, 30 + index],
                "prompt_token_count": 3,
                "label_token_ids": [11, 17],
                "requested_score_position": 2,
                "actual_last_evaluated_position": 2,
                "raw_logits_by_option_id": logits,
                "probabilities_by_option_id": softmax_by_option(logits),
                "model_receipt_hash": "sha256:model",
                "prefill_s": 0.1,
                "generated_tokens": 0,
                "error": None,
            }
        )
    reduction = reduce_capture(plan, rows, minimums={"training": 2, "calibration_tuning": 1})
    raw_dir = root / "raw"
    shards = write_raw_logit_shards(raw_dir, plan=plan, rows=rows)
    events = pilot.fixture_invocation_events(forwards=len(rows))
    receipts = _fixture_receipts()
    model_receipt = {"observed_offload": {"cuda_offload": True}}
    lease = {"release": {"released": True}}
    artifact = _artifact_from_evidence(
        groups=groups,
        reduction=reduction,
        shards=shards,
        events=events,
        preconditions=[{"check": "fixture", "passed": True}],
        source_hashes={},
        model_receipt=model_receipt,
        device_identity={"cpu": "fixture", "selected_cuda": {"uuid": "fixture"}},
        gpu_lease=lease,
        phase_spans=[],
        durations={
            "model_load": 1.0,
            "forward": 1.0,
            "generation": 0.0,
            "numeric_reduction": 0.1,
            "validation": 0.1,
        },
        validation_receipts=receipts,
        started_at="2026-09-21T00:00:00Z",
        started_ns=1_000_000_000,
        ended_ns=3_000_000_000,
        require_terminal=True,
    )
    artifact["raw_logit_root"] = "raw"
    artifact["role_minimums"] = {"training": 2, "calibration_tuning": 1}
    artifact["field_principles"] = _principles_for_artifact(artifact)
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return artifact


def independent_reduce(
    artifact: Mapping[str, Any], *, root: Path, require_terminal: bool
) -> JsonDict:
    """Reload exact raw shards and derive all three readiness values."""

    raw_root = root / str(artifact.get("raw_logit_root", RAW_DIR.as_posix()))
    manifest = artifact.get("raw_logit_shards")
    if not isinstance(manifest, list):
        return {"passed": False, "errors": ["raw_logit_shards_invalid"]}
    try:
        raw = reload_raw_logit_shards(raw_root, manifest)
        minimums = artifact.get("role_minimums")
        minimum_values = minimums if isinstance(minimums, Mapping) else ROLE_MINIMUMS
        reduction = reduce_capture(raw["plan"], raw["rows"], minimums=minimum_values)
    except CaptureError as exc:
        return {"passed": False, "errors": [str(exc)]}
    errors: list[str] = []
    stored = artifact.get("capture_reduction")
    reduced_public = {
        key: value
        for key, value in reduction.items()
        if key not in {"reconciled_rows", "group_rows"}
    }
    if stored != reduced_public:
        errors.append("capture_reduction_mismatch")
    if artifact.get("rows") != reduction["group_rows"]:
        errors.append("group_rows_mismatch")
    if require_terminal and not validation_names_passed(
        artifact.get("validation_receipts"), (*AFFECTED_CHECK_NAMES, *TERMINAL_CHECK_NAMES)
    ):
        errors.append("required_validation_failed")
    return {"passed": not errors, "errors": errors, "reduction": reduction}


def validate_artifact(
    value: object, *, root: Path = REPO_ROOT, require_validation: bool = True
) -> list[str]:
    """Cold-check identity, raw reduction, readiness, counters, and checksum."""

    if not isinstance(value, Mapping):
        return ["artifact_not_object"]
    artifact = dict(value)
    errors = [f"missing_field:{field}" for field in REQUIRED_FIELDS if field not in artifact]
    expected = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "MODEL_SPECS": MODEL_SPECS,
        "model_specs": MODEL_SPECS,
        "inference_substrate_class": INFERENCE_SUBSTRATE_CLASS,
        "execution_venue": EXECUTION_VENUE,
        "verifier_is_oracle": False,
    }
    errors.extend(
        f"identity_mismatch:{key}"
        for key, wanted in expected.items()
        if artifact.get(key) != wanted
    )
    independent = independent_reduce(artifact, root=root, require_terminal=require_validation)
    if independent.get("passed") is not True:
        errors.extend(str(error) for error in independent.get("errors", []))
    reduction = independent.get("reduction")
    counts = artifact.get("invocation_counts")
    events = artifact.get("current_invocation_events")
    if not isinstance(counts, Mapping) or not isinstance(events, list):
        errors.append("invocation_evidence_invalid")
    elif counts != pilot.reduce_invocation_events(
        [row for row in events if isinstance(row, Mapping)]
    ):
        errors.append("invocation_counts_mismatch")
    elif not pilot.invocation_counts_balanced(counts):
        errors.append("invocation_counts_unbalanced")
    if isinstance(reduction, Mapping):
        expected_ready = int(
            reduction.get("capture_complete_score") == 1
            and reduction.get("confirmatory_support_score") == 1
            and reduction.get("all_eligible_calls_valid") is True
            and artifact.get("flagged_adversarial") is False
            and (
                not require_validation
                or validation_names_passed(
                    artifact.get("validation_receipts"),
                    (*AFFECTED_CHECK_NAMES, *TERMINAL_CHECK_NAMES),
                )
            )
        )
        if artifact.get("fit_capture_ready_score") != expected_ready:
            errors.append("fit_capture_ready_score_mismatch")
        if artifact.get("capture_complete_score") != reduction.get("capture_complete_score"):
            errors.append("capture_complete_score_mismatch")
        if artifact.get("confirmatory_support_score") != reduction.get(
            "confirmatory_support_score"
        ):
            errors.append("confirmatory_support_score_mismatch")
    gates = artifact.get("acceptance_gate_results")
    if not isinstance(gates, list) or any(
        not isinstance(row, Mapping) or not row.get("principle") for row in gates
    ):
        errors.append("acceptance_gates_invalid")
    elif artifact.get("gate_check_summary") != gate_check_summary(gates):
        errors.append("gate_summary_mismatch")
    principles = artifact.get("field_principles")
    if not isinstance(principles, Mapping) or not set(REQUIRED_FIELDS).issubset(principles):
        errors.append("field_principles_invalid")
    if require_validation and not validation_names_passed(
        artifact.get("validation_receipts"), (*AFFECTED_CHECK_NAMES, *TERMINAL_CHECK_NAMES)
    ):
        errors.append("required_validation_failed")
    if artifact.get("reproducibility_checksum") != artifact_checksum(artifact):
        errors.append("reproducibility_checksum_mismatch")
    return list(dict.fromkeys(errors))


def _source_receipt(path: Path, root: Path) -> JsonDict:  # pragma: no cover - live provenance.
    """Hash exact bytes and retain historical artifact flags without changing them."""

    relative = path.relative_to(root).as_posix() if path.is_relative_to(root) else str(path)
    receipt: JsonDict = {"path": relative, "sha256": sha256_file(path)}
    value = load_json_object(path) if path.suffix == ".json" else {}
    receipt["original_flagged_adversarial"] = value.get("flagged_adversarial")
    receipt["original_verdict_class"] = value.get("verdict_class")
    return receipt


def _precondition(
    check: str, path: str, field: str, expected: Any, observed: Any
) -> JsonDict:  # pragma: no cover - live provenance.
    """Record one exact prerequisite before dependent work starts."""

    return {
        "check": check,
        "upstream": path,
        "path": path,
        "field": field,
        "expected": expected,
        "observed": observed,
        "op": "==",
        "passed": observed == expected,
    }


def collect_preconditions(
    root: Path,
) -> tuple[list[JsonDict], JsonDict, dict[str, JsonDict]]:  # pragma: no cover
    """Authenticate qualified inputs, historical flags, model bytes, and CUDA."""

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
        Path("python/carnot/experiment_7462_v654_option_protocol.py"),
        Path("python/carnot/experiment_7477_v655_native_readout_pilot.py"),
        V654_RESULT_PATH,
        V655_QUALIFICATION_PATH,
        V655_PILOT_PATH,
        PREDICTOR_PATH,
        EVALUATOR_PATH,
        GROUP_PATH,
        Path("openspec/capabilities/verification/spec.md"),
        MODULE_PATH,
        WRAPPER_PATH,
        TEST_PATH,
    )
    checks: list[JsonDict] = []
    hashes: dict[str, JsonDict] = {}
    for relative in paths:
        path = root / relative
        available = path.is_file() and path.stat().st_size > 0
        checks.append(
            _precondition(
                f"source_bytes:{relative.as_posix()}",
                relative.as_posix(),
                "bytes",
                "readable_nonempty_bytes",
                "readable_nonempty_bytes" if available else None,
            )
        )
        if available:
            hashes[relative.as_posix()] = _source_receipt(path, root)
    qualification = load_json_object(root / V655_QUALIFICATION_PATH)
    pilot_artifact = load_json_object(root / V655_PILOT_PATH)
    disqualified_protocol = load_json_object(root / V654_RESULT_PATH)
    for relative, artifact, field, expected in (
        (V655_QUALIFICATION_PATH, qualification, "option_protocol_ready_score", 1),
        (V655_QUALIFICATION_PATH, qualification, "flagged_adversarial", False),
        (V655_PILOT_PATH, pilot_artifact, "native_readout_ready_score", 1),
        (V655_PILOT_PATH, pilot_artifact, "flagged_adversarial", False),
        (V654_RESULT_PATH, disqualified_protocol, "verdict_class", "disqualified"),
        (V654_RESULT_PATH, disqualified_protocol, "flagged_adversarial", True),
    ):
        checks.append(
            _precondition(
                f"upstream:{relative.name}:{field}",
                relative.as_posix(),
                field,
                expected,
                artifact.get(field),
            )
        )
    groups = load_sealed_fit_groups(root)
    checks.append(
        _precondition(
            "sealed_fit_role_counts",
            GROUP_PATH.as_posix(),
            "role_counts",
            {"training": 180, "calibration_tuning": 60},
            role_counts(groups),
        )
    )
    model = cached_current_model(gpu_index=0)
    model_path = Path(str(model.get("model_path"))) if model else Path("/absent-model")
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
            _precondition(
                "force_live",
                "environment:CARNOT_FORCE_LIVE",
                "value",
                "1",
                os.environ.get("CARNOT_FORCE_LIVE"),
            ),
            _precondition(
                "model_identity",
                str(model_path),
                "hf_id",
                MODEL_HF_ID,
                model.get("hf_id") if model else None,
            ),
            _precondition("cached_gguf", str(model_path), "is_file", True, model_path.is_file()),
            _precondition(
                "embedded_tokenizer", str(model_path), "suffix", ".gguf", model_path.suffix.lower()
            ),
            _precondition("idle_cuda_device", "nvidia-smi", "idle_candidate", True, bool(idle)),
            _precondition(
                "current_task_not_quarantined",
                "ops/exclusion_manifest.yaml",
                EXPERIMENT_ID,
                False,
                "7479" in (root / "ops/exclusion_manifest.yaml").read_text(encoding="utf-8"),
            ),
            _precondition(
                "driving_requirement",
                SPEC_PATH.relative_to(root).as_posix(),
                "REQ-*",
                "REQ-VERIFY-7479",
                "REQ-VERIFY-7479"
                if "REQ-VERIFY-7479" in SPEC_PATH.read_text(encoding="utf-8")
                else None,
            ),
        )
    )
    if model_path.is_file():
        hashes[str(model_path)] = _source_receipt(model_path, root)
        historical_hash = pilot_artifact.get("prompt_identity", {}).get("model_sha256")
        checks.append(
            _precondition(
                "pilot_weight_hash_matches_current_bytes",
                str(model_path),
                "model_sha256",
                historical_hash,
                hashes[str(model_path)]["sha256"],
            )
        )
    return (
        checks,
        {
            "groups": groups,
            "model": model,
            "model_path": model_path,
            "gpu_inventory": gpus,
            "selected_gpu": selected,
        },
        hashes,
    )


class CaptureNativeScorer(parity.NativeScorer):  # pragma: no cover - live Qwen boundary.
    """Use the pilot scorer with the preregistered 2,048-token context."""

    def load(self) -> None:
        """Load all layers on the leased device and keep the embedded tokenizer."""

        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.gpu_index)
        import llama_cpp  # noqa: PLC0415

        self.backend = llama_cpp
        self.llm = llama_cpp.Llama(
            model_path=str(self.model_path),
            n_ctx=TOKEN_CEILING,
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


class CaptureNativeReadoutRunner(
    pilot.NativeReadoutRunner
):  # pragma: no cover - live Qwen boundary.
    """Reuse the Exp7477 receipt layer with a 2,048-token native scorer."""

    def __init__(self, model_path: Path, gpu_index: int) -> None:
        self.scorer = CaptureNativeScorer(model_path, gpu_index)
        self.reset_epoch = 0
        self.identity: JsonDict = {}


def progress(started: float, phase: str, event: str, **details: Any) -> None:  # pragma: no cover
    """Print each phase and slow-operation boundary with monotonic elapsed time."""

    suffix = " ".join(f"{key}={value}" for key, value in sorted(details.items()))
    print(
        f"[exp7479] phase={phase} event={event} elapsed_s={time.monotonic() - started:.3f}"
        + (f" {suffix}" if suffix else ""),
        flush=True,
    )


def _phase_span(
    phase: str, phase_started: float, run_started: float, completed: int, checkpoint: str
) -> JsonDict:  # pragma: no cover
    """Record one phase boundary and its last durable unit."""

    ended = time.monotonic()
    return {
        "phase": phase,
        "start_s": phase_started - run_started,
        "end_s": ended - run_started,
        "duration_s": ended - phase_started,
        "completed_units": completed,
        "checkpoint": checkpoint,
    }


def _terminal_commands(root: Path, candidate: Path) -> list[PlannedCommand]:  # pragma: no cover
    """Build fresh-process replay and strict readers for the measured candidate."""

    relative = candidate.relative_to(root).as_posix()
    python = ".venv/bin/python"
    commands = (
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
    return [
        PlannedCommand(
            command, "safety" if command.name == "adversarial_verify" else "completion", True
        )
        for command in commands
    ]


def _merge_scored_cell(
    cell: Mapping[str, Any], scored: Mapping[str, Any], model_hash: str
) -> JsonDict:  # pragma: no cover
    """Attach the label-free request identity to one native runner receipt."""

    row = {**deepcopy(dict(cell)), **deepcopy(dict(scored))}
    row["option_order"] = deepcopy(cell["option_order"])
    row["source_hash"] = cell["source_hash"]
    row["response_hash"] = cell["response_hash"]
    row["prompt_sha256"] = cell["prompt_sha256"]
    row["model_receipt_hash"] = model_hash
    row["attempted"] = True
    row["error"] = None
    row.pop("source_text", None)
    row.pop("response_text", None)
    return row


def run_experiment(
    root: Path, run_date: str, *, output: Path = RESULT_PATH
) -> JsonDict:  # pragma: no cover
    """Run one owned Qwen load, at most 520 forwards, and scoped validation."""

    if run_date != RUN_DATE:
        raise ValueError(f"run_date_mismatch:{run_date}")
    run_started = time.monotonic()
    started_ns = time.monotonic_ns()
    started_at = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    spans: list[JsonDict] = []
    events: list[JsonDict] = []
    durations: JsonDict = {
        "model_load": 0.0,
        "forward": 0.0,
        "generation": 0.0,
        "numeric_reduction": 0.0,
        "validation": 0.0,
    }

    progress(run_started, "preconditions", "start")
    phase_started = time.monotonic()
    progress(run_started, "preconditions", "before_authentication")
    preconditions, context, source_hashes = collect_preconditions(root)
    progress(run_started, "preconditions", "after_authentication", checked=len(preconditions))
    if not all(row["passed"] for row in preconditions):
        failed = next(row for row in preconditions if not row["passed"])
        raise RuntimeError(f"precondition_failed:{failed['check']}:{failed['observed']}")
    groups = context["groups"]
    unqualified_plan = build_capture_plan(groups)
    if len(unqualified_plan) != FORWARD_BUDGET:
        raise RuntimeError("forward_schedule_budget_mismatch")
    spans.append(
        _phase_span(
            "preconditions", phase_started, run_started, len(preconditions), "authenticated"
        )
    )
    progress(run_started, "preconditions", "complete", planned=len(unqualified_plan))

    model_path = Path(context["model_path"])
    gpu = dict(context["selected_gpu"])
    lease = lease_api.GpuLease.acquire(
        runtime_dir=LEASE_RUNTIME_DIR,
        task_id=EXPERIMENT_ID,
        device_uuid=str(gpu["uuid"]),
        expected_model=str(model_path),
        vram_before_mb=int(gpu["memory_used_mb"]),
        ttl_s=MAX_LIVE_SECONDS + 300.0,
    )
    owner = lease.owner_receipt()
    runner = CaptureNativeReadoutRunner(model_path, int(gpu["index"]))
    model_receipt: JsonDict = {}
    release: JsonDict = {}
    qualified_groups: list[JsonDict] = []
    plan: list[JsonDict] = []
    captured_rows: list[JsonDict] = []
    live_started = time.monotonic()
    inference_ok = False
    try:
        lease.transition("admitted")
        lease.transition("loading")
        phase_started = time.monotonic()
        progress(run_started, "model_load", "before_model_load", gpu=gpu["index"])
        events.append(pilot._event("native-load", "model_loads", "attempted"))
        load_started = time.monotonic()
        try:
            model_receipt = pilot._call_with_heartbeats(
                lambda: runner.load(weight_sha256=source_hashes[str(model_path)]["sha256"]),
                started=run_started,
                phase="model_load",
            )
        except BaseException:
            events.append(pilot._event("native-load", "model_loads", "failed"))
            raise
        events.append(pilot._event("native-load", "model_loads", "completed"))
        durations["model_load"] = time.monotonic() - load_started
        resident_mb = parity._gpu_memory(int(gpu["index"]))
        model_receipt["observed_offload"] = {
            "owned_pid": os.getpid(),
            "device_uuid": gpu["uuid"],
            "resident_vram_mb": resident_mb,
            "baseline_vram_mb": gpu["memory_used_mb"],
            "cuda_offload": resident_mb - int(gpu["memory_used_mb"]) > 1_000,
        }
        model_receipt["lease_id"] = owner["lease_id"]
        model_receipt["context_tokens"] = TOKEN_CEILING
        lease.transition("resident", vram_mb=resident_mb)
        lease.transition("inferencing")
        spans.append(_phase_span("model_load", phase_started, run_started, 1, "resident"))
        progress(run_started, "model_load", "after_model_load", resident_vram_mb=resident_mb)

        phase_started = time.monotonic()
        progress(run_started, "length_qualification", "before_tokenization", groups=len(groups))
        qualified_groups, plan = apply_token_ceiling(
            groups, unqualified_plan, tokenize=runner.scorer.tokenize
        )
        plan_for_storage = [
            {
                key: value
                for key, value in row.items()
                if key not in {"source_text", "response_text"}
            }
            for row in plan
        ]
        tokenizer_hash = canonical_hash(model_receipt["tokenizer_identity"])
        binding = checkpoint_binding(
            plan_for_storage, model_receipt["model_sha256"], tokenizer_hash
        )
        checkpoint_path = root / RAW_DIR / "checkpoint.json"
        if checkpoint_path.is_file():
            captured_rows = load_checkpoint(checkpoint_path, expected_binding=binding)
        spans.append(
            _phase_span(
                "length_qualification",
                phase_started,
                run_started,
                len(qualified_groups),
                "eligibility_frozen",
            )
        )
        progress(
            run_started,
            "length_qualification",
            "after_tokenization",
            excluded=sum(row["eligible"] is False for row in qualified_groups),
        )

        existing = {str(row["cell_id"]): row for row in captured_rows}
        plan_by_group: dict[str, list[JsonDict]] = {}
        for cell in plan:
            plan_by_group.setdefault(str(cell["group_id"]), []).append(cell)
        model_hash = canonical_hash(model_receipt)
        phase_started = time.monotonic()
        progress(run_started, "native_capture", "before_forward_loop", planned=len(plan))
        for unit_index, group in enumerate(qualified_groups, start=1):
            group_id = str(group["group_id"])
            for cell in plan_by_group[group_id]:
                cell_id = str(cell["cell_id"])
                if cell_id in existing:
                    continue
                stored_cell = {
                    key: value
                    for key, value in cell.items()
                    if key not in {"source_text", "response_text"}
                }
                if cell["disposition"] == "excluded":
                    existing[cell_id] = stored_cell
                    continue
                if time.monotonic() - live_started >= MAX_LIVE_SECONDS:
                    existing[cell_id] = stored_cell
                    continue
                events.append(pilot._event(cell_id, "forward_calls", "attempted"))
                call_started = time.monotonic()
                try:
                    scored = runner.score(
                        call_id=cell_id,
                        source=str(cell["source_text"]),
                        response=str(cell["response_text"]),
                        option_order=cell["option_order"],
                        row_kind=str(cell["arm"]),
                        group_id=group_id,
                    )
                    row = _merge_scored_cell(cell, scored, model_hash)
                except BaseException as exc:  # noqa: BLE001 - retain one terminal failed cell.
                    events.append(pilot._event(cell_id, "forward_calls", "failed"))
                    existing[cell_id] = {
                        **stored_cell,
                        "disposition": "failed",
                        "attempted": True,
                        "error": f"{type(exc).__name__}:{exc}",
                    }
                else:
                    events.append(pilot._event(cell_id, "forward_calls", "completed"))
                    existing[cell_id] = row
                durations["forward"] += time.monotonic() - call_started
            captured_rows = [
                existing[str(cell["cell_id"])] for cell in plan if str(cell["cell_id"]) in existing
            ]
            write_checkpoint(checkpoint_path, binding=binding, rows=captured_rows)
            progress(
                run_started,
                "native_capture",
                "group_complete",
                completed=unit_index,
                planned=len(qualified_groups),
                cells=len(captured_rows),
            )
        captured_rows = [
            existing.get(
                str(cell["cell_id"]),
                {
                    key: value
                    for key, value in cell.items()
                    if key not in {"source_text", "response_text"}
                },
            )
            for cell in plan
        ]
        write_checkpoint(checkpoint_path, binding=binding, rows=captured_rows)
        spans.append(
            _phase_span(
                "native_capture",
                phase_started,
                run_started,
                len(captured_rows),
                "all_groups_accounted",
            )
        )
        progress(
            run_started,
            "native_capture",
            "after_forward_loop",
            completed=sum(row.get("disposition") == "complete" for row in captured_rows),
        )
        reduction = reduce_capture(plan_for_storage, captured_rows)
        inference_ok = (
            reduction["capture_complete_score"] == 1
            and reduction["all_eligible_calls_valid"] is True
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
    lease_receipt = {**owner, "release": release}

    phase_started = time.monotonic()
    reduction_started = time.monotonic()
    progress(run_started, "reduction", "before_benchmark", cells=len(captured_rows))
    plan_for_storage = [
        {key: value for key, value in row.items() if key not in {"source_text", "response_text"}}
        for row in plan
    ]
    raw_dir = root / RAW_DIR
    shards = write_raw_logit_shards(raw_dir, plan=plan_for_storage, rows=captured_rows)
    reduction = reduce_capture(plan_for_storage, captured_rows)
    durations["numeric_reduction"] = time.monotonic() - reduction_started
    spans.append(
        _phase_span(
            "reduction", phase_started, run_started, len(captured_rows), "raw_shards_sealed"
        )
    )
    progress(
        run_started,
        "reduction",
        "after_benchmark",
        capture_complete=reduction["capture_complete_score"],
    )

    private_root = Path(tempfile.mkdtemp(prefix="exp7479-validation-", dir="/tmp"))
    commands = build_command_plan(root, VALIDATION_MANIFEST, private_root)
    plan_errors = validate_command_plan(root, VALIDATION_MANIFEST, commands)
    if plan_errors:
        raise RuntimeError("validation_plan_invalid:" + ",".join(plan_errors))
    validation_started = time.monotonic()
    phase_started = time.monotonic()
    progress(run_started, "affected_validation", "before_subprocesses", planned=len(commands))
    affected = run_categorized_commands(
        root,
        [PlannedCommand(command, "required_validation", True) for command in commands],
        log_dir=raw_dir / "validation/affected",
        heartbeat_s=60,
    )
    affected_reduction = reduce_affected_receipts(root, VALIDATION_MANIFEST, affected)
    spans.append(
        _phase_span(
            "affected_validation", phase_started, run_started, len(affected), "affected_checks"
        )
    )
    progress(
        run_started,
        "affected_validation",
        "after_subprocesses",
        passed=affected_reduction["passed"],
    )
    if not affected_reduction["passed"]:
        raise RuntimeError("affected_validation_failed")

    device_identity = {
        "cpu": platform.processor() or platform.machine(),
        "platform": platform.platform(),
        "cuda_inventory": context["gpu_inventory"],
        "selected_cuda": gpu,
    }
    candidate = _artifact_from_evidence(
        groups=qualified_groups,
        reduction=reduction,
        shards=shards,
        events=events,
        preconditions=preconditions,
        source_hashes=source_hashes,
        model_receipt=model_receipt,
        device_identity=device_identity,
        gpu_lease=lease_receipt,
        phase_spans=spans,
        durations=durations,
        validation_receipts=affected,
        started_at=started_at,
        started_ns=started_ns,
        ended_ns=time.monotonic_ns(),
        require_terminal=False,
    )
    candidate["raw_logit_root"] = RAW_DIR.as_posix()
    candidate["role_minimums"] = deepcopy(ROLE_MINIMUMS)
    candidate["field_principles"] = _principles_for_artifact(candidate)
    candidate["reproducibility_checksum"] = artifact_checksum(candidate)
    candidate_path = raw_dir / "measured_terminal_candidate.json"
    atomic_json(candidate_path, candidate)

    terminal_plan = _terminal_commands(root, candidate_path)
    phase_started = time.monotonic()
    progress(run_started, "terminal_validation", "before_subprocesses", planned=len(terminal_plan))
    terminal = run_categorized_commands(
        root, terminal_plan, log_dir=raw_dir / "validation/terminal", heartbeat_s=60
    )
    spans.append(
        _phase_span(
            "terminal_validation", phase_started, run_started, len(terminal), "terminal_readers"
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
        raise RuntimeError("terminal_validation_failed")
    durations["validation"] = time.monotonic() - validation_started

    final = _artifact_from_evidence(
        groups=qualified_groups,
        reduction=reduction,
        shards=shards,
        events=events,
        preconditions=preconditions,
        source_hashes=source_hashes,
        model_receipt=model_receipt,
        device_identity=device_identity,
        gpu_lease=lease_receipt,
        phase_spans=spans,
        durations=durations,
        validation_receipts=[*affected, *terminal],
        started_at=started_at,
        started_ns=started_ns,
        ended_ns=time.monotonic_ns(),
        require_terminal=True,
    )
    final["raw_logit_root"] = RAW_DIR.as_posix()
    final["role_minimums"] = deepcopy(ROLE_MINIMUMS)
    final["field_principles"] = _principles_for_artifact(final)
    final["reproducibility_checksum"] = artifact_checksum(final)
    independent = independent_reduce(final, root=root, require_terminal=True)
    if independent.get("passed") is not True:
        raise RuntimeError(f"independent_reduction_failed:{independent.get('errors')}")
    errors = validate_artifact(final, root=root, require_validation=True)
    if errors:
        raise RuntimeError("terminal_artifact_invalid:" + ",".join(errors))
    progress(run_started, "publish", "before_atomic_publish", path=output)
    atomic_json(root / output, final)
    progress(run_started, "publish", "after_atomic_publish", verdict=final["honest_verdict"])
    return final


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the fixed live command and fresh-process reader modes."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", required=True)
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--output", type=Path, default=RESULT_PATH)
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--independent-reduce", type=Path)
    return parser.parse_args(argv)


def _run_live_cli(args: argparse.Namespace, root: Path) -> int:  # pragma: no cover - live E2E.
    """Run the live-only branch after the testable reader modes are resolved."""

    result = run_experiment(root, args.date, output=args.output)
    print(
        json.dumps(
            {
                "artifact": str(root / args.output),
                "honest_verdict": result["honest_verdict"],
                "fit_capture_ready_score": result["fit_capture_ready_score"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run the live capture or one fresh-process terminal reader."""

    args = parse_args(argv)
    if args.date != RUN_DATE:
        raise SystemExit(f"--date must be {RUN_DATE}")
    root = args.root.resolve()
    if args.cold_replay is not None:
        artifact = load_json_object(args.cold_replay)
        errors = (
            validate_artifact(artifact, root=root, require_validation=False)
            if artifact
            else ["artifact_unreadable"]
        )
        print(json.dumps({"errors": errors}, sort_keys=True), flush=True)
        return int(bool(errors))
    if args.independent_reduce is not None:
        artifact = load_json_object(args.independent_reduce)
        reduced = (
            independent_reduce(artifact, root=root, require_terminal=False)
            if artifact
            else {"passed": False, "errors": ["artifact_unreadable"]}
        )
        printable = {key: value for key, value in reduced.items() if key != "reduction"}
        print(json.dumps(printable, sort_keys=True), flush=True)
        return int(reduced.get("passed") is not True)
    return _run_live_cli(args, root)  # pragma: no cover - exercised by the capability E2E.


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
