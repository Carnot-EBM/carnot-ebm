"""Measure the 4096-context native tool-source intervention pilot.

The module keeps public development selection separate from labels. It scores
two stable options with a cached Qwen GGUF and generates no tokens.

Spec refs: REQ-VERIFY-7535 and SCENARIO-VERIFY-7535-*.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from datetime import UTC, datetime
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import tempfile
import threading
import time
from typing import Any

from carnot import experiment_7533_v659_tool_protocol as protocol
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
from carnot.reporting.current_work_receipt import atomic_json


JsonDict = dict[str, Any]
RUN_DATE = "20260922"
MILESTONE = "2026.09.659"
VERSION = "v659"
EXPERIMENT_ID = "exp7535-v659-native-pilot"
SCHEMA = "carnot.exp7535.v659.native_pilot.v1"
MODEL_ID = "unsloth/Qwen3.8-27B-GGUF"
MODEL_SPECS = [MODEL_ID]
SELECTION_SEED = 659035
N_CTX = 4096
PILOT_GROUPS = 12
FORWARD_BUDGET = 72
CAPTURE_FORWARDS = 1440
CAPTURE_LIMIT_S = 3600.0
VALIDATION_RESERVE_S = 600.0
ADMISSION_WAIT_S = 300.0
GPU_IDLE_MAX_USED_MB = 500
OPTION_ORDERS = protocol.OPTION_ORDERS

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_PATH = Path("results/experiment_7535_v659_native_pilot.json")
RAW_DIR = Path("results/raw/experiment_7535_v659_native_pilot")
ROWS_PATH = RAW_DIR / "native_rows.jsonl"
CANDIDATE_PATH = RAW_DIR / "measured_terminal_candidate.json"
CHECKPOINT_PATH = RAW_DIR / "checkpoint.json"
MODULE_PATH = Path("python/carnot/experiment_7535_v659_native_pilot.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_7535_v659_native_pilot.py")
TEST_PATH = Path("tests/python/test_experiment_7535_v659_native_pilot.py")
SPEC_PATH = Path("openspec/capabilities/verification/spec.md")
UPSTREAM_PATH = Path("results/experiment_7533_v659_tool_protocol.json")

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
    Path("python/carnot/experiment_7477_v655_native_readout_pilot.py"),
    Path("python/carnot/experiment_7514_v657_service_trace.py"),
    Path("python/carnot/experiment_7531_b2_induction_gate_measurement.py"),
    Path("python/carnot/inference/sota_models.py"),
    UPSTREAM_PATH,
    SPEC_PATH,
    MODULE_PATH,
    WRAPPER_PATH,
    TEST_PATH,
)

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


class NativePilotError(ValueError):
    """Reject a panel, readout, forecast, or artifact that changed meaning."""


def sha256_text(value: str) -> str:
    """Hash exact UTF-8 bytes so retained prompt text stays identifiable."""

    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def canonical_hash(value: object) -> str:
    """Hash stable JSON bytes so a removed row changes the experiment identity."""

    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return sha256_text(payload)


def sha256_file(path: Path) -> str:  # pragma: no cover - live input boundary.
    """Hash one exact input incrementally without copying large model bytes."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _selection_rank(component_hash: object, stratum: int) -> str:
    return sha256_text(f"{SELECTION_SEED}:stratum:{stratum}:{component_hash}")


def _prompt_extent(row: Mapping[str, Any], token_count: Callable[[str], int]) -> int:
    values = [
        token_count(
            protocol.build_tool_prompt(
                str(row["context"]), str(row["question"]), str(row["answer"]), order
            )
        )
        for order in OPTION_ORDERS
    ]
    return max(int(value) for value in values)


def _donor(target: Mapping[str, Any], candidates: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    eligible = [
        row
        for row in candidates
        if row.get("tool_type") == target.get("tool_type")
        and row.get("component_hash") != target.get("component_hash")
        and row.get("context_hash") != target.get("context_hash")
        and row.get("answer_hash") != target.get("answer_hash")
    ]
    if not eligible:
        raise NativePilotError(f"development_donor_missing:{target.get('component_hash')}")
    return min(
        eligible,
        key=lambda row: (
            abs(int(row["prompt_max_tokens"]) - int(target["prompt_max_tokens"])),
            sha256_text(
                f"{SELECTION_SEED}:donor:{target.get('component_hash')}:{row.get('component_hash')}"
            ),
        ),
    )


def select_development_panel(
    candidates: Sequence[Mapping[str, Any]],
    *,
    excluded_component_hashes: set[str],
    token_count: Callable[[str], int],
) -> JsonDict:
    """Freeze twelve public training groups across complete-prompt lengths.

    The selector reads no evaluator fields. It uses one deterministic hash in
    each length stratum, then pairs each target with a different same-tool row.
    """

    public_keys = {
        "component_hash",
        "official_split",
        "tool_type",
        "context",
        "question",
        "answer",
        "context_hash",
        "answer_hash",
    }
    qualified: list[JsonDict] = []
    for source in candidates:
        if source.get("official_split") != "train":
            continue
        if str(source.get("component_hash")) in excluded_component_hashes:
            continue
        if not public_keys.issubset(source):
            continue
        row = {key: deepcopy(source[key]) for key in public_keys}
        extent = _prompt_extent(row, token_count)
        absent_extent = max(
            token_count(
                protocol.build_tool_prompt(
                    protocol.EMPTY_EVIDENCE_MARKER,
                    str(row["question"]),
                    str(row["answer"]),
                    order,
                )
            )
            for order in OPTION_ORDERS
        )
        row["prompt_max_tokens"] = max(extent, int(absent_extent))
        if int(row["prompt_max_tokens"]) <= N_CTX:
            qualified.append(row)
    qualified.sort(key=lambda row: (int(row["prompt_max_tokens"]), str(row["component_hash"])))
    if len(qualified) < PILOT_GROUPS:
        raise NativePilotError(f"development_capacity:{len(qualified)}")
    if len({str(row["tool_type"]) for row in qualified}) < 2:
        raise NativePilotError("tool_type_span")

    strata: list[list[JsonDict]] = [[] for _ in range(PILOT_GROUPS)]
    for index, row in enumerate(qualified):
        stratum = min(PILOT_GROUPS - 1, index * PILOT_GROUPS // len(qualified))
        strata[stratum].append(row)
    if any(not rows for rows in strata):  # pragma: no cover - capacity implies nonempty strata.
        raise NativePilotError("development_length_stratum_empty")
    selected = [
        min(rows, key=lambda row: _selection_rank(row["component_hash"], stratum))
        for stratum, rows in enumerate(strata)
    ]
    if len({str(row["tool_type"]) for row in selected}) < 2:
        primary = str(selected[0]["tool_type"])
        alternatives = [row for row in qualified if str(row["tool_type"]) != primary]
        if not alternatives:  # pragma: no cover - qualified tool span was already required.
            raise NativePilotError("tool_type_span")
        replacement = min(
            alternatives,
            key=lambda row: _selection_rank(row["component_hash"], PILOT_GROUPS - 1),
        )
        selected[-1] = replacement

    groups: list[JsonDict] = []
    for stratum, selected_row in enumerate(selected):
        target = deepcopy(selected_row)
        donor = _donor(target, qualified)
        target.update(
            {
                "role": "development_outside_exp7533_roles",
                "length_stratum": stratum,
                "donor_component_hash": donor["component_hash"],
                "donor_tool_type": donor["tool_type"],
                "donor_context": donor["context"],
                "donor_context_hash": donor["context_hash"],
                "selection_rank": _selection_rank(target["component_hash"], stratum),
            }
        )
        groups.append(target)
    groups.sort(key=lambda row: int(row["length_stratum"]))
    return {
        "seed": SELECTION_SEED,
        "eligible_count": len(qualified),
        "excluded_exp7533_role_count": len(excluded_component_hashes),
        "labels_read": 0,
        "groups": groups,
        "panel_sha256": canonical_hash(groups),
    }


def build_forward_schedule(
    groups: Sequence[Mapping[str, Any]], *, token_count: Callable[[str], int]
) -> list[JsonDict]:
    """Freeze three source conditions in both stable semantic option orders."""

    if len(groups) != PILOT_GROUPS:
        raise NativePilotError(f"pilot_group_count:{len(groups)}")
    schedule: list[JsonDict] = []
    for group_index, group in enumerate(groups):
        sources = {
            "original": str(group["context"]),
            "absent": protocol.EMPTY_EVIDENCE_MARKER,
            "donor": str(group["donor_context"]),
        }
        for condition, source in sources.items():
            for order_index, order in enumerate(OPTION_ORDERS):
                prompt = protocol.build_tool_prompt(
                    source, str(group["question"]), str(group["answer"]), order
                )
                count = int(token_count(prompt))
                if count > N_CTX:
                    raise NativePilotError(
                        f"prompt_overlength:{group.get('component_hash')}:{condition}:{count}"
                    )
                schedule.append(
                    {
                        "schedule_index": len(schedule),
                        "group_index": group_index,
                        "component_hash": group["component_hash"],
                        "donor_component_hash": group["donor_component_hash"],
                        "tool_type": group["tool_type"],
                        "length_stratum": group["length_stratum"],
                        "condition": condition,
                        "option_order_index": order_index,
                        "option_order": list(order),
                        "source_text": source,
                        "donor_context": str(group["donor_context"]),
                        "prompt": prompt,
                        "prompt_sha256": sha256_text(prompt),
                        "prompt_utf8_bytes": len(prompt.encode("utf-8")),
                        "prompt_token_count": count,
                        "readout_kind": "option_logits",
                        "generated_token_budget": 0,
                    }
                )
    if len(schedule) != FORWARD_BUDGET:  # pragma: no cover - fixed nested product is 12*3*2.
        raise NativePilotError(f"forward_schedule_budget:{len(schedule)}")
    return schedule


def _softmax_pair(logits: Mapping[str, Any], order: Sequence[str]) -> dict[str, float]:
    values = [float(logits[key]) for key in order]
    peak = max(values)
    weights = [math.exp(value - peak) for value in values]
    total = sum(weights)
    return {key: weight / total for key, weight in zip(order, weights, strict=True)}


def _row_checks(row: Mapping[str, Any], expected: Mapping[str, Any]) -> set[str]:
    failures: set[str] = set()
    identity_fields = (
        "schedule_index",
        "component_hash",
        "donor_component_hash",
        "condition",
        "option_order",
        "prompt",
        "prompt_sha256",
    )
    if any(row.get(key) != expected.get(key) for key in identity_fields):
        failures.add("schedule_identity")
    prompt = row.get("prompt")
    ids = row.get("prompt_token_ids")
    count = row.get("prompt_token_count")
    if (
        not isinstance(prompt, str)
        or row.get("prompt_sha256") != sha256_text(prompt)
        or row.get("prompt_utf8_bytes") != len(prompt.encode("utf-8"))
        or not isinstance(ids, list)
        or not isinstance(count, int)
        or len(ids) != count
        or count > N_CTX
    ):
        failures.add("prompt_token_custody")
    if row.get("requested_score_position") != count - 1 if isinstance(count, int) else True:
        failures.add("last_token_position")
    if row.get("actual_last_evaluated_position") != count - 1 if isinstance(count, int) else True:
        failures.add("last_token_position")
    order_value = row.get("option_order")
    order = list(order_value) if isinstance(order_value, list) else []
    mapping = row.get("label_to_option_id")
    if (
        tuple(order) not in OPTION_ORDERS
        or row.get("display_labels") != [" A", " B"]
        or not isinstance(mapping, Mapping)
        or mapping != dict(zip((" A", " B"), order, strict=True))
        or row.get("label_token_ids") is None
        or len(set(row.get("label_token_ids") or [])) != 2
    ):
        failures.add("semantic_order_mapping")
    logits = row.get("full_logits_by_option_id")
    probabilities = row.get("probabilities_by_option_id")
    try:
        if not isinstance(logits, Mapping) or set(logits) != set(order):
            raise ValueError
        if not isinstance(probabilities, Mapping) or set(probabilities) != set(order):
            raise ValueError
        expected_probabilities = _softmax_pair(logits, order)
        values = [float(probabilities[key]) for key in order]
        if any(not math.isfinite(value) for value in values):
            raise ValueError
        if not math.isclose(sum(values), 1.0, abs_tol=1e-9):
            raise ValueError
        if any(
            not math.isclose(float(probabilities[key]), expected_probabilities[key], abs_tol=1e-9)
            for key in order
        ):
            raise ValueError
    except (KeyError, TypeError, ValueError, OverflowError):
        failures.add("finite_probabilities")
    if row.get("generated_tokens") != 0:
        failures.add("zero_generation")
    if row.get("state_reset") is not True:
        failures.add("state_reset")
    return failures


def reduce_native_rows(
    rows: Sequence[Mapping[str, Any]], schedule: Sequence[Mapping[str, Any]]
) -> JsonDict:
    """Reduce raw forward evidence without treating prediction quality as validity."""

    failures: set[str] = set()
    if len(rows) != len(schedule) or len(schedule) != FORWARD_BUDGET:
        failures.add("schedule_identity")
    by_index = {
        int(row["schedule_index"]): row
        for row in rows
        if isinstance(row.get("schedule_index"), int)
    }
    if len(by_index) != len(rows):
        failures.add("schedule_identity")
    complete = 0
    errors = 0
    generated = 0
    condition_values: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for expected in schedule:
        index = int(expected["schedule_index"])
        row = by_index.get(index)
        if row is None:
            failures.add("schedule_identity")
            continue
        if row.get("disposition") != "complete" or row.get("error") is not None:
            errors += 1
            failures.add("complete_forward_custody")
            continue
        complete += 1
        generated += int(row.get("generated_tokens") or 0)
        failures.update(_row_checks(row, expected))
        probabilities = row.get("probabilities_by_option_id")
        if isinstance(probabilities, Mapping):
            try:
                condition_values[str(row["component_hash"])][str(row["condition"])].append(
                    float(probabilities["contains_unsupported"])
                )
            except (KeyError, TypeError, ValueError):
                failures.add("finite_probabilities")
    comparative_rows: list[JsonDict] = []
    for component_hash, values in sorted(condition_values.items()):
        if set(values) != {"original", "absent", "donor"} or any(
            len(cells) != 2 for cells in values.values()
        ):
            failures.add("semantic_order_mapping")
            continue
        absolute = {
            condition: statistics.fmean(cells) for condition, cells in sorted(values.items())
        }
        comparative_rows.append(
            {
                "component_hash": component_hash,
                "absolute_condition_probabilities": absolute,
                "original_minus_absent": absolute["original"] - absolute["absent"],
                "original_minus_donor": absolute["original"] - absolute["donor"],
                "disposition": "complete_comparative_row",
                "positive_claim": False,
            }
        )
    if len(comparative_rows) != PILOT_GROUPS:
        failures.add("comparative_row_count")
    return {
        "passed": not failures,
        "planned_forward_count": len(schedule),
        "attempted_forward_count": len(rows),
        "complete_forward_count": complete,
        "error_forward_count": errors,
        "generated_token_count": generated,
        "last_token_custody_valid": "last_token_position" not in failures,
        "prompt_token_custody_valid": "prompt_token_custody" not in failures,
        "semantic_order_mapping_valid": "semantic_order_mapping" not in failures,
        "finite_probabilities_valid": "finite_probabilities" not in failures,
        "failed_checks": sorted(failures),
        "comparative_rows": comparative_rows,
        "benefit_gate_applied": False,
    }


def _p95(values: Sequence[float]) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered or any(not math.isfinite(value) or value < 0 for value in ordered):
        raise NativePilotError("forward_durations_missing")
    position = (len(ordered) - 1) * 0.95
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def forecast_captures(
    *,
    load_seconds: float,
    forward_seconds: Sequence[float],
    fit_checkpoint_seconds: float,
    eval_checkpoint_seconds: float,
) -> JsonDict:
    """Forecast fixed fit and evaluation rosters without sample shrinking."""

    p95 = _p95(forward_seconds)

    def one(checkpoint_seconds: float) -> JsonDict:
        total = (
            float(load_seconds)
            + CAPTURE_FORWARDS * p95
            + float(checkpoint_seconds)
            + VALIDATION_RESERVE_S
        )
        return {
            "planned_forwards": CAPTURE_FORWARDS,
            "measured_load_seconds": float(load_seconds),
            "p95_forward_seconds": p95,
            "checkpoint_overhead_seconds": float(checkpoint_seconds),
            "validation_reserve_seconds": VALIDATION_RESERVE_S,
            "forecast_seconds": total,
            "limit_seconds": CAPTURE_LIMIT_S,
            "feasible": total <= CAPTURE_LIMIT_S,
        }

    fit = one(fit_checkpoint_seconds)
    evaluation = one(eval_checkpoint_seconds)
    return {
        "formula": "load + 1440*p95_forward + checkpoint + 600",
        "fit": fit,
        "evaluation": evaluation,
        "fit_capture_feasible_score": int(fit["feasible"]),
        "eval_capture_feasible_score": int(evaluation["feasible"]),
        "sample_shrinking_allowed": False,
        "transport_retuning_allowed": False,
    }


def reduce_invocation_events(events: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Count owned current operations and retain unfinished work as in-flight."""

    attempted = Counter(
        str(row.get("operation")) for row in events if row.get("state") == "attempted"
    )
    completed = Counter(
        str(row.get("operation")) for row in events if row.get("state") == "completed"
    )
    failures = sum(row.get("state") == "failed" for row in events)
    cancellations = sum(row.get("state") == "cancelled" for row in events)
    terminals = Counter(
        str(row.get("operation"))
        for row in events
        if row.get("state") in {"completed", "failed", "cancelled"}
    )
    return {
        "model_loads_attempted": attempted["model_load"],
        "model_loads_completed": completed["model_load"],
        "forward_calls_attempted": attempted["forward"],
        "forward_calls_completed": completed["forward"],
        "generation_calls_attempted": attempted["generation"],
        "generation_calls_completed": completed["generation"],
        "failures": failures,
        "cancellations": cancellations,
        "in_flight": sum(max(0, attempted[key] - terminals[key]) for key in attempted),
    }


def gate(
    check: str,
    category: str,
    expected: Any,
    observed: Any,
    op: str,
    passed: bool,
    principle: str,
    *,
    upstream: str,
    field: str,
    path: str | None = None,
) -> JsonDict:
    """Retain one gate's exact operands and the failure it prevents."""

    row: JsonDict = {
        "check": check,
        "category": category,
        "condition": f"{field} {op} expected",
        "expected": expected,
        "observed": observed,
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
    """Name every failure and preserve the first exact routing operand."""

    failed = [deepcopy(dict(row)) for row in gates if row.get("passed") is not True]
    return {
        "all_passed": not failed,
        "failed_count": len(failed),
        "failed_checks": [str(row.get("check")) for row in failed],
        "first_failure": failed[0] if failed else None,
    }


def artifact_checksum(value: Mapping[str, Any]) -> str:
    """Bind code, settings, roles, model identity, and every raw row receipt."""

    payload = deepcopy(dict(value))
    payload.pop("reproducibility_checksum", None)
    return canonical_hash(payload)


def _field_principles(fields: Sequence[str]) -> dict[str, str]:
    specific = {
        "schema": "Exact experiment, milestone, version, and date prevent contract drift.",
        "preconditions_checked": "Exact observations and hashes prevent fabricated readiness.",
        "MODEL_SPECS": "The current model ID prevents a hidden legacy substitution.",
        "model_specs": "Resolved path, hash, quantization, tokenizer, and runtime bind the model.",
        "model_invoked": "Current invocation status stays separate from historical calls.",
        "invocation_counts": "Balanced attempts and terminals expose failed or unfinished work.",
        "inference_substrate_class": "The actual load-without-generation class sets the duration floor.",
        "inference_substrate": "Native option scoring cannot be mistaken for embedding vectors or generation.",
        "execution_venue": "Real CUDA and CPU-only execution remain distinct evidence classes.",
        "duration_s": "Monotonic current work exposes implausible or stale compute claims.",
        "phase_spans": "Boundaries and completed units expose silent or unfinished operations.",
        "random_seed": "Frozen sampling, fitting, arrival, and bootstrap seeds prevent outcome selection.",
        "reproducibility_checksum": "One digest binds code, settings, roles, model, and raw receipts.",
        "rows": "Every compared development group retains absolute metrics and disposition.",
        "sample_size_budget": "Planned, attempted, complete, failed, censored, excluded, and unstarted units stay distinct.",
        "acceptance_gate_results": "Validity, readiness, support, and benefit cannot substitute for each other.",
        "gate_check_summary": "A block names exact upstream, field, expected, and observed values.",
        "honest_verdict": "A complete terminal prefix separates blocked, null, and positive outcomes.",
        "verdict_class": "The closed enum prevents external absence from becoming partial work.",
        "verifier_is_oracle": "Injected labels cannot establish natural factuality or formal correctness.",
        "flagged_adversarial": "Actual verifier findings cannot be cleared to open a gate.",
        "validation_receipts": "Exact commands, exits, hashes, and cold replay make validation auditable.",
        "native_tool_ready_score": "Bare 0/1 requires owned CUDA logit custody, not predictive quality.",
        "fit_capture_feasible_score": "Bare 0/1 binds the fixed fitting roster to the conservative forecast.",
        "eval_capture_feasible_score": "Bare 0/1 binds the fixed evaluation roster to its own forecast.",
        "pilot_cost_rows": "Per-forward length and duration prevent optimistic capture budgeting.",
        "readout_kind": "Option logits with zero output tokens distinguish scoring from generation.",
        "ownership_receipt": "GPU UUID and PID/start ticks prevent foreign compute from becoming evidence.",
    }
    return {
        field: specific.get(field, f"The {field} field preserves one auditable terminal operand.")
        for field in fields
    }


def _validation_passed(receipts: object) -> bool:
    if not isinstance(receipts, list):
        return False
    passed = {
        str(row.get("name"))
        for row in receipts
        if isinstance(row, Mapping) and row.get("passed") is True
    }
    return set((*AFFECTED_CHECK_NAMES, *TERMINAL_CHECK_NAMES)) <= passed


def _base_artifact(
    *,
    preconditions: Sequence[Mapping[str, Any]],
    source_hashes: Sequence[Mapping[str, Any]],
    duration_s: float,
    phase_spans: Sequence[Mapping[str, Any]],
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
        "title": "V659 4096-context native tool-source pilot",
        "preconditions_checked": [deepcopy(dict(row)) for row in preconditions],
        "source_artifact_hashes": [deepcopy(dict(row)) for row in source_hashes],
        "MODEL_SPECS": MODEL_SPECS,
        "model_specs": [],
        "model_invoked": False,
        "invocation_counts": reduce_invocation_events([]),
        "historical_invocation_counts": {},
        "inference_substrate_class": "model_load_no_generation",
        "inference_substrate": "live_llm_embedding_extraction",
        "readout_kind": "option_logits",
        "execution_venue": "host",
        "execution_device": "real_cuda_required",
        "duration_s": max(0.000001, float(duration_s)),
        "phase_spans": [deepcopy(dict(row)) for row in phase_spans],
        "duration_breakdown_s": {
            "current_measurement_and_validation": max(0.000001, float(duration_s)),
            "authoring": 0.0,
            "historical": 0.0,
        },
        "process_identity": {
            "pid": os.getpid(),
            "python": os.path.realpath(os.sys.executable),
            "clock": "time.monotonic",
        },
        "random_seed": {
            "sampling": SELECTION_SEED,
            "fitting": 659036,
            "arrival": 659037,
            "bootstrap": 659038,
        },
        "verifier_is_oracle": False,
        "flagged_adversarial": False,
        "adversarial_corrections": [],
        "positive_claim": False,
        "production_defaults_changed": False,
        "generator_weights_changed": False,
        "external_publication_authorized": False,
        "applicable_numbered_e2e": [],
        "private_llm_off_real_environment_smoke": "not_applicable_isolated_readout_pilot",
    }


def build_blocked_artifact(
    *,
    failed_gate: Mapping[str, Any],
    preconditions: Sequence[Mapping[str, Any]],
    source_hashes: Sequence[Mapping[str, Any]],
    duration_s: float,
    phase_spans: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Publish external absence without inventing model work or pilot rows."""

    artifact = _base_artifact(
        preconditions=preconditions,
        source_hashes=source_hashes,
        duration_s=duration_s,
        phase_spans=phase_spans,
    )
    gates = [deepcopy(dict(failed_gate))]
    reason = str(failed_gate.get("check") or "external_precondition")
    artifact.update(
        {
            "inference_substrate_class": "blocked_no_run",
            "inference_substrate": "blocked_no_run",
            "execution_device": "none_precondition_blocked",
            "planned_inference_substrate_class": "model_load_no_generation",
            "planned_inference_substrate": "live_llm_embedding_extraction",
            "rows": [],
            "raw_native_rows": None,
            "sample_size_budget": {
                "planned": FORWARD_BUDGET,
                "attempted": 0,
                "complete": 0,
                "excluded": 0,
                "failed": 0,
                "censored": 0,
                "unstarted": FORWARD_BUDGET,
            },
            "acceptance_gate_results": gates,
            "gate_check_summary": gate_check_summary(gates),
            "honest_verdict": f"complete_blocked_{reason}",
            "verdict_class": "blocked",
            "native_tool_ready_score": 0,
            "fit_capture_feasible_score": 0,
            "eval_capture_feasible_score": 0,
            "pilot_cost_rows": [],
            "capture_forecasts": None,
            "ownership_receipt": None,
            "development_panel": None,
            "transport_reduction": None,
            "validation_receipts": [],
            "complete": 1,
            "ready": 0,
            "benefit_measured": False,
            "honest_no_headroom_annotation": "external_precondition_absent",
        }
    )
    artifact["field_principles"] = _field_principles(tuple(artifact))
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return artifact


def _ready_gates(
    reduction: Mapping[str, Any], forecasts: Mapping[str, Any], validation_passed: bool
) -> list[JsonDict]:
    return [
        gate(
            "native_logit_custody",
            "validity",
            True,
            reduction.get("passed"),
            "is",
            reduction.get("passed") is True,
            "Invalid prompt, position, mapping, or probability evidence cannot support science.",
            upstream="raw_native_rows",
            field="transport_reduction.passed",
        ),
        gate(
            "fit_capture_forecast",
            "readiness",
            1,
            forecasts.get("fit_capture_feasible_score"),
            "==",
            forecasts.get("fit_capture_feasible_score") == 1,
            "A closed fixed fitting budget cannot become a ready capture.",
            upstream="capture_forecasts.fit",
            field="fit_capture_feasible_score",
        ),
        gate(
            "evaluation_capture_forecast",
            "readiness",
            1,
            forecasts.get("eval_capture_feasible_score"),
            "==",
            forecasts.get("eval_capture_feasible_score") == 1,
            "A closed fixed evaluation budget cannot become a ready capture.",
            upstream="capture_forecasts.evaluation",
            field="eval_capture_feasible_score",
        ),
        gate(
            "required_validation",
            "validity",
            True,
            validation_passed,
            "is",
            validation_passed,
            "Required scoped checks and cold replay must pass before publication.",
            upstream="validation_receipts",
            field="all_required_checks_passed",
        ),
        gate(
            "positive_claim",
            "benefit",
            False,
            False,
            "is",
            True,
            "Transport readiness and development effects do not establish predictive benefit.",
            upstream="development_only_pilot",
            field="positive_claim",
        ),
    ]


def build_complete_artifact(
    *,
    preconditions: Sequence[Mapping[str, Any]],
    source_hashes: Sequence[Mapping[str, Any]],
    model_specs: Sequence[Mapping[str, Any]],
    events: Sequence[Mapping[str, Any]],
    panel: Mapping[str, Any],
    rows_receipt: Mapping[str, Any],
    reduction: Mapping[str, Any],
    forecasts: Mapping[str, Any],
    ownership: Mapping[str, Any],
    pilot_cost_rows: Sequence[Mapping[str, Any]],
    validation_receipts: Sequence[Mapping[str, Any]],
    duration_s: float,
    phase_spans: Sequence[Mapping[str, Any]],
    require_validation: bool,
) -> JsonDict:
    """Build one complete null or disqualified terminal candidate."""

    validation_passed = (
        _validation_passed(list(validation_receipts)) if require_validation else True
    )
    gates = _ready_gates(reduction, forecasts, validation_passed)
    custody = reduction.get("passed") is True
    verdict_class = "null" if custody and validation_passed else "disqualified"
    verdict = (
        "complete_null_native_tool_transport_measured_benefit_unclaimed"
        if verdict_class == "null"
        else "complete_disqualified_native_tool_pilot_validation"
    )
    artifact = _base_artifact(
        preconditions=preconditions,
        source_hashes=source_hashes,
        duration_s=duration_s,
        phase_spans=phase_spans,
    )
    comparative = deepcopy(list(reduction.get("comparative_rows") or []))
    counts = reduce_invocation_events(events)
    artifact.update(
        {
            "model_specs": [deepcopy(dict(row)) for row in model_specs],
            "model_invoked": counts["model_loads_attempted"] > 0,
            "invocation_counts": counts,
            "current_invocation_events": [deepcopy(dict(row)) for row in events],
            "rows": comparative,
            "raw_native_rows": deepcopy(dict(rows_receipt)),
            "sample_size_budget": {
                "planned": FORWARD_BUDGET,
                "attempted": reduction.get("attempted_forward_count"),
                "complete": reduction.get("complete_forward_count"),
                "excluded": 0,
                "failed": reduction.get("error_forward_count"),
                "censored": 0,
                "unstarted": FORWARD_BUDGET - int(reduction.get("attempted_forward_count") or 0),
            },
            "development_panel": deepcopy(dict(panel)),
            "transport_reduction": deepcopy(dict(reduction)),
            "capture_forecasts": deepcopy(dict(forecasts)),
            "fit_capture_feasible_score": int(forecasts["fit_capture_feasible_score"]),
            "eval_capture_feasible_score": int(forecasts["eval_capture_feasible_score"]),
            "native_tool_ready_score": int(custody and validation_passed),
            "pilot_cost_rows": [deepcopy(dict(row)) for row in pilot_cost_rows],
            "ownership_receipt": deepcopy(dict(ownership)),
            "acceptance_gate_results": gates,
            "gate_check_summary": gate_check_summary(gates),
            "validation_receipts": [deepcopy(dict(row)) for row in validation_receipts],
            "honest_verdict": verdict,
            "verdict_class": verdict_class,
            "complete": 1,
            "ready": int(custody and validation_passed),
            "benefit_measured": False,
            "honest_no_headroom_annotation": "development_transport_only_no_predictive_claim",
        }
    )
    artifact["field_principles"] = _field_principles(tuple(artifact))
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return artifact


def _fixture_candidates() -> list[JsonDict]:
    rows: list[JsonDict] = []
    for index in range(48):
        tool = ("grep", "curl", "search")[index % 3]
        context = f"{tool} source {index} " + ("x " * (index + 2))
        answer = f"answer {index}"
        rows.append(
            {
                "component_hash": sha256_text(f"fixture-component-{index}"),
                "official_split": "train",
                "tool_type": tool,
                "context": context,
                "question": f"question {index}",
                "answer": answer,
                "context_hash": sha256_text(context.casefold()),
                "answer_hash": sha256_text(answer.casefold()),
            }
        )
    return rows


def _fixture_token_count(text: str) -> int:
    return len(text.split()) + 1


def _fixture_native_row(request: Mapping[str, Any], index: int) -> JsonDict:
    order = list(request["option_order"])
    logits = {order[0]: 2.0 + index / 100.0, order[1]: -1.0}
    probabilities = _softmax_pair(logits, order)
    prompt = str(request["prompt"])
    token_ids = list(range(1, int(request["prompt_token_count"]) + 1))
    return {
        **deepcopy(dict(request)),
        "call_id": f"fixture-{index:03d}",
        "disposition": "complete",
        "prompt_sha256": sha256_text(prompt),
        "prompt_utf8_bytes": len(prompt.encode("utf-8")),
        "prompt_token_ids": token_ids,
        "prompt_token_count": len(token_ids),
        "requested_score_position": len(token_ids) - 1,
        "actual_last_evaluated_position": len(token_ids) - 1,
        "display_labels": [" A", " B"],
        "label_token_ids": [10, 11],
        "label_to_option_id": dict(zip((" A", " B"), order, strict=True)),
        "full_logits_by_option_id": logits,
        "probabilities_by_option_id": probabilities,
        "generated_tokens": 0,
        "forward_seconds": 0.25 + index / 1000.0,
        "state_reset": True,
        "server_receipt": {
            "transport": "native_in_process",
            "pid": 123,
            "process_start_ticks": 456,
            "gpu_uuid": "GPU-fixture",
            "n_ctx": N_CTX,
        },
        "error": None,
    }


def build_artifact_for_test() -> JsonDict:
    """Build production-shaped circular fixture evidence through real reducers."""

    panel = select_development_panel(
        _fixture_candidates(), excluded_component_hashes=set(), token_count=_fixture_token_count
    )
    schedule = build_forward_schedule(panel["groups"], token_count=_fixture_token_count)
    native_rows = [_fixture_native_row(row, index) for index, row in enumerate(schedule)]
    reduction = reduce_native_rows(native_rows, schedule)
    forwards = [float(row["forward_seconds"]) for row in native_rows]
    forecasts = forecast_captures(
        load_seconds=2.0,
        forward_seconds=forwards,
        fit_checkpoint_seconds=1.0,
        eval_checkpoint_seconds=1.0,
    )
    events = [{"operation": "model_load", "state": state} for state in ("attempted", "completed")]
    for _ in native_rows:
        events.extend(
            [
                {"operation": "forward", "state": "attempted"},
                {"operation": "forward", "state": "completed"},
            ]
        )
    return build_complete_artifact(
        preconditions=[],
        source_hashes=[],
        model_specs=[
            {
                "hf_id": MODEL_ID,
                "model_path": "/fixture/Qwen3.8-27B-Q4_K_M.gguf",
                "model_sha256": "sha256:" + "a" * 64,
                "quantization": "Q4_K_M",
                "runtime": "llama_cpp_fixture",
                "embedded_tokenizer": True,
            }
        ],
        events=events,
        panel=panel,
        rows_receipt={
            "path": "/tmp/fixture-native-rows.jsonl",
            "sha256": canonical_hash(native_rows),
            "bytes": 1,
            "rows": FORWARD_BUDGET,
            "fixture_only": True,
        },
        reduction=reduction,
        forecasts=forecasts,
        ownership={
            "device_uuid": "GPU-fixture",
            "pid": 123,
            "pid_start_ticks": 456,
            "observed_offloaded_layers": 64,
        },
        pilot_cost_rows=[
            {
                "kind": "model_load",
                "measured_seconds": 2.0,
                "full_prompt_length": 0,
            },
            *[
                {
                    "kind": "native_forward",
                    "call_id": row["call_id"],
                    "full_prompt_length": row["prompt_token_count"],
                    "measured_seconds": row["forward_seconds"],
                }
                for row in native_rows
            ],
        ],
        validation_receipts=[],
        duration_s=30.0,
        phase_spans=[],
        require_validation=False,
    )


def validate_artifact(value: object, *, require_validation: bool = True) -> list[str]:
    """Cold-check identity, scores, counters, gates, checksum, and terminal state."""

    if not isinstance(value, Mapping):
        return ["artifact_not_object"]
    artifact = dict(value)
    errors: list[str] = []
    blocked = artifact.get("verdict_class") == "blocked"
    expected = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "version": VERSION,
        "run_date": RUN_DATE,
        "MODEL_SPECS": MODEL_SPECS,
        "inference_substrate_class": "blocked_no_run" if blocked else "model_load_no_generation",
        "inference_substrate": "blocked_no_run" if blocked else "live_llm_embedding_extraction",
        "readout_kind": "option_logits",
        "verifier_is_oracle": False,
        "positive_claim": False,
    }
    for key, wanted in expected.items():
        if artifact.get(key) != wanted:
            errors.append(f"{key}_mismatch")
    verdict_class = artifact.get("verdict_class")
    verdict = str(artifact.get("honest_verdict") or "")
    if verdict_class not in {
        "positive",
        "circular_positive",
        "null",
        "blocked",
        "disqualified",
        "partial",
    }:
        errors.append("verdict_class_invalid")
    if not verdict.startswith("complete_"):
        errors.append("honest_verdict_not_terminal")
    if artifact.get("reproducibility_checksum") != artifact_checksum(artifact):
        errors.append("reproducibility_checksum_mismatch")
    gates = artifact.get("acceptance_gate_results")
    if not isinstance(gates, list) or any(
        not isinstance(row, Mapping) or not row.get("principle") for row in gates
    ):
        errors.append("acceptance_gates_invalid")
    elif gate_check_summary(gates) != artifact.get("gate_check_summary"):
        errors.append("gate_check_summary_mismatch")
    for score in (
        "native_tool_ready_score",
        "fit_capture_feasible_score",
        "eval_capture_feasible_score",
    ):
        if type(artifact.get(score)) is not int or artifact.get(score) not in {0, 1}:
            errors.append(f"{score}_not_bare_binary")
    if verdict_class == "blocked":
        if artifact.get("rows") != [] or artifact.get("model_invoked") is not False:
            errors.append("blocked_artifact_fabricated_work")
        if artifact.get("native_tool_ready_score") != 0:
            errors.append("blocked_artifact_ready")
    else:
        counts = artifact.get("invocation_counts")
        events = artifact.get("current_invocation_events")
        if not isinstance(counts, Mapping) or not isinstance(events, list):
            errors.append("invocation_evidence_invalid")
        elif counts != reduce_invocation_events(
            [row for row in events if isinstance(row, Mapping)]
        ):
            errors.append("invocation_counts_mismatch")
    if require_validation and not _validation_passed(artifact.get("validation_receipts")):
        errors.append("required_validation_failed")
    principles = artifact.get("field_principles")
    if not isinstance(principles, Mapping) or not set(artifact) <= set(principles) | {
        "field_principles",
        "reproducibility_checksum",
    }:
        errors.append("field_principles_missing")
    return list(dict.fromkeys(errors))


def utc_now() -> str:  # pragma: no cover - live boundary.
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def progress(started: float, phase: str, event: str, **details: Any) -> None:  # pragma: no cover
    suffix = " ".join(f"{key}={value}" for key, value in sorted(details.items()))
    print(
        f"[exp7535] phase={phase} event={event} elapsed_s={time.monotonic() - started:.3f}"
        + (f" {suffix}" if suffix else ""),
        flush=True,
    )


def _phase_span(
    phase: str,
    phase_started: float,
    run_started: float,
    completed_units: int,
    checkpoint: str | None = None,
) -> JsonDict:  # pragma: no cover
    ended = time.monotonic()
    return {
        "phase": phase,
        "start_s": phase_started - run_started,
        "end_s": ended - run_started,
        "duration_s": ended - phase_started,
        "completed_units": completed_units,
        "checkpoint": checkpoint,
        "ended_at_utc": utc_now(),
    }


def _call_with_heartbeats(
    operation: Callable[[], Any], *, started: float, phase: str, pending: str
) -> Any:  # pragma: no cover
    result: list[Any] = []
    errors: list[BaseException] = []

    def target() -> None:
        try:
            result.append(operation())
        except BaseException as exc:  # noqa: BLE001 - owner re-raises exact failure.
            errors.append(exc)

    worker = threading.Thread(target=target, name=f"exp7535-{phase}", daemon=True)
    worker.start()
    while worker.is_alive():
        worker.join(timeout=60.0)
        if worker.is_alive():
            progress(started, phase, "pending", pending_operation=pending)
    if errors:
        raise errors[0]
    return result[0] if result else None


def _load_object(path: Path) -> JsonDict:  # pragma: no cover
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(value) if isinstance(value, Mapping) else {}


def _path_label(path: Path, root: Path) -> str:  # pragma: no cover
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _source_receipt(path: Path, root: Path) -> JsonDict:  # pragma: no cover
    return {
        "path": _path_label(path, root),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def _precondition(
    check: str,
    *,
    upstream: str,
    field: str,
    expected: Any,
    observed: Any,
    path: str | None = None,
    passed: bool | None = None,
) -> JsonDict:  # pragma: no cover
    return gate(
        check,
        "external_precondition",
        expected,
        observed,
        "==",
        observed == expected if passed is None else passed,
        "A missing or changed external input must block dependent model work.",
        upstream=upstream,
        field=field,
        path=path,
    )


def collect_preconditions(
    root: Path, started: float, run_date: str
) -> tuple[list[JsonDict], list[JsonDict], JsonDict]:  # pragma: no cover
    """Authenticate named bytes, upstream protocol, shards, model, and runtime."""

    checks: list[JsonDict] = [
        _precondition(
            "run_date",
            upstream="command_line",
            field="--date",
            expected=RUN_DATE,
            observed=run_date,
        )
    ]
    hashes: list[JsonDict] = []
    for relative in INPUT_PATHS:
        path = root / relative
        present = path.is_file() and path.stat().st_size > 0
        checks.append(
            _precondition(
                f"source_bytes:{relative.as_posix()}",
                upstream=relative.as_posix(),
                field="readable_nonempty_bytes",
                expected=True,
                observed=present,
                path=relative.as_posix(),
            )
        )
        if present:
            hashes.append(_source_receipt(path, root))
    specification = root / SPEC_PATH
    requirement_present = specification.is_file() and "REQ-VERIFY-7535" in specification.read_text(
        encoding="utf-8"
    )
    checks.append(
        _precondition(
            "requirement_present",
            upstream=SPEC_PATH.as_posix(),
            field="REQ-VERIFY-7535",
            expected=True,
            observed=requirement_present,
            path=SPEC_PATH.as_posix(),
        )
    )
    upstream = _load_object(root / UPSTREAM_PATH)
    for field, expected in (
        ("tool_protocol_ready_score", 1),
        ("verdict_class", "null"),
        ("flagged_adversarial", False),
        ("run_date", RUN_DATE),
    ):
        checks.append(
            _precondition(
                f"exp7533:{field}",
                upstream=UPSTREAM_PATH.as_posix(),
                field=field,
                expected=expected,
                observed=upstream.get(field),
                path=UPSTREAM_PATH.as_posix(),
            )
        )

    sealed = upstream.get("sealed_shards")
    if not isinstance(sealed, Mapping):
        checks.append(
            _precondition(
                "exp7533:sealed_shards",
                upstream=UPSTREAM_PATH.as_posix(),
                field="sealed_shards",
                expected="mapping",
                observed=None,
                path=UPSTREAM_PATH.as_posix(),
            )
        )
    else:
        for name, receipt_value in sealed.items():
            receipt = dict(receipt_value) if isinstance(receipt_value, Mapping) else {}
            raw_path = Path(str(receipt.get("path") or ""))
            path = raw_path if raw_path.is_absolute() else root / raw_path
            observed = sha256_file(path) if path.is_file() else None
            expected = receipt.get("sha256")
            checks.append(
                _precondition(
                    f"exp7533:sealed_shard:{name}",
                    upstream=UPSTREAM_PATH.as_posix(),
                    field=f"sealed_shards.{name}.sha256",
                    expected=expected,
                    observed=observed,
                    path=str(path),
                )
            )
            if path.is_file():
                hashes.append(_source_receipt(path, root))
    role_manifest = upstream.get("role_manifest")
    role_groups = role_manifest.get("groups") if isinstance(role_manifest, Mapping) else None
    role_hashes = {
        str(row.get("component_hash")) for row in (role_groups or []) if isinstance(row, Mapping)
    }
    checks.append(
        _precondition(
            "exp7533:frozen_role_components",
            upstream=UPSTREAM_PATH.as_posix(),
            field="role_manifest.groups.unique_component_count",
            expected=480,
            observed=len(role_hashes),
            path=UPSTREAM_PATH.as_posix(),
        )
    )
    model = cached_current_model(preferred_quant="Q4_K_M")
    model_path = Path(str(model.get("model_path"))) if model else Path("/nonexistent")
    model_ok = bool(
        model
        and model.get("hf_id") == MODEL_ID
        and "Q4_K_M" in model_path.name
        and model_path.is_file()
    )
    checks.append(
        _precondition(
            "cached_current_model",
            upstream="carnot.inference.sota_models.cached_current_model",
            field="hf_id_quantization_path",
            expected={"hf_id": MODEL_ID, "quantization": "Q4_K_M", "exists": True},
            observed={
                "hf_id": model.get("hf_id") if model else None,
                "quantization": "Q4_K_M" if "Q4_K_M" in model_path.name else None,
                "exists": model_path.is_file(),
            },
            path=str(model_path),
            passed=model_ok,
        )
    )
    model_hash: str | None = None
    if model_ok:
        progress(started, "preconditions", "before_model_hash", bytes=model_path.stat().st_size)
        model_hash = _call_with_heartbeats(
            lambda: sha256_file(model_path),
            started=started,
            phase="preconditions",
            pending="model_sha256",
        )
        progress(started, "preconditions", "after_model_hash", sha256=model_hash)
        hashes.append(
            {
                "path": str(model_path),
                "sha256": model_hash,
                "bytes": model_path.stat().st_size,
                "role": "current_model_weights",
            }
        )
    try:
        import llama_cpp  # noqa: PLC0415

        runtime_observed = {
            "available": True,
            "version": getattr(llama_cpp, "__version__", "unknown"),
            "module_path": str(Path(llama_cpp.__file__).resolve()),
        }
    except Exception as exc:  # noqa: BLE001 - exact external runtime observation.
        runtime_observed = {"available": False, "error": f"{type(exc).__name__}:{exc}"}
    checks.append(
        _precondition(
            "llama_cpp_runtime",
            upstream="python_environment",
            field="llama_cpp.available",
            expected=True,
            observed=runtime_observed["available"],
            path=str(runtime_observed.get("module_path")),
        )
    )
    return (
        checks,
        hashes,
        {
            "upstream": upstream,
            "excluded_role_hashes": role_hashes,
            "model": model or {},
            "model_path": model_path,
            "model_hash": model_hash,
            "runtime": runtime_observed,
        },
    )


def _gpu_inventory() -> list[JsonDict]:  # pragma: no cover - host hardware boundary.
    query = subprocess.run(
        (
            "nvidia-smi",
            "--query-gpu=index,uuid,name,memory.total,memory.used,memory.free",
            "--format=csv,noheader,nounits",
        ),
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    if query.returncode != 0:
        return []
    rows: list[JsonDict] = []
    for line in query.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) == 6:
            rows.append(
                {
                    "index": int(parts[0]),
                    "uuid": parts[1],
                    "name": parts[2],
                    "total_memory_mb": int(parts[3]),
                    "memory_used_mb": int(parts[4]),
                    "free_memory_mb": int(parts[5]),
                    "compute_apps": [],
                }
            )
    applications = subprocess.run(
        (
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ),
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    by_uuid = {str(row["uuid"]): row for row in rows}
    if applications.returncode == 0:
        for line in applications.stdout.splitlines():
            parts = [part.strip() for part in line.split(",")]
            if len(parts) == 4 and parts[0] in by_uuid:
                by_uuid[parts[0]]["compute_apps"].append(
                    {
                        "pid": int(parts[1]),
                        "process_name": parts[2],
                        "used_memory_mb": int(parts[3]),
                    }
                )
    return rows


def _admissible_gpu(inventory: Sequence[Mapping[str, Any]]) -> JsonDict | None:  # pragma: no cover
    own_pid = os.getpid()
    candidates: list[JsonDict] = []
    for source in inventory:
        row = deepcopy(dict(source))
        applications = [dict(app) for app in row.get("compute_apps") or []]
        foreign = [app for app in applications if app.get("pid") != own_pid]
        own = [app for app in applications if app.get("pid") == own_pid]
        own_memory = sum(int(app.get("used_memory_mb") or 0) for app in own)
        residual = max(0, int(row.get("memory_used_mb") or 0) - own_memory)
        row["own_pid"] = own_pid
        row["own_compute_apps"] = own
        row["foreign_compute_apps"] = foreign
        row["residual_used_memory_mb"] = residual
        if not foreign and residual < GPU_IDLE_MAX_USED_MB:
            candidates.append(row)
    if not candidates:
        return None
    return min(candidates, key=lambda row: (int(row["residual_used_memory_mb"]), int(row["index"])))


def wait_for_owned_gpu(started: float) -> tuple[JsonDict | None, JsonDict]:  # pragma: no cover
    """Wait at most 300 seconds for one idle GPU without touching foreign work."""

    configured = float(os.environ.get("CARNOT_EXP7535_ADMISSION_WAIT_S", ADMISSION_WAIT_S))
    wait_limit = min(ADMISSION_WAIT_S, max(0.0, configured))
    wait_started = time.monotonic()
    next_heartbeat = wait_started
    observations: list[JsonDict] = []
    while True:
        inventory = _gpu_inventory()
        selected = _admissible_gpu(inventory)
        observations.append(
            {
                "elapsed_s": time.monotonic() - wait_started,
                "inventory": inventory,
                "selected_uuid": selected.get("uuid") if selected else None,
            }
        )
        if selected is not None:
            return selected, {
                "wait_limit_s": wait_limit,
                "waited_s": time.monotonic() - wait_started,
                "observation_count": len(observations),
                "last_observation": observations[-1],
                "evictions_attempted": 0,
                "signals_sent": [],
            }
        now = time.monotonic()
        if now - wait_started >= wait_limit:
            return None, {
                "wait_limit_s": wait_limit,
                "waited_s": now - wait_started,
                "observation_count": len(observations),
                "last_observation": observations[-1] if observations else None,
                "evictions_attempted": 0,
                "signals_sent": [],
            }
        if now >= next_heartbeat:
            progress(
                started,
                "gpu_admission",
                "pending",
                completed_units=0,
                pending_operation="waiting_for_one_owned_gpu",
                foreign_gpu_count=sum(bool(row.get("compute_apps")) for row in inventory),
            )
            next_heartbeat = now + 60.0
        time.sleep(min(5.0, max(0.0, wait_limit - (now - wait_started))))


def _owned_vram_mb(pid: int, gpu_uuid: str) -> int | None:  # pragma: no cover
    inventory = _gpu_inventory()
    for gpu in inventory:
        if gpu.get("uuid") != gpu_uuid:
            continue
        for app in gpu.get("compute_apps") or []:
            if app.get("pid") == pid:
                return int(app.get("used_memory_mb") or 0)
    return None


def _load_panel_and_schedule(
    *,
    excluded_role_hashes: set[str],
    started: float,
) -> tuple[JsonDict, list[JsonDict], JsonDict]:  # pragma: no cover
    """Use the embedded GGUF vocabulary and only public parquet columns."""

    from llama_cpp import Llama  # noqa: PLC0415

    tokenizer = Llama(
        model_path=str(protocol.GGUF_PATH),
        vocab_only=True,
        n_ctx=N_CTX,
        verbose=False,
    )

    def token_count(text: str) -> int:
        try:
            return len(tokenizer.tokenize(text.encode("utf-8"), add_bos=True, special=True))
        except TypeError:
            return len(tokenizer.tokenize(text.encode("utf-8"), add_bos=True))

    try:
        train_paths, test_paths = protocol._dataset_paths()
        public_train = protocol._load_public_rows(train_paths, started=started)
        public_test = protocol._load_public_rows(test_paths, started=started)
        candidates, _ = protocol.build_components(
            [*public_train, *public_test], exposure_context_hashes=set()
        )
        train = [row for row in candidates if row.get("official_split") == "train"]
        qualified, _ = protocol._qualify_prompt_fit(train, token_count=token_count)
        panel = select_development_panel(
            qualified,
            excluded_component_hashes=excluded_role_hashes,
            token_count=token_count,
        )
        schedule = build_forward_schedule(panel["groups"], token_count=token_count)
        identity = {
            "representation": "embedded_gguf",
            "path": str(protocol.GGUF_PATH),
            "resolved_path": str(protocol.GGUF_PATH.resolve()),
            "n_ctx": N_CTX,
            "vocab_only": True,
            "model_weights_read": False,
        }
        return panel, schedule, identity
    finally:
        close = getattr(tokenizer, "close", None)
        if callable(close):
            close()
        del tokenizer
        gc.collect()


class NativeOptionRunner:  # pragma: no cover - real Qwen/CUDA boundary.
    """Own one llama.cpp model and expose final-token option logits only."""

    def __init__(self, model_path: Path, gpu_index: int, gpu_uuid: str) -> None:
        self.model_path = model_path
        self.gpu_index = gpu_index
        self.gpu_uuid = gpu_uuid
        self.llm: Any = None
        self.backend: Any = None
        self.label_token_ids: dict[str, int] = {}
        self.model_identity: JsonDict = {}
        self.reset_epoch = 0

    def load(self, *, model_hash: str, owner: Mapping[str, Any]) -> JsonDict:
        """Load the exact cached Qwen weights with a 4096-token context."""

        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.gpu_index)
        import llama_cpp  # noqa: PLC0415

        self.backend = llama_cpp
        self.llm = llama_cpp.Llama(
            model_path=str(self.model_path),
            n_ctx=N_CTX,
            n_batch=512,
            n_gpu_layers=-1,
            split_mode=0,
            main_gpu=0,
            logits_all=True,
            verbose=False,
        )
        for label in (" A", " B"):
            ids = self.tokenize(label, add_bos=False)
            if len(ids) != 1:
                raise NativePilotError(f"option_label_not_single_token:{label}:{ids}")
            self.label_token_ids[label] = ids[0]
        metadata = dict(getattr(self.llm, "metadata", {}) or {})
        backend_path = Path(llama_cpp.__file__).resolve()
        block_count = next(
            (
                int(value)
                for key, value in metadata.items()
                if str(key).endswith(".block_count") and str(value).isdigit()
            ),
            None,
        )
        self.model_identity = {
            "hf_id": MODEL_ID,
            "model_path": str(self.model_path),
            "model_sha256": model_hash,
            "quantization": "Q4_K_M",
            "runtime": {
                "name": "llama_cpp",
                "version": getattr(llama_cpp, "__version__", "unknown"),
                "module_path": str(backend_path),
                "module_sha256": sha256_file(backend_path),
            },
            "embedded_tokenizer": {
                "representation": "embedded_gguf",
                "metadata_model": metadata.get("tokenizer.ggml.model"),
                "bos_token_id": metadata.get("tokenizer.ggml.bos_token_id"),
                "eos_token_id": metadata.get("tokenizer.ggml.eos_token_id"),
                "label_token_ids": deepcopy(self.label_token_ids),
            },
            "n_ctx": N_CTX,
            "requested_offload": {"n_gpu_layers": -1, "split_mode": 0, "main_gpu": 0},
            "model_block_count": block_count,
            "owner": deepcopy(dict(owner)),
        }
        return deepcopy(self.model_identity)

    def tokenize(self, text: str, *, add_bos: bool) -> list[int]:
        try:
            return list(self.llm.tokenize(text.encode("utf-8"), add_bos=add_bos, special=True))
        except TypeError:
            return list(self.llm.tokenize(text.encode("utf-8"), add_bos=add_bos))

    def score(self, request: Mapping[str, Any], owner: Mapping[str, Any]) -> JsonDict:
        """Reset KV state, evaluate all prompt tokens, and retain two raw logits."""

        prompt = str(request["prompt"])
        input_ids = self.tokenize(prompt, add_bos=True)
        if len(input_ids) != int(request["prompt_token_count"]):
            raise NativePilotError(
                f"token_count_changed:{request.get('schedule_index')}:"
                f"{request.get('prompt_token_count')}:{len(input_ids)}"
            )
        boundaries = {
            label: self.tokenize(prompt + label, add_bos=True) == [*input_ids, token_id]
            for label, token_id in self.label_token_ids.items()
        }
        if not all(boundaries.values()):
            raise NativePilotError("option_token_boundary_changed")
        self.llm.reset()
        call_started = time.monotonic()
        self.llm.eval(input_ids)
        elapsed = time.monotonic() - call_started
        logits = self.llm.scores[len(input_ids) - 1]
        order = list(request["option_order"])
        display_logits = {
            label: float(logits[token_id]) for label, token_id in self.label_token_ids.items()
        }
        semantic_logits = {
            option_id: display_logits[label]
            for label, option_id in zip((" A", " B"), order, strict=True)
        }
        probabilities = _softmax_pair(semantic_logits, order)
        actual_tokens = getattr(
            self.llm, "n_tokens", getattr(self.llm, "_n_tokens", len(input_ids))
        )
        self.reset_epoch += 1
        return {
            **deepcopy(dict(request)),
            "call_id": f"native-{int(request['schedule_index']):03d}",
            "disposition": "complete",
            "prompt_token_ids": input_ids,
            "prompt_token_count": len(input_ids),
            "requested_score_position": len(input_ids) - 1,
            "actual_last_evaluated_position": int(actual_tokens) - 1,
            "display_labels": [" A", " B"],
            "label_token_ids": [self.label_token_ids[" A"], self.label_token_ids[" B"]],
            "label_to_option_id": dict(zip((" A", " B"), order, strict=True)),
            "display_logits": display_logits,
            "full_logits_by_option_id": semantic_logits,
            "probabilities_by_option_id": probabilities,
            "token_boundary_receipts": boundaries,
            "generated_tokens": 0,
            "forward_seconds": elapsed,
            "state_reset": True,
            "kv_state_id": f"reset-epoch-{self.reset_epoch:03d}",
            "server_receipt": {
                "transport": "native_in_process",
                "server_started": False,
                "pid": owner.get("pid"),
                "process_start_ticks": owner.get("pid_start_ticks"),
                "gpu_uuid": self.gpu_uuid,
                "n_ctx": N_CTX,
                "generated_tokens": 0,
            },
            "error": None,
        }

    def close(self) -> None:
        if self.llm is not None:
            close = getattr(self.llm, "close", None)
            if callable(close):
                close()
        self.llm = None
        gc.collect()


def _jsonl_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:  # pragma: no cover
    return b"".join(
        json.dumps(dict(row), sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
        + b"\n"
        for row in rows
    )


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> JsonDict:  # pragma: no cover
    payload = _jsonl_bytes(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    directory_fd = os.open(path.parent, os.O_DIRECTORY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    return {
        "path": path.relative_to(REPO_ROOT).as_posix(),
        "sha256": "sha256:" + hashlib.sha256(payload).hexdigest(),
        "bytes": len(payload),
        "rows": len(rows),
    }


def _read_jsonl(path: Path) -> list[JsonDict]:  # pragma: no cover
    rows: list[JsonDict] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            value = json.loads(line)
            if not isinstance(value, Mapping):
                raise NativePilotError("jsonl_row_not_object")
            rows.append(dict(value))
    return rows


def _receipt_path(receipt: Mapping[str, Any], root: Path) -> Path:  # pragma: no cover
    path = Path(str(receipt.get("path") or ""))
    return path if path.is_absolute() else root / path


def _receipt_valid(receipt: Mapping[str, Any], root: Path) -> bool:  # pragma: no cover
    path = _receipt_path(receipt, root)
    return (
        path.is_file()
        and path.stat().st_size == receipt.get("bytes")
        and sha256_file(path) == receipt.get("sha256")
        and path.stat().st_size < 20 * 1024 * 1024
    )


def independent_reduce(
    value: Mapping[str, Any], root: Path = REPO_ROOT
) -> JsonDict:  # pragma: no cover
    """Reload hash-bound schedule and forward rows without trusting headlines."""

    if value.get("verdict_class") == "blocked" or (
        value.get("rows") == []
        and value.get("model_invoked") is False
        and value.get("raw_native_rows") is None
    ):
        budget = value.get("sample_size_budget")
        passed = (
            value.get("rows") == []
            and value.get("model_invoked") is False
            and isinstance(budget, Mapping)
            and budget.get("unstarted") == FORWARD_BUDGET
        )
        return {"passed": passed, "mode": "blocked_no_measurement", "row_count": 0}
    raw = value.get("raw_native_rows")
    if not isinstance(raw, Mapping):
        return {"passed": False, "error": "raw_native_rows_missing"}
    schedule_receipt = raw.get("forward_schedule")
    rows_receipt = raw.get("native_rows")
    if not isinstance(schedule_receipt, Mapping) or not isinstance(rows_receipt, Mapping):
        return {"passed": False, "error": "raw_receipts_missing"}
    if not _receipt_valid(schedule_receipt, root) or not _receipt_valid(rows_receipt, root):
        return {"passed": False, "error": "raw_receipt_hash_mismatch"}
    schedule = _read_jsonl(_receipt_path(schedule_receipt, root))
    rows = _read_jsonl(_receipt_path(rows_receipt, root))
    reduction = reduce_native_rows(rows, schedule)
    declared = value.get("transport_reduction")
    return {
        "passed": reduction.get("passed") is True and reduction == declared,
        "row_count": len(rows),
        "schedule_count": len(schedule),
        "reduction_sha256": canonical_hash(reduction),
        "declared_reduction_sha256": canonical_hash(declared),
    }


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
    return [PlannedCommand(spec=spec, category=spec.scope, required=True) for spec in specs]


def _event(operation: str, state: str, call_id: str) -> JsonDict:  # pragma: no cover
    return {
        "operation": operation,
        "state": state,
        "call_id": call_id,
        "monotonic_ns": time.monotonic_ns(),
        "scope": "current",
    }


def _run_affected_validation(
    root: Path, started: float
) -> tuple[list[JsonDict], bool]:  # pragma: no cover
    private = Path(tempfile.mkdtemp(prefix="carnot-exp7535-validation-", dir="/tmp"))
    commands = build_command_plan(root, VALIDATION_MANIFEST, private)
    plan_errors = validate_command_plan(root, VALIDATION_MANIFEST, commands)
    if plan_errors:
        raise NativePilotError("validation_plan_invalid:" + ",".join(plan_errors))
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


def _refresh_artifact(
    artifact: JsonDict,
    *,
    receipts: Sequence[Mapping[str, Any]],
    spans: Sequence[Mapping[str, Any]],
    duration_s: float,
) -> None:  # pragma: no cover
    artifact["validation_receipts"] = [deepcopy(dict(row)) for row in receipts]
    artifact["phase_spans"] = [deepcopy(dict(row)) for row in spans]
    artifact["duration_s"] = max(0.000001, float(duration_s))
    artifact["duration_breakdown_s"]["current_measurement_and_validation"] = artifact["duration_s"]
    validation_ok = _validation_passed(artifact["validation_receipts"])
    for row in artifact.get("acceptance_gate_results") or []:
        if isinstance(row, dict) and row.get("check") == "required_validation":
            row["observed"] = validation_ok
            row["passed"] = validation_ok
    if artifact.get("verdict_class") != "blocked" and not validation_ok:
        artifact["honest_verdict"] = "complete_disqualified_required_validation"
        artifact["verdict_class"] = "disqualified"
        artifact["native_tool_ready_score"] = 0
        artifact["ready"] = 0
    artifact["gate_check_summary"] = gate_check_summary(
        artifact.get("acceptance_gate_results") or []
    )
    principle_fields = (*artifact.keys(), "field_principles", "reproducibility_checksum")
    artifact["field_principles"] = _field_principles(principle_fields)
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)


def _run_terminal_validation(
    root: Path,
    artifact: JsonDict,
    affected_receipts: Sequence[Mapping[str, Any]],
    spans: list[JsonDict],
    started: float,
) -> tuple[list[JsonDict], bool]:  # pragma: no cover
    candidate = root / CANDIDATE_PATH
    preliminary = [deepcopy(dict(row)) for row in affected_receipts]
    artifact["validation_receipts"] = preliminary
    artifact["field_principles"] = _field_principles(
        (*artifact.keys(), "field_principles", "reproducibility_checksum")
    )
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    atomic_json(candidate, artifact)
    commands = _terminal_commands(candidate)
    progress(started, "terminal_validation", "before_subprocesses", planned=len(commands))
    phase_started = time.monotonic()
    terminal = run_categorized_commands(
        root,
        commands,
        log_dir=root / RAW_DIR / "validation/terminal",
        heartbeat_s=60.0,
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
    progress(
        started,
        "terminal_validation",
        "after_subprocesses",
        completed_units=len(terminal),
    )
    receipts = [*preliminary, *[dict(row) for row in terminal]]
    passed = all(row.get("passed") is True for row in terminal)
    return receipts, passed


def _publish_terminal(
    root: Path,
    artifact: JsonDict,
    receipts: Sequence[Mapping[str, Any]],
    spans: Sequence[Mapping[str, Any]],
    started: float,
) -> int:  # pragma: no cover
    _refresh_artifact(
        artifact,
        receipts=receipts,
        spans=spans,
        duration_s=time.monotonic() - started,
    )
    errors = validate_artifact(artifact, require_validation=True)
    if errors:
        raise NativePilotError("terminal_artifact_invalid:" + ",".join(errors))
    progress(started, "publish", "before_atomic_write", path=RESULT_PATH.as_posix())
    atomic_json(root / RESULT_PATH, artifact)
    progress(started, "publish", "after_atomic_write", bytes=(root / RESULT_PATH).stat().st_size)
    return 0


def _finish_candidate(
    root: Path,
    artifact: JsonDict,
    affected: Sequence[Mapping[str, Any]],
    spans: list[JsonDict],
    started: float,
) -> int:  # pragma: no cover
    receipts, terminal_passed = _run_terminal_validation(root, artifact, affected, spans, started)
    if not terminal_passed and artifact.get("verdict_class") != "blocked":
        artifact["honest_verdict"] = "complete_disqualified_required_terminal_validation"
        artifact["verdict_class"] = "disqualified"
        artifact["native_tool_ready_score"] = 0
        artifact["ready"] = 0
    return _publish_terminal(root, artifact, receipts, spans, started)


def run_experiment(root: Path, run_date: str) -> int:  # pragma: no cover - live orchestration.
    """Authenticate, validate, admit one GPU, capture 72 forwards, and publish."""

    started = time.monotonic()
    spans: list[JsonDict] = []
    progress(started, "preconditions", "start")
    phase_started = time.monotonic()
    progress(started, "preconditions", "before_named_inputs")
    checks, source_hashes, context = collect_preconditions(root, started, run_date)
    progress(started, "preconditions", "after_named_inputs", completed_units=len(checks))
    spans.append(
        _phase_span("preconditions", phase_started, started, len(checks), "inputs_authenticated")
    )
    failed = next((row for row in checks if row.get("passed") is not True), None)

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
    if not affected_passed:
        validation_gate = gate(
            "required_affected_validation",
            "validity",
            True,
            False,
            "is",
            False,
            "Required scoped checks must pass before model work.",
            upstream="affected_validation_receipts",
            field="all_required_checks_passed",
        )
        artifact = build_blocked_artifact(
            failed_gate=validation_gate,
            preconditions=checks,
            source_hashes=source_hashes,
            duration_s=time.monotonic() - started,
            phase_spans=spans,
        )
        artifact["honest_verdict"] = "complete_disqualified_required_affected_validation"
        artifact["verdict_class"] = "disqualified"
        artifact["current_invocation_events"] = []
        artifact["field_principles"] = _field_principles(
            (*artifact.keys(), "field_principles", "reproducibility_checksum")
        )
        artifact["reproducibility_checksum"] = artifact_checksum(artifact)
        return _finish_candidate(root, artifact, affected, spans, started)
    if failed is not None:
        artifact = build_blocked_artifact(
            failed_gate=failed,
            preconditions=checks,
            source_hashes=source_hashes,
            duration_s=time.monotonic() - started,
            phase_spans=spans,
        )
        return _finish_candidate(root, artifact, affected, spans, started)

    progress(started, "gpu_admission", "before_wait", wait_limit_s=ADMISSION_WAIT_S)
    phase_started = time.monotonic()
    selected_gpu, admission = wait_for_owned_gpu(started)
    spans.append(
        _phase_span(
            "gpu_admission",
            phase_started,
            started,
            int(selected_gpu is not None),
            "one_gpu_selected" if selected_gpu else "wait_closed",
        )
    )
    progress(
        started,
        "gpu_admission",
        "after_wait",
        selected_uuid=selected_gpu.get("uuid") if selected_gpu else None,
        waited_s=admission["waited_s"],
    )
    admission_check = _precondition(
        "owned_gpu_available",
        upstream="nvidia-smi",
        field="admissible_single_gpu",
        expected=True,
        observed=selected_gpu is not None,
        path="nvidia-smi_compute_process_inventory",
    )
    checks.append(admission_check)
    if selected_gpu is None:
        artifact = build_blocked_artifact(
            failed_gate=admission_check,
            preconditions=checks,
            source_hashes=source_hashes,
            duration_s=time.monotonic() - started,
            phase_spans=spans,
        )
        artifact["gpu_admission_receipt"] = admission
        artifact["field_principles"] = _field_principles(
            (*artifact.keys(), "field_principles", "reproducibility_checksum")
        )
        artifact["reproducibility_checksum"] = artifact_checksum(artifact)
        return _finish_candidate(root, artifact, affected, spans, started)

    model_path = Path(context["model_path"])
    lease = lease_api.GpuLease.acquire(
        runtime_dir=root / RAW_DIR / "gpu_lease",
        task_id=EXPERIMENT_ID,
        device_uuid=str(selected_gpu["uuid"]),
        expected_model=str(model_path),
        vram_before_mb=int(selected_gpu["memory_used_mb"]),
        ttl_s=4800.0,
    )
    owner = lease.owner_receipt()
    runner = NativeOptionRunner(model_path, int(selected_gpu["index"]), str(selected_gpu["uuid"]))
    events: list[JsonDict] = []
    rows: list[JsonDict] = []
    schedule: list[JsonDict] = []
    panel: JsonDict = {}
    model_identity: JsonDict = {}
    load_seconds = 0.0
    checkpoint_seconds: list[float] = []
    ownership: JsonDict = {}
    inference_ok = False
    live_error: str | None = None
    schedule_receipt: JsonDict = {}
    rows_receipt: JsonDict = {}
    reduction: JsonDict = {}

    try:
        lease.transition("admitted")
        os.environ["CUDA_VISIBLE_DEVICES"] = str(selected_gpu["index"])
        phase_started = time.monotonic()
        progress(started, "panel_freeze", "before_tokenizer_and_public_rows")
        panel, schedule, tokenizer_identity = _call_with_heartbeats(
            lambda: _load_panel_and_schedule(
                excluded_role_hashes=set(context["excluded_role_hashes"]), started=started
            ),
            started=started,
            phase="panel_freeze",
            pending="vocab_tokenization_and_public_component_freeze",
        )
        schedule_receipt = _write_jsonl(root / RAW_DIR / "forward_schedule.jsonl", schedule)
        spans.append(
            _phase_span(
                "panel_freeze",
                phase_started,
                started,
                len(panel["groups"]),
                schedule_receipt["path"],
            )
        )
        progress(
            started,
            "panel_freeze",
            "after_tokenizer_and_public_rows",
            groups=len(panel["groups"]),
            forwards=len(schedule),
        )

        lease.transition("loading")
        phase_started = time.monotonic()
        progress(started, "model_load", "before", gpu_uuid=selected_gpu["uuid"])
        events.append(_event("model_load", "attempted", "qwen-load"))
        load_started = time.monotonic()
        try:
            model_identity = _call_with_heartbeats(
                lambda: runner.load(model_hash=str(context["model_hash"]), owner=owner),
                started=started,
                phase="model_load",
                pending="qwen_gguf_weight_load",
            )
        except BaseException:
            events.append(_event("model_load", "failed", "qwen-load"))
            raise
        load_seconds = time.monotonic() - load_started
        events.append(_event("model_load", "completed", "qwen-load"))
        owned_vram = _owned_vram_mb(os.getpid(), str(selected_gpu["uuid"]))
        block_count = model_identity.get("model_block_count")
        offload_real = isinstance(owned_vram, int) and owned_vram >= 15_000
        observed_layers = int(block_count) if offload_real and isinstance(block_count, int) else 0
        ownership = {
            **deepcopy(owner),
            "selected_gpu_index": selected_gpu["index"],
            "selected_gpu_uuid": selected_gpu["uuid"],
            "selected_gpu_name": selected_gpu["name"],
            "baseline_memory_used_mb": selected_gpu["memory_used_mb"],
            "owned_pid_vram_mb": owned_vram,
            "requested_offloaded_layers": -1,
            "measured_owned_offloaded_layers": observed_layers,
            "offload_measurement": "nvidia-smi owned PID VRAM plus GGUF block_count",
            "offload_real": offload_real,
            "admission_receipt": admission,
            "tokenizer_identity": tokenizer_identity,
            "signals_sent": [],
        }
        model_identity["observed_offload"] = {
            "gpu_uuid": selected_gpu["uuid"],
            "owned_pid": os.getpid(),
            "owned_pid_vram_mb": owned_vram,
            "offloaded_layers": observed_layers,
            "passed": offload_real and observed_layers > 0,
        }
        lease.transition("resident", vram_mb=int(owned_vram or 0))
        lease.transition("inferencing")
        spans.append(_phase_span("model_load", phase_started, started, 1, "qwen_resident"))
        progress(
            started,
            "model_load",
            "after",
            load_seconds=load_seconds,
            owned_vram_mb=owned_vram,
            offloaded_layers=observed_layers,
        )
        if not ownership["offload_real"] or ownership["measured_owned_offloaded_layers"] <= 0:
            raise NativePilotError("owned_cuda_offload_not_observed")

        phase_started = time.monotonic()
        progress(started, "native_forwards", "before_loop", planned=FORWARD_BUDGET)
        for index, request in enumerate(schedule):
            call_id = f"native-{index:03d}"
            progress(
                started,
                "native_forwards",
                "before_forward",
                completed_units=index,
                planned=FORWARD_BUDGET,
                call_id=call_id,
            )
            events.append(_event("forward", "attempted", call_id))
            try:
                row = runner.score(request, owner)
            except BaseException as exc:  # noqa: BLE001 - retain exact error row.
                row = {
                    **deepcopy(dict(request)),
                    "call_id": call_id,
                    "disposition": "failed",
                    "generated_tokens": 0,
                    "error": f"{type(exc).__name__}:{exc}",
                    "server_receipt": {
                        "transport": "native_in_process",
                        "pid": owner.get("pid"),
                        "process_start_ticks": owner.get("pid_start_ticks"),
                        "gpu_uuid": selected_gpu["uuid"],
                        "n_ctx": N_CTX,
                    },
                }
                events.append(_event("forward", "failed", call_id))
            else:
                events.append(_event("forward", "completed", call_id))
            rows.append(row)
            checkpoint_started = time.monotonic()
            atomic_json(
                root / CHECKPOINT_PATH,
                {
                    "schema": "carnot.exp7535.v659.checkpoint.v1",
                    "panel_sha256": panel["panel_sha256"],
                    "model_sha256": context["model_hash"],
                    "completed_units": len(rows),
                    "rows_sha256": canonical_hash(rows),
                },
            )
            checkpoint_seconds.append(time.monotonic() - checkpoint_started)
            lease.heartbeat()
            progress(
                started,
                "native_forwards",
                "after_forward",
                completed_units=len(rows),
                planned=FORWARD_BUDGET,
                disposition=row["disposition"],
            )
        rows_receipt = _write_jsonl(root / ROWS_PATH, rows)
        reduction = reduce_native_rows(rows, schedule)
        inference_ok = reduction["passed"] is True
        spans.append(
            _phase_span(
                "native_forwards",
                phase_started,
                started,
                len(rows),
                rows_receipt["path"],
            )
        )
        progress(
            started,
            "native_forwards",
            "after_loop",
            completed_units=len(rows),
            custody_passed=inference_ok,
        )
    except BaseException as exc:  # noqa: BLE001 - convert owned failure to terminal evidence.
        live_error = f"{type(exc).__name__}:{exc}"
        progress(started, "live_measurement", "error", error=live_error)
    finally:
        progress(started, "model_unload", "before")
        runner.close()
        progress(started, "model_unload", "after")
        phase = lease.document.get("phase")
        if phase in {"resident", "inferencing"}:
            lease.transition("unloading")
            after_vram = _owned_vram_mb(os.getpid(), str(selected_gpu["uuid"]))
            lease.transition(
                "validating",
                vram_mb=int(after_vram or 0),
                exit_code=0 if inference_ok else 1,
                unload_observed=after_vram in {None, 0},
            )
            lease.transition("terminal_complete" if inference_ok else "terminal_blocked")
        elif phase in {"preflight", "admitted", "loading"}:
            lease.transition("terminal_blocked")
        release = lease.release()
        ownership["release"] = release

    if schedule and not schedule_receipt:
        schedule_receipt = _write_jsonl(root / RAW_DIR / "forward_schedule.jsonl", schedule)
    if not rows_receipt:
        rows_receipt = _write_jsonl(root / ROWS_PATH, rows)
    if schedule and not reduction:
        reduction = reduce_native_rows(rows, schedule)
    if not reduction:
        reduction = {
            "passed": False,
            "planned_forward_count": FORWARD_BUDGET,
            "attempted_forward_count": len(rows),
            "complete_forward_count": 0,
            "error_forward_count": len(rows),
            "generated_token_count": 0,
            "failed_checks": ["live_measurement_before_schedule"],
            "comparative_rows": [],
            "benefit_gate_applied": False,
        }
    forward_seconds = [
        float(row["forward_seconds"])
        for row in rows
        if row.get("disposition") == "complete"
        and isinstance(row.get("forward_seconds"), int | float)
    ]
    checkpoint_projection = (
        CAPTURE_FORWARDS * _p95(checkpoint_seconds) if checkpoint_seconds else CAPTURE_LIMIT_S
    )
    if forward_seconds:
        forecasts = forecast_captures(
            load_seconds=load_seconds,
            forward_seconds=forward_seconds,
            fit_checkpoint_seconds=checkpoint_projection,
            eval_checkpoint_seconds=checkpoint_projection,
        )
    else:
        forecasts = {
            "formula": "load + 1440*p95_forward + checkpoint + 600",
            "fit": {"planned_forwards": CAPTURE_FORWARDS, "forecast_seconds": None},
            "evaluation": {"planned_forwards": CAPTURE_FORWARDS, "forecast_seconds": None},
            "fit_capture_feasible_score": 0,
            "eval_capture_feasible_score": 0,
            "sample_shrinking_allowed": False,
            "transport_retuning_allowed": False,
            "closed_reason": live_error or "no_complete_forward_durations",
        }
    if not ownership:
        ownership = {
            **deepcopy(owner),
            "selected_gpu_index": selected_gpu["index"],
            "selected_gpu_uuid": selected_gpu["uuid"],
            "measured_owned_offloaded_layers": 0,
            "offload_real": False,
            "admission_receipt": admission,
            "signals_sent": [],
            "release": release,
        }
    if not model_identity:
        model_identity = {
            "hf_id": MODEL_ID,
            "model_path": str(model_path),
            "model_sha256": context["model_hash"],
            "quantization": "Q4_K_M",
            "runtime": context["runtime"],
            "embedded_tokenizer": False,
            "load_error": live_error,
        }
    raw_receipts = {
        "forward_schedule": schedule_receipt,
        "native_rows": rows_receipt,
        "all_below_20_mib": all(
            int(receipt.get("bytes") or 0) < 20 * 1024 * 1024
            for receipt in (schedule_receipt, rows_receipt)
            if receipt
        ),
    }
    pilot_cost_rows: list[JsonDict] = [
        {
            "kind": "model_load",
            "full_prompt_length": 0,
            "measured_seconds": load_seconds,
        }
    ]
    pilot_cost_rows.extend(
        {
            "kind": "native_forward",
            "call_id": row.get("call_id"),
            "full_prompt_length": row.get("prompt_token_count"),
            "measured_seconds": row.get("forward_seconds"),
            "disposition": row.get("disposition"),
        }
        for row in rows
    )
    artifact = build_complete_artifact(
        preconditions=checks,
        source_hashes=source_hashes,
        model_specs=[model_identity],
        events=events,
        panel=panel,
        rows_receipt=raw_receipts,
        reduction=reduction,
        forecasts=forecasts,
        ownership=ownership,
        pilot_cost_rows=pilot_cost_rows,
        validation_receipts=affected,
        duration_s=time.monotonic() - started,
        phase_spans=spans,
        require_validation=False,
    )
    if live_error is not None:
        artifact["live_error"] = live_error
        artifact["honest_verdict"] = "complete_disqualified_owned_live_measurement"
        artifact["verdict_class"] = "disqualified"
        artifact["native_tool_ready_score"] = 0
        artifact["ready"] = 0
        artifact["field_principles"] = _field_principles(
            (*artifact.keys(), "field_principles", "reproducibility_checksum")
        )
        artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return _finish_candidate(root, artifact, affected, spans, started)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:  # pragma: no cover
    """Parse the fixed live command and fresh-process reader modes."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--date", required=True)
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--independent-reduce", type=Path)
    return parser.parse_args(argv)


def _argument_path(path: Path) -> Path:  # pragma: no cover
    return path if path.is_absolute() else REPO_ROOT / path


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover
    """Run the owned pilot or one read-only exact-candidate check."""

    args = parse_args(argv)
    if args.date != RUN_DATE:
        print(
            json.dumps(
                {
                    "mode": "argument_check",
                    "passed": False,
                    "expected": RUN_DATE,
                    "observed": args.date,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return 1
    if args.cold_replay is not None:
        artifact = _load_object(_argument_path(args.cold_replay))
        errors = validate_artifact(artifact, require_validation=False)
        print(
            json.dumps({"mode": "cold_replay", "passed": not errors, "errors": errors}),
            flush=True,
        )
        return int(bool(errors))
    if args.independent_reduce is not None:
        artifact = _load_object(_argument_path(args.independent_reduce))
        reduction = independent_reduce(artifact)
        print(
            json.dumps({"mode": "independent_reduce", **reduction}, sort_keys=True, default=str),
            flush=True,
        )
        return int(reduction.get("passed") is not True)
    return run_experiment(REPO_ROOT, args.date)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
