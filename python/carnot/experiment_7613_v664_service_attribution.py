"""Measure the durable recalibration service boundary by exclusive stage.

The experiment explains placement. It does not repeat the earlier aggregate
speed gate and does not predict accelerator latency.

Spec: REQ-CL-7613, REQ-HW-7613, and their matching scenarios.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from copy import deepcopy
from datetime import UTC, datetime
import json
import math
from pathlib import Path
import random
import shutil
import statistics
import tempfile
import time
from typing import Any

from carnot import experiment_7598_v663_rust_consumer as consumer
from carnot import experiment_7599_v663_board_continuity as board
from carnot.pipeline.calibrated_decision_service import DURABILITY_POLICY
from carnot.reporting import experiment_7303_validation_scope as validation
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file


JsonDict = dict[str, Any]
RUN_DATE = "20260924"
MILESTONE = "2026.09.664"
EXPERIMENT_ID = "exp7613-v664-service-attribution"
SCHEMA = "carnot.exp7613.v664.service_attribution.v1"
RANDOM_SEED = 7_613_001
REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_PATH = Path("results/experiment_7613_v664_service_attribution.json")
RAW_DIR = Path("results/raw/experiment_7613_v664_service_attribution")
MODULE_PATH = Path("python/carnot/experiment_7613_v664_service_attribution.py")
CLIENT_PATH = Path("python/carnot/pipeline/calibrated_decision_service.py")
RUST_PATH = Path("crates/carnot-core/src/bin/portable-recalibration-service.rs")
WRAPPER_PATH = Path("scripts/experiments/experiment_7613_v664_service_attribution.py")
TEST_PATH = Path("tests/python/test_experiment_7613_v664_service_attribution.py")
CL_SPEC_PATH = Path("openspec/capabilities/continuous-learning/spec.md")
HW_SPEC_PATH = Path("openspec/capabilities/hardware/spec.md")
PRIOR_CONSUMER_PATH = Path("results/experiment_7598_v663_rust_consumer.json")
PRIOR_BOARD_PATH = Path("results/experiment_7599_v663_board_continuity.json")
NOTES_PATH = Path("docs/research-notes/v664-service-placement.md")
MODEL_SPECS: list[JsonDict] = []
ZERO_INVOCATION_COUNTS = {
    "model_loads": 0,
    "forward_calls": 0,
    "generation_calls": 0,
    "current_llm_calls": 0,
    "input_tokens": 0,
    "output_tokens": 0,
}
MODES = ("cold", "warm")
BATCH_SIZES = (1, 8)
ARMS = ("rust", "python_inprocess")
REPEATS = 30
OVERHEAD_REPEATS = 10
WORKER_STAGE_NAMES = (
    "encoding",
    "update_arithmetic",
    "journal_write",
    "fsync",
    "acknowledgement",
    "reload",
)
VALIDATION_NAMES = (*validation.REQUIRED_CHECK_NAMES, "rust_tests", "rust_fmt", "rust_clippy")
TERMINAL_NAMES = (
    "fresh_process_cold_replay",
    "independent_raw_reduction",
    "adversarial_verify",
    "verdict_row_consistency_strict",
)


def _percentile(values: Sequence[float], fraction: float) -> float:
    """Return an interpolated percentile for a nonempty finite sequence."""

    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("percentile_requires_values")
    position = fraction * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _interval(values: Sequence[float]) -> JsonDict:
    """Keep descriptive uncertainty explicit without inventing independence."""

    return {
        "estimate": statistics.median(values),
        "lower95": _percentile(values, 0.025),
        "upper95": _percentile(values, 0.975),
        "independent_blocks": len(values),
    }


def _stage_row(
    *, mode: str, batch_size: int, repeat: int, arm: str, whole: int, arithmetic: int
) -> JsonDict:
    """Create one deterministic row for reducer and mutation tests."""

    worker = {
        "encoding": 1_000,
        "update_arithmetic": arithmetic,
        "journal_write": 8_000,
        "fsync": 40_000,
        "acknowledgement": 1_000,
        "reload": 5_000,
    }
    remainder = whole - sum(worker.values())
    return {
        "row_type": "consumer_stage",
        "unit_id": f"{mode}:{batch_size}:{repeat}:{arm}",
        "pair_id": f"{mode}:{batch_size}:{repeat}",
        "mode": mode,
        "batch_size": batch_size,
        "repeat": repeat,
        "arm": arm,
        "seed": RANDOM_SEED + repeat,
        "whole_service_ns": whole,
        "numerator": arithmetic,
        "denominator": whole,
        "metric_direction": "descriptive_fraction",
        "worker_stage_ns": worker,
        "caller_stage_ns": {"startup": 4_000 if mode == "cold" else 0},
        "caller_remainder_ns": remainder,
        "caller_remainder_label": "ipc_scheduling_and_unobserved_caller_work",
        "reconciliation_error_ns": 0,
        "clock_domains": {
            "caller": "caller_process_monotonic_duration",
            "worker": "worker_process_monotonic_duration",
        },
        "cross_clock_timestamp_subtraction": False,
        "durability_policy": DURABILITY_POLICY,
        "parity": True,
        "censored": False,
        "missing": False,
        "provenance": "paired durable request fixture",
    }


def synthetic_stage_rows() -> list[JsonDict]:
    """Return complete private rows for tests without benchmark claims."""

    rows = []
    for mode in MODES:
        for batch_size in BATCH_SIZES:
            for repeat in range(REPEATS):
                for arm in ARMS:
                    rows.append(
                        _stage_row(
                            mode=mode,
                            batch_size=batch_size,
                            repeat=repeat,
                            arm=arm,
                            whole=200_000 + batch_size * 20_000 + repeat * 100,
                            arithmetic=20_000 + batch_size * 500,
                        )
                    )
    return rows


def synthetic_overhead_rows() -> list[JsonDict]:
    """Return the frozen ten-on and ten-off overhead controls per stratum."""

    rows = []
    for mode in MODES:
        for batch_size in BATCH_SIZES:
            for repeat in range(OVERHEAD_REPEATS):
                for enabled, adjustment in ((True, 2_000), (False, 0)):
                    rows.append(
                        {
                            "row_type": "instrumentation_overhead",
                            "unit_id": f"{mode}:{batch_size}:{repeat}:{int(enabled)}",
                            "mode": mode,
                            "batch_size": batch_size,
                            "repeat": repeat,
                            "seed": RANDOM_SEED + 100_000 + repeat,
                            "telemetry_enabled": enabled,
                            "whole_service_ns": 200_000 + batch_size * 20_000 + adjustment,
                            "numerator": 200_000 + batch_size * 20_000 + adjustment,
                            "denominator": batch_size,
                            "metric_direction": "lower_is_better",
                            "censored": False,
                            "missing": False,
                            "provenance": "registered instrumentation overhead control",
                        }
                    )
    return rows


def reduce_instrumentation_overhead(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Estimate telemetry overhead without giving controls selection authority."""

    result: JsonDict = {}
    for mode in MODES:
        for batch_size in BATCH_SIZES:
            selected = [
                row
                for row in rows
                if row.get("mode") == mode and row.get("batch_size") == batch_size
            ]
            enabled = [
                float(row["whole_service_ns"])
                for row in selected
                if row.get("telemetry_enabled") is True
            ]
            disabled = [
                float(row["whole_service_ns"])
                for row in selected
                if row.get("telemetry_enabled") is False
            ]
            if len(enabled) != OVERHEAD_REPEATS or len(disabled) != OVERHEAD_REPEATS:
                raise ValueError("overhead_block_count")
            result[f"{mode}:{batch_size}"] = {
                "telemetry_on_blocks": len(enabled),
                "telemetry_off_blocks": len(disabled),
                "median_overhead_ns": statistics.median(enabled) - statistics.median(disabled),
                "median_overhead_fraction": statistics.median(enabled) / statistics.median(disabled)
                - 1.0,
                "report_selection_authority": False,
            }
    return result


def _validate_stage_row(row: Mapping[str, Any]) -> None:
    """Reject spans that cannot support clock-safe exclusive attribution."""

    worker = row.get("worker_stage_ns")
    if not isinstance(worker, Mapping) or set(worker) != set(WORKER_STAGE_NAMES):
        raise ValueError("stage_names")
    values = [int(worker[name]) for name in WORKER_STAGE_NAMES]
    if any(value < 0 for value in values):
        raise ValueError("stage_negative")
    whole = int(row.get("whole_service_ns") or 0)
    remainder = int(row.get("caller_remainder_ns") or 0)
    if whole <= 0 or remainder < 0:
        raise ValueError("stage_denominator")
    error = abs(whole - sum(values) - remainder)
    if error > max(1_000, whole * 0.01):
        raise ValueError("stage_double_count_or_gap")
    if row.get("durability_policy") != DURABILITY_POLICY:
        raise ValueError("durability_policy")
    if row.get("cross_clock_timestamp_subtraction") is not False:
        raise ValueError("stage_clock_domain")
    if row.get("parity") is not True or row.get("censored") is not False:
        raise ValueError("stage_parity")


def reduce_stage_rows(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Reduce only complete paired blocks and compute measured Amdahl bounds."""

    for row in rows:
        _validate_stage_row(row)
    strata: JsonDict = {}
    for mode in MODES:
        for batch_size in BATCH_SIZES:
            selected = [
                row
                for row in rows
                if row.get("mode") == mode and row.get("batch_size") == batch_size
            ]
            pairs: dict[str, set[str]] = {}
            for row in selected:
                pairs.setdefault(str(row.get("pair_id")), set()).add(str(row.get("arm")))
            if len(pairs) != REPEATS or any(arms != set(ARMS) for arms in pairs.values()):
                raise ValueError("stage_pair_count")
            rust = [row for row in selected if row.get("arm") == "rust"]
            fractions = [
                float(row["worker_stage_ns"]["update_arithmetic"]) / float(row["whole_service_ns"])
                for row in rust
            ]
            bounds = [1.0 / (1.0 - fraction) for fraction in fractions]
            key = f"{mode}:{batch_size}"
            strata[key] = {
                "mode": mode,
                "batch_size": batch_size,
                "paired_blocks": len(pairs),
                "denominator": "whole_durable_request",
                "arithmetic_fraction": _interval(fractions),
                "amdahl_upper_bound": {
                    **_interval(bounds),
                    "formula": "1/(1-f)",
                    "kind": "upper_bound_not_measured_speedup",
                },
                "projected_hardware_latency": None,
                "caller_remainder_label": "ipc_scheduling_and_unobserved_caller_work",
            }
    return {"stage_attribution_ready_score": 1, "strata": strata}


def _source(path: Path, root: Path) -> JsonDict:
    """Bind exact source bytes and historical terminal fields when present."""

    row = consumer.source_row(path, root)
    return deepcopy(row)


def collect_preconditions(root: Path) -> JsonDict:
    """Authenticate named producers and every dated board evidence file."""

    repo = root.resolve()
    prior_consumer = consumer.load_object(repo / PRIOR_CONSUMER_PATH)
    prior_board = consumer.load_object(repo / PRIOR_BOARD_PATH)
    checks: list[JsonDict] = []
    sources: dict[str, JsonDict] = {}
    required = (PRIOR_CONSUMER_PATH, PRIOR_BOARD_PATH, CL_SPEC_PATH, HW_SPEC_PATH, RUST_PATH)
    for relative in required:
        path = repo / relative
        observed = path.is_file() and path.stat().st_size > 0
        checks.append(
            consumer._check(
                "source_readable", relative.as_posix(), relative.as_posix(), "bytes", True, observed
            )
        )
        if observed:
            sources[relative.as_posix()] = _source(path, repo)
    checks.extend(
        [
            consumer._check(
                "exp7598_terminal",
                "Exp7598",
                PRIOR_CONSUMER_PATH.as_posix(),
                "honest_verdict",
                "complete_null_rust_consumer_ready_speed_gate_failed",
                prior_consumer.get("honest_verdict"),
            ),
            consumer._check(
                "exp7598_not_flagged",
                "Exp7598",
                PRIOR_CONSUMER_PATH.as_posix(),
                "flagged_adversarial",
                False,
                prior_consumer.get("flagged_adversarial"),
            ),
            consumer._check(
                "exp7599_board_score",
                "Exp7599",
                PRIOR_BOARD_PATH.as_posix(),
                "board_continuity_complete_score",
                1,
                prior_board.get("board_continuity_complete_score"),
            ),
            consumer._check(
                "exp7599_board_count",
                "Exp7599",
                PRIOR_BOARD_PATH.as_posix(),
                "board_rows",
                3,
                len(prior_board.get("board_rows") or []),
            ),
        ]
    )
    for row in prior_board.get("board_rows") or []:
        evidence = repo / str(row.get("evidence_artifact_path"))
        observed_hash = sha256_file(evidence) if evidence.is_file() else None
        expected_hash = row.get("evidence_artifact_sha256")
        checks.append(
            consumer._check(
                f"board_evidence:{row.get('board')}",
                "Exp7599",
                str(row.get("evidence_artifact_path")),
                "sha256",
                expected_hash,
                observed_hash,
            )
        )
        if evidence.is_file():
            sources[str(row.get("evidence_artifact_path"))] = _source(evidence, repo)
    failed = next((row for row in checks if row.get("passed") is not True), None)
    blocker = (
        {
            key: failed.get(key)
            for key in ("check", "upstream", "path", "field", "op", "expected", "observed")
        }
        if failed
        else None
    )
    return {
        "checks": checks,
        "blocker": blocker,
        "sources": sources,
        "prior_consumer": prior_consumer,
        "prior_board": prior_board,
    }


def progress(started: float, phase: str, event: str, **fields: Any) -> None:  # pragma: no cover
    """Emit one flushed boundary with truthful monotonic elapsed time."""

    detail = " ".join(f"{key}={value}" for key, value in fields.items())
    print(
        f"[exp7613] phase={phase} event={event} elapsed_s={time.monotonic() - started:.3f} {detail}".rstrip(),
        flush=True,
    )


def _worker_spans(arm: str, ack: Any) -> dict[str, int]:  # pragma: no cover
    """Map each arm to the same exclusive stage vocabulary."""

    if arm == "rust":
        return {name: int(ack.exclusive_stage_ns.get(name, 0)) for name in WORKER_STAGE_NAMES}
    legacy = ack.stage_ns
    return {
        "encoding": 0,
        "update_arithmetic": int(legacy.get("solve", 0)),
        "journal_write": int(legacy.get("write", 0)) + int(legacy.get("rename", 0)),
        "fsync": int(legacy.get("fsync", 0)),
        "acknowledgement": 0,
        "reload": int(legacy.get("reload", 0)),
    }


def _measure_block(
    service: Any,
    arm: str,
    events: Sequence[tuple[str, float, int]],
    *,
    include_setup: bool,
) -> JsonDict:  # pragma: no cover
    """Measure one identical predict-update-durable-ack-restart workload."""

    started = time.perf_counter_ns()
    worker = {name: 0 for name in WORKER_STAGE_NAMES}
    caller: dict[str, int] = {
        "startup": int(getattr(service, "setup_ns", 0)) if include_setup else 0
    }
    decisions: list[tuple[str, float, str]] = []
    for event_id, probability, label in events:
        decision = service.predict(event_id, probability)
        if not decision.available:
            raise RuntimeError(f"prediction_failed:{decision.error}")
        acknowledgment = service.release_feedback(event_id, label)
        if not acknowledgment.durable:
            raise RuntimeError(f"feedback_failed:{acknowledgment.error}")
        decisions.append((event_id, decision.error_probability, decision.action))
        for name, value in _worker_spans(arm, acknowledgment).items():
            worker[name] += value
        for source in (decision.caller_stage_ns, acknowledgment.caller_stage_ns):
            for name, value in source.items():
                caller[name] = caller.get(name, 0) + int(value)
    resumed = service.predict(f"{events[-1][0]}-resumed", 0.37)
    if not resumed.available:
        raise RuntimeError(f"resumed_prediction_failed:{resumed.error}")
    for name, value in resumed.caller_stage_ns.items():
        caller[name] = caller.get(name, 0) + int(value)
    measured = time.perf_counter_ns() - started
    whole = measured + caller["startup"]
    worker_sum = sum(worker.values())
    remainder = whole - worker_sum
    return {
        "whole_service_ns": whole,
        "worker_stage_ns": worker,
        "caller_stage_ns": caller,
        "caller_remainder_ns": remainder,
        "reconciliation_error_ns": abs(whole - worker_sum - remainder),
        "decisions": decisions,
        "resumed": (resumed.error_probability, resumed.action),
        "state": consumer.upstream._load_state(service.state_path).to_payload(),
        "durable_acknowledgments": len(events),
    }


def _measurement_row(
    result: Mapping[str, Any],
    *,
    mode: str,
    batch_size: int,
    repeat: int,
    arm: str,
    seed: int,
    order: Sequence[str],
    parity: bool,
) -> JsonDict:  # pragma: no cover
    """Retain absolute stage evidence for one independent paired arm."""

    whole = int(result["whole_service_ns"])
    arithmetic = int(result["worker_stage_ns"]["update_arithmetic"])
    return {
        "row_type": "consumer_stage",
        "unit_id": f"{mode}:{batch_size}:{repeat}:{arm}",
        "pair_id": f"{mode}:{batch_size}:{repeat}",
        "mode": mode,
        "batch_size": batch_size,
        "repeat": repeat,
        "arm": arm,
        "arm_order": list(order),
        "seed": seed,
        "whole_service_ns": whole,
        "numerator": arithmetic,
        "denominator": whole,
        "metric_direction": "descriptive_fraction",
        "worker_stage_ns": deepcopy(result["worker_stage_ns"]),
        "caller_stage_ns": deepcopy(result["caller_stage_ns"]),
        "caller_remainder_ns": int(result["caller_remainder_ns"]),
        "caller_remainder_label": "ipc_scheduling_and_unobserved_caller_work",
        "reconciliation_error_ns": int(result["reconciliation_error_ns"]),
        "clock_domains": {
            "caller": "caller_process_monotonic_duration",
            "worker": "worker_process_monotonic_duration"
            if arm == "rust"
            else "caller_process_local_comparator_duration",
        },
        "cross_clock_timestamp_subtraction": False,
        "durability_policy": DURABILITY_POLICY,
        "parity": parity,
        "durable_acknowledgments": int(result["durable_acknowledgments"]),
        "censored": False,
        "missing": False,
        "provenance": "public predict, release, durable ack, and resumed prediction",
    }


def measure_stage_rows(
    root: Path, scratch: Path, started: float
) -> list[JsonDict]:  # pragma: no cover
    """Measure 30 rotated paired blocks for each frozen stratum."""

    rows: list[JsonDict] = []
    completed = 0
    planned = len(MODES) * len(BATCH_SIZES) * REPEATS
    for mode_index, mode in enumerate(MODES):
        for batch_size in BATCH_SIZES:
            warm: dict[str, Any] = {}
            if mode == "warm":
                for arm in ARMS:
                    warm[arm] = consumer._open_service(
                        root,
                        arm,
                        scratch / f"warm-{batch_size}-{arm}.json",
                        telemetry_enabled=arm == "rust",
                    )
            try:
                for repeat in range(REPEATS):
                    seed = RANDOM_SEED + mode_index * 10_000 + batch_size * 100 + repeat
                    events = consumer._block_events(seed, batch_size)
                    order = list(ARMS)
                    if repeat % 2:
                        order.reverse()
                    results: dict[str, JsonDict] = {}
                    for arm in order:
                        service = warm.get(arm) or consumer._open_service(
                            root,
                            arm,
                            scratch / f"cold-{batch_size}-{repeat}-{arm}.json",
                            telemetry_enabled=arm == "rust",
                        )
                        try:
                            results[arm] = _measure_block(
                                service, arm, events, include_setup=mode == "cold"
                            )
                        finally:
                            if mode == "cold":
                                service.close()
                    parity_inputs = {
                        **results,
                        "python_service": results["python_inprocess"],
                    }
                    decision_parity, reload_parity = consumer._outcomes_match(parity_inputs)
                    parity = decision_parity and reload_parity
                    for arm in ARMS:
                        rows.append(
                            _measurement_row(
                                results[arm],
                                mode=mode,
                                batch_size=batch_size,
                                repeat=repeat,
                                arm=arm,
                                seed=seed,
                                order=order,
                                parity=parity,
                            )
                        )
                    completed += 1
                    progress(
                        started,
                        "benchmark",
                        "paired_block_complete",
                        completed=completed,
                        planned=planned,
                    )
            finally:
                for service in warm.values():
                    service.close()
    return rows


def measure_overhead_rows(
    root: Path, scratch: Path, started: float
) -> list[JsonDict]:  # pragma: no cover
    """Measure registered on/off controls that cannot select the main report."""

    rows: list[JsonDict] = []
    completed = 0
    planned = len(MODES) * len(BATCH_SIZES) * OVERHEAD_REPEATS
    for mode_index, mode in enumerate(MODES):
        for batch_size in BATCH_SIZES:
            warm: dict[bool, Any] = {}
            if mode == "warm":
                for enabled in (True, False):
                    warm[enabled] = consumer._open_service(
                        root,
                        "rust",
                        scratch / f"overhead-warm-{batch_size}-{int(enabled)}.json",
                        telemetry_enabled=enabled,
                    )
            try:
                for repeat in range(OVERHEAD_REPEATS):
                    seed = RANDOM_SEED + 100_000 + mode_index * 10_000 + batch_size * 100 + repeat
                    events = consumer._block_events(seed, batch_size)
                    order = (True, False) if repeat % 2 == 0 else (False, True)
                    for enabled in order:
                        service = warm.get(enabled) or consumer._open_service(
                            root,
                            "rust",
                            scratch / f"overhead-cold-{batch_size}-{repeat}-{int(enabled)}.json",
                            telemetry_enabled=enabled,
                        )
                        try:
                            result = _measure_block(
                                service, "rust", events, include_setup=mode == "cold"
                            )
                        finally:
                            if mode == "cold":
                                service.close()
                        rows.append(
                            {
                                "row_type": "instrumentation_overhead",
                                "unit_id": f"{mode}:{batch_size}:{repeat}:{int(enabled)}",
                                "mode": mode,
                                "batch_size": batch_size,
                                "repeat": repeat,
                                "seed": seed,
                                "telemetry_enabled": enabled,
                                "arm_order": [int(value) for value in order],
                                "whole_service_ns": int(result["whole_service_ns"]),
                                "numerator": int(result["whole_service_ns"]),
                                "denominator": batch_size * 2 + 1,
                                "metric_direction": "lower_is_better",
                                "durability_policy": DURABILITY_POLICY,
                                "censored": False,
                                "missing": False,
                                "provenance": "registered telemetry overhead control only",
                            }
                        )
                    completed += 1
                    progress(
                        started,
                        "overhead",
                        "paired_block_complete",
                        completed=completed,
                        planned=planned,
                    )
            finally:
                for service in warm.values():
                    service.close()
    return rows


def _provisional_receipts() -> list[JsonDict]:
    """Give pure tests complete names without claiming real command evidence."""

    return [
        {
            "name": name,
            "passed": True,
            "exit_code": 0,
            "command": "unit-test provisional receipt",
            "log_sha256": "sha256:" + "0" * 64,
            "provisional": True,
        }
        for name in (*VALIDATION_NAMES, *TERMINAL_NAMES)
    ]


def _gate(category: str, check: str, passed: bool, expected: Any, observed: Any) -> JsonDict:
    """Keep each acceptance category independent and principle annotated."""

    return {
        "category": category,
        "check": check,
        "passed": passed,
        "expected": expected,
        "observed": observed,
        "principle": "Validity, readiness, benefit, retention, and freshness are separate claims.",
    }


def _principles(keys: Sequence[str]) -> dict[str, str]:
    """Carry a one-line reporting rule beside every governed field."""

    special = {
        "honest_verdict": "Use a complete_ terminal prefix; completion alone is not scientific benefit.",
        "verdict_class": "Use exactly positive, circular_positive, null, blocked, disqualified, or partial.",
        "flagged_adversarial": "Persist the terminal reader result; flagged evidence never opens readiness.",
        "consumer_stage_rows": "Retain raw caller and worker durations with declared clock domains and remainder.",
        "stage_attribution_ready_score": "One requires exclusive clock-safe spans, equal durability, and measured overhead.",
        "amdahl_upper_bound": "Compute an explicit upper bound from measured arithmetic fraction, never accelerator gain.",
        "board_rows": "Keep all three dated scopes and changed-state prerequisites explicit.",
        "hardware_operations_issued": "An empty list prevents a current board execution claim.",
        "readiness": "Null prevents lifecycle readiness from becoming empirical value.",
    }
    return {
        key: special.get(key, "Bind this field to current raw evidence and independent reduction.")
        for key in keys
    }


def reproducibility_checksum(value: Mapping[str, Any]) -> str:
    """Bind configuration, source custody, raw rows, and reduction."""

    payload = deepcopy(dict(value))
    for key in (
        "started_at_utc",
        "completed_at_utc",
        "duration_s",
        "phase_spans",
        "validation_receipts",
        "terminal_reader_outcomes",
        "field_principles",
        "reproducibility_checksum",
    ):
        payload.pop(key, None)
    return canonical_hash(payload)


def build_artifact(
    root: Path,
    stage_rows: Sequence[Mapping[str, Any]],
    overhead_rows: Sequence[Mapping[str, Any]],
    *,
    validation_receipts: Sequence[Mapping[str, Any]] | None = None,
    duration_s: float = 0.01,
    phase_spans: Sequence[Mapping[str, Any]] = (),
) -> JsonDict:
    """Build one complete host-attribution artifact from immutable raw rows."""

    repo = root.resolve()
    context = collect_preconditions(repo)
    blocker = context["blocker"]
    receipts = [deepcopy(dict(row)) for row in (validation_receipts or _provisional_receipts())]
    if blocker is None:
        reduction = reduce_stage_rows(stage_rows)
        overhead = reduce_instrumentation_overhead(overhead_rows)
        verdict = "complete_null_service_attribution_preserves_exp7598_null"
        verdict_class = "null"
    else:
        reduction = {"stage_attribution_ready_score": 0, "strata": {}}
        overhead = {}
        verdict = f"complete_blocked_{str(blocker['check']).replace(':', '_')}"
        verdict_class = "blocked"
    board_rows = deepcopy(context["prior_board"].get("board_rows") or [])
    sources = deepcopy(context["sources"])
    generated_sources = (
        MODULE_PATH,
        CLIENT_PATH,
        WRAPPER_PATH,
        TEST_PATH,
        NOTES_PATH,
        consumer.RUST_BINARY,
        RAW_DIR / "affected_validation_manifest.json",
        RAW_DIR / "consumer_stage_rows.json",
        RAW_DIR / "instrumentation_overhead_rows.json",
    )
    for relative in generated_sources:
        path = repo / relative
        if path.is_file():
            sources[relative.as_posix()] = _source(path, repo)
    gates = [
        _gate(
            "validity",
            "exclusive_span_reconciliation",
            reduction["stage_attribution_ready_score"] == 1,
            1,
            reduction["stage_attribution_ready_score"],
        ),
        _gate("readiness", "public_client_default_unchanged", True, False, False),
        _gate("benefit", "empirical_accelerator_benefit", False, True, False),
        _gate(
            "retention",
            "exp7598_aggregate_null_preserved",
            context["prior_consumer"].get("honest_verdict")
            == "complete_null_rust_consumer_ready_speed_gate_failed",
            "complete_null_rust_consumer_ready_speed_gate_failed",
            context["prior_consumer"].get("honest_verdict"),
        ),
        _gate("freshness", "current_model_or_board_claim", True, False, False),
    ]
    gate_summary = [] if blocker is None else [deepcopy(blocker)]
    gate_summary.extend(
        row
        for row in context["prior_board"].get("gate_check_summary") or []
        if row.get("check") == "gatemate_receipt"
    )
    terminal = {
        str(row.get("name")): {
            "passed": row.get("passed") is True,
            "exit_code": row.get("exit_code"),
            "log_sha256": row.get("log_sha256"),
        }
        for row in receipts
        if row.get("name") in TERMINAL_NAMES
    }
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "worktree": str(repo),
        "started_at_utc": datetime.now(UTC).isoformat(),
        "completed_at_utc": datetime.now(UTC).isoformat(),
        "duration_s": float(duration_s),
        "phase_spans": [deepcopy(dict(row)) for row in phase_spans],
        "random_seed": {"arm_rotation": RANDOM_SEED, "overhead_controls": RANDOM_SEED + 100_000},
        "MODEL_SPECS": [],
        "model_specs": [],
        "target_model": "none:no_model_load",
        "no_model_load": True,
        "model_invoked": False,
        "invocation_counts": deepcopy(ZERO_INVOCATION_COUNTS),
        "historical_model_identity": "unsloth/Qwen3.8-27B-GGUF; provenance only",
        "inference_substrate": "host_rust_jsonl_and_inprocess_python_durable_service",
        "inference_substrate_class": "no_model_load",
        "planned_inference_substrate_class": "no_model_load",
        "actual_inference_substrate_class": "no_model_load",
        "source_artifact_hashes": sources,
        "preconditions_checked": deepcopy(context["checks"]),
        "rows": [deepcopy(dict(row)) for row in (*stage_rows, *overhead_rows, *board_rows)],
        "consumer_stage_rows": [deepcopy(dict(row)) for row in stage_rows],
        "instrumentation_overhead_rows": [deepcopy(dict(row)) for row in overhead_rows],
        "instrumentation_overhead": overhead,
        "sample_size_budget": {
            "paired_stage_blocks": {
                "intended": 120,
                "observed": len(stage_rows) // 2,
                "excluded": 0,
                "censored": 0,
                "arms_per_block": 2,
            },
            "telemetry_off_blocks": {
                "intended": 40,
                "observed": sum(row.get("telemetry_enabled") is False for row in overhead_rows),
                "excluded": 0,
                "censored": 0,
            },
        },
        "stage_attribution_ready_score": reduction["stage_attribution_ready_score"],
        "stage_reduction": reduction,
        "amdahl_upper_bound": {
            key: value["amdahl_upper_bound"] for key, value in reduction["strata"].items()
        },
        "board_rows": board_rows,
        "hardware_operations_issued": [],
        "acquisition_decision": {
            "purchase_authorized": False,
            "kernel_only_timing_justifies_purchase": False,
            "extropic_z1t": "research context only",
            "september_22_spintronic_lead": "research context only",
        },
        "prior_exp7598_honest_verdict": context["prior_consumer"].get("honest_verdict"),
        "prior_exp7598_aggregate_verdict_preserved": True,
        "public_client_opt_in_unchanged": True,
        "empirical_weights_changed": False,
        "projected_hardware_latency": None,
        "pyo3_crossing_claimed": False,
        "e2e_003_claimed": False,
        "readiness": None,
        "verifier_is_oracle": False,
        "acceptance_gate_results": gates,
        "gate_check_summary": gate_summary,
        "honest_verdict": verdict,
        "verdict_class": verdict_class,
        "flagged_adversarial": False,
        "validation_receipts": receipts,
        "terminal_reader_outcomes": terminal,
        "repository_health": {
            "unrelated_open_debt": [
                {
                    "command": "cargo clippy -p carnot-core --bin portable-recalibration-service -- -D warnings",
                    "path": "crates/carnot-core/src/adaptive_state.rs",
                    "line": 1700,
                    "lint": "clippy::needless_late_init",
                    "current_task_path": False,
                }
            ],
            "changed_rust_clippy_scope": "all warnings denied except the named pre-existing adaptive_state lint",
        },
    }
    artifact["field_principles"] = _principles(
        [*artifact, "field_principles", "reproducibility_checksum"]
    )
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def build_test_artifact(root: Path, scratch: Path) -> JsonDict:
    """Build deterministic private evidence for unit and mutation tests."""

    scratch.mkdir(parents=True, exist_ok=True)
    return build_artifact(root, synthetic_stage_rows(), synthetic_overhead_rows())


def independent_reduce(value: Mapping[str, Any]) -> JsonDict:
    """Recompute comparative rows without trusting producer summaries."""

    stage = value.get("consumer_stage_rows")
    overhead = value.get("instrumentation_overhead_rows")
    if not isinstance(stage, list) or not isinstance(overhead, list):
        raise ValueError("raw_rows_missing")
    return {
        "stage": reduce_stage_rows(stage),
        "overhead": reduce_instrumentation_overhead(overhead),
        "board_count": len(value.get("board_rows") or []),
    }


def _verify_sources(value: Mapping[str, Any], root: Path) -> list[str]:
    """Reject changed source bytes without reinterpreting historical artifacts."""

    errors = []
    sources = value.get("source_artifact_hashes")
    if not isinstance(sources, Mapping):
        return ["source_artifact_hashes"]
    for key, receipt in sources.items():
        if not isinstance(receipt, Mapping):
            errors.append(f"source_receipt:{key}")
            continue
        raw = Path(str(receipt.get("path", key)))
        path = raw if raw.is_absolute() else root / raw
        if not path.is_file() or sha256_file(path) != receipt.get("sha256"):
            errors.append(f"source_hash:{key}")
    return errors


def validate_artifact(value: Mapping[str, Any], *, root: Path | None = None) -> list[str]:
    """Reject custody drift, reduction drift, and broadened placement claims."""

    required = {
        "honest_verdict",
        "verdict_class",
        "flagged_adversarial",
        "gate_check_summary",
        "acceptance_gate_results",
        "rows",
        "sample_size_budget",
        "inference_substrate",
        "inference_substrate_class",
        "MODEL_SPECS",
        "invocation_counts",
        "duration_s",
        "random_seed",
        "reproducibility_checksum",
        "source_artifact_hashes",
        "validation_receipts",
        "verifier_is_oracle",
        "field_principles",
        "stage_attribution_ready_score",
        "consumer_stage_rows",
        "amdahl_upper_bound",
        "board_rows",
        "hardware_operations_issued",
    }
    errors = []
    missing = sorted(required.difference(value))
    if missing:
        errors.append("required_fields:" + ",".join(missing))
    if value.get("MODEL_SPECS") != [] or value.get("model_specs") != []:
        errors.append("model_specs")
    if (
        value.get("no_model_load") is not True
        or value.get("invocation_counts") != ZERO_INVOCATION_COUNTS
    ):
        errors.append("invocation_counts")
    if value.get("inference_substrate_class") != "no_model_load":
        errors.append("inference_substrate_class")
    if value.get("hardware_operations_issued") != []:
        errors.append("hardware_operations")
    if (
        value.get("prior_exp7598_honest_verdict")
        != "complete_null_rust_consumer_ready_speed_gate_failed"
    ):
        errors.append("prior_verdict")
    if value.get("prior_exp7598_aggregate_verdict_preserved") is not True:
        errors.append("prior_verdict_preservation")
    if (
        value.get("public_client_opt_in_unchanged") is not True
        or value.get("empirical_weights_changed") is not False
    ):
        errors.append("production_default")
    if (
        value.get("projected_hardware_latency") is not None
        or value.get("pyo3_crossing_claimed") is not False
    ):
        errors.append("claim_boundary")
    blocked = value.get("verdict_class") == "blocked"
    reduction = None
    if blocked:
        if value.get("stage_attribution_ready_score") != 0:
            errors.append("stage_attribution_ready_score")
        summary = value.get("gate_check_summary")
        first = summary[0] if isinstance(summary, list) and summary else {}
        if not all(
            key in first
            for key in ("check", "upstream", "path", "field", "op", "expected", "observed")
        ):
            errors.append("blocked_gate_check_summary")
    else:
        try:
            reduction = independent_reduce(value)
        except (KeyError, TypeError, ValueError) as error:
            errors.append(f"reduction:{error}")
        if reduction is not None:
            if value.get("stage_reduction") != reduction["stage"]:
                errors.append("stage_reduction")
            if value.get("instrumentation_overhead") != reduction["overhead"]:
                errors.append("instrumentation_overhead")
            if reduction["board_count"] != 3:
                errors.append("board_count")
            if (
                value.get("stage_attribution_ready_score")
                != reduction["stage"]["stage_attribution_ready_score"]
            ):
                errors.append("stage_attribution_ready_score")
    boards = {
        row.get("board"): row for row in value.get("board_rows") or [] if isinstance(row, Mapping)
    }
    if set(boards) != {"KV260", "PolarFire", "GateMate"}:
        errors.append("board_identity")
    else:
        if boards["KV260"].get("future_access") != "ssh kria" or boards["KV260"].get("k_max") != 5:
            errors.append("kv260_scope")
        if boards["PolarFire"].get("fpga_sampling_measured") is not False:
            errors.append("polarfire_scope")
        if boards["GateMate"].get("disposition") != "blocked_unchanged_physical_prerequisite":
            errors.append("gatemate_scope")
    categories = {
        row.get("category")
        for row in value.get("acceptance_gate_results") or []
        if isinstance(row, Mapping)
    }
    if categories != {"validity", "readiness", "benefit", "retention", "freshness"}:
        errors.append("acceptance_gate_results")
    names = {
        row.get("name")
        for row in value.get("validation_receipts") or []
        if isinstance(row, Mapping) and row.get("passed") is True and row.get("exit_code") == 0
    }
    if not set((*VALIDATION_NAMES, *TERMINAL_NAMES)).issubset(names):
        errors.append("validation_receipts")
    principles = value.get("field_principles")
    if not isinstance(principles, Mapping) or any(key not in principles for key in value):
        errors.append("field_principles")
    if value.get("reproducibility_checksum") != reproducibility_checksum(value):
        errors.append("reproducibility_checksum")
    if root is not None:
        errors.extend(_verify_sources(value, root.resolve()))
    return sorted(set(errors))


def cold_replay(path: Path, *, root: Path) -> list[str]:
    """Load one candidate in a fresh process and verify exact source custody."""

    value = consumer.load_object(path)
    return validate_artifact(value, root=root) if value else ["artifact_unreadable"]


def independent_replay(path: Path) -> list[str]:
    """Recompute the terminal reduction without trusting summary fields."""

    value = consumer.load_object(path)
    if not value:
        return ["artifact_unreadable"]
    try:
        reduced = independent_reduce(value)
    except (KeyError, TypeError, ValueError) as error:
        return [str(error)]
    errors = []
    if value.get("stage_reduction") != reduced["stage"]:
        errors.append("stage_reduction")
    if value.get("instrumentation_overhead") != reduced["overhead"]:
        errors.append("instrumentation_overhead")
    if reduced["board_count"] != 3:
        errors.append("board_count")
    return errors


def build_validation_commands(  # pragma: no cover
    root: Path, private_root: Path
) -> list[validation.CommandSpec]:
    """Freeze explicit Python and changed Rust checks without broad fallback."""

    tests = (TEST_PATH.as_posix(), consumer.TEST_PATH.as_posix())
    modules = (MODULE_PATH.as_posix(), CLIENT_PATH.as_posix())
    commands = validation.build_scoped_commands(
        root,
        tests,
        modules,
        static_paths=(WRAPPER_PATH.as_posix(),),
        basetemp=private_root / "pytest",
        coverage_file=private_root / "coverage" / ".coverage.exp7613",
    )
    python = str(root / ".venv/bin/python")
    pytest = str(root / ".venv/bin/pytest")
    commands.extend(
        [
            validation.CommandSpec(
                "consumer_e2e",
                (
                    pytest,
                    "-n",
                    "0",
                    "-o",
                    "addopts=",
                    "--no-cov",
                    f"--basetemp={private_root / 'pytest' / 'e2e'}",
                    f"{TEST_PATH.as_posix()}::test_scenario_cl_7613_telemetry_is_default_off_and_explicit_opt_in",
                    "-q",
                ),
                "request_update_durable_ack_restart",
                300.0,
            ),
            validation.CommandSpec(
                "rust_tests",
                ("cargo", "test", "-p", "carnot-core", "--bin", "portable-recalibration-service"),
                "changed_rust_binary",
                300.0,
            ),
            validation.CommandSpec(
                "rust_fmt",
                ("rustfmt", "--edition", "2021", "--check", RUST_PATH.as_posix()),
                "changed_rust_source",
            ),
            validation.CommandSpec(
                "rust_clippy",
                (
                    "cargo",
                    "clippy",
                    "-p",
                    "carnot-core",
                    "--bin",
                    "portable-recalibration-service",
                    "--",
                    "-D",
                    "warnings",
                    "-A",
                    "clippy::needless-late-init",
                ),
                "changed_rust_binary",
                300.0,
            ),
            validation.CommandSpec(
                "entrypoint_help",
                (python, "-u", WRAPPER_PATH.as_posix(), "--help"),
                "thin_entrypoint",
            ),
        ]
    )
    return commands


def terminal_commands(  # pragma: no cover
    candidate: Path, root: Path
) -> list[validation.CommandSpec]:
    """Build independent readers for one exact candidate path."""

    python = str(root / ".venv/bin/python")
    return [
        validation.CommandSpec(
            "fresh_process_cold_replay",
            (
                python,
                "-u",
                WRAPPER_PATH.as_posix(),
                "--root",
                str(root),
                "--cold-replay",
                str(candidate),
            ),
            "exact_candidate",
        ),
        validation.CommandSpec(
            "independent_raw_reduction",
            (
                python,
                "-u",
                WRAPPER_PATH.as_posix(),
                "--root",
                str(root),
                "--independent-reduce",
                str(candidate),
            ),
            "exact_candidate",
        ),
        validation.CommandSpec(
            "adversarial_verify",
            (python, "-u", "scripts/adversarial_verify.py", str(candidate)),
            "exact_candidate",
        ),
        validation.CommandSpec(
            "verdict_row_consistency_strict",
            (python, "-u", "scripts/verdict_row_consistency_lint.py", "--strict", str(candidate)),
            "exact_candidate",
        ),
    ]


def _all_passed(  # pragma: no cover
    receipts: Sequence[Mapping[str, Any]], names: Sequence[str]
) -> bool:
    """Require one clean exit for every frozen command name."""

    passed = {
        str(row.get("name"))
        for row in receipts
        if row.get("passed") is True and row.get("exit_code") == 0
    }
    return set(names).issubset(passed)


def _write_manifest(root: Path, raw_dir: Path) -> Path:  # pragma: no cover
    """Freeze affected files and exact checks before validation starts."""

    path = raw_dir / "affected_validation_manifest.json"
    value = {
        "experiment_id": EXPERIMENT_ID,
        "tests": [TEST_PATH.as_posix(), consumer.TEST_PATH.as_posix()],
        "changed_python_modules": [MODULE_PATH.as_posix(), CLIENT_PATH.as_posix()],
        "changed_rust": [RUST_PATH.as_posix()],
        "specs": [CL_SPEC_PATH.as_posix(), HW_SPEC_PATH.as_posix()],
        "docs": [NOTES_PATH.as_posix()],
        "worktree": str(root),
    }
    atomic_json(path, value)
    return path


def run_experiment(root: Path, run_date: str) -> JsonDict:  # pragma: no cover
    """Benchmark, validate exact bytes, and atomically publish the terminal result."""

    repo = root.resolve()
    started = time.monotonic()
    progress(started, "startup", "resolved_root", root=repo)
    if run_date != RUN_DATE:
        raise ValueError(f"run_date_must_equal_{RUN_DATE}")
    context = collect_preconditions(repo)
    progress(started, "preconditions", "authenticated", blocker=context["blocker"])
    raw_dir = repo / RAW_DIR
    raw_dir.mkdir(parents=True, exist_ok=True)
    private_root = Path(tempfile.mkdtemp(prefix="carnot-exp7613-", dir="/tmp"))
    (private_root / "coverage").mkdir(parents=True, exist_ok=True)
    (private_root / "pytest").mkdir(parents=True, exist_ok=True)
    manifest = _write_manifest(repo, raw_dir)
    progress(started, "manifest", "frozen", sha256=sha256_file(manifest))

    build = validation.CommandSpec(
        "rust_release_build",
        (
            "cargo",
            "build",
            "--release",
            "-p",
            "carnot-core",
            "--bin",
            "portable-recalibration-service",
        ),
        "benchmark_worker",
        300.0,
    )
    progress(started, "build", "before_subprocess")
    build_receipts = validation.run_commands(
        repo, [build], log_dir=raw_dir / "validation" / "build"
    )
    progress(
        started,
        "build",
        "after_subprocess",
        passed=_all_passed(build_receipts, ("rust_release_build",)),
    )
    if not _all_passed(build_receipts, ("rust_release_build",)):
        raise RuntimeError("rust_release_build_failed")

    if context["blocker"] is None:
        progress(started, "benchmark", "before_measurement", planned=120)
        stage_rows = measure_stage_rows(repo, private_root / "benchmark", started)
        progress(started, "benchmark", "after_measurement", rows=len(stage_rows))
        progress(started, "overhead", "before_measurement", planned=40)
        overhead_rows = measure_overhead_rows(repo, private_root / "overhead", started)
        progress(started, "overhead", "after_measurement", rows=len(overhead_rows))
    else:
        stage_rows = []
        overhead_rows = []
    atomic_json(raw_dir / "consumer_stage_rows.json", {"rows": stage_rows})
    atomic_json(raw_dir / "instrumentation_overhead_rows.json", {"rows": overhead_rows})

    commands = build_validation_commands(repo, private_root)
    progress(started, "validation", "before_subprocesses", planned=len(commands))
    affected = validation.run_commands(repo, commands, log_dir=raw_dir / "validation" / "affected")
    progress(
        started,
        "validation",
        "after_subprocesses",
        passed=_all_passed(
            affected,
            (
                *validation.REQUIRED_CHECK_NAMES,
                "consumer_e2e",
                "rust_tests",
                "rust_fmt",
                "rust_clippy",
                "entrypoint_help",
            ),
        ),
    )
    if not _all_passed(
        affected,
        (
            *validation.REQUIRED_CHECK_NAMES,
            "consumer_e2e",
            "rust_tests",
            "rust_fmt",
            "rust_clippy",
            "entrypoint_help",
        ),
    ):
        raise RuntimeError("affected_validation_failed")

    provisional = build_artifact(
        repo,
        stage_rows,
        overhead_rows,
        validation_receipts=[*build_receipts, *affected, *_provisional_receipts()],
        duration_s=time.monotonic() - started,
    )
    candidate = private_root / "candidate.json"
    atomic_json(candidate, provisional)
    progress(started, "terminal", "before_subprocesses", planned=len(TERMINAL_NAMES))
    terminal = validation.run_commands(
        repo, terminal_commands(candidate, repo), log_dir=raw_dir / "validation" / "terminal"
    )
    progress(
        started, "terminal", "after_subprocesses", passed=_all_passed(terminal, TERMINAL_NAMES)
    )
    if not _all_passed(terminal, TERMINAL_NAMES):
        raise RuntimeError("terminal_validation_failed")

    final = build_artifact(
        repo,
        stage_rows,
        overhead_rows,
        validation_receipts=[*build_receipts, *affected, *terminal],
        duration_s=time.monotonic() - started,
    )
    errors = validate_artifact(final, root=repo)
    if errors:
        raise RuntimeError("terminal_artifact_invalid:" + ",".join(errors))
    exact = private_root / "exact-candidate.json"
    atomic_json(exact, final)
    progress(started, "exact_terminal", "before_subprocesses", planned=len(TERMINAL_NAMES))
    exact_receipts = validation.run_commands(
        repo, terminal_commands(exact, repo), log_dir=raw_dir / "validation" / "exact"
    )
    progress(
        started,
        "exact_terminal",
        "after_subprocesses",
        passed=_all_passed(exact_receipts, TERMINAL_NAMES),
    )
    if not _all_passed(exact_receipts, TERMINAL_NAMES):
        raise RuntimeError("exact_terminal_validation_failed")
    atomic_json(raw_dir / "exact_terminal_validation_receipts.json", {"receipts": exact_receipts})
    destination = repo / RESULT_PATH
    progress(started, "publish", "before_atomic_write", path=destination)
    atomic_json(destination, final)
    if sha256_file(destination) != sha256_file(exact):
        raise RuntimeError("published_bytes_differ")
    progress(started, "publish", "after_atomic_write", bytes=destination.stat().st_size)
    shutil.rmtree(private_root)
    return final


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:  # pragma: no cover
    """Parse producer and strict read-only replay modes."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--independent-reduce", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover
    """Run one selected mode while keeping the repository wrapper thin."""

    args = parse_args(argv)
    root = args.root.resolve()
    if args.cold_replay is not None:
        errors = cold_replay(args.cold_replay, root=root)
        print(
            json.dumps({"validation_passed": not errors, "errors": errors}, sort_keys=True),
            flush=True,
        )
        return int(bool(errors))
    if args.independent_reduce is not None:
        errors = independent_replay(args.independent_reduce)
        print(
            json.dumps({"validation_passed": not errors, "errors": errors}, sort_keys=True),
            flush=True,
        )
        return int(bool(errors))
    run_experiment(root, args.date)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
