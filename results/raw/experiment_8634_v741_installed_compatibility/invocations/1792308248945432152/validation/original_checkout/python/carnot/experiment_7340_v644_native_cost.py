"""Measure complete acquired-constraint costs at the Exp7339 native boundary.

The experiment deliberately times the Python-facing boundary instead of a
Rust-only kernel.  That boundary includes request copying, PyO3 conversion,
evaluation, and detached Python results.  A historical persistent JSON service
is retained as a third arm so transport is visible rather than credited to the
native evaluator.  This is a host cost experiment, not a learning-speed or
hardware claim.

Spec refs: REQ-VERIFY-7340, SCENARIO-VERIFY-7340-*, REQ-PYBIND-7340,
and SCENARIO-PYBIND-7340-*.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from copy import deepcopy
from datetime import UTC, datetime
import hashlib
import json
import math
import os
from pathlib import Path
import random
import statistics
import subprocess
import sys
import time
from typing import Any, TextIO

from carnot import experiment_7326_v643_constraint_kernel as exp7326
from carnot import experiment_7339_v644_native_binding as exp7339
from carnot.reporting.experiment_7303_validation_scope import (
    CommandSpec,
    run_commands,
    run_scoped_validation,
)


JsonDict = dict[str, Any]

EXPERIMENT_ID = 7340
MILESTONE = "2026.09.644"
RUN_DATE = "20260916"
SCHEMA = "carnot.experiment_7340.v644_native_cost.v1"
REPO_ROOT = Path(__file__).resolve().parents[2]
UPSTREAM_RELATIVE_PATH = Path("results/experiment_7339_v644_native_binding.json")
RESULT_RELATIVE_PATH = Path("results/experiment_7340_v644_native_cost.json")
RAW_RELATIVE_PATH = Path("results/raw/experiment_7340_v644_native_cost")
EXPECTED_EXP7339_SHA256 = "sha256:5c448b48b56830bee5a440e5783af9b7f6cd0141e5c00b44df986781ecf22cbb"

ARMS = ("python_in_process", "rust_in_process", "persistent_service")
BATCH_SIZES = (1, 32, 256)
PAIRED_BLOCKS = 30
WARMUP_CALLS = 3
DEVELOPMENT_SEED = 7340001
EVALUATION_SEED = 7340002
RESAMPLING_SEED = 7340003
BOOTSTRAP_SAMPLES = 10_000
TARGET_CALIBRATION_NS = 1_000_000
MAX_REPETITIONS = 512

ZERO_INVOCATIONS: JsonDict = {
    "loads": {"attempted": 0, "completed": 0, "failed": 0, "cancelled": 0, "in_flight": 0},
    "generations": {
        "attempted": 0,
        "completed": 0,
        "failed": 0,
        "cancelled": 0,
        "in_flight": 0,
    },
}

FROZEN_PROTOCOL: JsonDict = {
    "schema": "carnot.exp7340.constraint_cost_protocol.v1",
    "sealed_before_timing": True,
    "batch_sizes": list(BATCH_SIZES),
    "randomized_paired_blocks_each": PAIRED_BLOCKS,
    "arms": ["python_in_process", "rust_in_process", "historical_json_service"],
    "warmup_calls_per_size_and_arm": WARMUP_CALLS,
    "timed_native_boundary": [
        "python_object_conversion",
        "request_allocation",
        "per_batch_marshalling",
        "rust_evaluation",
        "result_allocation",
        "python_result_conversion",
    ],
    "separately_charged": [
        "extension_build",
        "extension_import",
        "constraint_compilation",
        "historical_service_build",
        "historical_service_startup",
    ],
    "stopping_rule": "exactly 30 paired blocks for each fixed size; no outcome extension",
    "claim_boundary": "prototype parity and readiness only; Exp7340 owns speed inference",
}


def canonical_bytes(value: Any) -> bytes:
    """Encode evidence without whitespace or dictionary-order ambiguity."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def sha256_json(value: Any) -> str:
    """Bind a JSON value using the repository's explicit algorithm prefix."""

    return "sha256:" + hashlib.sha256(canonical_bytes(value)).hexdigest()


def sha256_file(path: Path) -> str:
    """Hash exact bytes so a similarly named producer cannot be substituted."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _check(
    upstream: str,
    check_name: str,
    field: str,
    expected: Any,
    observed: Any,
    passed: bool,
    principle: str,
) -> JsonDict:
    """Keep a failed precondition useful instead of reducing it to one Boolean."""

    return {
        "upstream": upstream,
        "check": check_name,
        "field": field,
        "expected_value": expected,
        "observed_value": observed,
        "passed": passed,
        "principle": principle,
    }


def gate_check_summary(checks: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Retain all preflight facts and point to the first failure in stable order."""

    copied = [dict(row) for row in checks]
    failures = [row for row in copied if row.get("passed") is not True]
    return {
        "passed": not failures,
        "check_count": len(copied),
        "failed_check_count": len(failures),
        "first_failure": failures[0] if failures else None,
        "checks": copied,
    }


def _read_object(path: Path) -> JsonDict:
    """Read one declared artifact object and reject non-object JSON."""

    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("artifact_not_object")
    return value


def collect_preconditions(
    root: Path,
    *,
    expected_upstream_sha256: str = EXPECTED_EXP7339_SHA256,
) -> tuple[list[JsonDict], JsonDict, JsonDict]:
    """Authenticate Exp7339 and only the top-level fields it declares.

    The nested values below ``binding_identity`` and ``source_artifact_hashes``
    remain part of those declared top-level fields.  No raw Exp7339 score is
    consumed until status, class, quarantine, readiness, protocol, binary, and
    retained-fixture checks have all been recorded.
    """

    path = root / UPSTREAM_RELATIVE_PATH
    upstream_name = str(path)
    checks = [
        _check(
            upstream_name,
            "exp7339_available",
            "path",
            True,
            path.is_file(),
            path.is_file(),
            "The exact same-milestone native binding result must exist.",
        )
    ]
    if not path.is_file():
        return checks, {}, {}
    actual_hash = sha256_file(path)
    try:
        upstream = _read_object(path)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
        checks.append(
            _check(
                upstream_name,
                "exp7339_parseable",
                "json",
                "one object",
                type(error).__name__,
                False,
                "Malformed producer bytes cannot authorize score consumption.",
            )
        )
        return checks, {"experiment_7339": actual_hash}, {}

    checks.extend(
        [
            _check(
                upstream_name,
                "exp7339_hash",
                "sha256",
                expected_upstream_sha256,
                actual_hash,
                actual_hash == expected_upstream_sha256,
                "The benchmark consumes the exact producer rather than a same-name replacement.",
            ),
            _check(
                upstream_name,
                "exp7339_terminal",
                "status",
                "complete",
                upstream.get("status"),
                upstream.get("status") == "complete",
                "A partial producer cannot supply the benchmark extension.",
            ),
            _check(
                upstream_name,
                "exp7339_eligible_class",
                "verdict_class",
                "not blocked, disqualified, or partial",
                upstream.get("verdict_class"),
                upstream.get("verdict_class") not in {"blocked", "disqualified", "partial"},
                "Ineligible evidence stays ineligible even when another field looks positive.",
            ),
            _check(
                upstream_name,
                "exp7339_not_quarantined",
                "flagged_adversarial",
                False,
                upstream.get("flagged_adversarial"),
                upstream.get("flagged_adversarial") is not True,
                "Quarantined evidence cannot supply a score or executable identity.",
            ),
            _check(
                upstream_name,
                "exp7339_ready",
                "native_binding_ready_score",
                1,
                upstream.get("native_binding_ready_score"),
                upstream.get("native_binding_ready_score") == 1,
                "The actual runtime binding protocol must have completed before cost inference.",
            ),
            _check(
                upstream_name,
                "exp7339_required_checks",
                "required_checks_passed",
                True,
                upstream.get("required_checks_passed"),
                upstream.get("required_checks_passed") is True,
                "An upstream validation failure cannot be hidden by its readiness score.",
            ),
            _check(
                upstream_name,
                "exp7339_frozen_protocol",
                "cost_protocol",
                FROZEN_PROTOCOL,
                upstream.get("cost_protocol"),
                upstream.get("cost_protocol") == FROZEN_PROTOCOL,
                "Batch sizes, arms, warmup, and stopping rule were frozen by Exp7339.",
            ),
        ]
    )

    gates = upstream.get("acceptance_gate_results")
    gate_values = list(gates.values()) if isinstance(gates, Mapping) else []
    all_upstream_gates = bool(gate_values) and all(
        isinstance(row, Mapping) and row.get("passed") is True for row in gate_values
    )
    checks.append(
        _check(
            upstream_name,
            "exp7339_acceptance_gates",
            "acceptance_gate_results.*.passed",
            True,
            all_upstream_gates,
            all_upstream_gates,
            "A failed upstream gate overrides a terminal-looking headline.",
        )
    )

    binding = upstream.get("binding_identity")
    binding_map = binding if isinstance(binding, Mapping) else {}
    extension = Path(str(binding_map.get("module_file", "")))
    extension_available = extension.is_file()
    checks.append(
        _check(
            upstream_name,
            "exp7339_extension_available",
            "binding_identity.module_file",
            True,
            extension_available,
            extension_available,
            "E2E-003 requires the actual imported binary, not only its source.",
        )
    )
    declared_extension_hash = binding_map.get("module_sha256")
    observed_extension_hash = sha256_file(extension) if extension_available else None
    checks.append(
        _check(
            upstream_name,
            "exp7339_extension_hash",
            "binding_identity.module_sha256",
            declared_extension_hash,
            observed_extension_hash,
            extension_available and observed_extension_hash == declared_extension_hash,
            "The benchmarked binary must be the binary Exp7339 proved.",
        )
    )

    build = binding_map.get("build")
    build_map = build if isinstance(build, Mapping) else {}
    service = Path(str(build_map.get("CARGO_TARGET_DIR", ""))) / (
        "release/examples/experiment_7326_constraint_kernel"
    )
    checks.append(
        _check(
            upstream_name,
            "exp7339_service_available",
            "binding_identity.build.CARGO_TARGET_DIR",
            True,
            service.is_file(),
            service.is_file(),
            "The persistent-service comparison must use the retained built mechanism.",
        )
    )

    source_hashes = upstream.get("source_artifact_hashes")
    source_map = source_hashes if isinstance(source_hashes, Mapping) else {}
    fixtures = root / exp7339.FIXTURE_PATH
    declared_fixture_hash = source_map.get("parity_fixtures")
    observed_fixture_hash = sha256_file(fixtures) if fixtures.is_file() else None
    checks.extend(
        [
            _check(
                upstream_name,
                "retained_fixtures_available",
                "source_artifact_hashes.parity_fixtures",
                True,
                fixtures.is_file(),
                fixtures.is_file(),
                "The benchmark replays retained acquired records instead of regenerated inputs.",
            ),
            _check(
                upstream_name,
                "retained_fixtures_hash",
                "source_artifact_hashes.parity_fixtures",
                declared_fixture_hash,
                observed_fixture_hash,
                fixtures.is_file() and observed_fixture_hash == declared_fixture_hash,
                "The exact retained fixture bytes define the request population.",
            ),
        ]
    )

    rust_identity = binding_map.get("rust_source_identity")
    rust_map = rust_identity if isinstance(rust_identity, Mapping) else {}
    for relative, expected in sorted(rust_map.items()):
        source = root / str(relative)
        observed = sha256_file(source) if source.is_file() else None
        checks.append(
            _check(
                upstream_name,
                f"rust_source_hash:{relative}",
                f"binding_identity.rust_source_identity.{relative}",
                expected,
                observed,
                source.is_file() and observed == expected,
                "Current evaluation sources must still match the extension's proved identity.",
            )
        )
    return (
        checks,
        {
            "experiment_7339": actual_hash,
            "loaded_extension": observed_extension_hash,
            "retained_fixtures": observed_fixture_hash,
        },
        upstream,
    )


def eligible_upstream_fixture_for_test(root: Path) -> JsonDict:
    """Create a tiny eligible producer layout for fail-closed unit tests."""

    extension = root / "extension/_rust.fixture.so"
    extension.parent.mkdir(parents=True, exist_ok=True)
    extension.write_bytes(b"fixture-extension")
    target = root / "target"
    service = target / "release/examples/experiment_7326_constraint_kernel"
    service.parent.mkdir(parents=True, exist_ok=True)
    service.write_bytes(b"fixture-service")
    fixtures = root / exp7339.FIXTURE_PATH
    fixtures.parent.mkdir(parents=True, exist_ok=True)
    fixtures.write_text("{}\n", encoding="utf-8")
    rust_sources: JsonDict = {}
    for index, relative in enumerate(
        (
            "crates/carnot-constraints/src/schedule.rs",
            "crates/carnot-python/src/lib.rs",
            "crates/carnot-python/src/schedule.rs",
        )
    ):
        source = root / relative
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text(f"fixture {index}\n", encoding="utf-8")
        rust_sources[relative] = sha256_file(source)
    return {
        "status": "complete",
        "verdict_class": "circular_positive",
        "flagged_adversarial": False,
        "native_binding_ready_score": 1,
        "required_checks_passed": True,
        "cost_protocol": deepcopy(FROZEN_PROTOCOL),
        "acceptance_gate_results": {"runtime": {"passed": True}},
        "binding_identity": {
            "module_file": str(extension),
            "module_sha256": sha256_file(extension),
            "compiled_execution": True,
            "build": {"CARGO_TARGET_DIR": str(target)},
            "rust_source_identity": rust_sources,
        },
        "source_artifact_hashes": {"parity_fixtures": sha256_file(fixtures)},
    }


def _percentile(values: Sequence[float], probability: float) -> float:
    """Interpolate a deterministic percentile for latency and bootstrap rows."""

    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("percentile_requires_values")
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def paired_bootstrap_ci95(
    values: Sequence[float],
    *,
    seed: int = RESAMPLING_SEED,
    samples: int = BOOTSTRAP_SAMPLES,
) -> JsonDict:
    """Bootstrap the paired mean without changing the fixed block denominator."""

    observed = [float(value) for value in values]
    if not observed:
        return {"mean": None, "ci95_lower": None, "ci95_upper": None, "blocks": 0}
    rng = random.Random(seed)
    means = [
        statistics.fmean(observed[rng.randrange(len(observed))] for _ in observed)
        for _ in range(samples)
    ]
    return {
        "mean": statistics.fmean(observed),
        "ci95_lower": _percentile(means, 0.025),
        "ci95_upper": _percentile(means, 0.975),
        "blocks": len(observed),
        "resamples": samples,
    }


def _stages_sum(row: Mapping[str, Any]) -> int | None:
    """Add only the four disjoint spans that define one complete boundary."""

    names = (
        "input_marshalling_ns",
        "evaluation_ns",
        "results_ns",
        "service_transport_ns",
    )
    values = [row.get(name) for name in names]
    if not all(
        isinstance(value, int) and not isinstance(value, bool) and value >= 0 for value in values
    ):
        return None
    return sum(values)


def synthetic_cost_rows(*, native_ratio: float) -> list[JsonDict]:
    """Create a complete deterministic protocol used to test reducers and gates."""

    rows: list[JsonDict] = []
    rng = random.Random(EVALUATION_SEED)
    for size in BATCH_SIZES:
        for block in range(PAIRED_BLOCKS):
            order = list(ARMS)
            rng.shuffle(order)
            totals = {
                "python_in_process": int(size * 1_000 * native_ratio),
                "rust_in_process": size * 1_000,
                "persistent_service": int(size * 3_000 * native_ratio),
            }
            for arm in ARMS:
                total = totals[arm]
                marshalling = total // 5
                results = total // 5
                service = total // 2 if arm == "persistent_service" else 0
                evaluation = total - marshalling - results - service
                row = {
                    "unit_id": f"size-{size}-block-{block:02d}",
                    "batch_size": size,
                    "block": block,
                    "seed": EVALUATION_SEED,
                    "arm": arm,
                    "arm_order": order.index(arm),
                    "order": order,
                    "repetitions": 1,
                    "input_marshalling_ns": marshalling,
                    "evaluation_ns": evaluation,
                    "results_ns": results,
                    "service_transport_ns": service,
                    "complete_boundary_ns": total,
                    "stage_overlap_check_passed": True,
                    "parity_matched": True,
                    "output_sha256": "sha256:fixture-output",
                    "requests_per_second": size * 1_000_000_000 / total,
                    "failures": 0,
                    "abstentions": 0,
                    "censored": False,
                    "contending_work": "none; paired arms execute serially",
                }
                rows.append(row)
    return rows


def reduce_cost_rows(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Reduce only complete three-arm blocks and keep every failed check visible."""

    cells: dict[tuple[int, int], list[Mapping[str, Any]]] = {}
    for row in rows:
        cells.setdefault((int(row["batch_size"]), int(row["block"])), []).append(row)
    complete_blocks = 0
    parity_mismatches = 0
    stage_failures = 0
    by_size: JsonDict = {}
    for size_index, size in enumerate(BATCH_SIZES):
        latencies: dict[str, list[float]] = {arm: [] for arm in ARMS}
        ratios: list[float] = []
        size_complete = 0
        for block in range(PAIRED_BLOCKS):
            block_rows = cells.get((size, block), [])
            by_arm = {str(row.get("arm")): row for row in block_rows}
            hashes = {str(row.get("output_sha256")) for row in block_rows}
            stage_valid = all(
                row.get("stage_overlap_check_passed") is True
                and _stages_sum(row) == row.get("complete_boundary_ns")
                for row in block_rows
            )
            parity_valid = len(hashes) == 1 and all(
                row.get("parity_matched") is True for row in block_rows
            )
            complete = (
                len(block_rows) == len(ARMS)
                and set(by_arm) == set(ARMS)
                and stage_valid
                and parity_valid
                and all(row.get("censored") is False for row in block_rows)
                and all(row.get("failures") == 0 for row in block_rows)
            )
            if complete:
                complete_blocks += 1
                size_complete += 1
            if block_rows and not parity_valid:
                parity_mismatches += 1
            stage_failures += sum(
                row.get("stage_overlap_check_passed") is not True
                or _stages_sum(row) != row.get("complete_boundary_ns")
                for row in block_rows
            )
            for arm, row in by_arm.items():
                repetitions = row.get("repetitions")
                elapsed = row.get("complete_boundary_ns")
                if (
                    arm in latencies
                    and isinstance(repetitions, int)
                    and repetitions > 0
                    and isinstance(elapsed, int)
                ):
                    latencies[arm].append(elapsed / repetitions)
            python = by_arm.get("python_in_process")
            rust = by_arm.get("rust_in_process")
            if python is not None and rust is not None:
                python_repetitions = int(python.get("repetitions", 0))
                rust_repetitions = int(rust.get("repetitions", 0))
                python_elapsed = int(python.get("complete_boundary_ns", 0))
                rust_elapsed = int(rust.get("complete_boundary_ns", 0))
                if python_repetitions > 0 and rust_repetitions > 0 and rust_elapsed > 0:
                    ratios.append(
                        (python_elapsed / python_repetitions) / (rust_elapsed / rust_repetitions)
                    )
        by_size[str(size)] = {
            "complete_blocks": size_complete,
            "latency_ns": {
                arm: {
                    "p50": _percentile(values, 0.50) if values else None,
                    "p95": _percentile(values, 0.95) if values else None,
                }
                for arm, values in latencies.items()
            },
            "throughput_ratio_native_over_python": paired_bootstrap_ci95(
                ratios, seed=RESAMPLING_SEED + size_index
            ),
        }
    expected_rows = len(BATCH_SIZES) * PAIRED_BLOCKS * len(ARMS)
    expected_blocks = len(BATCH_SIZES) * PAIRED_BLOCKS
    return {
        "row_count": len(rows),
        "expected_row_count": expected_rows,
        "paired_blocks": complete_blocks,
        "expected_paired_blocks": expected_blocks,
        "parity_mismatches": parity_mismatches,
        "stage_overlap_failures": stage_failures,
        "complete": len(rows) == expected_rows
        and complete_blocks == expected_blocks
        and parity_mismatches == 0
        and stage_failures == 0,
        "by_batch_size": by_size,
    }


def derive_scores(reduced: Mapping[str, Any]) -> JsonDict:
    """Keep accounting completion separate from the unchanged ten-x value gate."""

    complete = reduced.get("complete") is True
    by_size = reduced.get("by_batch_size")
    size_map = by_size if isinstance(by_size, Mapping) else {}
    ten_x = complete and all(
        isinstance(size_map.get(str(size)), Mapping)
        and isinstance(size_map[str(size)].get("throughput_ratio_native_over_python"), Mapping)
        and isinstance(
            size_map[str(size)]["throughput_ratio_native_over_python"].get("ci95_lower"),
            (int, float),
        )
        and size_map[str(size)]["throughput_ratio_native_over_python"]["ci95_lower"] >= 10
        for size in BATCH_SIZES
    )
    return {
        "native_cost_complete_score": int(complete),
        "native_ten_x_score": int(ten_x),
    }


def break_even_request_count(
    setup_ns: int,
    python_per_request_ns: float,
    rust_per_request_ns: float,
) -> int | None:
    """Return requests needed to repay setup, or unavailable when Rust is not cheaper."""

    savings = python_per_request_ns - rust_per_request_ns
    if savings <= 0:
        return None
    return math.ceil(setup_ns / savings)


def amortization_summary(*, setup_costs: Mapping[str, Any], reduced: Mapping[str, Any]) -> JsonDict:
    """Charge non-overlapping setup once and refuse an unsupported whole-path bound."""

    setup_ns = sum(
        int(value)
        for value in setup_costs.values()
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0
    )
    by_size = reduced.get("by_batch_size")
    size_map = by_size if isinstance(by_size, Mapping) else {}
    break_even: JsonDict = {}
    for size in BATCH_SIZES:
        size_row = size_map.get(str(size), {})
        latency = size_row.get("latency_ns", {}) if isinstance(size_row, Mapping) else {}
        python_row = latency.get("python_in_process", {}) if isinstance(latency, Mapping) else {}
        rust_row = latency.get("rust_in_process", {}) if isinstance(latency, Mapping) else {}
        python_batch = python_row.get("p50") if isinstance(python_row, Mapping) else None
        rust_batch = rust_row.get("p50") if isinstance(rust_row, Mapping) else None
        break_even[str(size)] = (
            break_even_request_count(setup_ns, float(python_batch) / size, float(rust_batch) / size)
            if isinstance(python_batch, (int, float)) and isinstance(rust_batch, (int, float))
            else None
        )
    return {
        "setup_ns": setup_ns,
        "setup_components_ns": dict(setup_costs),
        "break_even_requests_by_size": break_even,
        "v643_whole_learning_upper_bound": None,
        "v643_upper_bound_unavailable_reason": (
            "V643 does not expose an identity-matched non-overlapping whole-learning stage ledger"
        ),
        "whole_learning_speedup_claimed": False,
    }


def e2e_003_identity(extension: Path, binding_identity: Mapping[str, Any]) -> JsonDict:
    """Recheck that the binary used for runtime replay is the proved Exp7339 binary."""

    observed = {
        "module_file": str(extension),
        "module_sha256": sha256_file(extension) if extension.is_file() else None,
    }
    expected = {
        "module_file": str(binding_identity.get("module_file")),
        "module_sha256": binding_identity.get("module_sha256"),
    }
    passed = extension.is_file() and observed == expected
    return {
        "expected": expected,
        "observed": observed,
        "passed": passed,
        "principle": "E2E-003 binds the measured round trip to the exact loaded extension.",
    }


def _gate(expected: Any, observed: Any, passed: bool, principle: str) -> JsonDict:
    """Keep each frozen expectation beside its observation and decision."""

    return {"expected": expected, "observed": observed, "passed": passed, "principle": principle}


def _field_principles() -> JsonDict:
    """Explain schema fields without wrapping values that validators must execute."""

    return {
        "schema": "Version the artifact while preserving ordinary top-level experiment_id and milestone.",
        "status": "Publish a terminal result only after current work and affected validation.",
        "run_date": "Use 20260916 and retain actual UTC timestamps.",
        "preconditions_checked": "Name input identity, availability, and each failed check before work.",
        "MODEL_SPECS": "List current executable identities; no LLM is intended in this task.",
        "model_invoked": "True for any real model load or generation attempt, including a failed attempt.",
        "invocation_counts": "Separate attempted, completed, failed, cancelled, and in-flight work.",
        "inference_substrate": "Declare deterministic CPU evaluator work, not cited historical model work.",
        "inference_substrate_class": "Use the closed duration class exercised by current computation.",
        "execution_venue": "Use host; retained board receipts do not imply current board execution.",
        "duration_s": "Measure actual monotonic elapsed time without a padded floor.",
        "phase_spans": "Use disjoint spans, completed units, checkpoints, and pending operations.",
        "random_seed": "Freeze development, evaluation, and resampling seeds before outcomes.",
        "reproducibility_checksum": "Bind code, settings, inputs, evaluator identity, and raw evidence.",
        "source_artifact_hashes": "Authenticate exact producers, current sources, and task-owned raw rows.",
        "rows": "Retain every size, seed, arm, repetition count, complete time, and parity result.",
        "sample_size_budget": "Record planned, attempted, completed, censored, and the fixed stopping rule.",
        "acceptance_gate_results": "Record expected, observed, and passed while separating completion from value.",
        "gate_check_summary": "Every blocked result points to upstream, failed check, field, expected, and observed.",
        "verifier_is_oracle": "The shared execution authority defines correctness, so circularity remains explicit.",
        "honest_verdict": "Completed findings start complete_; external absence starts blocked_ and names the check.",
        "verdict_class": "Use only positive, circular_positive, null, blocked, disqualified, or partial.",
        "validation_receipts": "Retain exact command, scope, exit, duration, and log hash, including failures.",
        "repository_health": "Keep unrelated dated failures separate from affected required checks.",
        "field_principles": "Explain fields without wrapping executable scores or ordinary dictionaries.",
        "native_cost_complete_score": "One requires all ninety paired blocks, complete costs, and parity.",
        "native_ten_x_score": "One requires the unchanged ten-x lower-bound gate at every size.",
        "amortization": "Separate setup, per-request savings, and supported whole-path bounds.",
    }


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Bind deterministic evidence while excluding wall clocks and command duration noise."""

    excluded = {
        "started_at_utc",
        "completed_at_utc",
        "duration_s",
        "phase_spans",
        "validation_receipts",
        "repository_health",
        "reproducibility_checksum",
    }
    return sha256_json({key: value for key, value in artifact.items() if key not in excluded})


def _base_artifact(checks: Sequence[Mapping[str, Any]], hashes: Mapping[str, Any]) -> JsonDict:
    """Create one schema-complete candidate before scientific scores are assigned."""

    return {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "status": "partial",
        "run_date": RUN_DATE,
        "started_at_utc": datetime.now(UTC).isoformat(),
        "completed_at_utc": None,
        "MODEL_SPECS": [],
        "model_invoked": False,
        "invocation_counts": deepcopy(ZERO_INVOCATIONS),
        "inference_substrate": "cpu_exact_solver_or_simulator",
        "inference_substrate_class": "cpu_exact_solver_or_simulator",
        "execution_venue": "host",
        "preconditions_checked": [dict(row) for row in checks],
        "gate_check_summary": gate_check_summary(checks),
        "cost_protocol": deepcopy(FROZEN_PROTOCOL),
        "random_seed": {
            "development": DEVELOPMENT_SEED,
            "evaluation": EVALUATION_SEED,
            "resampling": RESAMPLING_SEED,
            "sealed_before_results": True,
        },
        "source_artifact_hashes": deepcopy(dict(hashes)),
        "rows": [],
        "cost_summary": {},
        "sample_size_budget": {
            "paired_blocks_per_size": PAIRED_BLOCKS,
            "batch_sizes": list(BATCH_SIZES),
            "paired_blocks_planned": 90,
            "paired_blocks_attempted": 0,
            "paired_blocks_completed": 0,
            "paired_blocks_censored": 0,
            "rows_planned": 270,
            "rows_attempted": 0,
            "rows_completed": 0,
            "rows_censored": 0,
            "stopping_rule": FROZEN_PROTOCOL["stopping_rule"],
        },
        "setup_costs": {},
        "amortization": {},
        "acceptance_gate_results": {},
        "native_cost_complete_score": 0,
        "native_ten_x_score": 0,
        "readiness_score": 0,
        "value_score": 0,
        "promotion_score": 0,
        "verifier_is_oracle": True,
        "whole_learning_speedup_claimed": False,
        "positive_control_passed": False,
        "false_negative_risk_checked": False,
        "honest_verdict": "partial: measurement has not completed",
        "verdict_class": "partial",
        "retirement": {"applied": False, "scope": None, "broad_native_claim": False},
        "phase_spans": [],
        "duration_s": 0.0,
        "validation_receipts": [],
        "repository_health": {
            "status": "healthy",
            "incident_open": False,
            "historical_failures": [],
            "historical_failure_count": 0,
            "affects_required_checks": False,
        },
        "methodology": {
            "design": "seeded paired complete-boundary host benchmark",
            "claim_boundary": "acquired-constraint evaluator only",
            "model_work": "none",
        },
        "field_principles": _field_principles(),
        "reproducibility_checksum": "pending",
    }


def blocked_artifact(checks: Sequence[Mapping[str, Any]], hashes: Mapping[str, Any]) -> JsonDict:
    """Publish external absence as terminal blocked evidence with no synthetic rows."""

    artifact = _base_artifact(checks, hashes)
    failure = artifact["gate_check_summary"]["first_failure"]
    check_name = failure["check"] if isinstance(failure, Mapping) else "unknown"
    artifact.update(
        {
            "status": "complete",
            "completed_at_utc": datetime.now(UTC).isoformat(),
            "honest_verdict": (
                f"blocked_{check_name}: upstream {failure['upstream']} field {failure['field']} "
                f"expected {failure['expected_value']!r}; observed {failure['observed_value']!r}"
            ),
            "verdict_class": "blocked",
        }
    )
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def _complete_artifact(
    artifact: JsonDict,
    rows: Sequence[Mapping[str, Any]],
    reduced: Mapping[str, Any],
    setup_costs: Mapping[str, Any],
    *,
    affected_validation_passed: bool,
    e2e_passed: bool,
    adverse_passed: bool,
) -> JsonDict:
    """Apply measured rows and derive a null without converting it into unfinished work."""

    scores = derive_scores(reduced)
    cost_complete = scores["native_cost_complete_score"] == 1
    ten_x = scores["native_ten_x_score"] == 1
    ready = cost_complete and affected_validation_passed and e2e_passed and adverse_passed
    artifact.update(scores)
    artifact.update(
        {
            "status": "complete",
            "completed_at_utc": datetime.now(UTC).isoformat(),
            "rows": [dict(row) for row in rows],
            "cost_summary": deepcopy(dict(reduced)),
            "setup_costs": deepcopy(dict(setup_costs)),
            "amortization": amortization_summary(setup_costs=setup_costs, reduced=reduced),
            "sample_size_budget": {
                "paired_blocks_per_size": PAIRED_BLOCKS,
                "batch_sizes": list(BATCH_SIZES),
                "paired_blocks_planned": 90,
                "paired_blocks_attempted": 90,
                "paired_blocks_completed": int(reduced.get("paired_blocks", 0)),
                "paired_blocks_censored": 90 - int(reduced.get("paired_blocks", 0)),
                "rows_planned": 270,
                "rows_attempted": len(rows),
                "rows_completed": sum(row.get("censored") is False for row in rows),
                "rows_censored": sum(row.get("censored") is True for row in rows),
                "stopping_rule": FROZEN_PROTOCOL["stopping_rule"],
            },
            "acceptance_gate_results": {
                "native_cost_completion": _gate(
                    {"paired_blocks": 90, "rows": 270, "parity_mismatches": 0},
                    dict(reduced),
                    cost_complete,
                    "Accounting completion requires every fixed block, cost span, and parity result.",
                ),
                "native_ten_x": _gate(
                    "CI95 lower bound >= 10 at sizes 1, 32, and 256",
                    {
                        str(size): reduced["by_batch_size"][str(size)][
                            "throughput_ratio_native_over_python"
                        ]["ci95_lower"]
                        for size in BATCH_SIZES
                    },
                    ten_x,
                    "The unchanged value gate cannot move after observing a smaller gain.",
                ),
                "e2e_003": _gate(
                    "benchmarked extension identity and zero mismatches",
                    {"identity": e2e_passed, "mismatches": reduced.get("parity_mismatches")},
                    e2e_passed and reduced.get("parity_mismatches") == 0,
                    "The loaded binary and ordinary Python result boundary must both be exercised.",
                ),
                "adverse_row_and_stage_checks": _gate(
                    "hash and stage mutations are rejected",
                    adverse_passed,
                    adverse_passed,
                    "A reducer that accepts corrupted rows cannot authorize terminal evidence.",
                ),
                "affected_validation": _gate(
                    "all scoped affected checks pass",
                    affected_validation_passed,
                    affected_validation_passed,
                    "Current worktree tests, coverage, lint, format, typing, and spec coverage are required.",
                ),
            },
            "readiness_score": int(ready),
            "value_score": int(ready and ten_x),
            "promotion_score": int(ready and ten_x),
            "positive_control_passed": e2e_passed and reduced.get("parity_mismatches") == 0,
            "false_negative_risk_checked": True,
        }
    )
    if not ready:
        artifact.update(
            {
                "native_ten_x_score": 0,
                "readiness_score": 0,
                "value_score": 0,
                "promotion_score": 0,
                "verdict_class": "disqualified",
                "honest_verdict": "complete_disqualified: required runtime or affected validation failed",
            }
        )
    elif ten_x:
        artifact.update(
            {
                "verdict_class": "positive",
                "honest_verdict": (
                    "complete_positive: the native complete-boundary throughput CI95 lower bound "
                    "reaches ten at every frozen size; no whole-learning or hardware claim follows"
                ),
            }
        )
    else:
        artifact.update(
            {
                "verdict_class": "null",
                "honest_verdict": (
                    "complete_null: exact parity and all ninety complete-boundary blocks finished, "
                    "but the unchanged native ten-x lower-bound gate failed"
                ),
                "retirement": {
                    "applied": True,
                    "scope": "this_exact_complete_boundary",
                    "reason": "repeated performance null at the acquired-constraint evaluator boundary",
                    "broad_native_claim": False,
                    "hardware_claim": False,
                },
            }
        )
    artifact["whole_learning_speedup_claimed"] = False
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def complete_artifact_fixture_for_test(root: Path, *, native_ratio: float) -> JsonDict:
    """Build deterministic terminal evidence for cold validation and tamper tests."""

    del root  # The fixture has no external paths and therefore cannot imply runtime proof.
    checks: list[JsonDict] = []
    artifact = _base_artifact(checks, {})
    rows = synthetic_cost_rows(native_ratio=native_ratio)
    reduced = reduce_cost_rows(rows)
    artifact = _complete_artifact(
        artifact,
        rows,
        reduced,
        {"cold_import_ns": 4_000, "constraint_compilation_ns": 5_000},
        affected_validation_passed=True,
        e2e_passed=True,
        adverse_passed=True,
    )
    artifact["acceptance_gate_results"]["terminal_validators"] = _gate(
        "both validators pass", True, True, "Fixture models completed terminal validation."
    )
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> list[str]:
    """Cold-check identity, scores, rows, safety claims, and deterministic checksum."""

    errors: list[str] = []

    def add(condition: bool, name: str) -> None:
        if condition:
            errors.append(name)

    add(artifact.get("schema") != SCHEMA, "schema")
    add(artifact.get("experiment_id") != EXPERIMENT_ID, "experiment_id")
    add(artifact.get("milestone") != MILESTONE, "milestone")
    add(artifact.get("run_date") != RUN_DATE, "run_date")
    add(artifact.get("status") != "complete", "status")
    add(artifact.get("MODEL_SPECS") != [], "MODEL_SPECS")
    add(artifact.get("model_invoked") is not False, "model_invoked")
    add(artifact.get("invocation_counts") != ZERO_INVOCATIONS, "invocation_counts")
    add(artifact.get("inference_substrate") != "cpu_exact_solver_or_simulator", "substrate")
    add(
        artifact.get("inference_substrate_class") != "cpu_exact_solver_or_simulator",
        "substrate_class",
    )
    add(artifact.get("execution_venue") != "host", "execution_venue")
    add(artifact.get("verifier_is_oracle") is not True, "verifier_is_oracle")
    add(artifact.get("whole_learning_speedup_claimed") is not False, "whole_learning_claim")
    add(artifact.get("cost_protocol") != FROZEN_PROTOCOL, "cost_protocol")
    verdict_class = artifact.get("verdict_class")
    add(
        verdict_class not in {"positive", "circular_positive", "null", "blocked", "disqualified"},
        "verdict_class",
    )
    verdict = artifact.get("honest_verdict")
    add(not isinstance(verdict, str), "honest_verdict")
    rows = artifact.get("rows")
    if verdict_class == "blocked":
        add(rows != [], "blocked_rows")
        add(artifact.get("native_cost_complete_score") != 0, "blocked_complete_score")
        add(artifact.get("native_ten_x_score") != 0, "blocked_ten_x_score")
        add(not isinstance(verdict, str) or not verdict.startswith("blocked_"), "blocked_prefix")
    else:
        add(not isinstance(rows, list), "rows")
        if isinstance(rows, list):
            reduced = reduce_cost_rows(rows)
            add(reduced != artifact.get("cost_summary"), "row_reduction")
            scores = derive_scores(reduced)
            add(
                scores["native_cost_complete_score"] != artifact.get("native_cost_complete_score"),
                "native_cost_complete_score",
            )
            if verdict_class not in {"disqualified"}:
                add(
                    scores["native_ten_x_score"] != artifact.get("native_ten_x_score"),
                    "native_ten_x_score",
                )
        add(not isinstance(verdict, str) or not verdict.startswith("complete_"), "complete_prefix")
    if verdict_class in {"blocked", "disqualified"}:
        for name in ("readiness_score", "value_score", "promotion_score", "native_ten_x_score"):
            add(artifact.get(name) != 0, f"ineligible_{name}")
    add(artifact.get("reproducibility_checksum") != reproducibility_checksum(artifact), "checksum")
    return sorted(set(errors))


def write_artifact(path: Path, artifact: Mapping[str, Any]) -> JsonDict:
    """Validate and atomically publish one canonical terminal document."""

    errors = validate_artifact(artifact)
    if errors:
        raise ValueError("artifact_validation_failed:" + ",".join(errors))
    exp7339._atomic_json(path, artifact)
    return {"path": str(path), "sha256": sha256_file(path)}


def prepare_scoped_basetemp(path: Path) -> Path:
    """Create the private parent required by the shipped scoped pytest runner."""

    return exp7326.prepare_scoped_basetemp(path)


class _PinnedArmWorker:  # pragma: no cover - measured subprocess boundary.
    """Own one long-lived Python or Rust arm and pin it before timed work."""

    def __init__(self, arm: str, root: Path, cpu: int | None) -> None:
        self.arm = arm
        started = time.perf_counter_ns()
        self.process = subprocess.Popen(  # noqa: S603
            (sys.executable, "-u", "-m", __name__, "--serve-arm", arm),
            cwd=root,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        if cpu is not None and hasattr(os, "sched_setaffinity"):
            os.sched_setaffinity(self.process.pid, {cpu})
        assert self.process.stdin is not None and self.process.stdout is not None
        self.input: TextIO = self.process.stdin
        self.output: TextIO = self.process.stdout
        pong = self.call({"operation": "ping"})
        if pong.get("kind") != "ready":
            raise RuntimeError(f"{arm}_worker_ping_failed")
        self.cold_start_ns = time.perf_counter_ns() - started
        self.cpu = cpu

    def call(self, message: Mapping[str, Any]) -> JsonDict:
        self.input.write(canonical_bytes(message).decode() + "\n")
        self.input.flush()
        line = self.output.readline()
        if not line:
            detail = self.process.stderr.read() if self.process.stderr is not None else ""
            raise RuntimeError(f"{self.arm}_worker_closed:{detail}")
        value = json.loads(line)
        if not isinstance(value, dict):
            raise RuntimeError(f"{self.arm}_worker_non_object")
        return value

    def close(self) -> None:
        try:
            self.call({"operation": "shutdown"})
        finally:
            self.process.wait(timeout=10)


def _serve_arm(arm: str) -> int:  # pragma: no cover - measured subprocess boundary.
    """Serve one in-process evaluator so each benchmark arm owns one host process."""

    evaluator: Any = None
    for line in sys.stdin:
        message = json.loads(line)
        operation = message.get("operation")
        if operation == "ping":
            response: JsonDict = {"kind": "ready"}
        elif operation == "initialize":
            if arm == "rust_in_process":
                started = time.perf_counter_ns()
                binding = exp7339.load_native_extension(Path(message["extension"]))
                import_ns = time.perf_counter_ns() - started
                started = time.perf_counter_ns()
                evaluator = binding.RustCompiledScheduleEvaluator(deepcopy(message["constraints"]))
                compilation_ns = time.perf_counter_ns() - started
            else:
                import_ns = 0
                compilation_ns = 0
            response = {
                "kind": "initialized",
                "extension_import_ns": import_ns,
                "constraint_compilation_ns": compilation_ns,
            }
        elif operation == "evaluate":
            requests = message["requests"]
            repetitions = int(message["repetitions"])
            marshalling_ns = 0
            evaluation_ns = 0
            results_ns = 0
            detached: list[JsonDict] = []
            for _ in range(repetitions):
                started = time.perf_counter_ns()
                if arm == "rust_in_process":
                    prepared = [exp7339.compact_request(request) for request in requests]
                else:
                    prepared = deepcopy(requests)
                marshalling_ns += time.perf_counter_ns() - started
                started = time.perf_counter_ns()
                if arm == "rust_in_process":
                    raw = evaluator.evaluate_batch(prepared)
                else:
                    raw = exp7326.evaluate_batch(prepared)
                evaluation_ns += time.perf_counter_ns() - started
                started = time.perf_counter_ns()
                detached = [deepcopy(dict(row)) for row in raw]
                results_ns += time.perf_counter_ns() - started
            response = {
                "kind": "result",
                "input_marshalling_ns": marshalling_ns,
                "evaluation_ns": evaluation_ns,
                "results_ns": results_ns,
                "service_transport_ns": 0,
                "results": detached,
            }
        elif operation == "shutdown":
            print('{"kind":"stopped"}', flush=True)
            return 0
        else:
            response = {"kind": "error", "error": "unknown_operation"}
        print(canonical_bytes(response).decode(), flush=True)
    return 0


def _service_call(
    service: exp7326.JsonLineService,
    requests: Sequence[Mapping[str, Any]],
    repetitions: int,
) -> JsonDict:  # pragma: no cover - measured subprocess boundary.
    """Measure JSON input, service wait, and result parse as disjoint real spans."""

    input_ns = 0
    transport_ns = 0
    result_ns = 0
    results: list[JsonDict] = []
    message = {"operation": "evaluate", "requests": requests}
    for _ in range(repetitions):
        started = time.perf_counter_ns()
        payload = canonical_bytes(message).decode() + "\n"
        input_ns += time.perf_counter_ns() - started
        started = time.perf_counter_ns()
        service.input.write(payload)
        service.input.flush()
        line = service.output.readline()
        transport_ns += time.perf_counter_ns() - started
        if not line:
            raise RuntimeError("persistent_service_closed")
        started = time.perf_counter_ns()
        response = json.loads(line)
        results = list(response["results"])
        result_ns += time.perf_counter_ns() - started
    return {
        "input_marshalling_ns": input_ns,
        "evaluation_ns": 0,
        "results_ns": result_ns,
        "service_transport_ns": transport_ns,
        "results": results,
        "service_attribution": "transport includes service parse, Rust evaluation, and serialization",
    }


def _available_cpus() -> list[int | None]:  # pragma: no cover - host-specific measurement.
    """Choose three allowed CPUs; a one-CPU host truthfully reuses that CPU."""

    if hasattr(os, "sched_getaffinity"):
        allowed = sorted(os.sched_getaffinity(0))
    else:
        allowed = []
    if not allowed:
        return [None, None, None]
    return [allowed[index % len(allowed)] for index in range(3)]


def _call_measured_arm(
    arm: str,
    python_worker: _PinnedArmWorker,
    rust_worker: _PinnedArmWorker,
    service: exp7326.JsonLineService,
    requests: Sequence[Mapping[str, Any]],
    repetitions: int,
) -> JsonDict:  # pragma: no cover - measured subprocess boundary.
    """Call exactly one pinned arm while the other experiment arms remain idle."""

    if arm == "python_in_process":
        return python_worker.call(
            {"operation": "evaluate", "requests": requests, "repetitions": repetitions}
        )
    if arm == "rust_in_process":
        return rust_worker.call(
            {"operation": "evaluate", "requests": requests, "repetitions": repetitions}
        )
    return _service_call(service, requests, repetitions)


def benchmark_complete_boundary(
    root: Path,
    extension: Path,
    service_path: Path,
    full_requests: Sequence[Mapping[str, Any]],
    constraints: Sequence[Mapping[str, Any]],
) -> tuple[list[JsonDict], JsonDict, JsonDict]:  # pragma: no cover - real benchmark.
    """Run 90 fixed paired blocks through three pinned persistent processes."""

    if len(full_requests) < 2:
        raise RuntimeError("development_evaluation_split_unavailable")
    split = min(16, max(1, len(full_requests) // 5))
    development = [deepcopy(dict(row)) for row in full_requests[:split]]
    evaluation = [deepcopy(dict(row)) for row in full_requests[split:]]
    cpus = _available_cpus()
    print("[exp7340] phase=worker_start event=before", flush=True)
    python_worker = _PinnedArmWorker("python_in_process", root, cpus[0])
    rust_worker = _PinnedArmWorker("rust_in_process", root, cpus[1])
    python_setup = python_worker.call({"operation": "initialize"})
    rust_setup = rust_worker.call(
        {
            "operation": "initialize",
            "extension": str(extension),
            "constraints": constraints,
        }
    )
    service_started = time.perf_counter_ns()
    service = exp7326.JsonLineService((str(service_path),), root)
    service_startup_ns = time.perf_counter_ns() - service_started
    if cpus[2] is not None and hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(service.process.pid, {cpus[2]})
    print("[exp7340] phase=worker_start event=after", flush=True)
    rows: list[JsonDict] = []
    calibration: JsonDict = {}
    rng = random.Random(EVALUATION_SEED)
    benchmark_started = time.monotonic()
    try:
        for size in BATCH_SIZES:
            development_batch = [
                deepcopy(development[index % len(development)]) for index in range(size)
            ]
            one_call: JsonDict = {}
            for arm in ARMS:
                measured = _call_measured_arm(
                    arm, python_worker, rust_worker, service, development_batch, 1
                )
                one_call[arm] = sum(
                    int(measured[name])
                    for name in (
                        "input_marshalling_ns",
                        "evaluation_ns",
                        "results_ns",
                        "service_transport_ns",
                    )
                )
            repetitions = min(
                MAX_REPETITIONS,
                max(1, math.ceil(TARGET_CALIBRATION_NS / max(1, min(one_call.values())))),
            )
            calibration[str(size)] = {
                "development_request_count": len(development_batch),
                "one_call_ns": one_call,
                "target_ns": TARGET_CALIBRATION_NS,
                "repetitions": repetitions,
                "selected_before_evaluation_blocks": True,
            }
            evaluation_batch = [
                deepcopy(evaluation[index % len(evaluation)]) for index in range(size)
            ]
            for _ in range(WARMUP_CALLS):
                for arm in ARMS:
                    _call_measured_arm(
                        arm, python_worker, rust_worker, service, evaluation_batch, repetitions
                    )
            for block in range(PAIRED_BLOCKS):
                order = list(ARMS)
                rng.shuffle(order)
                outputs: JsonDict = {}
                measurements: JsonDict = {}
                for arm in order:
                    result = _call_measured_arm(
                        arm, python_worker, rust_worker, service, evaluation_batch, repetitions
                    )
                    outputs[arm] = result.pop("results")
                    measurements[arm] = result
                hashes = {arm: sha256_json(outputs[arm]) for arm in ARMS}
                control_hash = hashes["python_in_process"]
                for arm in order:
                    measurement = measurements[arm]
                    complete_ns = sum(
                        int(measurement[name])
                        for name in (
                            "input_marshalling_ns",
                            "evaluation_ns",
                            "results_ns",
                            "service_transport_ns",
                        )
                    )
                    matched = hashes[arm] == control_hash
                    rows.append(
                        {
                            "unit_id": f"size-{size}-block-{block:02d}",
                            "batch_size": size,
                            "block": block,
                            "seed": EVALUATION_SEED,
                            "arm": arm,
                            "arm_order": order.index(arm),
                            "order": order,
                            "repetitions": repetitions,
                            "input_marshalling_ns": measurement["input_marshalling_ns"],
                            "evaluation_ns": measurement["evaluation_ns"],
                            "results_ns": measurement["results_ns"],
                            "service_transport_ns": measurement["service_transport_ns"],
                            "complete_boundary_ns": complete_ns,
                            "stage_overlap_check_passed": True,
                            "parity_matched": matched,
                            "output_sha256": hashes[arm],
                            "requests_per_second": (
                                size * repetitions * 1_000_000_000 / complete_ns
                            ),
                            "failures": int(not matched),
                            "abstentions": 0,
                            "censored": False,
                            "contending_work": (
                                "benchmark arms execute serially; other experiment workers idle; "
                                "ambient host work not controlled"
                            ),
                            "stage_attribution": measurement.get(
                                "service_attribution",
                                "worker-internal disjoint complete-boundary spans",
                            ),
                        }
                    )
                print(
                    f"[exp7340] phase=benchmark event=unit_complete size={size} "
                    f"block={block + 1}/{PAIRED_BLOCKS} "
                    f"elapsed_s={time.monotonic() - benchmark_started:.3f}",
                    flush=True,
                )
    finally:
        python_worker.close()
        rust_worker.close()
        service.close()
    setup = {
        "extension_import_ns": int(rust_setup["extension_import_ns"]),
        "constraint_compilation_ns": int(rust_setup["constraint_compilation_ns"]),
    }
    process_setup = {
        "python_worker_cold_start_ns": python_worker.cold_start_ns,
        "rust_worker_cold_start_ns": rust_worker.cold_start_ns,
        "persistent_service_cold_start_ns": service_startup_ns,
        "pinned_cpu_by_arm": {
            "python_in_process": cpus[0],
            "rust_in_process": cpus[1],
            "persistent_service": cpus[2],
        },
        "python_initialize": python_setup,
        "warmup_calls_per_size_and_arm": WARMUP_CALLS,
    }
    return rows, {**setup, **process_setup}, calibration


def _phase(
    spans: list[JsonDict], name: str, started: float, origin: float, units: int
) -> None:  # pragma: no cover - runtime clocks.
    """Append one disjoint completed phase span with no invented pending work."""

    end = time.monotonic() - origin
    spans.append(
        {
            "phase": name,
            "start_s": started - origin,
            "end_s": end,
            "completed_units": units,
            "checkpoint_positions": [units],
            "pending_operations": [],
        }
    )


def _source_hashes(root: Path) -> JsonDict:  # pragma: no cover - runtime identity.
    """Bind the exact worktree sources that define protocol, implementation, and tests."""

    paths = (
        "python/carnot/experiment_7340_v644_native_cost.py",
        "scripts/experiments/experiment_7340_v644_native_cost.py",
        "tests/python/test_experiment_7340_v644_native_cost.py",
        "python/carnot/experiment_7339_v644_native_binding.py",
        "python/carnot/experiment_7326_v643_constraint_kernel.py",
        "python/carnot/reporting/experiment_7303_validation_scope.py",
        "crates/carnot-constraints/src/schedule.rs",
        "crates/carnot-python/src/schedule.rs",
        "openspec/capabilities/constraint-verification/spec.md",
        "openspec/capabilities/python-bindings/spec.md",
        "ops/e2e-test-plan.md",
    )
    return {path: sha256_file(root / path) for path in paths}


def _adverse_checks(rows: Sequence[Mapping[str, Any]]) -> JsonDict:  # pragma: no cover
    """Prove row hashes and stage sums cannot be changed without failing reduction."""

    hash_mutation = deepcopy(list(rows))
    hash_mutation[0]["output_sha256"] = "sha256:adverse-row-hash"
    stage_mutation = deepcopy(list(rows))
    stage_mutation[0]["complete_boundary_ns"] += 1
    hash_reduced = reduce_cost_rows(hash_mutation)
    stage_reduced = reduce_cost_rows(stage_mutation)
    return {
        "hash_mutation_rejected": hash_reduced["complete"] is False
        and hash_reduced["parity_mismatches"] > 0,
        "stage_overlap_mutation_rejected": stage_reduced["complete"] is False
        and stage_reduced["stage_overlap_failures"] > 0,
        "passed": hash_reduced["complete"] is False
        and hash_reduced["parity_mismatches"] > 0
        and stage_reduced["complete"] is False
        and stage_reduced["stage_overlap_failures"] > 0,
    }


def _validation_receipt(
    name: str, command: str, scope: str, duration_s: float, path: Path, passed: bool
) -> JsonDict:  # pragma: no cover - runtime receipt.
    """Record one in-process evidence check using the exact raw file hash."""

    return {
        "name": name,
        "command": command,
        "scope": scope,
        "exit_code": 0 if passed else 1,
        "duration_s": duration_s,
        "log_path": str(path),
        "log_sha256": sha256_file(path),
        "passed": passed,
        "timed_out": False,
    }


def _terminal_validators(
    root: Path, candidate: Path, raw: Path
) -> list[JsonDict]:  # pragma: no cover - subprocess integration.
    """Run both required artifact validators against the measured candidate."""

    commands = [
        CommandSpec(
            "adversarial_verify",
            (str(root / ".venv/bin/python"), "-u", "scripts/adversarial_verify.py", str(candidate)),
            "measured terminal candidate",
        ),
        CommandSpec(
            "verdict_row_consistency_strict",
            (
                str(root / ".venv/bin/python"),
                "-u",
                "scripts/verdict_row_consistency_lint.py",
                "--strict",
                str(candidate),
            ),
            "measured terminal candidate",
        ),
    ]
    return run_commands(root, commands, log_dir=raw / "validation/terminal")


def run_experiment(root: Path) -> JsonDict:  # pragma: no cover - real end-to-end entrypoint.
    """Authenticate, benchmark, independently reduce, validate, and return terminal evidence."""

    origin = time.monotonic()
    spans: list[JsonDict] = []
    print("[exp7340] phase=preflight event=start", flush=True)
    started = time.monotonic()
    checks, hashes, upstream = collect_preconditions(root)
    hashes["current_sources"] = _source_hashes(root)
    artifact = _base_artifact(checks, hashes)
    if isinstance(upstream.get("repository_health"), Mapping):
        artifact["repository_health"] = deepcopy(upstream["repository_health"])
    _phase(spans, "preflight", started, origin, len(checks))
    artifact["phase_spans"] = spans
    if not artifact["gate_check_summary"]["passed"]:
        blocked = blocked_artifact(checks, hashes)
        blocked["phase_spans"] = spans
        blocked["duration_s"] = time.monotonic() - origin
        blocked["reproducibility_checksum"] = reproducibility_checksum(blocked)
        return blocked

    raw = root / RAW_RELATIVE_PATH
    raw.mkdir(parents=True, exist_ok=True)
    print("[exp7340] phase=protocol_seal event=start", flush=True)
    started = time.monotonic()
    protocol_path = raw / "cost_protocol.json"
    exp7339._atomic_json(protocol_path, FROZEN_PROTOCOL)
    artifact["source_artifact_hashes"]["cost_protocol"] = sha256_file(protocol_path)
    _phase(spans, "protocol_seal", started, origin, 1)

    print("[exp7340] phase=fixture_load event=start", flush=True)
    started = time.monotonic()
    fixtures = exp7339.read_jsonl(root / exp7339.FIXTURE_PATH)
    full_requests, constraints = exp7339._benchmark_requests(fixtures)
    _phase(spans, "fixture_load", started, origin, len(full_requests))

    binding = upstream["binding_identity"]
    extension = Path(binding["module_file"])
    service_path = Path(binding["build"]["CARGO_TARGET_DIR"]) / (
        "release/examples/experiment_7326_constraint_kernel"
    )
    print("[exp7340] phase=benchmark event=before", flush=True)
    started = time.monotonic()
    rows, setup_costs, calibration = benchmark_complete_boundary(
        root, extension, service_path, full_requests, constraints
    )
    row_path = raw / "cost_rows.jsonl"
    exp7339._atomic_jsonl(row_path, rows)
    artifact["source_artifact_hashes"]["cost_rows"] = sha256_file(row_path)
    _phase(spans, "benchmark", started, origin, len(rows))
    print("[exp7340] phase=benchmark event=after", flush=True)

    print("[exp7340] phase=independent_reduction event=start", flush=True)
    started = time.monotonic()
    reloaded_rows = exp7339.read_jsonl(row_path)
    reduced = reduce_cost_rows(reloaded_rows)
    adverse = _adverse_checks(reloaded_rows)
    reduction = {
        "raw_sha256_matched": sha256_file(row_path)
        == artifact["source_artifact_hashes"]["cost_rows"],
        "reduced": reduced,
        "adverse_checks": adverse,
        "calibration": calibration,
    }
    reduction_path = raw / "independent_reduction.json"
    exp7339._atomic_json(reduction_path, reduction)
    artifact["source_artifact_hashes"]["independent_reduction"] = sha256_file(reduction_path)
    reduction_passed = reduction["raw_sha256_matched"] is True and adverse["passed"] is True
    artifact["validation_receipts"].append(
        _validation_receipt(
            "independent_raw_reduction",
            "reload cost_rows.jsonl; reduce; mutate row hash and stage sum",
            "task-owned measured raw rows",
            time.monotonic() - started,
            reduction_path,
            reduction_passed,
        )
    )
    _phase(spans, "independent_reduction", started, origin, 3)

    e2e = e2e_003_identity(extension, binding)
    e2e_path = raw / "e2e_003.json"
    exp7339._atomic_json(e2e_path, e2e)
    artifact["validation_receipts"].append(
        _validation_receipt(
            "e2e_003_schedule_roundtrip",
            "benchmarked loaded extension identity plus all-block parity",
            "schedule input -> Rust energy -> detached Python output",
            0.0,
            e2e_path,
            e2e["passed"] is True and reduced["parity_mismatches"] == 0,
        )
    )

    print("[exp7340] phase=affected_validation event=before", flush=True)
    started = time.monotonic()
    validation = run_scoped_validation(
        root,
        ("tests/python/test_experiment_7340_v644_native_cost.py",),
        ("python/carnot/experiment_7340_v644_native_cost.py",),
        static_paths=("scripts/experiments/experiment_7340_v644_native_cost.py",),
        basetemp=prepare_scoped_basetemp(Path("/tmp/carnot-exp7340-scoped")),
        coverage_file=raw / ".coverage",
        log_dir=raw / "validation/scoped",
    )
    artifact["validation_receipts"].extend(validation["validation_receipts"])
    artifact.update(
        {
            name: validation[name]
            for name in (
                "required_checks_passed",
                "missing_required_commands",
                "failed_required_commands",
                "duplicate_required_commands",
            )
        }
    )
    artifact["repository_health"] = validation["repository_health"]
    _phase(spans, "affected_validation", started, origin, len(validation["validation_receipts"]))
    print("[exp7340] phase=affected_validation event=after", flush=True)

    _complete_artifact(
        artifact,
        rows,
        reduced,
        {
            "extension_import_ns": setup_costs["extension_import_ns"],
            "constraint_compilation_ns": setup_costs["constraint_compilation_ns"],
        },
        affected_validation_passed=validation["required_checks_passed"] is True,
        e2e_passed=e2e["passed"] is True and reduced["parity_mismatches"] == 0,
        adverse_passed=reduction_passed,
    )
    artifact["benchmark_environment"] = {
        key: value
        for key, value in setup_costs.items()
        if key not in {"extension_import_ns", "constraint_compilation_ns"}
    }
    artifact["development_calibration"] = calibration
    artifact["phase_spans"] = spans
    artifact["duration_s"] = time.monotonic() - origin
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:  # pragma: no cover
    """Parse the fixed execution date, cold validation, and worker modes."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", choices=[RUN_DATE])
    parser.add_argument("--validate", type=Path)
    parser.add_argument("--serve-arm", choices=["python_in_process", "rust_in_process"])
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - command-line integration.
    """Run the experiment and publish only validated terminal evidence."""

    args = _parse_args(argv)
    if args.serve_arm is not None:
        return _serve_arm(args.serve_arm)
    if args.validate is not None:
        errors = validate_artifact(_read_object(args.validate))
        print(json.dumps({"errors": errors}, sort_keys=True), flush=True)
        return int(bool(errors))
    if args.date != RUN_DATE:
        raise SystemExit("--date 20260916 is required")
    print("[exp7340] phase=startup event=start", flush=True)
    artifact = run_experiment(REPO_ROOT)
    if artifact["verdict_class"] == "blocked":
        write_artifact(REPO_ROOT / RESULT_RELATIVE_PATH, artifact)
        print(
            f"[exp7340] phase=terminal_write event=end path={REPO_ROOT / RESULT_RELATIVE_PATH}",
            flush=True,
        )
        return 0

    raw = REPO_ROOT / RAW_RELATIVE_PATH
    candidate = raw / "terminal_candidate.json"
    exp7339._atomic_json(candidate, artifact)
    print("[exp7340] phase=terminal_validators event=before", flush=True)
    started = time.monotonic()
    validators = _terminal_validators(REPO_ROOT, candidate, raw)
    validator_duration = time.monotonic() - started
    artifact["validation_receipts"].extend(validators)
    validators_passed = all(row["passed"] for row in validators)
    artifact["acceptance_gate_results"]["terminal_validators"] = _gate(
        "both terminal validators pass",
        {
            "passed": [row["name"] for row in validators if row["passed"]],
            "failed": [row["name"] for row in validators if not row["passed"]],
        },
        validators_passed,
        "Adversarial and strict row-consistency checks precede the atomic terminal write.",
    )
    if not validators_passed:
        artifact.update(
            {
                "native_ten_x_score": 0,
                "readiness_score": 0,
                "value_score": 0,
                "promotion_score": 0,
                "verdict_class": "disqualified",
                "honest_verdict": "complete_disqualified: terminal artifact validation failed",
            }
        )
    offset = float(artifact["duration_s"])
    artifact["phase_spans"].append(
        {
            "phase": "terminal_validators",
            "start_s": offset,
            "end_s": offset + validator_duration,
            "completed_units": len(validators),
            "checkpoint_positions": [len(validators)],
            "pending_operations": [],
        }
    )
    artifact["duration_s"] = offset + validator_duration
    artifact["completed_at_utc"] = datetime.now(UTC).isoformat()
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    write_artifact(REPO_ROOT / RESULT_RELATIVE_PATH, artifact)
    print(f"[exp7340] phase=terminal_validators event=after passed={validators_passed}", flush=True)
    print(
        f"[exp7340] phase=terminal_write event=end path={REPO_ROOT / RESULT_RELATIVE_PATH} "
        f"verdict={artifact['verdict_class']}",
        flush=True,
    )
    return int(artifact["verdict_class"] == "disqualified")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
