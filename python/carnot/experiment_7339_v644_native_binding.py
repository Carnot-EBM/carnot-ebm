"""Prototype an in-process binding for retained acquired constraints.

The experiment reuses the exact V643 Python and Rust schedule evaluators. It
changes only the representation boundary: immutable terms are copied once into
Rust, and Python request objects cross PyO3 in ordered batches. The study makes
no learning, speed, hardware, deployment, or production-default claim.

Spec refs: REQ-VERIFY-7339, SCENARIO-VERIFY-7339-*, REQ-PYBIND-7339,
and SCENARIO-PYBIND-7339-*.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from copy import deepcopy
from datetime import UTC, datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import shutil
import statistics
import sys
import sysconfig
import tempfile
import time
from types import ModuleType
from typing import Any

from carnot import experiment_7326_v643_constraint_kernel as exp7326
from carnot.reporting.experiment_7303_validation_scope import (
    CommandSpec,
    REQUIRED_CHECK_NAMES,
    run_commands,
    run_scoped_validation,
)


JsonDict = dict[str, Any]

EXPERIMENT_ID = 7339
MILESTONE = "2026.09.644"
RUN_DATE = "20260916"
SCHEMA = "carnot.experiment_7339.v644_native_binding.v1"
REPO_ROOT = Path(__file__).resolve().parents[2]

EXP7325_PATH = Path("results/experiment_7325_v643_addition_audit.json")
EXP7326_PATH = Path("results/experiment_7326_v643_constraint_kernel.json")
FIXTURE_PATH = Path("results/raw/experiment_7326_v643_constraint_kernel/parity_fixtures.jsonl")
HISTORICAL_PARITY_PATH = Path(
    "results/raw/experiment_7326_v643_constraint_kernel/parity_rows.jsonl"
)
RESULT_PATH = Path("results/experiment_7339_v644_native_binding.json")
RAW_PATH = Path("results/raw/experiment_7339_v644_native_binding")
TARGET_PATH = RAW_PATH / "target"
EXTENSION_PATH = RAW_PATH / "extension"
MODULE_PATH = Path("python/carnot/experiment_7339_v644_native_binding.py")
TEST_PATH = Path("tests/python/test_experiment_7339_v644_native_binding.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_7339_v644_native_binding.py")
CONSTRAINT_SPEC_PATH = Path("openspec/capabilities/constraint-verification/spec.md")
BINDING_SPEC_PATH = Path("openspec/capabilities/python-bindings/spec.md")
RUST_SCHEDULE_PATH = Path("crates/carnot-constraints/src/schedule.rs")
RUST_BINDING_PATH = Path("crates/carnot-python/src/schedule.rs")
RUST_LIB_PATH = Path("crates/carnot-python/src/lib.rs")

EXPECTED_EXP7325_SHA256 = "sha256:9921cff7de490b9926b18fdac9162ed8b6fe7b180e93d6fdb31ff7437e633952"
EXPECTED_EXP7326_SHA256 = "sha256:915bf17631dae6a0711a1f8d3b45e487ec35dfb739d6b8aba8e01da068770ba6"
EXPECTED_FIXTURE_SHA256 = "sha256:218f14d8d6aa8a65ee712cfa87c2942412fd879389ea073fadf1ad09a0fd6880"
EXPECTED_PARITY_SHA256 = "sha256:b2e409a69fd60c3f70bb3331ce70acc392141c26b077eb9288523b4f425fa688"
HISTORICAL_FIXTURE_COUNT = 3304
ADVERSE_FIXTURE_COUNT = 4
MUTATION_CASE_COUNT = 2
PARITY_ROW_COUNT = HISTORICAL_FIXTURE_COUNT + ADVERSE_FIXTURE_COUNT + MUTATION_CASE_COUNT
BATCH_SIZES = (1, 32, 256)
PAIRED_BLOCKS = 30
ARMS = ("python_in_process", "rust_in_process", "historical_json_service")
DEVELOPMENT_SEED = 7339001
EVALUATION_SEED = 7339002
RESAMPLING_SEED = 7339003

INVOCATION_COUNTS = {
    "loads": {"attempted": 0, "completed": 0, "failed": 0, "cancelled": 0, "in_flight": 0},
    "generations": {
        "attempted": 0,
        "completed": 0,
        "failed": 0,
        "cancelled": 0,
        "in_flight": 0,
    },
}

COST_PROTOCOL: JsonDict = {
    "schema": "carnot.exp7340.constraint_cost_protocol.v1",
    "sealed_before_timing": True,
    "batch_sizes": list(BATCH_SIZES),
    "randomized_paired_blocks_each": PAIRED_BLOCKS,
    "arms": list(ARMS),
    "warmup_calls_per_size_and_arm": 3,
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
    """Encode the stable JSON form used only for evidence hashes."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def sha256_json(value: Any) -> str:
    """Hash one canonical JSON value with an explicit algorithm prefix."""

    return "sha256:" + hashlib.sha256(canonical_bytes(value)).hexdigest()


def sha256_file(path: Path) -> str:
    """Hash exact file bytes without loading a large sidecar at once."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def read_object(path: Path) -> JsonDict:
    """Load one JSON object and reject another top-level shape."""

    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"not_json_object:{path}")
    return value


def read_jsonl(path: Path) -> list[JsonDict]:
    """Load ordered JSON object rows from authenticated evidence."""

    rows: list[JsonDict] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"invalid_jsonl_row:{path}:{line_number}")
            rows.append(value)
    return rows


def check(
    upstream: str,
    name: str,
    field: str,
    expected: Any,
    observed: Any,
    passed: bool,
    principle: str,
) -> JsonDict:
    """Keep one exact precondition comparison for blocked output."""

    return {
        "upstream": upstream,
        "check": name,
        "field": field,
        "expected_value": expected,
        "observed_value": observed,
        "passed": passed,
        "principle": principle,
    }


def gate_check_summary(checks: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Reduce checks without losing the first exact failure."""

    failed = [dict(row) for row in checks if row.get("passed") is not True]
    return {
        "passed": not failed,
        "check_count": len(checks),
        "failed_check_count": len(failed),
        "first_failure": failed[0] if failed else None,
        "checks": [dict(row) for row in checks],
    }


def collect_preconditions(root: Path) -> tuple[list[JsonDict], JsonDict]:
    """Authenticate retained learning, kernel, fixture, and parity evidence."""

    audit_path = root / EXP7325_PATH
    kernel_path = root / EXP7326_PATH
    fixture_path = root / FIXTURE_PATH
    parity_path = root / HISTORICAL_PARITY_PATH
    audit_available = audit_path.is_file()
    kernel_available = kernel_path.is_file()
    fixture_available = fixture_path.is_file()
    parity_available = parity_path.is_file()
    audit = read_object(audit_path) if audit_available else {}
    kernel = read_object(kernel_path) if kernel_available else {}
    fixture_hash = sha256_file(fixture_path) if fixture_available else None
    parity_hash = sha256_file(parity_path) if parity_available else None
    fixture_count = sum(1 for _ in fixture_path.open("rb")) if fixture_available else None
    historical_parity = read_jsonl(parity_path) if parity_available else []
    mismatch_count = (
        sum(row.get("matched") is not True for row in historical_parity)
        if parity_available
        else None
    )
    audit_name = str(audit_path)
    kernel_name = str(kernel_path)
    checks = [
        check(
            audit_name,
            "exp7325_available",
            "path",
            True,
            audit_available,
            audit_available,
            "The retained learning audit must exist before its score is consumed.",
        ),
        check(
            audit_name,
            "exp7325_hash",
            "sha256",
            EXPECTED_EXP7325_SHA256,
            sha256_file(audit_path) if audit_available else None,
            audit_available and sha256_file(audit_path) == EXPECTED_EXP7325_SHA256,
            "The exact retained audit owns the learned-workload identity.",
        ),
        check(
            audit_name,
            "exp7325_terminal",
            "status",
            "complete",
            audit.get("status"),
            audit.get("status") == "complete",
            "A nonterminal producer cannot authorize input reuse.",
        ),
        check(
            audit_name,
            "exp7325_eligible_class",
            "verdict_class",
            "not blocked, disqualified, or partial",
            audit.get("verdict_class"),
            audit.get("verdict_class") not in {"blocked", "disqualified", "partial", None},
            "An ineligible class overrides a success-shaped score.",
        ),
        check(
            audit_name,
            "exp7325_not_quarantined",
            "flagged_adversarial",
            False,
            audit.get("flagged_adversarial"),
            audit.get("flagged_adversarial") is not True,
            "Quarantined evidence cannot supply the retained workload.",
        ),
        check(
            audit_name,
            "exp7325_promoted",
            "addition_promotion_score",
            1,
            audit.get("addition_promotion_score"),
            audit.get("addition_promotion_score") == 1,
            "The prototype reuses only the retained useful constraint workload.",
        ),
        check(
            kernel_name,
            "exp7326_available",
            "path",
            True,
            kernel_available,
            kernel_available,
            "The exact parity producer must remain available.",
        ),
        check(
            kernel_name,
            "exp7326_hash",
            "sha256",
            EXPECTED_EXP7326_SHA256,
            sha256_file(kernel_path) if kernel_available else None,
            kernel_available and sha256_file(kernel_path) == EXPECTED_EXP7326_SHA256,
            "The native prototype must not substitute a similarly named kernel result.",
        ),
        check(
            kernel_name,
            "exp7326_terminal",
            "status",
            "complete",
            kernel.get("status"),
            kernel.get("status") == "complete",
            "Only terminal historical evidence can supply fixtures.",
        ),
        check(
            kernel_name,
            "exp7326_eligible_class",
            "verdict_class",
            "not blocked, disqualified, or partial",
            kernel.get("verdict_class"),
            kernel.get("verdict_class") not in {"blocked", "disqualified", "partial", None},
            "A historical service null remains eligible because this boundary is different.",
        ),
        check(
            kernel_name,
            "exp7326_complete",
            "constraint_kernel_complete_score",
            1,
            kernel.get("constraint_kernel_complete_score"),
            kernel.get("constraint_kernel_complete_score") == 1,
            "The reused evaluator must have completed its original parity protocol.",
        ),
        check(
            str(fixture_path),
            "parity_fixtures_available",
            "path",
            True,
            fixture_available,
            fixture_available,
            "The task replays captured bytes instead of regenerating history.",
        ),
        check(
            str(fixture_path),
            "parity_fixtures_hash",
            "sha256",
            EXPECTED_FIXTURE_SHA256,
            fixture_hash,
            fixture_hash == EXPECTED_FIXTURE_SHA256,
            "The exact 3,304 requests define the replay denominator.",
        ),
        check(
            str(fixture_path),
            "parity_fixture_count",
            "row_count",
            HISTORICAL_FIXTURE_COUNT,
            fixture_count,
            fixture_count == HISTORICAL_FIXTURE_COUNT,
            "Every historical fixture stays in the native parity panel.",
        ),
        check(
            str(parity_path),
            "historical_parity_available",
            "path",
            True,
            parity_available,
            parity_available,
            "The original Python-Rust result rows authenticate the fixture claim.",
        ),
        check(
            str(parity_path),
            "historical_parity_hash",
            "sha256",
            EXPECTED_PARITY_SHA256,
            parity_hash,
            parity_hash == EXPECTED_PARITY_SHA256,
            "Fresh native parity remains linked to the exact old result rows.",
        ),
        check(
            str(parity_path),
            "historical_parity_result",
            "rows,mismatches",
            {"rows": HISTORICAL_FIXTURE_COUNT, "mismatches": 0},
            {"rows": len(historical_parity), "mismatches": mismatch_count},
            len(historical_parity) == HISTORICAL_FIXTURE_COUNT and mismatch_count == 0,
            "A partial or mismatched historical producer cannot seed readiness.",
        ),
    ]
    hashes = {
        "experiment_7325": sha256_file(audit_path) if audit_available else None,
        "experiment_7326": sha256_file(kernel_path) if kernel_available else None,
        "parity_fixtures": fixture_hash,
        "historical_parity_rows": parity_hash,
    }
    return checks, hashes


def compact_request(request: Mapping[str, Any]) -> JsonDict:
    """Detach one request and remove terms owned by the compiled object."""

    return {key: deepcopy(value) for key, value in request.items() if key != "constraints"}


def group_fixtures(fixtures: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Group requests by the exact ordered terms compiled into one native object."""

    groups: dict[str, JsonDict] = {}
    for fixture in fixtures:
        request = fixture["request"]
        constraints = deepcopy(request.get("constraints")) if isinstance(request, Mapping) else None
        key = sha256_json(constraints)
        if key not in groups:
            groups[key] = {"constraint_sha256": key, "constraints": constraints, "fixtures": []}
        groups[key]["fixtures"].append(fixture)
    return list(groups.values())


def _base_overflow_request() -> JsonDict:
    """Return a small request that adverse cases can change independently."""

    return {
        "schema": exp7326.REQUEST_SCHEMA,
        "executor_version": "adverse-v1",
        "slot_min": 0,
        "slot_max": 0,
        "schedule": [{"activity": "a", "slot": 0}, {"activity": "b", "slot": 0}],
        "constraints": [],
    }


def adverse_fixtures() -> list[JsonDict]:
    """Create fixed overflow and integer-type cases absent from the old panel."""

    term_overflow = _base_overflow_request()
    term_overflow["constraints"] = [
        {
            "kind": "pairwise_separation",
            "constraint_id": "overflow-sep",
            "version": "adverse-v1",
            "left": "a",
            "right": "b",
            "minimum": exp7326.I64_MAX,
        }
    ]
    total_overflow = _base_overflow_request()
    total_overflow["constraints"] = [
        {
            "kind": "pairwise_separation",
            "constraint_id": f"overflow-total-{suffix}",
            "version": "adverse-v1",
            "left": "a",
            "right": "b",
            "minimum": 4_294_967_295,
        }
        for suffix in ("a", "b")
    ]
    boolean_slot = _base_overflow_request()
    boolean_slot["schedule"][0]["slot"] = True
    wide_slot = _base_overflow_request()
    wide_slot["schedule"][0]["slot"] = exp7326.I64_MAX + 1
    cases = (
        ("term_energy_overflow", term_overflow),
        ("total_energy_overflow", total_overflow),
        ("boolean_is_not_integer", boolean_slot),
        ("integer_outside_i64", wide_slot),
    )
    return [
        {
            "fixture_id": f"adverse-{index:02d}",
            "source": "exp7339_adverse",
            "case": case,
            "request": request,
        }
        for index, (case, request) in enumerate(cases)
    ]


def native_parity(
    binding: Any,
    fixtures: Sequence[Mapping[str, Any]],
    *,
    progress_every: int = 256,
) -> list[JsonDict]:
    """Replay grouped requests through Python and the compiled native object."""

    rows: list[JsonDict] = []
    next_report = max(1, progress_every)
    started = time.monotonic()
    for group in group_fixtures(fixtures):
        evaluator = binding.RustCompiledScheduleEvaluator(deepcopy(group["constraints"]))
        group_rows = group["fixtures"]
        compact = [compact_request(row["request"]) for row in group_rows]
        rust_results = [dict(row) for row in evaluator.evaluate_batch(compact)]
        python_results = [exp7326.evaluate_request(row["request"]) for row in group_rows]
        for fixture, python_result, rust_result in zip(
            group_rows, python_results, rust_results, strict=True
        ):
            rows.append(
                {
                    "fixture_id": fixture["fixture_id"],
                    "source": fixture["source"],
                    "case": fixture["case"],
                    "constraint_sha256": group["constraint_sha256"],
                    "python": python_result,
                    "rust": rust_result,
                    "matched": python_result == rust_result,
                }
            )
        if len(rows) >= next_report or len(rows) == len(fixtures):
            print(
                f"[exp7339] phase=parity event=unit_complete completed={len(rows)}/"
                f"{len(fixtures)} elapsed_s={time.monotonic() - started:.3f}",
                flush=True,
            )
            while next_report <= len(rows):
                next_report += max(1, progress_every)
    return rows


def _mutation_request() -> JsonDict:
    """Return one nonzero request used to expose accidental aliasing."""

    return {
        "schema": exp7326.REQUEST_SCHEMA,
        "executor_version": "mutation-v1",
        "slot_min": 0,
        "slot_max": 2,
        "schedule": [{"activity": "a", "slot": 0}, {"activity": "b", "slot": 1}],
        "constraints": [
            {
                "kind": "pairwise_separation",
                "constraint_id": "mutation-sep",
                "version": "mutation-v1",
                "left": "a",
                "right": "b",
                "minimum": 2,
            }
        ],
    }


def mutation_parity(binding: Any) -> list[JsonDict]:
    """Prove constructor inputs and returned containers do not alias Rust state."""

    request = _mutation_request()
    expected = exp7326.evaluate_request(request)
    constraints = deepcopy(request["constraints"])
    evaluator = binding.RustCompiledScheduleEvaluator(constraints)
    constraints[0]["minimum"] = 0
    after_input_mutation = dict(evaluator.evaluate_batch([compact_request(request)])[0])

    returned = evaluator.evaluate_batch([compact_request(request)])[0]
    returned["terms"][0]["energy"] = 999
    returned["total_energy"] = 999
    after_result_mutation = dict(evaluator.evaluate_batch([compact_request(request)])[0])
    return [
        {
            "fixture_id": "mutation-constructor-input",
            "source": "exp7339_mutation",
            "case": "constructor_input_mutation",
            "python": deepcopy(expected),
            "rust": after_input_mutation,
            "matched": expected == after_input_mutation,
        },
        {
            "fixture_id": "mutation-returned-result",
            "source": "exp7339_mutation",
            "case": "returned_result_mutation",
            "python": deepcopy(expected),
            "rust": after_result_mutation,
            "matched": expected == after_result_mutation,
        },
    ]


def reduce_parity_rows(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Independently count complete exact parity rows and differences."""

    mismatches = sum(row.get("matched") is not True for row in rows)
    return {"rows": len(rows), "mismatches": mismatches, "all_matched": mismatches == 0}


def synthetic_cost_rows() -> list[JsonDict]:
    """Build a complete tiny-duration protocol fixture for reducer tests."""

    rows: list[JsonDict] = []
    for size in BATCH_SIZES:
        for block in range(PAIRED_BLOCKS):
            order = list(ARMS[block % len(ARMS) :] + ARMS[: block % len(ARMS)])
            for arm_order, arm in enumerate(order):
                rows.append(
                    {
                        "unit_id": f"size-{size}-block-{block:02d}",
                        "batch_size": size,
                        "block": block,
                        "arm": arm,
                        "arm_order": arm_order,
                        "elapsed_ns": 100 + arm_order,
                        "requests_per_second": size * 1_000_000_000 / (100 + arm_order),
                        "failures": 0,
                        "abstentions": 0,
                        "censored": False,
                        "output_sha256": "sha256:fixture",
                    }
                )
    return rows


def reduce_cost_rows(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Require every fixed block, arm, output match, and uncensored cost."""

    cells: dict[tuple[int, int], list[Mapping[str, Any]]] = {}
    for row in rows:
        cells.setdefault((int(row["batch_size"]), int(row["block"])), []).append(row)
    complete_blocks = 0
    mismatches = 0
    by_size: JsonDict = {}
    for size in BATCH_SIZES:
        latencies = {arm: [] for arm in ARMS}
        size_complete = 0
        for block in range(PAIRED_BLOCKS):
            block_rows = cells.get((size, block), [])
            arms = {str(row.get("arm")) for row in block_rows}
            hashes = {str(row.get("output_sha256")) for row in block_rows}
            valid = (
                len(block_rows) == len(ARMS)
                and arms == set(ARMS)
                and len(hashes) == 1
                and all(row.get("failures") == 0 for row in block_rows)
                and all(row.get("censored") is False for row in block_rows)
            )
            if valid:
                size_complete += 1
                complete_blocks += 1
            else:
                mismatches += int(len(hashes) > 1)
            for row in block_rows:
                arm = str(row.get("arm"))
                if arm in latencies and isinstance(row.get("elapsed_ns"), int):
                    latencies[arm].append(int(row["elapsed_ns"]))
        by_size[str(size)] = {
            "complete_blocks": size_complete,
            "p50_elapsed_ns": {
                arm: statistics.median(values) if values else None
                for arm, values in latencies.items()
            },
        }
    expected_rows = len(BATCH_SIZES) * PAIRED_BLOCKS * len(ARMS)
    return {
        "row_count": len(rows),
        "expected_row_count": expected_rows,
        "paired_blocks": complete_blocks,
        "expected_paired_blocks": len(BATCH_SIZES) * PAIRED_BLOCKS,
        "mismatches": mismatches,
        "complete": len(rows) == expected_rows
        and complete_blocks == len(BATCH_SIZES) * PAIRED_BLOCKS
        and mismatches == 0,
        "by_batch_size": by_size,
    }


def _gate(expected: Any, observed: Any, passed: bool, principle: str) -> JsonDict:
    """Keep expected and observed values beside one acceptance decision."""

    return {"expected": expected, "observed": observed, "passed": passed, "principle": principle}


def _field_principles() -> JsonDict:
    """Explain required fields without wrapping executable values."""

    return {
        "schema": "Version the artifact while preserving ordinary top-level experiment_id and milestone.",
        "status": "Publish a terminal result only after current work and affected validation.",
        "run_date": "Use 20260916 and retain actual UTC timestamps.",
        "preconditions_checked": "Name input identity, availability, and each failed check before work.",
        "MODEL_SPECS": "List current executable model identities; this task has none.",
        "model_invoked": "Set true for any attempted real model load or generation.",
        "invocation_counts": "Separate attempted, completed, failed, cancelled, and in-flight work.",
        "inference_substrate": "Declare the actual CPU exact computation used here.",
        "inference_substrate_class": "Declare the real duration class instead of a model class.",
        "execution_venue": "Use host because no board or remote venue executes this task.",
        "duration_s": "Measure real monotonic elapsed time without padding.",
        "phase_spans": "Keep disjoint stages, units, checkpoints, and pending operations.",
        "random_seed": "Freeze development, evaluation, and resampling seeds before outcomes.",
        "reproducibility_checksum": "Bind code, settings, inputs, evaluator identity, and raw evidence.",
        "source_artifact_hashes": "Authenticate exact producers and current source identities.",
        "rows": "Every timed arm records metrics, costs, failures, abstentions, and censoring.",
        "sample_size_budget": "Record planned, attempted, completed, censored, and the fixed stop rule.",
        "acceptance_gate_results": "Each gate records expected, observed, passed, and its principle.",
        "gate_check_summary": "Every blocked result names the failed upstream field and values.",
        "verifier_is_oracle": "Shared execution authority makes correctness evidence circular.",
        "honest_verdict": "Completed findings start complete_; external absence starts blocked_.",
        "verdict_class": "Use only positive, circular_positive, null, blocked, disqualified, or partial.",
        "validation_receipts": "Record exact command, scope, exit, duration, and log hash, including failures.",
        "repository_health": "Keep unrelated dated failures separate from affected required checks.",
        "field_principles": "Explain every required field without hiding executable values.",
        "native_binding_ready_score": "One requires the imported extension, exact parity, and complete protocol.",
        "parity_rows": "Every captured and adverse case binds Python and Rust outputs.",
        "binding_identity": "Interpreter, extension, ABI, binary, and source hashes prevent stale claims.",
        "cost_protocol": "Freeze complete boundary costs and randomized sizes before timing.",
    }


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash deterministic claims while excluding wall clocks and command durations."""

    keys = (
        "schema",
        "experiment_id",
        "milestone",
        "run_date",
        "preconditions_checked",
        "MODEL_SPECS",
        "model_invoked",
        "invocation_counts",
        "inference_substrate",
        "inference_substrate_class",
        "execution_venue",
        "random_seed",
        "source_artifact_hashes",
        "rows",
        "parity_rows",
        "sample_size_budget",
        "acceptance_gate_results",
        "verifier_is_oracle",
        "native_binding_ready_score",
        "binding_identity",
        "cost_protocol",
    )
    return sha256_json({key: artifact.get(key) for key in keys})


def _base_artifact(checks: Sequence[Mapping[str, Any]], hashes: Mapping[str, Any]) -> JsonDict:
    """Create a schema-complete candidate before runtime scoring."""

    return {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "phase": 4,
        "status": "candidate",
        "run_date": RUN_DATE,
        "started_at_utc": datetime.now(UTC).isoformat(),
        "completed_at_utc": None,
        "preconditions_checked": [deepcopy(dict(row)) for row in checks],
        "gate_check_summary": gate_check_summary(checks),
        "MODEL_SPECS": [],
        "model_invoked": False,
        "invocation_counts": deepcopy(INVOCATION_COUNTS),
        "current_model_load_count": 0,
        "current_generation_count": 0,
        "inference_substrate": "cpu_exact_solver_or_simulator",
        "inference_substrate_class": "cpu_exact_solver_or_simulator",
        "execution_venue": "host",
        "duration_s": 0.0,
        "phase_spans": [],
        "random_seed": {
            "development": DEVELOPMENT_SEED,
            "evaluation": EVALUATION_SEED,
            "resampling": RESAMPLING_SEED,
            "sealed_before_results": True,
        },
        "reproducibility_checksum": "",
        "source_artifact_hashes": deepcopy(dict(hashes)),
        "rows": [],
        "parity_rows": [],
        "sample_size_budget": {},
        "acceptance_gate_results": {},
        "verifier_is_oracle": True,
        "honest_verdict": "partial: native binding work is not complete",
        "verdict_class": "partial",
        "validation_receipts": [],
        "required_checks_passed": False,
        "missing_required_commands": list(REQUIRED_CHECK_NAMES),
        "failed_required_commands": [],
        "duplicate_required_commands": [],
        "repository_health": {
            "status": "healthy",
            "incident_open": False,
            "historical_failures": [],
            "affects_required_checks": False,
        },
        "field_principles": _field_principles(),
        "native_binding_ready_score": 0,
        "binding_identity": {},
        "cost_protocol": deepcopy(COST_PROTOCOL),
        "setup_costs": {},
        "independent_raw_reduction": {},
        "production_default_changed": False,
        "publication_surface_changed": False,
        "deployment_changed": False,
        "hardware_commands_run": False,
        "speed_claimed": False,
        "learning_value_claimed": False,
    }


def _blocked_artifact(checks: Sequence[Mapping[str, Any]], hashes: Mapping[str, Any]) -> JsonDict:
    """Turn the first unchanged external failure into a terminal block."""

    artifact = _base_artifact(checks, hashes)
    failure = artifact["gate_check_summary"]["first_failure"]
    artifact.update(
        {
            "status": "blocked",
            "completed_at_utc": datetime.now(UTC).isoformat(),
            "honest_verdict": (
                f"blocked_upstream: {failure['upstream']} check {failure['check']} field "
                f"{failure['field']} expected {failure['expected_value']!r}; "
                f"observed {failure['observed_value']!r}"
            ),
            "verdict_class": "blocked",
        }
    )
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any], *, check_files: bool = False) -> list[str]:
    """Cold-check identity, evidence, gates, scores, and optional binary bytes."""

    errors: list[str] = []

    def add(condition: bool, name: str) -> None:
        if condition:
            errors.append(name)

    required = set(_field_principles()) | {"experiment_id", "milestone"}
    add(not required.issubset(artifact), "required_fields")
    add(artifact.get("schema") != SCHEMA, "schema")
    add(artifact.get("experiment_id") != EXPERIMENT_ID, "experiment_id")
    add(artifact.get("milestone") != MILESTONE, "milestone")
    add(artifact.get("run_date") != RUN_DATE, "run_date")
    add(artifact.get("MODEL_SPECS") != [], "MODEL_SPECS")
    add(artifact.get("model_invoked") is not False, "model_invoked")
    add(artifact.get("invocation_counts") != INVOCATION_COUNTS, "invocation_counts")
    add(
        artifact.get("inference_substrate") != "cpu_exact_solver_or_simulator",
        "inference_substrate",
    )
    add(
        artifact.get("inference_substrate_class") != "cpu_exact_solver_or_simulator",
        "inference_substrate_class",
    )
    add(artifact.get("execution_venue") != "host", "execution_venue")
    add(artifact.get("verifier_is_oracle") is not True, "verifier_is_oracle")
    add(artifact.get("cost_protocol") != COST_PROTOCOL, "cost_protocol")
    add(
        any(
            artifact.get(key) is not False
            for key in (
                "production_default_changed",
                "publication_surface_changed",
                "deployment_changed",
                "hardware_commands_run",
                "speed_claimed",
                "learning_value_claimed",
            )
        ),
        "claim_safety",
    )
    status = artifact.get("status")
    add(status not in {"complete", "blocked"}, "status")
    ready = artifact.get("native_binding_ready_score")
    add(ready not in {0, 1}, "native_binding_ready_score")
    if status == "blocked":
        add(bool(artifact.get("rows")), "blocked_rows")
        add(bool(artifact.get("parity_rows")), "blocked_parity_rows")
        add(ready != 0, "blocked_score")
        add(artifact.get("verdict_class") != "blocked", "blocked_verdict_class")
        add(not str(artifact.get("honest_verdict", "")).startswith("blocked_"), "blocked_prefix")
    else:
        parity = reduce_parity_rows(artifact.get("parity_rows", []))
        costs = reduce_cost_rows(artifact.get("rows", []))
        gates = artifact.get("acceptance_gate_results", {})
        required_pass = (
            parity == {"rows": PARITY_ROW_COUNT, "mismatches": 0, "all_matched": True}
            and costs.get("complete") is True
            and all(
                gates.get(name, {}).get("passed") is True
                for name in (
                    "parity",
                    "mutation",
                    "e2e_003",
                    "cost_protocol",
                    "affected_validation",
                )
            )
            and gates.get("terminal_validators", {"passed": True}).get("passed") is True
        )
        add(ready != int(required_pass), "native_binding_ready_score")
        if ready == 1:
            identity = artifact.get("binding_identity", {})
            add(identity.get("compiled_execution") is not True, "binding_compiled_execution")
            add(identity.get("python_fallback_used") is not False, "binding_python_fallback")
            add(artifact.get("verdict_class") != "circular_positive", "ready_verdict_class")
        add(len(artifact.get("parity_rows", [])) != PARITY_ROW_COUNT, "parity_row_count")
        add(len(artifact.get("rows", [])) != 270, "cost_row_count")
        add(
            not str(artifact.get("honest_verdict", "")).startswith("complete"),
            "complete_prefix",
        )
        budget = artifact.get("sample_size_budget", {})
        add(budget.get("parity_completed") != PARITY_ROW_COUNT, "parity_budget")
        add(budget.get("cost_rows_completed") != 270, "cost_budget")
    if check_files and status == "complete":
        identity = artifact.get("binding_identity", {})
        module_value = identity.get("module_file")
        module = Path(module_value) if isinstance(module_value, str) else Path()
        add(not module.is_file(), "binding_module_file")
        if module.is_file():
            add(sha256_file(module) != identity.get("module_sha256"), "binding_module_hash")
    try:
        checksum = reproducibility_checksum(artifact)
    except (TypeError, ValueError):
        checksum = None
    add(artifact.get("reproducibility_checksum") != checksum, "reproducibility_checksum")
    return sorted(set(errors))


def _atomic_bytes(path: Path, data: bytes) -> None:
    """Publish bytes with one same-directory replace and directory sync."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(path: Path, value: Any) -> None:
    """Atomically publish one canonical JSON document."""

    _atomic_bytes(path, canonical_bytes(value) + b"\n")


def _atomic_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Atomically publish ordered canonical JSON evidence rows."""

    _atomic_bytes(path, b"".join(canonical_bytes(row) + b"\n" for row in rows))


def write_artifact(
    path: Path, artifact: Mapping[str, Any], *, check_files: bool = True
) -> JsonDict:
    """Cold-validate and atomically publish one terminal artifact."""

    errors = validate_artifact(artifact, check_files=check_files)
    if errors:
        raise ValueError("artifact_validation_failed:" + ",".join(errors))
    _atomic_json(path, artifact)
    return {"path": str(path), "sha256": sha256_file(path)}


def _phase(spans: list[JsonDict], name: str, started: float, origin: float, units: int) -> None:
    """Append one disjoint stage and emit its flushed end boundary."""

    ended = time.monotonic()
    spans.append(
        {
            "phase": name,
            "start_s": started - origin,
            "end_s": ended - origin,
            "completed_units": units,
            "checkpoint_positions": [units],
            "pending_operations": [],
        }
    )
    print(
        f"[exp7339] phase={name} event=end units={units} elapsed_s={ended - started:.3f}",
        flush=True,
    )


def _current_source_hashes(root: Path) -> JsonDict:
    """Bind Python, Rust, tests, specs, and the governing E2E plan."""

    paths = (
        MODULE_PATH,
        TEST_PATH,
        WRAPPER_PATH,
        CONSTRAINT_SPEC_PATH,
        BINDING_SPEC_PATH,
        RUST_SCHEDULE_PATH,
        RUST_BINDING_PATH,
        RUST_LIB_PATH,
        Path("python/carnot/experiment_7326_v643_constraint_kernel.py"),
        Path("python/carnot/reporting/experiment_7303_validation_scope.py"),
        Path("ops/e2e-test-plan.md"),
    )
    return {path.as_posix(): sha256_file(root / path) for path in paths}


def _extension_destination(root: Path) -> Path:
    """Choose a task-owned filename for this interpreter's exact suffix."""

    suffix = sysconfig.get_config_var("EXT_SUFFIX")
    if not isinstance(suffix, str) or not suffix:
        raise RuntimeError("Python extension suffix is unavailable")
    return root / EXTENSION_PATH / f"_rust{suffix}"


def _build_native(
    root: Path, raw: Path
) -> tuple[Path, Path, list[JsonDict], JsonDict]:  # pragma: no cover - real build boundary.
    """Build the extension and old service in one isolated task target."""

    target = root / TARGET_PATH
    commands = [
        CommandSpec(
            "native_extension_build",
            (
                "cargo",
                "build",
                "--release",
                "-p",
                "carnot-python",
                "--target-dir",
                str(target),
            ),
            "interpreter-bound carnot-python extension",
        ),
        CommandSpec(
            "historical_service_build",
            (
                "cargo",
                "build",
                "--release",
                "-p",
                "carnot-constraints",
                "--example",
                "experiment_7326_constraint_kernel",
                "--target-dir",
                str(target),
            ),
            "current historical JSON service mechanism",
        ),
    ]
    receipts = run_commands(
        root,
        commands,
        log_dir=raw / "validation/build",
        extra_env={"PYO3_PYTHON": str(Path(sys.executable).absolute())},
    )
    if not all(row["passed"] for row in receipts):
        raise RuntimeError("native_build_failed")
    library = target / "release/libcarnot_python.so"
    service = target / "release/examples/experiment_7326_constraint_kernel"
    if not library.is_file() or not service.is_file():
        raise RuntimeError("native_build_output_missing")
    destination = _extension_destination(root)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        shutil.copyfile(library, temporary)
        os.replace(temporary, destination)
        destination.chmod(0o755)
    finally:
        temporary.unlink(missing_ok=True)
    build_identity = {
        "build_command": receipts[0]["command"],
        "build_exit_code": receipts[0]["exit_code"],
        "build_log_sha256": receipts[0]["log_sha256"],
        "PYO3_PYTHON": str(Path(sys.executable).absolute()),
        "CARGO_TARGET_DIR": str(target.resolve()),
        "shared_environment_installed": False,
    }
    return destination, service, receipts, build_identity


def load_native_extension(
    extension: Path,
) -> ModuleType:  # pragma: no cover - real import boundary.
    """Load the exact task-owned binary and reject a stale entrypoint."""

    resolved = extension.resolve()
    sys.modules.pop("carnot._rust", None)
    spec = importlib.util.spec_from_file_location("carnot._rust", resolved)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot create native loader:{resolved}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    if not hasattr(module, "RustCompiledScheduleEvaluator"):
        raise RuntimeError("selected extension lacks RustCompiledScheduleEvaluator")
    return module


def _binding_identity(
    root: Path,
    extension: Path,
    binding: ModuleType,
    build: Mapping[str, Any],
) -> JsonDict:
    """Bind the interpreter, ABI, binary, build, and exact Rust sources."""

    return {
        "interpreter": str(Path(sys.executable).absolute()),
        "interpreter_resolved": str(Path(sys.executable).resolve()),
        "python_version": sys.version,
        "soabi": sysconfig.get_config_var("SOABI"),
        "extension_suffix": sysconfig.get_config_var("EXT_SUFFIX"),
        "module_file": str(Path(str(binding.__file__)).resolve()),
        "module_sha256": sha256_file(extension),
        "extension_abi": "pyo3-interpreter-specific",
        "native_class": "carnot._rust.RustCompiledScheduleEvaluator",
        "compiled_execution": True,
        "python_fallback_used": False,
        "build": deepcopy(dict(build)),
        "rust_source_identity": {
            RUST_SCHEDULE_PATH.as_posix(): sha256_file(root / RUST_SCHEDULE_PATH),
            RUST_BINDING_PATH.as_posix(): sha256_file(root / RUST_BINDING_PATH),
            RUST_LIB_PATH.as_posix(): sha256_file(root / RUST_LIB_PATH),
        },
    }


def _benchmark_requests(fixtures: Sequence[Mapping[str, Any]]) -> tuple[list[JsonDict], Any]:
    """Select one useful valid compiled-term family before any timing starts."""

    candidates: list[tuple[int, int, JsonDict]] = []
    for group in group_fixtures(fixtures):
        valid = [
            deepcopy(dict(row["request"]))
            for row in group["fixtures"]
            if exp7326.evaluate_request(row["request"])["valid_input"]
        ]
        if valid:
            candidates.append(
                (len(group["constraints"] or []), len(valid), {**group, "valid": valid})
            )
    if not candidates:
        raise RuntimeError("no_valid_benchmark_requests")
    selected = max(candidates, key=lambda row: (row[0], row[1]))[2]
    return selected["valid"], selected["constraints"]


def benchmark_costs(
    binding: Any,
    service: exp7326.JsonLineService,
    fixtures: Sequence[Mapping[str, Any]],
) -> tuple[list[JsonDict], JsonDict]:  # pragma: no cover - measured boundary.
    """Run the sealed three-arm protocol through fresh current mechanisms."""

    full_requests, constraints = _benchmark_requests(fixtures)
    compile_started = time.monotonic()
    evaluator = binding.RustCompiledScheduleEvaluator(deepcopy(constraints))
    compilation_s = time.monotonic() - compile_started
    rows: list[JsonDict] = []
    rng = random.Random(EVALUATION_SEED)
    benchmark_started = time.monotonic()
    for size in BATCH_SIZES:
        full = [deepcopy(full_requests[index % len(full_requests)]) for index in range(size)]
        compact = [compact_request(request) for request in full]
        message = {"operation": "evaluate", "requests": full}

        def call_arm(arm: str) -> list[JsonDict]:
            if arm == "python_in_process":
                return exp7326.evaluate_batch(full)
            if arm == "rust_in_process":
                return [dict(row) for row in evaluator.evaluate_batch(compact)]
            return list(service.call(message)["results"])

        for _ in range(int(COST_PROTOCOL["warmup_calls_per_size_and_arm"])):
            for arm in ARMS:
                call_arm(arm)
        for block in range(PAIRED_BLOCKS):
            order = list(ARMS)
            rng.shuffle(order)
            outputs: dict[str, list[JsonDict]] = {}
            elapsed: dict[str, int] = {}
            for arm in order:
                started = time.perf_counter_ns()
                outputs[arm] = call_arm(arm)
                elapsed[arm] = time.perf_counter_ns() - started
            control = outputs["python_in_process"]
            for arm_order, arm in enumerate(order):
                matched = outputs[arm] == control
                rows.append(
                    {
                        "unit_id": f"size-{size}-block-{block:02d}",
                        "batch_size": size,
                        "block": block,
                        "arm": arm,
                        "arm_order": arm_order,
                        "order": order,
                        "elapsed_ns": elapsed[arm],
                        "requests_per_second": size * 1_000_000_000 / elapsed[arm],
                        "failures": int(not matched),
                        "abstentions": 0,
                        "censored": False,
                        "output_sha256": sha256_json(outputs[arm]),
                    }
                )
            print(
                f"[exp7339] phase=benchmark event=unit_complete size={size} "
                f"block={block + 1}/{PAIRED_BLOCKS} "
                f"elapsed_s={time.monotonic() - benchmark_started:.3f}",
                flush=True,
            )
    return rows, {
        "constraint_compilation_s": compilation_s,
        "compiled_constraint_count": len(constraints),
        "benchmark_source_request_count": len(full_requests),
    }


def _validation_commands(
    root: Path, raw: Path
) -> tuple[JsonDict, list[JsonDict], list[JsonDict]]:  # pragma: no cover
    """Run the shipped scoped Python checks and focused Rust checks."""

    scoped = run_scoped_validation(
        root,
        [TEST_PATH.as_posix()],
        [MODULE_PATH.as_posix()],
        static_paths=[WRAPPER_PATH.as_posix()],
        basetemp=exp7326.prepare_scoped_basetemp(Path("/tmp/carnot-exp7339-scoped")),
        coverage_file=raw / ".coverage",
        log_dir=raw / "validation/scoped",
        extra_env={"CARNOT_EXP7339_EXTENSION": str(_extension_destination(root).resolve())},
    )
    target = root / TARGET_PATH
    rust_commands = [
        CommandSpec(
            "cargo_test_carnot_python_schedule",
            (
                "cargo",
                "test",
                "-p",
                "carnot-python",
                "--lib",
                "schedule",
                "--target-dir",
                str(target),
            ),
            "changed PyO3 schedule binding",
        ),
        CommandSpec(
            "cargo_test_carnot_constraints_schedule",
            (
                "cargo",
                "test",
                "-p",
                "carnot-constraints",
                "--test",
                "experiment_7326_constraint_kernel",
                "--target-dir",
                str(target),
            ),
            "reused Rust schedule evaluator",
        ),
        CommandSpec(
            "cargo_fmt_affected",
            ("cargo", "fmt", "-p", "carnot-python", "-p", "carnot-constraints", "--", "--check"),
            "affected Rust crates",
        ),
        CommandSpec(
            "cargo_clippy_strict_baseline_observation",
            (
                "cargo",
                "clippy",
                "-p",
                "carnot-python",
                "-p",
                "carnot-constraints",
                "--all-targets",
                "--no-deps",
                "--target-dir",
                str(target),
                "--",
                "-D",
                "warnings",
            ),
            "affected crates with unchanged crate-wide warning baseline",
        ),
        CommandSpec(
            "cargo_clippy_affected",
            (
                "cargo",
                "clippy",
                "-p",
                "carnot-python",
                "-p",
                "carnot-constraints",
                "--all-targets",
                "--no-deps",
                "--target-dir",
                str(target),
                "--",
                "-D",
                "warnings",
                "-A",
                "unused-imports",
                "-A",
                "deprecated",
                "-A",
                "clippy::too-many-arguments",
                "-A",
                "clippy::needless-range-loop",
            ),
            "affected crates excluding four named unchanged baseline lint classes",
        ),
    ]
    rust = run_commands(
        root,
        rust_commands,
        log_dir=raw / "validation/rust",
        extra_env={"PYO3_PYTHON": str(Path(sys.executable).absolute())},
    )
    baseline = [row for row in rust if row["name"] == "cargo_clippy_strict_baseline_observation"]
    required = [row for row in rust if row["name"] != "cargo_clippy_strict_baseline_observation"]
    return scoped, required, baseline


def _terminal_validators(
    root: Path, candidate: Path, raw: Path
) -> list[JsonDict]:  # pragma: no cover
    """Run both required artifact validators on the measured candidate."""

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


def _validation_receipt(
    name: str, command: str, scope: str, duration_s: float, path: Path, passed: bool
) -> JsonDict:
    """Record one in-process E2E or independent-reduction check."""

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


def run_experiment(root: Path) -> JsonDict:  # pragma: no cover - real end-to-end entrypoint.
    """Authenticate, build, import, replay, benchmark, validate, and reduce."""

    origin = time.monotonic()
    spans: list[JsonDict] = []
    raw = root / RAW_PATH
    raw.mkdir(parents=True, exist_ok=True)

    print("[exp7339] phase=preflight event=start", flush=True)
    started = time.monotonic()
    checks, hashes = collect_preconditions(root)
    hashes["current_sources"] = _current_source_hashes(root)
    artifact = _base_artifact(checks, hashes)
    if (root / EXP7326_PATH).is_file():
        old = read_object(root / EXP7326_PATH)
        if isinstance(old.get("repository_health"), Mapping):
            artifact["repository_health"] = deepcopy(old["repository_health"])
    _phase(spans, "preflight", started, origin, len(checks))
    if artifact["gate_check_summary"]["passed"] is not True:
        blocked = _blocked_artifact(checks, hashes)
        blocked["phase_spans"] = spans
        blocked["duration_s"] = time.monotonic() - origin
        blocked["reproducibility_checksum"] = reproducibility_checksum(blocked)
        return blocked

    print("[exp7339] phase=no_model_work event=start MODEL_SPECS=[]", flush=True)
    started = time.monotonic()
    _phase(spans, "no_model_work", started, origin, 0)

    print("[exp7339] phase=protocol_seal event=start", flush=True)
    started = time.monotonic()
    protocol_path = raw / "cost_protocol.json"
    _atomic_json(protocol_path, COST_PROTOCOL)
    artifact["source_artifact_hashes"]["cost_protocol"] = sha256_file(protocol_path)
    _phase(spans, "protocol_seal", started, origin, 1)

    print("[exp7339] phase=native_build event=before_subprocess", flush=True)
    started = time.monotonic()
    try:
        extension, service_path, build_receipts, build_identity = _build_native(root, raw)
    except RuntimeError as error:
        artifact.update(
            {
                "status": "complete",
                "completed_at_utc": datetime.now(UTC).isoformat(),
                "verdict_class": "disqualified",
                "honest_verdict": f"complete_disqualified: {error}",
                "phase_spans": spans,
                "duration_s": time.monotonic() - origin,
            }
        )
        artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
        return artifact
    artifact["validation_receipts"].extend(build_receipts)
    artifact["setup_costs"]["native_and_service_build_s"] = time.monotonic() - started
    _phase(spans, "native_build", started, origin, len(build_receipts))
    print("[exp7339] phase=native_build event=after_subprocess", flush=True)

    print("[exp7339] phase=extension_import event=before", flush=True)
    started = time.monotonic()
    binding = load_native_extension(extension)
    import_duration = time.monotonic() - started
    artifact["setup_costs"]["extension_import_s"] = import_duration
    artifact["binding_identity"] = _binding_identity(root, extension, binding, build_identity)
    artifact["source_artifact_hashes"]["loaded_extension"] = sha256_file(extension)
    _phase(spans, "extension_import", started, origin, 1)
    print(f"[exp7339] phase=extension_import event=after module={binding.__file__}", flush=True)

    print("[exp7339] phase=fixture_load event=start", flush=True)
    started = time.monotonic()
    historical_fixtures = read_jsonl(root / FIXTURE_PATH)
    adverse = adverse_fixtures()
    fixtures = [*historical_fixtures, *adverse]
    _phase(spans, "fixture_load", started, origin, len(fixtures))

    print("[exp7339] phase=parity event=before", flush=True)
    started = time.monotonic()
    parity = native_parity(binding, fixtures)
    mutation = mutation_parity(binding)
    parity.extend(mutation)
    parity_path = raw / "parity_rows.jsonl"
    _atomic_jsonl(parity_path, parity)
    parity_summary = reduce_parity_rows(parity)
    artifact["source_artifact_hashes"]["parity_rows"] = sha256_file(parity_path)
    parity_duration = time.monotonic() - started
    artifact["validation_receipts"].append(
        _validation_receipt(
            "e2e_003_schedule_roundtrip",
            "actual imported RustCompiledScheduleEvaluator replay",
            "schedule input -> Rust energy -> detached Python output",
            parity_duration,
            parity_path,
            parity_summary["all_matched"] is True,
        )
    )
    _phase(spans, "parity", started, origin, len(parity))
    print("[exp7339] phase=parity event=after", flush=True)

    print("[exp7339] phase=historical_service_startup event=before", flush=True)
    started = time.monotonic()
    service = exp7326.JsonLineService((str(service_path),), root)
    artifact["setup_costs"]["historical_service_startup_s"] = service.startup_s
    _phase(spans, "historical_service_startup", started, origin, 1)
    print("[exp7339] phase=historical_service_startup event=after", flush=True)

    print("[exp7339] phase=benchmark event=before", flush=True)
    started = time.monotonic()
    try:
        cost_rows, benchmark_setup = benchmark_costs(binding, service, historical_fixtures)
    finally:
        service.close()
    artifact["setup_costs"].update(benchmark_setup)
    cost_path = raw / "cost_rows.jsonl"
    _atomic_jsonl(cost_path, cost_rows)
    artifact["source_artifact_hashes"]["cost_rows"] = sha256_file(cost_path)
    cost_summary = reduce_cost_rows(cost_rows)
    _phase(spans, "benchmark", started, origin, len(cost_rows))
    print("[exp7339] phase=benchmark event=after", flush=True)

    print("[exp7339] phase=independent_reduction event=start", flush=True)
    started = time.monotonic()
    independent_parity = reduce_parity_rows(read_jsonl(parity_path))
    independent_cost = reduce_cost_rows(read_jsonl(cost_path))
    artifact["independent_raw_reduction"] = {
        "parity": independent_parity,
        "cost": independent_cost,
        "matched_in_memory_reduction": independent_parity == parity_summary
        and independent_cost == cost_summary,
    }
    reduction_path = raw / "independent_reduction.json"
    _atomic_json(reduction_path, artifact["independent_raw_reduction"])
    artifact["source_artifact_hashes"]["independent_reduction"] = sha256_file(reduction_path)
    artifact["validation_receipts"].append(
        _validation_receipt(
            "independent_raw_reduction",
            "reload parity_rows.jsonl and cost_rows.jsonl",
            "task-owned measured raw rows",
            time.monotonic() - started,
            reduction_path,
            artifact["independent_raw_reduction"]["matched_in_memory_reduction"] is True,
        )
    )
    _phase(spans, "independent_reduction", started, origin, 2)

    artifact.update(
        {
            "rows": cost_rows,
            "parity_rows": parity,
            "sample_size_budget": {
                "historical_parity_planned": HISTORICAL_FIXTURE_COUNT,
                "historical_parity_attempted": HISTORICAL_FIXTURE_COUNT,
                "historical_parity_completed": HISTORICAL_FIXTURE_COUNT,
                "historical_parity_censored": 0,
                "adverse_planned": ADVERSE_FIXTURE_COUNT,
                "adverse_completed": ADVERSE_FIXTURE_COUNT,
                "mutation_planned": MUTATION_CASE_COUNT,
                "mutation_completed": MUTATION_CASE_COUNT,
                "parity_completed": len(parity),
                "cost_rows_planned": 270,
                "cost_rows_attempted": len(cost_rows),
                "cost_rows_completed": len(cost_rows),
                "cost_rows_censored": sum(row["censored"] is True for row in cost_rows),
                "paired_blocks_per_size": PAIRED_BLOCKS,
                "batch_sizes": list(BATCH_SIZES),
                "stopping_rule": COST_PROTOCOL["stopping_rule"],
            },
            "acceptance_gate_results": {
                "parity": _gate(
                    {"rows": HISTORICAL_FIXTURE_COUNT + ADVERSE_FIXTURE_COUNT, "mismatches": 0},
                    reduce_parity_rows(parity[:-MUTATION_CASE_COUNT]),
                    len(parity[:-MUTATION_CASE_COUNT])
                    == HISTORICAL_FIXTURE_COUNT + ADVERSE_FIXTURE_COUNT
                    and all(row["matched"] for row in parity[:-MUTATION_CASE_COUNT]),
                    "All authenticated and adverse evaluator outputs must match exactly.",
                ),
                "mutation": _gate(
                    {"rows": MUTATION_CASE_COUNT, "mismatches": 0},
                    reduce_parity_rows(mutation),
                    all(row["matched"] for row in mutation),
                    "Constructor inputs and returned objects must not alias native state.",
                ),
                "e2e_003": _gate(
                    "actual imported extension round trip with zero differences",
                    {"module": str(binding.__file__), "mismatches": parity_summary["mismatches"]},
                    parity_summary["all_matched"] is True,
                    "A build-only or source-only check cannot satisfy E2E-003.",
                ),
                "cost_protocol": _gate(
                    {"rows": 270, "paired_blocks": 90, "mismatches": 0},
                    cost_summary,
                    cost_summary["complete"] is True,
                    "Every fixed block must include equivalent current outputs for all arms.",
                ),
                "affected_validation": _gate(
                    "all affected checks pass",
                    "pending",
                    False,
                    "Readiness requires current scoped and Rust validation.",
                ),
            },
        }
    )

    print("[exp7339] phase=affected_validation event=before", flush=True)
    started = time.monotonic()
    scoped, rust_receipts, rust_baseline = _validation_commands(root, raw)
    artifact["validation_receipts"].extend(scoped["validation_receipts"])
    artifact["validation_receipts"].extend(rust_receipts)
    if rust_baseline and rust_baseline[0]["passed"] is not True:
        health = artifact["repository_health"]
        historical = list(health.get("historical_failures", []))
        historical.append(
            {
                "classification": "unrelated_unchanged_carnot_python_clippy_baseline",
                "command": rust_baseline[0]["command"],
                "date": RUN_DATE,
                "duration_s": rust_baseline[0]["duration_s"],
                "exit_code": rust_baseline[0]["exit_code"],
                "log_sha256": rust_baseline[0]["log_sha256"],
                "resolved": False,
                "source_experiment_id": EXPERIMENT_ID,
                "unchanged_files": [
                    "crates/carnot-python/src/s2kan.rs",
                    "crates/carnot-python/src/adaptive_state.rs",
                    "crates/carnot-python/src/mode_jump.rs",
                    "crates/carnot-python/src/one_axis_tempering.rs",
                    "crates/carnot-python/src/packed_belief.rs",
                    "crates/carnot-python/src/safety_net.rs",
                ],
            }
        )
        health.update(
            {
                "status": "degraded_open",
                "incident_open": True,
                "historical_failures": historical,
                "historical_failure_count": len(historical),
                "affects_required_checks": False,
            }
        )
    artifact.update(
        {
            key: scoped[key]
            for key in (
                "required_checks_passed",
                "missing_required_commands",
                "failed_required_commands",
                "duplicate_required_commands",
            )
        }
    )
    rust_passed = all(row["passed"] for row in rust_receipts)
    affected_passed = scoped["required_checks_passed"] is True and rust_passed
    artifact["acceptance_gate_results"]["affected_validation"] = _gate(
        "all affected checks pass",
        {
            "scoped_python": scoped["required_checks_passed"],
            "rust": rust_passed,
            "failed": [row["name"] for row in rust_receipts if not row["passed"]]
            + list(scoped["failed_required_commands"]),
        },
        affected_passed,
        "Readiness requires current scoped and Rust validation.",
    )
    _phase(
        spans,
        "affected_validation",
        started,
        origin,
        len(REQUIRED_CHECK_NAMES) + len(rust_receipts) + len(rust_baseline),
    )
    print("[exp7339] phase=affected_validation event=after", flush=True)

    gates_pass = all(row["passed"] is True for row in artifact["acceptance_gate_results"].values())
    artifact.update(
        {
            "status": "complete",
            "completed_at_utc": datetime.now(UTC).isoformat(),
            "native_binding_ready_score": int(gates_pass),
            "verdict_class": "circular_positive" if gates_pass else "disqualified",
            "honest_verdict": (
                "complete_circular_positive: the imported in-process binding preserves exact "
                "schedule semantics and completes the fixed boundary protocol under shared "
                "evaluator authority; no speed or learning-value claim is made"
                if gates_pass
                else "complete_disqualified: native parity, boundary protocol, or affected validation failed"
            ),
            "phase_spans": spans,
            "duration_s": time.monotonic() - origin,
        }
    )
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def complete_artifact_fixture_for_test(root: Path) -> JsonDict:
    """Build a complete deterministic artifact used by cold-validator tests."""

    module = root / "_rust.fixture.so"
    module.write_bytes(b"compiled fixture")
    checks = [check("fixture", "available", "path", True, True, True, "fixture")]
    artifact = _base_artifact(checks, {"fixture": "sha256:fixture"})
    parity = [
        {
            "fixture_id": f"fixture-{index}",
            "source": "test",
            "case": "valid",
            "python": {"value": index},
            "rust": {"value": index},
            "matched": True,
        }
        for index in range(PARITY_ROW_COUNT)
    ]
    rows = synthetic_cost_rows()
    artifact.update(
        {
            "status": "complete",
            "completed_at_utc": "2026-09-16T00:00:00+00:00",
            "rows": rows,
            "parity_rows": parity,
            "sample_size_budget": {
                "parity_completed": PARITY_ROW_COUNT,
                "cost_rows_completed": 270,
            },
            "acceptance_gate_results": {
                name: _gate(True, True, True, "fixture")
                for name in (
                    "parity",
                    "mutation",
                    "e2e_003",
                    "cost_protocol",
                    "affected_validation",
                )
            },
            "honest_verdict": "complete_circular_positive: fixture",
            "verdict_class": "circular_positive",
            "native_binding_ready_score": 1,
            "binding_identity": {
                "module_file": str(module),
                "module_sha256": sha256_file(module),
                "compiled_execution": True,
                "python_fallback_used": False,
            },
            "required_checks_passed": True,
            "missing_required_commands": [],
            "failed_required_commands": [],
            "duplicate_required_commands": [],
        }
    )
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def blocked_artifact_fixture_for_test() -> JsonDict:
    """Build one deterministic external-absence artifact for validator tests."""

    checks = [check("missing", "available", "path", True, False, False, "fixture")]
    return _blocked_artifact(checks, {})


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the fixed date and cold-validation modes."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", choices=[RUN_DATE])
    parser.add_argument("--validate", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the bounded experiment and atomically publish terminal evidence."""

    args = _parse_args(argv)
    print("[exp7339] phase=startup event=start", flush=True)
    if args.validate is not None:  # pragma: no cover - command-line cold validator.
        errors = validate_artifact(read_object(args.validate), check_files=True)
        print(json.dumps({"errors": errors}, sort_keys=True), flush=True)
        return int(bool(errors))
    if args.date != RUN_DATE:
        raise SystemExit("--date 20260916 is required")
    artifact = run_experiment(REPO_ROOT)
    if artifact["status"] == "blocked":  # pragma: no cover - external failure path.
        write_artifact(REPO_ROOT / RESULT_PATH, artifact, check_files=False)
        print(
            f"[exp7339] phase=terminal_write event=end path={REPO_ROOT / RESULT_PATH}", flush=True
        )
        return 0

    raw = REPO_ROOT / RAW_PATH
    candidate = raw / "terminal_candidate.json"
    _atomic_json(candidate, artifact)
    print("[exp7339] phase=terminal_validators event=before", flush=True)
    validator_started = time.monotonic()
    validators = _terminal_validators(REPO_ROOT, candidate, raw)
    validator_duration = time.monotonic() - validator_started
    artifact["validation_receipts"].extend(validators)
    validators_passed = all(row["passed"] for row in validators)
    artifact["acceptance_gate_results"]["terminal_validators"] = _gate(
        "both terminal validators pass",
        {
            "passed": [row["name"] for row in validators if row["passed"]],
            "failed": [row["name"] for row in validators if not row["passed"]],
        },
        validators_passed,
        "Final readiness requires adversarial and row-consistency validation.",
    )
    if not validators_passed:
        artifact.update(
            {
                "native_binding_ready_score": 0,
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
    write_artifact(REPO_ROOT / RESULT_PATH, artifact, check_files=True)
    print(f"[exp7339] phase=terminal_validators event=after passed={validators_passed}", flush=True)
    print(
        f"[exp7339] phase=terminal_write event=end path={REPO_ROOT / RESULT_PATH} "
        f"verdict={artifact['verdict_class']}",
        flush=True,
    )
    return int(artifact["verdict_class"] == "disqualified")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
