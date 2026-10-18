"""Measure exact Rust parity and host cost for acquired schedule constraints.

The reusable evaluator handles only integer pairwise separation and integer
sliding-window capacity. The experiment authenticates the promoted acquisition,
recovers finite captured schedules from their hashes, compares equivalent
serialized Python and Rust services, and reports host timing separately from a
software-only hardware placement projection.

Spec refs: REQ-VERIFY-7326 and SCENARIO-VERIFY-7326-*.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from copy import deepcopy
from datetime import UTC, datetime
import hashlib
import itertools
import json
import os
from pathlib import Path
import random
import statistics
import subprocess
import sys
import tempfile
import time
from typing import Any, TextIO

from carnot.reporting.experiment_7303_validation_scope import (
    CommandSpec,
    REQUIRED_CHECK_NAMES,
    run_commands,
    run_scoped_validation,
)


JsonDict = dict[str, Any]

EXPERIMENT_ID = 7326
MILESTONE = "2026.09.643"
RUN_DATE = "20260915"
SCHEMA = "carnot.experiment_7326.v643_constraint_kernel.v1"
REQUEST_SCHEMA = "carnot.constraint_kernel.request.v1"
PERSISTENT_ARM = "persistent_structural_acquisition"
I64_MIN = -(2**63)
I64_MAX = 2**63 - 1
U64_MAX = 2**64 - 1
SEEDED_FIXTURE_COUNT = 1000
CAPTURED_REQUEST_COUNT = 2304
BATCH_SIZES = (1, 32, 256)
PAIRED_BLOCKS = 30
DEVELOPMENT_SEED = 7326001
EVALUATION_SEED = 7326002
BOOTSTRAP_SEED = 7326003

REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = Path("python/carnot/experiment_7326_v643_constraint_kernel.py")
TEST_PATH = Path("tests/python/test_experiment_7326_v643_constraint_kernel.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_7326_v643_constraint_kernel.py")
SPEC_PATH = Path("openspec/capabilities/constraint-verification/spec.md")
UPSTREAM_PATH = Path("results/experiment_7325_v643_addition_audit.json")
RUST_PATHS = (
    Path("crates/carnot-constraints/src/schedule.rs"),
    Path("crates/carnot-constraints/src/lib.rs"),
    Path("crates/carnot-constraints/examples/experiment_7326_constraint_kernel.rs"),
    Path("crates/carnot-constraints/tests/experiment_7326_constraint_kernel.rs"),
)

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


def canonical_bytes(value: Any) -> bytes:
    """Return the exact compact, sorted JSON representation used for hashes."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def sha256_json(value: Any) -> str:
    """Hash canonical JSON with the repository's explicit algorithm prefix."""

    return "sha256:" + hashlib.sha256(canonical_bytes(value)).hexdigest()


def sha256_file(path: Path) -> str:
    """Hash exact file bytes without loading a large capture into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _invalid(error: str) -> JsonDict:
    """Return the one fail-closed shape shared with the Rust evaluator."""

    return {
        "valid_input": False,
        "feasible": False,
        "complete_oracle_certificate": False,
        "total_energy": None,
        "terms": [],
        "error": error,
    }


def _integer(value: Any, field: str) -> tuple[int | None, JsonDict | None]:
    """Accept a signed 64-bit integer but never Python's Boolean subtype."""

    if isinstance(value, bool) or not isinstance(value, int) or not I64_MIN <= value <= I64_MAX:
        return None, _invalid(f"invalid_integer:{field}")
    return value, None


def _text(value: Any, field: str, constraint_id: str) -> tuple[str | None, JsonDict | None]:
    """Read one required nonempty term string using Rust-compatible errors."""

    if not isinstance(value, str) or not value:
        return None, _invalid(f"missing_text:{constraint_id}:{field}")
    return value, None


def _term_id(term: Mapping[str, Any]) -> str:
    """Return a stable identifier even for an unknown malformed term."""

    value = term.get("constraint_id")
    return value if isinstance(value, str) and value else "missing"


def evaluate_request(request: Mapping[str, Any]) -> JsonDict:
    """Evaluate one schedule with exact nonnegative integer term energies."""

    if request.get("schema") != REQUEST_SCHEMA:
        return _invalid("invalid_schema")
    version = request.get("executor_version")
    if not isinstance(version, str) or not version:
        return _invalid("empty_executor_version")
    slot_min, error = _integer(request.get("slot_min"), "slot_min")
    if error is not None:
        return error
    slot_max, error = _integer(request.get("slot_max"), "slot_max")
    if error is not None:
        return error
    assert slot_min is not None and slot_max is not None
    if slot_min > slot_max:
        return _invalid("invalid_slot_domain")
    schedule = request.get("schedule")
    if not isinstance(schedule, list) or not schedule:
        return _invalid("empty_schedule")

    assignments: dict[str, int] = {}
    for assignment in schedule:
        if not isinstance(assignment, Mapping):
            return _invalid("invalid_assignment")
        activity = assignment.get("activity")
        if not isinstance(activity, str) or not activity:
            return _invalid("empty_activity")
        if activity in assignments:
            return _invalid(f"duplicate_activity:{activity}")
        slot, error = _integer(assignment.get("slot"), f"schedule:{activity}")
        if error is not None:
            return error
        assert slot is not None
        if slot < slot_min or slot > slot_max:
            return _invalid(f"slot_out_of_domain:{activity}")
        assignments[activity] = slot

    constraints = request.get("constraints")
    if not isinstance(constraints, list):
        return _invalid("invalid_constraints")
    output: list[JsonDict] = []
    total = 0
    for term in constraints:
        if not isinstance(term, Mapping):
            return _invalid("invalid_constraint")
        constraint_id = _term_id(term)
        if constraint_id == "missing":
            return _invalid("empty_constraint_id")
        kind = term.get("kind")
        if kind not in {"pairwise_separation", "sliding_window_capacity"}:
            return _invalid(f"unknown_constraint_kind:{constraint_id}")
        if term.get("version") != version:
            return _invalid(f"version_mismatch:{constraint_id}")

        if kind == "pairwise_separation":
            left, error = _text(term.get("left"), "left", constraint_id)
            if error is not None:
                return error
            right, error = _text(term.get("right"), "right", constraint_id)
            if error is not None:
                return error
            assert left is not None and right is not None
            if left == right:
                return _invalid(f"identical_pair:{constraint_id}")
            minimum, error = _integer(term.get("minimum"), f"{constraint_id}:minimum")
            if error is not None:
                return _invalid(f"missing_integer:{constraint_id}:minimum")
            assert minimum is not None
            if minimum < 0:
                return _invalid(f"negative_minimum:{constraint_id}")
            if left not in assignments:
                return _invalid(f"missing_activity:{constraint_id}:{left}")
            if right not in assignments:
                return _invalid(f"missing_activity:{constraint_id}:{right}")
            deficit = max(0, minimum - abs(assignments[left] - assignments[right]))
            energy = deficit * deficit
        else:
            window_size, error = _integer(term.get("window_size"), f"{constraint_id}:window_size")
            if error is not None:
                return _invalid(f"missing_integer:{constraint_id}:window_size")
            maximum, error = _integer(term.get("maximum"), f"{constraint_id}:maximum")
            if error is not None:
                return _invalid(f"missing_integer:{constraint_id}:maximum")
            assert window_size is not None and maximum is not None
            if window_size <= 0:
                return _invalid(f"nonpositive_window:{constraint_id}")
            if maximum < 0:
                return _invalid(f"negative_maximum:{constraint_id}")
            domain_width = slot_max - slot_min + 1
            if window_size > domain_width:
                return _invalid(f"window_exceeds_domain:{constraint_id}")
            if slot_max == I64_MAX:
                return _invalid(f"window_end_overflow:{constraint_id}")
            energy = 0
            for start in range(slot_min, slot_max + 2 - window_size):
                occupancy = sum(
                    start <= slot < start + window_size for slot in assignments.values()
                )
                excess = max(0, occupancy - maximum)
                energy += excess * excess
                if energy > U64_MAX:  # pragma: no cover - requires over 2^32 assignments.
                    return _invalid(f"energy_overflow:{constraint_id}")

        if energy > U64_MAX:
            return _invalid(f"energy_overflow:{constraint_id}")
        total += energy
        if total > U64_MAX:
            return _invalid(f"total_energy_overflow:{constraint_id}")
        output.append(
            {
                "constraint_id": constraint_id,
                "kind": kind,
                "energy": energy,
                "satisfied": energy == 0,
            }
        )

    return {
        "valid_input": True,
        "feasible": total == 0,
        "complete_oracle_certificate": False,
        "total_energy": total,
        "terms": output,
        "error": None,
    }


def evaluate_batch(requests: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Evaluate an ordered request batch without changing service semantics."""

    return [evaluate_request(request) for request in requests]


def service_response(message: Mapping[str, Any]) -> JsonDict:
    """Apply one newline-service operation for tests and subprocess use."""

    operation = message.get("operation")
    if operation == "ping":
        return {"kind": "ready"}
    if operation == "shutdown":
        return {"kind": "shutdown"}
    if operation == "evaluate":
        requests = message.get("requests")
        if not isinstance(requests, list):
            raise ValueError("invalid_requests")
        return {"results": evaluate_batch(requests)}
    raise ValueError("unknown_operation")


def _check(
    upstream: str,
    check: str,
    field: str,
    expected: Any,
    observed: Any,
    passed: bool,
    principle: str,
) -> JsonDict:
    """Keep every pre-gate comparison in one auditable shape."""

    return {
        "upstream": upstream,
        "check": check,
        "field": field,
        "expected_value": expected,
        "observed_value": observed,
        "passed": passed,
        "principle": principle,
    }


def gate_check_summary(checks: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Reduce preconditions without discarding the exact first failure."""

    failed = [dict(row) for row in checks if row.get("passed") is not True]
    return {
        "passed": not failed,
        "check_count": len(checks),
        "failed_check_count": len(failed),
        "first_failure": failed[0] if failed else None,
        "checks": [dict(row) for row in checks],
    }


def _read_object(path: Path) -> JsonDict:
    """Read one JSON object or raise a stable input error."""

    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"not_json_object:{path}")
    return value


def collect_preconditions(upstream_path: Path) -> tuple[list[JsonDict], JsonDict, Path]:
    """Authenticate Exp7325 and the exact raw rows it declares."""

    upstream_name = str(upstream_path)
    available = upstream_path.is_file()
    upstream = _read_object(upstream_path) if available else {}
    checks = [
        _check(
            upstream_name,
            "upstream_available",
            "path",
            True,
            available,
            available,
            "A missing promoted audit cannot authorize kernel work.",
        )
    ]
    fixed = (
        (
            "upstream_terminal",
            "status",
            "complete",
            upstream.get("status"),
            upstream.get("status") == "complete",
            "Only terminal external work can authorize kernel measurement.",
        ),
        (
            "addition_promoted",
            "addition_promotion_score",
            1,
            upstream.get("addition_promotion_score"),
            upstream.get("addition_promotion_score") == 1,
            "Only independently useful structural acquisition warrants a kernel.",
        ),
        (
            "upstream_not_disqualified",
            "verdict_class",
            "not disqualified",
            upstream.get("verdict_class"),
            upstream.get("verdict_class") != "disqualified",
            "An ineligible external class overrides a score of one.",
        ),
        (
            "upstream_not_blocked",
            "verdict_class",
            "not blocked",
            upstream.get("verdict_class"),
            upstream.get("verdict_class") != "blocked",
            "An ineligible external class overrides a score of one.",
        ),
        (
            "upstream_not_partial",
            "verdict_class",
            "not partial",
            upstream.get("verdict_class"),
            upstream.get("verdict_class") != "partial",
            "An ineligible external class overrides a score of one.",
        ),
        (
            "upstream_not_quarantined",
            "flagged_adversarial",
            False,
            upstream.get("flagged_adversarial"),
            upstream.get("flagged_adversarial") is not True,
            "Quarantined evidence cannot authorize kernel work.",
        ),
    )
    checks.extend(
        _check(upstream_name, name, field, expected, observed, passed, principle)
        for name, field, expected, observed, passed, principle in fixed
    )

    raw_receipt = (
        upstream.get("source_artifact_hashes", {}).get("raw_evidence", {}).get("rows", {})
        if available
        else {}
    )
    raw_value = raw_receipt.get("path")
    raw_path = Path(raw_value) if isinstance(raw_value, str) else Path()
    raw_available = bool(raw_value) and raw_path.is_file()
    checks.append(
        _check(
            str(raw_path),
            "acquired_rows_available",
            "path",
            True,
            raw_available,
            raw_available,
            "Exact acquired records must remain available.",
        )
    )
    observed_hash = sha256_file(raw_path) if raw_available else None
    expected_hash = raw_receipt.get("sha256")
    checks.append(
        _check(
            str(raw_path),
            "acquired_rows_hash",
            "sha256",
            expected_hash,
            observed_hash,
            expected_hash is not None and observed_hash == expected_hash,
            "Exact raw bytes, not an embedded summary, own the port.",
        )
    )
    if raw_available:
        with raw_path.open("rb") as stream:
            observed_count = sum(1 for _ in stream)
    else:
        observed_count = None
    expected_count = raw_receipt.get("row_count")
    checks.append(
        _check(
            str(raw_path),
            "captured_row_count",
            "row_count",
            expected_count,
            observed_count,
            expected_count is not None and observed_count == expected_count,
            "Every captured arm-request remains in the parity denominator.",
        )
    )
    hashes = {"upstream_artifact": sha256_file(upstream_path) if available else None}
    if raw_available:
        hashes["acquired_rows"] = observed_hash
    return checks, hashes, raw_path


def read_jsonl(path: Path) -> list[JsonDict]:
    """Load exact object rows from a task-owned authenticated JSONL file."""

    rows: list[JsonDict] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"invalid_jsonl_row:{path}:{line_number}")
            rows.append(value)
    return rows


def extract_acquired_records(rows: Sequence[Mapping[str, Any]]) -> tuple[list[JsonDict], list[str]]:
    """Authenticate and deduplicate every acquired atom from persistent rows."""

    records: list[JsonDict] = []
    seen: set[str] = set()
    errors: list[str] = []
    for row in rows:
        if row.get("arm") != PERSISTENT_ARM:
            continue
        atoms = row.get("new_atoms", [])
        if not isinstance(atoms, list):
            errors.append("invalid_new_atoms")
            continue
        for atom in atoms:
            if not isinstance(atom, Mapping):
                errors.append("invalid_atom")
                continue
            body = {
                key: deepcopy(atom.get(key))
                for key in ("kind", "version", "payload", "witness", "query_receipts")
            }
            atom_id = str(atom.get("atom_id"))
            if atom_id != sha256_json(body):
                errors.append(f"atom_hash:{atom_id}")
                continue
            if atom_id not in seen:
                records.append(deepcopy(dict(atom)))
                seen.add(atom_id)
    return records, sorted(set(errors))


def constraints_by_version(records: Sequence[Mapping[str, Any]]) -> dict[str, list[JsonDict]]:
    """Translate acquired atoms into the exact serialized kernel terms."""

    output: dict[str, list[JsonDict]] = {}
    for atom in records:
        version = str(atom["version"])
        payload = atom["payload"]
        atom_id = str(atom["atom_id"])
        if atom["kind"] == "pairwise_separation":
            left, right = payload["pair"]
            term = {
                "kind": "pairwise_separation",
                "constraint_id": atom_id,
                "version": version,
                "left": left,
                "right": right,
                "minimum": payload["minimum"],
            }
        elif atom["kind"] == "capacity":
            term = {
                "kind": "sliding_window_capacity",
                "constraint_id": atom_id,
                "version": version,
                "window_size": 1,
                "maximum": payload["maximum"],
            }
        else:
            raise ValueError(f"unsupported_acquired_kind:{atom['kind']}")
        output.setdefault(version, []).append(term)
    return output


def recover_captured_request(
    row: Mapping[str, Any], constraints: Sequence[Mapping[str, Any]]
) -> JsonDict:
    """Recover a returned finite plan only when its exact captured hash matches."""

    request_id = str(row["request_id"])
    request_index = int(row["request_index"])
    version = str(row["executor_version"])
    offset = request_index % 2
    slots = list(range(offset, offset + 4))
    schedule: list[JsonDict] = []
    if row.get("returned") is True:
        wanted = row.get("returned_plan_hash")
        for values in itertools.product(slots, repeat=4):
            assignments = dict(zip(("a", "b", "c", "d"), values, strict=True))
            plan = {"request_id": request_id, "assignments": assignments}
            if sha256_json(plan) == wanted:
                schedule = [
                    {"activity": activity, "slot": slot} for activity, slot in assignments.items()
                ]
                break
        if not schedule:
            raise ValueError(f"captured_plan_hash_not_recovered:{request_id}")
    return {
        "schema": REQUEST_SCHEMA,
        "executor_version": version,
        "slot_min": slots[0],
        "slot_max": slots[-1],
        "schedule": schedule,
        "constraints": [deepcopy(dict(term)) for term in constraints],
    }


def _synthetic_request(rng: random.Random, case: str, index: int) -> JsonDict:
    """Construct one bounded valid or semantic-malformed parity request."""

    version = f"seeded-executor-{index % 7}"
    slot_max = rng.randint(3, 8)
    schedule = [
        {"activity": name, "slot": rng.randint(0, slot_max)} for name in ("a", "b", "c", "d")
    ]
    terms = [
        {
            "kind": "pairwise_separation",
            "constraint_id": f"seeded-sep-{index}",
            "version": version,
            "left": "a",
            "right": "b",
            "minimum": rng.randint(0, 4),
        },
        {
            "kind": "sliding_window_capacity",
            "constraint_id": f"seeded-cap-{index}",
            "version": version,
            "window_size": rng.randint(1, min(3, slot_max + 1)),
            "maximum": rng.randint(0, 4),
        },
    ]
    request = {
        "schema": REQUEST_SCHEMA,
        "executor_version": version,
        "slot_min": 0,
        "slot_max": slot_max,
        "schedule": schedule,
        "constraints": terms,
    }
    if case == "invalid_slot":
        schedule[0]["slot"] = slot_max + 1
    elif case == "empty_schedule":
        request["schedule"] = []
    elif case == "version_mismatch":
        terms[0]["version"] = version + "-stale"
    elif case == "overlapping_windows":
        terms[1]["window_size"] = min(3, slot_max + 1)
        terms[1]["maximum"] = 1
    elif case == "overflow":
        request.update(
            {
                "slot_min": I64_MAX,
                "slot_max": I64_MAX,
                "schedule": [{"activity": "a", "slot": I64_MAX}],
                "constraints": [
                    {
                        "kind": "sliding_window_capacity",
                        "constraint_id": f"seeded-cap-{index}",
                        "version": version,
                        "window_size": 1,
                        "maximum": 1,
                    }
                ],
            }
        )
    return request


def seeded_fixtures(seed: int, *, count: int) -> list[JsonDict]:
    """Materialize the frozen valid and malformed finite parity budget."""

    rng = random.Random(seed)
    cases = (
        "valid",
        "invalid_slot",
        "empty_schedule",
        "version_mismatch",
        "overlapping_windows",
        "overflow",
    )
    return [
        {
            "fixture_id": f"seeded-{index:04d}",
            "source": "seeded",
            "case": cases[index % len(cases)],
            "request": _synthetic_request(rng, cases[index % len(cases)], index),
        }
        for index in range(count)
    ]


def _percentile(values: Sequence[float], probability: float) -> float:
    """Return a deterministic linearly interpolated percentile."""

    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def paired_speedup_interval(values: Sequence[float], *, seed: int) -> JsonDict:
    """Bootstrap a paired mean speedup without changing the frozen block count."""

    if not values:
        return {"n": 0, "point_estimate": None, "ci95_lower": None, "ci95_upper": None}
    rng = random.Random(seed)
    means = [statistics.fmean(rng.choice(values) for _ in values) for _ in range(10_000)]
    return {
        "n": len(values),
        "point_estimate": statistics.fmean(values),
        "ci95_lower": _percentile(means, 0.025),
        "ci95_upper": _percentile(means, 0.975),
    }


def derive_terminal_scores(
    *, parity_complete: bool, costs_complete: bool, speedup: bool
) -> tuple[int, int, str]:
    """Keep correctness completion separate from the predeclared 10x target."""

    complete = parity_complete and costs_complete
    if not complete:
        return 0, 0, "disqualified"
    if speedup:
        return 1, 1, "circular_positive"
    return 1, 0, "null"


def _signed_bit_width(values: Sequence[int]) -> int:
    """Return the smallest signed two's-complement width for observed integers."""

    width = 1
    while any(value < -(2 ** (width - 1)) or value > 2 ** (width - 1) - 1 for value in values):
        width += 1
    return width


def hardware_projection(
    records: Sequence[Mapping[str, Any]], fixtures: Sequence[Mapping[str, Any]]
) -> JsonDict:
    """Project sparse integer placement facts without inventing board timings."""

    separation_count = sum(row.get("kind") == "pairwise_separation" for row in records)
    capacity_count = sum(row.get("kind") == "capacity" for row in records)
    max_activities = max(
        (
            len(row.get("request", {}).get("schedule", []))
            for row in fixtures
            if isinstance(row.get("request"), Mapping)
        ),
        default=0,
    )
    valid_fixtures = [
        row
        for row in fixtures
        if isinstance(row.get("request"), Mapping)
        and evaluate_request(row["request"])["valid_input"]
    ]
    slots = [
        int(assignment["slot"])
        for row in valid_fixtures
        for assignment in row.get("request", {}).get("schedule", [])
        if isinstance(assignment, Mapping) and isinstance(assignment.get("slot"), int)
    ]
    return {
        "classification": "software_projection_not_fpga_or_tsu_execution",
        "separation_term_count": separation_count,
        "capacity_term_count": capacity_count,
        "sparse_coefficient_count": separation_count * 2 + capacity_count * max_activities,
        "integer_slot_bit_width": _signed_bit_width(slots or [0]),
        "serialized_constraint_bytes": len(canonical_bytes(records)),
        "maximum_coupling_degree": max(1 if separation_count else 0, max_activities - 1),
        "measured_board_timing": None,
        "execution_venue": "host",
        "storage_acknowledgment_semantics_changed": False,
    }


def _field_principles() -> JsonDict:
    """Explain required fields without wrapping their executable values."""

    return {
        "schema": "Version this artifact; keep ordinary top-level experiment_id and milestone.",
        "status": "Write terminal output only after current work and required validation.",
        "run_date": "Use 20260915; preserve actual UTC timestamps and monotonic phase spans.",
        "preconditions_checked": "Record input identities, availability, and the exact failed check.",
        "MODEL_SPECS": "Current executable identities only; no LLM runs in this task.",
        "model_invoked": "True for any actual attempted load or generation, including unusable results.",
        "invocation_counts": "Separate attempted, completed, failed, cancelled, and in-flight work.",
        "inference_substrate": "Describe actual computation using the recognized substrate literal.",
        "inference_substrate_class": "Use the closed substrate class actually exercised.",
        "execution_venue": "Use host; historical board work is not current board execution.",
        "duration_s": "Measure real elapsed time without sleeping or padding.",
        "phase_spans": "Record disjoint monotonic spans, units, boundaries, and pending operations.",
        "random_seed": "Seal independent development, evaluation, and bootstrap seeds before results.",
        "reproducibility_checksum": "Bind code, inputs, evaluator identity, settings, and raw evidence.",
        "source_artifact_hashes": "Authenticate exact producers and current executable sources.",
        "rows": "Emit every timing unit and arm with costs, failures, abstentions, and censoring.",
        "sample_size_budget": "Record planned, attempted, complete, censored, and the stopping rule.",
        "acceptance_gate_results": "Keep expected, observed, passed, and principle for every gate.",
        "gate_check_summary": "A blocked result names the upstream, check, field, expected, and observed.",
        "verifier_is_oracle": "Shared Python semantics are port authority, so positive science is circular.",
        "honest_verdict": "Completed findings use a terminal prefix; external absence starts blocked_.",
        "verdict_class": "Use the closed positive, circular-positive, null, blocked, disqualified, partial enum.",
        "validation_receipts": "Keep exact command, scope, exit, elapsed time, and log hash.",
        "repository_health": "Preserve unrelated dated failures without passing current checks.",
        "field_principles": "Explain why each field exists without wrapping executable values.",
        "constraint_kernel_complete_score": "One means exact parity and measured host cost are complete.",
        "kernel_speedup_score": "One requires a paired lower CI95 of at least 10 at every fixed size.",
        "kernel_rows": "Keep decisions, term energies, serialization cost, elapsed time, and paired blocks.",
        "hardware_placement": "Report actual sparse integer bytes and degree as a software projection.",
    }


def base_artifact(checks: Sequence[Mapping[str, Any]], hashes: Mapping[str, Any]) -> JsonDict:
    """Create a schema-complete candidate before scientific or validation scoring."""

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
            "bootstrap": BOOTSTRAP_SEED,
            "sealed_before_results": True,
        },
        "reproducibility_checksum": "",
        "source_artifact_hashes": deepcopy(dict(hashes)),
        "rows": [],
        "sample_size_budget": {},
        "acceptance_gate_results": {},
        "verifier_is_oracle": True,
        "honest_verdict": "partial: constraint kernel measurement has not completed",
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
        "constraint_kernel_complete_score": 0,
        "kernel_speedup_score": 0,
        "kernel_rows": {"parity": [], "cost": [], "serialization": [], "process_startup": []},
        "hardware_placement": {},
        "correctness_outcome": "pending",
        "performance_outcome": "pending",
        "whole_learning_speedup_claimed": False,
        "production_default_changed": False,
        "publication_surface_changed": False,
        "deployment_changed": False,
        "hardware_commands_run": False,
    }


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Bind deterministic evidence while excluding clocks and command durations."""

    keys = (
        "schema",
        "experiment_id",
        "milestone",
        "run_date",
        "preconditions_checked",
        "random_seed",
        "source_artifact_hashes",
        "sample_size_budget",
        "acceptance_gate_results",
        "verifier_is_oracle",
        "constraint_kernel_complete_score",
        "kernel_speedup_score",
        "kernel_rows",
        "hardware_placement",
        "correctness_outcome",
        "performance_outcome",
    )
    return sha256_json({key: artifact.get(key) for key in keys})


def validate_artifact(artifact: Mapping[str, Any]) -> list[str]:
    """Cold-check terminal identity, scores, budgets, evidence, and safety claims."""

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
    add(artifact.get("MODEL_SPECS") != [], "model_specs")
    add(artifact.get("model_invoked") is not False, "model_invoked")
    add(artifact.get("invocation_counts") != INVOCATION_COUNTS, "invocation_counts")
    add(artifact.get("inference_substrate") != "cpu_exact_solver_or_simulator", "substrate")
    add(
        artifact.get("inference_substrate_class") != "cpu_exact_solver_or_simulator",
        "substrate_class",
    )
    add(artifact.get("execution_venue") != "host", "execution_venue")
    add(artifact.get("verifier_is_oracle") is not True, "verifier_is_oracle")
    add(artifact.get("whole_learning_speedup_claimed") is not False, "whole_learning_claim")
    add(
        any(
            artifact.get(key) is not False
            for key in ("deployment_changed", "hardware_commands_run")
        ),
        "hardware_safety",
    )
    status = artifact.get("status")
    add(status not in {"complete", "blocked"}, "status")
    score = artifact.get("constraint_kernel_complete_score")
    speed = artifact.get("kernel_speedup_score")
    add(score not in {0, 1} or speed not in {0, 1}, "scores")
    if status == "blocked":
        add(bool(artifact.get("rows")), "blocked_rows")
        add(score != 0 or speed != 0, "blocked_scores")
        add(artifact.get("verdict_class") != "blocked", "blocked_verdict")
        add(not str(artifact.get("honest_verdict", "")).startswith("blocked_"), "blocked_prefix")
    else:
        add(not artifact.get("rows"), "rows")
        add(not str(artifact.get("honest_verdict", "")).startswith("complete"), "terminal_prefix")
        add(
            artifact.get("verdict_class") not in {"null", "circular_positive", "disqualified"},
            "verdict",
        )
        gates = artifact.get("acceptance_gate_results", {})
        scientific_complete = (
            gates.get("parity", {}).get("passed") is True
            and gates.get("complete_cost_rows", {}).get("passed") is True
        )
        if artifact.get("verdict_class") != "disqualified":
            add(score != int(scientific_complete), "complete_score")
            add(
                speed
                != int(
                    scientific_complete
                    and gates.get("speedup_lower_ci95", {}).get("passed") is True
                ),
                "speedup_score",
            )
        if score == 1:
            add(gates.get("parity", {}).get("passed") is not True, "parity_gate")
            add(gates.get("complete_cost_rows", {}).get("passed") is not True, "cost_gate")
        if speed == 1:
            add(gates.get("speedup_lower_ci95", {}).get("passed") is not True, "speedup_gate")
        budget = artifact.get("sample_size_budget", {})
        add(budget.get("captured_complete") != CAPTURED_REQUEST_COUNT, "captured_budget")
        add(budget.get("seeded_complete") != SEEDED_FIXTURE_COUNT, "seeded_budget")
        add(budget.get("paired_blocks_per_size") != PAIRED_BLOCKS, "paired_budget")
    add(artifact.get("reproducibility_checksum") != reproducibility_checksum(artifact), "checksum")
    return sorted(set(errors))


def _atomic_bytes(path: Path, data: bytes) -> None:
    """Publish bytes with one same-directory replace and directory sync."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temp_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_path, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temp_path.unlink(missing_ok=True)


def _atomic_json(path: Path, value: Any) -> None:
    """Atomically write one canonical JSON document."""

    _atomic_bytes(path, canonical_bytes(value) + b"\n")


def _atomic_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Atomically write canonical JSONL evidence."""

    _atomic_bytes(path, b"".join(canonical_bytes(row) + b"\n" for row in rows))


def write_artifact(path: Path, artifact: Mapping[str, Any]) -> JsonDict:
    """Validate and atomically publish one terminal experiment artifact."""

    errors = validate_artifact(artifact)
    if errors:
        raise ValueError("artifact_validation_failed:" + ",".join(errors))
    _atomic_json(path, artifact)
    return {"path": str(path), "sha256": sha256_file(path)}


def prepare_scoped_basetemp(path: Path) -> Path:
    """Create the private parent expected by the shipped scoped pytest commands."""

    path.mkdir(parents=True, exist_ok=True)
    return path


class JsonLineService:  # pragma: no cover - exercised by the measured entrypoint boundary.
    """Own one persistent newline-delimited evaluator subprocess."""

    def __init__(self, argv: Sequence[str], cwd: Path) -> None:
        self.argv = tuple(argv)
        self.started = time.monotonic()
        self.process = subprocess.Popen(  # noqa: S603
            self.argv,
            cwd=cwd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        assert self.process.stdin is not None and self.process.stdout is not None
        self.input: TextIO = self.process.stdin
        self.output: TextIO = self.process.stdout
        if self.call({"operation": "ping"}) != {"kind": "ready"}:
            raise RuntimeError("service_ping_failed")
        self.startup_s = time.monotonic() - self.started

    def call(self, message: Mapping[str, Any]) -> JsonDict:
        """Send and receive one bounded line while retaining exact JSON semantics."""

        self.input.write(canonical_bytes(message).decode() + "\n")
        self.input.flush()
        line = self.output.readline()
        if not line:
            detail = self.process.stderr.read() if self.process.stderr is not None else ""
            raise RuntimeError(f"service_closed:{detail}")
        value = json.loads(line)
        if not isinstance(value, dict):
            raise RuntimeError("service_non_object")
        return value

    def close(self) -> None:
        """Request clean shutdown and bound any broken service teardown."""

        if self.process.poll() is None:
            self.call({"operation": "shutdown"})
            self.process.wait(timeout=10)


def _gate(expected: Any, observed: Any, passed: bool, principle: str) -> JsonDict:
    """Build one scientific or validation gate row."""

    return {"expected": expected, "observed": observed, "passed": passed, "principle": principle}


def _source_hashes(repo_root: Path) -> JsonDict:
    """Bind current Python, Rust, test, spec, and governing source identities."""

    paths = (
        MODULE_PATH,
        TEST_PATH,
        WRAPPER_PATH,
        SPEC_PATH,
        Path("python/carnot/reporting/experiment_7303_validation_scope.py"),
        Path("python/carnot/memory/transactional_constraint_memory.py"),
        *RUST_PATHS,
    )
    return {path.as_posix(): sha256_file(repo_root / path) for path in paths}


def _phase(spans: list[JsonDict], name: str, start: float, origin: float, units: int) -> None:
    """Append one disjoint completed monotonic phase span."""

    end = time.monotonic()
    spans.append(
        {
            "phase": name,
            "start_s": start - origin,
            "end_s": end - origin,
            "units": units,
            "checkpoint_boundaries": 1,
            "pending_operations": [],
        }
    )
    print(f"[exp7326] phase={name} event=end units={units} elapsed_s={end - start:.3f}", flush=True)


def _captured_fixtures(
    rows: Sequence[Mapping[str, Any]], by_version: Mapping[str, Sequence[Mapping[str, Any]]]
) -> list[JsonDict]:
    """Recover every captured arm-request schedule with exact hash matching."""

    cache: dict[tuple[str, Any], JsonDict] = {}
    output: list[JsonDict] = []
    for index, row in enumerate(rows):
        key = (str(row["request_id"]), row.get("returned_plan_hash"))
        terms = by_version.get(str(row["executor_version"]), [])
        if key not in cache:
            cache[key] = recover_captured_request(row, terms)
        request = deepcopy(cache[key])
        request["constraints"] = [deepcopy(dict(term)) for term in terms]
        output.append(
            {
                "fixture_id": f"captured-{index:04d}",
                "source": "captured",
                "case": "returned" if row.get("returned") is True else "abstained",
                "request": request,
            }
        )
    return output


def _service_parity(
    fixtures: Sequence[Mapping[str, Any]], python: JsonLineService, rust: JsonLineService
) -> list[JsonDict]:  # pragma: no cover - measured subprocess boundary.
    """Compare every serialized decision and ordered per-term energy."""

    output: list[JsonDict] = []
    for start in range(0, len(fixtures), 256):
        chunk = fixtures[start : start + 256]
        requests = [row["request"] for row in chunk]
        python_results = python.call({"operation": "evaluate", "requests": requests})["results"]
        rust_results = rust.call({"operation": "evaluate", "requests": requests})["results"]
        for fixture, python_result, rust_result in zip(
            chunk, python_results, rust_results, strict=True
        ):
            output.append(
                {
                    "fixture_id": fixture["fixture_id"],
                    "source": fixture["source"],
                    "case": fixture["case"],
                    "python": python_result,
                    "rust": rust_result,
                    "matched": python_result == rust_result,
                }
            )
        print(
            f"[exp7326] phase=parity event=unit_complete completed={len(output)}/{len(fixtures)}",
            flush=True,
        )
    return output


def _benchmark(
    fixtures: Sequence[Mapping[str, Any]], python: JsonLineService, rust: JsonLineService
) -> tuple[list[JsonDict], list[JsonDict], list[JsonDict]]:  # pragma: no cover
    """Measure paired persistent-service blocks and separate JSON costs."""

    paired: list[JsonDict] = []
    rows: list[JsonDict] = []
    serialization: list[JsonDict] = []
    valid = [row["request"] for row in fixtures if evaluate_request(row["request"])["valid_input"]]
    for size_index, size in enumerate(BATCH_SIZES):
        requests = [deepcopy(valid[index % len(valid)]) for index in range(size)]
        message = {"operation": "evaluate", "requests": requests}
        encoded_started = time.perf_counter_ns()
        encoded = [canonical_bytes(message) for _ in range(PAIRED_BLOCKS)]
        encode_ns = time.perf_counter_ns() - encoded_started
        decode_started = time.perf_counter_ns()
        for value in encoded:
            json.loads(value)
        decode_ns = time.perf_counter_ns() - decode_started
        serialization.append(
            {
                "batch_size": size,
                "iterations": PAIRED_BLOCKS,
                "serialized_bytes": len(encoded[0]),
                "encode_elapsed_ns": encode_ns,
                "decode_elapsed_ns": decode_ns,
            }
        )
        for _ in range(3):
            python.call(message)
            rust.call(message)
        for block in range(PAIRED_BLOCKS):
            order = ("python", "rust") if block % 2 == 0 else ("rust", "python")
            elapsed: dict[str, int] = {}
            responses: dict[str, JsonDict] = {}
            for arm in order:
                service = python if arm == "python" else rust
                started = time.perf_counter_ns()
                responses[arm] = service.call(message)
                elapsed[arm] = time.perf_counter_ns() - started
            matched = responses["python"] == responses["rust"]
            ratio = elapsed["python"] / elapsed["rust"]
            paired.append(
                {
                    "batch_size": size,
                    "block": block,
                    "order": list(order),
                    "python_elapsed_ns": elapsed["python"],
                    "rust_elapsed_ns": elapsed["rust"],
                    "python_over_rust_speedup": ratio,
                    "outputs_matched": matched,
                    "complete": True,
                    "censored": False,
                }
            )
            for arm in ("python", "rust"):
                rows.append(
                    {
                        "unit_id": f"size-{size}-block-{block:02d}",
                        "batch_size": size,
                        "block": block,
                        "arm": arm,
                        "order": list(order),
                        "elapsed_ns": elapsed[arm],
                        "requests_per_second": size * 1_000_000_000 / elapsed[arm],
                        "failures": 0,
                        "abstentions": 0,
                        "censored": False,
                        "output_sha256": sha256_json(responses[arm]),
                    }
                )
            print(
                f"[exp7326] phase=benchmark event=unit_complete size={size} "
                f"block={block + 1}/{PAIRED_BLOCKS}",
                flush=True,
            )
        assert size_index < len(BATCH_SIZES)
    return paired, rows, serialization


def _mark_blocked(artifact: JsonDict) -> JsonDict:
    """Convert the first unchanged external failure into a terminal block."""

    failure = artifact["gate_check_summary"]["first_failure"]
    artifact.update(
        {
            "status": "blocked",
            "completed_at_utc": datetime.now(UTC).isoformat(),
            "verdict_class": "blocked",
            "honest_verdict": (
                f"blocked_upstream: {failure['upstream']} check {failure['check']} field "
                f"{failure['field']} expected {failure['expected_value']!r}; "
                f"observed {failure['observed_value']!r}"
            ),
            "constraint_kernel_complete_score": 0,
            "kernel_speedup_score": 0,
        }
    )
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def run_experiment(repo_root: Path) -> JsonDict:  # pragma: no cover - entrypoint integration.
    """Authenticate, measure, validate, and return one terminal artifact."""

    origin = time.monotonic()
    spans: list[JsonDict] = []
    print("[exp7326] phase=preflight event=start", flush=True)
    phase_start = time.monotonic()
    upstream_path = repo_root / UPSTREAM_PATH
    checks, hashes, raw_path = collect_preconditions(upstream_path)
    hashes["current_sources"] = _source_hashes(repo_root)
    artifact = base_artifact(checks, hashes)
    upstream = _read_object(upstream_path) if upstream_path.is_file() else {}
    if isinstance(upstream.get("repository_health"), Mapping):
        artifact["repository_health"] = deepcopy(upstream["repository_health"])
    _phase(spans, "preflight", phase_start, origin, len(checks))
    artifact["phase_spans"] = spans
    if not artifact["gate_check_summary"]["passed"]:
        blocked = _mark_blocked(artifact)
        blocked["duration_s"] = time.monotonic() - origin
        blocked["phase_spans"] = spans
        blocked["reproducibility_checksum"] = reproducibility_checksum(blocked)
        return blocked

    raw_dir = repo_root / "results/raw/experiment_7326_v643_constraint_kernel"
    raw_dir.mkdir(parents=True, exist_ok=True)
    print("[exp7326] phase=fixture_recovery event=start", flush=True)
    phase_start = time.monotonic()
    captured_rows = read_jsonl(raw_path)
    records, atom_errors = extract_acquired_records(captured_rows)
    if atom_errors or not records:
        failure = _check(
            str(raw_path),
            "acquired_atom_authentication",
            "new_atoms",
            "nonempty hash-valid records",
            atom_errors if atom_errors else [],
            False,
            "A failed or absent learned value cannot receive a production port.",
        )
        artifact["preconditions_checked"].append(failure)
        artifact["gate_check_summary"] = gate_check_summary(artifact["preconditions_checked"])
        _phase(spans, "fixture_recovery", phase_start, origin, len(captured_rows))
        return _mark_blocked(artifact)
    acquired_path = raw_dir / "acquired_records.json"
    _atomic_json(acquired_path, records)
    by_version = constraints_by_version(records)
    captured = _captured_fixtures(captured_rows, by_version)
    seeded = seeded_fixtures(EVALUATION_SEED, count=SEEDED_FIXTURE_COUNT)
    fixtures = [*captured, *seeded]
    fixture_path = raw_dir / "parity_fixtures.jsonl"
    _atomic_jsonl(fixture_path, fixtures)
    artifact["source_artifact_hashes"].update(
        {
            "acquired_records": sha256_file(acquired_path),
            "parity_fixtures": sha256_file(fixture_path),
        }
    )
    _phase(spans, "fixture_recovery", phase_start, origin, len(fixtures))

    validation_dir = raw_dir / "validation"
    print("[exp7326] phase=rust_build event=start", flush=True)
    phase_start = time.monotonic()
    build = run_commands(
        repo_root,
        [
            CommandSpec(
                "rust_example_build",
                (
                    "cargo",
                    "build",
                    "-p",
                    "carnot-constraints",
                    "--example",
                    "experiment_7326_constraint_kernel",
                ),
                "carnot-constraints 7326 serialized fixture",
            )
        ],
        log_dir=validation_dir / "build",
    )
    artifact["validation_receipts"].extend(build)
    _phase(spans, "rust_build", phase_start, origin, len(build))
    if not build[0]["passed"]:
        artifact.update(
            {
                "status": "complete",
                "verdict_class": "disqualified",
                "honest_verdict": "complete_disqualified: Rust 7326 fixture did not build",
                "completed_at_utc": datetime.now(UTC).isoformat(),
            }
        )
        artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
        return artifact

    print("[exp7326] phase=service_startup event=start", flush=True)
    phase_start = time.monotonic()
    python_service = JsonLineService(
        (
            sys.executable,
            "-u",
            "-m",
            "carnot.experiment_7326_v643_constraint_kernel",
            "--kernel-service",
        ),
        repo_root,
    )
    rust_service = JsonLineService(
        (str(repo_root / "target/debug/examples/experiment_7326_constraint_kernel"),),
        repo_root,
    )
    startup = [
        {"arm": "python", "elapsed_ns": int(python_service.startup_s * 1_000_000_000)},
        {"arm": "rust", "elapsed_ns": int(rust_service.startup_s * 1_000_000_000)},
    ]
    _phase(spans, "service_startup", phase_start, origin, 2)
    try:
        print("[exp7326] phase=parity event=start", flush=True)
        phase_start = time.monotonic()
        parity = _service_parity(fixtures, python_service, rust_service)
        parity_duration = time.monotonic() - phase_start
        parity_path = raw_dir / "parity_rows.jsonl"
        _atomic_jsonl(parity_path, parity)
        artifact["source_artifact_hashes"]["parity_rows"] = sha256_file(parity_path)
        artifact["validation_receipts"].append(
            {
                "name": "python_to_rust_serialized_roundtrip",
                "command": "persistent JSONL Python evaluator -> Rust experiment_7326_constraint_kernel example",
                "scope": "all 2304 captured and 1000 seeded requests with ordered per-term energies",
                "exit_code": 0 if all(row["matched"] for row in parity) else 1,
                "duration_s": parity_duration,
                "log_path": str(parity_path.relative_to(repo_root)),
                "log_sha256": sha256_file(parity_path),
                "passed": all(row["matched"] for row in parity),
                "timed_out": False,
            }
        )
        _phase(spans, "parity", phase_start, origin, len(parity))

        print("[exp7326] phase=benchmark event=start", flush=True)
        phase_start = time.monotonic()
        paired, timing_rows, serialization = _benchmark(fixtures, python_service, rust_service)
        _phase(spans, "benchmark", phase_start, origin, len(paired))
    finally:
        print("[exp7326] phase=service_teardown event=start", flush=True)
        phase_start = time.monotonic()
        python_service.close()
        rust_service.close()
        _phase(spans, "service_teardown", phase_start, origin, 2)

    parity_complete = len(parity) == len(fixtures) and all(row["matched"] for row in parity)
    costs_complete = len(paired) == len(BATCH_SIZES) * PAIRED_BLOCKS and all(
        row["complete"] and row["outputs_matched"] for row in paired
    )
    intervals = {
        str(size): paired_speedup_interval(
            [row["python_over_rust_speedup"] for row in paired if row["batch_size"] == size],
            seed=BOOTSTRAP_SEED + size,
        )
        for size in BATCH_SIZES
    }
    speedup_passed = all(
        row["ci95_lower"] is not None and row["ci95_lower"] >= 10 for row in intervals.values()
    )
    complete_score, speedup_score, verdict = derive_terminal_scores(
        parity_complete=parity_complete,
        costs_complete=costs_complete,
        speedup=speedup_passed,
    )
    artifact.update(
        {
            "status": "complete",
            "completed_at_utc": datetime.now(UTC).isoformat(),
            "rows": timing_rows,
            "sample_size_budget": {
                "captured_planned": CAPTURED_REQUEST_COUNT,
                "captured_attempted": len(captured),
                "captured_complete": len(captured),
                "captured_censored": 0,
                "seeded_planned": SEEDED_FIXTURE_COUNT,
                "seeded_attempted": len(seeded),
                "seeded_complete": len(seeded),
                "seeded_censored": 0,
                "paired_blocks_per_size": PAIRED_BLOCKS,
                "batch_sizes": list(BATCH_SIZES),
                "stopping_rule": "exactly 2304 captured rows, 1000 seeded fixtures, and 30 paired blocks per fixed size",
            },
            "acceptance_gate_results": {
                "parity": _gate(
                    {"rows": CAPTURED_REQUEST_COUNT + SEEDED_FIXTURE_COUNT, "mismatches": 0},
                    {"rows": len(parity), "mismatches": sum(not row["matched"] for row in parity)},
                    parity_complete,
                    "Every decision, error, total, and ordered term energy must agree bit exactly.",
                ),
                "complete_cost_rows": _gate(
                    {"batch_sizes": list(BATCH_SIZES), "paired_blocks_each": PAIRED_BLOCKS},
                    {"paired_rows": len(paired), "all_outputs_matched": costs_complete},
                    costs_complete,
                    "Correctness and paired host costs must both be complete.",
                ),
                "speedup_lower_ci95": _gate(
                    ">=10 at sizes 1, 32, and 256",
                    {size: row["ci95_lower"] for size, row in intervals.items()},
                    speedup_passed,
                    "Only the equivalent persistent service boundary owns the 10x score.",
                ),
            },
            "constraint_kernel_complete_score": complete_score,
            "kernel_speedup_score": speedup_score,
            "verdict_class": verdict,
            "correctness_outcome": (
                "complete_bit_exact_python_rust_parity"
                if parity_complete
                else "failed_python_rust_parity"
            ),
            "performance_outcome": {
                "paired_speedup_intervals": intervals,
                "ten_x_lower_bound_passed": speedup_passed,
                "scope": "persistent newline-delimited host service boundary",
            },
            "kernel_rows": {
                "parity": parity,
                "cost": paired,
                "serialization": serialization,
                "process_startup": startup,
            },
            "hardware_placement": hardware_projection(records, fixtures),
        }
    )
    if verdict == "circular_positive":
        artifact["honest_verdict"] = (
            "complete: acquired integer constraints have bit-exact Rust parity and meet the 10x "
            "paired host service-boundary gate under shared evaluator authority"
        )
    elif verdict == "null":
        artifact["honest_verdict"] = (
            "complete_null: acquired integer constraints have bit-exact Rust parity, but the "
            "predeclared 10x paired host service-boundary gate did not pass"
        )
    else:
        artifact["honest_verdict"] = (
            "complete_disqualified: parity or required cost rows were incomplete"
        )

    print("[exp7326] phase=scoped_validation event=start", flush=True)
    phase_start = time.monotonic()
    scoped_basetemp = prepare_scoped_basetemp(Path("/tmp/carnot-exp7326-scoped"))
    scoped = run_scoped_validation(
        repo_root,
        [TEST_PATH.as_posix()],
        [MODULE_PATH.as_posix()],
        static_paths=[WRAPPER_PATH.as_posix()],
        basetemp=scoped_basetemp,
        coverage_file=raw_dir / ".coverage",
        log_dir=validation_dir / "scoped",
        historical_failures=artifact["repository_health"].get("historical_failures", []),
    )
    artifact.update(
        {
            key: scoped[key]
            for key in (
                "required_checks_passed",
                "missing_required_commands",
                "failed_required_commands",
                "duplicate_required_commands",
                "repository_health",
            )
        }
    )
    artifact["validation_receipts"].extend(scoped["validation_receipts"])
    _phase(spans, "scoped_validation", phase_start, origin, len(REQUIRED_CHECK_NAMES))

    print("[exp7326] phase=rust_validation event=start", flush=True)
    phase_start = time.monotonic()
    rust_receipts = run_commands(
        repo_root,
        [
            CommandSpec(
                "cargo_test_carnot_constraints",
                ("cargo", "test", "-p", "carnot-constraints"),
                "changed Rust crate",
            ),
            CommandSpec(
                "cargo_fmt_carnot_constraints",
                ("cargo", "fmt", "-p", "carnot-constraints", "--", "--check"),
                "changed Rust crate",
            ),
            CommandSpec(
                "cargo_clippy_carnot_constraints",
                (
                    "cargo",
                    "clippy",
                    "-p",
                    "carnot-constraints",
                    "--all-targets",
                    "--no-deps",
                    "--",
                    "-D",
                    "warnings",
                ),
                "changed Rust crate",
            ),
        ],
        log_dir=validation_dir / "rust",
    )
    artifact["validation_receipts"].extend(rust_receipts)
    _phase(spans, "rust_validation", phase_start, origin, len(rust_receipts))
    affected_passed = artifact["required_checks_passed"] and all(
        row["passed"] for row in rust_receipts
    )
    if not affected_passed:
        failed = [row["name"] for row in rust_receipts if not row["passed"]]
        failed.extend(artifact["failed_required_commands"] or artifact["missing_required_commands"])
        artifact.update(
            {
                "constraint_kernel_complete_score": 0,
                "kernel_speedup_score": 0,
                "verdict_class": "disqualified",
                "honest_verdict": "complete_disqualified: affected validation failed: "
                + ",".join(failed),
            }
        )
    artifact["phase_spans"] = spans
    artifact["duration_s"] = time.monotonic() - origin
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def _terminal_validators(
    repo_root: Path, candidate: Path, log_dir: Path
) -> list[JsonDict]:  # pragma: no cover
    """Run both required artifact validators against the measured candidate."""

    return run_commands(
        repo_root,
        [
            CommandSpec(
                "adversarial_verify",
                (
                    str(repo_root / ".venv/bin/python"),
                    "-u",
                    "scripts/adversarial_verify.py",
                    str(candidate),
                ),
                "measured terminal candidate",
            ),
            CommandSpec(
                "verdict_row_consistency_strict",
                (
                    str(repo_root / ".venv/bin/python"),
                    "-u",
                    "scripts/verdict_row_consistency_lint.py",
                    "--strict",
                    str(candidate),
                ),
                "measured terminal candidate",
            ),
        ],
        log_dir=log_dir,
    )


def _serve() -> int:  # pragma: no cover - exercised through the serialized subprocess.
    """Serve ordered JSON requests until an explicit shutdown line arrives."""

    for line in sys.stdin:
        try:
            message = json.loads(line)
            response = service_response(message)
        except (TypeError, ValueError, json.JSONDecodeError):
            response = {"error": "invalid_request"}
        print(json.dumps(response, sort_keys=True, separators=(",", ":")), flush=True)
        if response == {"kind": "shutdown"}:
            break
    return 0


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:  # pragma: no cover
    """Parse the fixed execution date, cold validation, and service modes."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", choices=[RUN_DATE])
    parser.add_argument("--validate", type=Path)
    parser.add_argument("--kernel-service", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover
    """Run the bounded measurement and atomically publish only terminal evidence."""

    args = _parse_args(argv)
    if args.kernel_service:
        return _serve()
    print("[exp7326] phase=startup event=start", flush=True)
    if args.validate is not None:
        errors = validate_artifact(_read_object(args.validate))
        print(json.dumps({"errors": errors}, sort_keys=True), flush=True)
        return int(bool(errors))
    if args.date != RUN_DATE:
        raise SystemExit("--date 20260915 is required")

    artifact_path = REPO_ROOT / "results/experiment_7326_v643_constraint_kernel.json"
    raw_dir = REPO_ROOT / "results/raw/experiment_7326_v643_constraint_kernel"
    artifact = run_experiment(REPO_ROOT)
    if artifact["status"] == "blocked":
        write_artifact(artifact_path, artifact)
        print(f"[exp7326] phase=terminal_write event=end path={artifact_path}", flush=True)
        return 0

    candidate = raw_dir / "terminal_candidate.json"
    _atomic_json(candidate, artifact)
    print("[exp7326] phase=terminal_validators event=start", flush=True)
    validator_start_offset = float(artifact["duration_s"])
    validator_started = time.monotonic()
    validators = _terminal_validators(REPO_ROOT, candidate, raw_dir / "validation/terminal")
    validator_elapsed = time.monotonic() - validator_started
    artifact["validation_receipts"].extend(validators)
    if not all(row["passed"] for row in validators):
        artifact.update(
            {
                "constraint_kernel_complete_score": 0,
                "kernel_speedup_score": 0,
                "verdict_class": "disqualified",
                "honest_verdict": "complete_disqualified: terminal artifact validation failed",
            }
        )
    artifact["phase_spans"].append(
        {
            "phase": "terminal_validators",
            "start_s": validator_start_offset,
            "end_s": validator_start_offset + validator_elapsed,
            "units": len(validators),
            "checkpoint_boundaries": len(validators),
            "pending_operations": [],
        }
    )
    artifact["duration_s"] = validator_start_offset + validator_elapsed
    artifact["completed_at_utc"] = datetime.now(UTC).isoformat()
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    write_artifact(artifact_path, artifact)
    print(
        f"[exp7326] phase=terminal_write event=end path={artifact_path} verdict={artifact['verdict_class']}",
        flush=True,
    )
    return int(artifact["verdict_class"] == "disqualified")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
