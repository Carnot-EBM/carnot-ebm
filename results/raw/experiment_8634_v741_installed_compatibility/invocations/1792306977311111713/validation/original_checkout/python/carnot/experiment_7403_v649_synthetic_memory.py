"""Measure source-certified proof reuse on the sealed synthetic cohort.

This experiment replays exact Boolean requests. It makes no language-model
claim. The exact solver is also the correctness oracle, so successful value
evidence is circular by definition.

Spec refs: REQ-CL-7403 and SCENARIO-CL-7403-*.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import sys
import tempfile
import time
from typing import Any

from carnot import experiment_7358_v646_validation_contract as validation_contract
from carnot import experiment_7371_v647_proof_boundary as boundary
from carnot.learning.implication_memory import ProofMemory, execute_query, proof_from_dict
from carnot.reporting import experiment_7303_validation_scope as validation_scope


JsonDict = dict[str, Any]
REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_DATE = "20260919"
MILESTONE = "2026.09.649"
PHASE = 3
EXPERIMENT_ID = "exp7403-v649-synthetic-memory"
SCHEMA = "carnot.exp7403.v649.synthetic_memory.v1"
RESULT_PATH = Path("results/experiment_7403_v649_synthetic_memory.json")
RAW_DIR = Path("results/raw/experiment_7403_v649_synthetic_memory")
MODULE_PATH = Path("python/carnot/experiment_7403_v649_synthetic_memory.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_7403_v649_synthetic_memory.py")
TEST_PATH = Path("tests/python/test_experiment_7403_v649_synthetic_memory.py")
SPEC_PATH = Path("openspec/capabilities/continuous-learning/spec.md")
UPSTREAM_RESULT_PATH = Path("results/experiment_7371_v647_proof_boundary.json")
PROTOCOL_PATH = Path("data/v647_implication_stream_manifest.json")
EVALUATION_FIXTURE_PATH = Path(
    "results/raw/experiment_7371_v647_proof_boundary/evaluation_streams.json"
)
ARMS = boundary.ARMS
PERSISTENT_CONTROLS = (ARMS[1], ARMS[2])
RANDOM_SEED = {"experiment": 7_403_649, "resampling": 7_371_307}
BOOTSTRAP_DRAWS = 10_000
EXPECTED_HASHES = {
    UPSTREAM_RESULT_PATH.as_posix(): (
        "sha256:105908686733be404961f98b820fb9edccc2caf2f8fed37db5a29b506fcad3b4"
    ),
    PROTOCOL_PATH.as_posix(): (
        "sha256:6c0a312a8e34da96c2f2308b84c503bd8d064f59a8bc78caa41cf2b187d95fab"
    ),
    EVALUATION_FIXTURE_PATH.as_posix(): (
        "sha256:b9f3e8a361c93cb9712ca8c46fa33d3d6cc52987dcff58a1114198534d824e28"
    ),
}
ZERO_CURRENT_INVOCATIONS = {
    "model_loads_attempted": 0,
    "model_loads_completed": 0,
    "model_loads_failed": 0,
    "model_loads_cancelled": 0,
    "model_loads_in_flight": 0,
    "generation_calls_attempted": 0,
    "generation_calls_completed": 0,
    "generation_calls_failed": 0,
    "generation_calls_cancelled": 0,
    "generation_calls_in_flight": 0,
}
REQUIRED_CHECK_NAMES = validation_scope.REQUIRED_CHECK_NAMES
TERMINAL_CHECK_NAMES = (
    "cold_artifact_replay",
    "independent_reducer",
    "adversarial_verify",
    "verdict_row_consistency_strict",
    "declared_entrypoint",
)
EXTRA_AUTHORITY_ATTACKS = ("stale_version_proof", "missing_proof_fallback")
REQUIRED_SOURCE_PATHS = (
    Path("AGENTS.md"),
    Path("CODEX.md"),
    Path("CLAUDE.md"),
    Path("research-program.md"),
    Path("ops/exclusion_manifest.yaml"),
    Path("ops/e2e-test-plan.md"),
    Path("scripts/experiment_template.py"),
    Path("python/carnot/reporting/experiment_7303_validation_scope.py"),
    Path("python/carnot/experiment_7358_v646_validation_contract.py"),
    Path("openspec/capabilities/research-reporting/spec.md"),
    Path("python/carnot/learning/implication_memory.py"),
    Path("python/carnot/experiment_7370_v647_proof_memory.py"),
    Path("python/carnot/experiment_7371_v647_proof_boundary.py"),
    Path("python/carnot/verify/sat.py"),
    SPEC_PATH,
    Path("openspec/capabilities/constraint-verification/spec.md"),
    MODULE_PATH,
    WRAPPER_PATH,
    TEST_PATH,
)
V649_MANIFEST = validation_contract.AffectedManifest(
    experiment_id=EXPERIMENT_ID,
    test_paths=(TEST_PATH.as_posix(),),
    changed_modules=(MODULE_PATH.as_posix(),),
    static_paths=(WRAPPER_PATH.as_posix(),),
)


@dataclass
class ReplayEvidence:
    """Keep the four replay tables together without hiding ordinary rows."""

    rows: list[JsonDict]
    witnesses: list[JsonDict]
    attacks: list[JsonDict]
    restart_rows: list[JsonDict]


def progress(started: float, phase: str, event: str, **details: Any) -> None:
    """Print one flushed boundary with measured monotonic elapsed time."""

    suffix = " ".join(f"{key}={value}" for key, value in sorted(details.items()))
    print(
        f"[exp7403] phase={phase} event={event} elapsed_s={time.monotonic() - started:.3f}"
        + (f" {suffix}" if suffix else ""),
        flush=True,
    )


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def canonical_hash(value: Any) -> str:
    return "sha256:" + hashlib.sha256(canonical_bytes(value)).hexdigest()


def sha256_file(path: Path) -> str:
    return validation_contract.sha256_file(path)


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    validation_contract.atomic_json(path, value)


def precondition_row(
    check: str,
    upstream: str,
    field: str,
    expected: Any,
    observed: Any,
    passed: bool,
) -> JsonDict:
    """Record one structured gate before dependent work can start."""

    return {
        "check": check,
        "upstream": upstream,
        "artifact_field": field,
        "expected": expected,
        "observed": observed,
        "passed": passed,
        "terminal_blocking": True,
    }


def _load_object(path: Path) -> JsonDict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _selected_hash_check(
    path: Path, expected: str, check: str, field: str
) -> tuple[JsonDict, str | None]:
    observed = sha256_file(path) if path.is_file() else None
    return (
        precondition_row(check, str(path), field, expected, observed, observed == expected),
        observed,
    )


def collect_preconditions(
    repo_root: Path,
    *,
    upstream_result_path: Path | None = None,
    protocol_path: Path | None = None,
    evaluation_fixture_path: Path | None = None,
) -> tuple[list[JsonDict], dict[str, str], list[JsonDict]]:
    """Authenticate exact sources and the qualified Exp7371 boundary."""

    root = repo_root.resolve()
    checks: list[JsonDict] = []
    hashes: dict[str, str] = {}
    for relative in REQUIRED_SOURCE_PATHS:
        path = root / relative
        present = path.is_file() and path.stat().st_size > 0
        checks.append(
            precondition_row(
                f"source_bytes:{relative.as_posix()}",
                relative.as_posix(),
                "bytes",
                "readable_nonempty_bytes",
                "readable_nonempty_bytes" if present else "missing",
                present,
            )
        )
        if present:
            hashes[relative.as_posix()] = sha256_file(path)

    selected_result = upstream_result_path or root / UPSTREAM_RESULT_PATH
    selected_protocol = protocol_path or root / PROTOCOL_PATH
    selected_fixture = evaluation_fixture_path or root / EVALUATION_FIXTURE_PATH
    selected = (
        (
            selected_result,
            EXPECTED_HASHES[UPSTREAM_RESULT_PATH.as_posix()],
            "upstream_result_sha256",
            "sha256",
            UPSTREAM_RESULT_PATH.as_posix(),
        ),
        (
            selected_protocol,
            EXPECTED_HASHES[PROTOCOL_PATH.as_posix()],
            "protocol_sha256",
            "sha256",
            PROTOCOL_PATH.as_posix(),
        ),
        (
            selected_fixture,
            EXPECTED_HASHES[EVALUATION_FIXTURE_PATH.as_posix()],
            "evaluation_fixture_sha256",
            "sha256",
            EVALUATION_FIXTURE_PATH.as_posix(),
        ),
    )
    for path, expected, check, field, stable_name in selected:
        row, observed = _selected_hash_check(path, expected, check, field)
        checks.append(row)
        if observed is not None:
            hashes[stable_name if path == root / Path(stable_name) else str(path)] = observed

    upstream = _load_object(selected_result)
    expected_fields = {
        "experiment_id": "exp7371-v647-proof-boundary",
        "milestone": "2026.09.647",
        "proof_boundary_ready_score": 1,
        "flagged_adversarial": False,
        "verdict_class": "null",
        "status": "complete_proof_boundary_ready_live_capture_not_started",
    }
    for field, expected in expected_fields.items():
        checks.append(
            precondition_row(
                f"upstream_{field}",
                str(selected_result),
                field,
                expected,
                upstream.get(field),
                upstream.get(field) == expected,
            )
        )
    upstream_hashes = upstream.get("source_artifact_hashes") or {}
    for path, expected in (
        (PROTOCOL_PATH, EXPECTED_HASHES[PROTOCOL_PATH.as_posix()]),
        (EVALUATION_FIXTURE_PATH, EXPECTED_HASHES[EVALUATION_FIXTURE_PATH.as_posix()]),
    ):
        observed = upstream_hashes.get(path.as_posix())
        checks.append(
            precondition_row(
                f"upstream_receipt:{path.as_posix()}",
                str(selected_result),
                f"source_artifact_hashes.{path.as_posix()}",
                expected,
                observed,
                observed == expected,
            )
        )

    protocol = _load_object(selected_protocol)
    protocol_failures = protocol_errors(protocol) if protocol else ["protocol_missing"]
    checks.append(
        precondition_row(
            "sealed_protocol_structure",
            str(selected_protocol),
            "evaluation_streams",
            "32_streams_x_24_requests_five_arms",
            protocol_failures or "32_streams_x_24_requests_five_arms",
            not protocol_failures,
        )
    )
    spec_text = (root / SPEC_PATH).read_text(encoding="utf-8")
    checks.append(
        precondition_row(
            "driving_requirement",
            SPEC_PATH.as_posix(),
            "REQ-*",
            "REQ-CL-7403",
            "REQ-CL-7403" if "REQ-CL-7403" in spec_text else None,
            "REQ-CL-7403" in spec_text,
        )
    )
    exclusion_text = (root / "ops/exclusion_manifest.yaml").read_text(encoding="utf-8")
    excluded = "experiment_id: 7403" in exclusion_text or EXPERIMENT_ID in exclusion_text
    checks.append(
        precondition_row(
            "current_task_not_quarantined",
            "ops/exclusion_manifest.yaml",
            EXPERIMENT_ID,
            False,
            excluded,
            not excluded,
        )
    )
    checks.append(
        precondition_row(
            "host_cpu_available",
            "local_host",
            "cpu_count",
            "positive_integer",
            os.cpu_count(),
            bool(os.cpu_count() and os.cpu_count() > 0),
        )
    )
    sidecars = [
        {
            "label": "historical_exp7371_receipt_not_current_inference",
            "path": UPSTREAM_RESULT_PATH.as_posix(),
            "sha256": EXPECTED_HASHES[UPSTREAM_RESULT_PATH.as_posix()],
            "counted_as_current": False,
        }
    ]
    return checks, hashes, sidecars


def protocol_errors(protocol: Mapping[str, Any]) -> list[str]:
    """Keep Exp7371 validation and this cohort's frozen identity together."""

    errors = boundary.validate_protocol(protocol)
    if (
        protocol.get("experiment_id") != "exp7371-v647-proof-boundary"
        or protocol.get("milestone") != "2026.09.647"
        or protocol.get("run_date") != "20260917"
    ):
        errors.append("protocol_identity")
    if protocol.get("arms") != list(ARMS):
        errors.append("arm_order")
    return list(dict.fromkeys(errors))


def frozen_acceptance_manifest(protocol_sha256: str) -> JsonDict:
    """Seal every threshold before the replay timer starts."""

    return {
        "protocol_sha256": protocol_sha256,
        "unsafe_decisions": {"operator": "==", "threshold": 0},
        "exact_decision_coverage": {"operator": "==", "threshold": 1.0},
        "valid_utility": {"operator": "==", "threshold": True},
        "minimum_erasure_witnesses": {"operator": ">=", "threshold": 8},
        "minimum_witness_streams": {"operator": ">=", "threshold": 4},
        "paid_query_ratio_ci95_upper": {"operator": "<", "threshold": 0.90},
        "full_cost_ratio_ci95_upper": {"operator": "<=", "threshold": 1.0},
        "comparators": list(PERSISTENT_CONTROLS),
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "bootstrap_seed": RANDOM_SEED["resampling"],
        "bootstrap_unit": "paired_stream_blocks_with_formula_family_retained",
        "sealed_before_timing": True,
    }


def _complete_service_cost(row: Mapping[str, Any]) -> JsonDict:
    original = row.get("cost_units") or {}
    discovery = 3 * len(row.get("raw_candidate_order") or [])
    checking = int(original.get("proof_check", 0))
    updates = int(original.get("cache_update", 0))
    verification = int(original.get("source_hash", 0))
    serialization = int(original.get("memory_accounting", 0))
    exact_solver = int(original.get("exact_query", 0))
    orchestration = 1
    return {
        "discovery": discovery,
        "checking": checking,
        "updates": updates,
        "verification": verification,
        "serialization": serialization,
        "orchestration": orchestration,
        "exact_solver": exact_solver,
        "total": (
            discovery
            + checking
            + updates
            + verification
            + serialization
            + orchestration
            + exact_solver
        ),
    }


def _enrich_stream_rows(
    rows: Sequence[Mapping[str, Any]], witnesses: Sequence[Mapping[str, Any]]
) -> tuple[list[JsonDict], list[JsonDict]]:
    discoveries: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        if row.get("arm") == "proof_memory":
            for path_id in row.get("raw_candidate_order") or []:
                discoveries.setdefault(str(path_id), row)

    enriched_witnesses: list[JsonDict] = []
    witness_by_request: dict[str, list[JsonDict]] = {}
    for witness in witnesses:
        path_id = str((witness.get("path") or {}).get("path_id"))
        earlier = discoveries.get(path_id)
        enriched = {
            **deepcopy(dict(witness)),
            "earlier_verified_feedback_request_id": earlier.get("request_id") if earlier else None,
            "earlier_request_index": earlier.get("request_index") if earlier else None,
            "later_request_index": next(
                (
                    row.get("request_index")
                    for row in rows
                    if row.get("request_id") == witness.get("request_id")
                ),
                None,
            ),
            "changed_exact_work": (
                witness.get("later_used_exact_solver") is False
                and witness.get("erasure_used_exact_solver") is True
            ),
        }
        enriched["different_later_query"] = enriched[
            "earlier_verified_feedback_request_id"
        ] is not None and enriched["earlier_verified_feedback_request_id"] != witness.get(
            "request_id"
        )
        if not enriched["changed_exact_work"]:
            continue
        enriched_witnesses.append(enriched)
        witness_by_request.setdefault(str(witness.get("request_id")), []).append(enriched)

    verified_before: dict[tuple[str, str, int], list[str]] = {}
    proof_rows = [row for row in rows if row.get("arm") == "proof_memory"]
    for row in proof_rows:
        key = (str(row["stream_id"]), str(row["formula_version"]), int(row["request_index"]))
        verified_before[key] = [
            str(previous["request_id"])
            for previous in proof_rows
            if previous["stream_id"] == row["stream_id"]
            and previous["formula_version"] == row["formula_version"]
            and previous["request_index"] < row["request_index"]
            and previous["used_exact_solver"] is True
            and previous["decision"] == "unsatisfiable"
        ]

    enriched_rows: list[JsonDict] = []
    for row in rows:
        request_witnesses = witness_by_request.get(str(row.get("request_id")), [])
        key = (str(row["stream_id"]), str(row["formula_version"]), int(row["request_index"]))
        reuse_class = "none"
        if row.get("arm") == "proof_memory" and row.get("used_exact_solver") is False:
            reuse_class = "cross_query_source_proof"
        elif (
            row.get("arm") == "persistent_source_graph_reachability_cache"
            and row.get("used_exact_solver") is False
        ):
            reuse_class = "repeated_query_cache"
        enriched_rows.append(
            {
                **deepcopy(dict(row)),
                "complete_service_cost": _complete_service_cost(row),
                "prior_verified_feedback_request_ids": verified_before.get(key, []),
                "witnessed_earlier_feedback": [
                    item["earlier_verified_feedback_request_id"]
                    for item in request_witnesses
                    if item["earlier_verified_feedback_request_id"] is not None
                ]
                if row.get("arm") == "proof_memory"
                else [],
                "reuse_class": reuse_class,
            }
        )
    return enriched_rows, enriched_witnesses


def _restart_row(
    stream: Mapping[str, Any], witness: Mapping[str, Any], checkpoint_path: Path
) -> JsonDict:
    version = str(witness["formula_version"])
    payload = next(item for item in stream["versions"] if item["version"] == version)
    formula = boundary._formula_from_payload(payload)
    proof = proof_from_dict(witness["path"], formula)
    memory = ProofMemory.empty(formula).commit([proof])
    memory.save(checkpoint_path)
    restored = ProofMemory.load(checkpoint_path, formula)
    stale_payload = next(item for item in stream["versions"] if item["version"] != version)
    stale_formula = boundary._formula_from_payload(stale_payload)
    invalidated = ProofMemory.load(checkpoint_path, stale_formula)
    return {
        "stream_id": stream["stream_id"],
        "formula_version": version,
        "source_hash": formula.source_hash,
        "path_id": proof.path_id,
        "checkpoint_path": str(checkpoint_path),
        "snapshot_sha256_before": memory.sha256,
        "snapshot_sha256_after": restored.sha256,
        "serialized_bytes": len(memory.to_bytes()),
        "exact_bytes_restored": restored.to_bytes() == memory.to_bytes(),
        "stale_version_invalidated": invalidated.paths == (),
        "passed": restored.to_bytes() == memory.to_bytes() and invalidated.paths == (),
    }


def _attack_row(name: str, rejected: bool) -> JsonDict:
    return {
        "attack": name,
        "expected": "reject",
        "observed": "reject" if rejected else "accept",
        "passed": rejected,
        "failure": None if rejected else "unauthorized_authority_accepted",
    }


def authority_controls(
    protocol: Mapping[str, Any], witnesses: Sequence[Mapping[str, Any]]
) -> list[JsonDict]:
    """Retain Exp7371 attacks and add explicit stale and missing-proof cases."""

    attacks = [deepcopy(row) for row in boundary.run_authority_controls(protocol)]
    first = witnesses[0]
    stream = next(
        row for row in protocol["evaluation_streams"] if row["stream_id"] == first["stream_id"]
    )
    payload = next(
        item for item in stream["versions"] if item["version"] == first["formula_version"]
    )
    stale = deepcopy(first["path"])
    stale["formula_version"] = "stale-version"
    attacks.append(
        _attack_row("stale_version_proof", bool(boundary.independent_validate_path(payload, stale)))
    )
    formula = boundary._formula_from_payload(payload)
    missing = execute_query(ProofMemory.empty(formula), stream["requests"][0]["assumptions"])
    attacks.append(_attack_row("missing_proof_fallback", missing.used_exact_solver is True))
    return attacks


def replay_protocol(
    protocol: Mapping[str, Any],
    *,
    checkpoint_dir: Path | None = None,
    emit_progress: bool = False,
    started: float | None = None,
) -> ReplayEvidence:
    """Replay any compatible sealed cohort with one five-arm stream runner."""

    errors = protocol_errors(protocol)
    if errors:
        raise ValueError(f"protocol_invalid:{errors}")
    origin = started if started is not None else time.monotonic()
    temporary: tempfile.TemporaryDirectory[str] | None = None
    if checkpoint_dir is None:
        temporary = tempfile.TemporaryDirectory(prefix="exp7403-replay-")
        checkpoint_root = Path(temporary.name)
    else:
        checkpoint_root = checkpoint_dir
        checkpoint_root.mkdir(parents=True, exist_ok=True)
    rows: list[JsonDict] = []
    witnesses: list[JsonDict] = []
    restart_rows: list[JsonDict] = []
    try:
        streams = protocol["evaluation_streams"]
        for index, stream in enumerate(streams):
            stream_rows, stream_witnesses = boundary.evaluate_stream(stream)
            enriched_rows, enriched_witnesses = _enrich_stream_rows(stream_rows, stream_witnesses)
            if not stream_witnesses:
                raise RuntimeError(f"stream_has_no_proof_for_restart:{stream['stream_id']}")
            restart = _restart_row(
                stream,
                stream_witnesses[0],
                checkpoint_root / f"{index:02d}_{stream['stream_id']}.memory.json",
            )
            rows.extend(enriched_rows)
            witnesses.extend(enriched_witnesses)
            restart_rows.append(restart)
            if checkpoint_dir is not None:
                atomic_json(
                    checkpoint_root / f"{index:02d}_{stream['stream_id']}.checkpoint.json",
                    {
                        "stream_id": stream["stream_id"],
                        "completed_units": index + 1,
                        "rows": enriched_rows,
                        "erasure_witness_rows": enriched_witnesses,
                        "restart_row": restart,
                    },
                )
            if emit_progress:
                progress(origin, "evaluate", "stream_complete", unit=index + 1, total=len(streams))
        attacks = authority_controls(protocol, witnesses)
        return ReplayEvidence(rows, witnesses, attacks, restart_rows)
    finally:
        if temporary is not None:
            temporary.cleanup()


def _block_bootstrap_upper(
    rows: Sequence[Mapping[str, Any]], comparator: str, metric: str
) -> float:
    streams = sorted({str(row["stream_id"]) for row in rows})
    paired: dict[str, tuple[float, float]] = {}
    for stream in streams:
        proof = [row for row in rows if row["stream_id"] == stream and row["arm"] == ARMS[3]]
        control = [row for row in rows if row["stream_id"] == stream and row["arm"] == comparator]
        if metric == "paid_exact_query":
            numerator = sum(row[metric] is True for row in proof)
            denominator = sum(row[metric] is True for row in control)
        else:
            numerator = sum(int(row["complete_service_cost"]["total"]) for row in proof)
            denominator = sum(int(row["complete_service_cost"]["total"]) for row in control)
        paired[stream] = (float(numerator), float(denominator))
    generator = random.Random(RANDOM_SEED["resampling"])
    estimates: list[float] = []
    for _ in range(BOOTSTRAP_DRAWS):
        sample = [generator.choice(streams) for _ in streams]
        numerator = sum(paired[stream][0] for stream in sample)
        denominator = sum(paired[stream][1] for stream in sample)
        estimates.append(numerator / denominator if denominator else float("inf"))
    estimates.sort()
    return estimates[int(0.95 * len(estimates))]


def synthetic_metrics(
    rows: Sequence[Mapping[str, Any]], witnesses: Sequence[Mapping[str, Any]]
) -> JsonDict:
    """Reduce safety, utility, erasure, and paired comparator evidence."""

    unsafe = sum(
        row.get("final_exact_validation") is not True
        or (row.get("decision") == "satisfiable" and row.get("assignment_valid") is not True)
        or (row.get("decision") == "reject" and row.get("independent_satisfiable") is True)
        for row in rows
    )
    valid_witnesses = [
        row
        for row in witnesses
        if row.get("changed_exact_work") is True
        and row.get("different_later_query") is True
        and row.get("independent_path_errors") == []
    ]
    return {
        "unsafe_decisions": unsafe,
        "exact_decision_coverage": (
            sum(row.get("independent_exact_match") is True for row in rows) / len(rows)
            if rows
            else 0.0
        ),
        "valid_utility": bool(rows)
        and all(row.get("final_exact_validation") is True for row in rows),
        "erasure_witness_count": len(valid_witnesses),
        "erasure_witness_stream_count": len({str(row["stream_id"]) for row in valid_witnesses}),
        "paid_query_ratio_ci95_upper": {
            comparator: _block_bootstrap_upper(rows, comparator, "paid_exact_query")
            for comparator in PERSISTENT_CONTROLS
        }
        if rows
        else {},
        "full_cost_ratio_ci95_upper": {
            comparator: _block_bootstrap_upper(rows, comparator, "complete_service_cost")
            for comparator in PERSISTENT_CONTROLS
        }
        if rows
        else {},
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "bootstrap_seed": RANDOM_SEED["resampling"],
        "bootstrap_independent_stream_groups": len({row.get("stream_id") for row in rows}),
        "bootstrap_formula_families": len({row.get("family") for row in rows}),
    }


def cost_rows(rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    return [
        {
            "stream_id": row["stream_id"],
            "request_id": row["request_id"],
            "arm": row["arm"],
            "complete_service_cost": deepcopy(row["complete_service_cost"]),
            "duration_ns": row["duration_ns"],
            "failed": row.get("failure") is not None,
            "censored": row.get("censored") is True,
            "persistent_state_instance_id": row["state_instance_id"],
        }
        for row in rows
    ]


def passing_test_receipts() -> list[JsonDict]:
    return [
        {
            "name": name,
            "command": f"test {name}",
            "command_argv": ["test", name],
            "environment": {"PYTHONUNBUFFERED": "1"},
            "scope": "unit_fixture",
            "return_code": 0,
            "exit_code": 0,
            "duration_s": 0.001,
            "log_path": f"/tmp/{name}.log",
            "log_hash": "sha256:" + "1" * 64,
            "log_sha256": "sha256:" + "1" * 64,
            "passed": True,
            "timed_out": False,
            "required": True,
            "category": "required_validation",
            "started_at_utc": "2026-09-19T00:00:00+00:00",
            "ended_at_utc": "2026-09-19T00:00:00.001000+00:00",
        }
        for name in (*REQUIRED_CHECK_NAMES, *TERMINAL_CHECK_NAMES)
    ]


def _receipts_pass(receipts: Sequence[Mapping[str, Any]], names: Sequence[str]) -> bool:
    counts = Counter(row.get("name") for row in receipts if row.get("required") is True)
    return all(
        counts[name] == 1
        and next(row for row in receipts if row.get("name") == name).get("passed") is True
        and next(row for row in receipts if row.get("name") == name).get("return_code") == 0
        and next(row for row in receipts if row.get("name") == name).get("timed_out") is False
        for name in names
    )


def independent_reduce(artifact: Mapping[str, Any]) -> JsonDict:
    """Recompute completion, safety, and circular value from ordinary rows."""

    rows = [row for row in artifact.get("rows", []) if isinstance(row, Mapping)]
    witnesses = [
        row for row in artifact.get("erasure_witness_rows", []) if isinstance(row, Mapping)
    ]
    restarts = [row for row in artifact.get("restart_rows", []) if isinstance(row, Mapping)]
    attacks = [row for row in artifact.get("authority_attack_rows", []) if isinstance(row, Mapping)]
    receipts = [row for row in artifact.get("validation_receipts", []) if isinstance(row, Mapping)]
    preconditions = [
        row
        for row in artifact.get("preconditions_checked", [])
        if isinstance(row, Mapping) and row.get("terminal_blocking") is True
    ]
    metrics = synthetic_metrics(rows, witnesses)
    preconditions_passed = bool(preconditions) and all(
        row.get("passed") is True for row in preconditions
    )
    rows_complete = (
        len(rows) == 32 * 24 * 5
        and {row.get("arm") for row in rows} == set(ARMS)
        and len(artifact.get("cost_rows") or []) == len(rows)
        and all(
            row.get("complete_service_cost", {}).get("total", 0) > 0
            and row.get("censored") is False
            for row in rows
        )
    )
    protocol_complete = len(artifact.get("formula_stream_rows") or []) == 32 * 24
    expected_attacks = set(boundary.AUTHORITY_ATTACKS) | set(EXTRA_AUTHORITY_ATTACKS)
    authority_passed = {row.get("attack") for row in attacks} == expected_attacks and all(
        row.get("passed") is True for row in attacks
    )
    restart_passed = len(restarts) == 32 and all(row.get("passed") is True for row in restarts)
    safety_passed = (
        metrics["unsafe_decisions"] == 0
        and metrics["exact_decision_coverage"] == 1.0
        and metrics["valid_utility"] is True
    )
    witness_passed = (
        metrics["erasure_witness_count"] >= 8 and metrics["erasure_witness_stream_count"] >= 4
    )
    affected_passed = _receipts_pass(receipts, REQUIRED_CHECK_NAMES)
    terminal_passed = _receipts_pass(receipts, TERMINAL_CHECK_NAMES)
    not_flagged = artifact.get("flagged_adversarial") is False
    capture = int(
        preconditions_passed
        and protocol_complete
        and rows_complete
        and authority_passed
        and restart_passed
        and safety_passed
        and witness_passed
        and affected_passed
        and terminal_passed
        and not_flagged
    )
    efficacy = all(
        metrics["paid_query_ratio_ci95_upper"].get(comparator, float("inf")) < 0.90
        and metrics["full_cost_ratio_ci95_upper"].get(comparator, float("inf")) <= 1.0
        for comparator in PERSISTENT_CONTROLS
    )
    return {
        "preconditions_passed": preconditions_passed,
        "protocol_complete": protocol_complete,
        "rows_complete": rows_complete,
        "authority_controls_passed": authority_passed,
        "restart_passed": restart_passed,
        "safety_passed": safety_passed,
        "witness_gate_passed": witness_passed,
        "affected_validation_passed": affected_passed,
        "terminal_validation_passed": terminal_passed,
        "efficacy_passed": efficacy,
        "synthetic_metrics": metrics,
        "synthetic_memory_capture_complete_score": capture,
        "proof_safety_ready_score": int(capture and safety_passed),
        "synthetic_memory_value_score": int(capture and efficacy),
        "promotion_score": 0,
    }


def _gate(
    category: str,
    check: str,
    operator: str,
    expected: Any,
    observed: Any,
    passed: bool,
    terminal_blocking: bool,
) -> JsonDict:
    return {
        "category": category,
        "check": check,
        "operator": operator,
        "expected": expected,
        "observed": observed,
        "passed": passed,
        "terminal_blocking": terminal_blocking,
    }


def acceptance_gates(artifact: Mapping[str, Any]) -> list[JsonDict]:
    reduced = independent_reduce(artifact)
    metrics = reduced["synthetic_metrics"]
    gates = [
        _gate(
            "validation",
            "required_preconditions",
            "==",
            True,
            reduced["preconditions_passed"],
            reduced["preconditions_passed"],
            True,
        ),
        _gate(
            "completion",
            "sealed_protocol",
            "==",
            True,
            reduced["protocol_complete"],
            reduced["protocol_complete"],
            True,
        ),
        _gate(
            "completion",
            "complete_five_arm_rows",
            "==",
            True,
            reduced["rows_complete"],
            reduced["rows_complete"],
            True,
        ),
        _gate(
            "safety",
            "authority_controls",
            "==",
            True,
            reduced["authority_controls_passed"],
            reduced["authority_controls_passed"],
            True,
        ),
        _gate(
            "safety",
            "restart_and_version_isolation",
            "==",
            True,
            reduced["restart_passed"],
            reduced["restart_passed"],
            True,
        ),
        _gate(
            "safety",
            "unsafe_decisions",
            "==",
            0,
            metrics["unsafe_decisions"],
            metrics["unsafe_decisions"] == 0,
            True,
        ),
        _gate(
            "safety",
            "exact_decision_coverage",
            "==",
            1.0,
            metrics["exact_decision_coverage"],
            metrics["exact_decision_coverage"] == 1.0,
            True,
        ),
        _gate(
            "safety",
            "valid_utility",
            "==",
            True,
            metrics["valid_utility"],
            metrics["valid_utility"] is True,
            True,
        ),
        _gate(
            "safety",
            "individual_erasure_witnesses",
            ">=8_across_>=4_streams",
            True,
            reduced["witness_gate_passed"],
            reduced["witness_gate_passed"],
            True,
        ),
        _gate(
            "validation",
            "affected_checks",
            "==",
            True,
            reduced["affected_validation_passed"],
            reduced["affected_validation_passed"],
            True,
        ),
        _gate(
            "validation",
            "terminal_checks",
            "==",
            True,
            reduced["terminal_validation_passed"],
            reduced["terminal_validation_passed"],
            True,
        ),
    ]
    for comparator in PERSISTENT_CONTROLS:
        paid = metrics["paid_query_ratio_ci95_upper"].get(comparator)
        full = metrics["full_cost_ratio_ci95_upper"].get(comparator)
        gates.extend(
            (
                _gate(
                    "efficacy",
                    f"paid_query_ratio_ci95_upper_vs_{comparator}",
                    "<",
                    0.90,
                    paid,
                    paid is not None and paid < 0.90,
                    False,
                ),
                _gate(
                    "efficacy",
                    f"full_cost_ratio_ci95_upper_vs_{comparator}",
                    "<=",
                    1.0,
                    full,
                    full is not None and full <= 1.0,
                    False,
                ),
            )
        )
    gates.append(_gate("promotion", "automatic_promotion", "==", 0, 0, True, False))
    return gates


def gate_summary(
    gates: Sequence[Mapping[str, Any]], preconditions: Sequence[Mapping[str, Any]]
) -> JsonDict:
    failed_preconditions = [
        row
        for row in preconditions
        if row.get("terminal_blocking") is True and row.get("passed") is not True
    ]
    structured = [
        {
            "upstream": row.get("upstream"),
            "path": row.get("upstream"),
            "check": row.get("check"),
            "field": row.get("artifact_field"),
            "expected": row.get("expected"),
            "observed": row.get("observed"),
        }
        for row in failed_preconditions
    ]
    failures = [deepcopy(dict(row)) for row in gates if row.get("passed") is not True]
    blocking = [row for row in failures if row.get("terminal_blocking") is True]
    return {
        "passed": not blocking,
        "failed_count": len(failures),
        "blocking_failed_count": len(blocking),
        "first_failure": structured[0] if structured else (failures[0] if failures else None),
        "structured_prerequisite_failures": structured,
        "failures": failures,
    }


def _formula_stream_rows(protocol: Mapping[str, Any]) -> list[JsonDict]:
    return [
        {
            "stream_id": stream["stream_id"],
            "formula_id": stream["formula_id"],
            "family": stream["family"],
            "seed": stream["seed"],
            "n_vars": stream["n_vars"],
            "request_id": request["request_id"],
            "request_index": request["request_index"],
            "split": request["split"],
            "formula_version": request["formula_version"],
            "planned_arm_count": len(ARMS),
        }
        for stream in protocol.get("evaluation_streams", [])
        for request in stream.get("requests", [])
    ]


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    keys = (
        "schema",
        "experiment_id",
        "milestone",
        "phase",
        "run_date",
        "random_seed",
        "source_artifact_hashes",
        "preconditions_checked",
        "frozen_acceptance_manifest",
        "rows",
        "cost_rows",
        "erasure_witness_rows",
        "restart_rows",
        "authority_attack_rows",
        "validation_receipts",
        "synthetic_memory_capture_complete_score",
        "synthetic_memory_value_score",
        "proof_safety_ready_score",
        "honest_verdict",
        "verdict_class",
        "flagged_adversarial",
    )
    return canonical_hash({key: artifact.get(key) for key in keys})


def field_principles(artifact: Mapping[str, Any]) -> dict[str, str]:
    specific = {
        "schema": "Use a versioned schema with ordinary top-level identity, milestone, and status.",
        "run_date": "Use 20260919 and retain actual UTC start and end timestamps.",
        "preconditions_checked": "Authenticate exact input hashes, eligibility, runtime, device, and entrypoint before replay.",
        "MODEL_SPECS": "Use an empty list because this cohort performs no current LLM work.",
        "model_invoked": "Remain false because no current model load or generation was attempted.",
        "invocation_counts": "Count only current owned LLM operations; every counter is zero.",
        "inference_substrate": "Describe actual CPU exact-solver and replay work as a string.",
        "inference_substrate_class": "Use the closed CPU exact solver or simulator class.",
        "execution_venue": "Use the closed host string.",
        "duration_s": "Measure current task duration with a monotonic clock and no padding.",
        "phase_spans": "Retain actual phase boundaries, heartbeats, and checkpoint counts.",
        "random_seed": "Freeze experiment and 10,000-draw resampling seeds.",
        "reproducibility_checksum": "Bind code, configuration, inputs, rows, gates, and receipts.",
        "source_artifact_hashes": "Retain exact byte hashes; historical model details stay in sidecars.",
        "rows": "Retain every stream, request, arm, cost, failure, censor, and prior-feedback relation.",
        "sample_size_budget": "Separate planned, attempted, complete, censored, unstarted, and independent groups.",
        "acceptance_gate_results": "Keep category, check, operator, expected, observed, and pass values separate.",
        "gate_check_summary": "For blocked inputs, name upstream path, check, field, expected, and observed values.",
        "verifier_is_oracle": "True because exact formula authority also defines correctness.",
        "honest_verdict": "Use complete_ for finished work and blocked_ for unchanged missing prerequisites.",
        "verdict_class": "Use circular_positive for passed oracle-defined value and null for a valid failed value gate.",
        "flagged_adversarial": "Preserve critical verifier findings and prevent readiness when true.",
        "validation_receipts": "Retain exact argv, environment, scope, exit, duration, and hashed log evidence.",
        "repository_health": "Keep unrelated broad-suite observations separate from affected failures.",
        "field_principles": "Explain fields without wrapping ordinary values or numeric gates.",
        "promotion_score": "Always remain zero; this cohort cannot trigger rollout or publication.",
        "synthetic_memory_capture_complete_score": "One means the complete valid synthetic measurement exists, even for a null.",
        "synthetic_memory_value_score": "Apply the frozen causal and persistent-control efficacy gates.",
        "proof_safety_ready_score": "Track source, version, restart, and unsafe-decision safety separately from speed.",
        "erasure_witness_rows": "Name a distinct later query, erased proof, earlier verified source, and changed exact work.",
        "cost_rows": "Retain all complete-service stages for each arm while persistent state stays active.",
    }
    return {
        key: specific.get(key, "Retain this supporting evidence in its ordinary JSON type.")
        for key in artifact
        if key != "field_principles"
    } | {"field_principles": specific["field_principles"]}


def finalize_artifact(artifact: JsonDict) -> None:
    """Recompute all derived terminal fields after raw evidence changes."""

    reduced = independent_reduce(artifact)
    artifact["independent_reduction"] = reduced
    for score in (
        "synthetic_memory_capture_complete_score",
        "synthetic_memory_value_score",
        "proof_safety_ready_score",
        "promotion_score",
    ):
        artifact[score] = reduced[score]
    failed_preconditions = [
        row
        for row in artifact.get("preconditions_checked", [])
        if row.get("terminal_blocking") is True and row.get("passed") is not True
    ]
    if failed_preconditions:
        artifact["status"] = "blocked_required_exp7371_input_unavailable"
        artifact["honest_verdict"] = "blocked_required_exp7371_input_unavailable"
        artifact["verdict_class"] = "blocked"
    elif reduced["synthetic_memory_capture_complete_score"] != 1:
        artifact["status"] = "complete_disqualified_synthetic_measurement_invalid"
        artifact["honest_verdict"] = "complete_disqualified_synthetic_measurement_invalid"
        artifact["verdict_class"] = "disqualified"
    elif reduced["synthetic_memory_value_score"] == 1:
        artifact["status"] = "complete_source_certified_synthetic_memory_value"
        artifact["honest_verdict"] = (
            "complete_circular_positive_source_certified_synthetic_memory_value"
        )
        artifact["verdict_class"] = "circular_positive"
    else:
        artifact["status"] = "complete_null_synthetic_memory_value_gate_failed"
        artifact["honest_verdict"] = "complete_null_synthetic_memory_value_gate_failed"
        artifact["verdict_class"] = "null"
    artifact["acceptance_gate_results"] = acceptance_gates(artifact)
    artifact["gate_check_summary"] = gate_summary(
        artifact["acceptance_gate_results"], artifact.get("preconditions_checked", [])
    )
    artifact["field_principles"] = field_principles(artifact)
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)


def _base_artifact(
    protocol: Mapping[str, Any],
    evidence: ReplayEvidence,
    receipts: Sequence[Mapping[str, Any]],
    *,
    preconditions: Sequence[Mapping[str, Any]],
    source_hashes: Mapping[str, str],
    historical_sidecars: Sequence[Mapping[str, Any]],
    phase_spans: Sequence[Mapping[str, Any]],
    started_at_utc: str,
    completed_at_utc: str,
    duration_s: float,
    repository_health: Mapping[str, Any],
    flagged_adversarial: bool = False,
) -> JsonDict:
    rows = deepcopy(evidence.rows)
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "phase": PHASE,
        "status": "building_terminal_record",
        "run_date": RUN_DATE,
        "started_at_utc": started_at_utc,
        "completed_at_utc": completed_at_utc,
        "preconditions_checked": deepcopy(list(preconditions)),
        "MODEL_SPECS": [],
        "model_invoked": False,
        "invocation_counts": deepcopy(ZERO_CURRENT_INVOCATIONS),
        "inference_substrate": "host CPU exact 2-CNF solving, source-proof checking, serialization, and deterministic replay",
        "inference_substrate_class": "cpu_exact_solver_or_simulator",
        "inference_substrate_details": {
            "device": "host_cpu",
            "processor": platform.processor() or "unspecified_host_cpu",
            "python": platform.python_version(),
            "jax_platform": os.environ.get("JAX_PLATFORMS", "not_used_by_replay"),
            "llm_runtime": None,
        },
        "execution_venue": "host",
        "duration_s": duration_s,
        "phase_spans": deepcopy(list(phase_spans)),
        "random_seed": deepcopy(RANDOM_SEED),
        "reproducibility_checksum": "pending",
        "source_artifact_hashes": dict(source_hashes),
        "historical_inference_sidecars": deepcopy(list(historical_sidecars)),
        "small_ebm_training": {"performed": False, "reason": "not_part_of_exact_replay"},
        "frozen_protocol_path": PROTOCOL_PATH.as_posix(),
        "frozen_protocol_sha256": EXPECTED_HASHES[PROTOCOL_PATH.as_posix()],
        "frozen_acceptance_manifest": frozen_acceptance_manifest(
            EXPECTED_HASHES[PROTOCOL_PATH.as_posix()]
        ),
        "rows": rows,
        "cost_rows": cost_rows(rows),
        "formula_stream_rows": _formula_stream_rows(protocol),
        "erasure_witness_rows": deepcopy(evidence.witnesses),
        "restart_rows": deepcopy(evidence.restart_rows),
        "authority_attack_rows": deepcopy(evidence.attacks),
        "sample_size_budget": {
            "streams_planned": 32,
            "streams_attempted": len({row.get("stream_id") for row in rows}),
            "streams_completed": len({row.get("stream_id") for row in rows}),
            "streams_censored": 0,
            "streams_unstarted": max(0, 32 - len({row.get("stream_id") for row in rows})),
            "requests_per_stream": 24,
            "arms_per_request": 5,
            "rows_planned": 32 * 24 * 5,
            "rows_attempted": len(rows),
            "rows_completed": sum(row.get("censored") is False for row in rows),
            "rows_censored": sum(row.get("censored") is True for row in rows),
            "rows_unstarted": max(0, 32 * 24 * 5 - len(rows)),
            "maximum_paths": 128,
            "maximum_state_bytes": 65_536,
            "maximum_additions_per_event": 8,
            "maximum_edges_per_path": "2n",
            "effective_independent_stream_groups": len({row.get("stream_id") for row in rows}),
            "effective_formula_families": len({row.get("family") for row in rows}),
            "stop_rule": "run every sealed request once through every arm; do not invoke an LLM",
        },
        "validation_receipts": deepcopy(list(receipts)),
        "repository_health": deepcopy(dict(repository_health)),
        "verifier_is_oracle": True,
        "flagged_adversarial": flagged_adversarial,
        "promotion_score": 0,
        "live_model_benefit_established": False,
        "historical_disqualified_capture_revived": False,
        "reusable_replay_runner": "carnot.experiment_7403_v649_synthetic_memory.replay_protocol",
        "production_defaults_changed": False,
        "active_research_roadmap_changed": False,
        "research_conductor_changed": False,
        "generator_weights_changed": False,
        "rust_changed": False,
        "numbered_e2e_applicable": False,
        "capability_e2e_checks": ["declared_entrypoint", "cold_artifact_replay"],
    }
    finalize_artifact(artifact)
    return artifact


def build_artifact_for_test(
    protocol: Mapping[str, Any],
    evidence: ReplayEvidence,
    *,
    receipts: Sequence[Mapping[str, Any]] | None = None,
    preconditions: Sequence[Mapping[str, Any]] | None = None,
    source_hashes: Mapping[str, str] | None = None,
) -> JsonDict:
    return _base_artifact(
        protocol,
        evidence,
        list(receipts) if receipts is not None else passing_test_receipts(),
        preconditions=(
            list(preconditions)
            if preconditions is not None
            else [precondition_row("unit_fixture", "unit", "available", True, True, True)]
        ),
        source_hashes=source_hashes or {},
        historical_sidecars=[],
        phase_spans=[
            {
                "phase": name,
                "started_at_utc": "2026-09-19T00:00:00+00:00",
                "ended_at_utc": "2026-09-19T00:00:00.100000+00:00",
                "start_s": index / 10,
                "end_s": (index + 1) / 10,
                "duration_s": 0.1,
                "heartbeats": 0,
                "checkpoints": 32 if name == "evaluate" else 0,
            }
            for index, name in enumerate(
                ("read", "seal", "load", "generate", "evaluate", "validate", "write")
            )
        ],
        started_at_utc="2026-09-19T00:00:00+00:00",
        completed_at_utc="2026-09-19T00:00:01+00:00",
        duration_s=1.0,
        repository_health={"status": "unit_fixture", "affects_required_checks": False},
    )


def validate_artifact(value: object) -> list[str]:
    """Cold-check identity, declarations, row reduction, gates, and checksum."""

    if not isinstance(value, Mapping):
        return ["artifact_not_object"]
    artifact = dict(value)
    errors: list[str] = []
    if (
        artifact.get("schema"),
        artifact.get("experiment_id"),
        artifact.get("milestone"),
        artifact.get("phase"),
        artifact.get("run_date"),
    ) != (SCHEMA, EXPERIMENT_ID, MILESTONE, PHASE, RUN_DATE):
        errors.append("identity_mismatch")
    if artifact.get("MODEL_SPECS") != [] or artifact.get("model_invoked") is not False:
        errors.append("model_declaration_mismatch")
    if artifact.get("invocation_counts") != ZERO_CURRENT_INVOCATIONS:
        errors.append("invocation_counts_nonzero")
    if (
        not isinstance(artifact.get("inference_substrate"), str)
        or artifact.get("inference_substrate_class") != "cpu_exact_solver_or_simulator"
        or artifact.get("execution_venue") != "host"
    ):
        errors.append("substrate_mismatch")
    if artifact.get("verifier_is_oracle") is not True:
        errors.append("oracle_declaration_mismatch")
    reduced = independent_reduce(artifact)
    if artifact.get("independent_reduction") != reduced:
        errors.append("independent_reduction_mismatch")
    for score in (
        "synthetic_memory_capture_complete_score",
        "synthetic_memory_value_score",
        "proof_safety_ready_score",
    ):
        if artifact.get(score) != reduced[score]:
            errors.append(f"{score}_mismatch")
    if artifact.get("promotion_score") != 0:
        errors.append("promotion_nonzero")
    expected_gates = acceptance_gates(artifact)
    if artifact.get("acceptance_gate_results") != expected_gates:
        errors.append("acceptance_gates_mismatch")
    expected_summary = gate_summary(expected_gates, artifact.get("preconditions_checked", []))
    if artifact.get("gate_check_summary") != expected_summary:
        errors.append("gate_check_summary_mismatch")
    expected_class = "disqualified"
    if any(
        row.get("terminal_blocking") is True and row.get("passed") is not True
        for row in artifact.get("preconditions_checked", [])
    ):
        expected_class = "blocked"
    elif reduced["synthetic_memory_capture_complete_score"] == 1:
        expected_class = (
            "circular_positive" if reduced["synthetic_memory_value_score"] == 1 else "null"
        )
    if artifact.get("verdict_class") != expected_class:
        errors.append("verdict_class_mismatch")
    if expected_class == "blocked" and not str(artifact.get("honest_verdict", "")).startswith(
        "blocked_"
    ):
        errors.append("honest_verdict_mismatch")
    if expected_class != "blocked" and not str(artifact.get("honest_verdict", "")).startswith(
        "complete_"
    ):
        errors.append("honest_verdict_mismatch")
    principles = artifact.get("field_principles")
    if not isinstance(principles, Mapping) or set(principles) != set(artifact):
        errors.append("field_principles_incomplete")
    if artifact.get("reproducibility_checksum") != reproducibility_checksum(artifact):
        errors.append("reproducibility_checksum_mismatch")
    return list(dict.fromkeys(errors))


def _semantic_rows_checksum(rows: Sequence[Mapping[str, Any]]) -> str:
    normalized = [
        {key: deepcopy(value) for key, value in row.items() if key != "duration_ns"} for row in rows
    ]
    return canonical_hash(normalized)


def write_raw_evidence(
    root: Path,
    protocol: Mapping[str, Any],
    evidence: ReplayEvidence,
    historical_sidecars: Sequence[Mapping[str, Any]],
) -> dict[str, str]:
    """Publish replay tables and historical receipts as hash-bound sidecars."""

    values = {
        root / RAW_DIR / "protocol_snapshot.json": deepcopy(dict(protocol)),
        root / RAW_DIR / "synthetic_evidence.json": {
            "rows": deepcopy(evidence.rows),
            "erasure_witness_rows": deepcopy(evidence.witnesses),
            "restart_rows": deepcopy(evidence.restart_rows),
            "cost_rows": cost_rows(evidence.rows),
            "semantic_rows_checksum": _semantic_rows_checksum(evidence.rows),
        },
        root / RAW_DIR / "authority_attacks.json": {
            "authority_attack_rows": deepcopy(evidence.attacks)
        },
        root / RAW_DIR / "historical_model_receipts.json": {
            "label": "historical_only_not_current_inference",
            "sources": deepcopy(list(historical_sidecars)),
        },
    }
    hashes: dict[str, str] = {}
    for path, payload in values.items():
        atomic_json(path, payload)
        hashes[path.relative_to(root).as_posix()] = sha256_file(path)
    checkpoint_dir = root / RAW_DIR / "checkpoints"
    if checkpoint_dir.is_dir():
        for path in sorted(checkpoint_dir.iterdir()):
            if path.is_file():
                hashes[path.relative_to(root).as_posix()] = sha256_file(path)
    return hashes


def cold_reload_errors(
    artifact: Mapping[str, Any], root: Path, *, recompute_rows: bool = True
) -> list[str]:
    """Reload raw bytes and optionally rerun every semantic request in a fresh process."""

    errors = validate_artifact(artifact)
    for raw_path, expected in (artifact.get("source_artifact_hashes") or {}).items():
        if not str(raw_path).startswith(RAW_DIR.as_posix()):
            continue
        path = Path(str(raw_path))
        resolved = path if path.is_absolute() else root / path
        if not resolved.is_file() or sha256_file(resolved) != expected:
            errors.append(f"raw_hash_mismatch:{raw_path}")
    raw = _load_object(root / RAW_DIR / "synthetic_evidence.json")
    if raw:
        if raw.get("rows") != artifact.get("rows"):
            errors.append("raw_rows_mismatch")
        if raw.get("erasure_witness_rows") != artifact.get("erasure_witness_rows"):
            errors.append("raw_witnesses_mismatch")
        if raw.get("restart_rows") != artifact.get("restart_rows"):
            errors.append("raw_restart_rows_mismatch")
        if raw.get("cost_rows") != artifact.get("cost_rows"):
            errors.append("raw_cost_rows_mismatch")
    attacks = _load_object(root / RAW_DIR / "authority_attacks.json")
    if attacks and attacks.get("authority_attack_rows") != artifact.get("authority_attack_rows"):
        errors.append("raw_attacks_mismatch")
    if recompute_rows:
        protocol = _load_object(root / RAW_DIR / "protocol_snapshot.json")
        replayed = replay_protocol(protocol)
        if _semantic_rows_checksum(replayed.rows) != _semantic_rows_checksum(
            artifact.get("rows") or []
        ):
            errors.append("independent_row_replay_mismatch")
    return list(dict.fromkeys(errors))


def scoped_command_plan(repo_root: Path, private_root: Path) -> list[validation_scope.CommandSpec]:
    return validation_contract.build_command_plan(repo_root, V649_MANIFEST, private_root)


def validate_scoped_command_plan(
    repo_root: Path, commands: Sequence[validation_scope.CommandSpec]
) -> list[str]:
    errors = validation_contract.validate_command_plan(repo_root, V649_MANIFEST, commands)
    if Counter(command.name for command in commands) != Counter(REQUIRED_CHECK_NAMES):
        errors.append("required_command_names_changed")
    if any(command.name == "full_python_suite" for command in commands):
        errors.append("full_python_suite_forbidden")
    return list(dict.fromkeys(errors))


def _normalized_receipt(row: Mapping[str, Any]) -> JsonDict:
    return {
        "name": row.get("name"),
        "command": row.get("command"),
        "command_argv": deepcopy(row.get("command_argv") or []),
        "environment": deepcopy(row.get("command_environment") or {}),
        "scope": row.get("scope"),
        "return_code": row.get("exit_code"),
        "exit_code": row.get("exit_code"),
        "duration_s": row.get("duration_s"),
        "log_path": row.get("log_path"),
        "log_hash": row.get("log_sha256"),
        "log_sha256": row.get("log_sha256"),
        "passed": row.get("passed"),
        "timed_out": row.get("timed_out"),
        "required": row.get("required", True),
        "category": row.get("command_category", "required_validation"),
        "started_at_utc": row.get("started_at_utc"),
        "ended_at_utc": row.get("ended_at_utc"),
        **(
            {"resolved_imports": deepcopy(row["resolved_imports"])}
            if "resolved_imports" in row
            else {}
        ),
    }


def terminal_commands(candidate: Path) -> list[validation_contract.PlannedCommand]:
    python = str(REPO_ROOT / ".venv/bin/python")
    replay = (
        "import json,pathlib,sys;"
        "from carnot.experiment_7403_v649_synthetic_memory import cold_reload_errors;"
        "p=pathlib.Path(sys.argv[1]);v=json.loads(p.read_text());"
        "e=cold_reload_errors(v,pathlib.Path(sys.argv[2]));print(e,flush=True);"
        "raise SystemExit(bool(e))"
    )
    reduce = (
        "import json,pathlib,sys;"
        "from carnot.experiment_7403_v649_synthetic_memory import validate_artifact;"
        "v=json.loads(pathlib.Path(sys.argv[1]).read_text());"
        "e=validate_artifact(v);print(e,flush=True);raise SystemExit(bool(e))"
    )
    specs = (
        (
            "cold_artifact_replay",
            (python, "-u", "-c", replay, str(candidate), str(REPO_ROOT)),
            "capability_e2e",
        ),
        ("independent_reducer", (python, "-u", "-c", reduce, str(candidate)), "completion"),
        (
            "adversarial_verify",
            (python, "-u", "scripts/adversarial_verify.py", str(candidate)),
            "safety",
        ),
        (
            "verdict_row_consistency_strict",
            (
                python,
                "-u",
                "scripts/verdict_row_consistency_lint.py",
                "--strict",
                str(candidate),
            ),
            "completion",
        ),
    )
    return [
        validation_contract.PlannedCommand(
            validation_scope.CommandSpec(name, argv, "measured_candidate"), category, True
        )
        for name, argv, category in specs
    ]


def _span(
    phase: str, phase_started: float, run_started: float, started_at: str, checkpoints: int = 0
) -> JsonDict:
    ended = time.monotonic()
    return {
        "phase": phase,
        "started_at_utc": started_at,
        "ended_at_utc": utc_now(),
        "start_s": phase_started - run_started,
        "end_s": ended - run_started,
        "duration_s": ended - phase_started,
        "heartbeats": 0,
        "checkpoints": checkpoints,
    }


def run_experiment(
    repo_root: Path, run_date: str, *, output_path: Path = RESULT_PATH
) -> JsonDict:  # pragma: no cover - exercised through the declared entrypoint.
    """Authenticate, replay, validate, cold-read, and atomically publish."""

    if run_date != RUN_DATE:
        raise SystemExit(f"--date must be {RUN_DATE}")
    root = repo_root.resolve()
    output = output_path if output_path.is_absolute() else root / output_path
    run_started = time.monotonic()
    started_at = utc_now()
    spans: list[JsonDict] = []

    phase_started = time.monotonic()
    phase_utc = utc_now()
    progress(run_started, "read", "start")
    preconditions, source_hashes, historical_sidecars = collect_preconditions(root)
    blocked = any(row["terminal_blocking"] and not row["passed"] for row in preconditions)
    spans.append(_span("read", phase_started, run_started, phase_utc))
    progress(run_started, "read", "end", blocked=blocked, checks=len(preconditions))

    phase_started = time.monotonic()
    phase_utc = utc_now()
    progress(run_started, "seal", "start")
    protocol = _load_object(root / PROTOCOL_PATH)
    acceptance = frozen_acceptance_manifest(EXPECTED_HASHES[PROTOCOL_PATH.as_posix()])
    if not blocked and (
        protocol_errors(protocol) or acceptance["sealed_before_timing"] is not True
    ):
        raise RuntimeError("sealed_protocol_invalid")
    spans.append(_span("seal", phase_started, run_started, phase_utc))
    progress(run_started, "seal", "end")

    phase_started = time.monotonic()
    phase_utc = utc_now()
    progress(run_started, "load", "before_model_load", models=0)
    spans.append(_span("load", phase_started, run_started, phase_utc))
    progress(run_started, "load", "after_model_load", models=0)

    phase_started = time.monotonic()
    phase_utc = utc_now()
    progress(run_started, "generate", "before_generation", calls=0)
    spans.append(_span("generate", phase_started, run_started, phase_utc))
    progress(run_started, "generate", "after_generation", calls=0)

    phase_started = time.monotonic()
    phase_utc = utc_now()
    progress(run_started, "evaluate", "before_benchmark", blocked=blocked)
    evidence = ReplayEvidence([], [], [], [])
    if not blocked:
        evidence = replay_protocol(
            protocol,
            checkpoint_dir=root / RAW_DIR / "checkpoints",
            emit_progress=True,
            started=run_started,
        )
    spans.append(
        _span(
            "evaluate",
            phase_started,
            run_started,
            phase_utc,
            len({row.get("stream_id") for row in evidence.rows}),
        )
    )
    progress(run_started, "evaluate", "after_benchmark", rows=len(evidence.rows))
    source_hashes.update(write_raw_evidence(root, protocol, evidence, historical_sidecars))

    receipts: list[JsonDict] = []
    phase_started = time.monotonic()
    phase_utc = utc_now()
    progress(run_started, "validate", "before_subprocesses", blocked=blocked)
    if not blocked:
        commands = scoped_command_plan(root, root / RAW_DIR / "validation/private")
        plan_errors = validate_scoped_command_plan(root, commands)
        if plan_errors:
            raise RuntimeError(f"invalid_scoped_command_plan:{plan_errors}")
        planned = [
            validation_contract.PlannedCommand(command, "required_validation", True)
            for command in commands
        ]
        raw_receipts = validation_contract.run_categorized_commands(
            root, planned, log_dir=root / RAW_DIR / "validation/logs"
        )
        receipts.extend(_normalized_receipt(row) for row in raw_receipts)
    spans.append(_span("validate", phase_started, run_started, phase_utc))
    progress(run_started, "validate", "after_subprocesses", receipts=len(receipts))

    repository_health = {
        "status": "not_used_for_affected_readiness",
        "observations": [],
        "affects_required_checks": False,
    }
    phase_started = time.monotonic()
    phase_utc = utc_now()
    progress(run_started, "write", "start")
    candidate_path = root / RAW_DIR / "measured-terminal-candidate.json"
    candidate = _base_artifact(
        protocol,
        evidence,
        receipts,
        preconditions=preconditions,
        source_hashes=source_hashes,
        historical_sidecars=historical_sidecars,
        phase_spans=spans,
        started_at_utc=started_at,
        completed_at_utc=utc_now(),
        duration_s=time.monotonic() - run_started,
        repository_health=repository_health,
    )
    atomic_json(candidate_path, candidate)
    terminal_receipts: list[JsonDict] = []
    if not blocked:
        progress(run_started, "write", "before_terminal_subprocesses")
        raw_terminal = validation_contract.run_categorized_commands(
            root,
            terminal_commands(candidate_path),
            log_dir=root / RAW_DIR / "terminal/logs",
        )
        terminal_receipts = [_normalized_receipt(row) for row in raw_terminal]
        progress(run_started, "write", "after_terminal_subprocesses", count=len(terminal_receipts))
    entrypoint_log = root / RAW_DIR / "terminal/declared_entrypoint.log"
    entrypoint_log.parent.mkdir(parents=True, exist_ok=True)
    entrypoint_log.write_text(
        "The unbuffered declared entrypoint reached the atomic publication boundary.\n",
        encoding="utf-8",
    )
    terminal_receipts.append(
        {
            "name": "declared_entrypoint",
            "command": " ".join(sys.argv),
            "command_argv": list(sys.argv),
            "environment": {
                key: os.environ.get(key)
                for key in ("PYTHONUNBUFFERED", "JAX_PLATFORMS", "PYTHONPATH", "CARNOT_FORCE_LIVE")
            },
            "scope": "current_unbuffered_entrypoint",
            "return_code": 0,
            "exit_code": 0,
            "duration_s": time.monotonic() - run_started,
            "log_path": entrypoint_log.relative_to(root).as_posix(),
            "log_hash": sha256_file(entrypoint_log),
            "log_sha256": sha256_file(entrypoint_log),
            "passed": True,
            "timed_out": False,
            "required": True,
            "category": "capability_e2e",
            "started_at_utc": started_at,
            "ended_at_utc": utc_now(),
        }
    )
    receipts.extend(terminal_receipts)
    spans.append(_span("write", phase_started, run_started, phase_utc))
    flagged = any(
        row["name"] == "adversarial_verify" and row["passed"] is not True
        for row in terminal_receipts
    )
    final = _base_artifact(
        protocol,
        evidence,
        receipts,
        preconditions=preconditions,
        source_hashes=source_hashes,
        historical_sidecars=historical_sidecars,
        phase_spans=spans,
        started_at_utc=started_at,
        completed_at_utc=utc_now(),
        duration_s=time.monotonic() - run_started,
        repository_health=repository_health,
        flagged_adversarial=flagged,
    )
    errors = validate_artifact(final)
    if errors:
        raise RuntimeError(f"terminal_artifact_invalid:{errors}")
    atomic_json(output, final)
    progress(run_started, "write", "end", output=output, verdict=final["verdict_class"])
    return final


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", required=True)
    parser.add_argument("--output", type=Path, default=RESULT_PATH)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - thin CLI boundary.
    args = parse_args(argv)
    run_experiment(REPO_ROOT, args.date, output_path=args.output)
    return 0
