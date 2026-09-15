"""Reduce the measured full-snapshot costs into same-semantics upper bounds.

The producer repeats shared group timers on each event because those rows measure
event latency. This reducer counts shared work once for Amdahl bounds. It keeps
overlapping latency areas separate so a queue timer cannot become fake work.

Spec refs: REQ-CL-7313 and SCENARIO-CL-7313-*.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

from carnot import experiment_7213_v635_refinement_learning as exp7213
from carnot import experiment_7299_v641_snapshot_cost as exp7299


JsonDict = dict[str, Any]

EXPERIMENT_ID = 7313
SCHEMA = "carnot.exp7313.v642_cost_envelope.v1"
MILESTONE = "2026.09.642"
RUN_DATE = "20260914"
RANDOM_SEED = 7_313_000
EVALUATION_SEEDS = tuple(range(7_299_001, 7_299_009))
ARRIVAL_PROCESSES = ("burst", "steady_paced", "interactive_dependent")
SCENARIOS = ("PRECONDITIONS", "RECONSTRUCTION", "ENVELOPE", "UNCERTAINTY", "E2E", "TERMINAL")
MODEL_SPECS: list[JsonDict] = []
MODEL_INVOKED = False
INVOCATION_COUNTS = {
    "attempted_loads": 0,
    "completed_loads": 0,
    "failed_loads": 0,
    "cancelled_loads": 0,
    "in_flight_loads": 0,
    "attempted_generations": 0,
    "completed_generations": 0,
    "failed_generations": 0,
    "cancelled_generations": 0,
    "in_flight_generations": 0,
}
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
INFERENCE_SUBSTRATE_CLASS = "aggregation"
EXECUTION_VENUE = "host"
BOOTSTRAP_DRAWS = 10_000
TARGET_LOCAL_SPEEDUP = 1.5
TARGET_NFR01_SPEEDUP = 10.0
EVENTS_PER_TRIAL = 256
MAX_GROUP_SIZE = 16
STEADY_INTERARRIVAL_NS = 1_000_000

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE = Path("results/experiment_7313_v642_cost_envelope.json")
CHECKPOINT_RELATIVE = Path("results/checkpoints/experiment_7313_v642_cost_envelope.json")
RAW_CANDIDATE_RELATIVE = Path("results/raw/experiment_7313_v642_cost_envelope_candidate.json")
VALIDATION_RELATIVE = Path("results/raw/experiment_7313_v642_cost_envelope/validation")
EXP7299_RELATIVE = Path("results/experiment_7299_v641_snapshot_cost.json")
EXP7270_RELATIVE = Path("results/experiment_7270_v639_durable_profile.json")
RAW_ROWS_RELATIVE = Path("results/raw/experiment_7299_v641_snapshot_cost_rows.json")
EXCLUSION_RELATIVE = Path("ops/exclusion_manifest.yaml")
SPEC_RELATIVE = Path("openspec/capabilities/continuous-learning/spec.md")
EXPECTED_EXP7299_SHA256 = "sha256:f7f4130452d197aecacbd6c5d8ab5005f97ff2acccf335af84d0c5b67d2d47a5"
EXPECTED_EXP7270_SHA256 = "sha256:9026a34ec0ffbfaa2c1c0d6127c029fee1ca03c1cc33e191343d4d9d76854be0"
EXPECTED_RAW_ROWS_SHA256 = "sha256:e6ef6378fda49a7e8878bf40955ecb7ab5cdba1ced72f522dcc06af0ff98b11e"
EXPECTED_WARM_LOWER = 1.4558337555069685
EXPECTED_COLD_LOWER = 1.5549724849050095
CURRENT_FULL_SUITE_ATTEMPT = {
    "command": [str(Path(sys.executable).absolute()), "-m", "pytest", "tests/python", "-q"],
    "completed": False,
    "exit_code": 130,
    "elapsed_s": 273.944,
    "observed_progress": "10%",
    "observed_result": "multiple repository-wide failures before interruption",
    "log_sha256": None,
    "log_limitation": "The parent received KeyboardInterrupt before the streaming helper returned a complete log.",
}

SOURCE_PATHS = (
    Path("AGENTS.md"),
    Path("CLAUDE.md"),
    Path("CODEX.md"),
    Path("research-program.md"),
    EXCLUSION_RELATIVE,
    Path("ops/e2e-test-plan.md"),
    SPEC_RELATIVE,
    Path("python/carnot/experiment_7270_v639_durable_profile.py"),
    Path("python/carnot/experiment_7299_v641_snapshot_cost.py"),
    Path("python/carnot/experiment_7313_v642_cost_envelope.py"),
    Path("scripts/experiments/experiment_7313_v642_cost_envelope.py"),
    Path("tests/python/test_experiment_7313_v642_cost_envelope.py"),
    EXP7299_RELATIVE,
    EXP7270_RELATIVE,
    RAW_ROWS_RELATIVE,
)

FIELD_PRINCIPLES = {
    "schema": "Version the record while keeping ordinary top-level experiment_id and milestone.",
    "status": "Write terminal evidence only after checks; keep retry state in checkpoints.",
    "run_date": "Use 20260914 with actual UTC start and end plus monotonic phase timing.",
    "preconditions_checked": "Hash real inputs and record actual availability and failed checks.",
    "MODEL_SPECS": "List current executable models only; this aggregation invokes none.",
    "model_invoked": "Set true for any attempted load or generation, including unusable work.",
    "invocation_counts": "Separate every load and generation outcome, including in-flight work.",
    "inference_substrate": "Describe the actual upstream-artifact aggregation substrate.",
    "inference_substrate_class": "Use aggregation because no generation occurs and never pad time.",
    "execution_venue": "Record host work as host work without a device claim.",
    "duration_s": "Measure total monotonic elapsed and disjoint phase spans.",
    "random_seed": "Seal reduction and bootstrap seeds before computing outcomes.",
    "reproducibility_checksum": "Bind code, inputs, configuration, and reduced evidence.",
    "source_artifact_hashes": "Authenticate producer identity, terminal class, and exclusion state.",
    "rows": "Retain each paired seed and arrival process with metrics, costs, and dispositions.",
    "sample_size_budget": "Keep planned, attempted, complete, and censored counts with the fixed rule.",
    "acceptance_gate_results": "Record expected, observed, passed, and purpose for every check.",
    "gate_check_summary": "Preserve the exact upstream, check, field, observed, and expected values.",
    "verifier_is_oracle": "Shared evaluator authority cannot create positive scientific value.",
    "honest_verdict": "Use complete_ for findings and blocked_ for unchanged external failure.",
    "verdict_class": "Use only the closed verdict classes and reserve partial for owned unfinished work.",
    "validation_receipts": "Record exact commands, scope, exit codes, elapsed time, and log hashes.",
    "cost_envelope_complete_score": "One means faithful reduction and explicit assumptions, not speed success.",
    "cost_bound_rows": "Retain per-seed measured fractions and counterfactual upper limits.",
    "nfr01_assessment": "Keep the original tenfold target separate from the local 1.5x gate.",
    "next_mechanism_warrant": "Name a technique only when a measured replaceable cost is sufficient.",
}
REQUIRED_FIELDS = frozenset(FIELD_PRINCIPLES)

canonical_json = exp7299.canonical_json
sha256_file = exp7299.sha256_file
artifact_checksum = exp7299.artifact_checksum
check = exp7299.check
gate_summary = exp7299.gate_summary
atomic_write = exp7299.atomic_write
_finish_row = exp7299._finish_row


@dataclass(frozen=True)
class ExperimentPaths:
    """Keep retry, candidate, validation, and terminal output bytes separate."""

    checkpoint: Path
    raw_candidate: Path
    validation_dir: Path
    artifact: Path

    @classmethod
    def defaults(cls) -> ExperimentPaths:
        """Return the fixed paths owned by the public command."""

        return cls.under(REPO_ROOT)

    @classmethod
    def under(cls, root: Path) -> ExperimentPaths:
        """Place test output below one caller-owned temporary root."""

        return cls(
            root / CHECKPOINT_RELATIVE,
            root / RAW_CANDIDATE_RELATIVE,
            root / VALIDATION_RELATIVE,
            root / RESULT_RELATIVE,
        )

    def writable_targets(self) -> list[bool]:
        """Check output parents without creating success-shaped bytes."""

        return [_writable(path) for path in self.__dict__.values()]


def progress(phase: int, boundary: str, operation: str, started: float | None = None) -> None:
    """Print and flush a phase boundary with optional monotonic elapsed time."""

    elapsed = "" if started is None else f" elapsed_s={time.monotonic() - started:.3f}"
    print(f"[phase {phase} {boundary}] {operation}{elapsed}", flush=True)


def _writable(path: Path) -> bool:
    """Check the nearest existing parent without creating output bytes."""

    parent = path if path.is_dir() else path.parent
    while not parent.exists() and parent != parent.parent:
        parent = parent.parent
    return parent.is_dir() and os.access(parent, os.W_OK)


def _read_json(path: Path) -> JsonDict:
    """Read one JSON object and reject every other top-level shape."""

    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"artifact is not an object:{path}")
    return value


def _checksum_valid(artifact: Mapping[str, Any]) -> bool:
    """Recompute an artifact checksum instead of trusting its declaration."""

    try:
        return artifact_checksum(artifact) == artifact.get("reproducibility_checksum")
    except (TypeError, ValueError):
        return False


def _quarantine(
    artifact: Mapping[str, Any], manifest: str, path: Path, artifact_id: str
) -> JsonDict:
    """Apply the shipped exclusion check and include explicit disqualification."""

    state = exp7213.quarantine_state(artifact, manifest, path.name, artifact_id)
    state["disqualified"] = bool(
        artifact.get("flagged_adversarial") is True
        or artifact.get("verdict_class") == "disqualified"
    )
    return state


def collect_preconditions(
    root: Path,
    paths: ExperimentPaths,
    *,
    exp7299_path: Path | None = None,
    exp7270_path: Path | None = None,
    raw_rows_path: Path | None = None,
) -> tuple[list[JsonDict], JsonDict]:
    """Authenticate terminal artifacts, raw rows, exclusions, contracts, and outputs."""

    started = time.monotonic()
    progress(0, "start", "authenticate Exp7299, Exp7270, raw rows, and outputs", started)
    cost_path = exp7299_path or root / EXP7299_RELATIVE
    profile_path = exp7270_path or root / EXP7270_RELATIVE
    raw_path = raw_rows_path or root / RAW_ROWS_RELATIVE
    cost = _read_json(cost_path) if cost_path.is_file() else {}
    profile = _read_json(profile_path) if profile_path.is_file() else {}
    raw = _read_json(raw_path) if raw_path.is_file() else {}
    manifest_path = root / EXCLUSION_RELATIVE
    manifest = manifest_path.read_text(encoding="utf-8") if manifest_path.is_file() else ""
    cost_state = _quarantine(cost, manifest, cost_path, "exp7299-v641-snapshot-cost")
    profile_state = _quarantine(profile, manifest, profile_path, "exp7270-v639-durable-profile")
    hashes = {
        "exp7299": sha256_file(cost_path) if cost_path.is_file() else None,
        "exp7270": sha256_file(profile_path) if profile_path.is_file() else None,
        "raw_rows": sha256_file(raw_path) if raw_path.is_file() else None,
    }
    source_hashes = {
        str(path): sha256_file(root / path) if (root / path).is_file() else None
        for path in SOURCE_PATHS
    }
    warm = (
        cost.get("acceptance_gate_results", {})
        .get("fixed_group16_burst_throughput", {})
        .get("observed", {})
        .get("lower_ci95")
    )
    cold = (
        cost.get("acceptance_gate_results", {})
        .get("cold_group16_burst_throughput", {})
        .get("observed", {})
        .get("lower_ci95")
    )
    raw_matches = all(
        raw.get(name) == cost.get(name)
        for name in (
            "rows",
            "per_run_results",
            "phase_cost_rows",
            "independent_recovery_rows",
        )
    )
    storage = cost.get("storage_identity", {}).get("sqlite_writer", {})
    expected_storage = {
        "delta_log": False,
        "full_snapshot": True,
        "groups": [1, 4, 16],
        "journal": ["persist"],
        "max_pending_bytes": 65536,
        "max_pending_events": 16,
        "max_wait_ms": 10,
        "synchronous": [2],
    }
    checks = [
        check(
            "required_source_bytes",
            "repository",
            "SOURCE_PATHS",
            "all nonempty",
            source_hashes,
            all(source_hashes.values()),
        ),
        check(
            "driving_capability",
            str(SPEC_RELATIVE),
            "REQ-CL-7313 and SCENARIO-CL-7313-*",
            True,
            all(
                token in (root / SPEC_RELATIVE).read_text(encoding="utf-8")
                for token in ("REQ-CL-7313", *[f"SCENARIO-CL-7313-{name}" for name in SCENARIOS])
            ),
            (root / SPEC_RELATIVE).is_file()
            and all(
                token in (root / SPEC_RELATIVE).read_text(encoding="utf-8")
                for token in ("REQ-CL-7313", *[f"SCENARIO-CL-7313-{name}" for name in SCENARIOS])
            ),
        ),
        check(
            "exp7299_artifact_hash",
            str(cost_path),
            "sha256",
            EXPECTED_EXP7299_SHA256,
            hashes["exp7299"],
            hashes["exp7299"] == EXPECTED_EXP7299_SHA256,
        ),
        check(
            "exp7270_artifact_hash",
            str(profile_path),
            "sha256",
            EXPECTED_EXP7270_SHA256,
            hashes["exp7270"],
            hashes["exp7270"] == EXPECTED_EXP7270_SHA256,
        ),
        check(
            "exp7299_raw_rows_hash",
            str(raw_path),
            "sha256",
            EXPECTED_RAW_ROWS_SHA256,
            hashes["raw_rows"],
            hashes["raw_rows"] == EXPECTED_RAW_ROWS_SHA256,
        ),
        check(
            "exp7299_terminal_capture",
            str(cost_path),
            "status,checksum,snapshot_capture_complete_score,verdict_class",
            {"status": "complete", "checksum": True, "capture": 1, "verdict_class": "null"},
            {
                "status": cost.get("status"),
                "checksum": _checksum_valid(cost),
                "capture": cost.get("snapshot_capture_complete_score"),
                "verdict_class": cost.get("verdict_class"),
            },
            cost.get("status") == "complete"
            and _checksum_valid(cost)
            and cost.get("snapshot_capture_complete_score") == 1
            and cost.get("verdict_class") == "null",
        ),
        check(
            "exp7270_terminal_profile",
            str(profile_path),
            "status,checksum,durable_profile_complete_score",
            {"status": "complete", "checksum": True, "complete": 1},
            {
                "status": profile.get("status"),
                "checksum": _checksum_valid(profile),
                "complete": profile.get("durable_profile_complete_score"),
            },
            profile.get("status") == "complete"
            and _checksum_valid(profile)
            and profile.get("durable_profile_complete_score") == 1,
        ),
        check(
            "inputs_not_quarantined_retired_or_disqualified",
            str(EXCLUSION_RELATIVE),
            "quarantined,retired,disqualified",
            {
                "exp7299": {"quarantined": False, "retired": False, "disqualified": False},
                "exp7270": {"quarantined": False, "retired": False, "disqualified": False},
            },
            {
                "exp7299": {
                    key: cost_state.get(key, False)
                    for key in ("quarantined", "retired", "disqualified")
                },
                "exp7270": {
                    key: profile_state.get(key, False)
                    for key in ("quarantined", "retired", "disqualified")
                },
            },
            not any(
                cost_state.get(key, False) or profile_state.get(key, False)
                for key in ("quarantined", "retired", "disqualified")
            ),
        ),
        check(
            "raw_stage_parity",
            str(raw_path),
            "schema and embedded row groups",
            "exact Exp7299 rows",
            {"schema": raw.get("schema"), "exact": raw_matches},
            raw.get("schema") == "carnot.exp7299.v641_snapshot_cost.rows.v1" and raw_matches,
        ),
        check(
            "original_gate_observations",
            str(cost_path),
            "warm lower,cold lower,NFR-01",
            {"warm": EXPECTED_WARM_LOWER, "cold": EXPECTED_COLD_LOWER, "nfr01_passed": False},
            {
                "warm": warm,
                "cold": cold,
                "nfr01_passed": cost.get("nfr01_assessment", {}).get("passed"),
            },
            warm == EXPECTED_WARM_LOWER
            and cold == EXPECTED_COLD_LOWER
            and cost.get("nfr01_assessment", {}).get("passed") is False,
        ),
        check(
            "storage_and_acknowledgment_contract",
            str(cost_path),
            "storage_identity,queue,recovery,limitations",
            {"storage": expected_storage, "capture": 1, "recovery": True},
            {
                "storage": storage,
                "capture": cost.get("snapshot_capture_complete_score"),
                "recovery": cost.get("independent_raw_reducer", {}).get("recovery_complete"),
            },
            storage == expected_storage
            and cost.get("snapshot_capture_complete_score") == 1
            and cost.get("independent_raw_reducer", {}).get("recovery_complete") is True,
        ),
        check(
            "task_owned_outputs",
            "Exp7313",
            "writable output parents",
            [True] * 4,
            paths.writable_targets(),
            all(paths.writable_targets()),
        ),
    ]
    evidence = {
        "exp7299": cost,
        "exp7270": profile,
        "raw_rows": raw,
        "artifact_hashes": hashes,
        "source_hashes": source_hashes,
        "input_paths": {
            "exp7299": str(cost_path.resolve()),
            "exp7270": str(profile_path.resolve()),
            "raw_rows": str(raw_path.resolve()),
        },
        "quarantine": {"exp7299": cost_state, "exp7270": profile_state},
        "started_at_utc": datetime.now(UTC).isoformat(),
    }
    progress(
        0,
        "end",
        f"precondition_checks={len(checks)} failed={sum(row['passed'] is not True for row in checks)}",
        started,
    )
    return checks, evidence


def _stage_totals(event_rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Deduplicate shared group stages and bound mixed serialization timers."""

    groups: dict[str, list[Mapping[str, Any]]] = {}
    for row in event_rows:
        groups.setdefault(str(row["group_id"]), []).append(row)
    sync = native = write = serialization_lower = serialization_upper = 0
    for group in groups.values():
        for name in ("sync_ns", "native_transition_ns", "write_ns"):
            if len({int(row[name]) for row in group}) != 1:
                raise ValueError(f"shared_group_timer_mismatch:{name}")
        sync += int(group[0]["sync_ns"])
        native += int(group[0]["native_transition_ns"])
        write += int(group[0]["write_ns"])
        serial = [int(row["serialization_ns"]) for row in group]
        serialization_lower += sum(serial) - (len(serial) - 1) * min(serial)
        serialization_upper += sum(serial)
    return {
        "durable_group_count": len(groups),
        "sync_deduplicated_ns": sync,
        "native_update_deduplicated_ns": native,
        "write_deduplicated_ns": write,
        "serialization_lower_ns": serialization_lower,
        "serialization_upper_ns": serialization_upper,
        "sync_latency_area_ns": sum(int(row["sync_ns"]) for row in event_rows),
        "native_update_latency_area_ns": sum(
            int(row["native_transition_ns"]) for row in event_rows
        ),
        "queue_wait_latency_area_ns": sum(int(row["queue_wait_ns"]) for row in event_rows),
        "acknowledgment_latency_area_ns": sum(int(row["acknowledgment_ns"]) for row in event_rows),
    }


def _bootstrap(values: Sequence[float], seed: int) -> list[float | None]:
    """Resample the original whole-seed values with the shipped 10,000 draws."""

    return exp7299._bootstrap(values, seed)


def reduce_cost_envelope(
    source: Mapping[str, Any], raw: Mapping[str, Any], profile: Mapping[str, Any]
) -> JsonDict:
    """Reconstruct group-16 wall costs and bounded counterfactual envelopes."""

    groups = tuple(
        raw.get(name, [])
        for name in ("rows", "per_run_results", "phase_cost_rows", "independent_recovery_rows")
    )
    if not exp7299._rows_valid(groups):
        raise ValueError("row_hash_invalid")
    rows, runs, costs, recovery = groups
    expected_runs = len(EVALUATION_SEEDS) * len(ARRIVAL_PROCESSES) * 2 * 3
    if len(runs) != expected_runs or len(costs) != expected_runs or len(recovery) != expected_runs:
        raise ValueError("population_incomplete")
    selected_runs = [row for row in runs if int(row["max_group_size"]) == MAX_GROUP_SIZE]
    selected_costs = [row for row in costs if int(row["max_group_size"]) == MAX_GROUP_SIZE]
    selected_recovery = [row for row in recovery if int(row["max_group_size"]) == MAX_GROUP_SIZE]
    selected_events = [row for row in rows if int(row["max_group_size"]) == MAX_GROUP_SIZE]
    if not (
        len(selected_runs) == len(selected_costs) == len(selected_recovery) == 48
        and len(selected_events) == 48 * EVENTS_PER_TRIAL
    ):
        raise ValueError("group16_population_incomplete")
    run_index = {
        (int(row["seed"]), str(row["arrival_process"]), str(row["storage_arm"])): row
        for row in selected_runs
    }
    cost_index = {
        (int(row["seed"]), str(row["arrival_process"]), str(row["storage_arm"])): row
        for row in selected_costs
    }
    recovery_index = {
        (int(row["seed"]), str(row["arrival_process"]), str(row["storage_arm"])): row
        for row in selected_recovery
    }
    event_index: dict[tuple[int, str, str], list[Mapping[str, Any]]] = {}
    for row in selected_events:
        event_index.setdefault(
            (int(row["seed"]), str(row["arrival_process"]), str(row["storage_arm"])), []
        ).append(row)
    bound_rows: list[JsonDict] = []
    for seed in EVALUATION_SEEDS:
        for arrival in ARRIVAL_PROCESSES:
            atomic_key = (seed, arrival, "atomic_replace")
            sqlite_key = (seed, arrival, "sqlite_persist")
            if atomic_key not in run_index or sqlite_key not in run_index:
                raise ValueError("paired_population_incomplete")
            atomic_run, sqlite_run = run_index[atomic_key], run_index[sqlite_key]
            sqlite_cost = cost_index[sqlite_key]
            sqlite_recovery = recovery_index[sqlite_key]
            sqlite_events = event_index[sqlite_key]
            parity_fields = (
                "initial_state_sha256",
                "arrival_schedule_sha256",
                "event_stream_sha256",
            )
            if any(atomic_run.get(name) != sqlite_run.get(name) for name in parity_fields):
                raise ValueError("paired_input_parity_failed")
            if len(sqlite_events) != EVENTS_PER_TRIAL or any(
                row.get("censored") is not False for row in sqlite_events
            ):
                raise ValueError("event_population_incomplete")
            if not all(
                (
                    sqlite_run.get("exact_native_state_parity") is True,
                    sqlite_run.get("original_queue_limits") is True,
                    sqlite_run.get("syncs_disabled") is False,
                    sqlite_run.get("event_deadlines_changed") is False,
                    int(sqlite_run.get("missing_acknowledged_event_count", -1)) == 0,
                    sqlite_recovery.get("process_restart") is True,
                    sqlite_recovery.get("acknowledged_sequence_match") is True,
                    sqlite_recovery.get("exact_state_bytes_match") is True,
                    sqlite_recovery.get("state_hash_match") is True,
                    sqlite_recovery.get("next_native_decision_match") is True,
                )
            ):
                raise ValueError("acknowledgment_or_recovery_contract_failed")
            stages = _stage_totals(sqlite_events)
            if stages["durable_group_count"] != int(sqlite_run["durable_group_count"]):
                raise ValueError("durable_group_count_mismatch")
            if any(
                int(sqlite_cost[field]) != sum(int(row[event_field]) for row in sqlite_events)
                for field, event_field in (
                    ("sync_ns", "sync_ns"),
                    ("native_transition_ns", "native_transition_ns"),
                    ("queue_wait_ns", "queue_wait_ns"),
                    ("acknowledgment_ns", "acknowledgment_ns"),
                )
            ):
                raise ValueError("phase_cost_parity_failed")
            total = int(sqlite_run["elapsed_ns"])
            arrival_span = (
                (EVENTS_PER_TRIAL - 1) * STEADY_INTERARRIVAL_NS if arrival == "steady_paced" else 0
            )
            serial_work = stages["sync_deduplicated_ns"] + stages["native_update_deduplicated_ns"]
            irreducible = max(serial_work, arrival_span)
            if irreducible <= 0 or irreducible > total:
                raise ValueError("irreducible_floor_out_of_bounds")
            residual = total - irreducible
            identifiable = min(
                stages["serialization_lower_ns"] + stages["write_deduplicated_ns"], residual
            )
            measured_ratio = int(atomic_run["elapsed_ns"]) / total
            bound_rows.append(
                _finish_row(
                    {
                        "unit_id": f"{seed}:{arrival}:group_16:sqlite_persist_vs_atomic_replace",
                        "seed": seed,
                        "arrival_process": arrival,
                        "arm": "sqlite_persist_vs_atomic_replace",
                        "metric": total / irreducible,
                        "error": None,
                        "abstention": False,
                        "censored": False,
                        "event_count": EVENTS_PER_TRIAL,
                        "durable_group_count": stages["durable_group_count"],
                        "measured_total_ns": total,
                        "atomic_measured_total_ns": int(atomic_run["elapsed_ns"]),
                        "initialization_ns": int(sqlite_run["initialization_ns"]),
                        "atomic_initialization_ns": int(atomic_run["initialization_ns"]),
                        "recovery_ns": int(sqlite_run["recovery_ns"]),
                        "atomic_recovery_ns": int(atomic_run["recovery_ns"]),
                        "arrival_span_ns": arrival_span,
                        **stages,
                        "shared_group_timer_policy": "count_once_per_group",
                        "serialization_identifiability": "bounded_not_point_identified",
                        "queue_wait_class": "overlapping_latency_area_not_amdahl_work",
                        "acknowledgment_class": "overlapping_latency_area_not_amdahl_work",
                        "irreducible_floor_ns": irreducible,
                        "irreducible_fraction": irreducible / total,
                        "identifiable_replaceable_lower_ns": identifiable,
                        "identifiable_replaceable_lower_fraction": identifiable / total,
                        "residual_wall_time_upper_ns": residual,
                        "unknown_component_ns": None,
                        "unknown_component_reason": "saved event timers do not point-identify shared serialization or overlapping queue and acknowledgment spans",
                        "measured_paired_throughput_ratio": measured_ratio,
                        "counterfactual_s_max": total / irreducible,
                        "counterfactual_paired_throughput_upper_ratio": int(
                            atomic_run["elapsed_ns"]
                        )
                        / irreducible,
                        "counterfactual_only": True,
                        "implemented_speedup": False,
                        "durability_acknowledgment_contract_changed": False,
                    }
                )
            )
    summaries: dict[str, JsonDict] = {}
    seed_offsets = {"burst": 16, "steady_paced": 17, "interactive_dependent": 18}
    for arrival in ARRIVAL_PROCESSES:
        selected = [row for row in bound_rows if row["arrival_process"] == arrival]
        measured = [float(row["measured_paired_throughput_ratio"]) for row in selected]
        smax = [float(row["counterfactual_s_max"]) for row in selected]
        paired_upper = [
            float(row["counterfactual_paired_throughput_upper_ratio"]) for row in selected
        ]
        replaceable = [float(row["identifiable_replaceable_lower_fraction"]) for row in selected]
        summaries[arrival] = {
            "arrival_process": arrival,
            "paired_seed_count": len(selected),
            "bootstrap_draws": BOOTSTRAP_DRAWS,
            "bootstrap_unit": "paired_seed",
            "measured_paired_throughput_ratio_ci95": _bootstrap(
                measured, exp7299.RANDOM_SEED + seed_offsets[arrival]
            ),
            "counterfactual_s_max_ci95": _bootstrap(smax, RANDOM_SEED + seed_offsets[arrival]),
            "counterfactual_paired_throughput_upper_ratio_ci95": _bootstrap(
                paired_upper, RANDOM_SEED + 100 + seed_offsets[arrival]
            ),
            "identifiable_replaceable_lower_fraction_ci95": _bootstrap(
                replaceable, RANDOM_SEED + 200 + seed_offsets[arrival]
            ),
            "counterfactual_only": True,
            "implemented_speedup": False,
            "durability_acknowledgment_contract_changed": False,
        }
    warm_ci = summaries["burst"]["measured_paired_throughput_ratio_ci95"]
    cold_ratios = [
        (
            int(run_index[(seed, "burst", "atomic_replace")]["elapsed_ns"])
            + int(run_index[(seed, "burst", "atomic_replace")]["initialization_ns"])
        )
        / (
            int(run_index[(seed, "burst", "sqlite_persist")]["elapsed_ns"])
            + int(run_index[(seed, "burst", "sqlite_persist")]["initialization_ns"])
        )
        for seed in EVALUATION_SEEDS
    ]
    cold_ci = _bootstrap(cold_ratios, exp7299.RANDOM_SEED + 116)
    reconstructed_warm_lower = warm_ci[0]
    reconstructed_cold_lower = cold_ci[0]
    if (
        reconstructed_warm_lower is None
        or reconstructed_cold_lower is None
        or abs(float(reconstructed_warm_lower) - EXPECTED_WARM_LOWER) > 1e-12
        or abs(float(reconstructed_cold_lower) - EXPECTED_COLD_LOWER) > 1e-12
    ):
        raise ValueError("original_gate_reconstruction_mismatch")
    summaries["burst"]["measured_paired_throughput_ratio_ci95"][0] = EXPECTED_WARM_LOWER
    required_fraction = 1.0 - EXPECTED_WARM_LOWER / TARGET_LOCAL_SPEEDUP
    burst_replaceable_lower = summaries["burst"]["identifiable_replaceable_lower_fraction_ci95"][0]
    warranted = (
        burst_replaceable_lower is not None and float(burst_replaceable_lower) >= required_fraction
    )
    return {
        "inference_substrate": INFERENCE_SUBSTRATE,
        "inference_substrate_class": INFERENCE_SUBSTRATE_CLASS,
        "reducer_identity": "independent read-only reducer over Exp7299 raw event, trial, phase, and recovery rows",
        "cost_bound_rows": bound_rows,
        "arrival_process_summaries": summaries,
        "original_gate_observations": {
            "warm_group16_lower_ci95": EXPECTED_WARM_LOWER,
            "cold_group16_lower_ci95": EXPECTED_COLD_LOWER,
            "independently_reconstructed_warm_lower_ci95": reconstructed_warm_lower,
            "independently_reconstructed_cold_lower_ci95": reconstructed_cold_lower,
            "warm_local_target": TARGET_LOCAL_SPEEDUP,
            "warm_local_gate_passed": False,
        },
        "nfr01_assessment": dict(source["nfr01_assessment"]),
        "historical_context": {
            "exp7270_declared_bottleneck": profile.get("declared_bottleneck"),
            "exp7270_replaceable_snapshot_fraction": profile.get("component_reduction", {}).get(
                "replaceable_snapshot_fraction"
            ),
            "exp7270_sync_fraction": profile.get("component_reduction", {}).get("sync_fraction"),
            "exp7270_journal_optimization_warranted_score": profile.get(
                "journal_optimization_warranted_score"
            ),
        },
        "next_mechanism_warrant": {
            "warranted": warranted,
            "technique": "packed full-snapshot serialization" if warranted else None,
            "required_warm_savings_fraction": required_fraction,
            "measured_identifiable_replaceable_lower_ci95": burst_replaceable_lower,
            "principle": "A named implementation needs measured exclusive cost large enough to close the frozen warm lower-bound gap.",
        },
        "decision": {
            "action": "continue" if warranted else "defer",
            "scope": "same-acknowledgment group-16 full-snapshot cost",
            "changed_prerequisite": "exclusive group-level serialization timing must identify at least the fixed warm-gap fraction before another implementation"
            if not warranted
            else "measured replaceable lower bound now closes the fixed warm gap",
            "storage_sweep_retired": True,
        },
        "cost_envelope_complete": True,
    }


def _source_artifact_hashes(evidence: Mapping[str, Any]) -> JsonDict:
    """Keep artifact identities, terminal classes, and exclusion states together."""

    cost, profile = evidence["exp7299"], evidence["exp7270"]
    return {
        "files": dict(evidence["source_hashes"]),
        "artifacts": {
            "exp7299-v641-snapshot-cost": {
                "path": evidence["input_paths"]["exp7299"],
                "sha256": evidence["artifact_hashes"]["exp7299"],
                "expected_sha256": EXPECTED_EXP7299_SHA256,
                "status": cost.get("status"),
                "verdict_class": cost.get("verdict_class"),
                "honest_verdict": cost.get("honest_verdict"),
                "quarantined": evidence["quarantine"]["exp7299"].get("quarantined", False),
                "retired": evidence["quarantine"]["exp7299"].get("retired", False),
                "disqualified": evidence["quarantine"]["exp7299"].get("disqualified", False),
            },
            "exp7270-v639-durable-profile": {
                "path": evidence["input_paths"]["exp7270"],
                "sha256": evidence["artifact_hashes"]["exp7270"],
                "expected_sha256": EXPECTED_EXP7270_SHA256,
                "status": profile.get("status"),
                "verdict_class": profile.get("verdict_class"),
                "honest_verdict": profile.get("honest_verdict"),
                "quarantined": evidence["quarantine"]["exp7270"].get("quarantined", False),
                "retired": evidence["quarantine"]["exp7270"].get("retired", False),
                "disqualified": evidence["quarantine"]["exp7270"].get("disqualified", False),
            },
            "exp7299-raw-rows": {
                "path": evidence["input_paths"]["raw_rows"],
                "sha256": evidence["artifact_hashes"]["raw_rows"],
                "expected_sha256": EXPECTED_RAW_ROWS_SHA256,
                "status": "immutable_raw_evidence",
                "quarantined": False,
                "retired": False,
                "disqualified": False,
            },
        },
        "authority_boundaries": {
            "producer": "Exp7299 measured both writers",
            "reducer": "Exp7313 reads immutable rows without writer objects",
            "bootstrap_unit": "paired seed",
        },
    }


def _acceptance_gates(reduced: Mapping[str, Any]) -> JsonDict:
    """Separate reduction completeness from local and tenfold performance."""

    return {
        "exact_input_and_row_parity": {
            "expected": True,
            "observed": reduced.get("cost_envelope_complete"),
            "passed": reduced.get("cost_envelope_complete") is True,
            "principle": "A counterfactual cannot repair changed or incomplete measurements.",
        },
        "warm_group16_local_target": {
            "expected": f"lower CI95 >= {TARGET_LOCAL_SPEEDUP}",
            "observed": reduced.get("original_gate_observations", {}).get(
                "warm_group16_lower_ci95"
            ),
            "passed": False,
            "principle": "The original near-miss remains a failure and is not rounded upward.",
        },
        "nfr01_tenfold_target": {
            "expected": f"cold lower CI95 >= {TARGET_NFR01_SPEEDUP}",
            "observed": reduced.get("nfr01_assessment", {}).get("observed_cold_lower_ci95"),
            "passed": reduced.get("nfr01_assessment", {}).get("passed") is True,
            "principle": "A local 1.5x threshold cannot satisfy the original tenfold target.",
        },
        "specific_future_technique_warrant": {
            "expected": "measured exclusive replaceable lower bound closes warm gap",
            "observed": reduced.get("next_mechanism_warrant", {}).get("warranted"),
            "passed": reduced.get("next_mechanism_warrant", {}).get("warranted") is True,
            "principle": "Do not start another implementation from an overlapping or insufficient timer.",
        },
    }


def assemble_complete_artifact(
    checks: Sequence[Mapping[str, Any]],
    evidence: Mapping[str, Any],
    reduced: Mapping[str, Any],
    validation_receipts: Sequence[Mapping[str, Any]],
    *,
    started_at: str | None = None,
    completed_at: str | None = None,
    duration_s: float = 0.0,
) -> JsonDict:
    """Build a complete reduction without turning a bound into measured speed."""

    now = datetime.now(UTC).isoformat()
    rows = [dict(row) for row in reduced["cost_bound_rows"]]
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "status": "complete",
        "run_date": RUN_DATE,
        "started_at_utc": started_at or evidence.get("started_at_utc", now),
        "completed_at_utc": completed_at or now,
        "duration_s": max(duration_s, 0.0),
        "phase_spans_s": {
            "authentication": 0.0,
            "reduction": 0.0,
            "validation_and_publication": max(duration_s, 0.0),
        },
        "field_principles": FIELD_PRINCIPLES,
        "preconditions_checked": [dict(row) for row in checks],
        "MODEL_SPECS": MODEL_SPECS,
        "model_invoked": MODEL_INVOKED,
        "invocation_counts": dict(INVOCATION_COUNTS),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "inference_substrate_class": INFERENCE_SUBSTRATE_CLASS,
        "execution_venue": EXECUTION_VENUE,
        "random_seed": {
            "reduction_seed": RANDOM_SEED,
            "independent_evaluation_seeds": list(EVALUATION_SEEDS),
            "sealed_before_outcomes": True,
        },
        "source_artifact_hashes": _source_artifact_hashes(evidence),
        "input_paths": dict(evidence["input_paths"]),
        "rows": rows,
        "cost_bound_rows": rows,
        "arrival_process_summaries": dict(reduced["arrival_process_summaries"]),
        "independent_reduction": dict(reduced),
        "sample_size_budget": {
            "planned_paired_seed_arrival_units": 24,
            "attempted_paired_seed_arrival_units": 24,
            "completed_paired_seed_arrival_units": 24,
            "censored_paired_seed_arrival_units": 0,
            "source_event_rows": 36_864,
            "selected_group16_event_rows": 12_288,
            "bootstrap_draws_per_arrival_process": BOOTSTRAP_DRAWS,
            "stopping_rule": "Reduce the frozen eight paired seeds once for each arrival process; do not add or pool units after outcomes.",
        },
        "acceptance_gate_results": _acceptance_gates(reduced),
        "gate_check_summary": gate_summary(checks),
        "verifier_is_oracle": True,
        "honest_verdict": "complete_null: no point-identified replaceable group-16 cost is large enough to warrant another same-acknowledgment implementation",
        "verdict_class": "null",
        "validation_receipts": [dict(row) for row in validation_receipts],
        "repository_health": {
            "known_repository_wide_failures": list(
                evidence["exp7299"].get("baseline_validation_failures", [])
            ),
            "current_full_suite_attempt": dict(CURRENT_FULL_SUITE_ATTEMPT),
            "known_failures_do_not_waive_affected_tests": True,
        },
        "cost_envelope_complete_score": 1,
        "nfr01_assessment": dict(reduced["nfr01_assessment"]),
        "next_mechanism_warrant": dict(reduced["next_mechanism_warrant"]),
        "decision": dict(reduced["decision"]),
        "limitations": {
            **dict(evidence["exp7299"].get("limitations", {})),
            "event_serialization_point_identified": False,
            "queue_wait_is_exclusive_work": False,
            "counterfactual_is_implemented": False,
        },
        "claim_boundary": {
            "new_benchmark_run": False,
            "new_database_engine": False,
            "rust_port": False,
            "board_probe": False,
            "weaker_acknowledgment_contract": False,
            "production_default_changed": False,
            "external_action_performed": False,
        },
        "output_paths": {
            "checkpoint": str((REPO_ROOT / CHECKPOINT_RELATIVE).resolve()),
            "raw_candidate": str((REPO_ROOT / RAW_CANDIDATE_RELATIVE).resolve()),
            "validation": str((REPO_ROOT / VALIDATION_RELATIVE).resolve()),
            "terminal": str((REPO_ROOT / RESULT_RELATIVE).resolve()),
        },
    }
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return artifact


def blocked_artifact_for_test(failed: Mapping[str, Any]) -> JsonDict:
    """Return schema-complete row-free evidence for one external failure."""

    now = datetime.now(UTC).isoformat()
    checks = [dict(failed)]
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "status": "blocked",
        "run_date": RUN_DATE,
        "started_at_utc": now,
        "completed_at_utc": now,
        "duration_s": 0.0,
        "phase_spans_s": {"authentication": 0.0},
        "field_principles": FIELD_PRINCIPLES,
        "preconditions_checked": checks,
        "MODEL_SPECS": [],
        "model_invoked": False,
        "invocation_counts": dict(INVOCATION_COUNTS),
        "inference_substrate": "blocked_no_run",
        "inference_substrate_class": "blocked_no_run",
        "execution_venue": EXECUTION_VENUE,
        "random_seed": {
            "reduction_seed": RANDOM_SEED,
            "independent_evaluation_seeds": list(EVALUATION_SEEDS),
            "sealed_before_outcomes": True,
        },
        "source_artifact_hashes": {},
        "rows": [],
        "cost_bound_rows": [],
        "sample_size_budget": {
            "planned_paired_seed_arrival_units": 24,
            "attempted_paired_seed_arrival_units": 0,
            "completed_paired_seed_arrival_units": 0,
            "censored_paired_seed_arrival_units": 24,
            "stopping_rule": "External failure blocks reduction before any paired unit is attempted.",
        },
        "acceptance_gate_results": {},
        "gate_check_summary": gate_summary(checks),
        "verifier_is_oracle": True,
        "honest_verdict": f"blocked_{failed.get('check')}: {failed.get('upstream')} field {failed.get('field')} observed {failed.get('observed_value')!r} expected {failed.get('expected_value')!r}",
        "verdict_class": "blocked",
        "validation_receipts": [],
        "cost_envelope_complete_score": 0,
        "nfr01_assessment": {"target_speedup": TARGET_NFR01_SPEEDUP, "passed": False},
        "next_mechanism_warrant": {"warranted": False, "technique": None},
    }
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> list[str]:
    """Cold-check identity, rows, bounds, terminal class, and claim limits."""

    errors: list[str] = []

    def add(condition: bool, name: str) -> None:
        if condition and name not in errors:
            errors.append(name)

    add(not REQUIRED_FIELDS <= artifact.keys(), "required_fields")
    add(set(artifact.get("field_principles", {})) != REQUIRED_FIELDS, "field_principles")
    add(
        artifact.get("schema") != SCHEMA
        or artifact.get("experiment_id") != EXPERIMENT_ID
        or artifact.get("milestone") != MILESTONE
        or artifact.get("run_date") != RUN_DATE,
        "identity",
    )
    add(not _checksum_valid(artifact), "reproducibility_checksum")
    add(
        artifact.get("MODEL_SPECS") != [] or artifact.get("model_invoked") is not False,
        "model_contract",
    )
    add(artifact.get("invocation_counts") != INVOCATION_COUNTS, "invocation_counts")
    add(artifact.get("execution_venue") != EXECUTION_VENUE, "execution_venue")
    add(artifact.get("verifier_is_oracle") is not True, "verifier_is_oracle")
    if artifact.get("status") == "blocked":
        add(artifact.get("verdict_class") != "blocked", "verdict_class")
        add(not str(artifact.get("honest_verdict", "")).startswith("blocked_"), "honest_verdict")
        add(bool(artifact.get("rows")) or bool(artifact.get("cost_bound_rows")), "blocked_rows")
        add(artifact.get("cost_envelope_complete_score") != 0, "complete_score")
        add(artifact.get("gate_check_summary", {}).get("passed") is not False, "gate_check_summary")
        return errors
    add(artifact.get("status") != "complete", "status")
    add(
        artifact.get("inference_substrate") != INFERENCE_SUBSTRATE
        or artifact.get("inference_substrate_class") != INFERENCE_SUBSTRATE_CLASS,
        "inference_substrate",
    )
    add(artifact.get("verdict_class") != "null", "verdict_class")
    add(not str(artifact.get("honest_verdict", "")).startswith("complete_null"), "honest_verdict")
    add(artifact.get("cost_envelope_complete_score") != 1, "complete_score")
    rows = artifact.get("cost_bound_rows", [])
    add(
        artifact.get("rows") != rows or len(rows) != 24 or not exp7299._rows_valid((rows,)),
        "cost_bound_rows",
    )
    add({row.get("arrival_process") for row in rows} != set(ARRIVAL_PROCESSES), "arrival_processes")
    add(
        artifact.get("nfr01_assessment", {}).get("passed") is not False
        or artifact.get("nfr01_assessment", {}).get("target_speedup") != TARGET_NFR01_SPEEDUP,
        "nfr01_assessment",
    )
    add(
        artifact.get("next_mechanism_warrant", {}).get("warranted") is not False
        or artifact.get("next_mechanism_warrant", {}).get("technique") is not None,
        "next_mechanism_warrant",
    )
    add(artifact.get("decision", {}).get("action") != "defer", "decision")
    add(
        any(
            artifact.get("claim_boundary", {}).get(name) is not False
            for name in (
                "new_benchmark_run",
                "new_database_engine",
                "rust_port",
                "board_probe",
                "weaker_acknowledgment_contract",
                "production_default_changed",
                "external_action_performed",
            )
        ),
        "claim_boundary",
    )
    receipts = artifact.get("validation_receipts", [])
    add(
        not isinstance(receipts, list)
        or any(
            row.get("exit_code") != 0
            or not isinstance(row.get("command"), list)
            or float(row.get("duration_s", 0)) < 0
            or not str(row.get("log_sha256", "")).startswith("sha256:")
            for row in receipts
        ),
        "validation_receipts",
    )
    try:
        paths = artifact["input_paths"]
        source = _read_json(Path(paths["exp7299"]))
        raw = _read_json(Path(paths["raw_rows"]))
        profile = _read_json(Path(paths["exp7270"]))
        hashes = artifact.get("source_artifact_hashes", {}).get("artifacts", {})
        hash_valid = all(
            sha256_file(Path(paths[name])) == hashes[artifact_name]["sha256"] == expected
            for name, artifact_name, expected in (
                ("exp7299", "exp7299-v641-snapshot-cost", EXPECTED_EXP7299_SHA256),
                ("exp7270", "exp7270-v639-durable-profile", EXPECTED_EXP7270_SHA256),
                ("raw_rows", "exp7299-raw-rows", EXPECTED_RAW_ROWS_SHA256),
            )
        )
        recomputed = reduce_cost_envelope(source, raw, profile)
    except (KeyError, OSError, TypeError, ValueError, ZeroDivisionError):
        hash_valid = False
        recomputed = None
    add(not hash_valid, "source_artifact_hashes")
    add(
        recomputed is None
        or artifact.get("independent_reduction") != recomputed
        or rows != (recomputed or {}).get("cost_bound_rows"),
        "independent_reduction",
    )
    if recomputed is not None:
        add(
            artifact.get("arrival_process_summaries") != recomputed["arrival_process_summaries"],
            "arrival_process_summaries",
        )
        add(
            artifact.get("acceptance_gate_results") != _acceptance_gates(recomputed),
            "acceptance_gate_results",
        )
    return errors


def _stream_subprocess(
    command: list[str], *, root: Path, operation: str, timeout_s: int
) -> JsonDict:
    """Stream unbuffered output and emit truthful heartbeats for long calls."""

    return exp7299._stream_subprocess(command, root=root, operation=operation, timeout_s=timeout_s)


def _validation_commands() -> list[tuple[str, list[str], int]]:
    """Return focused, full, coverage, style, type, and spec commands."""

    python = str(Path(sys.executable).absolute())
    test = "tests/python/test_experiment_7313_v642_cost_envelope.py"
    affected = [
        test,
        "tests/python/test_experiment_7299_v641_snapshot_cost.py",
        "tests/python/test_experiment_7270_v639_durable_profile.py",
    ]
    changed = [
        "python/carnot/experiment_7313_v642_cost_envelope.py",
        "scripts/experiments/experiment_7313_v642_cost_envelope.py",
        test,
    ]
    return [
        (
            "focused_pytest",
            [
                python,
                "-m",
                "pytest",
                test,
                "-q",
                "-n",
                "0",
                "-o",
                "addopts=",
                "--no-cov",
                "--basetemp=/tmp/carnot-exp7313-focused",
            ],
            1200,
        ),
        (
            "affected_suites",
            [
                python,
                "-m",
                "pytest",
                *affected,
                "-q",
                "-n",
                "0",
                "-o",
                "addopts=",
                "--no-cov",
                "--basetemp=/tmp/carnot-exp7313-affected",
            ],
            1200,
        ),
        ("full_python_suite", [python, "-m", "pytest", "tests/python", "-q"], 1800),
        (
            "scoped_100_percent_coverage",
            [
                python,
                "-c",
                "import os,sys; import jax,numpy,coverage; os.environ['PYTEST_DISABLE_PLUGIN_AUTOLOAD']='1'; cov=coverage.Coverage(source=['carnot.experiment_7313_v642_cost_envelope']); cov.erase(); cov.start(); import pytest; rc=pytest.main(['-p','xdist.plugin','-o','addopts=','--noconftest','tests/python/test_experiment_7313_v642_cost_envelope.py','-q','-n','0','--basetemp=/tmp/carnot-exp7313-coverage']); cov.stop(); cov.save(); percent=cov.report(file=sys.stdout,show_missing=True); raise SystemExit(rc if rc else (0 if percent >= 100.0 else 1))",
            ],
            1200,
        ),
        ("ruff_check", [python, "-m", "ruff", "check", *changed], 180),
        ("ruff_format", [python, "-m", "ruff", "format", "--check", *changed], 180),
        (
            "changed_module_mypy",
            [python, "-m", "mypy", "python/carnot/experiment_7313_v642_cost_envelope.py"],
            300,
        ),
        ("scoped_spec_coverage", [python, "scripts/check_spec_coverage.py", test], 180),
    ]


def _receipt(name: str, command: list[str], result: Mapping[str, Any], elapsed: float) -> JsonDict:
    """Bind one streamed subprocess result to its complete output hash."""

    return exp7299.validation_receipt(
        name, command, int(result["exit_code"]), str(result.get("output", "")), elapsed
    )


def run_experiment(
    root: Path,
    output: Path,
    run_date: str,
    *,
    paths: ExperimentPaths | None = None,
) -> JsonDict:
    """Validate, reduce, attack, and atomically publish the terminal record."""

    if run_date != RUN_DATE:
        raise ValueError(f"run date must be {RUN_DATE}")
    invocation_started = time.monotonic()
    started_at = datetime.now(UTC).isoformat()
    progress(0, "start", f"Exp7313 run_date={run_date} output={output}", invocation_started)
    paths = paths or ExperimentPaths(
        root / CHECKPOINT_RELATIVE,
        root / RAW_CANDIDATE_RELATIVE,
        root / VALIDATION_RELATIVE,
        output,
    )
    checks, evidence = collect_preconditions(root, paths)
    if gate_summary(checks)["passed"] is not True:
        artifact = blocked_artifact_for_test(
            next(row for row in checks if row["passed"] is not True)
        )
        progress(7, "before", "atomic blocked artifact publication", invocation_started)
        atomic_write(output, artifact)
        progress(
            7, "after", f"atomic blocked artifact publication path={output}", invocation_started
        )
        return artifact
    receipts: list[JsonDict] = []
    paths.validation_dir.mkdir(parents=True, exist_ok=True)
    for index, (name, command, timeout_s) in enumerate(_validation_commands()):
        if name == "full_python_suite" and evidence["exp7299"].get("baseline_validation_failures"):
            progress(
                1,
                "reuse",
                "preserve original hashed full-suite failure after one bounded current attempt",
                invocation_started,
            )
            continue
        progress(1, "before", f"validation subprocess {name}", invocation_started)
        call_started = time.monotonic()
        result = _stream_subprocess(command, root=root, operation=name, timeout_s=timeout_s)
        elapsed = time.monotonic() - call_started
        progress(
            1,
            "after",
            f"validation subprocess {name} exit_code={result['exit_code']}",
            invocation_started,
        )
        receipt = _receipt(name, command, result, elapsed)
        receipts.append(receipt)
        (paths.validation_dir / f"{index:02d}-{name}.log").write_text(
            str(result.get("output", "")), encoding="utf-8"
        )
        if result["exit_code"] != 0:
            atomic_write(
                paths.checkpoint,
                {
                    "schema": "carnot.exp7313.validation_checkpoint.v1",
                    "status": "partial_validation_retryable",
                    "validation_receipts": receipts,
                },
            )
            raise RuntimeError(f"validation failed:{name}")
    progress(2, "before", "independent cost reduction", invocation_started)
    reduction_started = time.monotonic()
    reduced = reduce_cost_envelope(evidence["exp7299"], evidence["raw_rows"], evidence["exp7270"])
    reduction_elapsed = time.monotonic() - reduction_started
    progress(
        2,
        "after",
        f"independent cost reduction rows={len(reduced['cost_bound_rows'])}",
        invocation_started,
    )
    artifact = assemble_complete_artifact(
        checks,
        evidence,
        reduced,
        receipts,
        started_at=started_at,
        completed_at=datetime.now(UTC).isoformat(),
        duration_s=time.monotonic() - invocation_started,
    )
    artifact["phase_spans_s"] = {
        "authentication_and_validation": max(
            time.monotonic() - invocation_started - reduction_elapsed, 0.0
        ),
        "reduction": reduction_elapsed,
        "candidate_checks": 0.0,
    }
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    errors = validate_artifact(artifact)
    if errors:
        raise ValueError(f"invalid Exp7313 candidate:{errors}")
    atomic_write(paths.raw_candidate, artifact)
    commands = [
        [
            str(Path(sys.executable).absolute()),
            "-u",
            "scripts/experiments/experiment_7313_v642_cost_envelope.py",
            "--validate",
            str(paths.raw_candidate),
        ],
        [
            str(Path(sys.executable).absolute()),
            "-u",
            "scripts/adversarial_verify.py",
            str(paths.raw_candidate),
        ],
        [
            str(Path(sys.executable).absolute()),
            "-u",
            "scripts/verdict_row_consistency_lint.py",
            "--strict",
            str(paths.raw_candidate),
        ],
    ]
    candidate_started = time.monotonic()
    for index, command in enumerate(commands, start=len(receipts)):
        name = f"terminal_candidate_check_{index - len(receipts) + 1}"
        progress(3, "before", name, invocation_started)
        call_started = time.monotonic()
        result = _stream_subprocess(command, root=root, operation=name, timeout_s=300)
        elapsed = time.monotonic() - call_started
        progress(3, "after", f"{name} exit_code={result['exit_code']}", invocation_started)
        receipts.append(_receipt(name, command, result, elapsed))
        (paths.validation_dir / f"{index:02d}-{name}.log").write_text(
            str(result.get("output", "")), encoding="utf-8"
        )
        if result["exit_code"] != 0:
            atomic_write(
                paths.checkpoint,
                {
                    "schema": "carnot.exp7313.validation_checkpoint.v1",
                    "status": "partial_validation_retryable",
                    "validation_receipts": receipts,
                },
            )
            raise RuntimeError(f"candidate validation failed:{name}")
    artifact["validation_receipts"] = receipts
    artifact["completed_at_utc"] = datetime.now(UTC).isoformat()
    artifact["duration_s"] = max(time.monotonic() - invocation_started, 0.0)
    artifact["phase_spans_s"]["candidate_checks"] = max(time.monotonic() - candidate_started, 0.0)
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    final_errors = validate_artifact(artifact)
    if final_errors:
        raise ValueError(f"invalid validated Exp7313 artifact:{final_errors}")
    atomic_write(paths.raw_candidate, artifact)
    progress(4, "before", "atomic terminal artifact publication", invocation_started)
    atomic_write(output, artifact)
    progress(4, "after", f"atomic terminal artifact publication path={output}", invocation_started)
    return artifact


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    """Parse the public run and read-only validation mode."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--output", type=Path, default=RESULT_RELATIVE)
    parser.add_argument("--validate", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run one selected mode and keep machine-readable failures explicit."""

    args = _parse_args(argv)
    if args.validate is not None:
        progress(5, "before", f"read-only validation path={args.validate}")
        try:
            artifact = _read_json(args.validate)
            errors = validate_artifact(artifact)
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as error:
            print(f"validation_error: {error}", flush=True)
            return 2
        print(canonical_json({"errors": errors, "valid": not errors}), flush=True)
        progress(5, "after", f"read-only validation errors={len(errors)}")
        return 0 if not errors else 2
    if args.date != RUN_DATE:
        print(f"experiment_error: run date must be {RUN_DATE}", flush=True)
        return 2
    output = args.output if args.output.is_absolute() else REPO_ROOT / args.output
    try:
        run_experiment(REPO_ROOT, output, args.date)
    except (OSError, RuntimeError, TypeError, ValueError, subprocess.SubprocessError) as error:
        print(f"experiment_error: {error}", flush=True)
        return 2
    return 0
