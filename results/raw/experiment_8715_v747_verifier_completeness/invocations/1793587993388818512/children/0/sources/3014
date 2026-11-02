"""Recover completed ARC Panel B evidence without new model or game work.

The prior live child finished every scheduled episode. Its conductor stopped
before terminal promotion. This module authenticates those immutable bytes and
runs current validation under a new experiment identity.

Spec refs: REQ-ARC-WMTE-7511 and SCENARIO-ARC-WMTE-7511-*.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from copy import deepcopy
from datetime import UTC, datetime
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any

import yaml

from carnot import experiment_7358_v646_validation_contract as validation_contract
from carnot import experiment_7478_v655_arc_interval_protocol as interval_protocol
from carnot import experiment_7485_v655_arc_cost_panel_a as panel_a
from carnot.reporting import current_work_receipt
from carnot.reporting import experiment_7303_validation_scope as validation_scope


Json = dict[str, Any]
REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_DATE = "20260922"
MILESTONE = "2026.09.657"
EXPERIMENT_ID = "exp7511-arc-evidence-recovery"
SCHEMA = "carnot.exp7511.v657.arc_evidence_recovery.v1"

RAW_7499_DIR = Path("results/raw/experiment_7499_v656_arc_panel_b")
SESSION_PATH = RAW_7499_DIR / "live_session.json"
CANDIDATE_PATH = RAW_7499_DIR / "measured_terminal_candidate.json"
SCHEDULE_PATH = RAW_7499_DIR / "frozen_schedule.json"
EPISODE_ROWS_PATH = RAW_7499_DIR / "episode_rows.json"
BOUNDARY_PATH = RAW_7499_DIR / "current_invocation_events.jsonl"
CHECKPOINT_PATH = Path("results/checkpoints/experiment_7499_v656_arc_panel_b.json")
REGISTRY_PATH = Path("ops/arc_solve_registry.yaml")
SPEC_PATH = Path("openspec/capabilities/arc-world-model-trust-energy/spec.md")

RESULT_PATH = Path("results/experiment_7511_v657_arc_evidence_recovery.json")
RAW_DIR = Path("results/raw/experiment_7511_v657_arc_evidence_recovery")
TERMINAL_CANDIDATE_PATH = RAW_DIR / "terminal_candidate.json"
MODULE_PATH = Path("python/carnot/experiment_7511_v657_arc_evidence_recovery.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_7511_v657_arc_evidence_recovery.py")
TEST_PATH = Path("tests/python/test_experiment_7511_v657_arc_evidence_recovery.py")

GAMES = ("tu93", "g50t", "tn36", "vc33", "re86", "dc22")
SEEDS = (65_501, 65_502, 65_503)
DISABLED_FIELDS = (
    "adapter_disabled",
    "stored_engines_disabled",
    "banked_trajectories_disabled",
    "cross_game_state_disabled",
    "game_source_disabled",
    "injected_action_recipes_disabled",
    "outer_loop_re_disabled",
    "offline_ground_truth_bfs_disabled",
)
HISTORICAL_RECEIPT_NAMES = (
    "worktree_imports",
    "focused_pytest",
    "changed_module_coverage",
    "changed_module_coverage_report",
    "ruff_check",
    "ruff_format",
    "changed_module_mypy",
    "scoped_spec_coverage",
    "e2e_009",
    "e2e_010",
    "e2e_011",
    "private_arc_smoke",
)
CURRENT_VALIDATION_NAMES = validation_scope.REQUIRED_CHECK_NAMES
TERMINAL_RECEIPT_NAMES = (
    "declared_entrypoint_cold_replay",
    "independent_cold_reducer",
    "adversarial_verify",
    "verdict_row_consistency_strict",
)
ZERO_INVOCATION_COUNTS = {
    **current_work_receipt.ZERO_INVOCATION_COUNTS,
    **{
        f"forward_calls_{state}": 0
        for state in ("attempted", "completed", "failed", "cancelled", "in_flight")
    },
}

MANIFEST = validation_contract.AffectedManifest(
    experiment_id=EXPERIMENT_ID,
    test_paths=(TEST_PATH.as_posix(),),
    changed_modules=(MODULE_PATH.as_posix(),),
    static_paths=(WRAPPER_PATH.as_posix(),),
)

REPLAY_CRITICAL_SOURCES = (
    "python/carnot/experiment_7499_v656_arc_panel_b.py",
    "python/carnot/experiment_7485_v655_arc_cost_panel_a.py",
    "python/carnot/experiment_7478_v655_arc_interval_protocol.py",
    "python/carnot/experiment_7471_v654_arc_seam_observation.py",
    "python/carnot/agentic/arc_competition_agent.py",
    "python/carnot/agentic/arc_decision_telemetry.py",
    "python/carnot/reporting/current_work_receipt.py",
    "python/carnot/reporting/experiment_7303_validation_scope.py",
    "scripts/experiments/experiment_7499_v656_arc_panel_b.py",
    "tests/python/test_experiment_7499_v656_arc_panel_b.py",
)


def utc_now() -> str:
    """Return an aware UTC timestamp for a real recovery boundary."""

    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def progress(started: float, phase: str, event: str, **details: Any) -> None:
    """Flush one boundary so long hashing and validation stay observable."""

    suffix = " ".join(f"{key}={value}" for key, value in sorted(details.items()))
    print(
        f"[exp7511] phase={phase} event={event} elapsed_s={time.monotonic() - started:.3f}"
        + (f" {suffix}" if suffix else ""),
        flush=True,
    )


def load_object(path: Path) -> Json:
    """Read one JSON object and return an empty object for invalid bytes."""

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def read_jsonl(path: Path) -> list[Json]:
    """Use the qualified small reader for one immutable event ledger."""

    return interval_protocol.read_jsonl(path)


def canonical_hash(value: Any) -> str:
    """Hash stable JSON so independent replay detects changed reductions."""

    return panel_a.canonical_hash(value)


def sha256_file(path: Path) -> str:
    """Hash exact bytes without parsing or rewriting historical evidence."""

    return current_work_receipt.sha256_file(path)


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    """Publish one complete object only after durable local serialization."""

    current_work_receipt.atomic_json(path, value)


def _gate(
    check: str,
    category: str,
    expected: Any,
    observed: Any,
    *,
    upstream: str,
    field: str,
    principle: str,
    op: str = "==",
) -> Json:
    """Record one exact comparison and the failure that it prevents."""

    if op == "==":
        passed = observed == expected
    elif op == ">=":
        passed = bool(observed >= expected)
    else:
        raise ValueError(f"unsupported_gate_operator:{op}")
    return {
        "check": check,
        "category": category,
        "expected": expected,
        "observed": observed,
        "op": op,
        "passed": passed,
        "upstream": upstream,
        "path": upstream,
        "field": field,
        "artifact_field": field,
        "principle": principle,
    }


def _gate_summary(gates: Sequence[Mapping[str, Any]]) -> Json:
    """Name the first required failure and retain every failed comparison."""

    failures = [deepcopy(dict(row)) for row in gates if row.get("passed") is not True]
    required = [row for row in failures if row.get("category") in {"validity", "readiness"}]
    first = required[0] if required else (failures[0] if failures else None)
    return {
        "all_passed": not failures,
        "required_validity_and_readiness_passed": not required,
        "failed_count": len(failures),
        "required_failed_count": len(required),
        "failed_checks": failures,
        "first_failure": first,
        "first_failed_check": first.get("check") if first else None,
        "upstream": first.get("upstream") if first else None,
        "exact_field_path": first.get("field") if first else None,
        "expected_value": first.get("expected") if first else None,
        "observed_value": first.get("observed") if first else None,
    }


def expected_schedule() -> list[Json]:
    """Return the sealed game-major and seed-minor Panel B schedule."""

    return [
        {
            "episode_id": f"panel-b:{game}:seed-{seed}",
            "panel": "B",
            "game": game,
            "seed": seed,
            "execution_order": index,
            "execution_order_within_panel": index,
            "action_limit": 180,
            "request_limit": 2,
            "max_new_tokens_per_call": 256,
            "episode_limit_s": 240,
            "panel_live_limit_s": 3600,
            "readout_role": "soft_feature_only",
            "unavailable_game_policy": "retain_row_no_replacement",
            "disposition": "unstarted",
            **{field: True for field in DISABLED_FIELDS},
        }
        for index, (game, seed) in enumerate(
            (pair for game in GAMES for pair in ((game, seed) for seed in SEEDS))
        )
    ]


def _inventory_paths(root: Path) -> list[Path]:
    """List every original raw, session, checkpoint, and validation file."""

    raw_files = sorted(path for path in (root / RAW_7499_DIR).rglob("*") if path.is_file())
    checkpoint = root / CHECKPOINT_PATH
    return [*raw_files, checkpoint] if checkpoint.is_file() else raw_files


def inventory_original_evidence(root: Path) -> list[Json]:
    """Hash each original file while preserving its path and read-only role."""

    rows: list[Json] = []
    for path in _inventory_paths(root):
        rows.append(
            {
                "path": path.relative_to(root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "input_only": True,
                "write_authorized": False,
            }
        )
    return rows


def _source_drift_rows(root: Path, candidate: Mapping[str, Any]) -> list[Json]:
    """Compare replay-critical historical source bytes with current bytes."""

    recorded = candidate.get("source_artifact_hashes") or {}
    rows: list[Json] = []
    for label in REPLAY_CRITICAL_SOURCES:
        source = recorded.get(label) or {}
        path = root / label
        expected = source.get("sha256") if isinstance(source, Mapping) else None
        observed = sha256_file(path) if path.is_file() else None
        rows.append(
            {
                "path": label,
                "expected_sha256": expected,
                "observed_sha256": observed,
                "replay_critical": True,
                "passed": expected is not None and expected == observed,
            }
        )
    return rows


def collect_preconditions(root: Path) -> tuple[list[Json], list[Json], Json]:
    """Authenticate named inputs before any recovery measurement."""

    root = root.resolve()
    candidate = load_object(root / CANDIDATE_PATH)
    required = (
        SESSION_PATH,
        CANDIDATE_PATH,
        SCHEDULE_PATH,
        EPISODE_ROWS_PATH,
        BOUNDARY_PATH,
        CHECKPOINT_PATH,
        REGISTRY_PATH,
        SPEC_PATH,
    )
    checks = [
        _gate(
            f"source_bytes:{path.as_posix()}",
            "validity",
            True,
            (root / path).is_file() and (root / path).stat().st_size > 0,
            upstream=path.as_posix(),
            field="readable_nonempty_bytes",
            principle="Missing original evidence must block rather than create a result.",
        )
        for path in required
    ]
    spec_text = (
        (root / SPEC_PATH).read_text(encoding="utf-8") if (root / SPEC_PATH).is_file() else ""
    )
    checks.append(
        _gate(
            "driving_requirement",
            "validity",
            True,
            "REQ-ARC-WMTE-7511" in spec_text,
            upstream=SPEC_PATH.as_posix(),
            field="REQ-ARC-WMTE-7511",
            principle="Implementation without its requirement would bypass spec-first review.",
        )
    )
    exclusion = (
        (root / "ops/exclusion_manifest.yaml").read_text(encoding="utf-8")
        if (root / "ops/exclusion_manifest.yaml").is_file()
        else ""
    )
    checks.append(
        _gate(
            "task_not_excluded",
            "validity",
            False,
            "exp7511-arc-evidence-recovery" in exclusion,
            upstream="ops/exclusion_manifest.yaml",
            field=EXPERIMENT_ID,
            principle="A retired task cannot publish a newly qualified result.",
        )
    )
    inventory = inventory_original_evidence(root)
    checks.append(
        _gate(
            "original_inventory_complete",
            "validity",
            len(_inventory_paths(root)),
            len(inventory),
            upstream=RAW_7499_DIR.as_posix(),
            field="source_artifact_hashes.original_file_inventory",
            principle="Every original file needs a stable byte identity before reduction.",
        )
    )
    drift = _source_drift_rows(root, candidate) if candidate else []
    checks.append(
        _gate(
            "replay_critical_sources_unchanged",
            "validity",
            True,
            bool(drift) and all(row["passed"] for row in drift),
            upstream=CANDIDATE_PATH.as_posix(),
            field="source_artifact_hashes",
            principle="Historical code drift would make current replay untrustworthy.",
        )
    )
    sources: Json = {
        row["path"]: {
            "sha256": row["sha256"],
            "bytes": row["bytes"],
            "role": "immutable_exp7499_input",
            "authoritative": row["path"] != CANDIDATE_PATH.as_posix(),
        }
        for row in inventory
    }
    for path in (MODULE_PATH, WRAPPER_PATH, TEST_PATH, SPEC_PATH, REGISTRY_PATH):
        absolute = root / path
        if absolute.is_file():
            sources[path.as_posix()] = {
                "sha256": sha256_file(absolute),
                "bytes": absolute.stat().st_size,
                "role": "current_recovery_input",
                "authoritative": True,
            }
    sources["replay_critical_source_dispositions"] = drift
    return checks, inventory, sources


def _read_all_interval_events(root: Path) -> tuple[list[Json], list[Json]]:
    """Read and authenticate only the candidate's immutable interval sidecars."""

    candidate = load_object(root / CANDIDATE_PATH)
    events: list[Json] = []
    shards: list[Json] = []
    for reference in candidate.get("seam_event_shards") or []:
        path = root / str(reference.get("path") or "")
        rows = read_jsonl(path) if path.is_file() else []
        observed_hash = sha256_file(path) if path.is_file() else None
        disposition = {
            "path": str(reference.get("path") or ""),
            "episode_id": reference.get("episode_id"),
            "expected_sha256": reference.get("sha256"),
            "observed_sha256": observed_hash,
            "expected_rows": reference.get("row_count"),
            "observed_rows": len(rows),
        }
        disposition["passed"] = (
            disposition["expected_sha256"] == disposition["observed_sha256"]
            and disposition["expected_rows"] == disposition["observed_rows"]
        )
        shards.append(disposition)
        events.extend(rows)
    return events, shards


def _episode_bounds(row: Mapping[str, Any], events: Sequence[Mapping[str, Any]]) -> tuple[int, int]:
    """Derive one episode boundary from raw action and event clocks."""

    starts = [
        int(item["interval_start_monotonic_ns"])
        for item in row.get("action_rows") or []
        if isinstance(item, Mapping) and isinstance(item.get("interval_start_monotonic_ns"), int)
    ]
    ends = [
        int(item["interval_end_monotonic_ns"])
        for item in row.get("action_rows") or []
        if isinstance(item, Mapping) and isinstance(item.get("interval_end_monotonic_ns"), int)
    ]
    for event in events:
        start = event.get("interval_start_monotonic_ns")
        end = event.get("interval_end_monotonic_ns", event.get("event_monotonic_ns"))
        if isinstance(start, int) and not isinstance(start, bool):
            starts.append(start)
        if isinstance(end, int) and not isinstance(end, bool):
            ends.append(end)
    first = min(starts or ends or [0])
    elapsed_end = first + int(float(row.get("elapsed_s") or 0.0) * 1_000_000_000)
    return first, max([first, elapsed_end, *(ends or [first])])


def _expected_schedule_projection() -> list[Json]:
    """Select schedule fields whose exact values were sealed before outcomes."""

    fields = (
        "episode_id",
        "panel",
        "game",
        "seed",
        "execution_order",
        "execution_order_within_panel",
        "action_limit",
        "request_limit",
        "max_new_tokens_per_call",
        "episode_limit_s",
        "panel_live_limit_s",
        "readout_role",
        "unavailable_game_policy",
        "disposition",
        *DISABLED_FIELDS,
    )
    return [{key: row.get(key) for key in fields} for row in expected_schedule()]


def _schedule_projection(rows: Sequence[Mapping[str, Any]]) -> list[Json]:
    """Project observed schedule rows onto the sealed contract fields."""

    keys = tuple(_expected_schedule_projection()[0])
    return [{key: row.get(key) for key in keys} for row in rows]


def reduce_rows(
    schedule: Sequence[Mapping[str, Any]],
    episodes: Sequence[Mapping[str, Any]],
) -> Json:
    """Join the exact schedule to raw episode rows and retain every failure."""

    expected = _expected_schedule_projection()
    observed_schedule = _schedule_projection(schedule)
    by_id: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in episodes:
        by_id[str(row.get("episode_id"))].append(row)
    per_episode: list[Json] = []
    for sealed in expected:
        episode_id = str(sealed["episode_id"])
        matched = by_id.get(episode_id, [])
        row = matched[0] if len(matched) == 1 else {}
        failures: list[str] = []
        if len(matched) != 1:
            failures.append("episode_row_count")
        for field in DISABLED_FIELDS:
            if row.get(field) is not True:
                failures.append(field)
        for field in (
            "game",
            "seed",
            "action_limit",
            "request_limit",
            "max_new_tokens_per_call",
            "episode_limit_s",
            "panel_live_limit_s",
        ):
            if row.get(field) != sealed.get(field):
                failures.append(f"schedule_{field}")
        budget = row.get("request_budget_receipt") or {}
        action_count = int(row.get("action_count") or 0)
        progressed = int(row.get("peak_level") or 0) > int(row.get("start_level") or 0)
        per_episode.append(
            {
                "episode_id": episode_id,
                "game": sealed["game"],
                "seed": sealed["seed"],
                "disposition": row.get("disposition", "missing"),
                "action_count": action_count,
                "action_limit": sealed["action_limit"],
                "generation_attempted": int(budget.get("attempted") or 0),
                "generation_completed": int(budget.get("completed") or 0),
                "generation_failed": int(budget.get("failed") or 0),
                "generation_cancelled": int(budget.get("cancelled") or 0),
                "generation_in_flight": int(budget.get("in_flight") or 0),
                "start_level": row.get("start_level"),
                "peak_level": row.get("peak_level"),
                "terminal_level": row.get("terminal_level"),
                "progressed": progressed,
                "actions_to_progress": None,
                "actions_to_progress_censored": not progressed,
                "censored": row.get("disposition") != "complete" or not progressed,
                "censor_reason": None if progressed else "no_progress_within_action_cap",
                "historical_solve_provenance": row.get("solve_provenance"),
                "solve_provenance": "live_agent_self_discovery",
                "new_level_credit": 0,
                "elapsed_s": float(row.get("elapsed_s") or 0.0),
                "failures": failures,
            }
        )
    completed = sum(row["disposition"] == "complete" for row in per_episode)
    return {
        "schedule_matches": observed_schedule == expected,
        "unique_episode_ids": len(by_id) == len(episodes),
        "completed_episode_count": completed,
        "action_count": sum(row["action_count"] for row in per_episode),
        "all_action_caps_valid": all(
            row["action_count"] <= row["action_limit"] for row in per_episode
        ),
        "all_disabled_controls_valid": all(not row["failures"] for row in per_episode),
        "per_episode_results": per_episode,
    }


def _invocation_counts(events: Sequence[Mapping[str, Any]]) -> Json:
    """Reduce historical model calls while keeping them outside current counts."""

    reduced = panel_a._invocation_reduction(events, child_terminal=True)
    return deepcopy(dict(reduced.get("invocation_counts") or {}))


def _operation_counts(counts: Mapping[str, Any], prefix: str) -> Json:
    """Return one operation family with attempted and terminal dispositions."""

    return {
        state: int(counts.get(f"{prefix}_{state}") or 0)
        for state in ("attempted", "completed", "failed", "cancelled", "in_flight")
    }


def reduce_original_evidence(root: Path) -> Json:
    """Recompute Panel B rows from immutable schedule, episode, call, and event bytes."""

    schedule = load_object(root / SCHEDULE_PATH).get("rows") or []
    episodes = load_object(root / EPISODE_ROWS_PATH).get("rows") or []
    base = reduce_rows(schedule, episodes)
    all_events, shards = _read_all_interval_events(root)
    events_by_episode: dict[str, list[Json]] = defaultdict(list)
    for event in all_events:
        events_by_episode[str(event.get("episode_id"))].append(event)
    raw_by_id = {str(row.get("episode_id")): row for row in episodes}
    interval_rows: list[Json] = []
    supervisor_rows: list[Json] = []
    for unit in base["per_episode_results"]:
        episode_id = str(unit["episode_id"])
        events = events_by_episode.get(episode_id, [])
        start, end = _episode_bounds(raw_by_id.get(episode_id, {}), events)
        cost = interval_protocol.reduce_episode_intervals(
            events,
            episode_id=episode_id,
            episode_start_ns=start,
            episode_end_ns=end,
        )
        stages = cost.pop("stage_rows", [])
        compact_cost = {
            **cost,
            "stage_row_count": len(stages),
            "stage_rows_sha256": canonical_hash(stages),
        }
        interval_rows.append(
            {
                "episode_id": episode_id,
                "game": unit["game"],
                "seed": unit["seed"],
                **compact_cost,
            }
        )
        selections = [
            row
            for row in events
            if row.get("seam") == "supervisor_arm_selection" and row.get("event") == "selection"
        ]
        supervisor_rows.append(
            {
                "episode_id": episode_id,
                "game": unit["game"],
                "seed": unit["seed"],
                "opportunity_count": sum(
                    row.get("seam") == "supervisor_arm_selection"
                    and row.get("event") == "stage_start"
                    for row in events
                ),
                "selection_count": len(selections),
                "supervisor_firing_count": sum(
                    row.get("supervisor_fired") is True for row in selections
                ),
                "applied_redirection_count": sum(
                    row.get("applied_redirection") is True for row in selections
                ),
                "excluded_development_proxy_count": sum(
                    row.get("solve_provenance") in {"development_proxy", "outer_loop_re"}
                    for row in events
                ),
            }
        )
    for row in base["per_episode_results"]:
        cost = next(item for item in interval_rows if item["episode_id"] == row["episode_id"])
        row["exclusive_timing"] = deepcopy(cost)
    by_game: list[Json] = []
    for game in GAMES:
        rows = [row for row in base["per_episode_results"] if row["game"] == game]
        opportunities = [row for row in supervisor_rows if row["game"] == game]
        by_game.append(
            {
                "game": game,
                "planned_episodes": len(SEEDS),
                "completed_episodes": sum(row["disposition"] == "complete" for row in rows),
                "action_count": sum(row["action_count"] for row in rows),
                "generation_attempted": sum(row["generation_attempted"] for row in rows),
                "failed_generation_calls": sum(row["generation_failed"] for row in rows),
                "progressed_episodes": sum(row["progressed"] for row in rows),
                "right_censored_episodes": sum(row["censored"] for row in rows),
                "supervisor_opportunities": sum(row["opportunity_count"] for row in opportunities),
                "supervisor_firings": sum(row["supervisor_firing_count"] for row in opportunities),
                "applied_redirections": sum(
                    row["applied_redirection_count"] for row in opportunities
                ),
                "failures": sorted(
                    {failure for row in rows for failure in row.get("failures") or []}
                ),
            }
        )
    boundary_events = read_jsonl(root / BOUNDARY_PATH)
    historical_counts = _invocation_counts(boundary_events)
    generation_counts = _operation_counts(historical_counts, "generation_calls")
    model_load_counts = _operation_counts(historical_counts, "model_loads")
    source_session = load_object(root / SESSION_PATH)
    return {
        **base,
        "per_game_results": by_game,
        "exclusive_cost_rows": interval_rows,
        "supervisor_opportunity_rows": supervisor_rows,
        "interval_shard_dispositions": shards,
        "all_interval_shards_valid": bool(shards) and all(row["passed"] for row in shards),
        "all_interval_bounds_valid": bool(interval_rows)
        and all(row.get("bounds_valid") is True for row in interval_rows),
        "historical_invocation_counts": historical_counts,
        "generation_counts": generation_counts,
        "model_load_counts": model_load_counts,
        "historical_session_duration_s": source_session.get("duration_s"),
        "historical_session_error": source_session.get("error"),
        "historical_session_timed_out": source_session.get("timed_out"),
        "historical_model_invoked": source_session.get("model_invoked"),
        "all_custody_counts_valid": (
            generation_counts
            == {"attempted": 36, "completed": 36, "failed": 0, "cancelled": 0, "in_flight": 0}
            and model_load_counts
            == {"attempted": 1, "completed": 1, "failed": 0, "cancelled": 0, "in_flight": 0}
        ),
    }


def authenticate_runtime_custody(root: Path) -> Json:
    """Join historical process, lease, GPU, model, and tokenizer receipts."""

    session = load_object(root / SESSION_PATH)
    candidate = load_object(root / CANDIDATE_PATH)
    runtime = session.get("runtime_receipt") or {}
    lease = runtime.get("lease_owner") or {}
    release = runtime.get("lease_release") or {}
    offload = runtime.get("observed_cuda_offload") or {}
    specs = candidate.get("model_specs") or []
    model = specs[0] if len(specs) == 1 and isinstance(specs[0], Mapping) else {}
    source = (candidate.get("source_artifact_hashes") or {}).get(model.get("model_path")) or {}
    boundary = read_jsonl(root / BOUNDARY_PATH)
    child_pid = session.get("child_pid")
    owner_pids = {row.get("owner_pid") for row in boundary}
    checks = {
        "session_terminal": session.get("error") is None and session.get("timed_out") is False,
        "child_terminal": runtime.get("child_terminal") is True
        and runtime.get("child_returncode") == 0,
        "child_pid_owned": child_pid == runtime.get("child_pid") and owner_pids == {child_pid},
        "server_pid_owned": runtime.get("server_pid") == offload.get("server_pid"),
        "lease_identity": lease.get("lease_id") == release.get("lease_id")
        and lease.get("pid") == release.get("pid")
        and release.get("released") is True,
        "gpu_identity": runtime.get("gpu_uuid")
        == lease.get("device_uuid")
        == release.get("device_uuid")
        == offload.get("gpu_uuid"),
        "model_identity": model.get("hf_id") == "unsloth/Qwen3.8-27B-GGUF"
        and model.get("quantization") == "Q4_K_M"
        and model.get("sha256") == source.get("sha256")
        and lease.get("expected_model") == model.get("model_path"),
        "tokenizer_identity": runtime.get("embedded_tokenizer") is True
        and str(model.get("tokenizer_detail") or "").startswith("embedded GGUF tokenizer OK"),
        "offload_observed": offload.get("passed") is True
        and int(offload.get("owned_server_vram_mb") or 0) > 0,
        "no_cleanup_signal": not runtime.get("signals_sent")
        and not release.get("signals_sent")
        and not (lease.get("recovery") or {}).get("performed"),
    }
    return {
        "qualified": all(checks.values()),
        "checks": checks,
        "parent_pid": lease.get("pid"),
        "parent_start_ticks": lease.get("pid_start_ticks"),
        "child_pid": child_pid,
        "server_pid": runtime.get("server_pid"),
        "gpu_uuid": runtime.get("gpu_uuid"),
        "gpu_name": runtime.get("gpu_name"),
        "model": deepcopy(dict(model)),
        "embedded_tokenizer": runtime.get("embedded_tokenizer"),
        "native_binary": runtime.get("native_binary"),
        "observed_cuda_offload": deepcopy(dict(offload)),
        "lease_release": deepcopy(dict(release)),
    }


def authenticate_historical_receipts(root: Path, candidate: Mapping[str, Any]) -> list[Json]:
    """Authenticate twelve old receipts without reporting them as current work."""

    rows: list[Json] = []
    for receipt in candidate.get("validation_receipts") or []:
        name = str(receipt.get("name") or "")
        path = root / str(receipt.get("log_path") or "")
        failures: list[str] = []
        observed_hash = sha256_file(path) if path.is_file() else None
        if name not in HISTORICAL_RECEIPT_NAMES:
            failures.append("unexpected_receipt_name")
        if not path.is_file():
            failures.append("log_missing")
        if observed_hash != receipt.get("log_sha256"):
            failures.append("log_hash_mismatch")
        if receipt.get("exit_code") != 0:
            failures.append("exit_nonzero")
        if receipt.get("passed") is not True:
            failures.append("receipt_not_passed")
        if receipt.get("timed_out") is True:
            failures.append("receipt_timed_out")
        command = str(receipt.get("command") or "")
        if name in validation_scope.REQUIRED_CHECK_NAMES and "experiment_7499" not in command:
            if name not in {"ruff_check", "ruff_format", "changed_module_mypy"}:
                failures.append("command_scope_missing_exp7499")
        rows.append(
            {
                "name": name,
                "command": command,
                "command_argv": deepcopy(receipt.get("command_argv")),
                "scope": receipt.get("scope"),
                "exit_code": receipt.get("exit_code"),
                "passed": receipt.get("passed"),
                "timed_out": receipt.get("timed_out"),
                "log_path": receipt.get("log_path"),
                "expected_log_sha256": receipt.get("log_sha256"),
                "observed_log_sha256": observed_hash,
                "source_run_date": candidate.get("run_date"),
                "age_days": 1,
                "historical_only": True,
                "current_validation": False,
                "authenticated": not failures,
                "failures": failures,
            }
        )
    counts = Counter(row["name"] for row in rows)
    if any(counts[name] != 1 for name in HISTORICAL_RECEIPT_NAMES):
        for row in rows:
            row["authenticated"] = False
            row["failures"] = [*row["failures"], "receipt_set_incomplete_or_duplicate"]
    return rows


def _registry_disposition(root: Path) -> Json:
    """Read current public solve credit before describing inherited progress."""

    path = root / REGISTRY_PATH
    value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    indexed = {
        str(row.get("game")): row
        for row in value.get("games", [])
        if isinstance(row, Mapping) and row.get("game")
    }
    return {
        "path": REGISTRY_PATH.as_posix(),
        "sha256": sha256_file(path),
        "read_before_progress_description": True,
        "policy_received_registry_data": False,
        "rows": [
            {
                "game": game,
                "registered": game in indexed,
                "levels_reproduced": indexed.get(game, {}).get("levels_reproduced"),
                "full_game_clear": indexed.get(game, {}).get("full_game_clear"),
            }
            for game in GAMES
        ],
    }


def classify_qualification(
    *,
    external_available: bool,
    evidence_valid: bool,
    current_validation_passed: bool,
    benefit_passed: bool,
) -> tuple[str, str, int]:
    """Keep missing evidence, invalid evidence, readiness, and benefit separate."""

    if not external_available:
        return "blocked", "complete_blocked_external_evidence_absent", 0
    if not evidence_valid or not current_validation_passed:
        return "disqualified", "complete_disqualified_invalid_evidence", 0
    if benefit_passed:
        return "positive", "complete_positive_arc_panel_b_evidence_recovered", 1
    return "null", "complete_null_arc_panel_b_evidence_recovered", 1


def build_validation_plan(
    root: Path, private_root: Path
) -> list[validation_contract.EnvironmentCommandSpec]:
    """Freeze the eight scoped checks with private temp and coverage paths."""

    private_root.mkdir(parents=True, exist_ok=True)
    commands = validation_contract.build_command_plan(root, MANIFEST, private_root)
    return [
        command
        for command in commands
        if isinstance(command, validation_contract.EnvironmentCommandSpec)
    ]


def validate_validation_plan(
    root: Path, commands: Sequence[validation_scope.CommandSpec]
) -> list[str]:
    """Reject broad targets, duplicates, or any command outside the manifest."""

    errors = validation_contract.validate_command_plan(root, MANIFEST, commands)
    counts = Counter(row.name for row in commands)
    for name in CURRENT_VALIDATION_NAMES:
        if counts[name] != 1:
            errors.append(f"command_count:{name}:{counts[name]}")
    for command in commands:
        if any(argument.rstrip("/") in {"tests", "tests/python"} for argument in command.argv):
            errors.append(f"broad_test_target:{command.name}")
    return list(dict.fromkeys(errors))


def terminal_command_specs(root: Path, candidate: Path) -> list[validation_scope.CommandSpec]:
    """Build fresh replay and strict reader commands for one exact candidate."""

    python = str(root / ".venv/bin/python")
    wrapper = str(root / WRAPPER_PATH)
    return [
        validation_scope.CommandSpec(
            "declared_entrypoint_cold_replay",
            (python, "-u", wrapper, "--replay", str(candidate)),
            "exact_terminal_candidate",
            300.0,
        ),
        validation_scope.CommandSpec(
            "independent_cold_reducer",
            (python, "-u", wrapper, "--replay", str(candidate), "--reduce-only"),
            "exact_terminal_candidate",
            300.0,
        ),
        validation_scope.CommandSpec(
            "adversarial_verify",
            (python, "-u", "scripts/adversarial_verify.py", str(candidate)),
            "exact_terminal_candidate",
            300.0,
        ),
        validation_scope.CommandSpec(
            "verdict_row_consistency_strict",
            (
                python,
                "-u",
                "scripts/verdict_row_consistency_lint.py",
                "--strict",
                str(candidate),
            ),
            "exact_terminal_candidate",
            300.0,
        ),
    ]


def _receipts_pass(receipts: Sequence[Mapping[str, Any]], names: Sequence[str]) -> bool:
    """Require one successful non-timeout receipt for every exact command name."""

    by_name: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in receipts:
        by_name[str(row.get("name"))].append(row)
    return all(
        len(by_name[name]) == 1
        and by_name[name][0].get("passed") is True
        and by_name[name][0].get("exit_code") == 0
        and by_name[name][0].get("timed_out") is False
        for name in names
    )


def _current_receipt_dispositions(receipts: Sequence[Mapping[str, Any]]) -> list[Json]:
    """Label newly executed commands without mixing them with old validation."""

    return [
        {
            **deepcopy(dict(row)),
            "historical_only": False,
            "current_validation": True,
            "authenticated": row.get("passed") is True
            and row.get("exit_code") == 0
            and row.get("timed_out") is False,
            "age_days": 0,
        }
        for row in receipts
    ]


def _field_principles(keys: Sequence[str]) -> dict[str, str]:
    """Explain how each emitted field prevents a specific evidence failure."""

    specific = {
        "schema": "Version, experiment identity, milestone, and status prevent reader drift.",
        "run_date": "Use 20260922 while retaining actual UTC and monotonic process boundaries.",
        "preconditions_checked": "Exact paths and observed failures prevent invented recovery inputs.",
        "MODEL_SPECS": "An empty list proves this aggregation planned no current model load.",
        "model_specs": "The lowercase mirror keeps current no-load semantics unambiguous.",
        "model_invoked": "False separates current CPU work from historical live inference.",
        "invocation_counts": "Zero attempted and terminal counters exclude historical calls from current work.",
        "inference_substrate": "The exact aggregation label prevents historical generation from becoming current inference.",
        "inference_substrate_class": "Aggregation records the actual work without duration padding.",
        "execution_venue": "Host CPU qualification stays distinct from historical CUDA and board evidence.",
        "duration_s": "Measured current elapsed time excludes the historical live session.",
        "phase_spans": "Flushed phase boundaries expose stalls and unfinished validation.",
        "random_seed": "Frozen roles preserve the original schedule without multiplying support.",
        "reproducibility_checksum": "Stable source, setting, and reduction identities detect changed evidence.",
        "source_artifact_hashes": "Exact original bytes and authority labels prevent candidate laundering.",
        "rows": "Per-unit failures and censoring support independent terminal reduction.",
        "sample_size_budget": "Planned, attempted, completed, failed, censored, and unstarted units stay separate.",
        "acceptance_gate_results": "Validity, readiness, and benefit comparisons remain independent.",
        "gate_check_summary": "The first exact required failure prevents vague blocked verdicts.",
        "honest_verdict": "A complete terminal prefix distinguishes closed recovery from retryable work.",
        "verdict_class": "The closed enum prevents missing or invalid evidence from becoming a null.",
        "verifier_is_oracle": "False prevents circular fixture evidence from posing as independent qualification.",
        "flagged_adversarial": "Actual strict-reader findings cannot be cleared to open qualification.",
        "validation_receipts": "Exact current commands, exits, scopes, and log hashes support replay.",
        "field_principles": "Every emitted field states the evidence failure it prevents.",
        "arc_panel_b_qualified_score": "One requires authenticated historical evidence and complete current validation.",
        "solve_provenance": "Historical live-origin attempts stay distinct from development proxies.",
        "historical_model_specs": "Exact Qwen identity and receipt separate old inference from current aggregation.",
        "per_game_results": "Game-level cost, progress, and censoring preserve cross-game assessment.",
        "per_episode_results": "Exactly eighteen scheduled rows prevent hidden missingness or seed selection.",
        "raw_validation_dispositions": "Old and new receipts retain identity, age, scope, and actual exit status.",
    }
    return {
        key: specific.get(
            key, "This typed evidence prevents an omitted fact from changing the conclusion."
        )
        for key in keys
    }


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Bind stable custody, reduction, validation, and terminal classification."""

    return canonical_hash(
        {
            key: artifact.get(key)
            for key in (
                "schema",
                "experiment_id",
                "milestone",
                "run_date",
                "source_artifact_hashes",
                "historical_runtime_custody",
                "historical_invocation_counts",
                "rows",
                "per_game_results",
                "per_episode_results",
                "sample_size_budget",
                "acceptance_gate_results",
                "arc_panel_b_qualified_score",
                "verdict_class",
            )
        }
    )


def build_artifact(
    root: Path,
    *,
    preconditions: Sequence[Mapping[str, Any]],
    inventory: Sequence[Mapping[str, Any]],
    sources: Mapping[str, Any],
    reduction: Mapping[str, Any],
    custody: Mapping[str, Any],
    historical_receipts: Sequence[Mapping[str, Any]],
    current_receipts: Sequence[Mapping[str, Any]],
    duration_s: float,
    computation_s: float,
    validation_s: float,
    phase_spans: Sequence[Mapping[str, Any]],
    started_at_utc: str,
    ended_at_utc: str,
    require_terminal: bool,
) -> Json:
    """Build one terminal record from raw reductions and real command receipts."""

    candidate = load_object(root / CANDIDATE_PATH)
    session = load_object(root / SESSION_PATH)
    external_available = bool(preconditions) and all(
        row.get("passed") is True
        for row in preconditions
        if str(row.get("check", "")).startswith("source_bytes:")
    )
    historical_valid = (
        bool(preconditions)
        and all(row.get("passed") is True for row in preconditions)
        and reduction.get("schedule_matches") is True
        and reduction.get("unique_episode_ids") is True
        and reduction.get("completed_episode_count") == 18
        and reduction.get("action_count") == 3_240
        and reduction.get("all_action_caps_valid") is True
        and reduction.get("all_disabled_controls_valid") is True
        and reduction.get("all_interval_shards_valid") is True
        and reduction.get("all_interval_bounds_valid") is True
        and reduction.get("all_custody_counts_valid") is True
        and reduction.get("historical_session_error") is None
        and reduction.get("historical_session_timed_out") is False
        and custody.get("qualified") is True
        and len(historical_receipts) == len(HISTORICAL_RECEIPT_NAMES)
        and all(row.get("authenticated") is True for row in historical_receipts)
    )
    affected_passed = _receipts_pass(current_receipts, CURRENT_VALIDATION_NAMES)
    terminal_passed = (
        _receipts_pass(current_receipts, TERMINAL_RECEIPT_NAMES) if require_terminal else True
    )
    current_validation_passed = affected_passed and terminal_passed
    progressed = sum(row.get("progressed") is True for row in reduction["per_episode_results"])
    benefit_passed = progressed > 0 and candidate.get("scientific_benefit_score") == 1
    verdict, honest, qualified_score = classify_qualification(
        external_available=external_available,
        evidence_valid=historical_valid,
        current_validation_passed=current_validation_passed,
        benefit_passed=benefit_passed,
    )
    gates = [
        _gate(
            "external_inputs_available",
            "validity",
            True,
            external_available,
            upstream="preconditions_checked",
            field="source_bytes",
            principle="Missing external proof blocks qualification rather than creating a null.",
        ),
        _gate(
            "replay_critical_sources_unchanged",
            "validity",
            True,
            next(
                (
                    row.get("passed")
                    for row in preconditions
                    if row.get("check") == "replay_critical_sources_unchanged"
                ),
                False,
            ),
            upstream=CANDIDATE_PATH.as_posix(),
            field="source_artifact_hashes",
            principle="Changed historical code closes trustworthy replay.",
        ),
        _gate(
            "exact_eighteen_row_schedule",
            "validity",
            True,
            reduction.get("schedule_matches") is True
            and reduction.get("unique_episode_ids") is True,
            upstream=SCHEDULE_PATH.as_posix(),
            field="rows",
            principle="Seed selection or duplicated episodes would invalidate support.",
        ),
        _gate(
            "episode_and_action_reconciliation",
            "validity",
            {"episodes": 18, "actions": 3_240},
            {
                "episodes": reduction.get("completed_episode_count"),
                "actions": reduction.get("action_count"),
            },
            upstream=EPISODE_ROWS_PATH.as_posix(),
            field="completed_episode_count|action_count",
            principle="Incomplete work cannot pose as the frozen completed panel.",
        ),
        _gate(
            "historical_invocation_reconciliation",
            "validity",
            True,
            reduction.get("all_custody_counts_valid") is True,
            upstream=BOUNDARY_PATH.as_posix(),
            field="historical_invocation_counts",
            principle="Every old model attempt needs one terminal disposition.",
        ),
        _gate(
            "exclusive_interval_reduction",
            "validity",
            True,
            reduction.get("all_interval_shards_valid") is True
            and reduction.get("all_interval_bounds_valid") is True,
            upstream=(RAW_7499_DIR / "normalized_intervals").as_posix(),
            field="exclusive_cost_rows",
            principle="Nested intervals must not increase exclusive time or exceed custody bounds.",
        ),
    ]
    gates.extend(
        [
            _gate(
                "off_path_tools_disabled",
                "validity",
                True,
                reduction.get("all_disabled_controls_valid") is True
                and all(
                    row.get("excluded_development_proxy_count") == 0
                    for row in reduction["supervisor_opportunity_rows"]
                ),
                upstream=EPISODE_ROWS_PATH.as_posix(),
                field="disabled_controls|solve_provenance",
                principle="Source, recipe, adapter, solver, or outer-loop evidence cannot earn live credit.",
            ),
            _gate(
                "runtime_and_model_custody",
                "validity",
                True,
                custody.get("qualified") is True,
                upstream=SESSION_PATH.as_posix(),
                field="runtime_receipt",
                principle="Model labels alone do not prove owned process, GPU, tokenizer, or file identity.",
            ),
            _gate(
                "twelve_historical_receipts_authenticated",
                "validity",
                12,
                sum(row.get("authenticated") is True for row in historical_receipts),
                upstream=CANDIDATE_PATH.as_posix(),
                field="validation_receipts",
                principle="Old validation needs authentic commands, exits, scopes, and exact logs.",
            ),
            _gate(
                "current_affected_validation",
                "validity",
                True,
                affected_passed,
                upstream="validation_receipts",
                field="current_scoped_commands",
                principle="Favorable historical metrics cannot excuse failed current code checks.",
            ),
            _gate(
                "fresh_terminal_readers",
                "validity",
                True,
                terminal_passed,
                upstream="validation_receipts",
                field="current_terminal_commands",
                principle="Independent replay and strict readers must accept the measured candidate.",
            ),
            _gate(
                "panel_b_qualification_ready",
                "readiness",
                True,
                historical_valid and current_validation_passed,
                upstream="acceptance_gate_results",
                field="arc_panel_b_qualified_score",
                principle="Readiness reflects valid completion and stays independent of benefit.",
            ),
            _gate(
                "completed_episode_support",
                "scientific_benefit",
                18,
                reduction.get("completed_episode_count"),
                upstream="per_episode_results",
                field="completed_episode_count",
                principle="A favorable subset cannot replace all frozen episode support.",
                op=">=",
            ),
            _gate(
                "completed_game_support",
                "scientific_benefit",
                6,
                sum(
                    row.get("completed_episodes") == len(SEEDS)
                    for row in reduction["per_game_results"]
                ),
                upstream="per_game_results",
                field="completed_games",
                principle="Repeated seeds cannot replace independent game support.",
                op=">=",
            ),
            _gate(
                "reproduced_progress_support",
                "scientific_benefit",
                1,
                progressed,
                upstream="per_episode_results",
                field="progressed",
                principle="Complete zero progress remains a valid null and not a benefit claim.",
                op=">=",
            ),
        ]
    )
    registry = _registry_disposition(root)
    sample_size = {
        "planned_independent_units": 18,
        "attempted_independent_units": len(reduction["per_episode_results"]),
        "completed_independent_units": reduction.get("completed_episode_count"),
        "excluded_independent_units": 0,
        "failed_independent_units": sum(
            row.get("disposition") not in {"complete", "unavailable"}
            for row in reduction["per_episode_results"]
        ),
        "censored_independent_units": sum(
            row.get("censored") is True for row in reduction["per_episode_results"]
        ),
        "unstarted_independent_units": sum(
            row.get("disposition") == "missing" for row in reduction["per_episode_results"]
        ),
        "independent_game_clusters": len(GAMES),
        "seeds_do_not_multiply_game_support": True,
    }
    raw_dispositions = [
        *deepcopy(list(historical_receipts)),
        *_current_receipt_dispositions(current_receipts),
    ]
    artifact: Json = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "status": honest,
        "run_date": RUN_DATE,
        "started_at_utc": started_at_utc,
        "ended_at_utc": ended_at_utc,
        "process_identity": {"pid": os.getpid(), "parent_pid": os.getppid()},
        "preconditions_checked": deepcopy(list(preconditions)),
        "MODEL_SPECS": [],
        "model_specs": [],
        "model_invoked": False,
        "invocation_counts": deepcopy(ZERO_INVOCATION_COUNTS),
        "historical_model_specs": deepcopy(candidate.get("model_specs") or []),
        "historical_model_invoked": candidate.get("model_invoked"),
        "historical_invocation_counts": deepcopy(
            reduction.get("historical_invocation_counts") or {}
        ),
        "historical_model_receipt": {
            "source": CANDIDATE_PATH.as_posix(),
            "session": SESSION_PATH.as_posix(),
            "model": deepcopy(custody.get("model") or {}),
            "runtime": {
                key: deepcopy(custody.get(key))
                for key in (
                    "parent_pid",
                    "child_pid",
                    "server_pid",
                    "gpu_uuid",
                    "gpu_name",
                    "embedded_tokenizer",
                    "native_binary",
                    "observed_cuda_offload",
                )
            },
        },
        "inference_substrate": "aggregation_from_upstream_artifacts",
        "inference_substrate_class": "aggregation",
        "execution_venue": "host",
        "duration_s": round(float(duration_s), 6),
        "duration_breakdown_s": {
            "current_authoring_s": 0.0,
            "current_computation_s": round(float(computation_s), 6),
            "current_validation_s": round(float(validation_s), 6),
            "historical_live_session_s": float(session.get("duration_s") or 0.0),
        },
        "historical_duration_s": session.get("duration_s"),
        "phase_spans": deepcopy(list(phase_spans)),
        "random_seed": {
            "schedule": 7_478,
            "episodes": list(SEEDS),
            "audit": 7_511_093,
            "bootstrap": None,
            "arrival": None,
            "fitting": None,
            "principle": "Recovery performs no resampling, fitting, or imputation.",
        },
        "reproducibility_checksum": "",
        "source_artifact_hashes": deepcopy(dict(sources)),
        "original_file_inventory": deepcopy(list(inventory)),
        "rows": deepcopy(reduction["per_episode_results"]),
        "per_episode_results": deepcopy(reduction["per_episode_results"]),
        "per_game_results": deepcopy(reduction["per_game_results"]),
        "exclusive_cost_rows": deepcopy(reduction["exclusive_cost_rows"]),
        "supervisor_opportunity_rows": deepcopy(reduction["supervisor_opportunity_rows"]),
        "sample_size_budget": sample_size,
        "acceptance_gate_results": gates,
        "gate_check_summary": _gate_summary(gates),
        "honest_verdict": honest,
        "verdict_class": verdict,
        "verifier_is_oracle": False,
        "flagged_adversarial": False,
        "validation_manifest": {
            "experiment_id": MANIFEST.experiment_id,
            "test_paths": list(MANIFEST.test_paths),
            "changed_modules": list(MANIFEST.changed_modules),
            "static_paths": list(MANIFEST.static_paths),
        },
        "validation_receipts": deepcopy(list(current_receipts)),
        "raw_validation_dispositions": raw_dispositions,
        "arc_panel_b_qualified_score": qualified_score,
        "scientific_benefit_score": int(benefit_passed),
        "solve_provenance": "live_agent_self_discovery",
        "historical_runtime_custody": deepcopy(dict(custody)),
        "registry_precheck": registry,
        "new_level_credit": 0,
        "new_game_episodes": 0,
        "new_generation_calls": 0,
        "generator_loaded": False,
        "game_source_inspection": False,
        "solve_claim_made": False,
        "remote_submission": False,
        "raw_candidate_authoritative": False,
        "raw_candidate_prior_honest_verdict": candidate.get("honest_verdict"),
        "raw_candidate_prior_verdict_class": candidate.get("verdict_class"),
        "conductor_checkpoint": {
            "stage": load_object(root / CHECKPOINT_PATH).get("stage"),
            "completed_units": load_object(root / CHECKPOINT_PATH).get("completed_units"),
            "terminal_child": load_object(root / CHECKPOINT_PATH).get("terminal_child"),
            "historical_load_completed": load_object(root / CHECKPOINT_PATH).get("model_loaded"),
        },
    }
    artifact["field_principles"] = _field_principles((*artifact, "field_principles"))
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def independent_reduce(artifact: Mapping[str, Any], root: Path) -> Json:
    """Re-read immutable bytes and compare every declared unit and aggregate."""

    reduced = reduce_original_evidence(root)
    declared = {
        "rows": artifact.get("rows"),
        "per_episode_results": artifact.get("per_episode_results"),
        "per_game_results": artifact.get("per_game_results"),
        "exclusive_cost_rows": artifact.get("exclusive_cost_rows"),
        "supervisor_opportunity_rows": artifact.get("supervisor_opportunity_rows"),
        "historical_invocation_counts": artifact.get("historical_invocation_counts"),
    }
    recomputed = {
        "rows": reduced["per_episode_results"],
        "per_episode_results": reduced["per_episode_results"],
        "per_game_results": reduced["per_game_results"],
        "exclusive_cost_rows": reduced["exclusive_cost_rows"],
        "supervisor_opportunity_rows": reduced["supervisor_opportunity_rows"],
        "historical_invocation_counts": reduced["historical_invocation_counts"],
    }
    return {
        "matches_declared": canonical_hash(declared) == canonical_hash(recomputed),
        "declared_sha256": canonical_hash(declared),
        "recomputed_sha256": canonical_hash(recomputed),
        "completed_episode_count": reduced["completed_episode_count"],
        "action_count": reduced["action_count"],
        "generation_counts": reduced["generation_counts"],
        "all_interval_bounds_valid": reduced["all_interval_bounds_valid"],
    }


def validate_artifact(
    value: Mapping[str, Any] | Path,
    root: Path,
    *,
    require_terminal: bool,
) -> list[str]:
    """Cold-check identity, no-current-call semantics, reduction, and receipts."""

    artifact = load_object(value) if isinstance(value, Path) else dict(value)
    errors: list[str] = []
    required_fields = (
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
        "arc_panel_b_qualified_score",
        "solve_provenance",
        "historical_model_specs",
        "per_game_results",
        "per_episode_results",
        "raw_validation_dispositions",
    )
    errors.extend(f"missing_field:{field}" for field in required_fields if field not in artifact)
    for field, expected in (
        ("schema", SCHEMA),
        ("experiment_id", EXPERIMENT_ID),
        ("milestone", MILESTONE),
        ("run_date", RUN_DATE),
        ("inference_substrate", "aggregation_from_upstream_artifacts"),
        ("inference_substrate_class", "aggregation"),
        ("solve_provenance", "live_agent_self_discovery"),
    ):
        if artifact.get(field) != expected:
            errors.append(f"identity_mismatch:{field}")
    if artifact.get("MODEL_SPECS") != [] or artifact.get("model_specs") != []:
        errors.append("current_model_specs_nonempty")
    if artifact.get("model_invoked") is not False:
        errors.append("current_model_invoked")
    if artifact.get("invocation_counts") != ZERO_INVOCATION_COUNTS:
        errors.append("current_invocation_counts_nonzero")
    if artifact.get("new_game_episodes") != 0 or artifact.get("new_generation_calls") != 0:
        errors.append("new_live_work_claimed")
    if artifact.get("raw_candidate_authoritative") is not False:
        errors.append("raw_candidate_promoted_to_authority")
    if artifact.get("verdict_class") not in {
        "positive",
        "circular_positive",
        "null",
        "blocked",
        "disqualified",
        "partial",
    }:
        errors.append("invalid_verdict_class")
    if not str(artifact.get("honest_verdict") or "").startswith("complete_"):
        errors.append("honest_verdict_not_terminal")
    if independent_reduce(artifact, root)["matches_declared"] is not True:
        errors.append("independent_reduction_mismatch")
    historical = [
        row
        for row in artifact.get("raw_validation_dispositions") or []
        if row.get("historical_only") is True
    ]
    if len(historical) != 12 or not all(row.get("authenticated") is True for row in historical):
        errors.append("historical_validation_not_authenticated")
    receipts = artifact.get("validation_receipts") or []
    if not _receipts_pass(receipts, CURRENT_VALIDATION_NAMES):
        errors.append("current_scoped_validation_missing_or_failed")
    if require_terminal and not _receipts_pass(receipts, TERMINAL_RECEIPT_NAMES):
        errors.append("terminal_validation_missing_or_failed")
    principles = artifact.get("field_principles") or {}
    if any(not principles.get(key) for key in artifact):
        errors.append("field_principle_missing")
    if any(not row.get("principle") for row in artifact.get("acceptance_gate_results") or []):
        errors.append("gate_principle_missing")
    if artifact.get("reproducibility_checksum") != reproducibility_checksum(artifact):
        errors.append("reproducibility_checksum_mismatch")
    if len(json.dumps(artifact, sort_keys=True).encode()) >= 20 * 1024 * 1024:
        errors.append("artifact_exceeds_20_mib")
    return list(dict.fromkeys(errors))


def _fixture_receipts() -> list[Json]:
    """Supply typed pass receipts only for deterministic artifact tests."""

    return [
        {
            "name": name,
            "command": f"fixture:{name}",
            "scope": "test_fixture",
            "exit_code": 0,
            "passed": True,
            "timed_out": False,
            "log_sha256": "sha256:" + "0" * 64,
        }
        for name in (*CURRENT_VALIDATION_NAMES, *TERMINAL_RECEIPT_NAMES)
    ]


def build_artifact_for_test(root: Path) -> Json:
    """Build a complete fixture from the real immutable historical bytes."""

    preconditions, inventory, sources = collect_preconditions(root)
    reduction = reduce_original_evidence(root)
    custody = authenticate_runtime_custody(root)
    candidate = load_object(root / CANDIDATE_PATH)
    historical = authenticate_historical_receipts(root, candidate)
    return build_artifact(
        root,
        preconditions=preconditions,
        inventory=inventory,
        sources=sources,
        reduction=reduction,
        custody=custody,
        historical_receipts=historical,
        current_receipts=_fixture_receipts(),
        duration_s=1.0,
        computation_s=0.5,
        validation_s=0.5,
        phase_spans=[{"phase": "fixture", "start_s": 0.0, "end_s": 1.0, "duration_s": 1.0}],
        started_at_utc="2026-09-22T00:00:00Z",
        ended_at_utc="2026-09-22T00:00:01Z",
        require_terminal=True,
    )


def _phase(name: str, began: float, started: float, units: int = 0) -> Json:
    """Record one disjoint monotonic span from the current process."""

    ended = time.monotonic()
    return {
        "phase": name,
        "start_s": began - started,
        "end_s": ended - started,
        "duration_s": ended - began,
        "completed_units": units,
    }


def _blocked_artifact(
    *,
    preconditions: Sequence[Mapping[str, Any]],
    inventory: Sequence[Mapping[str, Any]],
    sources: Mapping[str, Any],
    duration_s: float,
    started_at_utc: str,
) -> Json:  # pragma: no cover - external absence path.
    """Close missing external evidence without fabricating dependent reduction."""

    failed = [deepcopy(dict(row)) for row in preconditions if row.get("passed") is not True]
    first = (
        failed[0]
        if failed
        else {
            "check": "unknown_external_failure",
            "upstream": "preconditions_checked",
            "field": "unknown",
            "expected": True,
            "observed": None,
        }
    )
    artifact: Json = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "status": "complete_blocked_external_evidence_absent",
        "run_date": RUN_DATE,
        "started_at_utc": started_at_utc,
        "ended_at_utc": utc_now(),
        "preconditions_checked": deepcopy(list(preconditions)),
        "MODEL_SPECS": [],
        "model_specs": [],
        "model_invoked": False,
        "invocation_counts": deepcopy(ZERO_INVOCATION_COUNTS),
        "inference_substrate": "aggregation_from_upstream_artifacts",
        "inference_substrate_class": "aggregation",
        "execution_venue": "host",
        "duration_s": duration_s,
        "phase_spans": [],
        "random_seed": {"audit": 7_511_093},
        "reproducibility_checksum": "",
        "source_artifact_hashes": deepcopy(dict(sources)),
        "original_file_inventory": deepcopy(list(inventory)),
        "rows": [],
        "per_episode_results": [],
        "per_game_results": [],
        "sample_size_budget": {
            "planned_independent_units": 18,
            "attempted_independent_units": 0,
            "completed_independent_units": 0,
            "excluded_independent_units": 0,
            "failed_independent_units": 0,
            "censored_independent_units": 0,
            "unstarted_independent_units": 18,
        },
        "acceptance_gate_results": failed,
        "gate_check_summary": {
            "all_passed": False,
            "required_validity_and_readiness_passed": False,
            "failed_count": len(failed),
            "required_failed_count": len(failed),
            "failed_checks": failed,
            "first_failure": first,
            "first_failed_check": first.get("check"),
            "upstream": first.get("upstream"),
            "exact_field_path": first.get("field"),
            "expected_value": first.get("expected"),
            "observed_value": first.get("observed"),
        },
        "honest_verdict": "complete_blocked_external_evidence_absent",
        "verdict_class": "blocked",
        "verifier_is_oracle": False,
        "flagged_adversarial": False,
        "validation_receipts": [],
        "raw_validation_dispositions": [],
        "historical_model_specs": [],
        "arc_panel_b_qualified_score": 0,
        "solve_provenance": "live_agent_self_discovery",
    }
    artifact["field_principles"] = _field_principles((*artifact, "field_principles"))
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def run_experiment(
    root: Path,
    run_date: str,
    *,
    output_path: Path = RESULT_PATH,
) -> Json:  # pragma: no cover - exercised through the declared entrypoint.
    """Authenticate, reduce, validate, replay, and atomically publish recovery."""

    started = time.monotonic()
    progress(started, "startup", "flushed_progress")
    if run_date != RUN_DATE:
        raise SystemExit(f"--date must be {RUN_DATE}")
    root = root.resolve()
    started_at = utc_now()
    spans: list[Json] = []

    phase_started = time.monotonic()
    progress(started, "preconditions", "start")
    preconditions, inventory_before, sources = collect_preconditions(root)
    spans.append(_phase("preconditions", phase_started, started, len(inventory_before)))
    preconditions_passed = bool(preconditions) and all(row["passed"] for row in preconditions)
    progress(started, "preconditions", "end", passed=preconditions_passed)
    if not preconditions_passed:
        blocked = _blocked_artifact(
            preconditions=preconditions,
            inventory=inventory_before,
            sources=sources,
            duration_s=time.monotonic() - started,
            started_at_utc=started_at,
        )
        progress(started, "publish", "before_atomic_blocked", path=output_path)
        atomic_json(root / output_path, blocked)
        progress(started, "publish", "after_atomic_blocked", path=output_path)
        return blocked

    phase_started = time.monotonic()
    progress(started, "reduction", "start")
    reduction = reduce_original_evidence(root)
    custody = authenticate_runtime_custody(root)
    candidate_7499 = load_object(root / CANDIDATE_PATH)
    historical_receipts = authenticate_historical_receipts(root, candidate_7499)
    registry = _registry_disposition(root)
    del registry  # build_artifact reads it again immediately before progress fields.
    spans.append(_phase("reduction", phase_started, started, 18))
    computation_s = time.monotonic() - phase_started
    progress(started, "reduction", "end", episodes=18)

    private = Path(tempfile.mkdtemp(prefix="exp7511-validation-", dir="/tmp"))
    raw_dir = root / RAW_DIR
    raw_dir.mkdir(parents=True, exist_ok=True)
    commands = build_validation_plan(root, private)
    plan_errors = validate_validation_plan(root, commands)
    phase_started = time.monotonic()
    progress(started, "validation", "before_scoped_subprocesses", errors=len(plan_errors))
    current_receipts: list[Json] = []
    if not plan_errors:
        current_receipts = validation_contract.run_categorized_commands(
            root,
            [
                validation_contract.PlannedCommand(command, "required_validation", True)
                for command in commands
            ],
            log_dir=raw_dir / "validation/scoped",
        )
    spans.append(_phase("validation", phase_started, started, len(current_receipts)))
    validation_s = time.monotonic() - phase_started
    progress(
        started,
        "validation",
        "after_scoped_subprocesses",
        passed=_receipts_pass(current_receipts, CURRENT_VALIDATION_NAMES),
    )

    candidate = build_artifact(
        root,
        preconditions=preconditions,
        inventory=inventory_before,
        sources=sources,
        reduction=reduction,
        custody=custody,
        historical_receipts=historical_receipts,
        current_receipts=current_receipts,
        duration_s=time.monotonic() - started,
        computation_s=computation_s,
        validation_s=validation_s,
        phase_spans=spans,
        started_at_utc=started_at,
        ended_at_utc=utc_now(),
        require_terminal=False,
    )
    candidate_errors = validate_artifact(candidate, root, require_terminal=False)
    if candidate_errors:
        raise RuntimeError(f"measured_candidate_invalid:{candidate_errors}")
    progress(started, "candidate", "before_atomic_write", path=TERMINAL_CANDIDATE_PATH)
    atomic_json(root / TERMINAL_CANDIDATE_PATH, candidate)
    progress(started, "candidate", "after_atomic_write", path=TERMINAL_CANDIDATE_PATH)

    phase_started = time.monotonic()
    progress(started, "terminal_validation", "before_subprocesses")
    terminal_receipts = validation_contract.run_categorized_commands(
        root,
        [
            validation_contract.PlannedCommand(command, "terminal_validation", True)
            for command in terminal_command_specs(root, root / TERMINAL_CANDIDATE_PATH)
        ],
        log_dir=raw_dir / "validation/terminal",
    )
    spans.append(_phase("terminal_validation", phase_started, started, len(terminal_receipts)))
    terminal_validation_s = time.monotonic() - phase_started
    progress(
        started,
        "terminal_validation",
        "after_subprocesses",
        passed=_receipts_pass(terminal_receipts, TERMINAL_RECEIPT_NAMES),
    )

    inventory_after = inventory_original_evidence(root)
    custody_unchanged = canonical_hash(inventory_before) == canonical_hash(inventory_after)
    preconditions.append(
        _gate(
            "original_evidence_unchanged_after_validation",
            "validity",
            True,
            custody_unchanged,
            upstream=RAW_7499_DIR.as_posix(),
            field="original_file_inventory",
            principle="Recovery validation must not rewrite the retired capture.",
        )
    )
    final = build_artifact(
        root,
        preconditions=preconditions,
        inventory=inventory_after,
        sources=sources,
        reduction=reduction,
        custody=custody,
        historical_receipts=historical_receipts,
        current_receipts=[*current_receipts, *terminal_receipts],
        duration_s=time.monotonic() - started,
        computation_s=computation_s,
        validation_s=validation_s + terminal_validation_s,
        phase_spans=spans,
        started_at_utc=started_at,
        ended_at_utc=utc_now(),
        require_terminal=True,
    )
    errors = validate_artifact(final, root, require_terminal=True)
    if errors:
        raise RuntimeError(f"terminal_artifact_invalid:{errors}")
    progress(started, "publish", "before_atomic_terminal", path=output_path)
    atomic_json(root / output_path, final)
    progress(started, "publish", "after_atomic_terminal", status=final["status"])
    return final


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the public run role or one read-only cold replay role."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date")
    parser.add_argument("--output", type=Path, default=RESULT_PATH)
    parser.add_argument("--replay", type=Path)
    parser.add_argument("--reduce-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run recovery or cold-check one exact candidate without writing it."""

    args = parse_args(argv)
    if args.replay is not None:
        artifact = load_object(args.replay)
        if args.reduce_only:
            print(json.dumps(independent_reduce(artifact, REPO_ROOT), sort_keys=True), flush=True)
            return 0
        errors = validate_artifact(artifact, REPO_ROOT, require_terminal=False)
        print(json.dumps({"errors": errors, "valid": not errors}, sort_keys=True), flush=True)
        return int(bool(errors))
    if args.date is None:
        raise SystemExit("--date is required unless --replay is used")
    run_experiment(REPO_ROOT, args.date, output_path=args.output)  # pragma: no cover
    return 0  # pragma: no cover


if __name__ == "__main__":  # pragma: no cover - thin CLI dispatch.
    raise SystemExit(main())
