"""Measure live ARC plan lineage on the roster sealed by Experiment 7562."""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
from copy import deepcopy
from datetime import UTC, datetime
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import tempfile
import threading
import time
from typing import Any

import yaml

from carnot import experiment_7471_v654_arc_seam_observation as exp7471
from carnot import experiment_7491_e6_timed_live_profile as e6
from carnot import experiment_7562_v661_arc_plan_lineage as lineage
from carnot.agentic.arc_decision_telemetry import (
    TELEMETRY_ENV_FLAG,
    TELEMETRY_PATH_ENV,
    load_telemetry,
)
from carnot.agentic.arc_inference_boundary import InvocationBoundaryLedger
from carnot.experiment_7358_v646_validation_contract import AffectedManifest
from carnot.reporting import current_work_receipt
from carnot.reporting import experiment_7303_validation_scope as validation_scope


Json = dict[str, Any]
REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_DATE = "20260923"
MILESTONE = "2026.09.661"
EXPERIMENT_ID = "exp7570-arc-live-lineage"
TASK_ID = "experiment_7570_v661_arc_live_lineage"
SCHEMA = "carnot.exp7570.v661.arc_live_lineage.v1"
MODEL_ID = "unsloth/Qwen3.8-27B-GGUF"
MODEL_SPECS = [MODEL_ID]
TOTAL_LIVE_LIMIT_S = 3000.0
ACTION_LIMIT = 600
INDUCTION_LIMIT = 2
MAX_NEW_TOKENS = 4096
EPISODE_LIMIT_S = 300.0
DEFAULT_EPISODE_RESERVE_S = EPISODE_LIMIT_S
GPU_INDEX = 1

RESULT_PATH = Path("results/experiment_7570_v661_arc_live_lineage.json")
UPSTREAM_PATH = Path("results/experiment_7562_v661_arc_plan_lineage.json")
SPEC_PATH = Path("openspec/capabilities/arc-world-model-trust-energy/spec.md")
REGISTRY_PATH = Path("ops/arc_solve_registry.yaml")
MODULE_PATH = Path("python/carnot/experiment_7570_v661_arc_live_lineage.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_7570_v661_arc_live_lineage.py")
TEST_PATH = Path("tests/python/test_experiment_7570_v661_arc_live_lineage.py")
RAW_DIR = Path("results/raw/experiment_7570_v661_arc_live_lineage")
SCHEDULE_PATH = RAW_DIR / "frozen_schedule.json"
SESSION_PATH = RAW_DIR / "live_session.json"
BOUNDARY_PATH = RAW_DIR / "current_invocation_events.jsonl"
RUNTIME_EVENT_PATH = RAW_DIR / "runtime_events.jsonl"
ACTION_PATH = RAW_DIR / "live_action_rows.jsonl"
TELEMETRY_PATH = RAW_DIR / "plan_lineage_telemetry.jsonl"
CHECKPOINT_PATH = Path("results/checkpoints/experiment_7570_v661_arc_live_lineage.json")

ZERO_INVOCATION_COUNTS = {
    f"{kind}_{state}": 0
    for kind in ("model_loads", "forward_calls", "generation_calls")
    for state in ("attempted", "completed", "failed", "cancelled", "in_flight")
}
VALID_TERMINAL_STAGES = frozenset(lineage.telemetry.PLAN_LINEAGE_TERMINAL_STAGES)
USEFUL_STAGES = frozenset({"executed_with_level_progress"})
CENSORED_STAGES = frozenset({"censored"})
UNUSABLE_STAGES = VALID_TERMINAL_STAGES - USEFUL_STAGES - CENSORED_STAGES
ALLOWED_UPSTREAM_CLASSES = frozenset({"null", "positive", "circular_positive"})
TERMINAL_COMMAND_NAMES = (
    "declared_entrypoint",
    "fresh_process_cold_replay",
    "independent_reduction",
    "adversarial_verify",
    "verdict_row_consistency_strict",
)
E2E_COMMAND_NAMES = ("e2e_011", "e2e_012", "e2e_013", "llm_off_environment_smoke")
REQUIRED_RECEIPT_NAMES = (
    *validation_scope.REQUIRED_CHECK_NAMES,
    *E2E_COMMAND_NAMES,
    *TERMINAL_COMMAND_NAMES,
)
AFFECTED_MANIFEST = AffectedManifest(
    experiment_id=EXPERIMENT_ID,
    test_paths=(TEST_PATH.as_posix(),),
    changed_modules=(MODULE_PATH.as_posix(),),
    static_paths=(WRAPPER_PATH.as_posix(),),
)


def utc_now() -> str:
    """Return one timezone-aware wall-clock boundary."""

    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def progress(started: float, phase: str, event: str, **details: Any) -> None:
    """Print one flushed boundary so slow live work never looks stalled."""

    suffix = " ".join(f"{key}={value}" for key, value in sorted(details.items()))
    print(
        f"[exp7570] phase={phase} event={event} elapsed_s={time.monotonic() - started:.3f}"
        + (f" {suffix}" if suffix else ""),
        flush=True,
    )


def canonical_hash(value: Any) -> str:
    """Hash stable JSON so changed raw evidence changes the run identity."""

    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def sha256_file(path: Path) -> str:
    """Hash exact file bytes in bounded chunks."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    """Publish one complete JSON object through an atomic rename."""

    current_work_receipt.atomic_json(path, value)


def load_json(path: Path) -> Json:
    """Load one JSON object and reject a non-object top level."""

    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"json_object_required:{path}")
    return value


def precondition_row(
    check: str,
    upstream: str,
    artifact_field: str,
    expected: Any,
    observed: Any,
    *,
    op: str = "==",
    passed: bool | None = None,
) -> Json:
    """Record the exact comparison that permits or blocks live work."""

    result = observed == expected if passed is None else passed
    return {
        "check": check,
        "upstream": upstream,
        "path": upstream,
        "artifact_field": artifact_field,
        "expected": expected,
        "observed": observed,
        "op": op,
        "passed": bool(result),
        "category": "validity",
        "principle": "Dependent evidence and resources must exist before measurement.",
    }


def collect_preconditions(root: Path) -> tuple[list[Json], Json]:
    """Authenticate the roadmap input and read-only resources before measurement."""

    path = root / UPSTREAM_PATH
    available = path.is_file() and path.stat().st_size > 0
    checks = [
        precondition_row(
            "exp7562_upstream_available",
            UPSTREAM_PATH.as_posix(),
            "path",
            "readable_file",
            "readable_file" if available else None,
        )
    ]
    upstream: Json = {}
    if available:
        try:
            upstream = load_json(path)
        except (OSError, ValueError, json.JSONDecodeError):
            checks[0].update(observed="malformed_json", passed=False)
    if upstream:
        comparisons = (
            ("exp7562_identity", "experiment_id", "exp7562-arc-plan-lineage"),
            ("exp7562_ready", "plan_lineage_ready_score", 1),
            ("exp7562_not_flagged", "flagged_adversarial", False),
        )
        for check, field, expected in comparisons:
            checks.append(
                precondition_row(
                    check, UPSTREAM_PATH.as_posix(), field, expected, upstream.get(field)
                )
            )
        observed_class = upstream.get("verdict_class")
        checks.append(
            precondition_row(
                "exp7562_verdict_class",
                UPSTREAM_PATH.as_posix(),
                "verdict_class",
                sorted(ALLOWED_UPSTREAM_CLASSES),
                observed_class,
                op="in",
                passed=observed_class in ALLOWED_UPSTREAM_CLASSES,
            )
        )

    required = (
        Path("AGENTS.md"),
        Path("CODEX.md"),
        Path("CLAUDE.md"),
        Path("research-program.md"),
        Path("ops/exclusion_manifest.yaml"),
        Path("ops/e2e-test-plan.md"),
        REGISTRY_PATH,
        SPEC_PATH,
        MODULE_PATH,
        WRAPPER_PATH,
        TEST_PATH,
    )
    for relative in required:
        present = (root / relative).is_file()
        checks.append(
            precondition_row(
                f"required_input:{relative.as_posix()}",
                relative.as_posix(),
                "path",
                "readable_file",
                "readable_file" if present else None,
            )
        )
    spec = (root / SPEC_PATH).read_text(encoding="utf-8") if (root / SPEC_PATH).is_file() else ""
    checks.append(
        precondition_row(
            "requirement_declared",
            SPEC_PATH.as_posix(),
            "REQ-ARC-WMTE-7570",
            True,
            "REQ-ARC-WMTE-7570" in spec,
        )
    )
    environment_dir = e6.resolve_environment_dir(root)
    available_games = (
        {item.name for item in environment_dir.iterdir() if item.is_dir()}
        if environment_dir is not None
        else set()
    )
    expected_games = {str(row.get("game")) for row in upstream.get("frozen_arc_roster", [])}
    missing_games = sorted(expected_games - available_games)
    checks.append(
        precondition_row(
            "withheld_environment_games_available",
            "environment_files_names_only",
            "missing_games",
            [],
            missing_games,
        )
    )
    return checks, upstream


def first_failed_precondition(checks: Sequence[Mapping[str, Any]]) -> Json | None:
    """Return the first failed comparison while preserving all rows elsewhere."""

    return next((deepcopy(dict(row)) for row in checks if row.get("passed") is not True), None)


def schedule_from_upstream(upstream: Mapping[str, Any]) -> list[Json]:
    """Copy the sealed Exp7562 roster and reject any changed intervention field."""

    rows = upstream.get("frozen_arc_roster")
    expected = lineage.freeze_arc_roster()
    if not isinstance(rows, list) or len(rows) != len(expected):
        raise ValueError("frozen_roster_mismatch:count")
    identity_keys = (
        "episode_id",
        "game",
        "seed",
        "adapter_withheld",
        "policy_action_limit",
        "induction_limit",
        "request_token_ceiling",
        "sampler",
        "checkpoint_scope",
    )
    if any(
        any(source.get(key) != sealed.get(key) for key in identity_keys)
        for source, sealed in zip(rows, expected, strict=True)
    ):
        raise ValueError("frozen_roster_mismatch:identity")
    return [
        {
            **deepcopy(dict(row)),
            "execution_order": index,
            "action_limit": int(row["policy_action_limit"]),
            "request_limit": int(row["induction_limit"]),
            "max_new_tokens_per_call": int(row["request_token_ceiling"]),
            "adapter_disabled": True,
            "stored_engines_disabled": True,
            "registry_trajectories_disabled": True,
            "banked_solutions_disabled": True,
            "game_source_read": False,
            "offline_ground_truth_bfs_disabled": True,
            "disposition": "unstarted",
        }
        for index, row in enumerate(rows)
    ]


def episode_p95(completed_durations: Sequence[float]) -> float:
    """Return the linear observed p95, or the fixed reserve before data exists."""

    values = sorted(float(value) for value in completed_durations if float(value) >= 0.0)
    if not values:
        return DEFAULT_EPISODE_RESERVE_S
    if len(values) == 1:
        return values[0]
    rank = 0.95 * (len(values) - 1)
    lower = math.floor(rank)
    upper = math.ceil(rank)
    return values[lower] + (rank - lower) * (values[upper] - values[lower])


def can_start_episode(remaining_s: float, completed_durations: Sequence[float]) -> bool:
    """Start only when the remaining generation budget can fit measured p95."""

    return float(remaining_s) >= episode_p95(completed_durations)


def remaining_episode_timeout_s(remaining_s: float) -> float:
    """Cap one episode alarm below the exact remaining live-work budget."""

    return max(1.0, min(float(EPISODE_LIMIT_S), float(remaining_s) - 1.0))


def _tokens(episode: Mapping[str, Any]) -> Json:
    rows = [row for row in episode.get("backend_usage_rows", []) if isinstance(row, Mapping)]
    return {
        "prompt": sum(int(row.get("prompt_tokens") or 0) for row in rows),
        "completion": sum(int(row.get("completion_tokens") or 0) for row in rows),
        "total": sum(int(row.get("total_tokens") or 0) for row in rows),
    }


def _outcome_counts(terminals: Sequence[Mapping[str, Any]]) -> Json:
    stages = [str(row.get("terminal_stage")) for row in terminals]
    return {
        "useful": sum(stage in USEFUL_STAGES for stage in stages),
        "unusable": sum(stage in UNUSABLE_STAGES for stage in stages),
        "unknown": sum(stage not in VALID_TERMINAL_STAGES for stage in stages),
        "censored": sum(stage in CENSORED_STAGES for stage in stages),
    }


def _supervisor_reduction(rows: Sequence[Mapping[str, Any]]) -> Json:
    selections = [row for row in rows if row.get("seam") == "supervisor_arm_selection"]
    transition_episodes = {
        str(row.get("episode_id"))
        for row in rows
        if row.get("seam") == "level_transition"
        and row.get("plan_linked") is True
        and int(row.get("level_delta") or 0) > 0
    }
    fired = Counter(
        str(row.get("chosen_arm"))
        for row in selections
        if row.get("chosen_arm") not in {None, "", "no_redirect"}
    )
    helped = Counter(
        str(row.get("chosen_arm"))
        for row in selections
        if row.get("chosen_arm") not in {None, "", "no_redirect"}
        and str(row.get("episode_id")) in transition_episodes
    )
    arms = [{"arm": arm, "fired": fired[arm], "helped": helped[arm]} for arm in sorted(fired)]
    unredirected = sum(row.get("chosen_arm") == "no_redirect" for row in selections)
    return {
        "arms": arms,
        "total_firings": sum(fired.values()),
        "total_helped": sum(helped.values()),
        "stagnations_unredirected": unredirected,
        "refinement_supported": bool(fired) and bool(helped),
        "help_definition": "later joined plan-linked level progress in the same episode",
        "causal_interpretation": "observational_temporal_join_only",
    }


def reduce_live_measurement(
    schedule: Sequence[Mapping[str, Any]],
    episodes: Sequence[Mapping[str, Any]],
    telemetry_rows: Sequence[Mapping[str, Any]],
) -> Json:
    """Reduce live custody without turning missing lifecycle rows into failures."""

    sealed = {str(row["episode_id"]): deepcopy(dict(row)) for row in schedule}
    observed = {
        str(row.get("episode_id")): deepcopy(dict(row))
        for row in episodes
        if row.get("episode_id") in sealed
    }
    rows: list[Json] = []
    for episode_id, schedule_row in sealed.items():
        row = {**schedule_row, **observed.get(episode_id, {})}
        row.setdefault("disposition", "unstarted")
        rows.append(row)
    lineage_reduction = lineage.reduce_lineage_rows(telemetry_rows)
    terminals = list(lineage_reduction["terminal_rows"])
    outcomes = _outcome_counts(terminals)
    unknown_ids = list(lineage_reduction.get("unknown_attempt_ids") or [])
    outcomes["unknown"] += len(unknown_ids)
    attempts = int(lineage_reduction.get("opportunity_count") or 0)
    valid_dispositions = attempts - outcomes["unknown"]
    disposition_fraction = valid_dispositions / attempts if attempts else None

    per_game: list[Json] = []
    for game in dict.fromkeys(str(row.get("game")) for row in rows):
        game_episodes = [row for row in rows if str(row.get("game")) == game]
        ids = {str(row.get("episode_id")) for row in game_episodes}
        game_terminals = [row for row in terminals if str(row.get("episode_id")) in ids]
        stage_counts = dict(
            sorted(Counter(str(row.get("terminal_stage")) for row in game_terminals).items())
        )
        game_tokens = [_tokens(row) for row in game_episodes]
        frame_changes = sum(
            first.get("state_sha256") != second.get("state_sha256")
            for episode in game_episodes
            for first, second in zip(
                episode.get("action_rows", []), episode.get("action_rows", [])[1:], strict=False
            )
        )
        plan_progress = sum(
            row.get("terminal_stage") == "executed_with_level_progress" for row in game_terminals
        )
        per_game.append(
            {
                "game": game,
                "planned_episodes": len(game_episodes),
                "attempted_episodes": sum(
                    row.get("disposition") != "unstarted" for row in game_episodes
                ),
                "completed_episodes": sum(
                    row.get("disposition") == "complete" for row in game_episodes
                ),
                "actions": sum(int(row.get("action_count") or 0) for row in game_episodes),
                "time_s": sum(float(row.get("elapsed_s") or 0.0) for row in game_episodes),
                "tokens": {
                    key: sum(item[key] for item in game_tokens)
                    for key in ("prompt", "completion", "total")
                },
                "attempt_stage_counts": stage_counts,
                "outcomes": _outcome_counts(game_terminals),
                "plan_derived_level_progress": plan_progress,
                "incidental_frame_change": max(0, frame_changes - plan_progress),
                "unidentifiable_attribution": sum(
                    stage not in VALID_TERMINAL_STAGES for stage in stage_counts
                ),
                "solve_provenance": "live_agent_self_discovery",
                "new_solve_credit": 0,
            }
        )
    dispositions = Counter(str(row.get("disposition")) for row in rows)
    censored_units = sum(
        count for disposition, count in dispositions.items() if disposition.startswith("censored")
    )
    failed_units = dispositions["failed"] + dispositions["complete_error"]
    supervisor = _supervisor_reduction(telemetry_rows)
    support = {
        "opportunities": attempts,
        "attempts": attempts,
        "valid_lifecycle_dispositions": valid_dispositions,
        **outcomes,
        "useful_games": sum(row["outcomes"]["useful"] > 0 for row in per_game),
        "unusable_games": sum(row["outcomes"]["unusable"] > 0 for row in per_game),
    }
    support_floor = {
        "minimum_opportunities": 1000,
        "minimum_attempts": 100,
        "minimum_games_per_outcome_class": 4,
        "observed": {
            "opportunities": attempts,
            "attempts": attempts,
            "useful_games": support["useful_games"],
            "unusable_games": support["unusable_games"],
        },
    }
    support_floor["met"] = (
        attempts >= 1000
        and attempts >= 100
        and support["useful_games"] >= 4
        and support["unusable_games"] >= 4
    )
    return {
        "lineage": lineage_reduction,
        "rows": rows,
        "per_game_results": per_game,
        "sample_size_budget": {
            "planned": len(rows),
            "attempted": len(rows) - dispositions["unstarted"],
            "completed": dispositions["complete"],
            "excluded": 0,
            "failed": failed_units,
            "censored": censored_units,
            "unstarted": dispositions["unstarted"],
        },
        "support_counts": support,
        "lineage_disposition_fraction": disposition_fraction,
        "plan_lineage_measured_score": int(
            attempts > 0
            and lineage_reduction["valid"]
            and disposition_fraction is not None
            and disposition_fraction >= 0.95
        ),
        "trajectory_supervisor": supervisor,
        "support_floor": support_floor,
        "numeric_gate_quality_claim": False,
        "gate_ready_to_ship": False,
    }


def _gate(
    check: str,
    category: str,
    expected: Any,
    observed: Any,
    passed: bool,
    principle: str,
    *,
    op: str = "==",
) -> Json:
    return {
        "check": check,
        "category": category,
        "expected": expected,
        "observed": observed,
        "op": op,
        "passed": bool(passed),
        "principle": principle,
    }


def _gate_summary(gates: Sequence[Mapping[str, Any]]) -> Json:
    failures = [deepcopy(dict(row)) for row in gates if row.get("passed") is not True]
    return {
        "all_passed": not failures,
        "failed_count": len(failures),
        "first_failure": failures[0] if failures else None,
        "failures": failures,
    }


def _receipts_pass(receipts: Sequence[Mapping[str, Any]]) -> bool:
    by_name = Counter(str(row.get("name")) for row in receipts)
    return all(
        by_name[name] == 1
        and next(row for row in receipts if row.get("name") == name).get("passed") is True
        and next(row for row in receipts if row.get("name") == name).get("exit_code") == 0
        for name in REQUIRED_RECEIPT_NAMES
    )


def _field_principles(keys: Sequence[str]) -> dict[str, Json]:
    specific = {
        "experiment_id": "Bind the exact task, milestone, and run date.",
        "preconditions_checked": "Prevent live work over missing upstream evidence.",
        "MODEL_SPECS": "Keep the mandated model identity explicit before loading.",
        "model_specs": "Bind the resolved file, hash, quantization, and runtime.",
        "model_invoked": "Separate current calls from historical model work.",
        "inference_substrate_class": "Apply the duration floor for actual work.",
        "inference_substrate": "Name the owned CUDA generation path precisely.",
        "execution_venue": "Keep the legal host venue separate from device identity.",
        "duration_s": "Monotonic current-work time exposes padding or omission.",
        "random_seed": "Freeze model, ordering, fitting, and bootstrap randomness.",
        "reproducibility_checksum": "Bind code, inputs, settings, raw rows, and model identity.",
        "rows": "Retain every planned episode; missing is not zero.",
        "sample_size_budget": "Keep completed, failed, censored, and unstarted units distinct.",
        "acceptance_gate_results": "Keep validity, readiness, and benefit independent.",
        "gate_check_summary": "Keep the first exact failure actionable.",
        "honest_verdict": "Use an unambiguous complete terminal prefix.",
        "verdict_class": "Use the closed scientific disposition vocabulary.",
        "verifier_is_oracle": "Do not turn probabilistic energy into source truth.",
        "flagged_adversarial": "Preserve quarantine determinations.",
        "validation_receipts": "Bind exact scoped, replay, reduction, and reader checks.",
        "arc_measurement_complete_score": "Measure complete custody, not empirical benefit.",
        "plan_lineage_measured_score": "Require attempts and 95 percent valid dispositions.",
        "per_game_results": "Retain stage denominators, tokens, time, outcomes, and censoring.",
        "solve_provenance": "Prevent repeated public progress from becoming new solve credit.",
        "trajectory_supervisor": "Zero firings cannot support arm refinement.",
        "gate_ready_to_ship": "Observation cannot authorize a suppression policy.",
    }
    return {
        key: {
            "principle": specific.get(
                key, "Keep this typed field explicit so independent readers detect drift."
            )
        }
        for key in keys
        if key != "field_principles"
    }


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Bind the stable candidate while excluding its self-referential fields."""

    copied = deepcopy(dict(artifact))
    copied.pop("reproducibility_checksum", None)
    copied.pop("field_principles", None)
    return canonical_hash(copied)


def build_blocked_artifact(
    run_date: str,
    checks: Sequence[Mapping[str, Any]],
    *,
    duration_s: float,
    schedule: Sequence[Mapping[str, Any]] | None = None,
) -> Json:
    """Publish external absence as blocked without fabricating model work."""

    failure = first_failed_precondition(checks) or precondition_row(
        "exp7562_upstream_available",
        UPSTREAM_PATH.as_posix(),
        "path",
        "readable_file",
        None,
    )
    rows = [deepcopy(dict(row)) for row in (schedule or lineage.freeze_arc_roster())]
    for row in rows:
        row["disposition"] = "unstarted"
    artifact: Json = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "run_date": run_date,
        "status": "complete_blocked_exp7562_not_ready",
        "honest_verdict": "complete_blocked_exp7562_not_ready",
        "verdict_class": "blocked",
        "positive_claim": False,
        "causal_efficacy_claim": False,
        "preconditions_checked": [deepcopy(dict(row)) for row in checks],
        "MODEL_SPECS": list(MODEL_SPECS),
        "model_specs": [],
        "model_invoked": False,
        "invocation_counts": deepcopy(ZERO_INVOCATION_COUNTS),
        "historical_model_calls": {},
        "planned_inference_substrate_class": "model_full_generation",
        "inference_substrate_class": "blocked_no_run",
        "inference_substrate": "blocked_no_run",
        "execution_venue": "host",
        "device_identity": {"kind": "cpu_preflight", "gpu": None},
        "duration_s": max(0.000001, float(duration_s)),
        "phase_spans": [],
        "process_identity": {"pid": os.getpid(), "worktree_root": str(REPO_ROOT)},
        "random_seed": {
            "model": 7570000,
            "episodes": [7570001, 7570002],
            "ordering": 7570003,
            "fitting": 7570004,
            "bootstrap": 7570005,
        },
        "source_artifact_hashes": {},
        "code_hashes": {},
        "raw_telemetry_rows": [],
        "raw_episode_rows": [],
        "rows": rows,
        "per_game_results": [],
        "sample_size_budget": {
            "planned": len(rows),
            "attempted": 0,
            "completed": 0,
            "excluded": 0,
            "failed": 0,
            "censored": 0,
            "unstarted": len(rows),
        },
        "support_counts": {
            "opportunities": 0,
            "attempts": 0,
            "valid_lifecycle_dispositions": 0,
            "useful": 0,
            "unusable": 0,
            "unknown": 0,
            "censored": 0,
            "useful_games": 0,
            "unusable_games": 0,
        },
        "support_floor": {
            "minimum_opportunities": 1000,
            "minimum_attempts": 100,
            "minimum_games_per_outcome_class": 4,
            "met": False,
        },
        "numeric_gate_quality_claim": False,
        "arc_measurement_complete_score": 0,
        "plan_lineage_measured_score": 0,
        "lineage_disposition_fraction": None,
        "trajectory_supervisor": {
            "arms": [],
            "total_firings": 0,
            "total_helped": 0,
            "stagnations_unredirected": 0,
            "refinement_supported": False,
        },
        "solve_provenance": {
            "required": "live_agent_self_discovery",
            "new_solve_claimed": False,
            "credited_levels": 0,
            "official_leaderboard_score_claimed": False,
        },
        "verifier_is_oracle": False,
        "flagged_adversarial": False,
        "gate_ready_to_ship": False,
        "production_defaults_changed": False,
        "generator_weights_changed": False,
        "game_source_read": False,
        "kernel_submitted": False,
        "acceptance_gate_results": [deepcopy(failure)],
        "gate_check_summary": {
            "all_passed": False,
            "failed_count": 1,
            "first_failure": deepcopy(failure),
            "failures": [deepcopy(failure)],
        },
        "validation_receipts": [],
    }
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    artifact["field_principles"] = _field_principles(tuple(artifact))
    return artifact


def _normalized_invocations(counts: Mapping[str, Any]) -> Json:
    normalized = deepcopy(ZERO_INVOCATION_COUNTS)
    for key in normalized:
        if key in counts:
            normalized[key] = int(counts[key])
    return normalized


def build_artifact(
    run_date: str,
    *,
    checks: Sequence[Mapping[str, Any]],
    schedule: Sequence[Mapping[str, Any]],
    episodes: Sequence[Mapping[str, Any]],
    telemetry_rows: Sequence[Mapping[str, Any]],
    reduced: Mapping[str, Any],
    invocation_counts: Mapping[str, Any],
    model_specs: Sequence[Mapping[str, Any]],
    runtime_receipt: Mapping[str, Any],
    validation_receipts: Sequence[Mapping[str, Any]],
    duration_s: float,
    phase_spans: Sequence[Mapping[str, Any]],
    source_hashes: Mapping[str, str],
    terminal: bool,
) -> Json:
    """Build a candidate or terminal artifact from published raw rows."""

    counts = _normalized_invocations(invocation_counts)
    invoked = counts["model_loads_attempted"] + counts["generation_calls_attempted"] > 0
    calls_balanced = all(
        counts[f"{kind}_attempted"]
        == sum(
            counts[f"{kind}_{state}"] for state in ("completed", "failed", "cancelled", "in_flight")
        )
        for kind in ("model_loads", "forward_calls", "generation_calls")
    )
    budget = reduced["sample_size_budget"]
    budget_accounted = sum(
        int(budget[key]) for key in ("completed", "excluded", "failed", "censored", "unstarted")
    ) == int(budget["planned"])
    offload = (
        runtime_receipt.get("offload_real") is True
        or (runtime_receipt.get("observed_cuda_offload") or {}).get("passed") is True
    )
    receipts_ok = _receipts_pass(validation_receipts) if terminal else False
    lineage_valid_or_null = bool(reduced["lineage"]["valid"])
    validity_gates = [
        _gate(
            "preconditions",
            "validity",
            True,
            all(row.get("passed") is True for row in checks),
            all(row.get("passed") is True for row in checks),
            "Missing input evidence must stop live measurement.",
        ),
        _gate(
            "budget_accounted",
            "validity",
            True,
            budget_accounted,
            budget_accounted,
            "Every frozen episode needs a terminal, censored, or unstarted disposition.",
        ),
        _gate(
            "authenticated_current_calls",
            "validity",
            True,
            invoked and calls_balanced and offload,
            invoked and calls_balanced and offload,
            "A live claim needs owned balanced calls and observed CUDA offload.",
        ),
        _gate(
            "lineage_reduction",
            "validity",
            True,
            lineage_valid_or_null,
            lineage_valid_or_null,
            "Malformed joins cannot become useful or unusable labels.",
        ),
        _gate(
            "required_validation",
            "validity",
            True,
            receipts_ok if terminal else "pending",
            receipts_ok,
            "Scoped checks and independent readers must pass before promotion.",
        ),
    ]
    support = reduced["support_counts"]
    benefit_gates = [
        _gate(
            "lineage_dispositions_at_least_95_percent",
            "readiness",
            0.95,
            reduced["lineage_disposition_fraction"],
            reduced["plan_lineage_measured_score"] == 1,
            "Lineage readiness needs nonzero attempts and nearly complete dispositions.",
            op=">=",
        ),
        _gate(
            "historical_gate_quality_floor",
            "benefit",
            {"opportunities": 1000, "attempts": 100, "games_per_class": 4},
            reduced["support_floor"]["observed"],
            reduced["support_floor"]["met"] is True,
            "Small diagnostics cannot support classifier quality or general efficacy.",
            op=">=",
        ),
        _gate(
            "suppression_policy_not_shipped",
            "readiness",
            False,
            False,
            True,
            "Observational joins cannot authorize a live suppression policy.",
        ),
    ]
    gates = [*validity_gates, *benefit_gates]
    complete = int(terminal and all(row["passed"] for row in validity_gates))
    verdict = (
        "complete_null_bounded_live_lineage_feasibility"
        if complete
        else "partial_terminal_validation_pending"
    )
    terminals = list(reduced["lineage"]["terminal_rows"])
    artifact: Json = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "run_date": run_date,
        "status": verdict,
        "honest_verdict": verdict,
        "verdict_class": "null" if complete else "partial",
        "positive_claim": False,
        "causal_efficacy_claim": False,
        "preconditions_checked": [deepcopy(dict(row)) for row in checks],
        "MODEL_SPECS": list(MODEL_SPECS),
        "model_specs": [deepcopy(dict(row)) for row in model_specs] if invoked else [],
        "model_invoked": invoked,
        "invocation_counts": counts,
        "historical_model_calls": {"exp7562": "no_current_calls_in_upstream"},
        "planned_inference_substrate_class": "model_full_generation",
        "inference_substrate_class": "model_full_generation",
        "inference_substrate": "owned_native_cuda_llama_cpp_qwen3.8_27b_gguf",
        "execution_venue": "host",
        "device_identity": {
            "gpu_uuid": runtime_receipt.get("gpu_uuid"),
            "gpu_index": runtime_receipt.get("gpu_index", GPU_INDEX),
            "gpu_name": runtime_receipt.get("gpu_name"),
        },
        "execution_venue_details": deepcopy(dict(runtime_receipt)),
        "duration_s": float(duration_s),
        "phase_spans": [deepcopy(dict(row)) for row in phase_spans],
        "process_identity": {"pid": os.getpid(), "worktree_root": str(REPO_ROOT)},
        "random_seed": {
            "model": 7570000,
            "episodes": [7570001, 7570002],
            "ordering": 7570003,
            "fitting": 7570004,
            "bootstrap": 7570005,
        },
        "source_artifact_hashes": deepcopy(dict(source_hashes)),
        "code_hashes": {},
        "protocol": {
            "total_live_limit_s": TOTAL_LIVE_LIMIT_S,
            "action_limit": ACTION_LIMIT,
            "induction_limit": INDUCTION_LIMIT,
            "request_token_ceiling": MAX_NEW_TOKENS,
            "sampler": "existing_live_default",
            "episode_start_rule": "remaining_budget_gte_completed_episode_p95",
        },
        "raw_telemetry_rows": [deepcopy(dict(row)) for row in telemetry_rows],
        "raw_telemetry_sha256": canonical_hash(telemetry_rows),
        "raw_episode_rows": [deepcopy(dict(row)) for row in episodes],
        "rows": deepcopy(list(reduced["rows"])),
        "per_game_results": deepcopy(list(reduced["per_game_results"])),
        "sample_size_budget": deepcopy(dict(budget)),
        "support_counts": deepcopy(dict(support)),
        "support_floor": deepcopy(dict(reduced["support_floor"])),
        "numeric_gate_quality_claim": False,
        "arc_measurement_complete_score": complete,
        "plan_lineage_measured_score": int(reduced["plan_lineage_measured_score"]),
        "lineage_disposition_fraction": reduced["lineage_disposition_fraction"],
        "lineage_reduction": {
            key: deepcopy(value)
            for key, value in reduced["lineage"].items()
            if key != "terminal_rows"
        },
        "rejection_reasons": dict(
            sorted(
                Counter(
                    str(row.get("closure_reason") or row.get("verifier_outcome") or "unknown")
                    for row in terminals
                    if row.get("terminal_stage")
                    in {"transport_failed", "parse_rejected", "verifier_rejected"}
                ).items()
            )
        ),
        "accepted_but_unused_plans": sum(
            row.get("terminal_stage") == "planned_not_executed" for row in terminals
        ),
        "trajectory_supervisor": deepcopy(dict(reduced["trajectory_supervisor"])),
        "curated_arm_change_proposed": None,
        "solve_provenance": {
            "required": "live_agent_self_discovery",
            "actual_attempts": "live_agent_self_discovery",
            "new_solve_claimed": False,
            "credited_levels": 0,
            "previously_reproduced_levels_are_new": False,
            "official_leaderboard_score_claimed": False,
            "registry_precheck_prevented_duplicate_credit": True,
        },
        "verifier_is_oracle": False,
        "flagged_adversarial": False,
        "gate_ready_to_ship": False,
        "production_defaults_changed": False,
        "suppression_policy_shipped": False,
        "new_supervisor_arm_shipped": False,
        "generator_weights_changed": False,
        "per_game_adapters": False,
        "registry_trajectories_used": False,
        "game_source_read": False,
        "offline_ground_truth_bfs_used": False,
        "kernel_submitted": False,
        "acceptance_gate_results": gates,
        "gate_check_summary": _gate_summary(gates),
        "validation_receipts": [deepcopy(dict(row)) for row in validation_receipts],
    }
    for path in (MODULE_PATH, WRAPPER_PATH, TEST_PATH, SPEC_PATH):
        absolute = REPO_ROOT / path
        if absolute.is_file():
            artifact["code_hashes"][path.as_posix()] = sha256_file(absolute)
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    artifact["field_principles"] = _field_principles(tuple(artifact))
    return artifact


def independent_reduce(artifact: Mapping[str, Any]) -> Json:
    """Rebuild lifecycle readiness and episode accounting from raw published rows."""

    reduced = reduce_live_measurement(
        artifact.get("rows") or [],
        artifact.get("raw_episode_rows") or [],
        artifact.get("raw_telemetry_rows") or [],
    )
    return {
        "planned": reduced["sample_size_budget"]["planned"],
        "attempted": reduced["sample_size_budget"]["attempted"],
        "completed": reduced["sample_size_budget"]["completed"],
        "lineage_valid": reduced["lineage"]["valid"],
        "lineage_errors": reduced["lineage"]["errors"],
        "plan_lineage_measured_score": reduced["plan_lineage_measured_score"],
        "lineage_disposition_fraction": reduced["lineage_disposition_fraction"],
        "lineage_reduction": {
            key: deepcopy(value)
            for key, value in reduced["lineage"].items()
            if key != "terminal_rows"
        },
        "support_counts": reduced["support_counts"],
        "sample_size_budget": reduced["sample_size_budget"],
    }


def validate_artifact(artifact: Mapping[str, Any], *, require_terminal: bool) -> list[str]:
    """Reject identity, custody, raw-reduction, readiness, or checksum drift."""

    errors: list[str] = []
    if artifact.get("experiment_id") != EXPERIMENT_ID:
        errors.append("experiment_id_mismatch")
    if artifact.get("milestone") != MILESTONE or artifact.get("run_date") != RUN_DATE:
        errors.append("experiment_identity_mismatch")
    if artifact.get("execution_venue") != "host":
        errors.append("execution_venue_invalid")
    if artifact.get("verdict_class") not in {
        "positive",
        "circular_positive",
        "null",
        "blocked",
        "disqualified",
        "partial",
    }:
        errors.append("verdict_class_invalid")
    for field in ("arc_measurement_complete_score", "plan_lineage_measured_score"):
        if type(artifact.get(field)) is not int or artifact.get(field) not in {0, 1}:
            errors.append(f"{field}_invalid")
    if artifact.get("gate_ready_to_ship") is not False:
        errors.append("gate_ready_to_ship_must_be_false")
    if artifact.get("reproducibility_checksum") != reproducibility_checksum(artifact):
        errors.append("reproducibility_checksum_mismatch")
    principles = artifact.get("field_principles")
    if not isinstance(principles, Mapping) or any(
        key != "field_principles" and key not in principles for key in artifact
    ):
        errors.append("field_principles_incomplete")
    if artifact.get("verdict_class") == "blocked":
        if artifact.get("model_invoked") is not False:
            errors.append("blocked_model_invoked")
        if any((artifact.get("invocation_counts") or {}).values()):
            errors.append("blocked_invocation_counts_nonzero")
        if artifact.get("inference_substrate_class") != "blocked_no_run":
            errors.append("blocked_substrate_invalid")
        if not str(artifact.get("honest_verdict") or "").startswith("complete_blocked_"):
            errors.append("blocked_terminal_prefix_missing")
        return sorted(set(errors))

    replay = independent_reduce(artifact)
    if replay["lineage_reduction"] != artifact.get("lineage_reduction"):
        errors.append("lineage_reduction_mismatch")
    if replay["sample_size_budget"] != artifact.get("sample_size_budget"):
        errors.append("sample_size_budget_mismatch")
    if replay["support_counts"] != artifact.get("support_counts"):
        errors.append("lineage_reduction_mismatch")
    if replay["plan_lineage_measured_score"] != artifact.get("plan_lineage_measured_score"):
        errors.append("plan_lineage_score_mismatch")
    counts = _normalized_invocations(artifact.get("invocation_counts") or {})
    if artifact.get("model_invoked") is not (
        counts["model_loads_attempted"] + counts["generation_calls_attempted"] > 0
    ):
        errors.append("model_invoked_mismatch")
    if require_terminal:
        if artifact.get("arc_measurement_complete_score") != 1:
            errors.append("terminal_completion_score_mismatch")
        if artifact.get("verdict_class") != "null":
            errors.append("terminal_verdict_class_mismatch")
        if not _receipts_pass(artifact.get("validation_receipts") or []):
            errors.append("required_validation_failed")
    elif artifact.get("verdict_class") != "partial":
        errors.append("candidate_verdict_class_mismatch")
    return sorted(set(errors))


def cold_replay(path: Path) -> Json:
    """Reload one candidate and independently reconstruct its joins."""

    artifact = load_json(path)
    terminal = artifact.get("arc_measurement_complete_score") == 1
    errors = validate_artifact(artifact, require_terminal=terminal)
    if errors:
        raise ValueError("cold_replay_invalid:" + ",".join(errors))
    return independent_reduce(artifact)


def configure_live_driver() -> None:  # pragma: no cover - host process wiring.
    """Point the qualified E6 process boundary at Exp7570-owned paths."""

    values = {
        "RUN_DATE": RUN_DATE,
        "EXPERIMENT_ID": 7570,
        "EXPERIMENT_NAME": EXPERIMENT_ID,
        "TASK_ID": TASK_ID,
        "SCHEMA": SCHEMA,
        "RESULT_PATH": RESULT_PATH,
        "RAW_DIR": RAW_DIR,
        "SCHEDULE_PATH": SCHEDULE_PATH,
        "SESSION_PATH": SESSION_PATH,
        "BOUNDARY_PATH": BOUNDARY_PATH,
        "RUNTIME_EVENT_PATH": RUNTIME_EVENT_PATH,
        "ACTION_PATH": ACTION_PATH,
        "CHECKPOINT_PATH": CHECKPOINT_PATH,
        "MODULE_PATH": MODULE_PATH,
        "WRAPPER_PATH": WRAPPER_PATH,
        "TEST_PATH": TEST_PATH,
        "EPISODE_SEEDS": (7570001, 7570002),
        "PANEL_GAMES": lineage.FROZEN_GAMES,
        "ACTION_LIMIT": ACTION_LIMIT,
        "REQUEST_LIMIT": INDUCTION_LIMIT,
        "MAX_NEW_TOKENS": MAX_NEW_TOKENS,
        "EPISODE_LIMIT_S": EPISODE_LIMIT_S,
        "TOTAL_LIVE_LIMIT_S": TOTAL_LIVE_LIMIT_S + 600.0,
    }
    for name, value in values.items():
        setattr(e6, name, value)
    e6._configure_live_driver()
    exp7471.AGGREGATE_LIVE_LIMIT_S = TOTAL_LIVE_LIMIT_S + 600.0


def _unstarted(schedule: Mapping[str, Any]) -> Json:  # pragma: no cover - live-only row.
    """Retain one unit that the aggregate budget prevented from starting."""

    return {
        **deepcopy(dict(schedule)),
        "disposition": "unstarted",
        "action_count": 0,
        "start_level": None,
        "peak_level": None,
        "terminal_level": None,
        "action_rows": [],
        "backend_usage_rows": [],
        "elapsed_s": 0.0,
        "solve_provenance": "unstarted",
        "trace_reproduction": {"attempted": False, "passed": False},
        "new_level_credit": 0,
        "recorder_error_count": 0,
        "error": None,
    }


def run_live_session(args: argparse.Namespace) -> int:  # pragma: no cover - owned GPU child.
    """Load one owned model and run roster units until measured p95 no longer fits."""

    started = time.monotonic()
    raw_dir = Path(args.raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    event_path = raw_dir / RUNTIME_EVENT_PATH.name
    action_path = raw_dir / ACTION_PATH.name
    schedule = load_json(Path(args.schedule_path)).get("rows") or []
    capture = exp7471.live_support.DurableRequestCapture(
        raw_dir, event_path, max_new_tokens=MAX_NEW_TOKENS
    )
    proposer: Any = None
    rows: list[Json] = []
    session: Json = {
        "child_pid": os.getpid(),
        "model_loaded": False,
        "model_invoked": False,
        "episodes": rows,
        "runtime_receipt": {},
        "error": None,
    }
    try:
        if int(args.gpu_index) != GPU_INDEX or os.environ.get("CUDA_VISIBLE_DEVICES") != str(
            GPU_INDEX
        ):
            raise RuntimeError("physical GPU 1 was not exclusively selected")
        capture.install()
        from carnot.agentic.arc_executable_world_model import LocalGGUFProposer

        progress(started, "model_load", "before", model_path=args.model_path)
        proposer = e6._construct_induction_proposer(
            args,
            max_tokens=MAX_NEW_TOKENS,
            proposer_type=LocalGGUFProposer,
            induction_codeonly=False,
        )
        if not proposer._ensure_server():
            raise RuntimeError("owned native CUDA llama-server failed to start")
        server_pid = getattr(proposer._proc, "pid", None)
        owned_vram = e6._owned_process_vram_mb(server_pid)
        offload_ok = (
            isinstance(owned_vram, int) and e6.OFFLOAD_MIN_MB <= owned_vram <= e6.OFFLOAD_MAX_MB
        )
        session["runtime_receipt"] = {
            "child_pid": os.getpid(),
            "server_pid": server_pid,
            "physical_gpu_index": GPU_INDEX,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "owned_server_vram_mb_after_load": owned_vram,
            "offload_real": offload_ok,
            "native_binary": proposer.last_launch_argv[0] if proposer.last_launch_argv else None,
            "server_command": list(proposer.last_launch_argv),
            "requested_n_gpu_layers": 999,
            "n_ctx": 49_152,
            "kv_quantization": "q8_0",
            "use_chat_template": True,
            "mtp": False,
            "max_new_tokens": MAX_NEW_TOKENS,
            "request_limit_per_episode": INDUCTION_LIMIT,
        }
        session["model_loaded"] = True
        atomic_json(
            Path(args.checkpoint_path),
            {
                "stage": "model_loaded",
                "model_loaded": True,
                "completed_units": 0,
                "server_pid": server_pid,
                "owned_server_vram_mb": owned_vram,
            },
        )
        progress(
            started,
            "model_load",
            "after",
            server_pid=server_pid,
            owned_server_vram_mb=owned_vram,
            offload_real=offload_ok,
        )
        if not offload_ok:
            raise RuntimeError("owned server did not show authenticated CUDA offload")
        live_started = time.monotonic()
        completed_durations: list[float] = []
        for index, sealed in enumerate(schedule):
            remaining_s = TOTAL_LIVE_LIMIT_S - (time.monotonic() - live_started)
            reserve_s = episode_p95(completed_durations)
            if not can_start_episode(remaining_s, completed_durations):
                rows.extend(_unstarted(row) for row in schedule[index:])
                progress(
                    started,
                    "episode_budget",
                    "stop_before_new_episode",
                    completed_units=index,
                    remaining_s=round(remaining_s, 3),
                    measured_p95_s=round(reserve_s, 3),
                )
                break
            progress(
                started,
                "episode",
                "before",
                episode_id=sealed["episode_id"],
                completed_units=index,
                remaining_s=round(remaining_s, 3),
                measured_p95_s=round(reserve_s, 3),
            )
            inherited_episode_limit = e6.EPISODE_LIMIT_S
            e6.EPISODE_LIMIT_S = remaining_episode_timeout_s(remaining_s)
            try:
                row = e6._run_policy_episode(sealed, proposer, capture, event_path, action_path)
            finally:
                e6.EPISODE_LIMIT_S = inherited_episode_limit
            rows.append(row)
            if row.get("disposition") == "complete":
                completed_durations.append(float(row.get("elapsed_s") or 0.0))
            atomic_json(raw_dir / "episode_rows.json", {"rows": rows})
            atomic_json(
                Path(args.checkpoint_path),
                {
                    "stage": "episodes",
                    "model_loaded": True,
                    "completed_units": len(rows),
                    "total_units": len(schedule),
                    "measured_episode_p95_s": episode_p95(completed_durations),
                },
            )
            progress(
                started,
                "episode",
                "after",
                episode_id=sealed["episode_id"],
                disposition=row["disposition"],
                duration_s=row["elapsed_s"],
                completed_units=len(rows),
            )
        if len(rows) < len(schedule):
            rows.extend(_unstarted(row) for row in schedule[len(rows) :])
        session["live_generation_game_duration_s"] = time.monotonic() - live_started
        session["measured_episode_p95_s"] = episode_p95(completed_durations)
        session["model_invoked"] = bool(
            InvocationBoundaryLedger(REPO_ROOT / BOUNDARY_PATH).read_events()
        )
    except BaseException as exc:
        session["error"] = f"{type(exc).__name__}: {exc}"[:500]
        progress(started, "live_child", "error", error=session["error"])
    finally:
        capture.restore()
        progress(started, "model_unload", "before")
        if proposer is not None:
            proposer.stop()
        progress(started, "model_unload", "after")
        session["duration_s"] = time.monotonic() - started
        atomic_json(Path(args.session_path), session)
        atomic_json(
            Path(args.checkpoint_path),
            {
                "stage": "child_terminal",
                "model_loaded": session["model_loaded"],
                "completed_units": len(rows),
                "terminal_child": True,
            },
        )
    return 0


def _phase(phase: str, phase_started: float, run_started: float) -> Json:
    ended = time.monotonic()
    return {
        "phase": phase,
        "start_s": phase_started - run_started,
        "end_s": ended - run_started,
        "duration_s": ended - phase_started,
        "ended_at_utc": utc_now(),
    }


def _model_spec(resources: Mapping[str, Any]) -> Json:
    spec = deepcopy(dict(resources.get("model_spec") or {}))
    spec.setdefault("repository", MODEL_ID)
    spec["resolved_file"] = str(resources.get("model_path"))
    spec["sha256"] = str(resources.get("model_hash"))
    spec["quantization"] = "Q4_K_M"
    spec["runtime"] = "owned_native_cuda_llama_cpp_server"
    spec["sampler"] = "existing_live_default"
    spec["max_new_tokens"] = MAX_NEW_TOKENS
    return spec


def _source_hashes(root: Path) -> dict[str, str]:
    paths = (UPSTREAM_PATH, SPEC_PATH, REGISTRY_PATH, MODULE_PATH, WRAPPER_PATH, TEST_PATH)
    return {path.as_posix(): sha256_file(root / path) for path in paths if (root / path).is_file()}


def _invocation_counts(events: Sequence[Mapping[str, Any]]) -> Json:
    reduced = exp7471._invocation_reduction(events, child_terminal=True)
    return _normalized_invocations(reduced["invocation_counts"])


def prepare_validation_directories(private: Path) -> Path:
    """Create parents required by pytest's private ``--basetemp`` children."""

    (private / "pytest").mkdir(parents=True, exist_ok=True)
    return private


def e2e_commands(
    root: Path, private: Path
) -> list[validation_scope.CommandSpec]:  # pragma: no cover - command declaration.
    """Declare the applicable ARC telemetry checks and private LLM-off smoke."""

    private.mkdir(parents=True, exist_ok=True)
    pytest = str(root / ".venv/bin/pytest")
    python = str(root / ".venv/bin/python")
    common = ("-n", "0", "-o", "addopts=", "--no-cov")
    return [
        validation_scope.CommandSpec(
            "e2e_011",
            (
                pytest,
                *common,
                f"--basetemp={private / 'e2e011'}",
                "tests/python/test_arc_decision_telemetry.py",
                "-q",
            ),
            "E2E-011 ARC decision telemetry parity",
        ),
        validation_scope.CommandSpec(
            "e2e_012",
            (
                pytest,
                *common,
                f"--basetemp={private / 'e2e012'}",
                "tests/python/test_experiment_7491_e6_timed_live_profile.py",
                "tests/python/test_experiment_7492_e6_timed_cost_profile.py",
                "tests/python/test_arc_decision_telemetry.py",
                "-q",
            ),
            "E2E-012 exclusive timing parity",
        ),
        validation_scope.CommandSpec(
            "e2e_013",
            (
                pytest,
                *common,
                f"--basetemp={private / 'e2e013'}",
                "tests/python/test_arc_decision_telemetry.py",
                "tests/python/test_experiment_7491_e6_timed_live_profile.py",
                "tests/python/test_experiment_7531_b2_induction_gate_measurement.py",
                "tests/python/test_semif_arc_readout_eval.py",
                "-q",
            ),
            "E2E-013 induction-attempt outcome telemetry",
        ),
        validation_scope.CommandSpec(
            "llm_off_environment_smoke",
            (
                "/usr/bin/env",
                "CARNOT_ARC_DISABLE_INDUCTION=1",
                python,
                "-u",
                "scripts/arc_loop_solve.py",
                "--mechanism",
                "e3",
                "--game",
                "r11l",
                "--max-actions",
                "12",
                "--output",
                str(private / "r11l-smoke.json"),
            ),
            "private LLM-off real-environment smoke",
        ),
    ]


def terminal_commands(
    root: Path, candidate: Path
) -> list[validation_scope.CommandSpec]:  # pragma: no cover - command declaration.
    """Declare the entrypoint, cold replay, reducer, and two strict readers."""

    python = str(root / ".venv/bin/python")
    return [
        validation_scope.CommandSpec(
            "declared_entrypoint",
            (
                python,
                "-u",
                WRAPPER_PATH.as_posix(),
                "--date",
                RUN_DATE,
                "--validate",
                str(candidate),
            ),
            "declared capability entrypoint read-only mode",
        ),
        validation_scope.CommandSpec(
            "fresh_process_cold_replay",
            (
                python,
                "-u",
                WRAPPER_PATH.as_posix(),
                "--date",
                RUN_DATE,
                "--cold-replay",
                str(candidate),
            ),
            "fresh-process cold replay",
        ),
        validation_scope.CommandSpec(
            "independent_reduction",
            (
                python,
                "-u",
                "-c",
                (
                    "import json,sys; from pathlib import Path; "
                    "from carnot.experiment_7570_v661_arc_live_lineage import "
                    "independent_reduce,load_json; "
                    "print(json.dumps(independent_reduce(load_json(Path(sys.argv[1]))),sort_keys=True))"
                ),
                str(candidate),
            ),
            "independent raw-row reduction",
        ),
        validation_scope.CommandSpec(
            "adversarial_verify",
            (python, "-u", "scripts/adversarial_verify.py", str(candidate)),
            "exact measured candidate",
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
            "exact measured candidate",
        ),
    ]


def _registry_receipt(root: Path, games: Sequence[str]) -> Json:  # pragma: no cover
    """Record prior public credit without passing registry data to the policy."""

    value = yaml.safe_load((root / REGISTRY_PATH).read_text(encoding="utf-8")) or {}
    indexed = {
        str(row.get("game")): row
        for row in value.get("games", [])
        if isinstance(row, Mapping) and row.get("game")
    }
    return {
        "path": REGISTRY_PATH.as_posix(),
        "sha256": sha256_file(root / REGISTRY_PATH),
        "rows": [
            {
                "game": game,
                "registered": game in indexed,
                "levels_reproduced": indexed.get(game, {}).get("levels_reproduced"),
                "full_game_clear": indexed.get(game, {}).get("full_game_clear"),
            }
            for game in games
        ],
        "new_credit_allowed": False,
        "policy_received_registry_data": False,
    }


def _failed_receipts(receipts: Sequence[Mapping[str, Any]]) -> list[str]:  # pragma: no cover
    return [str(row.get("name")) for row in receipts if row.get("passed") is not True]


def run_experiment(
    root: Path, run_date: str, *, output_path: Path = RESULT_PATH
) -> Json:  # pragma: no cover - host validation and GPU orchestration.
    """Run checks, one leased child, independent readers, and atomic publication."""

    if run_date != RUN_DATE:
        raise ValueError(f"run_date_mismatch:{run_date}")
    started = time.monotonic()
    phase_spans: list[Json] = []
    receipts: list[Json] = []
    progress(started, "startup", "begin", run_date=run_date)

    phase_started = time.monotonic()
    progress(started, "preconditions", "before")
    checks, upstream = collect_preconditions(root)
    schedule: list[Json] = []
    if not first_failed_precondition(checks):
        try:
            schedule = schedule_from_upstream(upstream)
        except ValueError as exc:
            checks.append(
                precondition_row(
                    "exp7562_frozen_roster",
                    UPSTREAM_PATH.as_posix(),
                    "frozen_arc_roster",
                    "exact_exp7562_roster",
                    str(exc),
                )
            )
    phase_spans.append(_phase("preconditions", phase_started, started))
    progress(
        started,
        "preconditions",
        "after",
        passed=first_failed_precondition(checks) is None,
    )
    if first_failed_precondition(checks):
        artifact = build_blocked_artifact(
            run_date, checks, duration_s=time.monotonic() - started, schedule=schedule or None
        )
        atomic_json(root / output_path, artifact)
        return artifact

    registry = _registry_receipt(root, [str(row["game"]) for row in schedule])
    private = prepare_validation_directories(Path(tempfile.mkdtemp(prefix="exp7570-validation-")))
    phase_started = time.monotonic()
    progress(started, "affected_validation", "before")
    affected = validation_scope.run_scoped_validation(
        root,
        AFFECTED_MANIFEST.test_paths,
        AFFECTED_MANIFEST.changed_modules,
        static_paths=AFFECTED_MANIFEST.static_paths,
        basetemp=private / "pytest",
        coverage_file=private / ".coverage.exp7570",
        log_dir=root / RAW_DIR / "validation/affected",
    )
    receipts.extend(affected["validation_receipts"])
    phase_spans.append(_phase("affected_validation", phase_started, started))
    progress(
        started,
        "affected_validation",
        "after",
        passed=affected["required_checks_passed"],
    )

    phase_started = time.monotonic()
    progress(started, "capability_e2e", "before")
    e2e_receipts = validation_scope.run_commands(
        root,
        e2e_commands(root, private / "capability"),
        log_dir=root / RAW_DIR / "validation/capability",
    )
    receipts.extend(e2e_receipts)
    phase_spans.append(_phase("capability_e2e", phase_started, started))
    progress(
        started,
        "capability_e2e",
        "after",
        failed=_failed_receipts(e2e_receipts),
    )
    if _failed_receipts(receipts):
        artifact = build_blocked_artifact(
            run_date, checks, duration_s=time.monotonic() - started, schedule=schedule
        )
        artifact.update(
            status="complete_disqualified_required_validation_failure",
            honest_verdict="complete_disqualified_required_validation_failure",
            verdict_class="disqualified",
            validation_receipts=receipts,
        )
        artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
        artifact["field_principles"] = _field_principles(tuple(artifact))
        atomic_json(root / output_path, artifact)
        return artifact

    configure_live_driver()
    for relative in (
        BOUNDARY_PATH,
        RUNTIME_EVENT_PATH,
        ACTION_PATH,
        SESSION_PATH,
        TELEMETRY_PATH,
        CHECKPOINT_PATH,
        RAW_DIR / "episode_rows.json",
    ):
        (root / relative).unlink(missing_ok=True)
    atomic_json(root / SCHEDULE_PATH, {"rows": schedule, "upstream": UPSTREAM_PATH.as_posix()})
    environment_dir = e6.resolve_environment_dir(root)
    if environment_dir is None:
        raise RuntimeError("validated environment directory disappeared")
    os.environ["CARNOT_ARC_PUBLIC_ENV_DIR"] = str(environment_dir)
    os.environ[TELEMETRY_ENV_FLAG] = "1"
    os.environ[TELEMETRY_PATH_ENV] = str(root / TELEMETRY_PATH)

    phase_started = time.monotonic()
    progress(started, "runtime_admission", "before")
    runtime_checks, runtime_hashes, resources = e6.collect_runtime_preconditions(root, started)
    for row in runtime_checks:
        row.setdefault("upstream", str(row.get("path") or "runtime"))
        row.setdefault("artifact_field", str(row.get("check") or "resource"))
        row.setdefault("op", row.get("operator", "=="))
        row.setdefault("category", "validity")
        row.setdefault("principle", "The live resource must authenticate before measurement.")
    checks.extend(runtime_checks)
    phase_spans.append(_phase("runtime_admission", phase_started, started))
    progress(
        started,
        "runtime_admission",
        "after",
        passed=all(row.get("passed") is True for row in runtime_checks),
    )
    if first_failed_precondition(checks):
        artifact = build_blocked_artifact(
            run_date, checks, duration_s=time.monotonic() - started, schedule=schedule
        )
        atomic_json(root / output_path, artifact)
        return artifact

    if resources.get("gpu") is None:
        raise RuntimeError("validated GPU resource disappeared")
    phase_started = time.monotonic()
    progress(started, "live_measurement", "before", planned_units=len(schedule))
    session = exp7471.run_child_with_lease(
        resources=resources,
        schedule_path=root / SCHEDULE_PATH,
        started=started,
    )
    phase_spans.append(_phase("live_measurement", phase_started, started))
    progress(
        started,
        "live_measurement",
        "after",
        observed_units=len(session.get("episodes") or []),
        error=session.get("error"),
    )
    runtime_receipt = deepcopy(dict(session.get("runtime_receipt") or {}))
    runtime_receipt["registry_precheck"] = registry
    checks.append(
        precondition_row(
            "authenticated_owned_cuda_offload",
            "live_session.runtime_receipt",
            "offload_real",
            True,
            runtime_receipt.get("offload_real") is True
            or (runtime_receipt.get("observed_cuda_offload") or {}).get("passed") is True,
        )
    )
    boundary_events = InvocationBoundaryLedger(root / BOUNDARY_PATH).read_events()
    counts = _invocation_counts(boundary_events)
    episodes = list(session.get("episodes") or [])
    telemetry_rows = load_telemetry(root / TELEMETRY_PATH)
    reduced = reduce_live_measurement(schedule, episodes, telemetry_rows)
    source_hashes: dict[str, Any] = _source_hashes(root)
    source_hashes.update(runtime_hashes)

    phase_started = time.monotonic()
    progress(started, "candidate", "before")
    candidate = build_artifact(
        run_date,
        checks=checks,
        schedule=schedule,
        episodes=episodes,
        telemetry_rows=telemetry_rows,
        reduced=reduced,
        invocation_counts=counts,
        model_specs=[_model_spec(resources)],
        runtime_receipt=runtime_receipt,
        validation_receipts=receipts,
        duration_s=time.monotonic() - started,
        phase_spans=phase_spans,
        source_hashes=source_hashes,
        terminal=False,
    )
    candidate_path = root / RAW_DIR / "measured_terminal_candidate.json"
    atomic_json(candidate_path, candidate)
    phase_spans.append(_phase("candidate", phase_started, started))
    progress(started, "candidate", "after", path=candidate_path)

    phase_started = time.monotonic()
    progress(started, "terminal_readers", "before")
    terminal_receipts = validation_scope.run_commands(
        root,
        terminal_commands(root, candidate_path),
        log_dir=root / RAW_DIR / "validation/terminal",
    )
    receipts.extend(terminal_receipts)
    phase_spans.append(_phase("terminal_readers", phase_started, started))
    progress(
        started,
        "terminal_readers",
        "after",
        failed=_failed_receipts(terminal_receipts),
    )
    final = build_artifact(
        run_date,
        checks=checks,
        schedule=schedule,
        episodes=episodes,
        telemetry_rows=telemetry_rows,
        reduced=reduced,
        invocation_counts=counts,
        model_specs=[_model_spec(resources)],
        runtime_receipt=runtime_receipt,
        validation_receipts=receipts,
        duration_s=time.monotonic() - started,
        phase_spans=phase_spans,
        source_hashes=source_hashes,
        terminal=True,
    )
    if _failed_receipts(receipts):
        final.update(
            status="complete_disqualified_required_validation_failure",
            honest_verdict="complete_disqualified_required_validation_failure",
            verdict_class="disqualified",
            arc_measurement_complete_score=0,
        )
        final["reproducibility_checksum"] = reproducibility_checksum(final)
        final["field_principles"] = _field_principles(tuple(final))
    errors = validate_artifact(final, require_terminal=not _failed_receipts(receipts))
    if errors:
        final.update(
            status="complete_disqualified_terminal_validation_failure",
            honest_verdict="complete_disqualified_terminal_validation_failure",
            verdict_class="disqualified",
            arc_measurement_complete_score=0,
            terminal_validation_errors=errors,
        )
        final["reproducibility_checksum"] = reproducibility_checksum(final)
        final["field_principles"] = _field_principles(tuple(final))
    encoded_size = len(json.dumps(final, sort_keys=True).encode("utf-8"))
    if encoded_size >= 20 * 1024 * 1024:
        raise RuntimeError(f"terminal_artifact_exceeds_20_mib:{encoded_size}")
    atomic_json(root / output_path, final)
    progress(
        started,
        "publish",
        "complete",
        verdict=final["honest_verdict"],
        bytes=encoded_size,
    )
    return final


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the thin public entrypoint and its read-only replay modes."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", choices=[RUN_DATE], required=True)
    parser.add_argument("--role", choices=("experiment", "live-session"), default="experiment")
    parser.add_argument("--model-path")
    parser.add_argument("--model-hash")
    parser.add_argument("--model-revision", default="unknown")
    parser.add_argument("--gpu-index", type=int)
    parser.add_argument("--port", type=int)
    parser.add_argument("--schedule-path", type=Path)
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--checkpoint-path", type=Path, default=CHECKPOINT_PATH)
    parser.add_argument("--session-path", type=Path, default=SESSION_PATH)
    parser.add_argument("--validate", type=Path)
    parser.add_argument("--cold-replay", type=Path)
    return parser.parse_args(argv)


def _run_with_heartbeat(args: argparse.Namespace) -> int:  # pragma: no cover - CLI guard.
    started = time.monotonic()
    stop = threading.Event()

    def emit() -> None:
        while not stop.wait(55.0):
            progress(started, "heartbeat", "pending_operation", role=args.role)

    thread = threading.Thread(target=emit, name="exp7570-progress", daemon=True)
    thread.start()
    try:
        if args.role == "live-session":
            configure_live_driver()
            return run_live_session(args)
        if args.validate is not None:
            errors = validate_artifact(load_json(args.validate), require_terminal=False)
            print(json.dumps({"errors": errors}, sort_keys=True), flush=True)
            return int(bool(errors))
        if args.cold_replay is not None:
            print(json.dumps(cold_replay(args.cold_replay), sort_keys=True), flush=True)
            return 0
        artifact = run_experiment(REPO_ROOT, args.date)
        return int(
            not str(artifact.get("honest_verdict") or "").startswith(
                ("complete_", "complete_blocked_")
            )
        )
    finally:
        stop.set()
        thread.join(timeout=1.0)


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - thin CLI.
    """Run the experiment or one read-only terminal check."""

    return _run_with_heartbeat(parse_args(argv))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
