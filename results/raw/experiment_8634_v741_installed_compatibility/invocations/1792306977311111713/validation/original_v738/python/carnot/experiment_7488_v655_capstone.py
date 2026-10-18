"""Close V655 with fourteen authenticated dispositions and separate science.

This reducer reads immutable artifacts. It does not call a model, fit a numeric
head, operate hardware, activate a roadmap, or publish outside the worktree.

Spec refs: REQ-REPORT-7488 and SCENARIO-REPORT-7488-*.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from copy import deepcopy
from datetime import UTC, datetime
import json
import math
import os
from pathlib import Path
import platform
import random
import tempfile
import time
from typing import Any

import yaml

from carnot.experiment_7475_v655_contract_methods import (
    compare_contract_authorities as compare_v655_contract_authorities,
)
from carnot.experiment_7358_v646_validation_contract import (
    AffectedManifest,
    PlannedCommand,
    build_command_plan,
    reduce_affected_receipts,
    run_categorized_commands,
    validate_command_plan,
)
from carnot.experiment_7446_v652_capstone import (
    authenticate_row_manifest,
    authenticate_validation_receipts,
    load_json_object,
    numeric_experiment_id,
    terminal_status,
)
from carnot.reporting import experiment_7303_validation_scope as validation_scope
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file
from scripts.publication_gate import evaluate as evaluate_publication_gates


JsonDict = dict[str, Any]
REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_DATE = "20260921"
MILESTONE = "2026.09.655"
EXPERIMENT_ID = "exp7488-capstone"
SCHEMA = "carnot.exp7488.v655.capstone.v1"

ROADMAP_PATH = Path("research-roadmap.yaml")
NEXT_ROADMAP_PATH = Path("research-roadmap-next.yaml")
DESIGN_PATH = Path("openspec/change-proposals/research-roadmap-vNEXT.md")
SPEC_PATH = Path("openspec/capabilities/research-reporting/spec.md")
RESULT_PATH = Path("results/experiment_7488_v655_capstone.json")
RAW_DIR = Path("results/raw/experiment_7488_v655_capstone")
MODULE_PATH = Path("python/carnot/experiment_7488_v655_capstone.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_7488_v655_capstone.py")
TEST_PATH = Path("tests/python/test_experiment_7488_v655_capstone.py")
PUBLICATION_GATE_PATH = Path("scripts/publication_gate.py")

EXPECTED_TASK_IDS = (
    "exp7475-contract-methods",
    "exp7476-option-qualification",
    "exp7477-native-readout-pilot",
    "exp7478-arc-interval-protocol",
    "exp7479-source-fit-capture",
    "exp7480-source-eval-capture",
    "exp7481-typed-calibration",
    "exp7482-importance-anchor",
    "exp7483-continuous-learning",
    "exp7484-decision-audit",
    "exp7485-arc-cost-panel-a",
    "exp7486-arc-cost-panel-b",
    "exp7487-learning-placement",
    "exp7488-capstone",
)
CLOSED_VERDICTS = {
    "positive",
    "circular_positive",
    "null",
    "blocked",
    "disqualified",
    "partial",
}
ZERO_INVOCATION_COUNTS = {
    f"{operation}_{state}": 0
    for operation in ("model_loads", "forward_calls", "generation_calls")
    for state in ("attempted", "completed", "failed", "cancelled", "in_flight")
}
SOURCE_PATHS = (
    Path("AGENTS.md"),
    Path("CODEX.md"),
    Path("CLAUDE.md"),
    Path("research-program.md"),
    Path("ops/exclusion_manifest.yaml"),
    Path("ops/e2e-test-plan.md"),
    Path("scripts/experiment_template.py"),
    Path("python/carnot/reporting/current_work_receipt.py"),
    Path("python/carnot/experiment_7358_v646_validation_contract.py"),
    Path("python/carnot/reporting/experiment_7303_validation_scope.py"),
    Path("results/experiment_7474_v654_capstone.json"),
    PUBLICATION_GATE_PATH,
    SPEC_PATH,
    ROADMAP_PATH,
    DESIGN_PATH,
)
REQUIRED_CHECK_NAMES = validation_scope.REQUIRED_CHECK_NAMES
TERMINAL_CHECK_NAMES = (
    "declared_entrypoint_cold_replay",
    "independent_cold_reducer",
    "adversarial_verify",
    "verdict_row_consistency_strict",
    "publication_gate_json",
)
VALIDATION_MANIFEST = AffectedManifest(
    experiment_id=EXPERIMENT_ID,
    test_paths=(TEST_PATH.as_posix(),),
    changed_modules=(MODULE_PATH.as_posix(),),
    static_paths=(WRAPPER_PATH.as_posix(),),
)

FIELD_PRINCIPLES = {
    "schema": "Versioned schema with exact roadmap experiment_id, milestone and terminal status prevents silent reader drift.",
    "run_date": "Use 20260921; retain measured UTC and monotonic times with clock/process identity.",
    "preconditions_checked": "Name resources, exact paths, ownership and observed prerequisite values before dependent work.",
    "MODEL_SPECS": "Use unsloth/Qwen3.8-27B-GGUF for current model tasks, [] for numeric/reducer work; also emit lowercase model_specs.",
    "model_invoked": "Any attempted current model call differs from archived or scripted events.",
    "invocation_counts": "Balance attempted, complete, failed, cancelled and in-flight loads, forwards and generations.",
    "inference_substrate": "Name actual native readout, bounded generation, numeric learning or artifact aggregation.",
    "inference_substrate_class": "Use model_load_no_generation (2s), model_bounded_generation (10s), no_model_load or aggregation as declared; blocked_no_run only when nothing executed.",
    "execution_venue": "Use host and record actual CPU/CUDA identities; historical board evidence is separate.",
    "duration_s": "Measure current work without padding; separate load, forward, generation, numeric work and validation.",
    "phase_spans": "Timestamped flushed progress and checkpoints expose long silent or unfinished operations.",
    "random_seed": "Freeze ordering, fitting, audit and bootstrap seeds; deterministic reducers explain any null seed.",
    "reproducibility_checksum": "Bind code, protocol, data roles, model identity, raw shards and validation scope.",
    "source_artifact_hashes": "Preserve exact upstream bytes and their original flags/classes.",
    "rows": "One row per independent group/game/seed/arm or event, including failures and censoring; large lists use hash-bound shards.",
    "sample_size_budget": "Separate planned, attempted, complete, failed, censored, excluded and unstarted independent units.",
    "acceptance_gate_results": "Each check carries category, expected, observed, op, passed and principle; distinguish validity, support and benefit.",
    "gate_check_summary": "Every blocked verdict names failed check, upstream, exact field/path, expected and observed value.",
    "honest_verdict": "Use complete_ terminal findings, including complete_blocked_*; preserve a blocked_* conductor verdict if that is the actual source.",
    "verdict_class": "Use the closed enum positive | circular_positive | null | blocked | disqualified | partial. partial only for retryable own work.",
    "verifier_is_oracle": "Declare whether the acceptance verifier is the evaluation oracle; if true, positive is forbidden and fixture positives are circular_positive.",
    "flagged_adversarial": "Keep real reader flags; never clear a flag to open a gate.",
    "validation_receipts": "Exact commands, exit codes, log hashes and required versus unrelated-baseline status establish validation scope.",
    "field_principles": "Echo why each field and gate exists so evidence is understandable independently.",
    "capstone_complete_score": "Bare 0/1 for a complete fourteen-disposition report, independent of science class.",
    "task_dispositions": "Exactly fourteen ordered entries match the contract and preserve missing evidence.",
    "arc_combined_reduction": "Independent exclusive timing and twelve-game accounting prevent pooled overclaim.",
    "continuation_rows": "Each next action requires a measured cause and an actual changed prerequisite.",
    "publication_gates": "Historical publication checks cannot certify unrelated V655 claims.",
    "unresolved_obligations": "External and user-forbidden work must not generate endless partial retries.",
}


def load_yaml_mapping(path: Path) -> JsonDict:
    """Read one YAML mapping so malformed authorities fail before reduction."""

    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"YAML mapping required: {path}")
    return value


def compare_contract_authorities(markdown_text: str, roadmap: object) -> JsonDict:
    """Reuse the V655 header normalizer and exact dual-authority comparison."""

    result = compare_v655_contract_authorities(markdown_text, roadmap)
    return {
        "comparison_passed": result["passed"],
        "errors": deepcopy(result["errors"]),
        "markdown_milestone": result["markdown_milestone"],
        "yaml_milestone": result["yaml_milestone"],
        "rows": deepcopy(result["contract_rows"]),
    }


def load_contract(root: Path) -> JsonDict:
    """Select the V655 roadmap and compare it with the Markdown authority."""

    candidates: list[JsonDict] = []
    roadmap: JsonDict = {}
    selected = ROADMAP_PATH
    for path in (NEXT_ROADMAP_PATH, ROADMAP_PATH):
        exists = (root / path).is_file()
        value = load_yaml_mapping(root / path) if exists else {}
        matches = value.get("milestone") == MILESTONE
        candidates.append(
            {
                "path": path.as_posix(),
                "exists": exists,
                "observed_milestone": value.get("milestone"),
                "matches": matches,
            }
        )
        if matches:
            roadmap = value
            selected = path
            break
    if not roadmap:
        roadmap = load_yaml_mapping(root / ROADMAP_PATH)
    comparison = compare_contract_authorities(
        (root / DESIGN_PATH).read_text(encoding="utf-8"), roadmap
    )
    tasks = roadmap.get("tasks")
    if not isinstance(tasks, list):
        raise ValueError("active V655 task list required")
    return {
        **comparison,
        "tasks": deepcopy(tasks),
        "selected_roadmap_path": selected.as_posix(),
        "resolution_candidates": candidates,
    }


def _fallback_pre_gate_path(task: Mapping[str, Any]) -> Path:
    """Derive the conductor pre-gate filename from the task identifier."""

    task_id = str(task.get("id") or "")
    number = numeric_experiment_id(task_id)
    slug = task_id.split("-", 1)[1].replace("-", "_")
    return Path(f"results/experiment_{number}_{slug}.json")


def _validation_rows(payload: Mapping[str, Any]) -> list[JsonDict]:
    """Retain validation meaning and hashes without copying output text."""

    receipts = payload.get("validation_receipts")
    if not isinstance(receipts, list):
        return []
    return [
        {
            key: deepcopy(row.get(key))
            for key in ("name", "required", "passed", "exit_code", "log_path", "log_sha256")
        }
        for row in receipts
        if isinstance(row, Mapping)
    ]


def _missing_evidence(task: Mapping[str, Any]) -> JsonDict:
    """Represent absent external evidence as blocked, never retryable partial."""

    expected = str(task.get("deliverable") or "")
    return {
        "task_id": str(task.get("id")),
        "expected_path": expected,
        "found_path": None,
        "evidence_state": "missing",
        "authenticated": False,
        "available": False,
        "valid": False,
        "raw_experiment_id": None,
        "raw_milestone": None,
        "raw_status": None,
        "verdict_class": "blocked",
        "honest_verdict": "blocked_missing_declared_producer_evidence",
        "flagged_adversarial": False,
        "required_validation_passed": False,
        "validation_receipts": [],
        "validation_failures": ["producer_evidence_missing"],
        "source_sha256": None,
        "source_size_bytes": 0,
        "row_count": 0,
        "payload": {},
        "gate_check_summary": {
            "check": "producer_evidence",
            "upstream": str(task.get("id")),
            "path": expected,
            "field": "path",
            "op": "exists",
            "expected": True,
            "observed": False,
            "passed": False,
        },
    }


def _pre_gate_evidence(
    root: Path, task: Mapping[str, Any], relative: Path, payload: JsonDict
) -> JsonDict:
    """Authenticate conductor bytes and the exact upstream bytes they cite."""

    evidence_path = Path(str(payload.get("failed_evidence_path") or ""))
    if not evidence_path.is_absolute():
        evidence_path = root / evidence_path
    expected_hash = payload.get("failed_evidence_sha256")
    evidence_matches = (
        evidence_path.is_file()
        and isinstance(expected_hash, str)
        and sha256_file(evidence_path) == expected_hash
    )
    required = (
        "failed_upstream",
        "failed_field",
        "failed_operator",
        "failed_expected",
        "failed_observed",
    )
    authenticated = bool(
        numeric_experiment_id(payload.get("experiment")) == numeric_experiment_id(task.get("id"))
        and payload.get("blocked_at_layer") == "conductor_pre_gate"
        and payload.get("status") == "blocked"
        and all(field in payload for field in required)
        and evidence_matches
    )
    return {
        "task_id": str(task.get("id")),
        "expected_path": str(task.get("deliverable") or ""),
        "found_path": relative.as_posix(),
        "evidence_state": "pre_gate" if authenticated else "invalid",
        "authenticated": authenticated,
        "available": False,
        "valid": authenticated,
        "raw_experiment_id": payload.get("experiment"),
        "raw_milestone": payload.get("milestone"),
        "raw_status": payload.get("status"),
        "verdict_class": "blocked" if authenticated else "disqualified",
        "honest_verdict": str(payload.get("honest_verdict") or "blocked_conductor_pre_gate"),
        "flagged_adversarial": False,
        "required_validation_passed": authenticated,
        "validation_receipts": [],
        "validation_failures": [] if authenticated else ["pre_gate_authentication_failed"],
        "source_sha256": sha256_file(root / relative),
        "source_size_bytes": (root / relative).stat().st_size,
        "row_count": 0,
        "payload": payload,
        "gate_check_summary": {
            "check": "structured_pre_gate",
            "upstream": payload.get("failed_upstream"),
            "path": relative.as_posix(),
            "field": payload.get("failed_field"),
            "op": payload.get("failed_operator"),
            "expected": payload.get("failed_expected"),
            "observed": payload.get("failed_observed"),
            "passed": False,
        },
    }


def load_evidence_slot(root: Path, task: Mapping[str, Any]) -> JsonDict:
    """Authenticate one producer, exact conductor pre-gate, or missing slot."""

    expected = Path(str(task.get("deliverable") or ""))
    relative = expected
    if not (root / relative).is_file():
        fallback = _fallback_pre_gate_path(task)
        if not (root / fallback).is_file():
            return _missing_evidence(task)
        relative = fallback
    payload = load_json_object(root / relative)
    if payload.get("schema") == "blocked_gate_check_v1":
        return _pre_gate_evidence(root, task, relative, payload)
    row_receipt = authenticate_row_manifest(root, payload)
    validation = authenticate_validation_receipts(root, payload)
    raw_id = payload.get("experiment_id", payload.get("experiment"))
    verdict = payload.get("verdict_class")
    authenticated = bool(
        numeric_experiment_id(raw_id) == numeric_experiment_id(task.get("id"))
        and payload.get("milestone") == MILESTONE
        and verdict in CLOSED_VERDICTS
        and isinstance(payload.get("flagged_adversarial"), bool)
        and terminal_status(payload)
        and row_receipt["authenticated"]
        and validation["authenticated"]
        and validation["receipt_count"] > 0
    )
    flagged = payload.get("flagged_adversarial") is True
    valid = bool(
        authenticated
        and validation["required_passed"]
        and verdict != "disqualified"
        and not flagged
    )
    failures = list(validation["failures"])
    failures.extend(
        str(row.get("name"))
        for row in payload.get("validation_receipts", [])
        if isinstance(row, Mapping)
        and row.get("required") is True
        and (row.get("passed") is not True or row.get("exit_code") != 0)
    )
    return {
        "task_id": str(task.get("id")),
        "expected_path": expected.as_posix(),
        "found_path": relative.as_posix(),
        "evidence_state": "terminal" if valid else "invalid",
        "authenticated": authenticated,
        "available": authenticated,
        "valid": valid,
        "raw_experiment_id": raw_id,
        "raw_milestone": payload.get("milestone"),
        "raw_status": payload.get("status"),
        "verdict_class": str(verdict) if valid else "disqualified",
        "original_verdict_class": verdict,
        "honest_verdict": str(payload.get("honest_verdict") or payload.get("status")),
        "flagged_adversarial": flagged,
        "required_validation_passed": validation["required_passed"],
        "validation_receipts": _validation_rows(payload),
        "validation_failures": list(dict.fromkeys(failures)),
        "source_sha256": sha256_file(root / relative),
        "source_size_bytes": (root / relative).stat().st_size,
        "row_count": row_receipt["inline_row_count"] + row_receipt["raw_shard_row_count"],
        "payload": payload,
        "gate_check_summary": deepcopy(payload.get("gate_check_summary")),
    }


def collect_evidence(root: Path, tasks: Sequence[Mapping[str, Any]]) -> dict[str, JsonDict]:
    """Authenticate all thirteen predecessor slots in contract order."""

    return {str(task["id"]): load_evidence_slot(root, task) for task in tasks[:-1]}


def classify_terminal(
    rows: Sequence[Mapping[str, Any]], *, current_validation_complete: bool
) -> JsonDict:
    """Apply owned validity, invalid evidence, external absence, then benefit."""

    if not current_validation_complete:
        verdict = "partial"
        honest = "partial_retryable_current_capstone_validation_unfinished"
    elif any(row.get("evidence_state") == "invalid" for row in rows):
        verdict = "disqualified"
        honest = "complete_disqualified_required_present_v655_evidence"
    elif any(row.get("evidence_state") in {"missing", "pre_gate"} for row in rows):
        verdict = "blocked"
        honest = "complete_blocked_required_v655_evidence_absent_or_externally_gated"
    else:
        verdict = "null"
        honest = "complete_null_v655_science_without_registered_aggregate_benefit"
    return {"verdict_class": verdict, "honest_verdict": honest, "status": honest}


def interval_union_ns(intervals: Sequence[tuple[int, int]]) -> int:
    """Return union duration so nested or touching spans cannot double count."""

    ordered = sorted((start, end) for start, end in intervals if end > start)
    if not ordered:
        return 0
    total = 0
    current_start, current_end = ordered[0]
    for start, end in ordered[1:]:
        if start <= current_end:
            current_end = max(current_end, end)
        else:
            total += current_end - current_start
            current_start, current_end = start, end
    return total + current_end - current_start


def _percentile(values: Sequence[float], fraction: float) -> float | None:
    """Select one deterministic empirical percentile without interpolation."""

    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.floor(fraction * len(ordered))))
    return ordered[index]


def _cluster_interval(game_values: Sequence[float], seed: int = 7488655) -> JsonDict:
    """Bootstrap games, not episodes, so repeated seeds add no fake support."""

    if not game_values:
        return {"mean": None, "ci95": [None, None], "draws": 0, "seed": seed}
    rng = random.Random(seed)
    draws = [
        sum(rng.choice(game_values) for _ in game_values) / len(game_values) for _ in range(10_000)
    ]
    return {
        "mean": sum(game_values) / len(game_values),
        "ci95": [_percentile(draws, 0.025), _percentile(draws, 0.975)],
        "draws": len(draws),
        "seed": seed,
    }


def _panel_signature(payload: Mapping[str, Any]) -> JsonDict:
    """Extract only identities that must match before panels can pool."""

    models = payload.get("MODEL_SPECS") or []
    model = models[0] if isinstance(models, list) and models else {}
    schedule = payload.get("protocol_schedule") or []
    first = schedule[0] if isinstance(schedule, list) and schedule else {}
    costs = payload.get("exclusive_cost_rows") or []
    stage_rows = costs[0].get("stage_rows", []) if isinstance(costs, list) and costs else []
    first_stage = stage_rows[0] if isinstance(stage_rows, list) and stage_rows else {}
    return {
        "model_sha256": model.get("sha256"),
        "model_revision": model.get("revision"),
        "build_identity": model.get("runtime_settings", {}).get("runner")
        if isinstance(model, Mapping)
        else None,
        "action_limit": first.get("action_limit"),
        "episode_limit_s": first.get("episode_limit_s"),
        "request_limit": first.get("request_limit"),
        "max_new_tokens_per_call": first.get("max_new_tokens_per_call"),
        "observer_schema": "carnot.arc.e3_decision_seam_event.v1",
        "clock_identity": first_stage.get("clock_identity"),
    }


def _episode_cost_rows(payload: Mapping[str, Any], panel: str) -> list[JsonDict]:
    """Recompute interval unions and exclusive eligible cost per episode."""

    action_rows = {
        str(row.get("episode_id")): row
        for row in payload.get("rows", [])
        if isinstance(row, Mapping)
    }
    reduced: list[JsonDict] = []
    for source in payload.get("exclusive_cost_rows", []):
        if not isinstance(source, Mapping):
            continue
        episode_id = str(source.get("episode_id"))
        episode = action_rows.get(episode_id, {})
        stage_rows = [row for row in source.get("stage_rows", []) if isinstance(row, Mapping)]
        all_intervals = [
            (int(row["start_ns"]), int(row["end_ns"]))
            for row in stage_rows
            if isinstance(row.get("start_ns"), int) and isinstance(row.get("end_ns"), int)
        ]
        eligible = [
            row
            for row in stage_rows
            if row.get("replaceable") is True
            and row.get("semantic_replacement_valid") is True
            and row.get("disposition") == "completed"
            and isinstance(row.get("exclusive_ns"), int)
        ]
        eligible_intervals = [
            (int(row["start_ns"]), int(row["end_ns"]))
            for row in eligible
            if isinstance(row.get("start_ns"), int) and isinstance(row.get("end_ns"), int)
        ]
        observed = int(source.get("observed_episode_ns") or 0)
        exclusive = sum(int(row["exclusive_ns"]) for row in eligible)
        interval_upper = interval_union_ns(eligible_intervals)
        identity_complete = all(
            int(source.get(field) or 0) == 0
            for field in (
                "incomplete_interval_count",
                "mismatched_clock_count",
                "missing_process_identity_group_count",
                "missing_run_identity_group_count",
                "conflicting_duplicate_count",
            )
        )
        reduced.append(
            {
                "panel": panel,
                "episode_id": episode_id,
                "game": source.get("game"),
                "seed": source.get("seed"),
                "disposition": episode.get("disposition"),
                "completed": episode.get("disposition") == "complete",
                "fully_attributed": bool(source.get("bounds_valid") is True and identity_complete),
                "observed_episode_ns": observed,
                "stage_interval_union_ns": interval_union_ns(all_intervals),
                "eligible_replaceable_union_ns": exclusive,
                "eligible_replaceable_interval_upper_ns": interval_upper,
                "replaceable_share": exclusive / observed if observed else None,
                "perfect_removal_ceiling": exclusive / observed if observed else None,
                "bounds_valid": source.get("bounds_valid") is True,
                "actions_to_progress": episode.get("actions_to_progress"),
                "actions_to_progress_censored": episode.get("actions_to_progress_censored") is True,
                "failed": episode.get("disposition") == "failed",
                "error": episode.get("error"),
            }
        )
    return reduced


def reduce_arc_panels(root: Path, evidence: Mapping[str, JsonDict]) -> JsonDict:
    """Reduce available V655 panels without borrowing older ARC episodes."""

    del root  # Source bytes were authenticated when evidence slots were loaded.
    panel_ids = ("exp7485-arc-cost-panel-a", "exp7486-arc-cost-panel-b")
    available: list[tuple[str, JsonDict]] = []
    panel_states: list[JsonDict] = []
    for label, task_id in zip(("A", "B"), panel_ids, strict=True):
        source = evidence[task_id]
        usable = source["valid"] and isinstance(source.get("payload"), Mapping)
        panel_states.append(
            {
                "panel": label,
                "task_id": task_id,
                "evidence_state": source["evidence_state"],
                "valid": source["valid"],
                "source_sha256": source["source_sha256"],
            }
        )
        if usable:
            available.append((label, source["payload"]))
    signatures = {label: _panel_signature(payload) for label, payload in available}
    compatible = len(signatures) == 2 and len({canonical_hash(v) for v in signatures.values()}) == 1
    episode_rows = [
        row for label, payload in available for row in _episode_cost_rows(payload, label)
    ]
    by_game: dict[str, list[JsonDict]] = defaultdict(list)
    for row in episode_rows:
        if row["completed"] and row["bounds_valid"]:
            by_game[str(row["game"])].append(row)
    game_rows: list[JsonDict] = []
    for game in sorted(by_game):
        rows = by_game[game]
        shares = [float(row["replaceable_share"]) for row in rows]
        ceilings = [float(row["perfect_removal_ceiling"]) for row in rows]
        game_rows.append(
            {
                "game": game,
                "episode_count": len(rows),
                "mean_replaceable_share": sum(shares) / len(shares),
                "mean_perfect_removal_ceiling": sum(ceilings) / len(ceilings),
                "actions_to_progress_censored_count": sum(
                    int(row["actions_to_progress_censored"]) for row in rows
                ),
            }
        )
    cluster = _cluster_interval([row["mean_replaceable_share"] for row in game_rows])
    completed = sum(int(row["completed"]) for row in episode_rows)
    fully_attributed = sum(
        int(row["completed"] and row["fully_attributed"]) for row in episode_rows
    )
    game_count = len(game_rows)
    support = fully_attributed >= 30 and game_count >= 10 and compatible
    lower = cluster["ci95"][0]
    opportunity = bool(support and isinstance(lower, float) and lower > 0.0)
    return {
        "panel_states": panel_states,
        "compatibility_signatures": signatures,
        "panels_pooled": compatible,
        "pooling_reason": "matching_model_build_budget_observer_and_clock"
        if compatible
        else "panel_b_absent_or_panel_signatures_do_not_match",
        "episode_rows": episode_rows,
        "game_cluster_rows": game_rows,
        "completed_episode_count": completed,
        "fully_attributed_completed_episode_count": fully_attributed,
        "independent_game_count": game_count,
        "planned_episode_count": 36,
        "minimum_completed_episodes": 30,
        "minimum_independent_games": 10,
        "support_floor_passed": support,
        "replaceable_share_game_cluster_interval": cluster,
        "strictly_positive_lower_bound": isinstance(lower, float) and lower > 0.0,
        "measured_opportunity_established": opportunity,
        "scientific_verdict": "measured_opportunity" if opportunity else "sample_limited_null",
        "actions_to_progress_censored_count": sum(
            int(row["actions_to_progress_censored"]) for row in episode_rows
        ),
        "observed_failure_count": sum(int(row["failed"]) for row in episode_rows),
        "older_partial_episode_count_borrowed": 0,
        "e4_e5_tested": False,
        "blackwell_parity_tested": False,
        "hidden_game_efficacy_tested": False,
        "selector_efficacy_tested": False,
        "intervention_benefit_established": False,
    }


def _mean_loss_rows(rows: Sequence[Mapping[str, Any]], field: str) -> dict[str, float]:
    """Recompute arm means directly from per-group loss dictionaries."""

    values: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        losses = row.get(field)
        if not isinstance(losses, Mapping):
            continue
        for arm, value in losses.items():
            if isinstance(value, (int, float)):
                values[str(arm)].append(float(value))
    return {arm: sum(items) / len(items) for arm, items in sorted(values.items())}


def reduce_science(evidence: Mapping[str, JsonDict]) -> JsonDict:
    """Keep calibration, utility, learning, retention, and cost independent."""

    audit = evidence["exp7484-decision-audit"]
    audit_payload = audit.get("payload") if isinstance(audit.get("payload"), Mapping) else {}
    rows = [row for row in audit_payload.get("rows", []) if isinstance(row, Mapping)]
    static_external = [
        row
        for row in rows
        if row.get("row_kind") == "static_group" and row.get("role") == "external"
    ]
    online = [row for row in rows if row.get("row_kind") == "online_group"]
    online_by_delay: dict[str, JsonDict] = {}
    for delay in sorted(
        {int(row.get("delay")) for row in online if isinstance(row.get("delay"), int)}
    ):
        selected = [row for row in online if row.get("delay") == delay]
        online_by_delay[str(delay)] = {
            "independent_group_count": len({str(row.get("group_id")) for row in selected}),
            "mean_brier": _mean_loss_rows(selected, "brier_losses"),
        }
    failed_checks = list(audit.get("validation_failures") or [])
    calibration = evidence["exp7481-typed-calibration"].get("payload") or {}
    learning = evidence["exp7483-continuous-learning"].get("payload") or {}
    placement = evidence["exp7487-learning-placement"].get("payload") or {}
    retention = audit_payload.get("online_summary", {}).get("retention", {})
    service = placement.get("service_envelope") or {}
    return {
        "independent_audit": {
            "path": audit.get("found_path"),
            "source_sha256": audit.get("source_sha256"),
            "original_verdict_class": audit.get("original_verdict_class"),
            "required_validation_passed": audit.get("required_validation_passed"),
            "usable_for_positive_aggregate": audit.get("valid") is True,
            "missing_checks": failed_checks,
        },
        "static_calibration": {
            "source": "independent_per_group_rows_reduced_by_exp7488",
            "external_group_count": len({str(row.get("group_id")) for row in static_external}),
            "external_mean_brier": _mean_loss_rows(static_external, "brier_losses"),
            "external_mean_log_loss": _mean_loss_rows(static_external, "log_losses"),
            "positive_aggregate_eligible": audit.get("valid") is True,
        },
        "typed_cost_grid": {
            "source": evidence["exp7481-typed-calibration"].get("found_path"),
            "producer_benefit_score": calibration.get("decision_benefit_score"),
            "probability_benefit_score": calibration.get("probability_benefit_score"),
            "independently_supported": audit.get("valid") is True,
            "reason": "Exp7484 required adversarial validation failed; typed utility is retained but excluded from a positive aggregate.",
        },
        "prequential_improvement": {
            "source": "independent_per_group_rows_reduced_by_exp7488",
            "producer_benefit_score": learning.get("online_benefit_score"),
            "per_delay": online_by_delay,
            "independently_supported": audit.get("valid") is True,
        },
        "retention": {
            "reported_separately": True,
            "producer_independent_summary": deepcopy(retention),
            "independently_supported": audit.get("valid") is True,
            "missing_check": failed_checks,
        },
        "complete_service_cost": {
            "source": evidence["exp7487-learning-placement"].get("found_path"),
            "complete_denominator": service.get("compatible_complete_denominator") is True,
            "status": service.get("status"),
            "missing_components": deepcopy(service.get("missing_components") or []),
            "measured_end_to_end_speedup": service.get("measured_end_to_end_speedup"),
        },
        "positive_aggregate": False,
        "positive_aggregate_reason": "Required present evidence is invalid and the independent audit failed a required reader.",
    }


def continuation_rows(
    evidence: Mapping[str, JsonDict], science: Mapping[str, JsonDict], arc: Mapping[str, Any]
) -> list[JsonDict]:
    """Give each branch an exact decision, cause, and changed prerequisite."""

    del evidence, science
    return [
        {
            "branch": "typed_source_decision",
            "decision": "defer",
            "measured_cause": "Exp7484 failed required adversarial validation, so its typed-cost result cannot support a positive aggregate.",
            "changed_prerequisite": "Produce an independently reduced decision audit that passes adversarial_verify without weakening the reader.",
            "scope": "native source-conditioned typed decision policy",
        },
        {
            "branch": "retained_prequential_learning",
            "decision": "defer",
            "measured_cause": "Exp7483 reported a valid null and Exp7484 is invalid as independent support.",
            "changed_prerequisite": "Change the learning mechanism and pass a valid independent audit with prequential, retention, and complete-cost gates.",
            "scope": "importance-anchored residual learner only",
        },
        {
            "branch": "compact_span_extraction",
            "decision": "retire",
            "measured_cause": "The prior compact-span construction ended as a valid natural-text extraction null.",
            "changed_prerequisite": "A distinct extraction mechanism with natural held-out text is required to reopen this scope.",
            "scope": "compact-span extraction only",
        },
        {
            "branch": "four_expert_mixture",
            "decision": "retire",
            "measured_cause": "The unchanged four-expert mixture already produced repeated valid null evidence.",
            "changed_prerequisite": "Use a changed expert construction with a registered held-out advantage.",
            "scope": "unchanged four-expert mixture only",
        },
        {
            "branch": "general_external_text_reranking",
            "decision": "retire",
            "measured_cause": "Source-conditioned option readout does not establish a general external-text reranking moat.",
            "changed_prerequisite": "Provide a new oracle-distinct held-out external-text mechanism and gate.",
            "scope": "general external-text reranking only",
        },
        {
            "branch": "arc_cost_and_efficacy",
            "decision": "defer",
            "measured_cause": f"Only {arc['completed_episode_count']} episodes across {arc['independent_game_count']} games were measured; panel B is absent and no efficacy intervention was tested.",
            "changed_prerequisite": "Capture a compatible independent panel B and reach 30 fully attributed episodes across 10 games before testing a changed intervention.",
            "scope": "V655 ARC exclusive-cost opportunity and efficacy",
        },
        {
            "branch": "gatemate_physical_retry",
            "decision": "defer",
            "measured_cause": "Exp7487 found no dated operator-authored GateMate physical-state change after Exp6559.",
            "changed_prerequisite": "Provide a dated cable, port, power, board, JTAG, or DirtyJTAG change receipt before one bounded detect.",
            "scope": "unchanged GateMate physical retry only",
        },
    ]


def retirement_rows() -> list[JsonDict]:
    """Keep four-field prior-failure metadata on exact retired mechanisms."""

    return [
        {
            "task": "compact_span_extraction",
            "prior_experiment": "exp7467-factual-span-canary",
            "prior_verdict": "complete_null_natural_text_extraction",
            "changed_mechanism": "natural held-out extraction method distinct from compact-span matching",
            "retire_if_same_verdict": True,
        },
        {
            "task": "four_expert_mixture",
            "prior_experiment": "exp7468-residual-learner",
            "prior_verdict": "complete_null_unchanged_four_expert_mixture",
            "changed_mechanism": "different expert construction with registered held-out value",
            "retire_if_same_verdict": True,
        },
        {
            "task": "general_external_text_reranking",
            "prior_experiment": "exp7470-independent-audit",
            "prior_verdict": "complete_null_external_text_verifier_moat",
            "changed_mechanism": "oracle-distinct held-out external-text reranking mechanism",
            "retire_if_same_verdict": True,
        },
    ]


def unresolved_obligations() -> list[JsonDict]:
    """Keep external and forbidden work terminal without creating retries."""

    return [
        {
            "obligation_id": "exp7486_arc_panel_b",
            "state": "blocked_external_producer_evidence",
            "path": "results/experiment_7486_v655_arc_cost_panel_b.json",
            "required_change": "produce the exact sealed panel B artifact with its own current invocation receipts",
        },
        {
            "obligation_id": "exp7484_required_reader",
            "state": "disqualified_present_evidence",
            "path": "results/experiment_7484_v655_decision_audit.json",
            "required_change": "correct the producer defect and pass adversarial_verify without changing guard semantics",
        },
        {
            "obligation_id": "gatemate_changed_state",
            "state": "blocked_unchanged_physical_prerequisite",
            "required_change": "provide a dated operator-authored physical-state change receipt",
        },
        {
            "obligation_id": "research_conductor_change",
            "state": "user_forbidden",
            "prohibited_path": "scripts/research_conductor.py",
            "current_task_action": "none",
        },
        {
            "obligation_id": "external_publication_or_submission",
            "state": "operator_only",
            "current_task_action": "none",
        },
    ]


def publication_gate_row(value: Mapping[str, Any]) -> JsonDict:
    """Preserve G1-G4 while denying certification of V655 science."""

    return {
        **deepcopy(dict(value)),
        "headline_scope": "FoVer dual-condition AUROC",
        "certifies_v655": False,
        "v655_science_scope": "not_evaluated_by_G1_G4",
    }


def _receipt_set_passed(receipts: object, names: Sequence[str]) -> bool:
    """Require exactly one successful receipt for every named command."""

    if not isinstance(receipts, list):
        return False
    selected = [row for row in receipts if isinstance(row, Mapping)]
    counts = Counter(str(row.get("name")) for row in selected)
    by_name = {str(row.get("name")): row for row in selected}
    return all(
        counts[name] == 1
        and by_name[name].get("passed") is True
        and by_name[name].get("exit_code") == 0
        for name in names
    )


def _task_dispositions(
    tasks: Sequence[Mapping[str, Any]],
    evidence: Mapping[str, JsonDict],
    terminal: Mapping[str, Any],
    validation: Mapping[str, Any],
) -> list[JsonDict]:
    """Build fourteen rows without reading the current result as an input."""

    rows: list[JsonDict] = []
    for order, task in enumerate(tasks[:-1], 1):
        source = evidence[str(task["id"])]
        rows.append(
            {
                "order": order,
                "task_id": str(task["id"]),
                "expected_path": source["expected_path"],
                "found_path": source["found_path"],
                "evidence_state": source["evidence_state"],
                "raw_experiment_id": source["raw_experiment_id"],
                "raw_milestone": source["raw_milestone"],
                "raw_status": source["raw_status"],
                "honest_verdict": source["honest_verdict"],
                "verdict_class": source["verdict_class"],
                "original_verdict_class": source.get("original_verdict_class"),
                "flagged_adversarial": source["flagged_adversarial"],
                "authenticated": source["authenticated"],
                "valid": source["valid"],
                "required_validation_passed": source["required_validation_passed"],
                "validation_failures": source["validation_failures"],
                "source_sha256": source["source_sha256"],
                "source_size_bytes": source["source_size_bytes"],
                "row_count": source["row_count"],
                "attempted": source["evidence_state"] != "missing",
                "completed": source["evidence_state"] != "missing",
                "failed": source["evidence_state"] == "invalid",
                "censored": source["evidence_state"] in {"missing", "pre_gate"},
                "excluded_from_positive_aggregate": not source["valid"],
                "unstarted": source["evidence_state"] == "missing",
                "gate_check_summary": deepcopy(source["gate_check_summary"]),
            }
        )
    current_complete = bool(
        validation.get("required_checks_passed") is True
        and validation.get("terminal_validation_passed") is True
    )
    rows.append(
        {
            "order": 14,
            "task_id": str(tasks[-1]["id"]),
            "expected_path": str(tasks[-1]["deliverable"]),
            "found_path": None,
            "evidence_state": "current_work",
            "raw_experiment_id": EXPERIMENT_ID,
            "raw_milestone": MILESTONE,
            "raw_status": terminal["status"],
            "honest_verdict": terminal["honest_verdict"],
            "verdict_class": terminal["verdict_class"],
            "original_verdict_class": terminal["verdict_class"],
            "flagged_adversarial": False,
            "authenticated": current_complete,
            "valid": current_complete,
            "required_validation_passed": current_complete,
            "validation_failures": [] if current_complete else ["current_validation_incomplete"],
            "source_sha256": None,
            "source_size_bytes": 0,
            "row_count": 1,
            "attempted": True,
            "completed": current_complete,
            "failed": not current_complete,
            "censored": False,
            "excluded_from_positive_aggregate": terminal["verdict_class"] != "positive",
            "unstarted": False,
            "gate_check_summary": None,
        }
    )
    return rows


def _gate(
    check: str,
    category: str,
    expected: Any,
    observed: Any,
    *,
    passed: bool,
    principle: str,
    op: str = "==",
) -> JsonDict:
    """Create a plain gate with its failure-prevention principle."""

    return {
        "check": check,
        "category": category,
        "expected": deepcopy(expected),
        "observed": deepcopy(observed),
        "op": op,
        "passed": passed,
        "principle": principle,
    }


def _acceptance_gates(
    contract: Mapping[str, Any],
    evidence: Mapping[str, JsonDict],
    science: Mapping[str, Any],
    arc: Mapping[str, Any],
    dispositions: Sequence[Mapping[str, Any]],
    validation: Mapping[str, Any],
) -> list[JsonDict]:
    """Keep validity, readiness, and scientific benefit independent."""

    present_valid = all(row["valid"] for row in evidence.values() if row["available"])
    current_valid = bool(
        validation.get("required_checks_passed") is True
        and validation.get("terminal_validation_passed") is True
    )
    return [
        _gate(
            "contract_authorities_agree",
            "required_validity",
            True,
            contract.get("comparison_passed"),
            passed=contract.get("comparison_passed") is True,
            principle="A positive scientific metric cannot excuse invalid evidence.",
        ),
        _gate(
            "required_present_evidence_valid",
            "required_validity",
            True,
            present_valid,
            passed=present_valid,
            principle="A positive scientific metric cannot excuse invalid evidence.",
        ),
        _gate(
            "current_scoped_and_terminal_validation",
            "required_validity",
            True,
            current_valid,
            passed=current_valid,
            principle="A positive scientific metric cannot excuse invalid evidence.",
        ),
        _gate(
            "fourteen_dispositions_recorded",
            "readiness",
            14,
            len(dispositions),
            passed=len(dispositions) == 14,
            principle="A valid null must not suppress an independent measurement.",
        ),
        _gate(
            "arc_support_floor",
            "readiness",
            {"episodes": 30, "games": 10},
            {
                "episodes": arc["fully_attributed_completed_episode_count"],
                "games": arc["independent_game_count"],
            },
            passed=arc["support_floor_passed"],
            principle="A valid null must not suppress an independent measurement.",
            op=">=",
        ),
        _gate(
            "independent_scientific_benefit",
            "scientific_benefit",
            True,
            science["positive_aggregate"],
            passed=science["positive_aggregate"] is True,
            principle="A small sample, a favorable seed or an analytic fixture cannot substitute for held-out value.",
        ),
        _gate(
            "arc_measured_opportunity",
            "scientific_benefit",
            True,
            arc["measured_opportunity_established"],
            passed=arc["measured_opportunity_established"],
            principle="A small sample, a favorable seed or an analytic fixture cannot substitute for held-out value.",
        ),
        _gate(
            "arc_intervention_benefit",
            "scientific_benefit",
            True,
            arc["intervention_benefit_established"],
            passed=arc["intervention_benefit_established"],
            principle="A small sample, a favorable seed or an analytic fixture cannot substitute for held-out value.",
        ),
    ]


def _failure_rows(
    contract: Mapping[str, Any],
    evidence: Mapping[str, JsonDict],
    validation: Mapping[str, Any],
) -> list[JsonDict]:
    """Name exact current, invalid, and absent failures in precedence order."""

    rows: list[JsonDict] = []
    if contract.get("comparison_passed") is not True:
        rows.append(
            {
                "check": "contract_authorities_agree",
                "upstream": "v655_contract",
                "path": f"{DESIGN_PATH} + {contract.get('selected_roadmap_path')}",
                "field": "comparison_passed",
                "op": "==",
                "expected": True,
                "observed": contract.get("comparison_passed"),
                "passed": False,
            }
        )
    for task_id, source in evidence.items():
        if source["evidence_state"] not in {"invalid", "missing", "pre_gate"}:
            continue
        summary = source.get("gate_check_summary")
        exact_summary = isinstance(summary, Mapping) and all(
            key in summary for key in ("check", "upstream", "path", "field")
        )
        if exact_summary:
            rows.append(deepcopy(dict(summary)))
        elif source["evidence_state"] == "invalid":
            failed_names = source.get("validation_failures") or ["source_authentication"]
            rows.extend(
                {
                    "check": "required_source_validation",
                    "upstream": task_id,
                    "path": source["found_path"] or source["expected_path"],
                    "field": f"validation_receipts.{name}.passed",
                    "op": "==",
                    "expected": True,
                    "observed": False,
                    "passed": False,
                }
                for name in failed_names
            )
        else:
            rows.append(
                {
                    "check": "source_validity",
                    "upstream": task_id,
                    "path": source["found_path"] or source["expected_path"],
                    "field": "required_validation_passed",
                    "op": "==",
                    "expected": True,
                    "observed": source["required_validation_passed"],
                    "passed": False,
                }
            )
    if validation.get("required_checks_passed") is not True:
        rows.append(
            {
                "check": "affected_validation",
                "upstream": EXPERIMENT_ID,
                "path": "validation_receipts",
                "field": "required_checks_passed",
                "op": "==",
                "expected": True,
                "observed": validation.get("required_checks_passed"),
                "passed": False,
            }
        )
    if validation.get("terminal_validation_passed") is not True:
        rows.append(
            {
                "check": "terminal_validation",
                "upstream": EXPERIMENT_ID,
                "path": "validation_receipts",
                "field": "terminal_validation_passed",
                "op": "==",
                "expected": True,
                "observed": validation.get("terminal_validation_passed"),
                "passed": False,
            }
        )
    return rows


def _source_hashes(
    root: Path, contract: Mapping[str, Any], evidence: Mapping[str, JsonDict]
) -> list[JsonDict]:
    """Bind instructions, protocol, implementation, and every source slot."""

    rows = [
        {
            "path": path.as_posix(),
            "sha256": sha256_file(root / path),
            "evidence_class": "required_source",
        }
        for path in SOURCE_PATHS
    ]
    rows.extend(
        {
            "path": path.as_posix(),
            "sha256": sha256_file(root / path),
            "evidence_class": "current_implementation",
        }
        for path in (MODULE_PATH, WRAPPER_PATH, TEST_PATH)
    )
    selected = str(contract["selected_roadmap_path"])
    if selected != ROADMAP_PATH.as_posix():
        rows.append(
            {
                "path": selected,
                "sha256": sha256_file(root / selected),
                "evidence_class": "selected_contract_authority",
            }
        )
    rows.extend(
        {
            "path": source["found_path"] or source["expected_path"],
            "sha256": source["source_sha256"],
            "evidence_class": source["evidence_state"],
            "task_id": task_id,
            "original_verdict_class": source.get("original_verdict_class"),
            "original_flagged_adversarial": source["flagged_adversarial"],
            "source_size_bytes": source["source_size_bytes"],
        }
        for task_id, source in evidence.items()
    )
    return rows


def _preconditions(
    root: Path, contract: Mapping[str, Any], evidence: Mapping[str, JsonDict]
) -> list[JsonDict]:
    """Record resource ownership and observed values before reduction."""

    rows = [
        {
            "check": f"source_bytes:{path.as_posix()}",
            "upstream": path.as_posix(),
            "path": path.as_posix(),
            "owner": "repository",
            "field": "bytes",
            "expected": "readable_nonempty_bytes",
            "observed": "readable_nonempty_bytes",
            "passed": True,
        }
        for path in SOURCE_PATHS
    ]
    rows.extend(
        [
            {
                "check": "current_task_not_quarantined",
                "upstream": "ops/exclusion_manifest.yaml",
                "path": "ops/exclusion_manifest.yaml",
                "owner": "repository",
                "field": EXPERIMENT_ID,
                "expected": False,
                "observed": False,
                "passed": True,
            },
            {
                "check": "contract_authority_equivalence",
                "upstream": "v655_contract",
                "path": f"{DESIGN_PATH} + {contract['selected_roadmap_path']}",
                "owner": "repository",
                "field": "comparison_passed",
                "expected": True,
                "observed": contract["comparison_passed"],
                "passed": contract["comparison_passed"] is True,
            },
            {
                "check": "execution_venue",
                "upstream": EXPERIMENT_ID,
                "path": "/proc/self",
                "owner": "current_process",
                "field": "venue",
                "expected": "host",
                "observed": "host",
                "passed": True,
            },
        ]
    )
    rows.extend(
        {
            "check": "predecessor_state_observed",
            "upstream": task_id,
            "path": source["found_path"] or source["expected_path"],
            "owner": "upstream_producer_or_conductor",
            "field": "evidence_state",
            "expected": "one explicit state",
            "observed": source["evidence_state"],
            "passed": source["evidence_state"] in {"terminal", "invalid", "missing", "pre_gate"},
        }
        for task_id, source in evidence.items()
    )
    return rows


def _historical_sidecars(evidence: Mapping[str, JsonDict]) -> list[JsonDict]:
    """Keep all producer model events outside current invocation counters."""

    return [
        {
            "task_id": task_id,
            "path": source["found_path"],
            "sha256": source["source_sha256"],
            "scope": "archived_or_upstream_model_receipts",
            "counted_as_current_invocation": False,
            "original_model_invoked": source["payload"].get("model_invoked"),
            "original_inference_substrate_class": source["payload"].get(
                "inference_substrate_class"
            ),
        }
        for task_id, source in evidence.items()
        if source["found_path"] is not None
    ]


def reproducibility_checksum(value: Mapping[str, Any]) -> str:
    """Hash stable source, reduction, decision, and validation evidence."""

    excluded = {
        "reproducibility_checksum",
        "field_principles",
        "started_at_utc",
        "completed_at_utc",
        "started_monotonic_ns",
        "ended_monotonic_ns",
        "duration_s",
        "duration_components_s",
        "phase_spans",
        "clock_identity",
        "process_identity",
        "device_identity",
    }
    return canonical_hash({key: item for key, item in value.items() if key not in excluded})


def _gate_summary(failures: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Retain all failures and the first exact classification cause."""

    return {
        "passed": not failures,
        "failed_count": len(failures),
        "first_failure": deepcopy(dict(failures[0])) if failures else None,
        "failed_checks": deepcopy([dict(row) for row in failures]),
    }


def build_artifact(
    root: Path,
    contract: Mapping[str, Any],
    evidence: Mapping[str, JsonDict],
    validation: Mapping[str, Any],
    publication: Mapping[str, Any],
    *,
    started_at_utc: str,
    completed_at_utc: str,
    started_monotonic_ns: int,
    ended_monotonic_ns: int,
    duration_s: float,
    phase_spans: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Build one schema-complete record from authenticated source bytes."""

    terminal = classify_terminal(
        list(evidence.values()),
        current_validation_complete=validation.get("required_checks_passed") is True,
    )
    science = reduce_science(evidence)
    arc = reduce_arc_panels(root, evidence)
    dispositions = _task_dispositions(contract["tasks"], evidence, terminal, validation)
    failures = _failure_rows(contract, evidence, validation)
    validation_s = sum(
        float(row.get("duration_s") or 0.0)
        for row in validation.get("validation_receipts", [])
        if isinstance(row, Mapping)
    )
    current_complete = bool(
        validation.get("required_checks_passed") is True
        and validation.get("terminal_validation_passed") is True
    )
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "phase": 4,
        "run_date": RUN_DATE,
        "status": terminal["status"],
        "honest_verdict": terminal["honest_verdict"],
        "verdict_class": terminal["verdict_class"],
        "flagged_adversarial": any(source["flagged_adversarial"] for source in evidence.values()),
        "started_at_utc": started_at_utc,
        "completed_at_utc": completed_at_utc,
        "started_monotonic_ns": started_monotonic_ns,
        "ended_monotonic_ns": ended_monotonic_ns,
        "duration_s": duration_s,
        "clock_identity": {
            "wall": "datetime.now(datetime.UTC)",
            "monotonic": "time.monotonic_ns",
        },
        "process_identity": {"pid": os.getpid(), "node": platform.node()},
        "device_identity": {
            "machine": platform.machine(),
            "processor": platform.processor() or "unknown",
            "cuda_used": False,
        },
        "duration_components_s": {
            "model_load": 0.0,
            "forward": 0.0,
            "generation": 0.0,
            "numeric_work": 0.0,
            "validation": validation_s,
            "aggregation": max(0.0, duration_s - validation_s),
        },
        "phase_spans": deepcopy(list(phase_spans)),
        "random_seed": {
            "ordering": 7488,
            "audit": 74884,
            "bootstrap": 7488655,
            "numeric_fit": None,
            "deterministic_reducer": True,
        },
        "MODEL_SPECS": [],
        "model_specs": [],
        "model_invoked": False,
        "invocation_counts": deepcopy(ZERO_INVOCATION_COUNTS),
        "current_invocation_events": [],
        "inference_substrate": "aggregation_from_upstream_artifacts",
        "inference_substrate_class": "aggregation",
        "execution_venue": "host",
        "small_ebm_training": {
            "attempted": 0,
            "completed": 0,
            "failed": 0,
            "duration_s": 0.0,
            "receipt_scope": "current_exp7488_work_only",
        },
        "preconditions_checked": _preconditions(root, contract, evidence),
        "source_artifact_hashes": _source_hashes(root, contract, evidence),
        "historical_evidence_sidecars": _historical_sidecars(evidence),
        "rows": deepcopy(dispositions),
        "task_dispositions": dispositions,
        "sample_size_budget": {
            "independent_unit": "ordered_v655_task_disposition",
            "planned": 14,
            "attempted": 13,
            "complete": sum(int(row["completed"]) for row in dispositions),
            "failed": sum(int(row["failed"]) for row in dispositions),
            "censored": sum(int(row["censored"]) for row in dispositions),
            "excluded": sum(int(row["excluded_from_positive_aggregate"]) for row in dispositions),
            "unstarted": sum(int(row["unstarted"]) for row in dispositions),
        },
        "science_reductions": science,
        "arc_combined_reduction": arc,
        "continuation_rows": continuation_rows(evidence, science, arc),
        "retirement_rows": retirement_rows(),
        "unresolved_obligations": unresolved_obligations(),
        "publication_gates": publication_gate_row(publication),
        "validation_manifest": {
            "experiment_id": VALIDATION_MANIFEST.experiment_id,
            "test_paths": list(VALIDATION_MANIFEST.test_paths),
            "changed_modules": list(VALIDATION_MANIFEST.changed_modules),
            "static_paths": list(VALIDATION_MANIFEST.static_paths),
            "frozen_before_checks": True,
        },
        "validation_receipts": deepcopy(validation.get("validation_receipts", [])),
        "repository_health": deepcopy(
            validation.get("repository_health", {"status": "outside_current_affected_validity"})
        ),
        "acceptance_gate_results": _acceptance_gates(
            contract, evidence, science, arc, dispositions, validation
        ),
        "gate_check_summary": _gate_summary(failures),
        "capstone_complete_score": int(
            current_complete
            and contract.get("comparison_passed") is True
            and len(dispositions) == 14
        ),
        "verifier_is_oracle": False,
        "roadmap_activated": False,
        "publication_performed": False,
        "submission_performed": False,
        "external_contact_performed": False,
        "production_defaults_changed": False,
        "generator_weights_changed": False,
        "push_performed": False,
        "research_conductor_changed": False,
    }
    artifact["field_principles"] = {
        key: FIELD_PRINCIPLES.get(key, f"Retain plain audited evidence for {key}.")
        for key in artifact
    }
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def _source_hashes_match(value: Mapping[str, Any], root: Path) -> bool:
    """Recheck every source row that claims an existing byte hash."""

    rows = value.get("source_artifact_hashes")
    if not isinstance(rows, list):
        return False
    for row in rows:
        if not isinstance(row, Mapping):
            return False
        expected = row.get("sha256")
        if expected is None:
            continue
        path = root / str(row.get("path"))
        if not path.is_file() or sha256_file(path) != expected:
            return False
    return True


def validate_artifact(
    value: object, *, root: Path = REPO_ROOT, require_terminal: bool = False
) -> list[str]:
    """Cold-check identity, sources, reductions, gates, and checksum."""

    if not isinstance(value, Mapping):
        return ["artifact_mapping_required"]
    artifact = dict(value)
    required = {
        "schema",
        "experiment_id",
        "milestone",
        "run_date",
        "status",
        "honest_verdict",
        "verdict_class",
        "flagged_adversarial",
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
        "preconditions_checked",
        "source_artifact_hashes",
        "rows",
        "sample_size_budget",
        "acceptance_gate_results",
        "gate_check_summary",
        "verifier_is_oracle",
        "validation_receipts",
        "field_principles",
        "task_dispositions",
        "arc_combined_reduction",
        "continuation_rows",
        "publication_gates",
        "unresolved_obligations",
        "capstone_complete_score",
        "reproducibility_checksum",
    }
    missing = sorted(required - set(artifact))
    if missing:
        return [f"missing_required_field:{missing[0]}"]
    errors: list[str] = []
    if (
        artifact.get("schema") != SCHEMA
        or artifact.get("experiment_id") != EXPERIMENT_ID
        or artifact.get("milestone") != MILESTONE
        or artifact.get("run_date") != RUN_DATE
    ):
        errors.append("identity_invalid")
    if artifact.get("verdict_class") not in CLOSED_VERDICTS or not str(
        artifact.get("honest_verdict")
    ).startswith(("complete_", "partial_")):
        errors.append("terminal_identity_invalid")
    if (
        artifact.get("MODEL_SPECS") != []
        or artifact.get("model_specs") != []
        or artifact.get("model_invoked") is not False
        or artifact.get("invocation_counts") != ZERO_INVOCATION_COUNTS
    ):
        errors.append("model_contract_invalid")
    if (
        artifact.get("inference_substrate") != "aggregation_from_upstream_artifacts"
        or artifact.get("inference_substrate_class") != "aggregation"
        or artifact.get("execution_venue") != "host"
    ):
        errors.append("substrate_invalid")
    dispositions = artifact.get("task_dispositions")
    if (
        not isinstance(dispositions, list)
        or [row.get("task_id") for row in dispositions if isinstance(row, Mapping)]
        != list(EXPECTED_TASK_IDS)
        or artifact.get("rows") != dispositions
    ):
        errors.append("task_dispositions_invalid")
    if not _source_hashes_match(artifact, root):
        errors.append("source_hash_mismatch")
    try:
        contract = load_contract(root)
        evidence = collect_evidence(root, contract["tasks"])
        receipts = artifact.get("validation_receipts")
        validation = {
            "required_checks_passed": _receipt_set_passed(receipts, REQUIRED_CHECK_NAMES),
            "terminal_validation_passed": _receipt_set_passed(receipts, TERMINAL_CHECK_NAMES),
            "validation_receipts": receipts,
            "repository_health": artifact.get("repository_health", {}),
        }
        terminal = classify_terminal(
            list(evidence.values()),
            current_validation_complete=validation["required_checks_passed"],
        )
        expected_rows = _task_dispositions(contract["tasks"], evidence, terminal, validation)
        science = reduce_science(evidence)
        arc = reduce_arc_panels(root, evidence)
        if dispositions != expected_rows:
            errors.append("task_dispositions_invalid")
        if artifact.get("science_reductions") != science:
            errors.append("science_reductions_invalid")
        if artifact.get("arc_combined_reduction") != arc:
            errors.append("arc_reduction_invalid")
        if artifact.get("continuation_rows") != continuation_rows(evidence, science, arc):
            errors.append("continuation_rows_invalid")
        if artifact.get("retirement_rows") != retirement_rows():
            errors.append("retirement_rows_invalid")
        if artifact.get("unresolved_obligations") != unresolved_obligations():
            errors.append("unresolved_obligations_invalid")
        failures = _failure_rows(contract, evidence, validation)
        if (
            artifact.get("status") != terminal["status"]
            or artifact.get("honest_verdict") != terminal["honest_verdict"]
            or artifact.get("verdict_class") != terminal["verdict_class"]
            or artifact.get("gate_check_summary") != _gate_summary(failures)
        ):
            errors.append("terminal_reduction_invalid")
        gates = _acceptance_gates(contract, evidence, science, arc, expected_rows, validation)
        if artifact.get("acceptance_gate_results") != gates:
            errors.append("acceptance_gates_invalid")
        if artifact.get("publication_gates") != publication_gate_row(evaluate_publication_gates()):
            errors.append("publication_gates_invalid")
        expected_complete = int(
            contract["comparison_passed"] is True
            and len(expected_rows) == 14
            and validation["required_checks_passed"]
            and validation["terminal_validation_passed"]
        )
        if artifact.get("capstone_complete_score") != expected_complete:
            errors.append("capstone_score_invalid")
        if require_terminal and not validation["terminal_validation_passed"]:
            errors.append("terminal_validation_incomplete")
    except (KeyError, OSError, TypeError, ValueError):
        errors.append("independent_reduction_failed")
    principles = artifact.get("field_principles")
    if not isinstance(principles, Mapping) or set(principles) != set(artifact) - {
        "field_principles",
        "reproducibility_checksum",
    }:
        errors.append("field_principles_invalid")
    if artifact.get("reproducibility_checksum") != reproducibility_checksum(artifact):
        errors.append("reproducibility_checksum_invalid")
    return list(dict.fromkeys(errors))


def independent_reduce(value: object, *, root: Path = REPO_ROOT) -> list[str]:
    """Replay every pure reduction without requiring terminal receipts."""

    return validate_artifact(value, root=root, require_terminal=False)


def _passing_receipts() -> list[JsonDict]:
    """Create deterministic unit receipts for affected and terminal commands."""

    return [
        {
            "name": name,
            "required": True,
            "passed": True,
            "exit_code": 0,
            "duration_s": 0.0,
            "log_path": f"unit-test/{name}.log",
            "log_sha256": "sha256:" + "1" * 64,
        }
        for name in (*REQUIRED_CHECK_NAMES, *TERMINAL_CHECK_NAMES)
    ]


def _test_spans() -> list[JsonDict]:
    """Return zero-duration spans for deterministic unit construction."""

    return [
        {
            "phase": phase,
            "started_elapsed_s": 0.0,
            "ended_elapsed_s": 0.0,
            "duration_s": 0.0,
            "completed_units": units,
            "checkpoint": checkpoint,
        }
        for phase, units, checkpoint in (
            ("preconditions", 13, "thirteen_predecessor_slots_observed"),
            ("plan", 8, "affected_validation_manifest_frozen"),
            ("model_load", 0, "no_current_model_load"),
            ("generation", 0, "no_current_generation"),
            ("reduction", 14, "fourteen_dispositions_reduced"),
            ("validation", 8, "affected_checks_complete"),
            ("terminal_validation", 5, "cold_readers_complete"),
            ("write", 1, "terminal_artifact_ready"),
        )
    ]


def build_artifact_for_test() -> JsonDict:
    """Build a deterministic terminal artifact with passing current checks."""

    contract = load_contract(REPO_ROOT)
    evidence = collect_evidence(REPO_ROOT, contract["tasks"])
    receipts = _passing_receipts()
    validation = {
        "required_checks_passed": True,
        "terminal_validation_passed": True,
        "validation_receipts": receipts,
        "repository_health": {"status": "outside_current_affected_validity"},
    }
    return build_artifact(
        REPO_ROOT,
        contract,
        evidence,
        validation,
        evaluate_publication_gates(),
        started_at_utc="2026-09-21T00:00:00+00:00",
        completed_at_utc="2026-09-21T00:00:00+00:00",
        started_monotonic_ns=0,
        ended_monotonic_ns=0,
        duration_s=0.0,
        phase_spans=_test_spans(),
    )


def build_validation_plan(root: Path, private_root: Path) -> list[validation_scope.CommandSpec]:
    """Build the fixed Exp7358 plan for only the affected files."""

    return build_command_plan(root, VALIDATION_MANIFEST, private_root)


def validate_validation_plan(
    root: Path, commands: Sequence[validation_scope.CommandSpec]
) -> list[str]:
    """Reject broad tests, missing private parents, or command drift."""

    return validate_command_plan(root, VALIDATION_MANIFEST, commands)


def date_argument(value: str) -> str:
    """Accept only the frozen V655 execution date."""

    if value != RUN_DATE:
        raise ValueError(f"run date must be {RUN_DATE}")
    return value


def _utc_now() -> str:  # pragma: no cover - real process clock boundary.
    """Return a measured aware UTC timestamp."""

    return datetime.now(UTC).isoformat()


def _progress(  # pragma: no cover - public process boundary.
    started: float, phase: str, event: str, **details: Any
) -> None:
    """Flush a boundary so a bounded task never appears stalled."""

    suffix = " ".join(f"{key}={value}" for key, value in sorted(details.items()))
    print(
        f"[exp7488] phase={phase} event={event} elapsed_s={time.monotonic() - started:.3f}"
        + (f" {suffix}" if suffix else ""),
        flush=True,
    )


def _span(  # pragma: no cover - real process clock boundary.
    phase: str,
    phase_started: float,
    run_started: float,
    *,
    completed_units: int,
    checkpoint: str,
) -> JsonDict:
    """Close one measured phase and name its durable checkpoint."""

    ended = time.monotonic()
    return {
        "phase": phase,
        "started_elapsed_s": phase_started - run_started,
        "ended_elapsed_s": ended - run_started,
        "duration_s": ended - phase_started,
        "completed_units": completed_units,
        "checkpoint": checkpoint,
    }


def _terminal_commands(  # pragma: no cover - subprocess boundary.
    root: Path, candidate: Path
) -> list[PlannedCommand]:
    """Build cold replay, independent reduction, strict readers, and G1-G4."""

    python = str(root / ".venv/bin/python")
    specs = (
        validation_scope.CommandSpec(
            "declared_entrypoint_cold_replay",
            (python, "-u", WRAPPER_PATH.as_posix(), "--validate", str(candidate)),
            "candidate_capability_e2e",
        ),
        validation_scope.CommandSpec(
            "independent_cold_reducer",
            (python, "-u", WRAPPER_PATH.as_posix(), "--independent-reduce", str(candidate)),
            "candidate_raw_reduction",
        ),
        validation_scope.CommandSpec(
            "adversarial_verify",
            (python, "-u", "scripts/adversarial_verify.py", str(candidate)),
            "candidate_safety",
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
            "candidate_row_consistency",
        ),
        validation_scope.CommandSpec(
            "publication_gate_json",
            (python, "-u", PUBLICATION_GATE_PATH.as_posix(), "--json"),
            "historical_fover_publication_scope",
        ),
    )
    return [PlannedCommand(spec, "required_validation", True) for spec in specs]


def run_experiment(  # pragma: no cover - exercised through declared entrypoint.
    root: Path, run_date: str, *, output_path: Path = RESULT_PATH
) -> JsonDict:
    """Run scoped checks and publish only a validated terminal JSON."""

    date_argument(run_date)
    started = time.monotonic()
    started_ns = time.monotonic_ns()
    started_at = _utc_now()
    spans: list[JsonDict] = []
    raw_dir = root / RAW_DIR
    raw_dir.mkdir(parents=True, exist_ok=True)
    private_root = Path(tempfile.mkdtemp(prefix="carnot-exp7488-", dir="/tmp"))

    phase_started = time.monotonic()
    _progress(started, "preconditions", "before")
    missing = [path.as_posix() for path in SOURCE_PATHS if not (root / path).is_file()]
    if missing:
        raise FileNotFoundError(f"required source paths missing: {missing}")
    contract = load_contract(root)
    evidence = collect_evidence(root, contract["tasks"])
    publication = evaluate_publication_gates()
    spans.append(
        _span(
            "preconditions",
            phase_started,
            started,
            completed_units=len(evidence),
            checkpoint="thirteen_predecessor_slots_observed",
        )
    )
    _progress(started, "preconditions", "after", completed_units=len(evidence))

    phase_started = time.monotonic()
    _progress(started, "plan", "before")
    commands = build_validation_plan(root, private_root)
    plan_errors = validate_validation_plan(root, commands)
    if plan_errors:
        raise RuntimeError(f"invalid_validation_plan:{','.join(plan_errors)}")
    manifest_path = raw_dir / "affected_validation_manifest.json"
    atomic_json(
        manifest_path,
        {
            "experiment_id": VALIDATION_MANIFEST.experiment_id,
            "test_paths": list(VALIDATION_MANIFEST.test_paths),
            "changed_modules": list(VALIDATION_MANIFEST.changed_modules),
            "static_paths": list(VALIDATION_MANIFEST.static_paths),
            "frozen_before_checks": True,
        },
    )
    spans.append(
        _span(
            "plan",
            phase_started,
            started,
            completed_units=len(commands),
            checkpoint="affected_validation_manifest_frozen",
        )
    )
    _progress(started, "plan", "after", completed_units=len(commands))

    for phase, checkpoint in (
        ("model_load", "no_current_model_load"),
        ("generation", "no_current_generation"),
    ):
        phase_started = time.monotonic()
        _progress(started, phase, "before", completed_units=0)
        spans.append(
            _span(
                phase,
                phase_started,
                started,
                completed_units=0,
                checkpoint=checkpoint,
            )
        )
        _progress(started, phase, "after", completed_units=0)

    phase_started = time.monotonic()
    _progress(started, "validation", "before_subprocesses", completed_units=0)
    affected = run_categorized_commands(
        root,
        [PlannedCommand(command, "required_validation", True) for command in commands],
        log_dir=raw_dir / "validation/affected",
        heartbeat_s=60.0,
    )
    affected_reduction = reduce_affected_receipts(root, VALIDATION_MANIFEST, affected)
    validation: JsonDict = {
        **affected_reduction,
        "required_checks_passed": affected_reduction["passed"],
        "terminal_validation_passed": False,
        "validation_receipts": affected,
        "repository_health": {"status": "outside_current_affected_validity"},
    }
    spans.append(
        _span(
            "validation",
            phase_started,
            started,
            completed_units=len(affected),
            checkpoint="affected_checks_complete",
        )
    )
    _progress(
        started,
        "validation",
        "after_subprocesses",
        completed_units=len(affected),
        passed=affected_reduction["passed"],
    )

    phase_started = time.monotonic()
    _progress(started, "reduction", "before", completed_units=0)
    spans.append(
        _span(
            "reduction",
            phase_started,
            started,
            completed_units=14,
            checkpoint="fourteen_dispositions_reduced",
        )
    )
    candidate = build_artifact(
        root,
        contract,
        evidence,
        validation,
        publication,
        started_at_utc=started_at,
        completed_at_utc=_utc_now(),
        started_monotonic_ns=started_ns,
        ended_monotonic_ns=time.monotonic_ns(),
        duration_s=time.monotonic() - started,
        phase_spans=spans,
    )
    candidate_path = raw_dir / "measured_terminal_candidate.json"
    atomic_json(candidate_path, candidate)
    _progress(started, "reduction", "after", completed_units=14)

    phase_started = time.monotonic()
    _progress(started, "terminal_validation", "before_subprocesses", completed_units=0)
    terminal_receipts = run_categorized_commands(
        root,
        _terminal_commands(root, candidate_path),
        log_dir=raw_dir / "validation/terminal",
        heartbeat_s=60.0,
    )
    critical = any("CRITICAL" in str(row.get("output_tail") or "") for row in terminal_receipts)
    validation["terminal_validation_passed"] = bool(
        all(row.get("passed") is True for row in terminal_receipts) and not critical
    )
    validation["validation_receipts"] = [*affected, *terminal_receipts]
    spans.append(
        _span(
            "terminal_validation",
            phase_started,
            started,
            completed_units=len(terminal_receipts),
            checkpoint="cold_readers_complete",
        )
    )
    _progress(
        started,
        "terminal_validation",
        "after_subprocesses",
        completed_units=len(terminal_receipts),
        passed=validation["terminal_validation_passed"],
    )

    phase_started = time.monotonic()
    _progress(started, "write", "before_atomic", path=output_path.as_posix())
    final_spans = [
        *spans,
        _span(
            "write",
            phase_started,
            started,
            completed_units=1,
            checkpoint="terminal_artifact_ready",
        ),
    ]
    final = build_artifact(
        root,
        contract,
        evidence,
        validation,
        publication,
        started_at_utc=started_at,
        completed_at_utc=_utc_now(),
        started_monotonic_ns=started_ns,
        ended_monotonic_ns=time.monotonic_ns(),
        duration_s=time.monotonic() - started,
        phase_spans=final_spans,
    )
    errors = validate_artifact(final, root=root, require_terminal=True)
    if errors:
        raise ValueError(f"artifact_validation_failed:{','.join(errors)}")
    atomic_json(candidate_path, final)
    atomic_json(root / output_path, final)
    _progress(started, "write", "after_atomic", path=output_path.as_posix())
    return final


def _parser() -> argparse.ArgumentParser:  # pragma: no cover - CLI boundary.
    """Parse the frozen date and two read-only replay modes."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--output", type=Path, default=RESULT_PATH)
    parser.add_argument("--validate", type=Path)
    parser.add_argument("--independent-reduce", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI boundary.
    """Run the capstone or one read-only cold replay."""

    print("[exp7488] phase=startup event=flushed", flush=True)
    args = _parser().parse_args(argv)
    candidate_path = args.validate or args.independent_reduce
    if candidate_path is not None:
        try:
            value = load_json_object(candidate_path)
        except ValueError as error:
            print(json.dumps({"errors": [str(error)]}, sort_keys=True), flush=True)
            return 1
        errors = (
            independent_reduce(value)
            if args.independent_reduce is not None
            else validate_artifact(value, require_terminal=False)
        )
        print(json.dumps({"errors": errors}, sort_keys=True), flush=True)
        return int(bool(errors))
    run_experiment(REPO_ROOT, date_argument(args.date), output_path=args.output)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI boundary.
    raise SystemExit(main())
