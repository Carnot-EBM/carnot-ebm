"""Measure live ARC supervisor opportunities on the frozen V658 panel.

The module reuses the shipped E3 runner, exclusive timer, validation scope,
and eligibility recorder. It adds only the protocol and terminal reduction for
this bounded live measurement.

Spec: REQ-ARC-7527 and SCENARIO-ARC-7527-*.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
from copy import deepcopy
from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time
from typing import Any

from carnot import experiment_7471_v654_arc_seam_observation as exp7471
from carnot import experiment_7491_e6_timed_live_profile as exp7491
from carnot import experiment_7526_v658_arc_eligibility as exp7526
from carnot.agentic.arc_inference_boundary import InvocationBoundaryLedger
from carnot.reporting import current_work_receipt


Json = dict[str, Any]
REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_DATE = "20260922"
MILESTONE = "2026.09.658"
EXPERIMENT_ID = "exp7527-v658-arc-opportunities"
TASK_ID = "experiment_7527_v658_arc_opportunities"
SCHEMA = "carnot.exp7527.v658.arc_opportunities.v1"
MODEL_ID = "unsloth/Qwen3.8-27B-GGUF"
MODEL_FILENAME = "Qwen3.8-27B-Q4_K_M.gguf"
MODEL_SPECS = [MODEL_ID]

PANEL_GAMES = ("g50t", "ka59", "vc33", "ft09", "su15", "bp35")
EPISODE_SEEDS = (658_027, 658_028)
SUPERVISOR_WINDOW = 120
ACTION_LIMIT = 840
EPISODE_LIMIT_S = 180.0
COLLECTION_LIMIT_S = 3000.0
REQUEST_LIMIT = exp7491.REQUEST_LIMIT
MAX_NEW_TOKENS = exp7491.MAX_NEW_TOKENS
OFFLOAD_MIN_MB = 17_000
OFFLOAD_MAX_MB = 20_000
WITHHELD_INPUTS = exp7491.WITHHELD_INPUTS
TERMINAL_DISPOSITIONS = {
    "complete",
    "complete_error",
    "failed",
    "censored_timeout",
    "censored_aggregate_limit",
    "censored_no_first_action",
    "unstarted",
    "unstarted_collection_cap",
}

SPEC_PATH = Path("openspec/capabilities/arc-agi/spec.md")
UPSTREAM_PATH = Path("results/experiment_7526_v658_arc_eligibility.json")
REGISTRY_PATH = Path("ops/arc_solve_registry.yaml")
RESULT_PATH = Path("results/experiment_7527_v658_arc_opportunities.json")
RAW_DIR = Path("results/raw/experiment_7527_v658_arc_opportunities")
SCHEDULE_PATH = RAW_DIR / "frozen_schedule.json"
SESSION_PATH = RAW_DIR / "live_session.json"
BOUNDARY_PATH = RAW_DIR / "current_invocation_events.jsonl"
RUNTIME_EVENT_PATH = RAW_DIR / "runtime_events.jsonl"
ACTION_PATH = RAW_DIR / "live_action_rows.jsonl"
CHECKPOINT_PATH = Path("results/checkpoints/experiment_7527_v658_arc_opportunities.json")
CANDIDATE_PATH = RAW_DIR / "measured_terminal_candidate.json"
MODULE_PATH = Path("python/carnot/experiment_7527_v658_arc_opportunities.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_7527_v658_arc_opportunities.py")
TEST_PATH = Path("tests/python/test_experiment_7527_v658_arc_opportunities.py")

FIELD_PRINCIPLES = {
    "schema": "Version, experiment identity, and milestone bind the terminal reader contract.",
    "run_date": "Real UTC, monotonic, process, and revision evidence prevents a timeless claim.",
    "preconditions_checked": "Exact observations and hashes prevent invented readiness.",
    "MODEL_SPECS": "Name the pinned generator only when current model work occurred.",
    "model_specs": "Bind the model path, bytes, quantization, revision, and runtime.",
    "model_invoked": "Separate current model work from historical provenance.",
    "invocation_counts": "Reconcile every load, forward, and generation disposition.",
    "inference_substrate_class": "The class selects the honest duration floor for actual work.",
    "inference_substrate": "State whether this run used live generation or no current model.",
    "execution_venue": "Host CPU and authenticated CUDA are different evidence classes.",
    "duration_s": "Measured work time prevents padded compute claims.",
    "phase_spans": "Phase and heartbeat timing makes waits and timeouts auditable.",
    "random_seed": "Frozen seeds prevent outcome-aware panel or bootstrap choices.",
    "reproducibility_checksum": "Bind code, protocol, rows, roles, and model identity.",
    "source_artifact_hashes": "Exact upstream bytes preserve exposure and flags.",
    "rows": "Every scheduled episode keeps its measured or censored disposition.",
    "sample_size_budget": "Separate planned, attempted, complete, failed, censored, and unstarted units.",
    "acceptance_gate_results": "Typed operands make validity and benefit independently reducible.",
    "gate_check_summary": "Name exact failures instead of hiding absent or null values.",
    "honest_verdict": "A terminal prefix preserves null, blocked, and disqualified meanings.",
    "verdict_class": "The closed enum prevents retrying an honest null as partial work.",
    "verifier_is_oracle": "No learned-verifier efficacy claim is made from the public outcome oracle.",
    "flagged_adversarial": "Actual findings cannot be cleared to open a gate.",
    "validation_receipts": "Exact commands, exits, and logs prove the affected checks ran.",
    "field_principles": "Each field states the failure that it prevents.",
    "arc_measurement_complete_score": "A bare one means all scheduled dispositions and current work reconcile.",
    "opportunity_support_score": "A bare one means support and eligibility are known, not that an arm works.",
    "per_game_results": "Game and seed rows preserve budgets, progress, cost, and shadow status.",
    "supervisor_opportunity_rows": "Each decision keeps predicates, selection, application, and horizon separate.",
    "solve_provenance": "Only replayed live self-discovery can receive solve provenance.",
    "registry_precheck": "Prior public clears prevent duplicate new-level credit.",
}


def utc_now() -> str:
    """Return one aware UTC timestamp for an observed boundary."""

    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def progress(started: float, phase: str, event: str, **details: Any) -> None:
    """Flush one truthful phase or pending-operation boundary."""

    suffix = " ".join(f"{key}={value}" for key, value in sorted(details.items()))
    print(
        f"[exp7527] phase={phase} event={event} elapsed_s={time.monotonic() - started:.3f}"
        + (f" {suffix}" if suffix else ""),
        flush=True,
    )


def canonical_hash(value: Any) -> str:
    """Hash stable JSON bytes so changed evidence changes its identity."""

    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    """Hash exact file bytes in bounded chunks."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def load_object(path: Path) -> Json:
    """Read one JSON object and return an empty mapping on invalid bytes."""

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(value) if isinstance(value, Mapping) else {}


def _source_row(root: Path, relative: Path, role: str = "required_input") -> Json:
    target = root / relative
    exists = target.is_file() and target.stat().st_size > 0
    return {
        "path": relative.as_posix(),
        "role": role,
        "exists": exists,
        "bytes": target.stat().st_size if exists else 0,
        "sha256": sha256_file(target) if exists else None,
    }


def _check(check: str, path: str, field: str, expected: Any, observed: Any, passed: bool) -> Json:
    return {
        "check": check,
        "path": path,
        "field": field,
        "expected": expected,
        "observed": observed,
        "passed": bool(passed),
    }


def _normalize_check_location(row: Mapping[str, Any]) -> Json:
    """Copy an inherited check and expose its exact resource under this contract."""

    copied = dict(row)
    if not copied.get("path"):
        copied["path"] = copied.get("upstream") or "runtime_preconditions"
    if not copied.get("field"):
        copied["field"] = copied.get("artifact_field") or copied.get("check")
    return copied


def authenticate_upstream_fields(upstream: Mapping[str, Any]) -> list[Json]:
    """Authenticate the producer's terminal identity before using its panel."""

    expected = {
        "schema": "carnot.exp7526.v658.arc_eligibility.v1",
        "experiment_id": "exp7526-arc-eligibility",
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "eligibility_receipt_ready_score": 1,
        "flagged_adversarial": False,
        "verdict_class": "null",
    }
    checks = [
        _check(
            f"exp7526:{field}",
            UPSTREAM_PATH.as_posix(),
            field,
            value,
            upstream.get(field),
            upstream.get(field) == value,
        )
        for field, value in expected.items()
    ]
    receipts = upstream.get("validation_receipts")
    required_pass = (
        isinstance(receipts, list)
        and bool(receipts)
        and all(
            row.get("exit_code") == 0
            for row in receipts
            if isinstance(row, Mapping) and row.get("required", True)
        )
    )
    checks.append(
        _check(
            "exp7526:required_validation_receipts",
            UPSTREAM_PATH.as_posix(),
            "validation_receipts[required].exit_code",
            0,
            0 if required_pass else None,
            required_pass,
        )
    )
    panel = upstream.get("panel_manifest")
    observed_panel = {
        "games": list(panel.get("games") or []) if isinstance(panel, Mapping) else [],
        "episode_seeds": list(panel.get("episode_seeds") or [])
        if isinstance(panel, Mapping)
        else [],
        "supervisor_window": panel.get("supervisor_window") if isinstance(panel, Mapping) else None,
        "action_cap": panel.get("action_cap") if isinstance(panel, Mapping) else None,
        "episode_cap_s": panel.get("episode_cap_s") if isinstance(panel, Mapping) else None,
        "collection_cap_s": panel.get("collection_cap_s") if isinstance(panel, Mapping) else None,
        "schedule_count": len(panel.get("schedule") or []) if isinstance(panel, Mapping) else 0,
    }
    expected_panel = {
        "games": list(PANEL_GAMES),
        "episode_seeds": list(EPISODE_SEEDS),
        "supervisor_window": SUPERVISOR_WINDOW,
        "action_cap": ACTION_LIMIT,
        "episode_cap_s": int(EPISODE_LIMIT_S),
        "collection_cap_s": int(COLLECTION_LIMIT_S),
        "schedule_count": 12,
    }
    checks.append(
        _check(
            "exp7526:frozen_panel",
            UPSTREAM_PATH.as_posix(),
            "panel_manifest",
            expected_panel,
            observed_panel,
            observed_panel == expected_panel,
        )
    )
    registry = upstream.get("registry_precheck")
    policy_received = (
        registry.get("policy_received_registry_data") if isinstance(registry, Mapping) else None
    )
    checks.append(
        _check(
            "exp7526:registry_withheld_from_policy",
            UPSTREAM_PATH.as_posix(),
            "registry_precheck.policy_received_registry_data",
            False,
            policy_received,
            policy_received is False,
        )
    )
    return checks


def collect_preconditions(root: Path) -> tuple[list[Json], dict[str, Json], Json]:
    """Check named files, exclusion status, and then authenticate Exp7526."""

    required = (
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
        Path("python/carnot/experiment_7491_e6_timed_live_profile.py"),
        Path("python/carnot/experiment_7471_v654_arc_seam_observation.py"),
        Path("python/carnot/experiment_7526_v658_arc_eligibility.py"),
        Path("results/experiment_7492_e6_timed_cost_profile.json"),
        Path("results/experiment_7512_v657_arc_opportunity.json"),
        UPSTREAM_PATH,
        REGISTRY_PATH,
        SPEC_PATH,
        MODULE_PATH,
        WRAPPER_PATH,
        TEST_PATH,
    )
    sources = {path.as_posix(): _source_row(root, path) for path in required}
    checks = [
        _check(
            f"required_file:{path.as_posix()}",
            path.as_posix(),
            "readable_nonempty_bytes",
            True,
            row["exists"],
            row["exists"] is True,
        )
        for path in required
        for row in (sources[path.as_posix()],)
    ]
    exclusion = (root / "ops/exclusion_manifest.yaml").read_text(encoding="utf-8")
    excluded = EXPERIMENT_ID in exclusion or TASK_ID in exclusion
    checks.append(
        _check(
            "task_not_excluded",
            "ops/exclusion_manifest.yaml",
            EXPERIMENT_ID,
            False,
            excluded,
            not excluded,
        )
    )
    upstream = (
        load_object(root / UPSTREAM_PATH) if sources[UPSTREAM_PATH.as_posix()]["exists"] else {}
    )
    checks.extend(authenticate_upstream_fields(upstream))
    requirement_present = "REQ-ARC-7527" in (root / SPEC_PATH).read_text(encoding="utf-8")
    checks.append(
        _check(
            "driving_requirement_present",
            SPEC_PATH.as_posix(),
            "REQ-ARC-7527",
            True,
            requirement_present,
            requirement_present,
        )
    )
    return checks, sources, upstream


def build_schedule(upstream: Mapping[str, Any]) -> list[Json]:
    """Copy the authenticated producer schedule and add fixed live limits."""

    failed = [row for row in authenticate_upstream_fields(upstream) if not row["passed"]]
    if failed:
        raise ValueError(f"unauthenticated_exp7526:{failed[0]['field']}")
    source_rows = upstream["panel_manifest"]["schedule"]
    return [
        {
            **deepcopy(dict(row)),
            "execution_order": int(row["order"]),
            "action_limit": ACTION_LIMIT,
            "episode_limit_s": EPISODE_LIMIT_S,
            "request_limit": REQUEST_LIMIT,
            "max_new_tokens_per_call": MAX_NEW_TOKENS,
            "supervisor_window": SUPERVISOR_WINDOW,
            "supervisor_mode": "shadow",
            "withheld_inputs": list(WITHHELD_INPUTS),
            "adapter_disabled": True,
            "stored_solutions_disabled": True,
        }
        for row in source_rows
    ]


def registry_precheck(root: Path, games: Sequence[str]) -> Json:
    """Bind prior public clears without passing registry data to the policy."""

    import yaml

    path = root / REGISTRY_PATH
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    values = loaded.get("games", []) if isinstance(loaded, Mapping) else []
    indexed = {
        str(row.get("game")): row
        for row in values
        if isinstance(row, Mapping) and row.get("game") is not None
    }
    return {
        "path": REGISTRY_PATH.as_posix(),
        "sha256": sha256_file(path),
        "read_before_outcomes": True,
        "policy_received_registry_data": False,
        "rows": [
            {
                "game": game,
                "registered": game in indexed,
                "levels_reproduced": indexed.get(game, {}).get("levels_reproduced"),
                "full_game_clear": indexed.get(game, {}).get("full_game_clear"),
                "new_level_credit_allowed": False,
            }
            for game in games
        ],
    }


def _episode_supervisor_rows(row: Mapping[str, Any]) -> list[Json]:
    values = row.get("supervisor_rows")
    return (
        [dict(value) for value in values if isinstance(value, Mapping)]
        if isinstance(values, list)
        else []
    )


def reduce_panel(rows: Sequence[Mapping[str, Any]]) -> Json:
    """Reduce support without treating shadow selection as a causal arm."""

    complete = [row for row in rows if row.get("disposition") == "complete"]
    complete_games = {str(row.get("game")) for row in complete}
    exposure = sum(int(row.get("window_exposure_count") or 0) for row in rows)
    explicit = sum(int(row.get("explicit_eligibility_count") or 0) for row in rows)
    rate = explicit / exposure if exposure > 0 else None
    opportunities = [
        {
            "episode_id": row.get("episode_id"),
            "game": row.get("game"),
            "seed": row.get("seed"),
            **item,
        }
        for row in rows
        for item in _episode_supervisor_rows(row)
    ]
    eligible_count = sum(int(row.get("eligible_count") or 0) for row in rows)
    selected_count = sum(int(row.get("selected_count") or 0) for row in rows)
    applied_count = sum(int(row.get("applied_count") or 0) for row in rows)
    eligible_choices = [row for row in opportunities if row.get("eligible") is True]
    choice_games = {str(row.get("game")) for row in eligible_choices}
    choice_arms = {str(row.get("arm")) for row in eligible_choices}
    choice_modes = {str(row.get("mode")) for row in eligible_choices}
    future_ready = (
        len(eligible_choices) >= 12
        and len(choice_games) >= 3
        and len(choice_arms) >= 2
        and len(choice_modes) == 1
        and next(iter(choice_modes), "") in {"shadow", "applied"}
    )
    support = (
        len(complete) >= 10
        and complete_games == set(PANEL_GAMES)
        and isinstance(rate, float)
        and rate >= 0.9
    )
    return {
        "row_count": len(rows),
        "complete_episode_count": len(complete),
        "complete_game_count": len(complete_games),
        "complete_games": sorted(complete_games),
        "window_exposure_count": exposure,
        "explicit_eligibility_count": explicit,
        "eligibility_authentication_rate": rate,
        "eligible_opportunity_count": eligible_count,
        "selected_opportunity_count": selected_count,
        "applied_opportunity_count": applied_count,
        "supervisor_opportunity_rows": opportunities,
        "future_causal_trial_ready": future_ready,
        "future_causal_trial_support": {
            "actual_eligible_choices": len(eligible_choices),
            "games": sorted(choice_games),
            "selectable_arms": sorted(choice_arms),
            "application_modes": sorted(choice_modes),
        },
        "opportunity_support_score": int(support),
        "effect_estimate": None,
        "benefit_supported": False,
        "arm_refinement": None,
        "new_level_credit": 0,
    }


def reduce_invocations(events: Sequence[Mapping[str, Any]], *, child_terminal: bool) -> Json:
    """Reduce current boundary events and declare only work that occurred."""

    normalized = [
        {
            **dict(row),
            "operation": row.get("operation") or row.get("kind"),
            "state": row.get("state") or row.get("event"),
        }
        for row in events
    ]
    current = exp7471.live_support.reduce_current_invocations(
        normalized, child_terminal=child_terminal
    )
    counts = dict(current["invocation_counts"])
    if not any(counts.values()) and normalized:
        for operation, prefix in (
            ("model_load", "model_loads"),
            ("forward", "forward_calls"),
            ("generation", "generation_calls"),
        ):
            matching = [row for row in normalized if row.get("operation") == operation]
            for state in ("attempted", "completed", "failed", "cancelled"):
                counts[f"{prefix}_{state}"] = sum(row.get("state") == state for row in matching)
            counts[f"{prefix}_in_flight"] = max(
                0,
                counts[f"{prefix}_attempted"]
                - counts[f"{prefix}_completed"]
                - counts[f"{prefix}_failed"]
                - counts[f"{prefix}_cancelled"],
            )
    generated = counts.get("generation_calls_attempted", 0) > 0
    loaded = counts.get("model_loads_attempted", 0) > 0
    if generated:
        substrate_class = "model_full_generation"
        substrate = "live_llm_inference"
    elif loaded:
        substrate_class = "model_load_no_generation"
        substrate = "live_llm_inference"
    else:
        substrate_class = "blocked_no_run"
        substrate = "aggregation_from_upstream_artifacts"
    return {
        **current,
        "invocation_counts": counts,
        "model_invoked": generated or loaded,
        "inference_substrate_class": substrate_class,
        "inference_substrate": substrate,
        "MODEL_SPECS": list(MODEL_SPECS) if generated or loaded else [],
    }


def _gate(
    check: str,
    category: str,
    expected: Any,
    observed: Any,
    op: str,
    principle: str,
) -> Json:
    if op == "==":
        passed = observed == expected
    elif op == ">=":
        passed = isinstance(observed, (int, float)) and observed >= expected
    else:
        raise ValueError(f"unsupported_gate_operator:{op}")
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
        "failures": failures,
        "first_failure": failures[0] if failures else None,
        "required_validity_and_readiness_passed": all(
            row.get("passed") is True
            for row in gates
            if row.get("category") in {"validity", "readiness"}
        ),
    }


def _artifact_checksum(artifact: Mapping[str, Any]) -> str:
    copied = deepcopy(dict(artifact))
    copied["reproducibility_checksum"] = ""
    return canonical_hash(copied)


def _git_revision(root: Path) -> str | None:
    try:
        completed = subprocess.run(
            ("git", "rev-parse", "HEAD"),
            cwd=root,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return completed.stdout.strip() if completed.returncode == 0 else None


def _interface_stratum(row: Mapping[str, Any]) -> str:
    labels = {
        str(action.get("action"))
        for action in row.get("action_rows", [])
        if isinstance(action, Mapping) and str(action.get("action")) != "RESET"
    }
    if "ACTION6" in labels:
        return "coordinate_action_exposed"
    if labels:
        return "discrete_action_only_observed"
    return "no_post_reset_action_observed"


def _per_game_results(rows: Sequence[Mapping[str, Any]]) -> list[Json]:
    results = []
    for game in PANEL_GAMES:
        episodes = [row for row in rows if row.get("game") == game]
        results.append(
            {
                "game": game,
                "interface_strata": sorted({_interface_stratum(row) for row in episodes}),
                "episode_count": len(episodes),
                "complete_count": sum(row.get("disposition") == "complete" for row in episodes),
                "actions": sum(int(row.get("action_count") or 0) for row in episodes),
                "peak_levels": [row.get("peak_level") for row in episodes],
                "model_calls": sum(
                    int((row.get("request_budget_receipt") or {}).get("attempted") or 0)
                    for row in episodes
                ),
                "model_tokens": sum(
                    int(usage.get("input_tokens") or 0) + int(usage.get("output_tokens") or 0)
                    for row in episodes
                    for usage in row.get("backend_usage_rows", [])
                    if isinstance(usage, Mapping)
                ),
                "exclusive_phase_durations": [row.get("normalized_cost") for row in episodes],
                "window_exposure_count": sum(
                    int(row.get("window_exposure_count") or 0) for row in episodes
                ),
                "eligible_count": sum(int(row.get("eligible_count") or 0) for row in episodes),
                "selected_count": sum(int(row.get("selected_count") or 0) for row in episodes),
                "applied_count": sum(int(row.get("applied_count") or 0) for row in episodes),
                "errors": [row.get("error") for row in episodes if row.get("error")],
                "censored_outcome_horizons": [
                    row.get("censored_outcome_horizon") for row in episodes
                ],
                "episodes": [
                    {
                        key: row.get(key)
                        for key in (
                            "episode_id",
                            "seed",
                            "disposition",
                            "action_count",
                            "start_level",
                            "peak_level",
                            "terminal_level",
                            "elapsed_s",
                            "window_exposure_count",
                            "eligible_count",
                            "selected_count",
                            "applied_count",
                            "solve_provenance",
                            "error",
                        )
                    }
                    for row in episodes
                ],
            }
        )
    return results


def build_artifact(
    root: Path,
    *,
    upstream: Mapping[str, Any],
    schedule: Sequence[Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
    invocation: Mapping[str, Any],
    runtime_receipt: Mapping[str, Any],
    model_specs: Sequence[Mapping[str, Any]],
    preconditions: Sequence[Mapping[str, Any]],
    sources: Mapping[str, Any],
    validation_receipts: Sequence[Mapping[str, Any]],
    phase_spans: Sequence[Mapping[str, Any]],
    registry_receipt: Mapping[str, Any],
    started_at_utc: str,
    ended_at_utc: str,
    duration_s: float,
) -> Json:
    """Build a terminal record whose readiness does not depend on benefit."""

    copied_rows = [deepcopy(dict(row)) for row in rows]
    reduced = reduce_panel(copied_rows)
    dispositions = Counter(str(row.get("disposition")) for row in copied_rows)
    required_receipts = [row for row in validation_receipts if row.get("required", True)]
    validation_passed = bool(required_receipts) and all(
        row.get("passed") is True and row.get("exit_code", 0) == 0 for row in required_receipts
    )
    all_dispositions = len(copied_rows) == len(schedule) == 12 and all(
        row.get("disposition") in TERMINAL_DISPOSITIONS for row in copied_rows
    )
    full_generation = invocation.get("inference_substrate_class") == "model_full_generation"
    offload_real = runtime_receipt.get("offload_real") is True
    measurement_complete = int(
        validation_passed and all_dispositions and full_generation and offload_real
    )
    gates = [
        _gate(
            "required_scoped_and_terminal_checks",
            "validity",
            1,
            int(validation_passed),
            "==",
            "Favorable science cannot excuse invalid evidence.",
        ),
        _gate(
            "owned_cuda_full_generation",
            "validity",
            1,
            int(full_generation and offload_real),
            "==",
            "A model string without an owned CUDA generation is not live evidence.",
        ),
        _gate(
            "authenticated_measured_dispositions",
            "readiness",
            1,
            measurement_complete,
            "==",
            "Censoring is auditable, but a missing scheduled disposition is not.",
        ),
        _gate(
            "complete_episode_support",
            "support",
            10,
            reduced["complete_episode_count"],
            ">=",
            "Fewer than ten complete episodes cannot support an opportunity statement.",
        ),
        _gate(
            "complete_game_support",
            "support",
            6,
            reduced["complete_game_count"],
            ">=",
            "Repeated seeds cannot replace all six natural game strata.",
        ),
        _gate(
            "authenticated_eligibility_rate",
            "support",
            0.9,
            reduced["eligibility_authentication_rate"],
            ">=",
            "Unknown eligibility at more than ten percent of boundaries suppresses support.",
        ),
        _gate(
            "causal_arm_effect_supported",
            "benefit",
            1,
            0,
            "==",
            "Shadow recommendations are not randomized treatments and cannot establish efficacy.",
        ),
    ]
    required_valid = all(
        gate["passed"] for gate in gates if gate["category"] in {"validity", "readiness"}
    )
    if not required_valid:
        verdict_class = "disqualified"
        honest = "complete_disqualified_required_validation_or_live_evidence_failed"
    elif reduced["opportunity_support_score"] == 0:
        verdict_class = "null"
        honest = "complete_null_insufficient_authenticated_opportunity_support"
    elif reduced["eligible_opportunity_count"] == 0:
        verdict_class = "null"
        honest = "complete_null_zero_eligible_supervisor_opportunities"
    else:
        verdict_class = "null"
        honest = "complete_null_opportunities_observed_without_causal_arm_effect"
    sample_budget = {
        "planned": len(schedule),
        "attempted": sum(
            row.get("disposition") not in {"unstarted", "unstarted_collection_cap"}
            for row in copied_rows
        ),
        "completed": dispositions["complete"],
        "excluded": 0,
        "failed": dispositions["failed"] + dispositions["complete_error"],
        "censored": sum(
            count for name, count in dispositions.items() if name.startswith("censored_")
        ),
        "unstarted": dispositions["unstarted"] + dispositions["unstarted_collection_cap"],
        "independent_game_clusters": len(PANEL_GAMES),
    }
    artifact: Json = {
        "schema": SCHEMA,
        "version": 1,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "status": honest,
        "run_date": RUN_DATE,
        "started_at_utc": started_at_utc,
        "ended_at_utc": ended_at_utc,
        "process_identity": {
            "pid": os.getpid(),
            "cwd": str(root.resolve()),
            "source_revision": _git_revision(root),
            "clock_identity": "time.monotonic",
        },
        "preconditions_checked": [deepcopy(dict(row)) for row in preconditions],
        "MODEL_SPECS": deepcopy(list(invocation.get("MODEL_SPECS") or [])),
        "model_specs": [deepcopy(dict(row)) for row in model_specs]
        if invocation.get("model_invoked")
        else [],
        "model_invoked": invocation.get("model_invoked") is True,
        "invocation_counts": deepcopy(dict(invocation.get("invocation_counts") or {})),
        "inference_substrate_class": invocation.get("inference_substrate_class"),
        "inference_substrate": invocation.get("inference_substrate"),
        "execution_venue": "host",
        "execution_venue_detail": {
            "host_cpu_orchestration": True,
            "real_cuda_generation": offload_real,
            "runtime_receipt": deepcopy(dict(runtime_receipt)),
        },
        "duration_s": max(0.000001, float(duration_s)),
        "duration_breakdown_s": {
            "current_work": max(0.000001, float(duration_s)),
            "authoring": 0.0,
            "validation": sum(float(row.get("duration_s") or 0.0) for row in validation_receipts),
            "historical_capture": 0.0,
        },
        "phase_spans": [deepcopy(dict(row)) for row in phase_spans],
        "random_seed": {
            "selection": 658_026,
            "episodes": list(EPISODE_SEEDS),
            "fitting": 658_029,
            "arrival": 658_030,
            "bootstrap": 658_031,
        },
        "reproducibility_checksum": "",
        "source_artifact_hashes": deepcopy(dict(sources)),
        "upstream_receipt": {
            "path": UPSTREAM_PATH.as_posix(),
            "sha256": (sources.get(UPSTREAM_PATH.as_posix()) or {}).get("sha256"),
            "experiment_id": upstream.get("experiment_id"),
            "eligibility_receipt_ready_score": upstream.get("eligibility_receipt_ready_score"),
        },
        "protocol_schedule": [deepcopy(dict(row)) for row in schedule],
        "rows": copied_rows,
        "raw_reduction": reduced,
        "raw_reduction_checksum": canonical_hash(reduced),
        "sample_size_budget": sample_budget,
        "acceptance_gate_results": gates,
        "gate_check_summary": _gate_summary(gates),
        "honest_verdict": honest,
        "verdict_class": verdict_class,
        "verifier_is_oracle": False,
        "flagged_adversarial": False,
        "validation_receipts": [deepcopy(dict(row)) for row in validation_receipts],
        "validation_manifest": {
            "test_paths": [TEST_PATH.as_posix()],
            "changed_modules": [MODULE_PATH.as_posix()],
            "static_paths": [WRAPPER_PATH.as_posix()],
        },
        "arc_measurement_complete_score": measurement_complete,
        "opportunity_support_score": reduced["opportunity_support_score"],
        "per_game_results": _per_game_results(copied_rows),
        "natural_strata_comparison": {
            "design": "descriptive_game_and_observed_interface_strata",
            "randomized_treatment_comparison": False,
            "selected_versus_unselected_effect_estimate": None,
        },
        "supervisor_opportunity_rows": reduced["supervisor_opportunity_rows"],
        "effect_estimate": None,
        "arm_refinement": None,
        "future_causal_trial": reduced["future_causal_trial_support"],
        "future_causal_trial_ready": reduced["future_causal_trial_ready"],
        "solve_provenance": "live_agent_self_discovery",
        "new_level_credit": 0,
        "registry_precheck": deepcopy(dict(registry_receipt)),
        "public_development_generalization_proxy": True,
        "official_hidden_score": False,
    }
    artifact["field_principles"] = {
        key: FIELD_PRINCIPLES.get(
            key, "Retain this typed field so an independent reader can audit the result."
        )
        for key in (*artifact.keys(), "field_principles")
    }
    artifact["reproducibility_checksum"] = _artifact_checksum(artifact)
    return artifact


def build_blocked_artifact(
    root: Path,
    *,
    failed_check: Mapping[str, Any],
    schedule: Sequence[Mapping[str, Any]],
    preconditions: Sequence[Mapping[str, Any]],
    sources: Mapping[str, Any],
    duration_s: float,
    invocation: Mapping[str, Any] | None = None,
    model_specs: Sequence[Mapping[str, Any]] = (),
) -> Json:
    """Close unchanged external absence with exact expected and observed values."""

    current = dict(invocation or reduce_invocations([], child_terminal=True))
    rows = [
        {
            **deepcopy(dict(row)),
            "disposition": "unstarted",
            "action_count": 0,
            "window_exposure_count": 0,
            "explicit_eligibility_count": 0,
            "eligible_count": 0,
            "selected_count": 0,
            "applied_count": 0,
            "supervisor_rows": [],
            "censored_outcome_horizon": "unstarted",
        }
        for row in schedule
    ]
    first_failure = {
        key: failed_check.get(key) for key in ("check", "path", "field", "expected", "observed")
    }
    artifact: Json = {
        "schema": SCHEMA,
        "version": 1,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "status": f"complete_blocked_{failed_check.get('check', 'prerequisite')}",
        "run_date": RUN_DATE,
        "process_identity": {"pid": os.getpid(), "cwd": str(root.resolve())},
        "preconditions_checked": [deepcopy(dict(row)) for row in preconditions],
        "MODEL_SPECS": deepcopy(list(current.get("MODEL_SPECS") or [])),
        "model_specs": [deepcopy(dict(row)) for row in model_specs],
        "model_invoked": current.get("model_invoked") is True,
        "invocation_counts": deepcopy(dict(current.get("invocation_counts") or {})),
        "inference_substrate_class": current.get("inference_substrate_class"),
        "inference_substrate": current.get("inference_substrate"),
        "execution_venue": "host",
        "duration_s": max(0.000001, float(duration_s)),
        "phase_spans": [],
        "random_seed": {"selection": 658_026, "episodes": list(EPISODE_SEEDS)},
        "reproducibility_checksum": "",
        "source_artifact_hashes": deepcopy(dict(sources)),
        "rows": rows,
        "raw_reduction": reduce_panel(rows),
        "sample_size_budget": {
            "planned": len(schedule),
            "attempted": 0,
            "completed": 0,
            "excluded": 0,
            "failed": 0,
            "censored": 0,
            "unstarted": len(schedule),
            "independent_game_clusters": len({row.get("game") for row in schedule}),
        },
        "acceptance_gate_results": [],
        "gate_check_summary": {
            "all_passed": False,
            "failed_count": 1,
            "failures": [first_failure],
            "first_failure": first_failure,
            "required_validity_and_readiness_passed": False,
        },
        "honest_verdict": f"complete_blocked_{failed_check.get('check', 'prerequisite')}",
        "verdict_class": "blocked",
        "verifier_is_oracle": False,
        "flagged_adversarial": False,
        "validation_receipts": [],
        "arc_measurement_complete_score": 0,
        "opportunity_support_score": 0,
        "per_game_results": _per_game_results(rows),
        "supervisor_opportunity_rows": [],
        "solve_provenance": "live_agent_self_discovery",
        "new_level_credit": 0,
        "registry_precheck": {},
    }
    artifact["raw_reduction_checksum"] = canonical_hash(artifact["raw_reduction"])
    artifact["field_principles"] = {
        key: FIELD_PRINCIPLES.get(
            key, "Retain this typed field so an independent reader can audit the blocker."
        )
        for key in (*artifact.keys(), "field_principles")
    }
    artifact["reproducibility_checksum"] = _artifact_checksum(artifact)
    return artifact


def _fixture_invocation() -> Json:
    counts = {
        "model_loads_attempted": 1,
        "model_loads_completed": 1,
        "model_loads_failed": 0,
        "model_loads_cancelled": 0,
        "model_loads_in_flight": 0,
        "forward_calls_attempted": 0,
        "forward_calls_completed": 0,
        "forward_calls_failed": 0,
        "forward_calls_cancelled": 0,
        "forward_calls_in_flight": 0,
        "generation_calls_attempted": 1,
        "generation_calls_completed": 1,
        "generation_calls_failed": 0,
        "generation_calls_cancelled": 0,
        "generation_calls_in_flight": 0,
    }
    return {
        "model_invoked": True,
        "invocation_counts": counts,
        "inference_substrate_class": "model_full_generation",
        "inference_substrate": "live_llm_inference",
        "MODEL_SPECS": list(MODEL_SPECS),
    }


def build_artifact_for_test(
    root: Path,
    upstream: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    *,
    validation_passed: bool = True,
    offload_real: bool = True,
) -> Json:
    """Build a deterministic complete fixture without loading a model or game."""

    schedule = build_schedule(upstream)
    return build_artifact(
        root,
        upstream=upstream,
        schedule=schedule,
        rows=rows,
        invocation=_fixture_invocation(),
        runtime_receipt={
            "offload_real": offload_real,
            "owned_server_vram_mb_after_load": 18_000,
            "quantization": "Q4_K_M",
        },
        model_specs=[
            {
                "repository": MODEL_ID,
                "filename": MODEL_FILENAME,
                "resolved_path": "/fixture/Qwen3.8-27B-Q4_K_M.gguf",
                "sha256": "sha256:fixture",
                "quantization": "Q4_K_M",
                "runtime": "owned_native_cuda_llama_server",
            }
        ],
        preconditions=[{"check": "fixture", "passed": True}],
        sources={},
        validation_receipts=[
            {
                "name": "fixture",
                "required": True,
                "passed": validation_passed,
                "exit_code": 0 if validation_passed else 1,
            }
        ],
        phase_spans=[],
        registry_receipt=deepcopy(dict(upstream.get("registry_precheck") or {})),
        started_at_utc="2026-09-22T00:00:00Z",
        ended_at_utc="2026-09-22T00:01:01Z",
        duration_s=61.0,
    )


def validate_artifact(artifact: Mapping[str, Any], *, require_terminal: bool) -> list[str]:
    """Cold-check identity, row reduction, scores, principles, and checksum."""

    required = {
        "schema",
        "version",
        "experiment_id",
        "milestone",
        "run_date",
        "preconditions_checked",
        "MODEL_SPECS",
        "model_specs",
        "model_invoked",
        "invocation_counts",
        "inference_substrate_class",
        "inference_substrate",
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
        "arc_measurement_complete_score",
        "opportunity_support_score",
        "per_game_results",
        "supervisor_opportunity_rows",
        "solve_provenance",
        "registry_precheck",
    }
    errors = [f"missing_field:{key}" for key in sorted(required - set(artifact))]
    if artifact.get("schema") != SCHEMA:
        errors.append("schema_mismatch")
    if artifact.get("experiment_id") != EXPERIMENT_ID:
        errors.append("experiment_id_mismatch")
    if artifact.get("milestone") != MILESTONE:
        errors.append("milestone_mismatch")
    if type(artifact.get("arc_measurement_complete_score")) is not int or artifact.get(
        "arc_measurement_complete_score"
    ) not in {0, 1}:
        errors.append("measurement_score_not_bare_numeric")
    if type(artifact.get("opportunity_support_score")) is not int or artifact.get(
        "opportunity_support_score"
    ) not in {0, 1}:
        errors.append("opportunity_score_not_bare_numeric")
    rows = artifact.get("rows")
    if not isinstance(rows, list):
        errors.append("rows_not_list")
    else:
        reduced = reduce_panel([row for row in rows if isinstance(row, Mapping)])
        if artifact.get("raw_reduction") != reduced:
            errors.append("independent_reduction_mismatch")
        if artifact.get("raw_reduction_checksum") != canonical_hash(reduced):
            errors.append("raw_reduction_checksum_mismatch")
        if artifact.get("supervisor_opportunity_rows") != reduced["supervisor_opportunity_rows"]:
            errors.append("supervisor_rows_mismatch")
    substrate_class = artifact.get("inference_substrate_class")
    model_invoked = artifact.get("model_invoked") is True
    if substrate_class == "model_full_generation":
        if not model_invoked or artifact.get("inference_substrate") != "live_llm_inference":
            errors.append("full_generation_declaration_mismatch")
        if artifact.get("MODEL_SPECS") != MODEL_SPECS or not artifact.get("model_specs"):
            errors.append("full_generation_model_specs_missing")
        if float(artifact.get("duration_s") or 0.0) < 60.0:
            errors.append("full_generation_duration_below_floor")
    principles = artifact.get("field_principles")
    if not isinstance(principles, Mapping) or any(key not in principles for key in artifact):
        errors.append("field_principles_incomplete")
    if artifact.get("reproducibility_checksum") != _artifact_checksum(artifact):
        errors.append("reproducibility_checksum_mismatch")
    if require_terminal:
        if not str(artifact.get("honest_verdict", "")).startswith("complete_"):
            errors.append("terminal_prefix_missing")
        if artifact.get("verdict_class") not in {
            "positive",
            "circular_positive",
            "null",
            "blocked",
            "disqualified",
            "partial",
        }:
            errors.append("verdict_class_invalid")
    return list(dict.fromkeys(errors))


def independent_reduce(artifact: Mapping[str, Any]) -> Json:
    """Recompute all comparative claims from the episode rows alone."""

    rows = artifact.get("rows")
    typed = (
        [dict(row) for row in rows if isinstance(row, Mapping)] if isinstance(rows, list) else []
    )
    reduced = reduce_panel(typed)
    errors = []
    if artifact.get("raw_reduction") != reduced:
        errors.append("independent_reduction_mismatch")
    if artifact.get("raw_reduction_checksum") != canonical_hash(reduced):
        errors.append("raw_reduction_checksum_mismatch")
    if artifact.get("supervisor_opportunity_rows") != reduced["supervisor_opportunity_rows"]:
        errors.append("supervisor_rows_mismatch")
    return {
        "rows": typed,
        "reduced": reduced,
        "errors": errors,
        "matches_declared": not errors,
    }


def _affected_manifest() -> Any:
    from carnot import experiment_7358_v646_validation_contract as contract

    return contract.AffectedManifest(
        experiment_id=EXPERIMENT_ID,
        test_paths=(TEST_PATH.as_posix(),),
        changed_modules=(MODULE_PATH.as_posix(),),
        static_paths=(WRAPPER_PATH.as_posix(),),
    )


def build_validation_plan(root: Path, private_root: Path) -> list[Any]:
    """Freeze scoped checks and the applicable ARC capability E2Es."""

    from carnot import experiment_7358_v646_validation_contract as contract
    from carnot.reporting import experiment_7303_validation_scope as scope

    private_root.mkdir(parents=True, exist_ok=True)
    commands: list[Any] = list(
        contract.build_command_plan(root, _affected_manifest(), private_root)
    )
    common = ("-n", "0", "-o", "addopts=", "--no-cov")
    for name, test in (
        ("e2e_009", "tests/python/test_arc_induction_state_persistence.py"),
        ("e2e_010", "tests/python/test_arc_tool_grammar_transport.py"),
        ("e2e_011", "tests/python/test_arc_decision_telemetry.py"),
    ):
        parent = private_root / name
        parent.mkdir(parents=True, exist_ok=True)
        commands.append(
            scope.CommandSpec(
                name,
                (
                    ".venv/bin/pytest",
                    *common,
                    f"--basetemp={parent / 'basetemp'}",
                    test,
                    "-q",
                ),
                "capability_e2e",
                900.0,
            )
        )
    smoke = private_root / "private_arc_smoke"
    smoke.mkdir(parents=True, exist_ok=True)
    commands.append(
        contract.EnvironmentCommandSpec(
            "private_arc_smoke",
            (
                ".venv/bin/python",
                "-u",
                "scripts/arc_loop_solve.py",
                "--mechanism",
                "e3",
                "--game",
                "r11l",
                "--max-actions",
                "12",
                "--output",
                str(smoke / "receipt.json"),
            ),
            "private_real_environment_smoke",
            900.0,
            (("CARNOT_ARC_DISABLE_INDUCTION", "1"),),
        )
    )
    return commands


def validate_validation_plan(root: Path, commands: Sequence[Any]) -> list[str]:
    """Reject broad substitutions, duplicate checks, and command drift."""

    from carnot import experiment_7358_v646_validation_contract as contract
    from carnot.reporting import experiment_7303_validation_scope as scope

    affected = [row for row in commands if row.name in scope.REQUIRED_CHECK_NAMES]
    errors = contract.validate_command_plan(root, _affected_manifest(), affected)
    expected = {
        *scope.REQUIRED_CHECK_NAMES,
        "e2e_009",
        "e2e_010",
        "e2e_011",
        "private_arc_smoke",
    }
    counts = {name: sum(row.name == name for row in commands) for name in expected}
    errors.extend(f"command_count:{name}:{count}" for name, count in counts.items() if count != 1)
    return list(dict.fromkeys(errors))


def terminal_command_specs(root: Path, candidate: Path) -> list[Any]:
    """Build fresh-process replay and strict readers for one exact candidate."""

    from carnot.reporting import experiment_7303_validation_scope as scope

    python = str(root / ".venv/bin/python")
    wrapper = str(root / WRAPPER_PATH)
    return [
        scope.CommandSpec(
            "declared_entrypoint_cold_replay",
            (python, "-u", wrapper, "--replay", str(candidate)),
            "exact_terminal_candidate",
            300.0,
        ),
        scope.CommandSpec(
            "independent_cold_reducer",
            (python, "-u", wrapper, "--replay", str(candidate), "--reduce-only"),
            "exact_terminal_candidate",
            300.0,
        ),
        scope.CommandSpec(
            "adversarial_verify",
            (python, "-u", "scripts/adversarial_verify.py", str(candidate)),
            "exact_terminal_candidate",
            300.0,
        ),
        scope.CommandSpec(
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
    indexed = {str(row.get("name")): row for row in receipts}
    return all(
        indexed.get(name, {}).get("passed") is True
        and indexed.get(name, {}).get("exit_code") == 0
        and indexed.get(name, {}).get("timed_out") is not True
        for name in names
    )


def _phase(
    name: str, began: float, started: float, units: int, checkpoint: str | None = None
) -> Json:
    ended = time.monotonic()
    return {
        "phase": name,
        "start_s": began - started,
        "end_s": ended - started,
        "duration_s": ended - began,
        "completed_units": units,
        "checkpoint": checkpoint,
        "ended_at_utc": utc_now(),
    }


def _configure_shared_runner() -> None:
    """Point the qualified shared process and timing helpers at this run."""

    values = {
        "TASK_ID": TASK_ID,
        "RAW_DIR": RAW_DIR,
        "SCHEDULE_PATH": SCHEDULE_PATH,
        "SESSION_PATH": SESSION_PATH,
        "BOUNDARY_PATH": BOUNDARY_PATH,
        "RUNTIME_EVENT_PATH": RUNTIME_EVENT_PATH,
        "ACTION_PATH": ACTION_PATH,
        "CHECKPOINT_PATH": CHECKPOINT_PATH,
        "WRAPPER_PATH": WRAPPER_PATH,
        "RUN_DATE": RUN_DATE,
        "EPISODE_SEEDS": EPISODE_SEEDS,
        "ACTION_LIMIT": ACTION_LIMIT,
        "REQUEST_LIMIT": REQUEST_LIMIT,
        "MAX_NEW_TOKENS": MAX_NEW_TOKENS,
        "EPISODE_LIMIT_S": EPISODE_LIMIT_S,
        "AGGREGATE_LIVE_LIMIT_S": COLLECTION_LIMIT_S,
        "SUPERVISOR_THRESHOLD": SUPERVISOR_WINDOW,
    }
    for name, value in values.items():
        setattr(exp7471, name, value)
    for name, value in (
        ("ACTION_LIMIT", ACTION_LIMIT),
        ("REQUEST_LIMIT", REQUEST_LIMIT),
        ("MAX_NEW_TOKENS", MAX_NEW_TOKENS),
        ("EPISODE_LIMIT_S", EPISODE_LIMIT_S),
    ):
        setattr(exp7491, name, value)


def _unstarted_row(schedule: Mapping[str, Any], disposition: str) -> Json:
    row = exp7491._unstarted_row(schedule, disposition)
    row.update(
        {
            "window_exposure_count": 0,
            "explicit_eligibility_count": 0,
            "eligible_count": 0,
            "selected_count": 0,
            "applied_count": 0,
            "supervisor_rows": [],
            "censored_outcome_horizon": disposition,
        }
    )
    return row


def _attach_eligibility(row: Json, path: Path, root: Path) -> Json:
    """Join exact live eligibility rows without reconstructing missing facts."""

    observations = exp7526._read_jsonl(path)
    exposed = [
        item
        for item in observations
        if isinstance(item.get("action_id"), int) and int(item["action_id"]) >= SUPERVISOR_WINDOW
    ]
    explicit = [
        item
        for item in exposed
        if item.get("observation_status") == "complete"
        and all(
            arm.get("eligible") in {True, False}
            for arm in item.get("arm_rows", [])
            if isinstance(arm, Mapping)
        )
    ]
    supervisor_rows = [
        {
            "action_id": observation.get("action_id"),
            "level_id": observation.get("level_id"),
            "predicate_inputs": deepcopy(observation.get("predicate_inputs")),
            "observation_status": observation.get("observation_status"),
            "application_disposition": observation.get("application_disposition"),
            "old_state_hash": observation.get("old_state_hash"),
            "new_state_hash": observation.get("new_state_hash"),
            "outcome_horizon": (
                "censored_at_episode_end"
                if str(row.get("disposition", "")).startswith("censored_")
                else "observed_to_episode_terminal"
            ),
            **deepcopy(dict(arm)),
        }
        for observation in exposed
        for arm in observation.get("arm_rows", [])
        if isinstance(arm, Mapping)
    ]
    row.update(
        {
            "window_exposure_count": len(exposed),
            "explicit_eligibility_count": len(explicit),
            "eligible_count": sum(item.get("eligible") is True for item in supervisor_rows),
            "selected_count": sum(item.get("selected") is True for item in supervisor_rows),
            "applied_count": sum(item.get("applied") is True for item in supervisor_rows),
            "supervisor_rows": supervisor_rows,
            "eligibility_sidecar": {
                "path": path.relative_to(root).as_posix(),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
                "observation_count": len(observations),
            }
            if path.is_file()
            else None,
            "censored_outcome_horizon": (
                str(row.get("disposition"))
                if str(row.get("disposition", "")).startswith("censored_")
                else "episode_terminal"
            ),
        }
    )
    return row


def _install_generation_progress(proposer: Any, started: float, checkpoint: Path) -> None:
    """Print around each real generation while leaving its inputs and result unchanged."""

    original = proposer.generate
    serial = 0

    def observed(*args: Any, **kwargs: Any) -> Any:
        nonlocal serial
        serial += 1
        current_work_receipt.atomic_json(
            checkpoint,
            {"stage": "generation", "generation_index": serial, "completed_units": 0},
        )
        progress(started, "generation", "before", generation_index=serial)
        try:
            return original(*args, **kwargs)
        finally:
            progress(started, "generation", "after", generation_index=serial)

    proposer.generate = observed


def run_live_session(args: argparse.Namespace) -> int:  # pragma: no cover - owned live child.
    """Load one owned server and run the frozen twelve-episode panel."""

    started = time.monotonic()
    deadline_ns = int(
        os.environ.get(
            "CARNOT_7527_DEADLINE_MONOTONIC_NS",
            str(time.monotonic_ns() + int(COLLECTION_LIMIT_S * 1_000_000_000)),
        )
    )
    root = REPO_ROOT
    raw_dir = Path(args.raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    event_path = raw_dir / RUNTIME_EVENT_PATH.name
    action_path = raw_dir / ACTION_PATH.name
    schedule = load_object(Path(args.schedule_path)).get("rows") or []
    capture = exp7471.live_support.DurableRequestCapture(raw_dir, event_path)
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
        _configure_shared_runner()
        capture.install()
        from carnot.agentic.arc_executable_world_model import LocalGGUFProposer

        progress(started, "model_load", "before", model_path=args.model_path)
        proposer = LocalGGUFProposer(
            repo_substr="Qwen3.8-27B",
            model_path=exp7471.live_support._absolute_model_path(args.model_path),
            port=int(args.port),
            mtp=False,
            kv_quant="q8_0",
            use_chat_template=True,
            n_gpu_layers=999,
            n_ctx=49_152,
            max_tokens=MAX_NEW_TOKENS,
            timeout=int(EPISODE_LIMIT_S),
            tries=1,
        )
        proposer.model_repository = MODEL_ID
        proposer.model_revision = str(args.model_revision)
        proposer.requested_model_filename = MODEL_FILENAME
        proposer.requested_model_path = exp7471.live_support._absolute_model_path(args.model_path)
        if not proposer._ensure_server():
            raise RuntimeError("owned native CUDA llama-server failed to start")
        server_pid = getattr(proposer._proc, "pid", None)
        owned_vram = exp7491._owned_process_vram_mb(server_pid)
        offload_real = (
            isinstance(owned_vram, int) and OFFLOAD_MIN_MB <= owned_vram <= OFFLOAD_MAX_MB
        )
        session["runtime_receipt"] = {
            "child_pid": os.getpid(),
            "server_pid": server_pid,
            "physical_gpu_index": int(args.gpu_index),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "owned_server_vram_mb_after_load": owned_vram,
            "offload_expected_range_mb": [OFFLOAD_MIN_MB, OFFLOAD_MAX_MB],
            "offload_real": offload_real,
            "native_binary": proposer.last_launch_argv[0] if proposer.last_launch_argv else None,
            "server_command": list(proposer.last_launch_argv),
            "requested_n_gpu_layers": 999,
            "resolved_model_path": proposer.requested_model_path,
            "resolved_model_sha256": str(args.model_hash),
            "model_revision": str(args.model_revision),
            "quantization": "Q4_K_M",
            "n_ctx": 49_152,
            "kv_quantization": "q8_0",
            "embedded_tokenizer": True,
            "use_chat_template": True,
            "mtp": False,
            "max_new_tokens": MAX_NEW_TOKENS,
            "request_limit_per_episode": REQUEST_LIMIT,
            "supervisor_mode": "shadow",
            "supervisor_window": SUPERVISOR_WINDOW,
        }
        session["model_loaded"] = True
        current_work_receipt.atomic_json(
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
            offload_real=offload_real,
        )
        if not offload_real:
            raise RuntimeError("owned server did not show the required CUDA offload")
        _install_generation_progress(proposer, started, Path(args.checkpoint_path))

        for index, sealed in enumerate(schedule):
            remaining_s = (deadline_ns - time.monotonic_ns()) / 1_000_000_000
            if remaining_s <= EPISODE_LIMIT_S:
                rows.extend(
                    _unstarted_row(item, "unstarted_collection_cap") for item in schedule[index:]
                )
                break
            episode_id = str(sealed["episode_id"])
            safe_id = episode_id.replace(":", "__")
            eligibility_path = raw_dir / "eligibility" / f"{safe_id}.jsonl"
            eligibility_path.parent.mkdir(parents=True, exist_ok=True)
            eligibility_path.unlink(missing_ok=True)
            os.environ[exp7526.RECORDER_ENV_FLAG] = "1"
            os.environ[exp7526.RECORDER_PATH_ENV] = str(eligibility_path)
            progress(
                started,
                "episode",
                "before_benchmark",
                episode_id=episode_id,
                completed_units=index,
            )
            row = exp7491._run_policy_episode(sealed, proposer, capture, event_path, action_path)
            _attach_eligibility(row, eligibility_path, root)
            rows.append(row)
            current_work_receipt.atomic_json(raw_dir / "episode_rows.json", {"rows": rows})
            current_work_receipt.atomic_json(
                Path(args.checkpoint_path),
                {
                    "stage": "episodes",
                    "model_loaded": True,
                    "completed_units": len(rows),
                    "total_units": len(schedule),
                    "last_episode": episode_id,
                },
            )
            progress(
                started,
                "episode",
                "after_benchmark",
                episode_id=episode_id,
                disposition=row["disposition"],
                completed_units=len(rows),
                window_exposures=row["window_exposure_count"],
            )
        session["model_invoked"] = bool(
            InvocationBoundaryLedger(root / BOUNDARY_PATH).read_events()
        )
    except BaseException as exc:
        session["error"] = f"{type(exc).__name__}: {exc}"[:500]
        progress(started, "live_child", "error", error=session["error"])
    finally:
        os.environ.pop(exp7526.RECORDER_ENV_FLAG, None)
        os.environ.pop(exp7526.RECORDER_PATH_ENV, None)
        capture.restore()
        progress(started, "model_unload", "before")
        if proposer is not None:
            proposer.stop()
        progress(started, "model_unload", "after")
        session["duration_s"] = time.monotonic() - started
        current_work_receipt.atomic_json(Path(args.session_path), session)
        current_work_receipt.atomic_json(
            Path(args.checkpoint_path),
            {
                "stage": "child_terminal",
                "model_loaded": session["model_loaded"],
                "completed_units": len(rows),
                "terminal_child": True,
            },
        )
    return 0


def collect_runtime_preconditions(
    root: Path, started: float
) -> tuple[list[Json], dict[str, Any], Json]:  # pragma: no cover - host resources.
    """Resolve the public environment, pinned model, native server, and idle CUDA."""

    environment_dir = exp7491.resolve_environment_dir(root)
    available = (
        {path.name for path in environment_dir.iterdir() if path.is_dir()}
        if environment_dir is not None
        else set()
    )
    missing_games = sorted(set(PANEL_GAMES) - available)
    checks = [
        _check(
            "public_environment_panel_available",
            str(environment_dir) if environment_dir is not None else "environment_files",
            "missing_games",
            [],
            missing_games,
            not missing_games,
        )
    ]
    progress(started, "runtime_preconditions", "before_model_cuda_probe")
    base_checks, hashes, resources = exp7471._runtime_preconditions(root, started)
    progress(started, "runtime_preconditions", "after_model_cuda_probe")
    checks.extend(_normalize_check_location(row) for row in base_checks)
    resources["environment_dir"] = environment_dir
    model_path = resources.get("model_path")
    model_name = Path(str(model_path)).name if model_path else None
    checks.append(
        _check(
            "pinned_model_filename",
            str(model_path),
            "filename",
            MODEL_FILENAME,
            model_name,
            model_name == MODEL_FILENAME,
        )
    )
    return checks, dict(hashes), dict(resources)


def _retag_failure(artifact: Json, *, verdict: str, verdict_class: str) -> Json:
    artifact["status"] = verdict
    artifact["honest_verdict"] = verdict
    artifact["verdict_class"] = verdict_class
    artifact["arc_measurement_complete_score"] = 0
    artifact["opportunity_support_score"] = 0
    artifact["field_principles"] = {
        key: FIELD_PRINCIPLES.get(
            key, "Retain this typed field so an independent reader can audit the failure."
        )
        for key in (*artifact.keys(), "field_principles")
    }
    artifact["reproducibility_checksum"] = _artifact_checksum(artifact)
    return artifact


def _publish(root: Path, artifact: Mapping[str, Any], started: float) -> None:
    progress(started, "publish", "before_atomic_terminal", path=RESULT_PATH)
    current_work_receipt.atomic_json(root / RESULT_PATH, artifact)
    progress(
        started,
        "publish",
        "after_atomic_terminal",
        verdict=artifact.get("honest_verdict"),
    )


def run_experiment(root: Path, run_date: str) -> Json:  # pragma: no cover - orchestration.
    """Precheck, validate, measure, cold-replay, and publish exactly once."""

    from carnot import experiment_7358_v646_validation_contract as contract
    from carnot.reporting import experiment_7303_validation_scope as scope

    started = time.monotonic()
    started_at = utc_now()
    progress(started, "startup", "flushed_progress")
    if run_date != RUN_DATE:
        raise SystemExit(f"--date must be {RUN_DATE}")
    root = root.resolve()
    spans: list[Json] = []

    began = time.monotonic()
    progress(started, "preconditions", "start")
    preconditions, sources, upstream = collect_preconditions(root)
    spans.append(_phase("preconditions", began, started, len(preconditions)))
    failed = [row for row in preconditions if row.get("passed") is not True]
    progress(started, "preconditions", "end", passed=not failed)
    schedule: list[Json] = []
    if not failed:
        schedule = build_schedule(upstream)
    if failed:
        artifact = build_blocked_artifact(
            root,
            failed_check=failed[0],
            schedule=schedule,
            preconditions=preconditions,
            sources=sources,
            duration_s=time.monotonic() - started,
        )
        _publish(root, artifact, started)
        return artifact

    registry = registry_precheck(root, PANEL_GAMES)
    current_work_receipt.atomic_json(
        root / SCHEDULE_PATH,
        {"rows": schedule, "upstream": UPSTREAM_PATH.as_posix(), "registry": registry},
    )
    progress(started, "protocol", "frozen", units=len(schedule))

    private = Path(tempfile.mkdtemp(prefix="exp7527-validation-", dir="/tmp"))
    plan = build_validation_plan(root, private / "scoped")
    plan_errors = validate_validation_plan(root, plan)
    if plan_errors:
        raise RuntimeError(f"validation_plan_invalid:{plan_errors}")
    progress(started, "validation_manifest", "frozen", commands=len(plan))
    began = time.monotonic()
    progress(started, "validation", "before_subprocesses", commands=len(plan))
    receipts = contract.run_categorized_commands(
        root,
        [contract.PlannedCommand(row, "required_validation", True) for row in plan],
        log_dir=root / RAW_DIR / "validation/scoped",
        heartbeat_s=60.0,
    )
    spans.append(_phase("required_validation", began, started, len(receipts)))
    progress(started, "validation", "after_subprocesses", completed=len(receipts))
    required_names = (
        *scope.REQUIRED_CHECK_NAMES,
        "e2e_009",
        "e2e_010",
        "e2e_011",
        "private_arc_smoke",
    )
    if not _receipts_pass(receipts, required_names):
        failure = _check(
            "required_validation_failed",
            "validation_receipts",
            "required exit codes",
            0,
            [row.get("exit_code") for row in receipts if row.get("required", True)],
            False,
        )
        artifact = build_blocked_artifact(
            root,
            failed_check=failure,
            schedule=schedule,
            preconditions=preconditions,
            sources=sources,
            duration_s=time.monotonic() - started,
        )
        artifact["validation_receipts"] = receipts
        _retag_failure(
            artifact,
            verdict="complete_disqualified_required_validation_failed",
            verdict_class="disqualified",
        )
        _publish(root, artifact, started)
        return artifact

    began = time.monotonic()
    progress(started, "runtime_preconditions", "start")
    runtime_checks, runtime_hashes, resources = collect_runtime_preconditions(root, started)
    preconditions.extend(runtime_checks)
    sources.update(
        {
            str(path): {"path": str(path), **dict(value)}
            for path, value in runtime_hashes.items()
            if isinstance(value, Mapping)
        }
    )
    spans.append(_phase("runtime_preconditions", began, started, len(runtime_checks)))
    failed_runtime = [row for row in runtime_checks if row.get("passed") is not True]
    progress(started, "runtime_preconditions", "end", passed=not failed_runtime)
    if failed_runtime:
        artifact = build_blocked_artifact(
            root,
            failed_check=failed_runtime[0],
            schedule=schedule,
            preconditions=preconditions,
            sources=sources,
            duration_s=time.monotonic() - started,
        )
        artifact["validation_receipts"] = receipts
        artifact["reproducibility_checksum"] = _artifact_checksum(artifact)
        _publish(root, artifact, started)
        return artifact
    if resources.get("gpu") is None or resources.get("environment_dir") is None:
        raise RuntimeError("validated runtime resources disappeared")

    _configure_shared_runner()
    for relative in (
        BOUNDARY_PATH,
        RUNTIME_EVENT_PATH,
        ACTION_PATH,
        SESSION_PATH,
        CHECKPOINT_PATH,
        RAW_DIR / "episode_rows.json",
    ):
        (root / relative).unlink(missing_ok=True)
    os.environ["CARNOT_ARC_PUBLIC_ENV_DIR"] = str(resources["environment_dir"])
    remaining_budget_s = max(0.0, COLLECTION_LIMIT_S - (time.monotonic() - started))
    if remaining_budget_s <= EPISODE_LIMIT_S:
        failure = _check(
            "collection_budget_before_model_load",
            "time.monotonic",
            "remaining_budget_s",
            f">{EPISODE_LIMIT_S}",
            remaining_budget_s,
            False,
        )
        artifact = build_blocked_artifact(
            root,
            failed_check=failure,
            schedule=schedule,
            preconditions=preconditions,
            sources=sources,
            duration_s=time.monotonic() - started,
        )
        _retag_failure(
            artifact,
            verdict="complete_partial_owned_collection_budget_exhausted",
            verdict_class="partial",
        )
        _publish(root, artifact, started)
        return artifact
    exp7471.AGGREGATE_LIVE_LIMIT_S = remaining_budget_s
    os.environ["CARNOT_7527_DEADLINE_MONOTONIC_NS"] = str(
        time.monotonic_ns() + int(remaining_budget_s * 1_000_000_000)
    )
    began = time.monotonic()
    progress(
        started,
        "live_collection",
        "before_model_load_generation_benchmark",
        planned_units=len(schedule),
        gpu_index=resources["gpu"].get("index"),
    )
    session = exp7471.run_child_with_lease(
        resources=resources,
        schedule_path=root / SCHEDULE_PATH,
        started=started,
    )
    observed_rows = [dict(row) for row in session.get("episodes", []) if isinstance(row, Mapping)]
    spans.append(
        _phase(
            "live_collection",
            began,
            started,
            len(observed_rows),
            CHECKPOINT_PATH.as_posix(),
        )
    )
    progress(
        started,
        "live_collection",
        "after_model_load_generation_benchmark",
        observed_units=len(observed_rows),
    )
    boundary_events = InvocationBoundaryLedger(root / BOUNDARY_PATH).read_events()
    invocation = reduce_invocations(boundary_events, child_terminal=True)
    runtime_receipt = dict(session.get("runtime_receipt") or {})
    model_specs = [dict(resources.get("model_spec") or {})]
    model_specs[0].update(
        {
            "repository": MODEL_ID,
            "filename": MODEL_FILENAME,
            "resolved_path": str(resources.get("model_path")),
            "sha256": resources.get("model_hash"),
            "quantization": "Q4_K_M",
            "runtime": "owned_native_cuda_llama_server",
        }
    )
    for relative, role in (
        (SCHEDULE_PATH, "frozen_protocol"),
        (SESSION_PATH, "live_session"),
        (BOUNDARY_PATH, "current_invocation_ledger"),
        (RUNTIME_EVENT_PATH, "request_response_events"),
        (ACTION_PATH, "raw_action_events"),
        (RAW_DIR / "episode_rows.json", "episode_checkpoint"),
    ):
        path = root / relative
        if path.is_file():
            sources[relative.as_posix()] = _source_row(root, relative, role)
    for row in observed_rows:
        for sidecar in (row.get("eligibility_sidecar"),):
            if isinstance(sidecar, Mapping) and isinstance(sidecar.get("path"), str):
                relative = Path(str(sidecar["path"]))
                if (root / relative).is_file():
                    sources[relative.as_posix()] = _source_row(
                        root, relative, "eligibility_measurement"
                    )

    if invocation["inference_substrate_class"] != "model_full_generation":
        failure = _check(
            "current_full_generation_required",
            BOUNDARY_PATH.as_posix(),
            "inference_substrate_class",
            "model_full_generation",
            invocation["inference_substrate_class"],
            False,
        )
        artifact = build_blocked_artifact(
            root,
            failed_check=failure,
            schedule=schedule,
            preconditions=preconditions,
            sources=sources,
            duration_s=time.monotonic() - started,
            invocation=invocation,
            model_specs=model_specs,
        )
        artifact["validation_receipts"] = receipts
        _retag_failure(
            artifact,
            verdict="complete_partial_owned_full_generation_unfinished",
            verdict_class="partial",
        )
        _publish(root, artifact, started)
        return artifact

    observed_by_id = {
        str(row.get("episode_id")): row for row in observed_rows if row.get("episode_id")
    }
    rows = [
        observed_by_id.get(str(sealed["episode_id"]), _unstarted_row(sealed, "unstarted"))
        for sealed in schedule
    ]
    candidate = build_artifact(
        root,
        upstream=upstream,
        schedule=schedule,
        rows=rows,
        invocation=invocation,
        runtime_receipt=runtime_receipt,
        model_specs=model_specs,
        preconditions=preconditions,
        sources=sources,
        validation_receipts=receipts,
        phase_spans=spans,
        registry_receipt=registry,
        started_at_utc=started_at,
        ended_at_utc=utc_now(),
        duration_s=time.monotonic() - started,
    )
    errors = validate_artifact(candidate, require_terminal=False)
    if errors:
        raise RuntimeError(f"candidate_invalid:{errors}")
    progress(started, "candidate", "before_atomic_write", path=CANDIDATE_PATH)
    current_work_receipt.atomic_json(root / CANDIDATE_PATH, candidate)
    progress(started, "candidate", "after_atomic_write", path=CANDIDATE_PATH)

    began = time.monotonic()
    terminal_specs = terminal_command_specs(root, root / CANDIDATE_PATH)
    progress(
        started,
        "terminal_validation",
        "before_subprocesses",
        commands=len(terminal_specs),
    )
    terminal_receipts = contract.run_categorized_commands(
        root,
        [contract.PlannedCommand(row, "terminal_reader", True) for row in terminal_specs],
        log_dir=root / RAW_DIR / "validation/terminal",
        heartbeat_s=60.0,
    )
    spans.append(_phase("terminal_validation", began, started, len(terminal_receipts)))
    receipts.extend(terminal_receipts)
    progress(
        started,
        "terminal_validation",
        "after_subprocesses",
        completed=len(terminal_receipts),
    )

    final = build_artifact(
        root,
        upstream=upstream,
        schedule=schedule,
        rows=rows,
        invocation=invocation,
        runtime_receipt=runtime_receipt,
        model_specs=model_specs,
        preconditions=preconditions,
        sources=sources,
        validation_receipts=receipts,
        phase_spans=spans,
        registry_receipt=registry,
        started_at_utc=started_at,
        ended_at_utc=utc_now(),
        duration_s=time.monotonic() - started,
    )
    terminal_names = (
        "declared_entrypoint_cold_replay",
        "independent_cold_reducer",
        "adversarial_verify",
        "verdict_row_consistency_strict",
    )
    if not _receipts_pass(receipts, terminal_names):
        final["flagged_adversarial"] = any(
            row.get("name") == "adversarial_verify" and row.get("passed") is not True
            for row in terminal_receipts
        )
        _retag_failure(
            final,
            verdict="complete_disqualified_required_terminal_validation_failed",
            verdict_class="disqualified",
        )
    errors = validate_artifact(final, require_terminal=True)
    if errors:
        raise RuntimeError(f"terminal_artifact_invalid:{errors}")
    _publish(root, final, started)
    return final


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse producer, owned-child, or read-only replay roles."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date")
    parser.add_argument("--role", choices=("experiment", "live-session"), default="experiment")
    parser.add_argument("--replay", type=Path)
    parser.add_argument("--reduce-only", action="store_true")
    parser.add_argument("--model-path")
    parser.add_argument("--model-hash")
    parser.add_argument("--model-revision", default="unknown")
    parser.add_argument("--gpu-index", type=int)
    parser.add_argument("--port", type=int)
    parser.add_argument("--schedule-path", type=Path)
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--checkpoint-path", type=Path, default=CHECKPOINT_PATH)
    parser.add_argument("--session-path", type=Path, default=SESSION_PATH)
    args = parser.parse_args(argv)
    if args.replay is None and args.date is None:
        parser.error("--date is required unless --replay is used")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    """Run the producer or independently inspect one exact candidate."""

    args = parse_args(argv)
    if args.replay is not None:
        artifact = load_object(args.replay)
        replay = independent_reduce(artifact)
        errors = (
            replay["errors"]
            if args.reduce_only
            else validate_artifact(artifact, require_terminal=False)
        )
        print(
            json.dumps(
                {
                    "matches_declared": replay["matches_declared"],
                    "row_count": len(replay["rows"]),
                    "reduction_errors": replay["errors"],
                    "validation_errors": errors,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return int(bool(errors))
    if args.role == "live-session":
        _configure_shared_runner()
        return run_live_session(args)
    artifact = run_experiment(REPO_ROOT, str(args.date))
    return int(
        not str(artifact.get("honest_verdict", "")).startswith(
            ("complete_", "success_", "passed_", "shipped_")
        )
    )


__all__ = [
    "EPISODE_SEEDS",
    "MODEL_ID",
    "PANEL_GAMES",
    "REPO_ROOT",
    "RESULT_PATH",
    "UPSTREAM_PATH",
    "WITHHELD_INPUTS",
    "authenticate_upstream_fields",
    "build_artifact_for_test",
    "build_blocked_artifact",
    "build_schedule",
    "build_validation_plan",
    "independent_reduce",
    "main",
    "reduce_invocations",
    "reduce_panel",
    "validate_artifact",
    "validate_validation_plan",
]
