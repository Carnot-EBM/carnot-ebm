"""Capture V655 ARC cost Panel A through the qualified live E3 driver.

This module keeps orchestration thin by reusing the Experiment 7471 live
driver and the Experiment 7478 interval reducer. Its small reducer adds the
Panel A identities and reports costs without treating model work as removable.

Spec refs: REQ-ARC-WMTE-7485 and SCENARIO-ARC-WMTE-7485-*.
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
import tempfile
import time
from typing import Any

import yaml

from carnot import experiment_7358_v646_validation_contract as validation_contract
from carnot import experiment_7471_v654_arc_seam_observation as live_driver
from carnot import experiment_7478_v655_arc_interval_protocol as interval_protocol
from carnot.agentic.arc_inference_boundary import InvocationBoundaryLedger
from carnot.reporting import current_work_receipt
from carnot.reporting import experiment_7303_validation_scope as validation_scope


JsonDict = dict[str, Any]
REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_DATE = "20260921"
MILESTONE = "2026.09.655"
EXPERIMENT_ID = "exp7485-v655-arc-cost-panel-a"
TASK_ID = "experiment_7485_v655_arc_cost_panel_a"
SCHEMA = "carnot.exp7485.v655.arc_cost_panel_a.v1"

MODEL_ID = "unsloth/Qwen3.8-27B-GGUF"
MODEL_SPECS = [MODEL_ID]
INFERENCE_SUBSTRATE_CLASS = "model_bounded_generation"
EXECUTION_VENUE = "host"
PANEL_GAMES = ("sk48", "tr87", "s5i5", "lp85", "lf52", "cn04")
EPISODE_SEEDS = (65_501, 65_502, 65_503)
ACTION_LIMIT = 180
REQUEST_LIMIT = 2
MAX_NEW_TOKENS = 256
EPISODE_LIMIT_S = 240.0
PANEL_LIVE_LIMIT_S = 3600.0

SPEC_PATH = Path("openspec/capabilities/arc-world-model-trust-energy/spec.md")
UPSTREAM_PATH = Path("results/experiment_7478_v655_arc_interval_protocol.json")
REGISTRY_PATH = Path("ops/arc_solve_registry.yaml")
RESULT_PATH = Path("results/experiment_7485_v655_arc_cost_panel_a.json")
RAW_DIR = Path("results/raw/experiment_7485_v655_arc_cost_panel_a")
SCHEDULE_PATH = RAW_DIR / "frozen_schedule.json"
SESSION_PATH = RAW_DIR / "live_session.json"
BOUNDARY_PATH = RAW_DIR / "current_invocation_events.jsonl"
RUNTIME_EVENT_PATH = RAW_DIR / "runtime_events.jsonl"
ACTION_PATH = RAW_DIR / "live_action_rows.jsonl"
CALLBACK_SEAM_PATH = RAW_DIR / "callback_seam_events.jsonl"
NORMALIZED_DIR = RAW_DIR / "normalized_intervals"
CHECKPOINT_PATH = Path("results/checkpoints/experiment_7485_v655_arc_cost_panel_a.json")
CANDIDATE_PATH = RAW_DIR / "measured_terminal_candidate.json"
MODULE_PATH = Path("python/carnot/experiment_7485_v655_arc_cost_panel_a.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_7485_v655_arc_cost_panel_a.py")
TEST_PATH = Path("tests/python/test_experiment_7485_v655_arc_cost_panel_a.py")

REQUIRED_TERMINAL_NAMES = (
    "declared_entrypoint_cold_replay",
    "independent_cold_reducer",
    "adversarial_verify",
    "verdict_row_consistency_strict",
)
CAPABILITY_E2E_NAMES = ("e2e_009", "e2e_010", "e2e_011", "private_arc_smoke")
TERMINAL_DISPOSITIONS = {
    "complete",
    "complete_error",
    "failed",
    "censored_timeout",
    "censored_aggregate_limit",
    "censored_no_first_action",
    "unavailable",
    "unstarted",
}
WORK_CLASS_BY_SEAM = {
    "candidate_action_selection": "unattributed_decision_seam",
    "hypothesis_gate": "world_model_construction",
    "supervisor_arm_selection": "replaceable_decision",
    "induction_timing": "replaceable_decision",
    "downstream_generation": "text_generation",
}

REQUIRED_ARTIFACT_FIELDS = (
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
    "arc_panel_a_complete_score",
    "per_game_results",
    "solve_provenance",
    "offline_reproduced",
    "reproduced_levels",
    "exclusive_cost_rows",
    "generalization_scope",
)

FIELD_PRINCIPLES = {
    "schema": "Versioned schema with exact roadmap experiment_id, milestone and terminal status prevents silent reader drift.",
    "run_date": "Use 20260921; retain measured UTC and monotonic times with clock/process identity.",
    "preconditions_checked": "Name resources, exact paths, ownership and observed prerequisite values before dependent work.",
    "MODEL_SPECS": "Use unsloth/Qwen3.8-27B-GGUF for current model tasks and never substitute a smaller headline model.",
    "model_specs": "Mirror the authenticated current model identity for lowercase readers.",
    "model_invoked": "Any attempted current model call differs from archived or scripted events.",
    "invocation_counts": "Balance attempted, complete, failed, cancelled and in-flight loads, forwards and generations.",
    "inference_substrate": "Name the actual native readout or bounded generation path used by current work.",
    "inference_substrate_class": "Use model_bounded_generation after a generation attempt and record the actual earlier stopping class otherwise.",
    "execution_venue": "Use host and record actual CPU/CUDA identities; historical board evidence is separate.",
    "duration_s": "Measure current work without padding; separate load, generation, episode and validation time.",
    "phase_spans": "Timestamped flushed progress and checkpoints expose long silent or unfinished operations.",
    "random_seed": "Freeze ordering, fitting, audit and bootstrap seeds; deterministic reducers explain any null seed.",
    "reproducibility_checksum": "Bind code, protocol, data roles, model identity, raw shards and validation scope.",
    "source_artifact_hashes": "Preserve exact upstream bytes and their original flags and classes.",
    "rows": "One row per game and seed keeps failures, censoring and unstarted units visible.",
    "sample_size_budget": "Separate planned, attempted, complete, failed, censored, excluded and unstarted independent units.",
    "acceptance_gate_results": "Each check carries category, expected, observed, operator, result and its failure-prevention principle.",
    "gate_check_summary": "Every blocked verdict names failed check, upstream, exact field or path, expected and observed value.",
    "honest_verdict": "Use complete terminal findings and preserve a blocked source verdict when it is the actual result.",
    "verdict_class": "Use the closed terminal enum and reserve partial for retryable owned work.",
    "verifier_is_oracle": "Declare that the public environment is the evaluation oracle, which forbids a positive verdict.",
    "flagged_adversarial": "Keep real reader flags and never clear a flag to open a gate.",
    "validation_receipts": "Exact commands, exits, log hashes and required status establish validation scope.",
    "field_principles": "Echo why each field and gate exists so evidence is understandable independently.",
    "arc_panel_a_complete_score": "Bare 0/1 requires all eighteen dispositions and valid current instrument receipts.",
    "per_game_results": "Game and seed rows retain progress, costs, failures and censoring.",
    "solve_provenance": "Use live_agent_self_discovery exactly; fresh trace reproduction is recorded separately.",
    "offline_reproduced": "True only when a current live-agent trace replays in a fresh environment.",
    "reproduced_levels": "Current reproduced progress does not add duplicate public registry credit.",
    "exclusive_cost_rows": "Episode-scoped nonoverlapping intervals bound removable work honestly.",
    "generalization_scope": "Public adapter-withheld proxy; hidden-game efficacy is unmeasured.",
}

MANIFEST = validation_contract.AffectedManifest(
    experiment_id=EXPERIMENT_ID,
    test_paths=(TEST_PATH.as_posix(),),
    changed_modules=(MODULE_PATH.as_posix(),),
    static_paths=(WRAPPER_PATH.as_posix(),),
)

INPUT_PATHS = (
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
    UPSTREAM_PATH,
    Path("python/carnot/experiment_7478_v655_arc_interval_protocol.py"),
    Path("python/carnot/experiment_7471_v654_arc_seam_observation.py"),
    Path("python/carnot/agentic/arc_competition_agent.py"),
    Path("python/carnot/agentic/arc_decision_telemetry.py"),
    REGISTRY_PATH,
    SPEC_PATH,
    MODULE_PATH,
    WRAPPER_PATH,
    TEST_PATH,
)


def utc_now() -> str:
    """Return one UTC timestamp for a durable experiment boundary."""

    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def progress(started: float, phase: str, event: str, **details: Any) -> None:  # pragma: no cover
    """Flush one phase or long-operation boundary to the conductor."""

    suffix = " ".join(f"{key}={value}" for key, value in sorted(details.items()))
    print(
        f"[exp7485] phase={phase} event={event} elapsed_s={time.monotonic() - started:.3f}"
        + (f" {suffix}" if suffix else ""),
        flush=True,
    )


def canonical_hash(value: Any) -> str:
    """Hash stable JSON bytes so independent replay detects any changed row."""

    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def artifact_checksum(value: Mapping[str, Any]) -> str:
    """Bind every terminal field except the checksum that stores the result."""

    copied = deepcopy(dict(value))
    copied["reproducibility_checksum"] = ""
    return canonical_hash(copied)


def load_object(path: Path) -> JsonDict:
    """Read one JSON object and fail closed for missing or malformed bytes."""

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(value) if isinstance(value, Mapping) else {}


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
) -> JsonDict:
    """Attach one failure-prevention principle to an exact comparison."""

    passed = observed == expected if op == "==" else observed >= expected if op == ">=" else False
    return {
        "check": check,
        "category": category,
        "expected": deepcopy(expected),
        "observed": deepcopy(observed),
        "op": op,
        "passed": bool(passed),
        "upstream": upstream,
        "path": upstream,
        "field": field,
        "artifact_field": field,
        "principle": principle,
    }


def _gate_summary(gates: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Expose every failed gate while separating required and benefit checks."""

    failures = [deepcopy(dict(row)) for row in gates if row.get("passed") is not True]
    required = [row for row in failures if row.get("category") in {"validity", "readiness"}]
    first = failures[0] if failures else None
    return {
        "all_passed": not failures,
        "failed_count": len(failures),
        "required_failed_count": len(required),
        "required_validity_and_readiness_passed": not required,
        "failed_checks": failures,
        "first_failed_check": first.get("check") if first else None,
        "upstream": first.get("upstream") if first else None,
        "exact_field_path": first.get("artifact_field") if first else None,
        "expected_value": first.get("expected") if first else None,
        "observed_value": first.get("observed") if first else None,
    }


def _source_record(path: Path, role: str, flags: Mapping[str, Any] | None = None) -> JsonDict:
    """Bind an input's exact bytes and preserve upstream flags separately."""

    row: JsonDict = {
        "path": path.as_posix(),
        "role": role,
        "bytes": path.stat().st_size,
        "sha256": current_work_receipt.sha256_file(path),
    }
    if flags is not None:
        row["original_flags"] = deepcopy(dict(flags))
    return row


def build_panel_schedule() -> list[JsonDict]:
    """Select Panel A directly from the outcome-blind Experiment 7478 seal."""

    manifest = interval_protocol.build_arc_schedule_manifest()
    rows = [deepcopy(row) for row in manifest["rows"] if row.get("panel") == "A"]
    return [
        {
            **row,
            "execution_order": index,
            "adapter_disabled": True,
            "stored_engines_disabled": True,
            "banked_trajectories_disabled": True,
            "cross_game_state_disabled": True,
            "game_source_disabled": True,
            "offline_ground_truth_bfs_disabled": True,
        }
        for index, row in enumerate(rows)
    ]


def collect_preconditions(
    root: Path, *, force_live: str | None = None
) -> tuple[list[JsonDict], JsonDict, JsonDict, JsonDict]:
    """Authenticate source bytes, protocol flags, games and prior solve credit."""

    checks: list[JsonDict] = []
    hashes: JsonDict = {}
    for relative in INPUT_PATHS:
        path = root / relative
        available = path.is_file() and path.stat().st_size > 0
        checks.append(
            _gate(
                f"source_bytes:{relative.as_posix()}",
                "validity",
                True,
                available,
                upstream=relative.as_posix(),
                field="bytes",
                principle="Missing source bytes would make current evidence non-replayable.",
            )
        )
        if available:
            hashes[relative.as_posix()] = _source_record(path, "current_input")

    upstream = load_object(root / UPSTREAM_PATH)
    flags = {
        key: upstream.get(key)
        for key in (
            "status",
            "honest_verdict",
            "verdict_class",
            "flagged_adversarial",
            "arc_interval_protocol_ready_score",
        )
    }
    if (root / UPSTREAM_PATH).is_file():
        hashes[UPSTREAM_PATH.as_posix()] = _source_record(
            root / UPSTREAM_PATH, "structured_prerequisite", flags
        )
    for field, expected in (
        ("arc_interval_protocol_ready_score", 1),
        ("verdict_class", "null"),
        ("flagged_adversarial", False),
    ):
        checks.append(
            _gate(
                f"exp7478.{field}",
                "validity",
                expected,
                upstream.get(field),
                upstream=UPSTREAM_PATH.as_posix(),
                field=field,
                principle="Panel capture requires the qualified unflagged interval instrument.",
            )
        )
    checks.append(
        _gate(
            "force_live",
            "validity",
            "1",
            os.environ.get("CARNOT_FORCE_LIVE") if force_live is None else force_live,
            upstream="environment",
            field="CARNOT_FORCE_LIVE",
            principle="A live panel cannot silently substitute scripted or simulated events.",
        )
    )
    spec = (root / SPEC_PATH).read_text(encoding="utf-8") if (root / SPEC_PATH).is_file() else ""
    checks.append(
        _gate(
            "driving_requirement",
            "validity",
            True,
            "REQ-ARC-WMTE-7485" in spec,
            upstream=SPEC_PATH.as_posix(),
            field="REQ-ARC-WMTE-7485",
            principle="Implementation without its requirement would bypass spec-first review.",
        )
    )
    exclusion = (root / "ops/exclusion_manifest.yaml").read_text(encoding="utf-8")
    checks.append(
        _gate(
            "task_not_excluded",
            "validity",
            False,
            "7485" in exclusion or EXPERIMENT_ID in exclusion,
            upstream="ops/exclusion_manifest.yaml",
            field=EXPERIMENT_ID,
            principle="A quarantined scope must not publish fresh scientific evidence.",
        )
    )
    available = set(live_driver.accessible_games(root))
    missing_games = [game for game in PANEL_GAMES if game not in available]
    checks.append(
        _gate(
            "exact_panel_games_available",
            "validity",
            [],
            missing_games,
            upstream="environment_files",
            field="panel_a_games",
            principle="An unavailable game remains explicit and cannot be outcome-replaced.",
        )
    )
    registry_value = yaml.safe_load((root / REGISTRY_PATH).read_text(encoding="utf-8")) or {}
    indexed = {
        str(row.get("game")): row
        for row in registry_value.get("games", [])
        if isinstance(row, Mapping) and row.get("game")
    }
    registry = {
        "rows": [
            {
                "game": game,
                "registered": game in indexed,
                "levels_reproduced": indexed.get(game, {}).get("levels_reproduced"),
                "full_game_clear": indexed.get(game, {}).get("full_game_clear"),
            }
            for game in PANEL_GAMES
        ],
        "policy_received_registry_data": False,
        "new_credit_allowed": False,
    }
    return checks, hashes, upstream, registry


def normalize_interval_events(
    events: Sequence[Mapping[str, Any]], *, owner_pid: int
) -> list[JsonDict]:
    """Add the qualified identity and work class without changing raw choices."""

    normalized: list[JsonDict] = []
    for index, source in enumerate(events):
        row = deepcopy(dict(source))
        seam = str(row.get("seam") or "unattributed")
        row.setdefault("schema", "carnot.arc.e3_decision_interval.v2")
        row.setdefault("run_id", EXPERIMENT_ID)
        row.setdefault("process_id", owner_pid)
        row.setdefault("decision_id", f"{row.get('episode_id')}:{seam}:unidentified-{index}")
        row.setdefault("parent_decision_id", None)
        row.setdefault("clock_identity", "time.monotonic_ns")
        row.setdefault("work_class", WORK_CLASS_BY_SEAM.get(seam, "unattributed_decision_seam"))
        if row.get("event") in {"stage_end", "stage_terminal"}:
            row.setdefault("disposition", "completed")
        normalized.append(row)
    return normalized


def _events_from_shards(
    shards: Sequence[Mapping[str, Any]], root: Path = REPO_ROOT
) -> list[JsonDict]:
    """Read hash-bound JSONL shards or explicit in-test rows."""

    rows: list[JsonDict] = []
    for shard in shards:
        inline = shard.get("inline_rows")
        if isinstance(inline, list):
            rows.extend(dict(row) for row in inline if isinstance(row, Mapping))
            continue
        raw = shard.get("path")
        if not isinstance(raw, str):
            continue
        path = Path(raw)
        path = path if path.is_absolute() else root / path
        rows.extend(live_driver.read_jsonl(path))
    return rows


def _actions_to_progress(row: Mapping[str, Any]) -> int | None:
    existing = row.get("actions_to_progress")
    if isinstance(existing, int) and not isinstance(existing, bool):
        return existing
    start = row.get("start_level")
    if not isinstance(start, int):
        return None
    for action in row.get("action_rows") or []:
        if isinstance(action, Mapping) and int(action.get("level") or 0) > start:
            return int(action.get("action_index") or 0)
    return None


def _episode_bounds(row: Mapping[str, Any], events: Sequence[Mapping[str, Any]]) -> tuple[int, int]:
    explicit_start = row.get("episode_start_ns")
    explicit_end = row.get("episode_end_ns")
    if (
        isinstance(explicit_start, int)
        and not isinstance(explicit_start, bool)
        and isinstance(explicit_end, int)
        and not isinstance(explicit_end, bool)
        and explicit_end >= explicit_start
    ):
        return explicit_start, explicit_end
    action_rows = [item for item in row.get("action_rows") or [] if isinstance(item, Mapping)]
    starts = [
        int(item["interval_start_monotonic_ns"])
        for item in action_rows
        if isinstance(item.get("interval_start_monotonic_ns"), int)
    ]
    ends = [
        int(item["interval_end_monotonic_ns"])
        for item in action_rows
        if isinstance(item.get("interval_end_monotonic_ns"), int)
    ]
    for event in events:
        for field, target in (
            ("interval_start_monotonic_ns", starts),
            ("interval_end_monotonic_ns", ends),
            ("event_monotonic_ns", ends),
        ):
            value = event.get(field)
            if isinstance(value, int) and not isinstance(value, bool):
                target.append(value)
    start = min(starts or ends or [0])
    measured_end = max(ends or starts or [start])
    elapsed_end = start + max(0, int(float(row.get("elapsed_s") or 0.0) * 1_000_000_000))
    return start, max(start, measured_end, elapsed_end)


def _unstarted_row(sealed: Mapping[str, Any]) -> JsonDict:
    return {
        **deepcopy(dict(sealed)),
        "disposition": "unstarted",
        "action_count": 0,
        "start_level": None,
        "peak_level": None,
        "terminal_level": None,
        "action_rows": [],
        "request_budget_receipt": None,
        "trace_reproduction": {"attempted": False, "passed": False},
        "solve_provenance": "unstarted",
        "elapsed_s": 0.0,
        "new_level_credit": 0,
        "error": None,
    }


def reduce_panel(
    schedule: Sequence[Mapping[str, Any]],
    episode_rows: Sequence[Mapping[str, Any]],
    seam_events: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Reduce eighteen sealed units through the exclusive interval protocol."""

    observed = {
        str(row.get("episode_id")): dict(row)
        for row in episode_rows
        if isinstance(row, Mapping) and row.get("episode_id")
    }
    rows: list[JsonDict] = []
    exclusive: list[JsonDict] = []
    for sealed in schedule:
        episode_id = str(sealed["episode_id"])
        row = (
            {**deepcopy(dict(sealed)), **deepcopy(observed[episode_id])}
            if episode_id in observed
            else _unstarted_row(sealed)
        )
        events = [
            dict(event) for event in seam_events if str(event.get("episode_id")) == episode_id
        ]
        start, end = _episode_bounds(row, events)
        cost = interval_protocol.reduce_episode_intervals(
            events, episode_id=episode_id, episode_start_ns=start, episode_end_ns=end
        )
        for stage in cost["stage_rows"]:
            stage["replaceable"] = (
                stage.get("work_class") in interval_protocol.REPLACEABLE_WORK_CLASSES
            )
            stage["semantic_replacement_valid"] = False
        actions_to_progress = _actions_to_progress(row)
        reproduction = row.get("trace_reproduction") or {}
        progressed = actions_to_progress is not None
        reproduced = progressed and reproduction.get("passed") is True
        selections = [event for event in events if event.get("event") == "selection"]
        selected = [
            str(item) for event in selections for item in event.get("selected_candidate_ids") or []
        ]
        request_tokens = {
            "input": sum(int(event.get("input_tokens") or 0) for event in events),
            "output": sum(int(event.get("output_tokens") or 0) for event in events),
        }
        failed_calls = sum(
            event.get("seam") == "downstream_generation" and event.get("disposition") == "failed"
            for event in events
        )
        row.update(
            {
                "actions_to_progress": actions_to_progress,
                "actions_to_progress_censored": actions_to_progress is None,
                "offline_reproduced": bool(reproduced),
                "reproduced_levels": int(row.get("peak_level") or 0) if reproduced else 0,
                "solve_provenance": (
                    "live_agent_self_discovery"
                    if reproduced
                    else "unreproduced_live_progress"
                    if progressed
                    else str(row.get("solve_provenance") or "no_level_reached")
                ),
                "new_level_credit": 0,
                "exclusive_cost": cost,
                "request_tokens": request_tokens,
                "failed_generation_calls": failed_calls,
                "induction_completion": any(
                    event.get("seam") == "hypothesis_gate"
                    and event.get("event") in {"stage_end", "stage_terminal"}
                    and event.get("disposition") == "completed"
                    for event in events
                ),
                "verifier_feedback_exposed": any(
                    event.get("seam") == "hypothesis_gate" and event.get("event") == "gate_result"
                    for event in events
                ),
                "supervision_opportunity": any(
                    event.get("seam") == "supervisor_arm_selection"
                    and event.get("event") == "stage_start"
                    for event in events
                ),
                "no_op_decisions": sum(
                    item in {"no_redirect", "defer_induction", "skip_induction"}
                    for item in selected
                ),
                "delegate_decisions": sum(
                    item not in {"no_redirect", "defer_induction", "skip_induction"}
                    for item in selected
                ),
            }
        )
        rows.append(row)
        exclusive.append(
            {
                "episode_id": episode_id,
                "game": row.get("game"),
                "seed": row.get("seed"),
                **deepcopy(cost),
            }
        )

    dispositions = Counter(str(row.get("disposition")) for row in rows)
    censored = sum(count for name, count in dispositions.items() if name.startswith("censored_"))
    failed = dispositions["failed"] + dispositions["complete_error"]
    excluded = dispositions["unavailable"]
    unstarted = dispositions["unstarted"]
    complete = dispositions["complete"]
    per_game: list[JsonDict] = []
    for game in PANEL_GAMES:
        game_rows = [row for row in rows if row.get("game") == game]
        per_game.append(
            {
                "game": game,
                "seed_rows": [
                    {
                        key: row.get(key)
                        for key in (
                            "seed",
                            "episode_id",
                            "disposition",
                            "action_count",
                            "actions_to_progress",
                            "actions_to_progress_censored",
                            "offline_reproduced",
                            "reproduced_levels",
                            "request_tokens",
                            "failed_generation_calls",
                            "induction_completion",
                            "verifier_feedback_exposed",
                            "supervision_opportunity",
                            "no_op_decisions",
                            "delegate_decisions",
                            "exclusive_cost",
                        )
                    }
                    for row in game_rows
                ],
                "completed_seeds": sum(row.get("disposition") == "complete" for row in game_rows),
                "any_reproduced_progress": any(
                    row.get("offline_reproduced") is True for row in game_rows
                ),
                "right_censored": any(row.get("disposition") != "complete" for row in game_rows),
            }
        )
    all_dispositions = len(rows) == len(schedule) == 18 and all(
        row.get("disposition") in TERMINAL_DISPOSITIONS for row in rows
    )
    return {
        "rows": rows,
        "per_game_results": per_game,
        "exclusive_cost_rows": exclusive,
        "sample_size_budget": {
            "planned_independent_units": len(schedule),
            "attempted_independent_units": len(rows) - unstarted,
            "complete_independent_units": complete,
            "failed_independent_units": failed,
            "censored_independent_units": censored,
            "excluded_independent_units": excluded,
            "unstarted_independent_units": unstarted,
            "independent_game_clusters": len(per_game),
        },
        "all_dispositions_present": all_dispositions,
        "all_interval_bounds_valid": all(row["bounds_valid"] for row in exclusive),
    }


def compact_reduction(reduced: Mapping[str, Any]) -> JsonDict:
    """Replace repeated large action and stage lists with hash-bound summaries."""

    compacted = deepcopy(dict(reduced))
    rows = compacted.get("rows") or []
    for row in rows:
        if "action_rows" in row:
            actions = row.pop("action_rows")
            row["action_outcome_count"] = len(actions)
            row["action_outcomes_sha256"] = canonical_hash(actions)
        cost = row.get("exclusive_cost") or {}
        stages = cost.pop("stage_rows", [])
        cost["stage_row_count"] = len(stages)
        cost["stage_rows_sha256"] = canonical_hash(stages)
        row["episode_start_ns"] = cost.get("episode_start_ns")
        row["episode_end_ns"] = cost.get("episode_end_ns")
    for cost in compacted.get("exclusive_cost_rows") or []:
        stages = cost.pop("stage_rows", [])
        cost["stage_row_count"] = len(stages)
        cost["stage_rows_sha256"] = canonical_hash(stages)
    row_by_id = {str(row.get("episode_id")): row for row in rows}
    for game in compacted.get("per_game_results") or []:
        for seed_row in game.get("seed_rows") or []:
            source = row_by_id.get(str(seed_row.get("episode_id")), {})
            seed_row["exclusive_cost"] = deepcopy(source.get("exclusive_cost") or {})
    return compacted


def _invocation_reduction(
    boundary_events: Sequence[Mapping[str, Any]], *, child_terminal: bool
) -> JsonDict:
    """Reduce only current boundary events and retain the actual stopping class."""

    current = live_driver.live_support.reduce_current_invocations(
        boundary_events, child_terminal=child_terminal
    )
    counts = deepcopy(current_work_receipt.ZERO_INVOCATION_COUNTS)
    counts.update(current["invocation_counts"])
    for state in ("attempted", "completed", "failed", "cancelled", "in_flight"):
        counts[f"forward_calls_{state}"] = 0
    generated = counts["generation_calls_attempted"] > 0
    loaded = counts["model_loads_attempted"] > 0
    return {
        **current,
        "invocation_counts": counts,
        "inference_substrate_class": (
            "model_bounded_generation"
            if generated
            else "model_load_no_generation"
            if loaded
            else "no_model_load"
        ),
        "inference_substrate": (
            "owned_native_cuda_llama_cpp_qwen3.8_27b_gguf"
            if generated
            else "owned_native_cuda_model_load_only"
            if loaded
            else "pre_model_validation_only"
        ),
    }


def _receipts_pass(receipts: Sequence[Mapping[str, Any]], names: Sequence[str]) -> bool:
    by_name = {str(row.get("name")): row for row in receipts}
    return all(
        name in by_name
        and by_name[name].get("passed") is True
        and by_name[name].get("exit_code") == 0
        and by_name[name].get("timed_out") is not True
        for name in names
    )


def build_terminal_artifact(
    *,
    started_at_utc: str,
    ended_at_utc: str,
    duration_s: float,
    phase_spans: Sequence[Mapping[str, Any]],
    preconditions: Sequence[Mapping[str, Any]],
    source_hashes: Mapping[str, Any],
    registry_precheck: Mapping[str, Any],
    schedule: Sequence[Mapping[str, Any]],
    episode_rows: Sequence[Mapping[str, Any]],
    seam_shards: Sequence[Mapping[str, Any]],
    boundary_events: Sequence[Mapping[str, Any]],
    runtime_receipt: Mapping[str, Any],
    model_specs: Sequence[Mapping[str, Any]],
    validation_receipts: Sequence[Mapping[str, Any]],
    require_terminal: bool,
) -> JsonDict:
    """Build one terminal record only from current raw rows and receipts."""

    seam_events = _events_from_shards(seam_shards)
    reduced = compact_reduction(reduce_panel(schedule, episode_rows, seam_events))
    invocation = _invocation_reduction(
        boundary_events, child_terminal=bool(runtime_receipt.get("child_terminal", True))
    )
    counts = invocation["invocation_counts"]
    balanced = all(
        counts[f"{prefix}_attempted"]
        == sum(
            counts[f"{prefix}_{state}"]
            for state in ("completed", "failed", "cancelled", "in_flight")
        )
        for prefix in ("model_loads", "forward_calls", "generation_calls")
    )
    load_attempted = counts["model_loads_attempted"] > 0
    offload = bool((runtime_receipt.get("observed_cuda_offload") or {}).get("passed"))
    affected_ok = (
        _receipts_pass(validation_receipts, validation_scope.REQUIRED_CHECK_NAMES)
        if require_terminal
        else True
    )
    e2e_ok = _receipts_pass(validation_receipts, CAPABILITY_E2E_NAMES) if require_terminal else True
    terminal_ok = (
        _receipts_pass(validation_receipts, REQUIRED_TERMINAL_NAMES) if require_terminal else True
    )
    validity = [
        _gate(
            "preconditions_checked",
            "validity",
            True,
            bool(preconditions) and all(row.get("passed") is True for row in preconditions),
            upstream="preconditions_checked",
            field="passed",
            principle="A positive scientific metric cannot excuse invalid evidence.",
        ),
        _gate(
            "exact_panel_schedule",
            "validity",
            True,
            canonical_hash(schedule) == canonical_hash(build_panel_schedule()),
            upstream="frozen_schedule",
            field="rows",
            principle="Outcome-driven replacement would invalidate the generalization sample.",
        ),
        _gate(
            "eighteen_episode_dispositions",
            "validity",
            True,
            reduced["all_dispositions_present"],
            upstream="raw_episode_rows",
            field="disposition",
            principle="A positive scientific metric cannot excuse invalid evidence.",
        ),
        _gate(
            "exclusive_interval_accounting",
            "validity",
            True,
            reduced["all_interval_bounds_valid"],
            upstream="exclusive_cost_rows",
            field="bounds_valid",
            principle="Overlapping model work cannot be counted twice or relabeled removable.",
        ),
        _gate(
            "invocation_accounting_balanced",
            "validity",
            True,
            balanced,
            upstream="current_invocation_events",
            field="invocation_counts",
            principle="Attempted calls must keep exactly one terminal or in-flight disposition.",
        ),
        _gate(
            "current_model_load_attempted",
            "validity",
            True,
            load_attempted,
            upstream="current_invocation_events",
            field="model_loads_attempted",
            principle="Archived model receipts cannot stand in for current Panel A work.",
        ),
        _gate(
            "actual_cuda_offload_receipt",
            "validity",
            True,
            offload,
            upstream="execution_venue_details.observed_cuda_offload",
            field="passed",
            principle="Requested GPU layers do not prove that the owned model occupied CUDA memory.",
        ),
        _gate(
            "affected_validation",
            "validity",
            True,
            affected_ok,
            upstream="validation_receipts",
            field="affected_check_names",
            principle="A positive scientific metric cannot excuse invalid evidence.",
        ),
        _gate(
            "capability_e2e",
            "validity",
            True,
            e2e_ok,
            upstream="validation_receipts",
            field="E2E-009|E2E-010|E2E-011|private_arc_smoke",
            principle="Unit fixtures cannot replace the actual ARC plumbing checks.",
        ),
        _gate(
            "terminal_readers",
            "validity",
            True,
            terminal_ok,
            upstream="validation_receipts",
            field="terminal_command_names",
            principle="A positive scientific metric cannot excuse invalid evidence.",
        ),
    ]
    readiness = [
        _gate(
            "panel_a_instrument_complete",
            "readiness",
            True,
            reduced["all_dispositions_present"] and balanced and load_attempted and offload,
            upstream="rows|current_invocation_events|execution_venue_details",
            field="arc_panel_a_complete_score",
            principle="A valid null must not suppress an independent measurement.",
        )
    ]
    historical_complete = 8
    support = historical_complete + reduced["sample_size_budget"]["complete_independent_units"]
    progressed_rows = [row for row in reduced["rows"] if row.get("actions_to_progress") is not None]
    retention = bool(progressed_rows) and all(
        row.get("offline_reproduced") is True for row in progressed_rows
    )
    benefit = [
        _gate(
            "e6_episode_support_floor",
            "scientific_benefit",
            30,
            support,
            upstream="exp7471_plus_panel_a_complete_units",
            field="sample_size_budget.pooled_complete_independent_units",
            principle="A small sample, a favorable seed or an analytic fixture cannot substitute for held-out value.",
            op=">=",
        ),
        _gate(
            "e6_game_support_floor",
            "scientific_benefit",
            10,
            4 + reduced["sample_size_budget"]["independent_game_clusters"],
            upstream="exp7471_plus_panel_a_game_clusters",
            field="sample_size_budget.pooled_independent_game_clusters",
            principle="Repeated seeds cannot substitute for independent game support.",
            op=">=",
        ),
        _gate(
            "effect_size_threshold",
            "scientific_benefit",
            True,
            False,
            upstream="observation_only_no_intervention",
            field="effect_size",
            principle="A timing observation cannot substitute for a held-out intervention effect.",
        ),
        _gate(
            "retention_threshold",
            "scientific_benefit",
            True,
            retention,
            upstream="current_trace_reproduction",
            field="retention",
            principle="Cost savings cannot excuse lost ARC progress.",
        ),
        _gate(
            "multiplicity_threshold",
            "scientific_benefit",
            "not_applicable_no_hypothesis_family",
            "not_applicable_no_hypothesis_family",
            upstream="observation_scope",
            field="multiplicity",
            principle="One favorable comparison cannot bypass a declared hypothesis-family correction.",
        ),
    ]
    gates = [*validity, *readiness, *benefit]
    required_ok = all(row["passed"] for row in (*validity, *readiness))
    complete_score = int(
        reduced["all_dispositions_present"] and balanced and load_attempted and offload
    )
    status = (
        "complete_null_live_arc_cost_panel_a"
        if required_ok
        else "complete_disqualified_live_arc_cost_panel_a"
    )
    reproduced_rows = [row for row in reduced["rows"] if row.get("offline_reproduced") is True]
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "status": status,
        "run_date": RUN_DATE,
        "started_at_utc": started_at_utc,
        "ended_at_utc": ended_at_utc,
        "clock_identity": {"wall": "datetime.now(UTC)", "interval": "time.monotonic_ns"},
        "preconditions_checked": deepcopy(list(preconditions)),
        "MODEL_SPECS": deepcopy(list(model_specs)) if load_attempted else [MODEL_ID],
        "model_specs": deepcopy(list(model_specs)) if load_attempted else [MODEL_ID],
        "model_invoked": bool(invocation["model_invoked"]),
        "invocation_counts": deepcopy(counts),
        "current_invocation_rows": deepcopy(invocation.get("call_rows") or []),
        "inference_substrate": invocation["inference_substrate"],
        "inference_substrate_class": invocation["inference_substrate_class"],
        "planned_inference_substrate_class": INFERENCE_SUBSTRATE_CLASS,
        "execution_venue": EXECUTION_VENUE,
        "execution_venue_details": deepcopy(dict(runtime_receipt)),
        "requested_offload": {"n_gpu_layers": 999, "device": "one_owned_cuda_gpu"},
        "observed_offload": deepcopy(runtime_receipt.get("observed_cuda_offload") or {}),
        "duration_s": round(float(duration_s), 6),
        "duration_breakdown_s": {
            str(row.get("phase")): round(float(row.get("duration_s") or 0.0), 6)
            for row in phase_spans
        },
        "phase_spans": deepcopy(list(phase_spans)),
        "random_seed": {
            "selection": 7_478,
            "episodes": list(EPISODE_SEEDS),
            "ordering": 7_478,
            "audit": 7_485_091,
            "bootstrap": 7_485_095,
        },
        "reproducibility_checksum": "",
        "source_artifact_hashes": deepcopy(dict(source_hashes)),
        "registry_precheck": deepcopy(dict(registry_precheck)),
        "protocol_schedule": deepcopy(list(schedule)),
        "rows": reduced["rows"],
        "sample_size_budget": reduced["sample_size_budget"],
        "per_game_results": reduced["per_game_results"],
        "exclusive_cost_rows": reduced["exclusive_cost_rows"],
        "seam_event_shards": deepcopy(list(seam_shards)),
        "acceptance_gate_results": gates,
        "gate_check_summary": _gate_summary(gates),
        "honest_verdict": status,
        "verdict_class": "null" if required_ok else "disqualified",
        "verifier_is_oracle": True,
        "flagged_adversarial": False,
        "validation_manifest": {
            "experiment_id": MANIFEST.experiment_id,
            "test_paths": list(MANIFEST.test_paths),
            "changed_modules": list(MANIFEST.changed_modules),
            "static_paths": list(MANIFEST.static_paths),
        },
        "validation_receipts": deepcopy(list(validation_receipts)),
        "arc_panel_a_complete_score": complete_score,
        "scientific_benefit_score": int(all(row["passed"] for row in benefit)),
        "solve_provenance": "live_agent_self_discovery",
        "offline_reproduced": bool(reproduced_rows),
        "reproduced_levels": sum(int(row.get("reproduced_levels") or 0) for row in reproduced_rows),
        "new_level_credit": 0,
        "generalization_scope": "public_adapter_withheld_proxy",
        "hidden_game_efficacy_claim": False,
        "selector_efficacy_claim": False,
        "e0_parity_claim": False,
        "e4_e5_readiness_claim": False,
        "panel_b_dependency": False,
        "remote_submission": False,
        "scored_path_changed": False,
        "research_conductor_changed": False,
        "small_ebm_training": {"performed": False},
    }
    artifact["field_principles"] = {
        key: FIELD_PRINCIPLES.get(
            key, "Retain this typed field so independent readers can audit the experiment."
        )
        for key in artifact
    }
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return artifact


def independent_reduce(artifact: Mapping[str, Any]) -> JsonDict:
    """Recompute every per-unit and cost claim from frozen rows and raw shards."""

    reduced = compact_reduction(
        reduce_panel(
            artifact.get("protocol_schedule") or [],
            artifact.get("rows") or [],
            _events_from_shards(artifact.get("seam_event_shards") or []),
        )
    )
    declared = {
        key: artifact.get(key)
        for key in ("rows", "sample_size_budget", "per_game_results", "exclusive_cost_rows")
    }
    recomputed = {
        key: reduced[key]
        for key in ("rows", "sample_size_budget", "per_game_results", "exclusive_cost_rows")
    }
    return {
        **recomputed,
        "matches_declared": canonical_hash(declared) == canonical_hash(recomputed),
    }


def validate_artifact(value: Mapping[str, Any] | Path, *, require_terminal: bool) -> list[str]:
    """Cold-check identity, reduction, invocation balance, claims and checksum."""

    artifact = load_object(value) if isinstance(value, Path) else dict(value)
    errors = [
        f"missing_field:{field}" for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact
    ]
    for field, expected in (
        ("schema", SCHEMA),
        ("experiment_id", EXPERIMENT_ID),
        ("milestone", MILESTONE),
        ("run_date", RUN_DATE),
        ("generalization_scope", "public_adapter_withheld_proxy"),
        ("solve_provenance", "live_agent_self_discovery"),
    ):
        if artifact.get(field) != expected:
            errors.append(f"identity_mismatch:{field}")
    if independent_reduce(artifact)["matches_declared"] is not True:
        errors.append("independent_reduction_mismatch")
    counts = artifact.get("invocation_counts") or {}
    for prefix in ("model_loads", "forward_calls", "generation_calls"):
        attempted = int(counts.get(f"{prefix}_attempted") or 0)
        terminal = sum(
            int(counts.get(f"{prefix}_{state}") or 0)
            for state in ("completed", "failed", "cancelled", "in_flight")
        )
        if attempted != terminal:
            errors.append(f"unbalanced_invocations:{prefix}")
    if artifact.get("verdict_class") not in {
        "positive",
        "circular_positive",
        "null",
        "blocked",
        "disqualified",
        "partial",
    }:
        errors.append("invalid_verdict_class")
    if artifact.get("verifier_is_oracle") is True and artifact.get("verdict_class") == "positive":
        errors.append("oracle_positive_forbidden")
    if artifact.get("new_level_credit") != 0:
        errors.append("registered_public_credit_nonzero")
    if require_terminal and not _receipts_pass(
        artifact.get("validation_receipts") or [],
        (*validation_scope.REQUIRED_CHECK_NAMES, *CAPABILITY_E2E_NAMES, *REQUIRED_TERMINAL_NAMES),
    ):
        errors.append("required_validation_missing_or_failed")
    if artifact.get("reproducibility_checksum") != artifact_checksum(artifact):
        errors.append("reproducibility_checksum_mismatch")
    return list(dict.fromkeys(errors))


def _fixture_receipts() -> list[JsonDict]:
    return [
        {"name": name, "passed": True, "exit_code": 0, "timed_out": False}
        for name in (
            *validation_scope.REQUIRED_CHECK_NAMES,
            *CAPABILITY_E2E_NAMES,
            *REQUIRED_TERMINAL_NAMES,
        )
    ]


def _fixture_boundary_events() -> list[JsonDict]:
    rows: list[JsonDict] = []
    for operation_index, operation in enumerate(("model_load", "generation")):
        call_id = f"fixture-{operation}"
        started = 10 + operation_index * 10
        for state_index, state in enumerate(("attempted", "completed")):
            rows.append(
                {
                    "schema": "carnot.arc_inference_boundary_event.v1",
                    "call_id": call_id,
                    "event_id": canonical_hash({"call_id": call_id, "state": state}),
                    "operation": operation,
                    "owner_pid": 1,
                    "child_pid": 2,
                    "state": state,
                    "recorded_monotonic_ns": started + state_index,
                    "started_monotonic_ns": started,
                    "ended_monotonic_ns": started + 1 if state == "completed" else None,
                    "model_identity": {
                        "model_repository": MODEL_ID,
                        "model_revision": "fixture-revision",
                        "model_filename": "Qwen3.8-27B-Q4_K_M.gguf",
                        "model_path": "/tmp/Qwen3.8-27B-Q4_K_M.gguf",
                    },
                }
            )
    return rows


def build_artifact_for_test() -> JsonDict:
    """Build a terminal-shaped fixture through the production reducers."""

    schedule = build_panel_schedule()
    episodes = []
    for sealed in schedule:
        episodes.append(
            {
                **deepcopy(sealed),
                "disposition": "complete",
                "action_count": 1,
                "start_level": 0,
                "peak_level": 0,
                "terminal_level": 0,
                "action_rows": [
                    {
                        "action_index": 1,
                        "level": 0,
                        "later_progress": False,
                        "interval_start_monotonic_ns": 1_000,
                        "interval_end_monotonic_ns": 2_000,
                    }
                ],
                "request_budget_receipt": {
                    "attempted": 0,
                    "completed": 0,
                    "failed": 0,
                    "cancelled": 0,
                    "in_flight": 0,
                },
                "trace_reproduction": {"attempted": False, "passed": False},
                "solve_provenance": "no_level_reached",
                "elapsed_s": 0.000001,
                "new_level_credit": 0,
                "error": None,
            }
        )
    events = normalize_interval_events(
        [
            {
                "episode_id": schedule[0]["episode_id"],
                "decision_id": "fixture-decision",
                "seam": "induction_timing",
                "event": "stage_start",
                "event_monotonic_ns": 1_100,
                "interval_start_monotonic_ns": 1_100,
            },
            {
                "episode_id": schedule[0]["episode_id"],
                "decision_id": "fixture-decision",
                "seam": "induction_timing",
                "event": "stage_end",
                "event_monotonic_ns": 1_200,
                "interval_end_monotonic_ns": 1_200,
            },
        ],
        owner_pid=1,
    )
    shard = {
        "episode_id": schedule[0]["episode_id"],
        "path": "inline_fixture",
        "sha256": canonical_hash(events),
        "row_count": len(events),
        "inline_rows": events,
    }
    return build_terminal_artifact(
        started_at_utc="2026-09-21T00:00:00Z",
        ended_at_utc="2026-09-21T00:01:00Z",
        duration_s=60.0,
        phase_spans=[{"phase": "fixture", "duration_s": 60.0, "completed_units": 18}],
        preconditions=[{"check": "fixture", "passed": True}],
        source_hashes={},
        registry_precheck={"rows": [], "new_credit_allowed": False},
        schedule=schedule,
        episode_rows=episodes,
        seam_shards=[shard],
        boundary_events=_fixture_boundary_events(),
        runtime_receipt={
            "child_terminal": True,
            "observed_cuda_offload": {"passed": True, "owned_server_vram_mb": 1},
        },
        model_specs=[{"hf_id": MODEL_ID, "quantization": "Q4_K_M"}],
        validation_receipts=_fixture_receipts(),
        require_terminal=True,
    )


def build_validation_plan(root: Path, private_root: Path) -> list[validation_scope.CommandSpec]:
    """Freeze affected checks plus the three shared ARC capability checks."""

    private_root.mkdir(parents=True, exist_ok=True)
    affected = validation_contract.build_command_plan(root, MANIFEST, private_root / "affected")
    shared = interval_protocol.build_validation_plan(root, private_root / "shared")
    return [*affected, *(row for row in shared if row.name in CAPABILITY_E2E_NAMES)]


def validate_validation_plan(
    root: Path, commands: Sequence[validation_scope.CommandSpec]
) -> list[str]:
    """Reject command drift, duplicate checks and broad test targets."""

    affected = [row for row in commands if row.name in validation_scope.REQUIRED_CHECK_NAMES]
    errors = validation_contract.validate_command_plan(root, MANIFEST, affected)
    counts = Counter(row.name for row in commands)
    for name in (*validation_scope.REQUIRED_CHECK_NAMES, *CAPABILITY_E2E_NAMES):
        if counts[name] != 1:
            errors.append(f"command_count:{name}:{counts[name]}")
    for row in commands:
        if any(argument.rstrip("/") in {"tests", "tests/python"} for argument in row.argv):
            errors.append(f"broad_test_target:{row.name}")
    return errors


def terminal_command_specs(root: Path, candidate: Path) -> list[validation_scope.CommandSpec]:
    """Build cold replay, independent reduction and unchanged strict readers."""

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
            (python, "-u", "scripts/verdict_row_consistency_lint.py", "--strict", str(candidate)),
            "exact_terminal_candidate",
            300.0,
        ),
    ]


def _configure_live_driver() -> None:  # pragma: no cover - owned live child configuration.
    """Point the qualified live driver at this task's isolated owned paths."""

    values = {
        "TASK_ID": TASK_ID,
        "RAW_DIR": RAW_DIR,
        "SCHEDULE_PATH": SCHEDULE_PATH,
        "SESSION_PATH": SESSION_PATH,
        "BOUNDARY_PATH": BOUNDARY_PATH,
        "RUNTIME_EVENT_PATH": RUNTIME_EVENT_PATH,
        "ACTION_PATH": ACTION_PATH,
        "CALLBACK_SEAM_PATH": CALLBACK_SEAM_PATH,
        "CHECKPOINT_PATH": CHECKPOINT_PATH,
        "WRAPPER_PATH": WRAPPER_PATH,
        "RUN_DATE": RUN_DATE,
        "EPISODE_SEEDS": EPISODE_SEEDS,
        "ACTION_LIMIT": ACTION_LIMIT,
        "REQUEST_LIMIT": REQUEST_LIMIT,
        "MAX_NEW_TOKENS": MAX_NEW_TOKENS,
        "EPISODE_LIMIT_S": EPISODE_LIMIT_S,
        "AGGREGATE_LIVE_LIMIT_S": PANEL_LIVE_LIMIT_S,
    }
    for name, value in values.items():
        setattr(live_driver, name, value)


def _normalize_live_shards(
    root: Path, episodes: Sequence[Mapping[str, Any]], owner_pid: int
) -> list[JsonDict]:  # pragma: no cover
    """Persist qualified rows while keeping each original raw shard immutable."""

    shards: list[JsonDict] = []
    source_shards = live_driver._seam_shards(root, episodes)
    callback = live_driver._write_callback_seam_shard(
        root, load_object(root / SESSION_PATH).get("runtime_receipt", {}).get("gpu_uuid")
    )
    if callback is not None:
        source_shards.append(callback)
    for index, source in enumerate(source_shards):
        raw = Path(str(source["path"]))
        raw = raw if raw.is_absolute() else root / raw
        rows = normalize_interval_events(live_driver.read_jsonl(raw), owner_pid=owner_pid)
        output = (
            root
            / NORMALIZED_DIR
            / f"{index:02d}_{str(source.get('episode_id')).replace(':', '__')}.jsonl"
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        output.unlink(missing_ok=True)
        for row in rows:
            live_driver._append_jsonl(output, row)
        shards.append(
            {
                "episode_id": source.get("episode_id"),
                "path": output.relative_to(root).as_posix(),
                "sha256": current_work_receipt.sha256_file(output),
                "bytes": output.stat().st_size,
                "row_count": len(rows),
                "source_raw_sha256": source.get("sha256"),
            }
        )
    return shards


def _phase(
    spans: list[JsonDict], name: str, began: float, started: float, units: int
) -> None:  # pragma: no cover
    ended = time.monotonic()
    spans.append(
        {
            "phase": name,
            "start_s": round(began - started, 6),
            "end_s": round(ended - started, 6),
            "duration_s": round(ended - began, 6),
            "completed_units": units,
            "ended_at_utc": utc_now(),
        }
    )


def _blocked_artifact(
    *,
    started_at: str,
    started: float,
    spans: Sequence[Mapping[str, Any]],
    checks: Sequence[Mapping[str, Any]],
    hashes: Mapping[str, Any],
    registry: Mapping[str, Any],
    schedule: Sequence[Mapping[str, Any]],
    receipts: Sequence[Mapping[str, Any]],
) -> JsonDict:  # pragma: no cover
    """Publish an honest pre-model block without fabricated live receipts."""

    artifact = build_terminal_artifact(
        started_at_utc=started_at,
        ended_at_utc=utc_now(),
        duration_s=time.monotonic() - started,
        phase_spans=spans,
        preconditions=checks,
        source_hashes=hashes,
        registry_precheck=registry,
        schedule=schedule,
        episode_rows=[],
        seam_shards=[],
        boundary_events=[],
        runtime_receipt={"child_terminal": True},
        model_specs=[],
        validation_receipts=receipts,
        require_terminal=False,
    )
    artifact["status"] = "blocked_panel_a_precondition"
    artifact["honest_verdict"] = artifact["status"]
    artifact["verdict_class"] = "blocked"
    artifact["arc_panel_a_complete_score"] = 0
    artifact["gate_check_summary"] = _gate_summary(artifact["acceptance_gate_results"])
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return artifact


def run_experiment(root: Path, run_date: str) -> JsonDict:  # pragma: no cover - orchestration.
    """Validate, run one owned live child, reduce, cold-check and publish."""

    started = time.monotonic()
    started_at = utc_now()
    spans: list[JsonDict] = []
    receipts: list[JsonDict] = []
    progress(started, "startup", "begin", run_date=run_date)
    phase = time.monotonic()
    progress(started, "preconditions", "before")
    checks, hashes, _upstream, registry = collect_preconditions(root)
    checks.insert(
        0,
        _gate(
            "run_date",
            "validity",
            RUN_DATE,
            run_date,
            upstream="command_line",
            field="--date",
            principle="A wrong run date would detach evidence from its roadmap task.",
        ),
    )
    schedule = build_panel_schedule()
    current_work_receipt.atomic_json(
        root / SCHEDULE_PATH,
        {"rows": schedule, "registry_precheck": registry, "panel_b_dependency": False},
    )
    _phase(spans, "preconditions", phase, started, len(checks))
    progress(started, "preconditions", "after", passed=all(row["passed"] for row in checks))
    if not all(row["passed"] for row in checks):
        artifact = _blocked_artifact(
            started_at=started_at,
            started=started,
            spans=spans,
            checks=checks,
            hashes=hashes,
            registry=registry,
            schedule=schedule,
            receipts=receipts,
        )
        current_work_receipt.atomic_json(root / RESULT_PATH, artifact)
        return artifact

    private = Path(tempfile.mkdtemp(prefix="exp7485-validation-", dir="/tmp"))
    phase = time.monotonic()
    progress(started, "validation", "before_affected_and_e2e_subprocesses")
    plan = build_validation_plan(root, private / "scoped")
    errors = validate_validation_plan(root, plan)
    if errors:
        raise RuntimeError(f"validation_plan_invalid:{errors}")
    receipts.extend(
        validation_contract.run_categorized_commands(
            root,
            [validation_contract.PlannedCommand(row, "required_validation", True) for row in plan],
            log_dir=root / RAW_DIR / "validation/affected_and_e2e",
            heartbeat_s=60.0,
        )
    )
    _phase(spans, "affected_and_capability_validation", phase, started, len(receipts))
    progress(
        started, "validation", "after_affected_and_e2e_subprocesses", completed_units=len(receipts)
    )
    if not _receipts_pass(
        receipts, (*validation_scope.REQUIRED_CHECK_NAMES, *CAPABILITY_E2E_NAMES)
    ):
        artifact = _blocked_artifact(
            started_at=started_at,
            started=started,
            spans=spans,
            checks=checks,
            hashes=hashes,
            registry=registry,
            schedule=schedule,
            receipts=receipts,
        )
        artifact["status"] = "complete_disqualified_affected_validation"
        artifact["honest_verdict"] = artifact["status"]
        artifact["verdict_class"] = "disqualified"
        artifact["reproducibility_checksum"] = artifact_checksum(artifact)
        current_work_receipt.atomic_json(root / RESULT_PATH, artifact)
        return artifact

    _configure_live_driver()
    phase = time.monotonic()
    progress(started, "runtime_preconditions", "before_cached_model_cuda_lease_checks")
    runtime_checks, runtime_hashes, resources = live_driver._runtime_preconditions(root, started)
    checks.extend(runtime_checks)
    hashes.update(runtime_hashes)
    _phase(spans, "runtime_preconditions", phase, started, len(runtime_checks))
    progress(
        started,
        "runtime_preconditions",
        "after_cached_model_cuda_lease_checks",
        passed=all(row.get("passed") is True for row in checks),
    )
    if not all(row.get("passed") is True for row in checks):
        artifact = _blocked_artifact(
            started_at=started_at,
            started=started,
            spans=spans,
            checks=checks,
            hashes=hashes,
            registry=registry,
            schedule=schedule,
            receipts=receipts,
        )
        current_work_receipt.atomic_json(root / RESULT_PATH, artifact)
        return artifact

    for relative in (
        BOUNDARY_PATH,
        RUNTIME_EVENT_PATH,
        ACTION_PATH,
        CALLBACK_SEAM_PATH,
        SESSION_PATH,
        CHECKPOINT_PATH,
        CANDIDATE_PATH,
        RAW_DIR / "episode_rows.json",
    ):
        (root / relative).unlink(missing_ok=True)
    phase = time.monotonic()
    progress(started, "live", "before_model_load_generation_benchmark", planned_units=18)
    session = live_driver.run_child_with_lease(
        resources=resources, schedule_path=root / SCHEDULE_PATH, started=started
    )
    episodes = [dict(row) for row in session.get("episodes") or [] if isinstance(row, Mapping)]
    progress(
        started,
        "live",
        "after_model_load_generation_benchmark",
        completed_units=len(episodes),
    )
    _phase(spans, "live_model_and_episodes", phase, started, len(episodes))
    boundary_events = InvocationBoundaryLedger(root / BOUNDARY_PATH).read_events()
    seam_shards = _normalize_live_shards(
        root, episodes, int(session.get("child_pid") or os.getpid())
    )
    for relative, role in (
        (SCHEDULE_PATH, "frozen_protocol"),
        (BOUNDARY_PATH, "current_invocation_ledger"),
        (RUNTIME_EVENT_PATH, "request_event_shard"),
        (ACTION_PATH, "action_outcome_shard"),
    ):
        path = root / relative
        if path.is_file():
            hashes[relative.as_posix()] = _source_record(path, role)
    for shard in seam_shards:
        path = root / str(shard["path"])
        hashes[str(shard["path"])] = _source_record(path, "qualified_interval_shard")

    phase = time.monotonic()
    candidate = build_terminal_artifact(
        started_at_utc=started_at,
        ended_at_utc=utc_now(),
        duration_s=time.monotonic() - started,
        phase_spans=spans,
        preconditions=checks,
        source_hashes=hashes,
        registry_precheck=registry,
        schedule=schedule,
        episode_rows=episodes,
        seam_shards=seam_shards,
        boundary_events=boundary_events,
        runtime_receipt=session.get("runtime_receipt") or {},
        model_specs=[resources["model_spec"]],
        validation_receipts=receipts,
        require_terminal=False,
    )
    candidate_errors = validate_artifact(candidate, require_terminal=False)
    if candidate_errors:
        raise RuntimeError(f"candidate_invalid:{candidate_errors}")
    current_work_receipt.atomic_json(root / CANDIDATE_PATH, candidate)
    _phase(spans, "independent_reduction", phase, started, len(candidate["rows"]))

    phase = time.monotonic()
    progress(started, "terminal_validation", "before_subprocesses", candidate=CANDIDATE_PATH)
    terminal = validation_scope.run_commands(
        root,
        terminal_command_specs(root, root / CANDIDATE_PATH),
        log_dir=root / RAW_DIR / "validation/terminal",
        heartbeat_s=60.0,
    )
    receipts.extend(terminal)
    _phase(spans, "terminal_validation", phase, started, len(terminal))
    progress(started, "terminal_validation", "after_subprocesses", completed_units=len(terminal))

    artifact = build_terminal_artifact(
        started_at_utc=started_at,
        ended_at_utc=utc_now(),
        duration_s=time.monotonic() - started,
        phase_spans=spans,
        preconditions=checks,
        source_hashes=hashes,
        registry_precheck=registry,
        schedule=schedule,
        episode_rows=episodes,
        seam_shards=seam_shards,
        boundary_events=boundary_events,
        runtime_receipt=session.get("runtime_receipt") or {},
        model_specs=[resources["model_spec"]],
        validation_receipts=receipts,
        require_terminal=True,
    )
    errors = validate_artifact(artifact, require_terminal=True)
    if errors:
        raise RuntimeError(f"terminal_artifact_invalid:{errors}")
    progress(started, "publish", "before_atomic_terminal_write", path=RESULT_PATH)
    current_work_receipt.atomic_json(root / RESULT_PATH, artifact)
    progress(
        started,
        "publish",
        "after_atomic_terminal_write",
        path=RESULT_PATH,
        complete=artifact["arc_panel_a_complete_score"],
    )
    return artifact


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the public experiment, live-child and cold-replay roles."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", choices=[RUN_DATE])
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
        parser.error("--date or --replay is required")
    return args


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - thin CLI.
    """Run the host experiment, its owned child, or a fresh cold replay."""

    args = parse_args(argv)
    if args.replay is not None:
        artifact = load_object(args.replay)
        reduced = independent_reduce(artifact)
        errors = [] if args.reduce_only else validate_artifact(artifact, require_terminal=False)
        print(
            json.dumps(
                {
                    "matches_declared": reduced.get("matches_declared"),
                    "row_count": len(reduced.get("rows") or []),
                    "validation_errors": errors,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return int(bool(errors) or reduced.get("matches_declared") is not True)
    if args.role == "live-session":
        _configure_live_driver()
        return live_driver.run_live_session(args)
    artifact = run_experiment(REPO_ROOT, str(args.date))
    return 0 if artifact.get("verdict_class") in {"null", "blocked"} else 1


__all__ = [
    "CAPABILITY_E2E_NAMES",
    "INFERENCE_SUBSTRATE_CLASS",
    "MODEL_SPECS",
    "REQUIRED_ARTIFACT_FIELDS",
    "REQUIRED_TERMINAL_NAMES",
    "SPEC_PATH",
    "build_artifact_for_test",
    "build_panel_schedule",
    "build_validation_plan",
    "collect_preconditions",
    "independent_reduce",
    "main",
    "normalize_interval_events",
    "reduce_panel",
    "terminal_command_specs",
    "validate_artifact",
    "validate_validation_plan",
]
