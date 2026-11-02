"""Diagnose and harden the ARC scored-policy invocation boundary.

The current experiment does not load an LLM. It reduces the immutable Exp7376
receipts, runs a scripted CPU transport through the submitted agent factory,
and validates the timeout accounting that future bounded children can reuse.

Spec refs: REQ-ARC-WMTE-7384 and SCENARIO-ARC-WMTE-7384-*.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
from copy import deepcopy
from datetime import UTC, datetime
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import tempfile
import time
from typing import Any
import urllib.request

from carnot import experiment_7358_v646_validation_contract as validation_contract
from carnot.reporting import experiment_7303_validation_scope as validation_scope


JsonDict = dict[str, Any]
REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ID = "exp7384-arc-invocation-boundary"
MILESTONE = "2026.09.648"
RUN_DATE = "20260918"
SCHEMA = "carnot.exp7384.v648_arc_invocation_boundary.v1"
RESULT_PATH = Path("results/experiment_7384_v648_arc_invocation_boundary.json")
RAW_DIR = Path("results/raw/experiment_7384_v648_arc_invocation_boundary")
MODULE_PATH = Path("python/carnot/experiment_7384_v648_arc_invocation_boundary.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_7384_v648_arc_invocation_boundary.py")
TEST_PATH = Path("tests/python/test_experiment_7384_v648_arc_invocation_boundary.py")
SPEC_PATH = Path("openspec/capabilities/arc-world-model-trust-energy/spec.md")
HISTORICAL_ARTIFACT = Path("results/experiment_7376_v647_arc_outcomes.json")
HISTORICAL_RAW = Path("results/raw/experiment_7376_v647_arc_outcomes")
REQUIRED_CHECK_NAMES = validation_scope.REQUIRED_CHECK_NAMES
REQUIRED_E2E_NAMES = ("e2e_009", "e2e_010", "e2e_offline_smoke")
ZERO_INVOCATION_COUNTS = {
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
    "usable_answers": 0,
    "raw_receipts": [],
}

INPUT_PATHS = (
    Path("AGENTS.md"),
    Path("CLAUDE.md"),
    Path("CODEX.md"),
    Path("research-program.md"),
    Path("ops/exclusion_manifest.yaml"),
    Path("ops/e2e-test-plan.md"),
    Path("scripts/experiment_template.py"),
    Path("python/carnot/reporting/experiment_7303_validation_scope.py"),
    Path("python/carnot/experiment_7358_v646_validation_contract.py"),
    Path("python/carnot/experiment_7376_v647_arc_outcomes.py"),
    Path("python/carnot/experiment_7345_v645_arc_resume_check.py"),
    Path("python/carnot/experiment_7336_v644_arc_resume.py"),
    Path("python/carnot/experiment_7318_v643_arc_authority.py"),
    Path("python/carnot/experiment_7263_v639_arc_live.py"),
    Path("python/carnot/agentic/arc_competition_agent.py"),
    Path("scripts/arc_loop_solve.py"),
    Path("ops/arc_solve_registry.yaml"),
    Path("tests/python/test_arc_induction_state_persistence.py"),
    Path("tests/python/test_arc_tool_grammar_transport.py"),
    Path("openspec/capabilities/research-reporting/spec.md"),
    SPEC_PATH,
    HISTORICAL_ARTIFACT,
    HISTORICAL_RAW / "live_session.json",
    HISTORICAL_RAW / "episode_rows.json",
    HISTORICAL_RAW / "frozen_schedule.json",
    HISTORICAL_RAW / "receipt_events.jsonl",
)


def utc_now() -> str:
    """Return one aware timestamp for an observed experiment boundary."""

    return datetime.now(UTC).isoformat()


def progress(started: float, phase: str, event: str, **details: Any) -> None:
    """Flush a truthful boundary so long validation work stays observable."""

    suffix = " ".join(f"{key}={value}" for key, value in sorted(details.items()))
    print(
        f"[exp7384] phase={phase} event={event} "
        f"elapsed_s={time.monotonic() - started:.3f}" + (f" {suffix}" if suffix else ""),
        flush=True,
    )


def sha256_file(path: Path) -> str:
    """Hash exact source bytes without loading a large file into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def canonical_hash(value: Any) -> str:
    """Hash stable JSON bytes so reducers can detect silent evidence drift."""

    data = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    return "sha256:" + hashlib.sha256(data).hexdigest()


def artifact_checksum(value: Mapping[str, Any]) -> str:
    """Bind every terminal field except the checksum slot that contains the result."""

    return canonical_hash(
        {key: item for key, item in value.items() if key != "reproducibility_checksum"}
    )


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    """Replace a JSON artifact only after its complete bytes reach local storage."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def load_object(path: Path) -> JsonDict:
    """Return one JSON object, or an empty object for missing or malformed bytes."""

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(value) if isinstance(value, Mapping) else {}


def gate_row(
    check: str,
    category: str,
    expected: Any,
    observed: Any,
    *,
    upstream: str,
    artifact_field: str,
    operator: str = "==",
) -> JsonDict:
    """Keep a gate's value, comparison, and evidence path separate."""

    if operator == "==":
        passed = observed == expected
    elif operator == ">=":
        passed = observed is not None and observed >= expected
    elif operator == "is":
        passed = observed is expected
    else:
        raise ValueError(f"unsupported gate operator: {operator}")
    return {
        "check": check,
        "category": category,
        "upstream": upstream,
        "artifact_field": artifact_field,
        "expected": expected,
        "observed": observed,
        "operator": operator,
        "passed": passed,
    }


def gate_summary(gates: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Retain every failed gate and expose the first exact failure."""

    failed = [deepcopy(dict(row)) for row in gates if row.get("passed") is not True]
    return {
        "all_passed": not failed,
        "failed_count": len(failed),
        "failed_checks": failed,
        "first_failure": failed[0] if failed else None,
    }


def recover_timeout_accounting(
    *,
    schedule: Sequence[Mapping[str, Any]],
    terminal_session: Mapping[str, Any] | None,
    durable_episode_rows: Sequence[Mapping[str, Any]],
    progress_rows: Mapping[str, Mapping[str, Any]],
    timed_out: bool,
) -> JsonDict:
    """Recover completed work before classifying the active and later units.

    A bounded parent can stop a child between its per-episode checkpoint and its
    final session write. The durable rows are real completed work. Only a unit
    with its own progress receipt is started-but-censored; later units are
    unstarted and must not be rewritten as zero-action attempts.
    """

    terminal_rows = (
        list(terminal_session.get("episodes") or [])
        if isinstance(terminal_session, Mapping)
        else []
    )
    source = "terminal_session" if terminal_rows else "durable_episode_checkpoint"
    available = terminal_rows if terminal_rows else list(durable_episode_rows)
    by_id = {
        str(row.get("episode_id")): deepcopy(dict(row))
        for row in available
        if isinstance(row, Mapping) and row.get("episode_id") is not None
    }
    rows: list[JsonDict] = []
    errors: list[str] = []
    last_confirmed_event: str | None = None
    for sealed in schedule:
        episode_id = str(sealed.get("episode_id"))
        if episode_id in by_id:
            row = by_id[episode_id]
            row.setdefault("disposition", "complete")
            row.setdefault("censored", False)
            if row.get("error"):
                errors.append(str(row["error"]))
            rows.append(row)
            last_confirmed_event = "episode_checkpoint_written"
            continue
        progress_row = deepcopy(dict(progress_rows.get(episode_id) or {}))
        started = bool(
            progress_row
            and (
                progress_row.get("started_at")
                or progress_row.get("last_event")
                or int(progress_row.get("actions") or 0) > 0
            )
        )
        if started:
            last_confirmed_event = str(progress_row.get("last_event") or "episode_started")
            if progress_row.get("error"):
                errors.append(str(progress_row["error"]))
            rows.append(
                {
                    **deepcopy(dict(sealed)),
                    "disposition": "started_censored_timeout",
                    "censored": True,
                    "censoring_reason": "aggregate_episode_work_timeout",
                    "action_count": int(progress_row.get("actions") or 0),
                    "last_confirmed_event": last_confirmed_event,
                    "progress_receipt": progress_row,
                    "error": progress_row.get("error") or "aggregate_episode_work_timeout",
                }
            )
        else:
            rows.append(
                {
                    **deepcopy(dict(sealed)),
                    "disposition": "unstarted_timeout" if timed_out else "unstarted",
                    "censored": False,
                    "censoring_reason": None,
                    "action_count": 0,
                    "error": None,
                }
            )
    if isinstance(terminal_session, Mapping) and terminal_session.get("error"):
        errors.append(str(terminal_session["error"]))
    completed = sum(str(row.get("disposition", "")).startswith("complete") for row in rows)
    censored = sum(row.get("disposition") == "started_censored_timeout" for row in rows)
    unstarted = sum(str(row.get("disposition", "")).startswith("unstarted") for row in rows)
    if terminal_rows:
        last_confirmed_event = "child_terminal_session_write"
    return {
        "source": source,
        "rows": rows,
        "accounting": {
            "planned_units": len(schedule),
            "attempted_units": completed + censored,
            "completed_units": completed,
            "censored_units": censored,
            "unstarted_units": unstarted,
        },
        "earliest_exception": errors[0] if errors else None,
        "last_confirmed_event": last_confirmed_event,
        "first_missing_event": None if terminal_rows else "child_terminal_session_write",
    }


def reduce_invocation_events(events: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Reduce call identities so attempted and terminal states cannot contradict."""

    calls: dict[tuple[str, str], set[str]] = {}
    usable: set[tuple[str, str]] = set()
    earliest_exception: str | None = None
    for index, event in enumerate(events):
        operation = str(event.get("operation") or "")
        state = str(event.get("state") or "")
        call_id = str(event.get("call_id") or f"missing-{index}")
        calls.setdefault((operation, call_id), set()).add(state)
        if event.get("usable") is True:
            usable.add((operation, call_id))
        if earliest_exception is None and event.get("error"):
            earliest_exception = str(event["error"])

    def count(operation: str, state: str) -> int:
        return sum(state in states for (kind, _), states in calls.items() if kind == operation)

    load_attempted = count("model_load", "attempted")
    load_completed = count("model_load", "completed")
    load_failed = count("model_load", "failed")
    load_cancelled = count("model_load", "cancelled")
    gen_attempted = count("generation", "attempted")
    gen_completed = count("generation", "completed")
    gen_failed = count("generation", "failed")
    gen_cancelled = count("generation", "cancelled")
    load_terminal = load_completed + load_failed + load_cancelled
    gen_terminal = gen_completed + gen_failed + gen_cancelled
    counts = {
        "model_loads_attempted": load_attempted,
        "model_loads_completed": load_completed,
        "model_loads_failed": load_failed,
        "model_loads_cancelled": load_cancelled,
        "model_loads_in_flight": max(0, load_attempted - load_terminal),
        "generation_calls_attempted": gen_attempted,
        "generation_calls_completed": gen_completed,
        "generation_calls_failed": gen_failed,
        "generation_calls_cancelled": gen_cancelled,
        "generation_calls_in_flight": max(0, gen_attempted - gen_terminal),
        "usable_answers": sum(operation == "generation" for operation, _ in usable),
    }
    if load_failed and not load_completed:
        state = "failed_load"
    elif load_completed and not gen_attempted:
        state = "successful_load_no_generation"
    elif gen_failed and not gen_completed:
        state = "attempted_failed_generation"
    elif gen_completed:
        state = "successful_generation"
    elif load_attempted:
        state = "load_in_flight_or_cancelled"
    else:
        state = "no_invocation"
    return {
        "state": state,
        "counts": counts,
        "earliest_exception": earliest_exception,
        "raw_receipts": [deepcopy(dict(row)) for row in events],
    }


def _historical_progress_rows(
    raw: Path, schedule: Sequence[Mapping[str, Any]]
) -> dict[str, JsonDict]:
    """Read each scheduled unit's own progress receipt without inferring missing work."""

    rows: dict[str, JsonDict] = {}
    for sealed in schedule:
        episode_id = str(sealed.get("episode_id"))
        path = raw / episode_id.replace(":", "__") / "run_progress.json"
        value = load_object(path)
        if value:
            rows[episode_id] = value
    return rows


def _historical_boundary_rows(events: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Project authenticated historical events without treating them as current calls."""

    rows: list[JsonDict] = []
    for event in events:
        rows.append(
            {
                "source": "historical_exp7376_receipt_events",
                "operation": event.get("operation"),
                "state": event.get("state"),
                "call_id": event.get("call_id"),
                "owner_pid": event.get("owner_pid"),
                "child_pid": event.get("child_pid"),
                "started_monotonic_ns": event.get("started_monotonic_ns"),
                "ended_monotonic_ns": event.get("ended_monotonic_ns"),
                "recorded_monotonic_ns": event.get("recorded_monotonic_ns"),
                "historical_only": True,
            }
        )
    return rows


def diagnose_historical_exp7376(repo_root: Path) -> JsonDict:
    """Reload Exp7376 raw bytes and reproduce its terminal accounting loss."""

    root = repo_root.resolve()
    raw = root / HISTORICAL_RAW
    artifact = load_object(root / HISTORICAL_ARTIFACT)
    live_session = load_object(raw / "live_session.json")
    schedule = list(load_object(raw / "frozen_schedule.json").get("rows") or [])
    durable_rows = list(load_object(raw / "episode_rows.json").get("rows") or [])
    events: list[JsonDict] = []
    try:
        for line in (raw / "receipt_events.jsonl").read_text(encoding="utf-8").splitlines():
            value = json.loads(line)
            if isinstance(value, Mapping):
                events.append(dict(value))
    except (OSError, json.JSONDecodeError):
        events = []
    progress_rows = _historical_progress_rows(raw, schedule)
    corrected = recover_timeout_accounting(
        schedule=schedule,
        terminal_session=None if not live_session.get("episodes") else live_session,
        durable_episode_rows=durable_rows,
        progress_rows=progress_rows,
        timed_out=bool(live_session.get("timed_out")),
    )
    terminal_actions = sum(int(row.get("action_count") or 0) for row in artifact.get("rows", []))
    durable_actions = sum(
        int(row.get("action_count") or 0)
        for row in durable_rows
        if str(row.get("disposition")) == "complete"
    )
    invocation_reduction = reduce_invocation_events(events)
    accounting = corrected["accounting"]
    cause_reproduced = bool(
        live_session.get("error") == "live_child_did_not_write_session"
        and terminal_actions == 0
        and durable_actions > 0
        and accounting["completed_units"] == len(durable_rows)
        and accounting["attempted_units"] > accounting["completed_units"]
    )
    root_cause = {
        "observed_cause": (
            "The aggregate parent timeout terminated the owned child before its final session "
            "write. The fallback ignored five durable completed episode rows and the active "
            "sixth progress row. It then synthesized six zero-action censored rows."
        ),
        "failing_reproduction": {
            "terminal_session_error": live_session.get("error"),
            "terminal_summary_action_count": terminal_actions,
            "durable_completed_action_count": durable_actions,
            "old_accounting": artifact.get("sample_size_budget"),
        },
        "corrected_result": accounting,
        "corrected": cause_reproduced,
        "next_observation_if_unresolved": (
            None
            if cause_reproduced
            else "record child environment reset, policy construction, and first-action spans"
        ),
    }
    return {
        "cause_reproduced": cause_reproduced,
        "terminal_summary_action_count": terminal_actions,
        "durable_completed_action_count": durable_actions,
        "corrected_accounting": accounting,
        "last_confirmed_event": corrected["last_confirmed_event"],
        "first_missing_event": corrected["first_missing_event"],
        "earliest_exception": corrected["earliest_exception"],
        "historical_invocations": invocation_reduction,
        "boundary_event_rows": _historical_boundary_rows(events),
        "root_cause_receipt": root_cause,
        "rows": corrected["rows"],
        "entrypoint_receipt": {
            "entrypoint": "carnot.experiment_7263_v639_arc_live.run_child_with_lease",
            "child_entrypoint": "carnot.experiment_7263_v639_arc_live.run_live_session",
            "wrapper": "scripts/experiments/experiment_7376_v647_arc_outcomes.py",
            "arguments": {
                "schedule_path": HISTORICAL_RAW.joinpath("frozen_schedule.json").as_posix(),
                "raw_dir": HISTORICAL_RAW.as_posix(),
                "checkpoint_path": "results/checkpoints/experiment_7376_v647_arc_outcomes.json",
                "session_path": HISTORICAL_RAW.joinpath("live_session.json").as_posix(),
                "remaining_s": 1800.0,
            },
        },
    }


def _span_event(
    boundary: str,
    started_ns: int,
    ended_ns: int,
    *,
    state: str = "completed",
    detail: Mapping[str, Any] | None = None,
) -> JsonDict:
    """Build one current-process monotonic boundary row."""

    return {
        "source": "current_scripted_scored_path",
        "boundary": boundary,
        "state": state,
        "owner_pid": os.getpid(),
        "child_pid": None,
        "started_monotonic_ns": started_ns,
        "ended_monotonic_ns": ended_ns,
        "duration_s": max(0.0, (ended_ns - started_ns) / 1_000_000_000),
        "detail": dict(detail or {}),
        "historical_only": False,
    }


def run_scripted_factory_path(work_dir: Path) -> JsonDict:
    """Drive make_carnot_agent through HTTP tool feedback to a scored action."""

    from carnot import experiment_7234_v637_arc_scored_dryrun as scored
    from carnot import experiment_7318_v643_arc_authority as authority
    from carnot.agentic import arc_executable_world_model as e3

    work_dir.mkdir(parents=True, exist_ok=True)
    answers = [
        authority._chat_reply(json.dumps({"name": "diff_grids", "arguments": {"t": 0}})),
        authority._chat_reply(
            json.dumps(
                {
                    "name": "run_engine_on_transitions",
                    "arguments": {"code": authority._ENGINE_CODE},
                }
            )
        ),
    ]
    proposer = e3.LocalGGUFProposer(ffn_cpu_layers=0, mtp=False, max_tokens=1024, tries=1)
    proposer._ensure_server = lambda: True
    payloads: list[JsonDict] = []
    events: list[JsonDict] = []

    def request(req: Any, timeout: float | None = None) -> io.BytesIO:
        del timeout
        request_started = time.monotonic_ns()
        payload = json.loads(req.data)
        payloads.append(payload)
        response = answers.pop(0)
        request_ended = time.monotonic_ns()
        events.append(
            _span_event(
                "http_request_entry_exit",
                request_started,
                request_ended,
                detail={"request_index": len(payloads) - 1, "scripted_transport": True},
            )
        )
        return io.BytesIO(json.dumps(response).encode())

    environment = {
        "CARNOT_ARC_INDUCE_THINK": "0",
        "CARNOT_ARC_INDUCE_TOOL_LOOP": "selfparse",
        "CARNOT_ARC_INDUCE_TOOL_GRAMMAR": "1",
        "CARNOT_ARC_INDUCE_TOOL_TURNS": "2",
        "CARNOT_ARC_GENERATOR_SEED": "7384",
        "CARNOT_ARC_LLM_BACKEND": "llamacpp",
        "CARNOT_ARC_STALL_REFACTOR_LOOP": "0",
        "CARNOT_ARC_CEGIS_ACCEPT_SPLIT": "1",
        "CARNOT_ARC_STRUCTURED_NAV": "0",
        "CARNOT_ARC_LIVE_TTT": "0",
    }
    old_environment = {key: os.environ.get(key) for key in environment}
    old_disabled = os.environ.pop("CARNOT_ARC_DISABLE_INDUCTION", None)
    old_urlopen = urllib.request.urlopen
    old_e3_dir = e3.E3_DIR
    try:
        os.environ.update(environment)
        e3.E3_DIR = work_dir / "engines"
        urllib.request.urlopen = request
        reset_started = time.monotonic_ns()
        transitions = authority._transition_rows()
        reset_ended = time.monotonic_ns()
        events.append(
            _span_event(
                "environment_reset",
                reset_started,
                reset_ended,
                detail={"adapter_withheld": True, "transition_count": len(transitions)},
            )
        )
        policy_started = time.monotonic_ns()
        policy, factory = scored.build_disposable_submitted_policy("r11l", proposer)
        policy_ended = time.monotonic_ns()
        events.append(
            _span_event(
                "policy_initialization",
                policy_started,
                policy_ended,
                detail={"factory": factory["factory"], "policy_class": type(policy).__name__},
            )
        )
        policy.transitions = transitions
        policy.root_grid = transitions[0].grid
        policy.cell = 1
        policy._episode_transition_start = 0
        policy.program_synthesis_filter_enabled = False
        policy.active_probe_controller_enabled = False
        policy.think_arm_fallback_enabled = False
        policy.max_refinement_rounds = 1
        consume_started = time.monotonic_ns()
        policy._induce_and_plan()
        consume_ended = time.monotonic_ns()
        events.append(
            _span_event(
                "result_consumption",
                consume_started,
                consume_ended,
                detail={"http_requests": len(payloads), "plan_installed": bool(policy.plan)},
            )
        )
        action_started = time.monotonic_ns()
        move = policy._next_plan_move() if policy.plan else (None, None)
        action_ended = time.monotonic_ns()
        events.append(
            _span_event(
                "first_action",
                action_started,
                action_ended,
                detail={"action": move[0], "data": move[1]},
            )
        )
    finally:
        urllib.request.urlopen = old_urlopen
        e3.E3_DIR = old_e3_dir
        for key, value in old_environment.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        if old_disabled is not None:
            os.environ["CARNOT_ARC_DISABLE_INDUCTION"] = old_disabled
    later_has_result = len(payloads) == 2 and any(
        '"after": 1' in str(message.get("content", ""))
        for message in payloads[1].get("messages", [])
    )
    action_reached = move[0] not in {None, "RESET"}
    return {
        "factory": factory["factory"],
        "policy_class": type(policy).__name__,
        "adapter_disabled": factory["adapter_disabled"],
        "http_request_count": len(payloads),
        "tool_result_consumed_by_later_request": later_has_result,
        "subsequent_scored_policy_action": action_reached,
        "environment_action": {"action": move[0], "data": move[1]},
        "boundary_event_rows": events,
        "model_invoked": False,
        "solve_credited": False,
        "passed": bool(later_has_result and action_reached and len(payloads) == 2),
    }


def affected_manifest() -> validation_contract.AffectedManifest:
    """Name the exact files sent through the Exp7358 command planner."""

    return validation_contract.AffectedManifest(
        experiment_id=EXPERIMENT_ID,
        test_paths=(TEST_PATH.as_posix(),),
        changed_modules=(MODULE_PATH.as_posix(),),
        static_paths=(WRAPPER_PATH.as_posix(),),
    )


def build_validation_plan(root: Path, private_root: Path) -> list[validation_scope.CommandSpec]:
    """Build the actual Exp7358 plan, including command-local coverage data."""

    return validation_contract.build_command_plan(root, affected_manifest(), private_root)


def validate_validation_plan(
    root: Path, commands: Sequence[validation_scope.CommandSpec]
) -> list[str]:
    """Reject command drift before any validation child starts."""

    return validation_contract.validate_command_plan(root, affected_manifest(), commands)


def e2e_command_specs(root: Path, private: Path) -> list[validation_scope.CommandSpec]:
    """Return the exact E2E-009, E2E-010, and LLM-off smoke commands."""

    private.mkdir(parents=True, exist_ok=True)
    python = str(root / ".venv/bin/python")
    pytest = str(root / ".venv/bin/pytest")
    return [
        validation_scope.CommandSpec(
            "e2e_009",
            (
                pytest,
                "-n",
                "0",
                "-o",
                "addopts=",
                "--no-cov",
                f"--basetemp={private / 'e2e009'}",
                "tests/python/test_arc_induction_state_persistence.py",
                "-q",
            ),
            "E2E-009 cross-call ARC induction memory",
        ),
        validation_scope.CommandSpec(
            "e2e_010",
            (
                pytest,
                "-n",
                "0",
                "-o",
                "addopts=",
                "--no-cov",
                f"--basetemp={private / 'e2e010'}",
                "tests/python/test_arc_tool_grammar_transport.py",
                "-q",
            ),
            "E2E-010 local grammar tool transport",
        ),
        validation_scope.CommandSpec(
            "e2e_offline_smoke",
            (
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
                str(private / "r11l-twelve-action-smoke.json"),
            ),
            "E2E-009 LLM-disabled twelve-action environment smoke",
        ),
    ]


def _required_receipts_pass(receipts: Sequence[Mapping[str, Any]], names: Sequence[str]) -> bool:
    """Require one successful, non-timeout receipt for every named command."""

    counts = Counter(str(row.get("name")) for row in receipts)
    return all(
        counts[name] == 1
        and next(row for row in receipts if row.get("name") == name).get("passed") is True
        and next(row for row in receipts if row.get("name") == name).get("exit_code") == 0
        and next(row for row in receipts if row.get("name") == name).get("timed_out") is False
        for name in names
    )


def terminal_command_specs(root: Path, candidate: Path) -> list[validation_scope.CommandSpec]:
    """Build independent reduction and unchanged terminal safety readers."""

    python = str(root / ".venv/bin/python")
    return [
        validation_scope.CommandSpec(
            "independent_reducer",
            (
                python,
                "-u",
                "-c",
                (
                    "import json,sys; "
                    "from carnot.experiment_7384_v648_arc_invocation_boundary import "
                    "independent_reduce_file; "
                    "r=independent_reduce_file(sys.argv[1]); print(json.dumps(r,sort_keys=True)); "
                    "raise SystemExit(0 if r['matches_declared'] else 1)"
                ),
                str(candidate),
            ),
            "independent terminal reduction",
        ),
        validation_scope.CommandSpec(
            "adversarial_verify",
            (python, "-u", "scripts/adversarial_verify.py", str(candidate)),
            "unchanged adversarial verifier",
        ),
        validation_scope.CommandSpec(
            "verdict_row_consistency_strict",
            (python, "-u", "scripts/verdict_row_consistency_lint.py", "--strict", str(candidate)),
            "strict row-to-verdict consistency",
        ),
    ]


def collect_preconditions(root: Path) -> tuple[list[JsonDict], dict[str, JsonDict], JsonDict]:
    """Authenticate exact sources and the disqualified historical producer."""

    checks: list[JsonDict] = []
    hashes: dict[str, JsonDict] = {}
    for relative in (*INPUT_PATHS, MODULE_PATH, WRAPPER_PATH, TEST_PATH):
        path = root / relative
        available = path.is_file() and path.stat().st_size > 0
        checks.append(
            gate_row(
                f"source_bytes:{relative.as_posix()}",
                "precondition",
                "readable_nonempty_bytes",
                "readable_nonempty_bytes" if available else None,
                upstream=relative.as_posix(),
                artifact_field="bytes",
            )
        )
        if available:
            hashes[relative.as_posix()] = {
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
                "role": "historical_diagnostic_evidence"
                if relative == HISTORICAL_ARTIFACT
                else "current_input",
            }
    spec = (root / SPEC_PATH).read_text(encoding="utf-8") if (root / SPEC_PATH).is_file() else ""
    checks.append(
        gate_row(
            "driving_requirement",
            "precondition",
            True,
            "REQ-ARC-WMTE-7384" in spec,
            upstream=SPEC_PATH.as_posix(),
            artifact_field="REQ-ARC-WMTE-7384",
        )
    )
    exclusion = (
        (root / "ops/exclusion_manifest.yaml").read_text(encoding="utf-8")
        if (root / "ops/exclusion_manifest.yaml").is_file()
        else ""
    )
    checks.append(
        gate_row(
            "current_task_not_quarantined",
            "precondition",
            False,
            "experiment_id: 7384" in exclusion or "exp7384" in exclusion,
            upstream="ops/exclusion_manifest.yaml",
            artifact_field=EXPERIMENT_ID,
        )
    )
    historical = load_object(root / HISTORICAL_ARTIFACT)
    expected_historical = {
        "experiment_id": "exp7376-arc-outcomes",
        "status": "complete_disqualified_required_evidence",
        "verdict_class": "disqualified",
        "flagged_adversarial": True,
        "inference_substrate_class": "model_load_no_generation",
    }
    observed_historical = {key: historical.get(key) for key in expected_historical}
    checks.append(
        gate_row(
            "historical_producer_identity",
            "historical_diagnostic_only",
            expected_historical,
            observed_historical,
            upstream=HISTORICAL_ARTIFACT.as_posix(),
            artifact_field="producer_identity_class_and_flag",
        )
    )
    historical_receipt = {
        **observed_historical,
        "path": HISTORICAL_ARTIFACT.as_posix(),
        "sha256": hashes.get(HISTORICAL_ARTIFACT.as_posix(), {}).get("sha256"),
        "original_invocation_counts": deepcopy(historical.get("invocation_counts") or {}),
        "historical_only": True,
        "authorizes_current_readiness": False,
    }
    return checks, hashes, historical_receipt


def _field_principles(keys: Sequence[str]) -> dict[str, str]:
    """Explain why each ordinary top-level field is retained."""

    specific = {
        "schema": "A versioned schema lets strict readers reject incompatible records.",
        "status": "A terminal status prevents a bootstrap file from looking successful.",
        "run_date": "The requested date and UTC timestamps identify the actual execution.",
        "preconditions_checked": "Exact resource checks prevent invented dependent results.",
        "MODEL_SPECS": "An empty list states that no current LLM was required or loaded.",
        "model_invoked": "False separates scripted transport from a current LLM attempt.",
        "invocation_counts": "Typed zero counts prevent historical calls from becoming current calls.",
        "inference_substrate": "The CPU identity and lease explain where current computation ran.",
        "inference_substrate_class": "The closed class applies the correct no-model duration floor.",
        "execution_venue": "The closed host value keeps location separate from compute class.",
        "duration_s": "Measured monotonic time detects fabricated or padded execution.",
        "phase_spans": "Measured boundaries make slow or missing phases auditable.",
        "random_seed": "Frozen experiment and scripted transport seeds support repetition.",
        "reproducibility_checksum": "The checksum binds code, settings, sources, and raw rows.",
        "source_artifact_hashes": "Byte hashes preserve exact producer and source identity.",
        "rows": "Per-unit outcomes expose completion, censoring, costs, and failures.",
        "sample_size_budget": "Planned, attempted, completed, censored, and unstarted work stay distinct.",
        "acceptance_gate_results": "Expected, observed, operator, and pass state prevent gate reinterpretation.",
        "gate_check_summary": "Every failure names its upstream path and exact field.",
        "verifier_is_oracle": "The flag discloses when a formal evaluator defines truth.",
        "honest_verdict": "The terminal verdict states the completed diagnostic scope.",
        "verdict_class": "The closed class separates null plumbing evidence from promotion.",
        "flagged_adversarial": "A critical independent finding excludes the current producer.",
        "validation_receipts": "Executed argv, environment, timing, exit, and log hash support audit.",
        "repository_health": "Unrelated health remains separate from required affected checks.",
        "field_principles": "This map explains why every output field exists.",
        "promotion_score": "Zero forbids automatic rollout or publication from this diagnostic.",
        "arc_invocation_ready_score": "One means only that invocation plumbing passed its narrow gates.",
        "boundary_event_rows": "Monotonic spans locate the last observed and first missing boundary.",
        "root_cause_receipt": "The receipt ties the failing reproduction to the corrected accounting.",
        "solve_provenance": "Development proxy prevents scripted actions from receiving solve credit.",
        "generalization_activity": "The field records reusable hardening from the failed live attempt.",
    }
    return {
        key: specific.get(key, f"The {key} field preserves directly auditable experiment evidence.")
        for key in keys
    }


def build_terminal_artifact(
    *,
    run_date: str,
    started_at_utc: str,
    ended_at_utc: str,
    duration_s: float,
    preconditions: Sequence[Mapping[str, Any]],
    diagnosis: Mapping[str, Any],
    scripted_path: Mapping[str, Any],
    validation_receipts: Sequence[Mapping[str, Any]],
    source_hashes: Mapping[str, Any],
    phase_spans: Sequence[Mapping[str, Any]],
    terminal_lints_passed: bool,
    historical_model_receipt: Mapping[str, Any] | None = None,
) -> JsonDict:
    """Build one terminal record from raw diagnostic and validation evidence."""

    current_preconditions_passed = bool(preconditions) and all(
        row.get("passed") is True for row in preconditions
    )
    scoped_passed = _required_receipts_pass(validation_receipts, REQUIRED_CHECK_NAMES)
    e2e_passed = _required_receipts_pass(validation_receipts, REQUIRED_E2E_NAMES)
    accounting = dict(diagnosis.get("corrected_accounting") or {})
    accounting_exact = (
        bool(accounting)
        and (
            int(accounting.get("planned_units") or 0)
            == int(accounting.get("completed_units") or 0)
            + int(accounting.get("censored_units") or 0)
            + int(accounting.get("unstarted_units") or 0)
        )
        and (
            int(accounting.get("attempted_units") or 0)
            == int(accounting.get("completed_units") or 0)
            + int(accounting.get("censored_units") or 0)
        )
    )
    cause_corrected = bool(
        diagnosis.get("cause_reproduced")
        and (diagnosis.get("root_cause_receipt") or {}).get("corrected") is True
    )
    scored_action = bool(
        scripted_path.get("passed") is True
        and scripted_path.get("tool_result_consumed_by_later_request") is True
        and scripted_path.get("subsequent_scored_policy_action") is True
    )
    gates = [
        gate_row(
            "preconditions",
            "precondition",
            True,
            current_preconditions_passed,
            upstream="preconditions_checked",
            artifact_field="all_current_inputs_authenticated",
        ),
        gate_row(
            "historical_cause_reproduced_and_corrected",
            "completion",
            True,
            cause_corrected,
            upstream=HISTORICAL_RAW.as_posix(),
            artifact_field="root_cause_receipt.corrected",
        ),
        gate_row(
            "timeout_accounting_exact",
            "completion",
            True,
            accounting_exact,
            upstream="corrected_timeout_accounting",
            artifact_field="planned=completed+censored+unstarted",
        ),
        gate_row(
            "scripted_scored_action_reachable",
            "required_validation",
            True,
            scored_action,
            upstream="scripted_scored_path",
            artifact_field="subsequent_scored_policy_action",
        ),
        gate_row(
            "affected_validation",
            "required_validation",
            True,
            scoped_passed,
            upstream="validation_receipts",
            artifact_field="required_scoped_commands",
        ),
        gate_row(
            "e2e_plumbing",
            "required_validation",
            True,
            e2e_passed,
            upstream="validation_receipts",
            artifact_field="E2E-009/E2E-010/offline_smoke",
        ),
        gate_row(
            "terminal_safety",
            "safety",
            True,
            terminal_lints_passed,
            upstream="terminal_validation_receipts",
            artifact_field="all_terminal_readers_passed",
        ),
        gate_row(
            "current_llm_invocations",
            "safety",
            ZERO_INVOCATION_COUNTS,
            ZERO_INVOCATION_COUNTS,
            upstream="current_process",
            artifact_field="invocation_counts",
        ),
        gate_row(
            "scientific_efficacy",
            "scientific_efficacy",
            None,
            None,
            upstream="protocol",
            artifact_field="credited_solve",
            operator="is",
        ),
        gate_row(
            "automatic_promotion",
            "promotion",
            0,
            0,
            upstream="protocol",
            artifact_field="promotion_score",
        ),
    ]
    ready = int(all(row["passed"] for row in gates))
    if not current_preconditions_passed:
        verdict_class = "blocked"
        honest_verdict = "blocked_required_input_or_current_gate"
    elif ready:
        verdict_class = "null"
        honest_verdict = "complete_null_invocation_boundary_ready_no_efficacy_claim"
    else:
        verdict_class = "disqualified"
        honest_verdict = "complete_disqualified_required_evidence"
    receipt = dict(historical_model_receipt or {})
    historical = {
        key: deepcopy(receipt[key])
        for key in (
            "experiment_id",
            "status",
            "verdict_class",
            "flagged_adversarial",
            "inference_substrate_class",
            "path",
            "sha256",
            "sidecar_path",
            "sidecar_sha256",
            "historical_only",
            "authorizes_current_readiness",
        )
        if key in receipt
    }
    historical.setdefault("historical_only", True)
    historical.setdefault("authorizes_current_readiness", False)
    historical["scope"] = "historical"
    rows = []
    for raw_row in diagnosis.get("rows", []):
        if not isinstance(raw_row, Mapping):
            continue
        rows.append(
            {
                "unit_id": raw_row.get("episode_id"),
                "game": raw_row.get("game"),
                "seed": raw_row.get("seed"),
                "arm": "historical_exp7376_timeout_accounting",
                "disposition": raw_row.get("disposition"),
                "censored": raw_row.get("censored"),
                "metrics": {
                    "environment_action_count": int(raw_row.get("action_count") or 0),
                    "levels_completed": int(raw_row.get("levels") or 0),
                },
                "costs": {
                    "historical_request_attempt_count": int(
                        raw_row.get("generation_calls_attempted") or 0
                    ),
                    "current_llm_calls": 0,
                },
                "failure": raw_row.get("error"),
                "scope": "historical",
            }
        )
    rows.append(
        {
            "unit_id": "scripted_factory_path",
            "arm": "make_carnot_agent_scripted_transport",
            "disposition": "complete" if scripted_path.get("passed") else "complete_error",
            "censored": False,
            "metric": {
                "tool_result_consumed": scripted_path.get("tool_result_consumed_by_later_request"),
                "scored_action_reached": scripted_path.get("subsequent_scored_policy_action"),
            },
            "cost": {
                "current_llm_calls": 0,
                "scripted_http_requests": scripted_path.get("http_request_count"),
            },
            "failure": None if scripted_path.get("passed") else "scripted_scored_path_failed",
        }
    )
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "status": honest_verdict,
        "run_date": run_date,
        "started_at_utc": started_at_utc,
        "ended_at_utc": ended_at_utc,
        "duration_s": round(float(duration_s), 6),
        "preconditions_checked": [deepcopy(dict(row)) for row in preconditions],
        "MODEL_SPECS": [],
        "model_invoked": False,
        "invocation_counts": deepcopy(ZERO_INVOCATION_COUNTS),
        "historical_model_receipts": historical,
        "inference_substrate": {
            "value": "deterministic_qa_regression_no_llm",
            "principle": "Current work is CPU log reduction and scripted transport with no model load.",
            "cpu": platform.processor() or platform.machine(),
            "platform": platform.platform(),
            "jax_platform": os.environ.get("JAX_PLATFORMS", "cpu"),
            "resource_lease": "host_process_only_no_gpu_lease",
        },
        "inference_substrate_class": "no_model_load",
        "execution_venue": "host",
        "small_ebm_training": {
            "performed": False,
            "kind": "none",
            "current_llm_invocation": False,
        },
        "phase_spans": [deepcopy(dict(row)) for row in phase_spans],
        "random_seed": {"experiment": 7384, "scripted_transport": 7384, "resampling": None},
        "source_artifact_hashes": deepcopy(dict(source_hashes)),
        "rows": rows,
        "sample_size_budget": {
            **accounting,
            "stopping_rule": "historical aggregate timeout at 1800 seconds; current scripted fixture is bounded by two replies",
            "remaining_work": 0 if ready else 1,
        },
        "acceptance_gate_results": gates,
        "gate_check_summary": gate_summary(gates),
        "verifier_is_oracle": False,
        "honest_verdict": honest_verdict,
        "verdict_class": verdict_class,
        "flagged_adversarial": not terminal_lints_passed,
        "validation_receipts": [deepcopy(dict(row)) for row in validation_receipts],
        "required_check_names": [*REQUIRED_CHECK_NAMES, *REQUIRED_E2E_NAMES],
        "repository_health": {
            "observed_at_run_date": run_date,
            "status": "not_assessed_by_scoped_experiment",
            "unrelated_failures": [],
            "affects_required_checks": False,
        },
        "promotion_score": 0,
        "arc_invocation_ready_score": ready,
        "boundary_event_rows": [
            *deepcopy(list(diagnosis.get("boundary_event_rows") or [])),
            *deepcopy(list(scripted_path.get("boundary_event_rows") or [])),
        ],
        "last_confirmed_event": diagnosis.get("last_confirmed_event"),
        "first_missing_event": diagnosis.get("first_missing_event"),
        "root_cause_receipt": deepcopy(dict(diagnosis.get("root_cause_receipt") or {})),
        "scripted_scored_path": deepcopy(dict(scripted_path)),
        "solve_provenance": "development_proxy",
        "credited_solve": False,
        "generalization_activity": (
            "Reusable live-policy invocation and timeout-accounting hardening based on the "
            "real Exp7376 failed generalization attempt."
        ),
        "production_defaults_changed": False,
        "curated_arm_selection_changed": False,
        "active_research_roadmap_changed": False,
        "research_conductor_changed": False,
    }
    artifact["field_principles"] = _field_principles(
        [*artifact, "field_principles", "reproducibility_checksum"]
    )
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return artifact


def validate_artifact(value: Mapping[str, Any]) -> list[str]:
    """Cold-check identity, current-call truth, readiness, and checksum."""

    errors: list[str] = []
    if value.get("schema") != SCHEMA or value.get("experiment_id") != EXPERIMENT_ID:
        errors.append("schema or experiment identity mismatch")
    if value.get("milestone") != MILESTONE or value.get("run_date") != RUN_DATE:
        errors.append("milestone or run date mismatch")
    if value.get("MODEL_SPECS") != [] or value.get("model_invoked") is not False:
        errors.append("current model declaration mismatch")
    if value.get("invocation_counts") != ZERO_INVOCATION_COUNTS:
        errors.append("current invocation counts mismatch")
    if value.get("inference_substrate_class") != "no_model_load":
        errors.append("inference substrate class mismatch")
    if value.get("execution_venue") != "host":
        errors.append("execution venue mismatch")
    if value.get("promotion_score") != 0 or value.get("credited_solve") is not False:
        errors.append("unsafe promotion or solve credit")
    gates = value.get("acceptance_gate_results") or []
    expected_ready = int(
        value.get("verdict_class") == "null"
        and bool(gates)
        and all(row.get("passed") is True for row in gates)
        and value.get("flagged_adversarial") is False
    )
    if value.get("arc_invocation_ready_score") != expected_ready:
        errors.append("readiness reduction mismatch")
    if value.get("verdict_class") in {"blocked", "disqualified", "partial"} and value.get(
        "arc_invocation_ready_score"
    ):
        errors.append("unsafe readiness on non-ready artifact")
    principles = value.get("field_principles")
    if not isinstance(principles, Mapping) or not set(value) <= set(principles):
        errors.append("field principles mismatch")
    if value.get("reproducibility_checksum") != artifact_checksum(value):
        errors.append("checksum mismatch")
    return errors


def independent_reduce_file(path: str | Path) -> JsonDict:
    """Reload a candidate and independently recompute its declared readiness."""

    artifact = load_object(Path(path))
    errors = validate_artifact(artifact)
    accounting = dict(artifact.get("sample_size_budget") or {})
    accounting_exact = int(accounting.get("planned_units") or 0) == int(
        accounting.get("completed_units") or 0
    ) + int(accounting.get("censored_units") or 0) + int(accounting.get("unstarted_units") or 0)
    declared = int(artifact.get("arc_invocation_ready_score") or 0)
    reduced = int(not errors and accounting_exact)
    return {
        "declared_arc_invocation_ready_score": declared,
        "reduced_arc_invocation_ready_score": reduced,
        "accounting_exact": accounting_exact,
        "validation_errors": errors,
        "matches_declared": declared == reduced,
    }


def _phase(phase: str, phase_started: float, run_started: float, units: int) -> JsonDict:
    """Measure one disjoint phase from monotonic process time."""

    now = time.monotonic()
    return {
        "phase": phase,
        "started_elapsed_s": round(phase_started - run_started, 6),
        "ended_elapsed_s": round(now - run_started, 6),
        "duration_s": round(now - phase_started, 6),
        "completed_units": units,
        "checkpoint_at_utc": utc_now(),
    }


def run_experiment(
    args: argparse.Namespace,
) -> JsonDict:  # pragma: no cover - actual experiment entrypoint.
    """Execute CPU diagnosis, required checks, terminal readers, and atomic write."""

    started = time.monotonic()
    started_at = utc_now()
    phases: list[JsonDict] = []
    progress(started, "startup", "begin", run_date=args.date)

    phase_started = time.monotonic()
    progress(started, "read", "before_preconditions")
    preconditions, source_hashes, historical_receipt = collect_preconditions(REPO_ROOT)
    phases.append(_phase("read", phase_started, started, len(preconditions)))
    progress(
        started,
        "read",
        "after_preconditions",
        passed=all(row["passed"] for row in preconditions),
    )

    phase_started = time.monotonic()
    progress(started, "evaluate", "before_historical_reduction")
    diagnosis = diagnose_historical_exp7376(REPO_ROOT)
    progress(started, "evaluate", "after_historical_reduction", cause=diagnosis["cause_reproduced"])
    historical_sidecar = REPO_ROOT / RAW_DIR / "historical_exp7376_invocation_receipts.json"
    atomic_json(
        historical_sidecar,
        {
            "schema": "carnot.exp7384.historical_invocation_sidecar.v1",
            "scope": "historical",
            "counts_as_current_invocation": False,
            "producer": historical_receipt,
            "raw_event_reduction": diagnosis["historical_invocations"],
        },
    )
    historical_receipt = {
        key: value
        for key, value in historical_receipt.items()
        if key != "original_invocation_counts"
    }
    historical_receipt.update(
        {
            "sidecar_path": historical_sidecar.relative_to(REPO_ROOT).as_posix(),
            "sidecar_sha256": sha256_file(historical_sidecar),
        }
    )
    source_hashes[historical_sidecar.relative_to(REPO_ROOT).as_posix()] = {
        "path": str(historical_sidecar.resolve()),
        "sha256": sha256_file(historical_sidecar),
        "role": "historical_diagnostic_sidecar",
        "historical_only": True,
    }
    progress(started, "evaluate", "before_scripted_scored_path")
    scripted = run_scripted_factory_path(REPO_ROOT / RAW_DIR / "scripted-path")
    progress(started, "evaluate", "after_scripted_scored_path", passed=scripted["passed"])
    raw_evidence = REPO_ROOT / RAW_DIR / "independent_reduction_input.json"
    atomic_json(raw_evidence, {"diagnosis": diagnosis, "scripted_scored_path": scripted})
    source_hashes[raw_evidence.relative_to(REPO_ROOT).as_posix()] = {
        "path": str(raw_evidence.resolve()),
        "sha256": sha256_file(raw_evidence),
        "role": "current_raw_evidence",
    }
    phases.append(_phase("evaluate", phase_started, started, len(diagnosis.get("rows", [])) + 1))

    private = Path(tempfile.mkdtemp(prefix="exp7384-validation-"))
    phase_started = time.monotonic()
    progress(started, "validate", "before_scoped_subprocesses")
    plan = build_validation_plan(REPO_ROOT, private / "scoped")
    plan_errors = validate_validation_plan(REPO_ROOT, plan)
    if plan_errors:
        raise RuntimeError(f"validation plan drift: {plan_errors}")
    planned = [
        validation_contract.PlannedCommand(command, "required_validation", True) for command in plan
    ]
    scoped_receipts = validation_contract.run_categorized_commands(
        REPO_ROOT,
        planned,
        log_dir=REPO_ROOT / RAW_DIR / "validation" / "scoped",
        heartbeat_s=60.0,
    )
    progress(started, "validate", "after_scoped_subprocesses", completed_units=len(scoped_receipts))
    progress(started, "validate", "before_e2e_subprocesses")
    e2e_receipts = validation_scope.run_commands(
        REPO_ROOT,
        e2e_command_specs(REPO_ROOT, private / "e2e"),
        log_dir=REPO_ROOT / RAW_DIR / "validation" / "e2e",
        extra_env={"CARNOT_ARC_DISABLE_INDUCTION": "1"},
        heartbeat_s=60.0,
    )
    progress(started, "validate", "after_e2e_subprocesses", completed_units=len(e2e_receipts))
    phases.append(
        _phase("validate", phase_started, started, len(scoped_receipts) + len(e2e_receipts))
    )

    validation_receipts = [*scoped_receipts, *e2e_receipts]
    candidate_path = REPO_ROOT / RAW_DIR / "measured_terminal_candidate.json"
    candidate = build_terminal_artifact(
        run_date=args.date,
        started_at_utc=started_at,
        ended_at_utc=utc_now(),
        duration_s=time.monotonic() - started,
        preconditions=preconditions,
        diagnosis=diagnosis,
        scripted_path=scripted,
        validation_receipts=validation_receipts,
        source_hashes=source_hashes,
        phase_spans=phases,
        terminal_lints_passed=True,
        historical_model_receipt=historical_receipt,
    )
    atomic_json(candidate_path, candidate)
    progress(started, "validate", "before_terminal_subprocesses")
    terminal_receipts = validation_scope.run_commands(
        REPO_ROOT,
        terminal_command_specs(REPO_ROOT, candidate_path),
        log_dir=REPO_ROOT / RAW_DIR / "validation" / "terminal",
        heartbeat_s=60.0,
    )
    progress(
        started, "validate", "after_terminal_subprocesses", completed_units=len(terminal_receipts)
    )
    terminal_passed = all(row.get("passed") is True for row in terminal_receipts)
    phases.append(_phase("validate", phase_started, started, len(terminal_receipts)))

    phase_started = time.monotonic()
    final_receipts = [*validation_receipts, *terminal_receipts]
    artifact = build_terminal_artifact(
        run_date=args.date,
        started_at_utc=started_at,
        ended_at_utc=utc_now(),
        duration_s=time.monotonic() - started,
        preconditions=preconditions,
        diagnosis=diagnosis,
        scripted_path=scripted,
        validation_receipts=final_receipts,
        source_hashes=source_hashes,
        phase_spans=[*phases, _phase("write", phase_started, started, 1)],
        terminal_lints_passed=terminal_passed,
        historical_model_receipt=historical_receipt,
    )
    errors = validate_artifact(artifact)
    if errors:
        raise RuntimeError(f"terminal artifact validation failed: {errors}")
    atomic_json(candidate_path, artifact)
    atomic_json(REPO_ROOT / RESULT_PATH, artifact)
    progress(
        started,
        "write",
        "after_atomic_publication",
        path=RESULT_PATH.as_posix(),
        ready=artifact["arc_invocation_ready_score"],
    )
    return artifact


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the fixed execution date accepted by the thin entrypoint."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", required=True, choices=[RUN_DATE])
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - thin CLI handoff.
    """Run the diagnostic and return success only after a terminal artifact exists."""

    artifact = run_experiment(parse_args(argv))
    return 0 if str(artifact.get("status", "")).startswith(("complete_", "blocked_")) else 1
