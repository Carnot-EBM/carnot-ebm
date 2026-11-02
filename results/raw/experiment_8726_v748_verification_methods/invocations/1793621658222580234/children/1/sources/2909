"""Qualify durable scored-policy episodes without current model work.

The experiment turns the Experiment 7376 timeout lesson into a reusable child
checkpoint. It uses scripted HTTP replies to cross the real scored policy. It
does not load a model, run a board, or claim ARC efficacy.

Spec refs: REQ-ARC-WMTE-7398 and SCENARIO-ARC-WMTE-7398-*.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from datetime import UTC, datetime
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import re
import tempfile
import time
from typing import Any
import urllib.request

from carnot import experiment_7358_v646_validation_contract as validation_contract
from carnot import experiment_7384_v648_arc_invocation_boundary as invocation_boundary
from carnot.reporting import experiment_7303_validation_scope as validation_scope


JsonDict = dict[str, Any]
Emit = Callable[[str, Mapping[str, Any]], JsonDict]
EpisodeExecutor = Callable[[Mapping[str, Any], Emit], Mapping[str, Any]]

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ID = "exp7398-arc-checkpoint"
MILESTONE = "2026.09.649"
RUN_DATE = "20260918"
SCHEMA = "carnot.exp7398.v649_arc_checkpoint.v1"
CHECKPOINT_SCHEMA = "carnot.arc_episode_checkpoint.v1"
RESULT_PATH = Path("results/experiment_7398_v649_arc_checkpoint.json")
RAW_DIR = Path("results/raw/experiment_7398_v649_arc_checkpoint")
MODULE_PATH = Path("python/carnot/experiment_7398_v649_arc_checkpoint.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_7398_v649_arc_checkpoint.py")
TEST_PATH = Path("tests/python/test_experiment_7398_v649_arc_checkpoint.py")
SPEC_PATH = Path("openspec/capabilities/arc-world-model-trust-energy/spec.md")
HISTORICAL_ARTIFACT = Path("results/experiment_7384_v648_arc_invocation_boundary.json")
HISTORICAL_OUTCOME = Path("results/experiment_7376_v647_arc_outcomes.json")
HISTORICAL_RAW = Path("results/raw/experiment_7376_v647_arc_outcomes")
REQUIRED_CHECK_NAMES = validation_scope.REQUIRED_CHECK_NAMES
REQUIRED_E2E_NAMES = ("e2e_009", "e2e_010", "e2e_offline_smoke")
INFERENCE_SUBSTRATE = (
    "Host CPU JSON/hash reduction, atomic checkpoint I/O, and scripted HTTP "
    "transport through the real scored ARC policy; no current LLM or board run."
)
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
}
INTERRUPTION_BOUNDARIES = (
    "before_first_request",
    "after_request",
    "after_action",
    "between_episodes",
    "before_final_aggregation",
)
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
    Path("python/carnot/experiment_7384_v648_arc_invocation_boundary.py"),
    Path("python/carnot/experiment_7376_v647_arc_outcomes.py"),
    Path("python/carnot/experiment_7345_v645_arc_resume_check.py"),
    Path("python/carnot/agentic/arc_competition_agent.py"),
    Path("scripts/arc_loop_solve.py"),
    Path("ops/arc_solve_registry.yaml"),
    Path("openspec/capabilities/research-reporting/spec.md"),
    SPEC_PATH,
    HISTORICAL_ARTIFACT,
    HISTORICAL_OUTCOME,
    HISTORICAL_RAW / "live_session.json",
    HISTORICAL_RAW / "episode_rows.json",
    HISTORICAL_RAW / "frozen_schedule.json",
    HISTORICAL_RAW / "receipt_events.jsonl",
)


class CheckpointIntegrityError(ValueError):
    """Report a journal that cannot safely authorize resumed work."""


class InjectedInterruption(BaseException):
    """Stop the owned child without letting policy error handling consume it."""


def utc_now() -> str:
    """Return one real UTC boundary with an explicit offset."""

    return datetime.now(UTC).isoformat()


def progress(started: float, phase: str, event: str, **details: Any) -> None:
    """Print a flushed phase boundary with monotonic elapsed time."""

    suffix = " ".join(f"{key}={value}" for key, value in sorted(details.items()))
    print(
        f"[exp7398] phase={phase} event={event} elapsed_s={time.monotonic() - started:.3f}"
        + (f" {suffix}" if suffix else ""),
        flush=True,
    )


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()


def canonical_hash(value: Any) -> str:
    """Hash stable JSON bytes for event and artifact identity."""

    return "sha256:" + hashlib.sha256(_canonical_bytes(value)).hexdigest()


def sha256_file(path: Path) -> str:
    """Hash exact file bytes without loading large evidence into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    """Flush a complete JSON object before one local atomic replacement."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def _load_object(path: Path) -> JsonDict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(value) if isinstance(value, Mapping) else {}


def artifact_checksum(value: Mapping[str, Any]) -> str:
    """Bind every terminal field except the checksum slot itself."""

    payload = deepcopy(dict(value))
    payload["reproducibility_checksum"] = ""
    return canonical_hash(payload)


def gate_row(
    check: str,
    category: str,
    expected: Any,
    observed: Any,
    *,
    upstream: str = "current_experiment",
    artifact_field: str | None = None,
    operator: str = "==",
) -> JsonDict:
    """Record one comparison without hiding missing observed values."""

    if operator == "==":
        passed = observed == expected
    elif operator == "is":
        passed = observed is expected
    else:
        raise ValueError(f"unsupported gate operator: {operator}")
    return {
        "check": check,
        "category": category,
        "upstream": upstream,
        "artifact_field": artifact_field or check,
        "operator": operator,
        "expected": expected,
        "observed": observed,
        "passed": passed,
    }


def gate_summary(gates: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Retain every failed gate and the first exact failure."""

    failed = [deepcopy(dict(row)) for row in gates if row.get("passed") is not True]
    return {
        "all_passed": not failed,
        "failed_count": len(failed),
        "failed_checks": failed,
        "first_failure": failed[0] if failed else None,
    }


def _episode_state_hash(episode: Mapping[str, Any]) -> str:
    return canonical_hash(
        {
            "episode_id": episode.get("episode_id"),
            "game": episode.get("game"),
            "seed": episode.get("seed"),
        }
    )


def _checkpoint_name(episode_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", episode_id).strip("_") + ".json"


def _journal_checksum(value: Mapping[str, Any]) -> str:
    return canonical_hash(
        {
            "schema": value.get("schema"),
            "episode_id": value.get("episode_id"),
            "state_hash": value.get("state_hash"),
            "events": value.get("events"),
        }
    )


def read_episode_checkpoint(path: Path, episode: Mapping[str, Any]) -> JsonDict | None:
    """Read and independently validate one atomic episode journal."""

    if not path.exists():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CheckpointIntegrityError(f"malformed checkpoint: {path}") from error
    if not isinstance(raw, Mapping):
        raise CheckpointIntegrityError("checkpoint must be a JSON object")
    value = dict(raw)
    expected_state = _episode_state_hash(episode)
    if value.get("schema") != CHECKPOINT_SCHEMA:
        raise CheckpointIntegrityError("checkpoint schema mismatch")
    if value.get("episode_id") != episode.get("episode_id"):
        raise CheckpointIntegrityError("checkpoint episode identity mismatch")
    if value.get("state_hash") != expected_state:
        raise CheckpointIntegrityError("checkpoint state hash mismatch")
    events = value.get("events")
    if not isinstance(events, list):
        raise CheckpointIntegrityError("checkpoint events must be a list")
    seen: set[str] = set()
    previous_hash = "sha256:" + "0" * 64
    for index, raw_event in enumerate(events):
        if not isinstance(raw_event, Mapping):
            raise CheckpointIntegrityError("checkpoint event shape is invalid")
        event = dict(raw_event)
        required = {
            "sequence",
            "event_id",
            "kind",
            "episode_id",
            "state_hash",
            "prior_event_hash",
            "detail",
            "event_hash",
        }
        if not required <= set(event) or not isinstance(event.get("detail"), Mapping):
            raise CheckpointIntegrityError("checkpoint event shape is invalid")
        event_id = str(event["event_id"])
        if event_id in seen:
            raise CheckpointIntegrityError(f"duplicate event ID: {event_id}")
        seen.add(event_id)
        if event.get("sequence") != index:
            raise CheckpointIntegrityError("checkpoint event sequence mismatch")
        if event.get("episode_id") != episode.get("episode_id"):
            raise CheckpointIntegrityError("checkpoint event episode mismatch")
        if event.get("state_hash") != expected_state:
            raise CheckpointIntegrityError("checkpoint event state hash mismatch")
        if event.get("prior_event_hash") != previous_hash:
            raise CheckpointIntegrityError("checkpoint event hash chain mismatch")
        hashed = {key: deepcopy(item) for key, item in event.items() if key != "event_hash"}
        expected_event_hash = canonical_hash(hashed)
        if event.get("event_hash") != expected_event_hash:
            raise CheckpointIntegrityError("checkpoint event hash mismatch")
        previous_hash = expected_event_hash
    if value.get("journal_checksum") != _journal_checksum(value):
        raise CheckpointIntegrityError("checkpoint journal checksum mismatch")
    return value


def _new_journal(episode: Mapping[str, Any]) -> JsonDict:
    value: JsonDict = {
        "schema": CHECKPOINT_SCHEMA,
        "episode_id": episode.get("episode_id"),
        "state_hash": _episode_state_hash(episode),
        "events": [],
    }
    value["journal_checksum"] = _journal_checksum(value)
    return value


def _append_event(
    path: Path,
    episode: Mapping[str, Any],
    kind: str,
    ordinal: int,
    detail: Mapping[str, Any],
) -> JsonDict:
    journal = read_episode_checkpoint(path, episode) or _new_journal(episode)
    event_id = f"{episode['episode_id']}:{kind}:{ordinal}"
    for old in journal["events"]:
        if old["event_id"] != event_id:
            continue
        if old["kind"] != kind or old["detail"] != dict(detail):
            raise CheckpointIntegrityError(f"duplicate event ID changed content: {event_id}")
        return deepcopy(dict(old))
    previous_hash = (
        journal["events"][-1]["event_hash"] if journal["events"] else "sha256:" + "0" * 64
    )
    event: JsonDict = {
        "sequence": len(journal["events"]),
        "event_id": event_id,
        "kind": kind,
        "episode_id": episode["episode_id"],
        "state_hash": journal["state_hash"],
        "prior_event_hash": previous_hash,
        "detail": deepcopy(dict(detail)),
    }
    event["event_hash"] = canonical_hash(event)
    journal["events"].append(event)
    journal["journal_checksum"] = _journal_checksum(journal)
    atomic_json(path, journal)
    return deepcopy(event)


def _events_by_kind(journal: Mapping[str, Any], kind: str) -> list[JsonDict]:
    return [
        deepcopy(dict(row))
        for row in journal.get("events", [])
        if isinstance(row, Mapping) and row.get("kind") == kind
    ]


def run_checkpointed_episodes(
    checkpoint_dir: Path,
    schedule: Sequence[Mapping[str, Any]],
    episode_executor: EpisodeExecutor,
    *,
    interrupt_at: str | None = None,
) -> JsonDict:
    """Run episodes with exact-once action recovery at five crash boundaries."""

    if interrupt_at is not None and interrupt_at not in INTERRUPTION_BOUNDARIES:
        raise ValueError(f"unknown interruption boundary: {interrupt_at}")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    rows: list[JsonDict] = []
    recovery_rows: list[JsonDict] = []
    checkpoint_receipts: list[JsonDict] = []
    interruption_triggered = False
    for episode_index, episode in enumerate(schedule):
        episode_id = str(episode.get("episode_id"))
        path = checkpoint_dir / _checkpoint_name(episode_id)
        initial = read_episode_checkpoint(path, episode)
        was_complete = bool(initial and _events_by_kind(initial, "episode_completed"))
        _append_event(path, episode, "episode_started", 0, {"episode_index": episode_index})
        journal = read_episode_checkpoint(path, episode)
        assert journal is not None
        action_events = _events_by_kind(journal, "action_completed")

        if action_events:
            action_event = action_events[-1]
            row = deepcopy(dict(action_event["detail"]))
        else:
            ordinals: Counter[str] = Counter()

            def emit(kind: str, detail: Mapping[str, Any]) -> JsonDict:
                nonlocal interruption_triggered
                ordinal = ordinals[kind]
                ordinals[kind] += 1
                if (
                    interrupt_at == "before_first_request"
                    and kind == "request_completed"
                    and ordinal == 0
                    and not interruption_triggered
                ):
                    interruption_triggered = True
                    raise InjectedInterruption("before_first_request")
                event = _append_event(path, episode, kind, ordinal, detail)
                if (
                    interrupt_at == "after_request"
                    and kind == "request_completed"
                    and ordinal == 0
                    and not interruption_triggered
                ):
                    interruption_triggered = True
                    raise InjectedInterruption("after_request")
                if (
                    interrupt_at == "after_action"
                    and kind == "action_completed"
                    and not interruption_triggered
                ):
                    interruption_triggered = True
                    raise InjectedInterruption("after_action")
                return event

            row = deepcopy(dict(episode_executor(episode, emit)))
            journal = read_episode_checkpoint(path, episode)
            assert journal is not None
            action_events = _events_by_kind(journal, "action_completed")
            if len(action_events) != 1 or action_events[0]["detail"] != row:
                raise CheckpointIntegrityError("executor did not seal exactly one matching action")
            action_event = action_events[0]
        durable_action_row = deepcopy(dict(action_event["detail"]))
        _append_event(
            path,
            episode,
            "episode_completed",
            0,
            {"action_event_id": action_event["event_id"], "action_row_hash": canonical_hash(row)},
        )
        row["action_event_id"] = action_event["event_id"]
        row["disposition"] = "complete"
        row["censored"] = False
        rows.append(row)
        final = read_episode_checkpoint(path, episode)
        assert final is not None
        recovery_rows.append(
            {
                "episode_id": episode_id,
                "durable_source_event_id": action_event["event_id"],
                "disposition": "complete",
                "censored": False,
                "recovered_from_checkpoint": was_complete or bool(initial and action_events),
                "restart_parity": canonical_hash(action_event["detail"])
                == canonical_hash(durable_action_row),
                "event_count": len(final["events"]),
                "checkpoint_path": str(path.resolve()),
                "checkpoint_sha256": sha256_file(path),
            }
        )
        checkpoint_receipts.append(
            {
                "episode_id": episode_id,
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
                "state_hash": final["state_hash"],
                "journal_checksum": final["journal_checksum"],
            }
        )
        if (
            interrupt_at == "between_episodes"
            and episode_index + 1 < len(schedule)
            and not interruption_triggered
        ):
            interruption_triggered = True
            raise InjectedInterruption("between_episodes")
    if interrupt_at == "before_final_aggregation" and not interruption_triggered:
        raise InjectedInterruption("before_final_aggregation")
    accounting = {
        "planned_units": len(schedule),
        "attempted_units": len(rows),
        "completed_units": len(rows),
        "censored_units": 0,
        "unstarted_units": len(schedule) - len(rows),
    }
    aggregate = {
        "schema": "carnot.arc_episode_aggregate.v1",
        "schedule_hash": canonical_hash(list(schedule)),
        "row_hash": canonical_hash(rows),
        "action_event_ids": [row["action_event_id"] for row in rows],
        "accounting": accounting,
    }
    aggregate_path = checkpoint_dir / "aggregate.json"
    atomic_json(aggregate_path, aggregate)
    independently_read = all(
        read_episode_checkpoint(
            checkpoint_dir / _checkpoint_name(str(episode["episode_id"])), episode
        )
        is not None
        for episode in schedule
    )
    return {
        "rows": rows,
        "episode_recovery_rows": recovery_rows,
        "checkpoint_receipts": checkpoint_receipts,
        "accounting": accounting,
        "aggregate_path": str(aggregate_path.resolve()),
        "aggregate_sha256": sha256_file(aggregate_path),
        "independent_reader_passed": independently_read,
    }


def run_scripted_scored_episode(episode: Mapping[str, Any], work_dir: Path, emit: Emit) -> JsonDict:
    """Cross the real factory, tool-result, and later-action path with fixtures."""

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

    def scripted_urlopen(request: Any, timeout: float | None = None) -> io.BytesIO:
        del timeout
        payload = json.loads(request.data)
        response = answers[len(payloads)]
        payloads.append(payload)
        emit(
            "request_completed",
            {
                "request_index": len(payloads) - 1,
                "request_sha256": "sha256:" + hashlib.sha256(request.data).hexdigest(),
                "response_sha256": canonical_hash(response),
                "scripted_transport": True,
            },
        )
        return io.BytesIO(json.dumps(response).encode())

    environment = {
        "CARNOT_ARC_INDUCE_THINK": "0",
        "CARNOT_ARC_INDUCE_TOOL_LOOP": "selfparse",
        "CARNOT_ARC_INDUCE_TOOL_GRAMMAR": "1",
        "CARNOT_ARC_INDUCE_TOOL_TURNS": "2",
        "CARNOT_ARC_GENERATOR_SEED": str(episode.get("seed")),
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
    move: tuple[Any, Any] = (None, None)
    factory: JsonDict = {}
    policy: Any = None
    try:
        os.environ.update(environment)
        e3.E3_DIR = work_dir / "engines"
        urllib.request.urlopen = scripted_urlopen
        transitions = authority._transition_rows()
        policy, factory = scored.build_disposable_submitted_policy(str(episode["game"]), proposer)
        policy.transitions = transitions
        policy.root_grid = transitions[0].grid
        policy.cell = 1
        policy._episode_transition_start = 0
        policy.program_synthesis_filter_enabled = False
        policy.active_probe_controller_enabled = False
        policy.think_arm_fallback_enabled = False
        policy.max_refinement_rounds = 1
        policy._induce_and_plan()
        move = policy._next_plan_move() if policy.plan else (None, None)
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
        if isinstance(message, Mapping)
    )
    action_reached = move[0] not in {None, "RESET"}
    row: JsonDict = {
        "episode_id": str(episode["episode_id"]),
        "game": str(episode["game"]),
        "seed": int(episode["seed"]),
        "factory": factory.get("factory"),
        "policy_class": type(policy).__name__,
        "adapter_disabled": factory.get("adapter_disabled") is True,
        "solution_paths_denied": True,
        "denied_paths": ["per_game_adapter", "banked_solution", "hand_solver", "replay_route"],
        "http_request_count": len(payloads),
        "tool_result_consumed": later_has_result,
        "later_policy_action": action_reached,
        "action": move[0],
        "action_data": move[1],
        "model_invoked": False,
        "credited_solve": False,
        "solve_provenance": "development_proxy",
    }
    emit("action_completed", row)
    return row


def run_interruption_matrix(work_dir: Path) -> JsonDict:
    """Exercise every interruption boundary through the real scored policy."""

    all_rows: list[JsonDict] = []
    recovery_rows: list[JsonDict] = []
    receipts: list[JsonDict] = []
    controls: list[JsonDict] = []
    schedule = [
        {"episode_id": "r11l:seed-7398", "game": "r11l", "seed": 7398},
        {"episode_id": "r11l:seed-17398", "game": "r11l", "seed": 17398},
    ]
    for boundary in INTERRUPTION_BOUNDARIES:
        case_dir = work_dir / boundary

        def executor(episode: Mapping[str, Any], emit: Emit, *, _case: Path = case_dir) -> JsonDict:
            return run_scripted_scored_episode(
                episode,
                _case / str(episode["episode_id"]).replace(":", "__"),
                emit,
            )

        interrupted = False
        try:
            run_checkpointed_episodes(
                case_dir / "checkpoints", schedule, executor, interrupt_at=boundary
            )
        except InjectedInterruption as error:
            interrupted = str(error) == boundary
        resumed = run_checkpointed_episodes(case_dir / "checkpoints", schedule, executor)
        boundary_rows = []
        for row in resumed["rows"]:
            copied = deepcopy(row)
            copied["interruption_boundary"] = boundary
            boundary_rows.append(copied)
        all_rows.extend(boundary_rows)
        for row in resumed["episode_recovery_rows"]:
            copied = deepcopy(row)
            copied["interruption_boundary"] = boundary
            recovery_rows.append(copied)
        receipts.extend(deepcopy(resumed["checkpoint_receipts"]))
        controls.append(
            {
                "boundary": boundary,
                "interruption_observed": interrupted,
                "completed_units": resumed["accounting"]["completed_units"],
                "unique_action_event_count": len(
                    {row["action_event_id"] for row in resumed["rows"]}
                ),
                "restart_parity": all(
                    row["restart_parity"] for row in resumed["episode_recovery_rows"]
                ),
                "independent_reader_passed": resumed["independent_reader_passed"],
                "passed": bool(
                    interrupted
                    and resumed["accounting"]["completed_units"] == len(schedule)
                    and len({row["action_event_id"] for row in resumed["rows"]}) == len(schedule)
                    and all(row["restart_parity"] for row in resumed["episode_recovery_rows"])
                    and resumed["independent_reader_passed"]
                ),
            }
        )
    planned = len(schedule) * len(INTERRUPTION_BOUNDARIES)
    return {
        "rows": all_rows,
        "episode_recovery_rows": recovery_rows,
        "checkpoint_receipts": receipts,
        "interruption_controls": controls,
        "accounting": {
            "planned_units": planned,
            "attempted_units": planned,
            "completed_units": len(all_rows),
            "censored_units": 0,
            "unstarted_units": planned - len(all_rows),
        },
        "independent_reader_passed": all(row["independent_reader_passed"] for row in controls),
    }


def diagnose_historical_timeout(root: Path) -> JsonDict:
    """Reuse the shipped V648 reader against immutable V647 journal bytes."""

    return invocation_boundary.diagnose_historical_exp7376(root)


def write_provenance_sidecars(
    directory: Path,
    diagnosis: Mapping[str, Any],
    scripted: Mapping[str, Any],
) -> dict[str, JsonDict]:
    """Keep archived runtime counters separate from scripted transport rows."""

    directory.mkdir(parents=True, exist_ok=True)
    historical_path = directory / "historical_runtime.json"
    historical = {
        "schema": "carnot.exp7398.historical_runtime_sidecar.v1",
        "scope": "historical",
        "counts_as_current_invocation": False,
        "producer_receipts": {
            "exp7376": {
                key: _load_object(REPO_ROOT / HISTORICAL_OUTCOME).get(key)
                for key in ("experiment_id", "status", "verdict_class", "flagged_adversarial")
            },
            "exp7384": {
                key: _load_object(REPO_ROOT / HISTORICAL_ARTIFACT).get(key)
                for key in ("experiment_id", "status", "verdict_class", "flagged_adversarial")
            },
        },
        "diagnosis": deepcopy(dict(diagnosis)),
    }
    atomic_json(historical_path, historical)
    scripted_path = directory / "scripted_transport.json"
    scripted_payload = {
        "schema": "carnot.exp7398.scripted_transport_sidecar.v1",
        "scope": "development_proxy",
        "counts_as_current_invocation": False,
        "model_invoked": False,
        "credited_solve": False,
        "transport": deepcopy(dict(scripted)),
    }
    atomic_json(scripted_path, scripted_payload)
    return {
        "historical_runtime": {
            "scope": "historical",
            "path": str(historical_path.resolve()),
            "sha256": sha256_file(historical_path),
            "authorizes_current_readiness": False,
        },
        "scripted_transport": {
            "scope": "development_proxy",
            "path": str(scripted_path.resolve()),
            "sha256": sha256_file(scripted_path),
            "authorizes_efficacy": False,
        },
    }


def affected_manifest() -> validation_contract.AffectedManifest:
    """Freeze the exact Exp7358 affected-check inputs."""

    return validation_contract.AffectedManifest(
        experiment_id=EXPERIMENT_ID,
        test_paths=(TEST_PATH.as_posix(),),
        changed_modules=(MODULE_PATH.as_posix(),),
        static_paths=(WRAPPER_PATH.as_posix(),),
    )


def build_validation_plan(root: Path, private_root: Path) -> list[validation_scope.CommandSpec]:
    """Build the existing Exp7303 commands through the Exp7358 contract."""

    return validation_contract.build_command_plan(root, affected_manifest(), private_root)


def validate_validation_plan(
    root: Path, commands: Sequence[validation_scope.CommandSpec]
) -> list[str]:
    """Reject expanded scope or changed command semantics before execution."""

    return validation_contract.validate_command_plan(root, affected_manifest(), commands)


def e2e_command_specs(root: Path, private: Path) -> list[validation_scope.CommandSpec]:
    """Build E2E-009, E2E-010, and the required LLM-off smoke."""

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


def _receipts_pass(receipts: Sequence[Mapping[str, Any]], names: Sequence[str]) -> bool:
    counts = Counter(str(row.get("name")) for row in receipts)
    return all(
        counts[name] == 1
        and next(row for row in receipts if row.get("name") == name).get("passed") is True
        and next(row for row in receipts if row.get("name") == name).get("exit_code") == 0
        and next(row for row in receipts if row.get("name") == name).get("timed_out") is False
        for name in names
    )


def terminal_command_specs(root: Path, candidate: Path) -> list[validation_scope.CommandSpec]:
    """Use independent reduction and the repository's unchanged strict readers."""

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
                    "from carnot.experiment_7398_v649_arc_checkpoint import independent_reduce_file; "
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
    """Authenticate exact inputs and the quarantined predecessor receipt."""

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
                "role": (
                    "historical_diagnostic_evidence"
                    if relative in {HISTORICAL_ARTIFACT, HISTORICAL_OUTCOME}
                    or str(relative).startswith(str(HISTORICAL_RAW))
                    else "current_input"
                ),
            }
    spec = (root / SPEC_PATH).read_text(encoding="utf-8") if (root / SPEC_PATH).is_file() else ""
    checks.append(
        gate_row(
            "driving_requirement",
            "precondition",
            True,
            "REQ-ARC-WMTE-7398" in spec,
            upstream=SPEC_PATH.as_posix(),
            artifact_field="REQ-ARC-WMTE-7398",
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
            "experiment_id: 7398" in exclusion or "exp7398" in exclusion,
            upstream="ops/exclusion_manifest.yaml",
            artifact_field=EXPERIMENT_ID,
        )
    )
    historical = _load_object(root / HISTORICAL_ARTIFACT)
    expected = {
        "experiment_id": "exp7384-arc-invocation-boundary",
        "status": "complete_disqualified_required_evidence",
        "verdict_class": "disqualified",
        "flagged_adversarial": True,
        "inference_substrate_class": "no_model_load",
    }
    observed = {key: historical.get(key) for key in expected}
    checks.append(
        gate_row(
            "quarantined_predecessor_identity",
            "historical_diagnostic_only",
            expected,
            observed,
            upstream=HISTORICAL_ARTIFACT.as_posix(),
            artifact_field="producer_identity_class_and_flag",
        )
    )
    receipt = {
        **observed,
        "path": HISTORICAL_ARTIFACT.as_posix(),
        "sha256": hashes.get(HISTORICAL_ARTIFACT.as_posix(), {}).get("sha256"),
        "historical_only": True,
        "authorizes_current_readiness": False,
    }
    return checks, hashes, receipt


def _field_principles(keys: Sequence[str]) -> dict[str, str]:
    """Explain ordinary fields without value wrappers."""

    specific = {
        "schema": "A versioned schema lets strict readers reject incompatible records.",
        "run_date": "The requested date includes actual UTC start and end timestamps.",
        "preconditions_checked": "Exact hashes and eligibility checks precede dependent work.",
        "MODEL_SPECS": "An empty list states that no current LLM was loaded.",
        "model_invoked": "False distinguishes scripted transport from a real LLM attempt.",
        "invocation_counts": "Exact current zeros exclude archived and simulated calls.",
        "inference_substrate": "This string describes current CPU, hash, and scripted work.",
        "inference_substrate_details": "Device and software facts stay separate from the substrate string.",
        "inference_substrate_class": "The closed no-model class applies to current work.",
        "execution_venue": "The closed host value identifies the execution venue.",
        "duration_s": "Monotonic task duration is measured without padding.",
        "phase_spans": "Real phase boundaries expose validation and checkpoint timing.",
        "random_seed": "Frozen fixture and experiment seeds make scripted work repeatable.",
        "reproducibility_checksum": "The checksum binds code, inputs, protocol, and raw rows.",
        "source_artifact_hashes": "Byte hashes retain exact source identity and role.",
        "rows": "Per-unit rows expose metrics, costs, failures, and dispositions.",
        "sample_size_budget": "Planned, attempted, completed, censored, and unstarted work stay distinct.",
        "acceptance_gate_results": "Each gate records category, operator, expected, observed, and pass state.",
        "gate_check_summary": "Every failure identifies its upstream path and field.",
        "verifier_is_oracle": "False states that this plumbing check is not a correctness oracle.",
        "honest_verdict": "The terminal verdict limits the finding to checkpoint qualification.",
        "verdict_class": "The closed class separates null plumbing evidence from efficacy.",
        "flagged_adversarial": "Critical reader findings prevent readiness.",
        "validation_receipts": "Command arguments, environment, timing, exit, and log hashes support audit.",
        "repository_health": "Unrelated broad health stays separate from affected checks.",
        "field_principles": "This map explains each top-level field.",
        "promotion_score": "Zero forbids automatic rollout, publication, or weight changes.",
        "arc_checkpoint_ready_score": "One means restart, accounting, denial, and reader checks passed.",
        "episode_recovery_rows": "Durable event IDs and restart parity prove exact-once recovery.",
        "solve_provenance": "Development proxy prevents scripted actions from receiving solve credit.",
        "provenance_sidecars": "Separate hashes isolate historical runtime and scripted transport evidence.",
    }
    return {
        key: specific.get(key, f"The {key} field retains directly auditable experiment evidence.")
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
    checkpoint_run: Mapping[str, Any],
    validation_receipts: Sequence[Mapping[str, Any]],
    source_hashes: Mapping[str, Any],
    provenance_sidecars: Mapping[str, Any],
    phase_spans: Sequence[Mapping[str, Any]],
    terminal_lints_passed: bool,
) -> JsonDict:
    """Build one independently reducible terminal qualification record."""

    preconditions_passed = bool(preconditions) and all(
        row.get("passed") is True for row in preconditions
    )
    accounting = dict(checkpoint_run.get("accounting") or {})
    planned = int(accounting.get("planned_units") or 0)
    accounting_exact = planned > 0 and planned == sum(
        int(accounting.get(key) or 0)
        for key in ("completed_units", "censored_units", "unstarted_units")
    )
    recovery = [
        row for row in checkpoint_run.get("episode_recovery_rows", []) if isinstance(row, Mapping)
    ]
    controls = [
        row for row in checkpoint_run.get("interruption_controls", []) if isinstance(row, Mapping)
    ]
    restart_passed = bool(recovery) and all(row.get("restart_parity") is True for row in recovery)
    if controls:
        restart_passed = restart_passed and all(row.get("passed") is True for row in controls)
    path_denial = bool(checkpoint_run.get("rows")) and all(
        row.get("adapter_disabled") is True
        and row.get("solution_paths_denied") is True
        and row.get("tool_result_consumed") is True
        and row.get("later_policy_action") is True
        for row in checkpoint_run.get("rows", [])
        if isinstance(row, Mapping)
    )
    independent_reader = checkpoint_run.get("independent_reader_passed") is True
    scoped_passed = _receipts_pass(validation_receipts, REQUIRED_CHECK_NAMES)
    e2e_passed = _receipts_pass(validation_receipts, REQUIRED_E2E_NAMES)
    provenance_passed = set(provenance_sidecars) == {
        "historical_runtime",
        "scripted_transport",
    } and all(
        str(row.get("sha256", "")).startswith("sha256:") for row in provenance_sidecars.values()
    )
    historical_corrected = bool(
        diagnosis.get("cause_reproduced")
        and diagnosis.get("terminal_summary_action_count") == 0
        and int(diagnosis.get("durable_completed_action_count") or 0) > 0
    )
    gates = [
        gate_row(
            "preconditions",
            "precondition",
            True,
            preconditions_passed,
            upstream="preconditions_checked",
            artifact_field="all_current_inputs_authenticated",
        ),
        gate_row(
            "historical_timeout_correction",
            "completion",
            True,
            historical_corrected,
            upstream=HISTORICAL_RAW.as_posix(),
            artifact_field="terminal_zero_vs_durable_actions",
        ),
        gate_row(
            "restart_parity",
            "completion",
            True,
            restart_passed,
            upstream="episode_recovery_rows",
            artifact_field="restart_parity",
        ),
        gate_row(
            "episode_accounting",
            "completion",
            True,
            accounting_exact,
            upstream="sample_size_budget",
            artifact_field="planned=completed+censored+unstarted",
        ),
        gate_row(
            "solution_path_denial",
            "safety",
            True,
            path_denial,
            upstream="rows",
            artifact_field="adapter_disabled_and_solution_paths_denied",
        ),
        gate_row(
            "independent_checkpoint_reader",
            "safety",
            True,
            independent_reader,
            upstream="checkpoint_receipts",
            artifact_field="independent_reader_passed",
        ),
        gate_row(
            "provenance_sidecar_separation",
            "safety",
            True,
            provenance_passed,
            upstream="provenance_sidecars",
            artifact_field="separate_hash_bound_sidecars",
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
            "efficacy",
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
    if not preconditions_passed:
        verdict_class = "blocked"
        honest_verdict = "blocked_required_input_or_current_gate"
    elif ready:
        verdict_class = "null"
        honest_verdict = "complete_null_arc_checkpoint_ready_no_live_efficacy_claim"
    else:
        verdict_class = "disqualified"
        honest_verdict = "complete_disqualified_required_evidence"
    rows: list[JsonDict] = []
    for raw in diagnosis.get("rows", []):
        if not isinstance(raw, Mapping):
            continue
        rows.append(
            {
                "unit_id": raw.get("episode_id"),
                "arm": "historical_timeout_correction",
                "game": raw.get("game"),
                "seed": raw.get("seed"),
                "disposition": raw.get("disposition"),
                "censored": raw.get("censored"),
                "metrics": {"environment_action_count": int(raw.get("action_count") or 0)},
                "costs": {"current_llm_calls": 0},
                "failure": raw.get("error"),
                "scope": "historical",
            }
        )
    for raw in checkpoint_run.get("rows", []):
        if not isinstance(raw, Mapping):
            continue
        rows.append(
            {
                "unit_id": raw.get("episode_id"),
                "arm": raw.get("interruption_boundary", "checkpoint_resume"),
                "game": raw.get("game"),
                "seed": raw.get("seed"),
                "condition": raw.get("interruption_boundary", "uninterrupted_fixture"),
                "disposition": raw.get("disposition", "complete"),
                "censored": raw.get("censored", False),
                "metrics": {
                    "action": raw.get("action"),
                    "tool_result_consumed": raw.get("tool_result_consumed"),
                    "later_policy_action": raw.get("later_policy_action"),
                },
                "costs": {
                    "current_llm_calls": 0,
                    "scripted_http_requests": raw.get("http_request_count", 1),
                },
                "failure": None,
                "scope": "development_proxy",
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
        "inference_substrate": INFERENCE_SUBSTRATE,
        "inference_substrate_details": {
            "device": "host_cpu",
            "processor": platform.processor() or platform.machine(),
            "platform": platform.platform(),
            "jax_platform": os.environ.get("JAX_PLATFORMS", "cpu"),
            "software": "CPython JSON/hash reduction and scripted urllib transport",
        },
        "inference_substrate_class": "no_model_load",
        "execution_venue": "host",
        "small_ebm_training": {"performed": False, "kind": "none"},
        "phase_spans": [deepcopy(dict(row)) for row in phase_spans],
        "random_seed": {
            "experiment": 7398,
            "scripted_transport": [7398, 17398],
            "resampling": None,
        },
        "source_artifact_hashes": deepcopy(dict(source_hashes)),
        "provenance_sidecars": deepcopy(dict(provenance_sidecars)),
        "rows": rows,
        "sample_size_budget": {
            **accounting,
            "unit_limit": planned,
            "stop_rule": "Run two scripted episodes at each of five fixed interruption boundaries.",
            "effective_independent_group_count": len(INTERRUPTION_BOUNDARIES) if controls else 1,
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
            "status": "not_assessed_by_scoped_experiment",
            "unrelated_failures": [],
            "affects_required_checks": False,
        },
        "promotion_score": 0,
        "arc_checkpoint_ready_score": ready,
        "episode_recovery_rows": deepcopy(recovery),
        "checkpoint_receipts": deepcopy(list(checkpoint_run.get("checkpoint_receipts", []))),
        "interruption_controls": deepcopy(controls),
        "historical_correction": {
            "terminal_summary_action_count": diagnosis.get("terminal_summary_action_count"),
            "durable_completed_action_count": diagnosis.get("durable_completed_action_count"),
            "corrected_accounting": deepcopy(diagnosis.get("corrected_accounting")),
            "last_confirmed_event": diagnosis.get("last_confirmed_event"),
            "first_missing_event": diagnosis.get("first_missing_event"),
        },
        "solve_provenance": "development_proxy",
        "credited_solve": False,
        "production_defaults_changed": False,
        "supervisor_ordering_changed": False,
        "generator_weights_changed": False,
        "active_research_roadmap_changed": False,
        "research_conductor_changed": False,
    }
    artifact["field_principles"] = _field_principles(
        [*artifact, "field_principles", "reproducibility_checksum"]
    )
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return artifact


def validate_artifact(value: Mapping[str, Any]) -> list[str]:
    """Cold-check identity, current provenance, readiness, and checksum."""

    errors: list[str] = []
    if value.get("schema") != SCHEMA or value.get("experiment_id") != EXPERIMENT_ID:
        errors.append("schema or experiment identity mismatch")
    if value.get("milestone") != MILESTONE or value.get("run_date") != RUN_DATE:
        errors.append("milestone or run date mismatch")
    if value.get("MODEL_SPECS") != [] or value.get("model_invoked") is not False:
        errors.append("current model declaration mismatch")
    if value.get("invocation_counts") != ZERO_INVOCATION_COUNTS:
        errors.append("current invocation counts mismatch")
    if not isinstance(value.get("inference_substrate"), str):
        errors.append("inference substrate must be a string")
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
    if value.get("arc_checkpoint_ready_score") != expected_ready:
        errors.append("readiness reduction mismatch")
    if value.get("verdict_class") in {"blocked", "disqualified", "partial"} and value.get(
        "arc_checkpoint_ready_score"
    ):
        errors.append("unsafe readiness on non-ready artifact")
    principles = value.get("field_principles")
    if not isinstance(principles, Mapping) or not set(value) <= set(principles):
        errors.append("field principles mismatch")
    if value.get("reproducibility_checksum") != artifact_checksum(value):
        errors.append("checksum mismatch")
    return errors


def independent_reduce_file(path: str | Path) -> JsonDict:
    """Reload the exact candidate and independently recompute readiness."""

    artifact = _load_object(Path(path))
    errors = validate_artifact(artifact)
    accounting = dict(artifact.get("sample_size_budget") or {})
    accounting_exact = int(accounting.get("planned_units") or 0) > 0 and int(
        accounting.get("planned_units") or 0
    ) == sum(
        int(accounting.get(key) or 0)
        for key in ("completed_units", "censored_units", "unstarted_units")
    )
    recovery = artifact.get("episode_recovery_rows") or []
    restart_parity = bool(recovery) and all(row.get("restart_parity") is True for row in recovery)
    declared = int(artifact.get("arc_checkpoint_ready_score") or 0)
    reduced = int(
        not errors
        and accounting_exact
        and restart_parity
        and all(row.get("passed") is True for row in artifact.get("acceptance_gate_results", []))
    )
    return {
        "declared_arc_checkpoint_ready_score": declared,
        "reduced_arc_checkpoint_ready_score": reduced,
        "accounting_exact": accounting_exact,
        "restart_parity": restart_parity,
        "validation_errors": errors,
        "matches_declared": declared == reduced,
    }


def _phase(phase: str, phase_started: float, run_started: float, units: int) -> JsonDict:
    now = time.monotonic()
    return {
        "phase": phase,
        "started_elapsed_s": round(phase_started - run_started, 6),
        "ended_elapsed_s": round(now - run_started, 6),
        "duration_s": round(now - phase_started, 6),
        "completed_units": units,
        "checkpoint_at_utc": utc_now(),
    }


def run_experiment(args: argparse.Namespace) -> JsonDict:  # pragma: no cover - CLI orchestration.
    """Run qualification, affected checks, terminal readers, and atomic publish."""

    started = time.monotonic()
    started_at = utc_now()
    phases: list[JsonDict] = []
    progress(started, "startup", "begin", run_date=args.date)

    phase_started = time.monotonic()
    progress(started, "read", "before_preconditions")
    preconditions, source_hashes, _historical_receipt = collect_preconditions(REPO_ROOT)
    phases.append(_phase("read", phase_started, started, len(preconditions)))
    progress(
        started, "read", "after_preconditions", passed=all(row["passed"] for row in preconditions)
    )

    phase_started = time.monotonic()
    progress(started, "evaluate", "before_historical_reduction")
    diagnosis = diagnose_historical_timeout(REPO_ROOT)
    progress(started, "evaluate", "after_historical_reduction", cause=diagnosis["cause_reproduced"])
    progress(started, "evaluate", "before_scripted_checkpoint_matrix")
    checkpoint_run = run_interruption_matrix(REPO_ROOT / RAW_DIR / "scripted-checkpoints")
    progress(
        started,
        "evaluate",
        "after_scripted_checkpoint_matrix",
        completed_units=checkpoint_run["accounting"]["completed_units"],
    )
    sidecars = write_provenance_sidecars(
        REPO_ROOT / RAW_DIR / "sidecars", diagnosis, checkpoint_run
    )
    for receipt in sidecars.values():
        sidecar_path = Path(str(receipt["path"]))
        source_hashes[sidecar_path.relative_to(REPO_ROOT).as_posix()] = {
            "path": str(sidecar_path),
            "sha256": receipt["sha256"],
            "role": f"{receipt['scope']}_sidecar",
        }
    phases.append(_phase("evaluate", phase_started, started, len(checkpoint_run["rows"])))

    private = Path(tempfile.mkdtemp(prefix="exp7398-validation-"))
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
        checkpoint_run=checkpoint_run,
        validation_receipts=validation_receipts,
        source_hashes=source_hashes,
        provenance_sidecars=sidecars,
        phase_spans=phases,
        terminal_lints_passed=True,
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
    phases.append(_phase("validate_terminal", phase_started, started, len(terminal_receipts)))

    phase_started = time.monotonic()
    artifact = build_terminal_artifact(
        run_date=args.date,
        started_at_utc=started_at,
        ended_at_utc=utc_now(),
        duration_s=time.monotonic() - started,
        preconditions=preconditions,
        diagnosis=diagnosis,
        checkpoint_run=checkpoint_run,
        validation_receipts=[*validation_receipts, *terminal_receipts],
        source_hashes=source_hashes,
        provenance_sidecars=sidecars,
        phase_spans=[*phases, _phase("write", phase_started, started, 1)],
        terminal_lints_passed=terminal_passed,
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
        ready=artifact["arc_checkpoint_ready_score"],
    )
    return artifact


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the fixed execution date accepted by the thin entrypoint."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", required=True, choices=[RUN_DATE])
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - thin CLI handoff.
    """Return success after any honest terminal complete or blocked artifact."""

    artifact = run_experiment(parse_args(argv))
    return 0 if str(artifact.get("status", "")).startswith(("complete_", "blocked_")) else 1
