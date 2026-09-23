"""Qualify corrected B2 live-agent evidence without running the model again.

The source run is historical GPU work. This module only authenticates and
reduces its completed bytes, so current-work counters always stay at zero.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any

from carnot.experiment_7358_v646_validation_contract import AffectedManifest
from carnot.reporting import experiment_7303_validation_scope as validation_scope


JsonDict = dict[str, Any]
REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_DATE = "20260923"
MILESTONE = "2026.09.660"
EXPERIMENT_ID = "exp7556-arc-corrected-custody"
SCHEMA = "carnot.exp7556.v660.arc_corrected_custody.v1"
RESULT_PATH = Path("results/experiment_7556_v660_arc_corrected_custody.json")
LOCAL_SOURCE_PATH = Path("results/experiment_10008_b2_induction_gate_measurement_v2.json")
EXTERNAL_ROOT = Path("/home/ianblenke/carnot-wt-b2c")
OLD_SOURCE_PATH = Path("results/experiment_7531_b2_induction_gate_measurement.json")
RAW_RELATIVE = Path("results/raw/experiment_10008_b2_induction_gate_measurement_v2")
REGISTRY_PATH = Path("ops/arc_solve_registry.yaml")
SPEC_PATH = Path("openspec/capabilities/arc-world-model-trust-energy/spec.md")
MODULE_PATH = Path("python/carnot/experiment_7556_v660_arc_corrected_custody.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_7556_v660_arc_corrected_custody.py")
TEST_PATH = Path("tests/python/test_experiment_7556_v660_arc_corrected_custody.py")
MODEL_SPECS: list[JsonDict] = []
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
}
TERMINAL_CHECK_NAMES = (
    "fresh_process_cold_replay",
    "independent_reduction",
    "adversarial_verify",
    "verdict_row_consistency_strict",
)
REQUIRED_RECEIPT_NAMES = (*validation_scope.REQUIRED_CHECK_NAMES, *TERMINAL_CHECK_NAMES)
AFFECTED_MANIFEST = AffectedManifest(
    experiment_id=EXPERIMENT_ID,
    test_paths=(TEST_PATH.as_posix(),),
    changed_modules=(MODULE_PATH.as_posix(),),
    static_paths=(WRAPPER_PATH.as_posix(),),
)


def utc_now() -> str:
    """Return a stable UTC spelling for command and publication boundaries."""

    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def progress(started: float, phase: str, event: str, **details: Any) -> None:
    """Flush one truthful phase boundary for the conductor's silence guard."""

    suffix = " ".join(f"{key}={value}" for key, value in sorted(details.items()))
    print(
        f"[exp7556] phase={phase} event={event} elapsed_s={time.monotonic() - started:.3f}"
        + (f" {suffix}" if suffix else ""),
        flush=True,
    )


def sha256_file(path: Path) -> str:
    """Hash exact bytes so later source drift invalidates the custody receipt."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def canonical_hash(value: Any) -> str:
    """Hash stable JSON so row removal or mutation changes the result identity."""

    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    """Publish complete JSON bytes only after the temporary file reaches disk."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def load_json(path: Path) -> JsonDict:
    """Load one required JSON object and reject arrays or scalar substitutes."""

    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"json_object_required:{path}")
    return value


def load_jsonl(path: Path) -> list[JsonDict]:
    """Load every non-empty JSONL row and reject malformed or non-object rows."""

    rows: list[JsonDict] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"jsonl_object_required:{path}:{line_number}")
        rows.append(value)
    return rows


def precondition_row(
    check: str,
    upstream: str,
    artifact_field: str,
    expected: Any,
    observed: Any,
) -> JsonDict:
    """Describe one exact comparison so a blocker cannot hide its cause."""

    return {
        "check": check,
        "upstream": upstream,
        "path": upstream,
        "artifact_field": artifact_field,
        "expected": expected,
        "observed": observed,
        "op": "==",
        "passed": observed == expected,
        "principle": "A dependent claim needs the named prerequisite value.",
    }


def locate_corrected_source(root: Path, external_root: Path = EXTERNAL_ROOT) -> Path | None:
    """Prefer the current worktree and use only the documented B2C fallback."""

    local = root.resolve() / LOCAL_SOURCE_PATH
    external = external_root.resolve() / LOCAL_SOURCE_PATH
    if local.is_file():
        return local
    if external.is_file():
        return external
    return None


def _pid_alive(pid: int) -> bool:
    """Report a live non-zombie process without changing or signalling it."""

    stat = Path(f"/proc/{pid}/stat")
    try:
        fields = stat.read_text(encoding="utf-8").split()
    except OSError:
        return False
    return len(fields) > 2 and fields[2] != "Z"


def _source_root(source_path: Path) -> Path:
    """Recover the worktree root from its results artifact location."""

    return source_path.resolve().parents[1]


def collect_preconditions(
    root: Path,
    external_root: Path = EXTERNAL_ROOT,
    *,
    pid_probe: Callable[[int], bool] = _pid_alive,
) -> tuple[Path | None, list[JsonDict]]:
    """Check exact inputs and producer state before any evidence reduction."""

    source = locate_corrected_source(root, external_root)
    checks = [
        precondition_row(
            "corrected_b2_artifact_available",
            "experiment_10008_b2_induction_gate_measurement_v2",
            "path",
            "readable_terminal_artifact",
            "readable_terminal_artifact" if source is not None else None,
        )
    ]
    required = (
        Path("AGENTS.md"),
        Path("CODEX.md"),
        Path("CLAUDE.md"),
        Path("research-program.md"),
        Path("ops/exclusion_manifest.yaml"),
        Path("ops/e2e-test-plan.md"),
        OLD_SOURCE_PATH,
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
    spec_has_req = False
    try:
        spec_has_req = "REQ-ARC-WMTE-7556" in (root / SPEC_PATH).read_text(encoding="utf-8")
    except OSError:
        pass
    checks.append(
        precondition_row(
            "spec_requirement_present",
            SPEC_PATH.as_posix(),
            "REQ-ARC-WMTE-7556",
            True,
            spec_has_req,
        )
    )
    checks.append(
        precondition_row(
            "aggregation_resource_available",
            "current_host",
            "resource_class",
            "host_filesystem_no_model_load",
            "host_filesystem_no_model_load",
        )
    )
    if source is None:
        return None, checks
    try:
        source_value = load_json(source)
    except (OSError, json.JSONDecodeError, ValueError):
        source_value = {}
    checks.extend(
        (
            precondition_row(
                "source_status_terminal",
                str(source),
                "status",
                "complete_b2_measurement",
                source_value.get("status"),
            ),
            precondition_row(
                "source_historical_verdict",
                str(source),
                "honest_verdict",
                "complete_feasibility_only_sample_floor_not_met",
                source_value.get("honest_verdict"),
            ),
        )
    )
    session_path = _source_root(source) / RAW_RELATIVE / "live_session.json"
    session = load_json(session_path) if session_path.is_file() else {}
    runtime = session.get("runtime_receipt")
    runtime = runtime if isinstance(runtime, Mapping) else {}
    child_pid = session.get("child_pid")
    active = pid_probe(child_pid) if isinstance(child_pid, int) else None
    checks.extend(
        (
            precondition_row(
                "producer_child_terminal",
                str(session_path),
                "child_terminal",
                True,
                runtime.get("child_terminal"),
            ),
            precondition_row(
                "producer_child_exit",
                str(session_path),
                "child_returncode",
                0,
                runtime.get("child_returncode"),
            ),
            precondition_row(
                "producer_lease_released",
                str(session_path),
                "lease_release.released",
                True,
                (runtime.get("lease_release") or {}).get("released"),
            ),
            precondition_row(
                "producer_process_exited",
                str(session_path),
                "child_pid_active",
                False,
                active,
            ),
        )
    )
    return source, checks


def _raw_citation_checks(
    source: Mapping[str, Any], source_root: Path
) -> tuple[list[JsonDict], bool]:
    """Authenticate each high-level raw file cited by the source artifact."""

    rows: list[JsonDict] = []
    for citation in source.get("cited_artifacts") or []:
        if not isinstance(citation, Mapping):
            continue
        label = str(citation.get("path") or "")
        if not label.startswith(RAW_RELATIVE.as_posix() + "/"):
            continue
        path = source_root / label
        observed = sha256_file(path) if path.is_file() else None
        expected = citation.get("sha256")
        rows.append(
            {
                "path": label,
                "role": citation.get("role"),
                "expected_sha256": expected,
                "observed_sha256": observed,
                "passed": observed == expected,
            }
        )
    required_roles = {
        "b2_protocol",
        "live_session",
        "current_invocation_ledger",
        "request_events",
        "action_events",
        "b2_induction_telemetry",
    }
    observed_roles = {str(row["role"]) for row in rows if row["passed"] is True}
    return rows, required_roles <= observed_roles and all(row["passed"] for row in rows)


def _invocation_summary(events: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Count historical calls and require one terminal state per attempt."""

    calls: dict[str, list[str]] = defaultdict(list)
    operations: dict[str, str] = {}
    owner_pids: set[int] = set()
    for row in events:
        call_id = str(row.get("call_id") or "")
        calls[call_id].append(str(row.get("state") or ""))
        operations[call_id] = str(row.get("operation") or "")
        if isinstance(row.get("owner_pid"), int):
            owner_pids.add(int(row["owner_pid"]))
    incomplete = [
        call_id
        for call_id, states in calls.items()
        if states.count("attempted") != 1
        or sum(state in {"completed", "failed", "cancelled"} for state in states) != 1
    ]
    counts = deepcopy(ZERO_INVOCATION_COUNTS)
    for call_id, states in calls.items():
        prefix = "model_loads" if operations[call_id] == "model_load" else "generation_calls"
        if operations[call_id] not in {"model_load", "generation"}:
            incomplete.append(call_id)
            continue
        counts[f"{prefix}_attempted"] += states.count("attempted")
        for state in ("completed", "failed", "cancelled"):
            counts[f"{prefix}_{state}"] += states.count(state)
        counts[f"{prefix}_in_flight"] += int(
            "attempted" in states
            and not any(state in states for state in ("completed", "failed", "cancelled"))
        )
    return {
        **counts,
        "owner_pids": sorted(owner_pids),
        "incomplete_call_ids": sorted(set(incomplete)),
        "all_calls_terminal": not incomplete,
    }


def _event_file(path_label: str, raw_dir: Path) -> Path:
    """Map the original B2C absolute path to the copied immutable raw tree."""

    marker = RAW_RELATIVE.name + "/"
    if marker not in path_label:
        return Path("/__invalid_exp7556_event_path__")
    suffix = path_label.split(marker, 1)[1]
    return raw_dir / suffix


def _response_custody(
    raw_dir: Path, runtime_events: Sequence[Mapping[str, Any]], source: Mapping[str, Any]
) -> tuple[JsonDict, dict[str, list[JsonDict]]]:
    """Reduce exact response bytes and verify their transport-event hashes."""

    requests = [row for row in runtime_events if row.get("event") == "server_request"]
    responses = [row for row in runtime_events if row.get("event") == "server_response"]
    errors = [row for row in runtime_events if row.get("event") == "server_error"]
    byte_hashes_match = True
    requested_caps: set[int] = set()
    for row in (*requests, *responses):
        key = "request_path" if row.get("event") == "server_request" else "response_path"
        hash_key = "request_sha256" if key == "request_path" else "response_sha256"
        path = _event_file(str(row.get(key) or ""), raw_dir)
        observed = sha256_file(path) if path.is_file() else None
        byte_hashes_match = byte_hashes_match and observed == row.get(hash_key)
        cap = row.get("requested_max_tokens")
        if isinstance(cap, int) and not isinstance(cap, bool):
            requested_caps.add(cap)

    by_episode: dict[str, list[JsonDict]] = defaultdict(list)
    finish = Counter()
    token_histogram = Counter()
    content_bytes = 0
    reasoning_bytes = 0
    response_bytes = 0
    for event in responses:
        path = _event_file(str(event.get("response_path") or ""), raw_dir)
        value = load_json(path)
        choices = value.get("choices") or []
        choice = choices[0] if choices and isinstance(choices[0], Mapping) else {}
        message = choice.get("message") if isinstance(choice, Mapping) else {}
        message = message if isinstance(message, Mapping) else {}
        usage = value.get("usage") if isinstance(value.get("usage"), Mapping) else {}
        completion = usage.get("completion_tokens")
        completion = completion if isinstance(completion, int) else None
        record = {
            "episode_id": event.get("episode_id"),
            "call_index": event.get("call_index"),
            "path": path.relative_to(raw_dir.parent.parent.parent).as_posix(),
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
            "finish_reason": choice.get("finish_reason"),
            "completion_tokens": completion,
            "prompt_tokens": usage.get("prompt_tokens"),
            "content_bytes": len(str(message.get("content") or "").encode()),
            "reasoning_bytes": len(str(message.get("reasoning_content") or "").encode()),
        }
        by_episode[str(event.get("episode_id"))].append(record)
        finish[str(record["finish_reason"])] += 1
        if completion is not None:
            token_histogram[str(completion)] += 1
        content_bytes += int(record["content_bytes"])
        reasoning_bytes += int(record["reasoning_bytes"])
        response_bytes += int(record["bytes"])
    for rows in by_episode.values():
        rows.sort(key=lambda row: int(row.get("call_index") or 0))
    source_distribution = source.get("completion_tokens_distribution") or {}
    histogram = dict(sorted(token_histogram.items()))
    summary = {
        "request_count": len(requests),
        "completed_response_count": len(responses),
        "failed_response_count": len(errors),
        "requested_token_caps": sorted(requested_caps),
        "completion_token_total": sum(int(key) * count for key, count in histogram.items()),
        "completion_token_histogram": histogram,
        "finish_reason_histogram": dict(sorted(finish.items())),
        "content_byte_total": content_bytes,
        "reasoning_byte_total": reasoning_bytes,
        "response_byte_total": response_bytes,
        "transport_hashes_match": byte_hashes_match,
        "source_distribution_matches": histogram == source_distribution.get("histogram"),
        "second_saturation_observed": len(histogram) == 1 and histogram.get("4096") == 33,
    }
    return summary, by_episode


def _attempt_rows(
    telemetry: Sequence[Mapping[str, Any]],
    actions: Sequence[Mapping[str, Any]],
    responses: Mapping[str, Sequence[Mapping[str, Any]]],
) -> list[JsonDict]:
    """Join each fired attempt to its distinct response and later live actions."""

    action_groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in actions:
        action_groups[str(row.get("episode_id"))].append(row)
    source_attempts = [row for row in telemetry if row.get("record_type") == "induction_attempt"]
    attempt_groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in source_attempts:
        attempt_groups[str(row.get("episode_id"))].append(row)
    joined: list[JsonDict] = []
    for episode_id, attempts in attempt_groups.items():
        attempts.sort(key=lambda row: float(row.get("monotonic_timestamp_s") or 0.0))
        response_rows = list(responses.get(episode_id) or [])
        for index, attempt in enumerate(attempts):
            response = response_rows[index] if index < len(response_rows) else None
            step = int(attempt.get("step_index") or 0)
            later_actions = [
                row
                for row in action_groups.get(episode_id, [])
                if isinstance(row.get("action_index"), int) and int(row["action_index"]) > step
            ]
            level_up = attempt.get("level_up_progress") is True
            frame_change = attempt.get("frame_change_progress") is True
            if level_up:
                progress_kind = "attributable_level_up"
            elif frame_change:
                progress_kind = "incidental_frame_change_only"
            else:
                progress_kind = "no_progress_observed"
            joined.append(
                {
                    "attempt_id": attempt.get("attempt_id"),
                    "episode_id": episode_id,
                    "game": attempt.get("game_id"),
                    "seed": int(episode_id.rsplit("seed-", 1)[1]),
                    "monotonic_timestamp_s": attempt.get("monotonic_timestamp_s"),
                    "step_index": step,
                    "progress_window_actions": attempt.get("progress_window_actions"),
                    "progress_window_censored": attempt.get("progress_window_censored"),
                    "response_disposition": (
                        "completed" if response is not None else "censored_no_response"
                    ),
                    "request_call_index": response.get("call_index") if response else index,
                    "response_sha256": response.get("sha256") if response else None,
                    "finish_reason": response.get("finish_reason") if response else None,
                    "completion_tokens": response.get("completion_tokens") if response else None,
                    "prompt_tokens": response.get("prompt_tokens") if response else None,
                    "content_bytes": response.get("content_bytes") if response else None,
                    "reasoning_bytes": response.get("reasoning_bytes") if response else None,
                    "planned": attempt.get("planned") is True,
                    "verifier_result": attempt.get("verifier_result"),
                    "frame_change_progress": frame_change,
                    "level_up_progress": level_up,
                    "progress_attribution": progress_kind,
                    "later_live_action_count": len(later_actions),
                    "action_provenance": (
                        "observed_live_actions" if later_actions else "no_later_action_observed"
                    ),
                    "disposition": "complete" if response is not None else "censored",
                }
            )
    return sorted(joined, key=lambda row: str(row["attempt_id"]))


def _schedule_rows(
    schedule: Sequence[Mapping[str, Any]],
    episodes: Sequence[Mapping[str, Any]],
    attempts: Sequence[Mapping[str, Any]],
) -> list[JsonDict]:
    """Retain every planned game and seed without converting missing work to zero."""

    by_episode = {str(row.get("episode_id")): row for row in episodes}
    attempt_ids: dict[str, list[str]] = defaultdict(list)
    for row in attempts:
        attempt_ids[str(row.get("episode_id"))].append(str(row.get("attempt_id")))
    rows: list[JsonDict] = []
    for planned in schedule:
        episode_id = str(planned.get("episode_id"))
        episode = by_episode.get(episode_id, {})
        disposition = str(episode.get("disposition") or "unstarted")
        rows.append(
            {
                "unit_id": episode_id,
                "game": planned.get("game"),
                "seed": planned.get("seed"),
                "execution_order": planned.get("execution_order"),
                "disposition": disposition,
                "action_count": episode.get("action_count") if disposition != "unstarted" else None,
                "start_level": episode.get("start_level"),
                "peak_level": episode.get("peak_level"),
                "terminal_level": episode.get("terminal_level"),
                "new_level_credit": episode.get("new_level_credit"),
                "solve_provenance": episode.get("solve_provenance"),
                "induction_attempt_ids": attempt_ids.get(episode_id, []),
                "censored": disposition == "censored",
                "failure": episode.get("error"),
            }
        )
    return rows


def _per_game(
    rows: Sequence[Mapping[str, Any]], attempts: Sequence[Mapping[str, Any]]
) -> list[JsonDict]:
    """Group complete unit and attempt records without discarding their identities."""

    games = sorted({str(row.get("game")) for row in rows})
    return [
        {
            "game": game,
            "units": [deepcopy(dict(row)) for row in rows if row.get("game") == game],
            "attempts": [deepcopy(dict(row)) for row in attempts if row.get("game") == game],
        }
        for game in games
    ]


def reduce_corrected_evidence(
    root: Path,
    source_path: Path,
    *,
    pid_probe: Callable[[int], bool] = _pid_alive,
) -> JsonDict:
    """Recompute custody and all scientific rows from the completed B2C bytes."""

    before = source_path.stat()
    source = load_json(source_path)
    source_root = _source_root(source_path)
    raw_dir = source_root / RAW_RELATIVE
    session = load_json(raw_dir / "live_session.json")
    schedule_value = load_json(raw_dir / "frozen_schedule.json")
    schedule = schedule_value.get("rows") or schedule_value.get("schedule") or []
    episodes = session.get("episodes") or []
    boundary = load_jsonl(raw_dir / "current_invocation_events.jsonl")
    runtime_events = load_jsonl(raw_dir / "runtime_events.jsonl")
    actions = load_jsonl(raw_dir / "live_action_rows.jsonl")
    telemetry = load_jsonl(raw_dir / "induction_gate_telemetry.jsonl")
    after = source_path.stat()
    cited_rows, raw_hashes_match = _raw_citation_checks(source, source_root)
    historical_calls = _invocation_summary(boundary)
    response, responses_by_episode = _response_custody(raw_dir, runtime_events, source)
    attempts = _attempt_rows(telemetry, actions, responses_by_episode)
    rows = _schedule_rows(schedule, episodes, attempts)
    runtime = session.get("runtime_receipt") or {}
    child_pid = int(session.get("child_pid"))
    run_ids = sorted(
        {str(row.get("run_id")) for row in telemetry if isinstance(row.get("run_id"), str)}
    )
    session_run_id = run_ids[0] if len(run_ids) == 1 else None
    expected_run_id = f"induction_gate_telemetry:{child_pid}"
    completed_episodes = [row for row in episodes if row.get("disposition") == "complete"]
    policy_entries = [row.get("policy_entry") or {} for row in completed_episodes]
    controls = {
        "factory": "make_carnot_agent",
        "policy_class": "E3AgentPolicy",
        "adapter_disabled": all(row.get("adapter_disabled") is True for row in episodes),
        "banked_trajectories_disabled": all(
            row.get("banked_trajectories_disabled") is True for row in episodes
        ),
        "stored_engines_disabled": all(
            row.get("stored_engines_disabled") is True for row in episodes
        ),
        "game_source_read": any(row.get("game_source_read") is True for row in episodes),
    }
    policy_valid = bool(policy_entries) and all(
        row.get("factory") == controls["factory"]
        and row.get("policy_class") == controls["policy_class"]
        for row in policy_entries
    )
    credited = [
        row
        for row in rows
        if isinstance(row.get("new_level_credit"), int) and row["new_level_credit"] > 0
    ]
    credited_with_actions = sum(
        bool(row.get("induction_attempt_ids"))
        and row.get("solve_provenance") == "live_agent_self_discovery"
        for row in credited
    )
    model_specs = source.get("MODEL_SPECS") or []
    model = model_specs[0] if len(model_specs) == 1 and isinstance(model_specs[0], Mapping) else {}
    model_valid = (
        model.get("hf_id") == "unsloth/Qwen3.8-27B-GGUF"
        and model.get("quantization") == "Q4_K_M"
        and (model.get("runtime_settings") or {}).get("max_new_tokens_per_call") == 4096
        and str(model.get("sha256") or "").startswith("sha256:")
    )
    action_rows_valid = bool(actions) and all(
        row.get("event") == "action_end"
        and isinstance(row.get("action_index"), int)
        and str(row.get("state_sha256") or "").startswith("sha256:")
        for row in actions
    )
    sample_budget = {
        "planned": len(rows),
        "attempted": sum(row["disposition"] != "unstarted" for row in rows),
        "completed": sum(row["disposition"] == "complete" for row in rows),
        "excluded": sum(row["disposition"] == "excluded" for row in rows),
        "failed": sum(row["disposition"] == "failed" for row in rows),
        "censored": sum(row["disposition"] == "censored" for row in rows),
        "unstarted": sum(row["disposition"] == "unstarted" for row in rows),
    }
    attempt_budget = {
        "attempted": len(attempts),
        "completed_responses": sum(row["response_disposition"] == "completed" for row in attempts),
        "failed_responses": sum(row["response_disposition"] == "failed" for row in attempts),
        "censored_responses": sum(
            row["response_disposition"] == "censored_no_response" for row in attempts
        ),
    }
    source_stable = (
        before.st_ino == after.st_ino
        and before.st_size == after.st_size
        and before.st_mtime_ns == after.st_mtime_ns
    )
    lease_release = runtime.get("lease_release") or {}
    producer_active = pid_probe(child_pid)
    handoff = {
        "selected_location": str(source_path.resolve()),
        "original_location": str((EXTERNAL_ROOT / LOCAL_SOURCE_PATH).resolve()),
        "source_sha256": sha256_file(source_path),
        "source_stable": source_stable,
        "terminal_status": source.get("status"),
        "producer_pid": child_pid,
        "producer_exited": not producer_active,
        "producer_returncode": runtime.get("child_returncode"),
        "session_run_id": session_run_id,
        "expected_session_run_id": expected_run_id,
        "lease_released": lease_release.get("released"),
        "raw_hashes_match": raw_hashes_match,
        "historical_calls_terminal": historical_calls["all_calls_terminal"],
        "source_raw_hashes": cited_rows,
    }
    join_summary = {
        "attempt_count": len(attempts),
        "world_model_output_count": sum((row.get("content_bytes") or 0) > 0 for row in attempts),
        "planned_count": sum(row.get("planned") is True for row in attempts),
        "verifier_observed_count": sum(
            row.get("verifier_result") not in (None, "not_observed") for row in attempts
        ),
        "frame_change_progress_count": sum(
            row.get("frame_change_progress") is True for row in attempts
        ),
        "level_up_progress_count": sum(row.get("level_up_progress") is True for row in attempts),
        "credited_level_count": sum(int(row.get("new_level_credit") or 0) for row in credited),
        "credited_levels_with_action_provenance": credited_with_actions,
        "action_provenance_complete": action_rows_valid,
        "later_progress_inferred_from_frame_change_only": True,
    }
    old = load_json(root / OLD_SOURCE_PATH)
    source_hashes = {
        LOCAL_SOURCE_PATH.as_posix(): sha256_file(source_path),
        OLD_SOURCE_PATH.as_posix(): sha256_file(root / OLD_SOURCE_PATH),
        REGISTRY_PATH.as_posix(): sha256_file(root / REGISTRY_PATH),
        **{str(row["path"]): str(row["observed_sha256"]) for row in cited_rows},
    }
    custody_qualified = all(
        (
            source.get("status") == "complete_b2_measurement",
            source_stable,
            not producer_active,
            runtime.get("child_terminal") is True,
            runtime.get("child_returncode") == 0,
            lease_release.get("released") is True,
            raw_hashes_match,
            historical_calls["all_calls_terminal"] is True,
            session_run_id == expected_run_id,
            policy_valid,
            controls["adapter_disabled"] is True,
            controls["banked_trajectories_disabled"] is True,
            controls["stored_engines_disabled"] is True,
            controls["game_source_read"] is False,
            model_valid,
            response["transport_hashes_match"] is True,
        )
    )
    return {
        "custody_qualified": custody_qualified,
        "handoff_receipt": handoff,
        "source_artifact_hashes": source_hashes,
        "historical_model_calls": historical_calls,
        "live_policy_path": controls,
        "model_identity": deepcopy(dict(model)),
        "model_identity_qualified": model_valid,
        "response_custody": response,
        "rows": rows,
        "attempt_rows": attempts,
        "per_game_results": _per_game(rows, attempts),
        "sample_size_budget": sample_budget,
        "induction_attempt_budget": attempt_budget,
        "join_summary": join_summary,
        "registry_receipt": {
            "path": REGISTRY_PATH.as_posix(),
            "sha256": source_hashes[REGISTRY_PATH.as_posix()],
            "reviewed_before_level_claim": True,
            "new_level_claimed": False,
        },
        "historical_exp7531": {
            "path": OLD_SOURCE_PATH.as_posix(),
            "honest_verdict": old.get("honest_verdict"),
            "false_negative_risk": deepcopy(old.get("false_negative_risk")),
            "further_correction_2026_09_22": deepcopy(old.get("further_correction_2026_09_22")),
            "corrigendum": deepcopy(old.get("corrigendum")),
        },
        "source_historical_verdict": source.get("honest_verdict"),
        "source_positive_control": deepcopy(source.get("positive_control")),
        "source_positive_control_diagnostic": deepcopy(source.get("positive_control_diagnostic")),
        "source_possible_second_completion_cap": source.get("possible_second_completion_cap"),
        "source_started_at_utc": source.get("started_at_utc"),
        "source_ended_at_utc": source.get("ended_at_utc"),
    }


def validate_reduction(value: Mapping[str, Any]) -> list[str]:
    """Reject custody, cap, session, and solve-provenance mutations."""

    errors: list[str] = []
    handoff = value.get("handoff_receipt") or {}
    if handoff.get("producer_exited") is not True:
        errors.append("producer_still_active")
    if handoff.get("producer_returncode") != 0 or handoff.get("lease_released") is not True:
        errors.append("producer_not_terminal")
    if handoff.get("session_run_id") != handoff.get("expected_session_run_id"):
        errors.append("session_identity_mismatch")
    if handoff.get("source_stable") is not True or handoff.get("raw_hashes_match") is not True:
        errors.append("source_or_raw_hash_mismatch")
    if handoff.get("historical_calls_terminal") is not True:
        errors.append("historical_call_incomplete")
    response = value.get("response_custody") or {}
    if response.get("requested_token_caps") != [4096]:
        errors.append("corrected_cap_mismatch")
    if response.get("transport_hashes_match") is not True:
        errors.append("transport_hash_mismatch")
    if response.get("source_distribution_matches") is not True:
        errors.append("source_distribution_mismatch")
    if response.get("second_saturation_observed") is not True:
        errors.append("second_saturation_missing")
    if value.get("model_identity_qualified") is not True:
        errors.append("model_identity_mismatch")
    policy = value.get("live_policy_path") or {}
    expected_policy = {
        "factory": "make_carnot_agent",
        "policy_class": "E3AgentPolicy",
        "adapter_disabled": True,
        "banked_trajectories_disabled": True,
        "stored_engines_disabled": True,
        "game_source_read": False,
    }
    if policy != expected_policy:
        errors.append("live_policy_path_mismatch")
    rows = value.get("rows") or []
    attempts = value.get("attempt_rows") or []
    if len(rows) != 144 or len({(row.get("game"), row.get("seed")) for row in rows}) != 144:
        errors.append("schedule_reconstruction_mismatch")
    if len(attempts) != 35:
        errors.append("attempt_reconstruction_mismatch")
    join = value.get("join_summary") or {}
    if join.get("credited_level_count") != join.get("credited_levels_with_action_provenance"):
        errors.append("credited_level_missing_action_provenance")
    if join.get("action_provenance_complete") is not True:
        errors.append("action_provenance_incomplete")
    if value.get("custody_qualified") is not (not errors):
        errors.append("custody_qualified_mismatch")
    return list(dict.fromkeys(errors))


def _gate(
    check: str,
    category: str,
    expected: Any,
    observed: Any,
    principle: str,
) -> JsonDict:
    """Keep readiness and benefit comparisons machine-readable and separate."""

    return {
        "check": check,
        "category": category,
        "expected": expected,
        "observed": observed,
        "op": "==",
        "passed": observed == expected,
        "principle": principle,
    }


def _receipts_pass(receipts: Sequence[Mapping[str, Any]]) -> bool:
    """Require each scoped and terminal command exactly once with a clean exit."""

    by_name: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in receipts:
        by_name[str(row.get("name"))].append(row)
    return all(
        len(by_name[name]) == 1
        and by_name[name][0].get("passed") is True
        and by_name[name][0].get("exit_code") == 0
        and by_name[name][0].get("timed_out") is False
        for name in REQUIRED_RECEIPT_NAMES
    )


def _gate_summary(gates: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Name the first failed gate while retaining every later failure."""

    failures = [deepcopy(dict(row)) for row in gates if row.get("passed") is not True]
    return {
        "all_passed": not failures,
        "failed_count": len(failures),
        "first_failure": failures[0] if failures else None,
        "failures": failures,
    }


def _code_hashes(root: Path) -> JsonDict:
    """Bind the implementation, entrypoint, test, and governing requirement."""

    paths = (MODULE_PATH, WRAPPER_PATH, TEST_PATH, SPEC_PATH)
    return {
        path.as_posix(): sha256_file(root / path) if (root / path).is_file() else None
        for path in paths
    }


def _checksum_payload(artifact: Mapping[str, Any]) -> JsonDict:
    """Select stable method and evidence fields for independent replay."""

    keys = (
        "experiment_id",
        "milestone",
        "run_date",
        "random_seed",
        "code_hashes",
        "source_artifact_hashes",
        "historical_model_calls",
        "live_policy_path",
        "historical_model_identity",
        "response_custody",
        "rows",
        "induction_attempt_rows",
        "sample_size_budget",
        "induction_attempt_budget",
        "solve_provenance",
        "acceptance_gate_results",
        "honest_verdict",
        "verdict_class",
        "corrected_arc_ready_score",
    )
    return {key: deepcopy(artifact.get(key)) for key in keys}


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Bind code, settings, source identities, model identity, and raw rows."""

    return canonical_hash(_checksum_payload(artifact))


def _field_principles(keys: Sequence[str]) -> dict[str, str]:
    """Explain the failure prevented by every published top-level field."""

    specific = {
        "experiment_id": "Bind the exact V660 task instead of a nearby ARC run.",
        "preconditions_checked": "Show input and resource checks before reduction.",
        "MODEL_SPECS": "Keep this no-load task from inheriting historical model identity.",
        "model_specs": "Match the canonical no-load model list spelling.",
        "model_invoked": "Prevent historical Qwen calls from becoming current calls.",
        "invocation_counts": "Expose every current load and generation disposition.",
        "inference_substrate_class": "Apply the aggregation duration policy without padding.",
        "inference_substrate": "Name aggregation from upstream artifacts exactly.",
        "execution_venue": "Use the legal host enum instead of host_cpu.",
        "duration_s": "Measure current work instead of copying historical runtime.",
        "random_seed": "Bind historical ordering without inventing current sampling.",
        "reproducibility_checksum": "Detect changed code, sources, settings, or rows.",
        "rows": "Keep every game and seed disposition; missing is not zero.",
        "sample_size_budget": "Separate planned, completed, censored, and unstarted units.",
        "acceptance_gate_results": "Keep validity, readiness, and benefit independent.",
        "gate_check_summary": "Name exact expected and observed values for failures.",
        "honest_verdict": "Use a terminal complete prefix without overstating benefit.",
        "verdict_class": "Use the closed scientific terminal vocabulary.",
        "verifier_is_oracle": "Prevent oracle-defined evidence from becoming causal proof.",
        "flagged_adversarial": "Preserve safety findings instead of clearing a gate.",
        "validation_receipts": "Retain exact commands, exits, hashes, and cold replay.",
        "corrected_arc_ready_score": "Separate authenticated evidence readiness from efficacy.",
        "historical_model_calls": "Keep old GPU activity separate from current aggregation.",
        "per_game_results": "Retain each game, seed, attempt, and censoring disposition.",
        "solve_provenance": "Deny headline credit without live self-discovery actions.",
        "handoff_receipt": "Bind original location, producer exit, session, and raw hashes.",
    }
    return {
        key: specific.get(key, "Retain this value so independent replay can detect drift.")
        for key in keys
    }


def build_artifact(
    root: Path,
    run_date: str,
    reduced: Mapping[str, Any],
    *,
    preconditions_checked: Sequence[Mapping[str, Any]],
    validation_receipts: Sequence[Mapping[str, Any]],
    duration_s: float,
    phase_spans: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Build a ready-but-null terminal candidate from authenticated evidence."""

    reduction_valid = not validate_reduction(reduced)
    validation_passed = _receipts_pass(validation_receipts)
    ready = int(reduction_valid and validation_passed)
    joins = reduced.get("join_summary") or {}
    gates = [
        _gate(
            "raw_custody",
            "validity",
            True,
            reduction_valid,
            "Invalid or active evidence cannot support science.",
        ),
        _gate(
            "response_byte_accounting",
            "support",
            True,
            (reduced.get("response_custody") or {}).get("source_distribution_matches"),
            "Token totals must come from authenticated response bytes.",
        ),
        _gate(
            "required_scoped_and_terminal_validation",
            "validity",
            True,
            validation_passed,
            "Only passing scoped checks and cold replay can promote readiness.",
        ),
        _gate(
            "corrected_arc_evidence_ready",
            "readiness",
            1,
            ready,
            "A valid null remains auditable and readiness is not benefit.",
        ),
        _gate(
            "induction_efficacy_observed",
            "benefit",
            True,
            bool(
                joins.get("planned_count", 0) > 0
                and joins.get("verifier_observed_count", 0) > 0
                and joins.get("level_up_progress_count", 0) > 0
            ),
            "Configuration repair without a plan-to-action-to-progress join is not efficacy.",
        ),
    ]
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "run_date": run_date,
        "status": "complete_null_corrected_b2_authenticated_no_efficacy",
        "honest_verdict": "complete_null_corrected_b2_authenticated_no_efficacy",
        "verdict_class": "null",
        "positive_claim": False,
        "no_headroom": False,
        "no_headroom_annotation": (
            "The frame-change proxy is saturated, so absence of measured headroom is "
            "not evidence that induction has no underlying headroom."
        ),
        "flagged_adversarial": False,
        "verifier_is_oracle": False,
        "corrected_arc_ready_score": ready,
        "MODEL_SPECS": [],
        "model_specs": [],
        "model_invoked": False,
        "invocation_counts": deepcopy(ZERO_INVOCATION_COUNTS),
        "historical_model_calls": deepcopy(reduced.get("historical_model_calls") or {}),
        "historical_model_identity": deepcopy(reduced.get("model_identity") or {}),
        "inference_substrate": "aggregation_from_upstream_artifacts",
        "inference_substrate_class": "aggregation",
        "planned_inference_substrate_class": "aggregation",
        "execution_venue": "host",
        "device_compute": {
            "current_work": "host_cpu_json_reduction",
            "cuda_used_by_current_work": False,
            "historical_cuda_work_only": True,
        },
        "duration_s": float(duration_s),
        "phase_spans": [deepcopy(dict(row)) for row in phase_spans],
        "process_identity": {"pid": os.getpid(), "worktree_root": str(root.resolve())},
        "duration_accounting": {
            "current_work_s": float(duration_s),
            "historical_runtime_s": 10_570.330257908994,
            "authoring_included": False,
            "validation_included": True,
        },
        "random_seed": {
            "historical_episode_seeds": sorted({row["seed"] for row in reduced["rows"]}),
            "ordering_seed": 7531,
            "current_sampling_seed": None,
        },
        "preconditions_checked": [deepcopy(dict(row)) for row in preconditions_checked],
        "source_artifact_hashes": deepcopy(reduced.get("source_artifact_hashes") or {}),
        "code_hashes": _code_hashes(root),
        "handoff_receipt": deepcopy(reduced.get("handoff_receipt") or {}),
        "live_policy_path": deepcopy(reduced.get("live_policy_path") or {}),
        "response_custody": deepcopy(reduced.get("response_custody") or {}),
        "join_summary": deepcopy(joins),
        "rows": deepcopy(reduced.get("rows") or []),
        "induction_attempt_rows": deepcopy(reduced.get("attempt_rows") or []),
        "per_game_results": deepcopy(reduced.get("per_game_results") or []),
        "sample_size_budget": deepcopy(reduced.get("sample_size_budget") or {}),
        "induction_attempt_budget": deepcopy(reduced.get("induction_attempt_budget") or {}),
        "solve_provenance": {
            "required_for_credit": "live_agent_self_discovery",
            "credited_level_count": joins.get("credited_level_count", 0),
            "credited_levels_with_action_provenance": joins.get(
                "credited_levels_with_action_provenance", 0
            ),
            "development_proxy_headline_credit": False,
            "outer_loop_re_headline_credit": False,
        },
        "registry_receipt": deepcopy(reduced.get("registry_receipt") or {}),
        "historical_exp7531": deepcopy(reduced.get("historical_exp7531") or {}),
        "historical_verdicts": {
            "experiment_7531": (reduced.get("historical_exp7531") or {}).get("honest_verdict"),
            "experiment_10008": reduced.get("source_historical_verdict"),
        },
        "positive_control": deepcopy(reduced.get("source_positive_control") or {}),
        "positive_control_diagnostic": deepcopy(
            reduced.get("source_positive_control_diagnostic") or {}
        ),
        "possible_second_completion_cap": reduced.get("source_possible_second_completion_cap"),
        "retired_budget_raise_reopened": False,
        "production_defaults_changed": False,
        "publication_mode": "none",
        "remote_submission": False,
        "acceptance_gate_results": gates,
        "gate_check_summary": _gate_summary(gates),
        "validation_receipts": [deepcopy(dict(row)) for row in validation_receipts],
    }
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    artifact["field_principles"] = _field_principles(tuple(artifact))
    return artifact


def build_blocked_artifact(
    run_date: str,
    checks: Sequence[Mapping[str, Any]],
    *,
    duration_s: float,
    source_path: Path | None,
) -> JsonDict:
    """Publish external absence or active production as blocked, never partial."""

    failed = next((deepcopy(dict(row)) for row in checks if row.get("passed") is not True), None)
    gate = failed or precondition_row(
        "corrected_b2_artifact_available",
        "experiment_10008_b2_induction_gate_measurement_v2",
        "path",
        "readable_terminal_artifact",
        None,
    )
    gate["category"] = "validity"
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "run_date": run_date,
        "status": "complete_blocked_corrected_b2_not_available",
        "honest_verdict": "complete_blocked_corrected_b2_not_available",
        "verdict_class": "blocked",
        "positive_claim": False,
        "no_headroom": False,
        "flagged_adversarial": False,
        "verifier_is_oracle": False,
        "corrected_arc_ready_score": 0,
        "MODEL_SPECS": [],
        "model_specs": [],
        "model_invoked": False,
        "invocation_counts": deepcopy(ZERO_INVOCATION_COUNTS),
        "historical_model_calls": {},
        "inference_substrate": "blocked_no_run",
        "inference_substrate_class": "blocked_no_run",
        "planned_inference_substrate_class": "aggregation",
        "execution_venue": "host",
        "duration_s": float(duration_s),
        "phase_spans": [],
        "process_identity": {"pid": os.getpid()},
        "random_seed": {"current_sampling_seed": None},
        "preconditions_checked": [deepcopy(dict(row)) for row in checks],
        "source_artifact_hashes": {},
        "code_hashes": {},
        "handoff_receipt": {
            "selected_location": str(source_path) if source_path else None,
            "terminal_status": None,
            "producer_exited": None,
            "raw_hashes_match": None,
        },
        "rows": [],
        "induction_attempt_rows": [],
        "per_game_results": [],
        "sample_size_budget": {
            "planned": 144,
            "attempted": 0,
            "completed": 0,
            "excluded": 0,
            "failed": 0,
            "censored": 0,
            "unstarted": 144,
        },
        "induction_attempt_budget": {
            "attempted": 0,
            "completed_responses": 0,
            "failed_responses": 0,
            "censored_responses": 0,
        },
        "solve_provenance": {
            "required_for_credit": "live_agent_self_discovery",
            "credited_level_count": 0,
        },
        "acceptance_gate_results": [gate],
        "gate_check_summary": {
            "all_passed": False,
            "failed_count": 1,
            "first_failure": deepcopy(gate),
            "failures": [deepcopy(gate)],
        },
        "validation_receipts": [],
    }
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    artifact["field_principles"] = _field_principles(tuple(artifact))
    return artifact


def independent_reduce(artifact: Mapping[str, Any]) -> JsonDict:
    """Recompute readiness and scientific non-benefit from published rows."""

    rows = artifact.get("rows") or []
    attempts = artifact.get("induction_attempt_rows") or []
    receipts_passed = _receipts_pass(artifact.get("validation_receipts") or [])
    custody_passed = (
        (artifact.get("handoff_receipt") or {}).get("raw_hashes_match") is True
        and len(rows) == 144
        and len(attempts) == 35
        and (artifact.get("response_custody") or {}).get("second_saturation_observed") is True
    )
    benefit = any(
        row.get("planned") is True
        and row.get("verifier_result") not in (None, "not_observed")
        and row.get("level_up_progress") is True
        for row in attempts
    )
    return {
        "custody_passed": custody_passed,
        "required_validation_passed": receipts_passed,
        "row_count": len(rows),
        "attempt_count": len(attempts),
        "benefit_observed": benefit,
        "corrected_arc_ready_score": int(custody_passed and receipts_passed),
        "verdict_class": "null" if custody_passed and receipts_passed and not benefit else None,
    }


def validate_artifact(artifact: Mapping[str, Any], *, require_terminal: bool = False) -> list[str]:
    """Cold-check identity, current work, readiness, rows, and checksum."""

    errors: list[str] = []
    if artifact.get("experiment_id") != EXPERIMENT_ID:
        errors.append("experiment_id_mismatch")
    if artifact.get("milestone") != MILESTONE or artifact.get("run_date") != RUN_DATE:
        errors.append("experiment_identity_mismatch")
    if not str(artifact.get("honest_verdict") or "").startswith("complete_"):
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
    score = artifact.get("corrected_arc_ready_score")
    if type(score) is not int or score not in {0, 1}:
        errors.append("corrected_arc_ready_score_invalid")
    if artifact.get("MODEL_SPECS") != [] or artifact.get("model_specs") != []:
        errors.append("current_model_specs_not_empty")
    if artifact.get("model_invoked") is not False:
        errors.append("current_model_invoked")
    if artifact.get("invocation_counts") != ZERO_INVOCATION_COUNTS:
        errors.append("current_invocation_counts_nonzero")
    if artifact.get("execution_venue") != "host":
        errors.append("execution_venue_invalid")
    if artifact.get("positive_claim") is not False:
        errors.append("positive_claim_invalid")
    if artifact.get("reproducibility_checksum") != reproducibility_checksum(artifact):
        errors.append("reproducibility_checksum_mismatch")
    principles = artifact.get("field_principles") or {}
    if set(artifact) - {"field_principles"} > set(principles):
        errors.append("field_principles_incomplete")
    if artifact.get("verdict_class") == "blocked":
        if score != 0 or artifact.get("honest_verdict") != (
            "complete_blocked_corrected_b2_not_available"
        ):
            errors.append("blocked_classification_mismatch")
        first = (artifact.get("gate_check_summary") or {}).get("first_failure")
        if not isinstance(first, Mapping):
            errors.append("blocked_gate_summary_missing")
        return list(dict.fromkeys(errors))
    if artifact.get("inference_substrate") != "aggregation_from_upstream_artifacts":
        errors.append("inference_substrate_mismatch")
    if artifact.get("inference_substrate_class") != "aggregation":
        errors.append("inference_substrate_class_mismatch")
    if len(artifact.get("rows") or []) != 144:
        errors.append("row_count_mismatch")
    if len(artifact.get("induction_attempt_rows") or []) != 35:
        errors.append("attempt_count_mismatch")
    solve = artifact.get("solve_provenance") or {}
    if solve.get("credited_level_count") != solve.get("credited_levels_with_action_provenance"):
        errors.append("solve_provenance_mismatch")
    validation_passed = _receipts_pass(artifact.get("validation_receipts") or [])
    if require_terminal and not validation_passed:
        errors.append("required_validation_failed")
    if require_terminal and score != 1:
        errors.append("ready_score_mismatch")
    if artifact.get("verdict_class") != "null":
        errors.append("scientific_verdict_mismatch")
    return list(dict.fromkeys(errors))


def cold_replay(path: Path) -> JsonDict:
    """Reload a candidate in a fresh process and recompute its terminal reduction."""

    artifact = load_json(path)
    errors = validate_artifact(artifact, require_terminal=False)
    if errors:
        raise ValueError("cold_replay_invalid:" + ",".join(errors))
    return independent_reduce(artifact)


def _terminal_commands(root: Path, candidate: Path) -> list[validation_scope.CommandSpec]:
    """Declare bounded cold replay, independent reduction, and strict readers."""

    python = str(root / ".venv/bin/python")
    wrapper = WRAPPER_PATH.as_posix()
    return [
        validation_scope.CommandSpec(
            "fresh_process_cold_replay",
            (python, "-u", wrapper, "--date", RUN_DATE, "--cold-replay", str(candidate)),
            "terminal_candidate",
        ),
        validation_scope.CommandSpec(
            "independent_reduction",
            (python, "-u", wrapper, "--date", RUN_DATE, "--validate", str(candidate)),
            "terminal_candidate",
        ),
        validation_scope.CommandSpec(
            "adversarial_verify",
            (python, "-u", "scripts/adversarial_verify.py", str(candidate)),
            "terminal_candidate",
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
            "terminal_candidate",
        ),
    ]


def _phase_span(phase: str, phase_started: float, run_started: float) -> JsonDict:
    """Record one monotonic phase interval relative to this process start."""

    ended = time.monotonic()
    return {
        "phase": phase,
        "start_s": phase_started - run_started,
        "end_s": ended - run_started,
        "duration_s": ended - phase_started,
    }


def run_experiment(  # pragma: no cover - exercised by the declared capability entrypoint.
    root: Path,
    run_date: str,
    *,
    output_path: Path = RESULT_PATH,
) -> JsonDict:
    """Authenticate, validate, cold-replay, and atomically publish Exp7556."""

    if run_date != RUN_DATE:
        raise ValueError(f"run_date_mismatch:{run_date}")
    started = time.monotonic()
    spans: list[JsonDict] = []
    progress(started, "preconditions", "start")
    phase_started = time.monotonic()
    source, checks = collect_preconditions(root)
    spans.append(_phase_span("preconditions", phase_started, started))
    progress(
        started,
        "preconditions",
        "end",
        passed=all(row["passed"] for row in checks),
        source=source,
    )
    if source is None or not all(row["passed"] for row in checks):
        blocked = build_blocked_artifact(
            run_date,
            checks,
            duration_s=time.monotonic() - started,
            source_path=source,
        )
        progress(started, "publication", "before_atomic_blocked", path=output_path)
        atomic_json(root / output_path, blocked)
        progress(started, "publication", "after_atomic_blocked", path=output_path)
        return blocked

    progress(started, "reduction", "start")
    phase_started = time.monotonic()
    reduced = reduce_corrected_evidence(root, source)
    reduction_errors = validate_reduction(reduced)
    spans.append(_phase_span("reduction", phase_started, started))
    progress(started, "reduction", "end", errors=len(reduction_errors))
    if reduction_errors:
        raise RuntimeError("corrected_evidence_invalid:" + ",".join(reduction_errors))

    private_root = Path(tempfile.mkdtemp(prefix="exp7556-validation-", dir="/tmp"))
    raw_dir = root / "results/raw/experiment_7556_v660_arc_corrected_custody"
    coverage_file = private_root / ".coverage.exp7556"
    (private_root / "pytest").mkdir(parents=True, exist_ok=True)
    commands = validation_scope.build_scoped_commands(
        root,
        AFFECTED_MANIFEST.test_paths,
        AFFECTED_MANIFEST.changed_modules,
        static_paths=AFFECTED_MANIFEST.static_paths,
        basetemp=private_root / "pytest",
        coverage_file=coverage_file,
    )
    progress(started, "validation", "before_scoped_subprocesses", units=len(commands))
    phase_started = time.monotonic()
    affected = validation_scope.run_commands(
        root,
        commands,
        log_dir=raw_dir / "validation/affected",
        extra_env={"COVERAGE_FILE": str(coverage_file)},
    )
    for row in affected:
        row["command_environment"] = {"COVERAGE_FILE": str(coverage_file)}
    spans.append(_phase_span("validation", phase_started, started))
    progress(
        started,
        "validation",
        "after_scoped_subprocesses",
        passed=all(row["passed"] for row in affected),
    )
    if not all(row["passed"] for row in affected):
        raise RuntimeError("scoped_validation_failed")

    candidate = build_artifact(
        root,
        run_date,
        reduced,
        preconditions_checked=checks,
        validation_receipts=affected,
        duration_s=time.monotonic() - started,
        phase_spans=spans,
    )
    candidate_path = raw_dir / "terminal_candidate.json"
    atomic_json(candidate_path, candidate)
    progress(started, "terminal_validation", "before_subprocesses", units=4)
    phase_started = time.monotonic()
    terminal = validation_scope.run_commands(
        root,
        _terminal_commands(root, candidate_path),
        log_dir=raw_dir / "validation/terminal",
    )
    spans.append(_phase_span("terminal_validation", phase_started, started))
    progress(
        started,
        "terminal_validation",
        "after_subprocesses",
        passed=all(row["passed"] for row in terminal),
    )
    if not all(row["passed"] for row in terminal):
        raise RuntimeError("terminal_validation_failed")

    final = build_artifact(
        root,
        run_date,
        reduced,
        preconditions_checked=checks,
        validation_receipts=[*affected, *terminal],
        duration_s=time.monotonic() - started,
        phase_spans=spans,
    )
    errors = validate_artifact(final, require_terminal=True)
    if errors:
        raise RuntimeError("terminal_artifact_invalid:" + ",".join(errors))
    progress(started, "publication", "before_atomic_terminal", path=output_path)
    atomic_json(candidate_path, final)
    atomic_json(root / output_path, final)
    progress(started, "publication", "after_atomic_terminal", path=output_path)
    return final


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the thin public entrypoint and its cold-reader modes."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", required=True)
    parser.add_argument("--output", type=Path, default=RESULT_PATH)
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--validate", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - thin CLI.
    """Dispatch one experiment run or a read-only fresh-process check."""

    args = parse_args(argv)
    if args.cold_replay is not None:
        print(json.dumps(cold_replay(args.cold_replay), sort_keys=True), flush=True)
        return 0
    if args.validate is not None:
        value = load_json(args.validate)
        errors = validate_artifact(value, require_terminal=False)
        if errors:
            raise ValueError("artifact_invalid:" + ",".join(errors))
        print(json.dumps(independent_reduce(value), sort_keys=True), flush=True)
        return 0
    run_experiment(REPO_ROOT, args.date, output_path=args.output)
    return 0


if __name__ == "__main__":  # pragma: no cover - module execution convenience.
    raise SystemExit(main())
