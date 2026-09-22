"""Run the instrumented E6 live-loop cost profile.

The observer imports and extends Experiment 7471. It is off by default. It
records timing only. It does not change policy inputs or return values.

Spec: REQ-ARC-WMTE-7491 and SCENARIO-ARC-WMTE-7491-*.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from copy import deepcopy
from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
import random
import signal
import subprocess
import time
from types import MethodType
from typing import Any

from carnot import experiment_7471_v654_arc_seam_observation as exp7471
from carnot.agentic.arc_decision_telemetry import (
    NOOP_RECORDER,
    NoOpDecisionTelemetryRecorder,
    TELEMETRY_EPISODE_ENV,
    TELEMETRY_PATH_ENV,
    load_telemetry,
)
from carnot.agentic.arc_inference_boundary import InvocationBoundaryLedger
from carnot.agentic.arc_request_budget import attach_request_budget
from carnot.reporting import current_work_receipt


JsonDict = dict[str, Any]
REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_DATE = "20260921"
EXPERIMENT_ID = 7491
EXPERIMENT_NAME = "exp7491-e6-timed-live-profile"
TASK_ID = "experiment_7491_e6_timed_live_profile"
SCHEMA = "carnot.arc.e6_timed_live_profile.v1"
MODEL_ID = exp7471.MODEL_ID
MODEL_FILENAME = exp7471.MODEL_FILENAME
MODEL_SPECS = [MODEL_ID]
INFERENCE_SUBSTRATE = "owned_native_cuda_llama_cpp_qwen3.8_27b_gguf"
INFERENCE_SUBSTRATE_CLASS = "model_bounded_generation"
EXECUTION_VENUE = "host"

E4_GAMES = ("sk48", "tr87", "tu93", "s5i5", "lp85", "tn36", "lf52", "cn04", "re86")
PANEL_GAMES = (
    "sb26",
    "vc33",
    "su15",
    "g50t",
    "m0r0",
    "dc22",
    "wa30",
    "ka59",
    "bp35",
    "sp80",
    "ft09",
    "ar25",
)
EPISODE_SEEDS = (7_491_001, 7_491_002, 7_491_003)
PANEL_PREFIX = "carnot-e6-timed-2026-09-21:"
ACTION_LIMIT = exp7471.ACTION_LIMIT
REQUEST_LIMIT = exp7471.REQUEST_LIMIT
MAX_NEW_TOKENS = exp7471.MAX_NEW_TOKENS
EPISODE_LIMIT_S = exp7471.EPISODE_LIMIT_S
TOTAL_LIVE_LIMIT_S = 3 * 60 * 60.0
GPU_INDEX = 1
GPU_IDLE_MAX_USED_MB = 500
OFFLOAD_MIN_MB = 17_000
OFFLOAD_MAX_MB = 20_000

SPEC_PATH = Path("openspec/capabilities/arc-world-model-trust-energy/spec.md")
FROZEN_PANEL_PATH = Path("docs/research-notes/semif-e6-timed-frozen-game-panel-2026-09-21.json")
RESULT_PATH = Path("results/experiment_7491_e6_timed_live_profile.json")
RAW_DIR = Path("results/raw/experiment_7491_e6_timed_live_profile")
SCHEDULE_PATH = RAW_DIR / "frozen_schedule.json"
SESSION_PATH = RAW_DIR / "live_session.json"
BOUNDARY_PATH = RAW_DIR / "current_invocation_events.jsonl"
RUNTIME_EVENT_PATH = RAW_DIR / "runtime_events.jsonl"
ACTION_PATH = RAW_DIR / "live_action_rows.jsonl"
CHECKPOINT_PATH = Path("results/checkpoints/experiment_7491_e6_timed_live_profile.json")
MODULE_PATH = Path("python/carnot/experiment_7491_e6_timed_live_profile.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_7491_e6_timed_live_profile.py")
TEST_PATH = Path("tests/python/test_experiment_7491_e6_timed_live_profile.py")
PRIOR_E6_PATH = Path("results/experiment_7490_e6_live_loop_cost_profile.json")

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

WITHHELD_INPUTS = (
    "per_game_adapter",
    "stored_engine",
    "banked_trajectory",
    "banked_solution",
    "cross_game_state",
    "game_source",
    "offline_ground_truth_bfs",
    "per_game_model",
    "registry_policy_input",
)

DECISION_POINTS = (
    "candidate_selection",
    "induction_and_generation",
    "world_model_verification",
    "supervisor",
    "planner",
    "environment",
)


def utc_now() -> str:
    """Return one aware UTC timestamp."""

    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def progress(started: float, step: str, event: str, **details: Any) -> None:
    """Print one flushed progress line."""

    suffix = " ".join(f"{key}={value}" for key, value in sorted(details.items()))
    print(
        f"[exp7491] step={step} event={event} elapsed_s={time.monotonic() - started:.3f}"
        + (f" {suffix}" if suffix else ""),
        flush=True,
    )


def canonical_hash(value: Any) -> str:
    """Hash canonical JSON bytes."""

    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    """Hash a file in bounded chunks."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    """Write one JSON object atomically to an explicit path."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(dict(value), indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def load_json(path: Path) -> JsonDict:
    """Load one JSON object or return an empty object."""

    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(value) if isinstance(value, Mapping) else {}


def read_span_rows(path: Path) -> list[JsonDict]:
    """Read complete span rows."""

    if not path.is_file():
        return []
    rows: list[JsonDict] = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        value = json.loads(line)
        if isinstance(value, Mapping):
            rows.append(dict(value))
    return rows


def _interval_union(intervals: Sequence[tuple[int, int]]) -> int:
    merged: list[list[int]] = []
    for start, end in sorted(intervals):
        if end <= start:
            continue
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return sum(end - start for start, end in merged)


def reconcile_spans(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Reconcile every complete parent to its direct child union and gap."""

    by_id = {str(row.get("decision_id")): row for row in rows if row.get("decision_id")}
    children: dict[str, list[Mapping[str, Any]]] = {}
    errors: list[str] = []
    for row in rows:
        parent = row.get("parent_decision_id")
        if parent is not None:
            children.setdefault(str(parent), []).append(row)
    reconciliations: list[JsonDict] = []
    for decision_id, row in by_id.items():
        start = row.get("start_monotonic_ns")
        end = row.get("end_monotonic_ns")
        exclusive = row.get("exclusive_ns")
        if not all(isinstance(value, int) for value in (start, end, exclusive)):
            errors.append(f"incomplete:{decision_id}")
            continue
        assert isinstance(start, int)
        assert isinstance(end, int)
        assert isinstance(exclusive, int)
        direct_intervals: list[tuple[int, int]] = []
        for child in children.get(decision_id, []):
            child_start = child.get("start_monotonic_ns")
            child_end = child.get("end_monotonic_ns")
            if not isinstance(child_start, int) or not isinstance(child_end, int):
                errors.append(f"incomplete_child:{decision_id}")
                continue
            if child_start < start or child_end > end:
                errors.append(f"child_outside_parent:{child.get('decision_id')}")
            if child.get("concurrent") is not True:
                direct_intervals.append((max(start, child_start), min(end, child_end)))
        duration = max(0, end - start)
        child_union = _interval_union(direct_intervals)
        gap = duration - child_union
        passed = gap >= 0 and exclusive == gap
        if not passed:
            errors.append(f"exclusive_mismatch:{decision_id}")
        reconciliations.append(
            {
                "decision_id": decision_id,
                "duration_ns": duration,
                "direct_child_union_ns": child_union,
                "gap_ns": gap,
                "exclusive_ns": exclusive,
                "passed": passed,
            }
        )
    roots = [row for row in rows if row.get("parent_decision_id") is None]
    for root in roots:
        start = root.get("start_monotonic_ns")
        end = root.get("end_monotonic_ns")
        if isinstance(start, int) and isinstance(end, int):
            descendants = [
                row
                for row in rows
                if isinstance(row.get("exclusive_ns"), int)
                and isinstance(row.get("start_monotonic_ns"), int)
                and isinstance(row.get("end_monotonic_ns"), int)
                and start <= int(row["start_monotonic_ns"])
                and int(row["end_monotonic_ns"]) <= end
            ]
            if sum(int(row["exclusive_ns"]) for row in descendants) != end - start:
                errors.append(f"root_descendant_mismatch:{root.get('decision_id')}")
    return {
        "passed": not errors and bool(rows),
        "errors": errors,
        "rows": reconciliations,
        "span_count": len(rows),
    }


class _TimingRecorder(NoOpDecisionTelemetryRecorder):
    """Compose E6 timing with an optional decision recorder without double calls."""

    enabled = True

    def __init__(
        self,
        observer: E6TimedObserver,
        delegate: Any = NOOP_RECORDER,
    ) -> None:
        self.observer = observer
        self.delegate = delegate
        # NoOpDecisionTelemetryRecorder.error_count is a plain writable class attribute,
        # not a property, so a real property here would be an incompatible override
        # (mypy: "Cannot override writeable attribute with read-only property"). Track
        # the observer's count by keeping this attribute synced in count_error() below
        # instead of computing it lazily.
        self.error_count = observer.error_count

    def count_error(self) -> None:
        self.observer.count_error()
        self.error_count = self.observer.error_count + int(
            getattr(self.delegate, "error_count", 0) or 0
        )

    def begin_policy_step(self, policy: Any, latest: Any) -> None:
        self.delegate.begin_policy_step(policy, latest)

    def record_event(self, seam: str, payload: Mapping[str, Any], wall_time_s: float) -> None:
        self.delegate.record_event(seam, payload, wall_time_s)

    def record_candidate_action(
        self,
        options: Sequence[Mapping[str, Any]],
        *,
        ranking_changed: bool,
        wall_time_s: float,
        state: Mapping[str, Any] | None = None,
        state_summary: str = "",
    ) -> None:
        self.delegate.record_candidate_action(
            options,
            ranking_changed=ranking_changed,
            wall_time_s=wall_time_s,
            state=state,
            state_summary=state_summary,
        )

    def record_induction_decision(
        self,
        policy: Any,
        *,
        stalled: bool,
        won: bool,
        decision: tuple[bool, str | None],
        wall_time_s: float,
    ) -> None:
        self.delegate.record_induction_decision(
            policy,
            stalled=stalled,
            won=won,
            decision=decision,
            wall_time_s=wall_time_s,
        )

    def time_supervisor_selection(self, supervisor: Any, snapshot: Any) -> Any:
        with self.observer.span("supervisor"):
            return self.delegate.time_supervisor_selection(supervisor, snapshot)

    def time_world_model_selection(
        self,
        selector: Callable[..., Any],
        transitions: Sequence[Any],
        candidates: Sequence[Any],
        *,
        acceptance_threshold: float | None = None,
        **kwargs: Any,
    ) -> Any:
        with self.observer.span("world_model_verification", verifier_kind="candidate_selector"):
            return self.delegate.time_world_model_selection(
                selector,
                transitions,
                candidates,
                acceptance_threshold=acceptance_threshold,
                **kwargs,
            )

    def time_world_model_verification(
        self,
        verifier: Any,
        engine: Any,
        *,
        candidates: Sequence[Any] = (),
        candidate_source: str = "loaded_engine",
    ) -> Any:
        with self.observer.span(
            "world_model_verification",
            verifier_kind=type(verifier).__name__,
            candidate_source=candidate_source,
        ):
            return self.delegate.time_world_model_verification(
                verifier,
                engine,
                candidates=candidates,
                candidate_source=candidate_source,
            )

    def complete_induction(
        self,
        policy: Any,
        attempt: Mapping[str, Any],
        wall_time_s: float,
    ) -> None:
        self.delegate.complete_induction(policy, attempt, wall_time_s)

    def observe_induction_progress(self, policy: Any, latest: Any) -> None:
        self.delegate.observe_induction_progress(policy, latest)

    def finish_episode(self, *, level_end: int | None, actions_used: int | None = None) -> None:
        self.delegate.finish_episode(level_end=level_end, actions_used=actions_used)
        self.error_count = self.observer.error_count + int(
            getattr(self.delegate, "error_count", 0) or 0
        )


class E6TimedObserver(exp7471.E3SeamObserver):
    """Add hierarchical exclusive spans to the Experiment 7471 observer."""

    def __init__(
        self,
        episode_id: str,
        event_path: Path,
        *,
        enabled: bool = False,
        clock_ns: Callable[[], int] = time.monotonic_ns,
    ) -> None:
        super().__init__(episode_id, event_path, clock_ns=clock_ns)
        self.enabled = bool(enabled)
        self._span_rows: list[JsonDict] = []
        self._open_span_ids: list[str] = []
        self._span_by_id: dict[str, JsonDict] = {}
        self._finished = False
        self._error_count = 0
        self._usage_by_request: dict[str, JsonDict] = {}

    @property
    def error_count(self) -> int:
        return self._error_count

    def count_error(self) -> None:
        try:
            self._error_count += 1
        except Exception:
            return

    @contextmanager
    def span(
        self,
        seam: str,
        terminal_disposition: str = "completed",
        *,
        parent_decision_id: str | None = None,
        concurrent: bool = False,
        **fields: Any,
    ) -> Iterator[str | None]:
        """Record one span and never intercept the wrapped result or error."""

        if not self.enabled:
            yield None
            return
        try:
            decision_id = self._decision_id(str(seam))
            parent = (
                parent_decision_id
                if parent_decision_id is not None
                else self._open_span_ids[-1]
                if self._open_span_ids
                else None
            )
            started = self.clock_ns()
            row: JsonDict = {
                "schema": "carnot.arc.e6_exclusive_span.v1",
                "episode_id": self.episode_id,
                "decision_id": decision_id,
                "parent_decision_id": parent,
                "seam": str(seam),
                "start_monotonic_ns": started,
                "end_monotonic_ns": None,
                "exclusive_ns": None,
                "concurrent": bool(concurrent),
                "terminal_disposition": terminal_disposition,
                "clock_identity": "time.monotonic_ns",
                **deepcopy(fields),
            }
            self._span_rows.append(row)
            self._span_by_id[decision_id] = row
            self._open_span_ids.append(decision_id)
        except Exception:
            self.count_error()
            yield None
            return
        try:
            yield decision_id
        except BaseException as exc:
            row["terminal_disposition"] = f"error:{type(exc).__name__}"
            raise
        finally:
            try:
                row["end_monotonic_ns"] = self.clock_ns()
                if self._open_span_ids and self._open_span_ids[-1] == decision_id:
                    self._open_span_ids.pop()
                elif decision_id in self._open_span_ids:
                    self._open_span_ids.remove(decision_id)
            except Exception:
                self.count_error()

    def episode(self, terminal_disposition: str = "completed") -> Any:
        """Open one episode parent span."""

        return self.span("episode", terminal_disposition)

    def annotate(self, decision_id: str | None, **fields: Any) -> None:
        """Attach already-observed metadata to one span."""

        if not self.enabled or decision_id is None:
            return
        try:
            row = self._span_by_id[decision_id]
            row.update(deepcopy(fields))
        except Exception:
            self.count_error()

    def add_completed_span(
        self,
        seam: str,
        *,
        start_ns: int,
        end_ns: int,
        parent_decision_id: str | None,
        terminal_disposition: str,
        concurrent: bool = False,
        **fields: Any,
    ) -> str | None:
        """Add a transport span whose durable timestamps already exist."""

        if not self.enabled:
            return None
        try:
            decision_id = self._decision_id(seam)
            row: JsonDict = {
                "schema": "carnot.arc.e6_exclusive_span.v1",
                "episode_id": self.episode_id,
                "decision_id": decision_id,
                "parent_decision_id": parent_decision_id,
                "seam": seam,
                "start_monotonic_ns": int(start_ns),
                "end_monotonic_ns": int(end_ns),
                "exclusive_ns": None,
                "concurrent": bool(concurrent),
                "terminal_disposition": str(terminal_disposition),
                "clock_identity": "time.monotonic_ns",
                **deepcopy(fields),
            }
            self._span_rows.append(row)
            self._span_by_id[decision_id] = row
            return decision_id
        except Exception:
            self.count_error()
            return None

    def record_backend_usage(
        self,
        *,
        request_id: str,
        prompt_tokens: int,
        completion_tokens: int,
        total_tokens: int,
        usage_source: str,
    ) -> None:
        """Join backend token use to a generation span by request identity."""

        if not self.enabled:
            return
        try:
            if prompt_tokens + completion_tokens != total_tokens:
                raise ValueError("backend token total does not reconcile")
            usage = {
                "request_id": str(request_id),
                "prompt_tokens": int(prompt_tokens),
                "completion_tokens": int(completion_tokens),
                "total_tokens": int(total_tokens),
                "usage_source": str(usage_source),
            }
            self._usage_by_request[str(request_id)] = usage
            matches = [row for row in self._span_rows if row.get("request_id") == str(request_id)]
            if not matches:
                raise KeyError(request_id)
            for row in matches:
                row.update(deepcopy(usage))
        except Exception:
            self.count_error()

    def _install_candidates(self, policy: Any) -> None:
        explorer = policy.explorer
        original = explorer._candidates

        def observed(_explorer: Any, *args: Any, **kwargs: Any) -> Any:
            with self.span("candidate_selection") as decision_id:
                result = original(*args, **kwargs)
                self.annotate(
                    decision_id,
                    candidate_count=len(result) if isinstance(result, Sequence) else None,
                )
                return result

        explorer._candidates = MethodType(observed, explorer)

    def _install_induction(self, policy: Any) -> None:
        original = policy._induce_and_plan_timed

        def observed(_policy: Any, *args: Any, **kwargs: Any) -> Any:
            with self.span("induction_and_generation") as decision_id:
                result = original(*args, **kwargs)
                self.annotate(
                    decision_id,
                    plan_disposition="plan_found"
                    if bool(getattr(policy, "plan", None))
                    else "no_plan",
                )
                return result

        policy._induce_and_plan_timed = MethodType(observed, policy)

    def _install_planner(self, policy: Any) -> None:
        original = policy._call_plan_in_model

        def observed(_policy: Any, *args: Any, **kwargs: Any) -> Any:
            with self.span("planner") as decision_id:
                result = original(*args, **kwargs)
                self.annotate(
                    decision_id,
                    plan_disposition="plan_found" if bool(result) else "no_plan",
                )
                return result

        policy._call_plan_in_model = MethodType(observed, policy)

    def install(self, policy: Any) -> Any:
        """Install exact passive wrappers only for an enabled profiling run."""

        if not self.enabled:
            return policy
        if getattr(policy, "_exp7491_timed_observer", None) is not None:
            raise ValueError("policy already has an Experiment 7491 observer")
        policy._exp7491_timed_observer = self
        self._install_candidates(policy)
        self._install_induction(policy)
        self._install_planner(policy)
        policy._decision_telemetry = _TimingRecorder(
            self,
            getattr(policy, "_decision_telemetry", NOOP_RECORDER),
        )
        return policy

    def _compute_exclusive(self) -> None:
        children: dict[str, list[JsonDict]] = {}
        for row in self._span_rows:
            parent = row.get("parent_decision_id")
            if isinstance(parent, str):
                children.setdefault(parent, []).append(row)
        for row in self._span_rows:
            start = row.get("start_monotonic_ns")
            end = row.get("end_monotonic_ns")
            if not isinstance(start, int) or not isinstance(end, int):
                continue
            intervals = [
                (
                    max(start, int(child["start_monotonic_ns"])),
                    min(end, int(child["end_monotonic_ns"])),
                )
                for child in children.get(str(row.get("decision_id")), [])
                if child.get("concurrent") is not True
                and isinstance(child.get("start_monotonic_ns"), int)
                and isinstance(child.get("end_monotonic_ns"), int)
            ]
            row["exclusive_ns"] = max(0, end - start - _interval_union(intervals))

    def finish(self, terminal_disposition: str) -> None:
        """Close, reconcile, and write once. Failures stay counted."""

        if not self.enabled or self._finished:
            return
        self._finished = True
        try:
            now = self.clock_ns() if self._open_span_ids else 0
            for decision_id in list(reversed(self._open_span_ids)):
                row = self._span_by_id.get(decision_id)
                if row is not None and row.get("end_monotonic_ns") is None:
                    row["end_monotonic_ns"] = now
                    row["terminal_disposition"] = terminal_disposition
            self._open_span_ids.clear()
            for row in self._span_rows:
                if row.get("seam") == "episode":
                    row["terminal_disposition"] = terminal_disposition
            self._compute_exclusive()
            self.event_path.parent.mkdir(parents=True, exist_ok=True)
            with self.event_path.open("w", encoding="utf-8") as handle:
                for row in self._span_rows:
                    handle.write(json.dumps(row, sort_keys=True, default=str) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
        except Exception:
            self.count_error()


def _rank_game(game: str) -> str:
    return hashlib.sha256(f"{PANEL_PREFIX}{game}".encode()).hexdigest()


def build_frozen_schedule(public_games: Sequence[str]) -> list[JsonDict]:
    """Rebuild the exact outcome-blind 36-unit schedule."""

    remaining = sorted(
        (game for game in set(map(str, public_games)) if game not in E4_GAMES),
        key=lambda game: (_rank_game(game), game),
    )
    selected = tuple(remaining[:12])
    if selected != PANEL_GAMES:
        raise ValueError(f"frozen panel mismatch: {selected}")
    rows: list[JsonDict] = []
    for game in selected:
        for seed in EPISODE_SEEDS:
            rows.append(
                {
                    "episode_id": f"{game}:seed-{seed}",
                    "game": game,
                    "seed": seed,
                    "execution_order": len(rows),
                }
            )
    return rows


def protocol_schedule(public_games: Sequence[str]) -> list[JsonDict]:
    """Add fixed live limits without changing the frozen identities."""

    return [
        {
            **row,
            "action_limit": ACTION_LIMIT,
            "episode_limit_s": EPISODE_LIMIT_S,
            "request_limit": REQUEST_LIMIT,
            "max_new_tokens_per_call": MAX_NEW_TOKENS,
            "adapter_disabled": True,
            "stored_engines_disabled": True,
            "banked_trajectories_disabled": True,
            "cross_game_state_disabled": True,
            "hidden_game_source_disabled": True,
            "offline_ground_truth_bfs_disabled": True,
            "withheld_inputs": list(WITHHELD_INPUTS),
        }
        for row in build_frozen_schedule(public_games)
    ]


def _gpu_one_reading(
    inventory: Sequence[Mapping[str, Any]], *, own_pid: int | None = None
) -> JsonDict | None:
    """Describe physical GPU 1 without importing or initializing a CUDA runtime."""

    for source in inventory:
        row = dict(source)
        if row.get("index") != GPU_INDEX:
            continue
        total = row.get("total_memory_mb")
        free = row.get("free_memory_mb")
        if not isinstance(total, int) or not isinstance(free, int):
            return None
        used = total - free
        compute_apps = [
            dict(app) for app in row.get("compute_apps") or [] if isinstance(app, Mapping)
        ]
        own_apps = [app for app in compute_apps if app.get("pid") == own_pid]
        foreign_apps = [app for app in compute_apps if app.get("pid") != own_pid]
        own_memory = sum(int(app.get("used_memory_mb") or 0) for app in own_apps)
        row["used_memory_mb"] = used
        row["own_pid"] = own_pid
        row["own_compute_apps"] = own_apps
        row["foreign_compute_apps"] = foreign_apps
        row["own_process_memory_mb"] = own_memory
        row["residual_used_memory_mb"] = max(0, used - own_memory)
        return row
    return None


def select_gpu_one(
    inventory: Sequence[Mapping[str, Any]], *, own_pid: int | None = None
) -> JsonDict | None:
    """Return GPU 1 when only this PID uses it and residual use is below 500 MiB."""

    row = _gpu_one_reading(inventory, own_pid=own_pid)
    if row is None:
        return None
    if row["residual_used_memory_mb"] >= GPU_IDLE_MAX_USED_MB:
        return None
    if row["foreign_compute_apps"]:
        return None
    return row


def _git_primary_root(root: Path) -> Path | None:
    """Resolve the shared primary checkout without a hardcoded path."""

    completed = subprocess.run(
        ("git", "rev-parse", "--path-format=absolute", "--git-common-dir"),
        cwd=root,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    if completed.returncode != 0:
        return None
    common = Path(completed.stdout.strip())
    return common.parent if common.name == ".git" else None


def resolve_environment_dir(root: Path) -> Path | None:
    """Find the public simulator directory without reading game source."""

    candidates = [root / "environment_files"]
    primary = _git_primary_root(root)
    if primary is not None:
        candidates.append(primary / "environment_files")
    return next((path for path in candidates if path.is_dir()), None)


def _check(check: str, expected: Any, observed: Any, *, passed: bool, path: str) -> JsonDict:
    return {
        "check": check,
        "expected": expected,
        "observed": observed,
        "passed": bool(passed),
        "path": path,
    }


def collect_static_preconditions(root: Path) -> tuple[list[JsonDict], list[JsonDict], Path | None]:
    """Validate frozen and source bytes before any runtime probe."""

    cited: list[JsonDict] = []
    checks: list[JsonDict] = []
    for relative in (
        SPEC_PATH,
        FROZEN_PANEL_PATH,
        MODULE_PATH,
        WRAPPER_PATH,
        TEST_PATH,
        PRIOR_E6_PATH,
        Path("python/carnot/experiment_7471_v654_arc_seam_observation.py"),
    ):
        path = root / relative
        present = path.is_file() and path.stat().st_size > 0
        checks.append(
            _check(
                f"source:{relative}",
                "readable_nonempty",
                "readable_nonempty" if present else None,
                passed=present,
                path=relative.as_posix(),
            )
        )
        if present:
            cited.append(
                {
                    "path": relative.as_posix(),
                    "sha256": sha256_file(path),
                    "bytes": path.stat().st_size,
                    "role": "protocol_or_source",
                }
            )
    frozen = load_json(root / FROZEN_PANEL_PATH)
    try:
        rebuilt = build_frozen_schedule(frozen.get("public_survey_games") or [])
    except ValueError:
        rebuilt = []
    checks.append(
        _check(
            "frozen_panel_identity",
            frozen.get("schedule"),
            rebuilt,
            passed=rebuilt == frozen.get("schedule") and len(rebuilt) == 36,
            path=FROZEN_PANEL_PATH.as_posix(),
        )
    )
    environment_dir = resolve_environment_dir(root)
    available = (
        {path.name for path in environment_dir.iterdir() if path.is_dir()}
        if environment_dir is not None
        else set()
    )
    missing = sorted(set(PANEL_GAMES) - available)
    checks.append(
        _check(
            "public_environment_panel_available",
            [],
            missing,
            passed=not missing,
            path="environment_files",
        )
    )
    return checks, cited, environment_dir


def collect_runtime_preconditions(
    root: Path, started: float
) -> tuple[list[JsonDict], list[JsonDict], JsonDict]:  # pragma: no cover - host resources.
    """Admit GPU 1 before CUDA initialization, then identify our own allocation."""

    inventory_before = exp7471.live_support.shipped_live.live_base._gpu_inventory()
    observed_before = _gpu_one_reading(inventory_before)
    gpu_before = select_gpu_one(inventory_before)
    admission = _check(
        "physical_gpu_1_idle_before_runtime_preflight",
        {
            "index": GPU_INDEX,
            "total_used_memory_mb_lt": GPU_IDLE_MAX_USED_MB,
            "foreign_compute_processes": 0,
        },
        observed_before,
        passed=gpu_before is not None,
        path="nvidia-smi_before_in_process_cuda_or_llama_cpp_initialization",
    )
    if gpu_before is None:
        return [admission], [], {"gpu": None}

    base_checks, base_hashes, resources = exp7471._runtime_preconditions(root, started)
    checks = [admission]
    checks.extend(dict(row) for row in base_checks if row.get("check") != "owned_gpu")
    own_pid = os.getpid()
    inventory_after = exp7471.live_support.shipped_live.live_base._gpu_inventory()
    observed_after = _gpu_one_reading(inventory_after, own_pid=own_pid)
    gpu_after = select_gpu_one(inventory_after, own_pid=own_pid)
    checks.append(
        _check(
            "physical_gpu_1_idle_after_runtime_preflight",
            {
                "index": GPU_INDEX,
                "allowed_compute_pid": own_pid,
                "residual_used_memory_mb_lt": GPU_IDLE_MAX_USED_MB,
                "foreign_compute_processes": 0,
            },
            observed_after,
            passed=gpu_after is not None,
            path="nvidia-smi_after_runtime_preflight",
        )
    )
    resources["gpu"] = gpu_after
    cited = [
        {"path": path, **dict(value)}
        for path, value in base_hashes.items()
        if isinstance(value, Mapping)
    ]
    return checks, cited, dict(resources)


def _owned_process_vram_mb(pid: int | None) -> int | None:  # pragma: no cover - host GPU.
    if not isinstance(pid, int):
        return None
    completed = subprocess.run(
        (
            "nvidia-smi",
            "--query-compute-apps=pid,used_memory",
            "--format=csv,noheader,nounits",
        ),
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    for line in completed.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) == 2 and parts[0] == str(pid):
            try:
                return int(parts[1])
            except ValueError:
                return None
    return None


def _find_parent_span(observer: E6TimedObserver, start_ns: int, end_ns: int) -> str | None:
    candidates = [
        row
        for row in observer._span_rows
        if row.get("seam") == "induction_and_generation"
        and row.get("request_id") is None
        and isinstance(row.get("start_monotonic_ns"), int)
        and isinstance(row.get("end_monotonic_ns"), int)
        and int(row["start_monotonic_ns"]) <= start_ns
        and end_ns <= int(row["end_monotonic_ns"])
    ]
    if not candidates:
        candidates = [
            row
            for row in observer._span_rows
            if row.get("seam") == "action_decision"
            and isinstance(row.get("start_monotonic_ns"), int)
            and isinstance(row.get("end_monotonic_ns"), int)
            and int(row["start_monotonic_ns"]) <= start_ns
            and end_ns <= int(row["end_monotonic_ns"])
        ]
    if not candidates:
        return next(
            (
                str(row["decision_id"])
                for row in observer._span_rows
                if row.get("seam") == "episode"
            ),
            None,
        )
    parent = min(
        candidates,
        key=lambda row: int(row["end_monotonic_ns"]) - int(row["start_monotonic_ns"]),
    )
    return str(parent["decision_id"])


def add_request_spans(
    observer: E6TimedObserver,
    budget_receipt: Mapping[str, Any],
    transport_rows: Sequence[Mapping[str, Any]],
) -> list[JsonDict]:
    """Join durable transport spans and usage through request IDs."""

    callbacks = {
        int(row.get("reservation_index") or 0): dict(row)
        for row in budget_receipt.get("callback_rows", [])
        if isinstance(row, Mapping)
    }
    usage_rows: list[JsonDict] = []
    for transport in transport_rows:
        index = int(transport.get("call_index") or 0)
        callback = callbacks.get(index, {})
        request_id = callback.get("request_id")
        started_s = transport.get("request_started_monotonic")
        elapsed_s = transport.get("elapsed_s")
        if not isinstance(request_id, str) or not isinstance(started_s, (int, float)):
            observer.count_error()
            continue
        if not isinstance(elapsed_s, (int, float)):
            elapsed_s = float(callback.get("elapsed_s") or 0.0)
        start_ns = int(float(started_s) * 1_000_000_000)
        end_ns = int((float(started_s) + max(0.0, float(elapsed_s))) * 1_000_000_000)
        disposition = str(callback.get("disposition") or "in_flight")
        observer.add_completed_span(
            "induction_and_generation",
            start_ns=start_ns,
            end_ns=end_ns,
            parent_decision_id=_find_parent_span(observer, start_ns, end_ns),
            terminal_disposition=disposition,
            request_id=request_id,
            generation_transport=True,
        )
        response_path = transport.get("response_path")
        response = load_json(Path(response_path)) if isinstance(response_path, str) else {}
        usage = response.get("usage")
        if not isinstance(usage, Mapping):
            continue
        prompt = usage.get("prompt_tokens")
        completion = usage.get("completion_tokens")
        total = usage.get("total_tokens")
        if not all(isinstance(value, int) for value in (prompt, completion, total)):
            observer.count_error()
            continue
        assert isinstance(prompt, int)
        assert isinstance(completion, int)
        assert isinstance(total, int)
        observer.record_backend_usage(
            request_id=request_id,
            prompt_tokens=prompt,
            completion_tokens=completion,
            total_tokens=total,
            usage_source=str(response_path),
        )
        usage_rows.append(
            {
                "episode_id": observer.episode_id,
                "request_id": request_id,
                "prompt_tokens": prompt,
                "completion_tokens": completion,
                "total_tokens": total,
                "usage_source": str(response_path),
            }
        )
    return usage_rows


def normalize_episode_cost(
    episode: Mapping[str, Any], spans: Sequence[Mapping[str, Any]]
) -> JsonDict:
    """Convert exclusive spans into the six E6 decision points."""

    totals = dict.fromkeys(DECISION_POINTS, 0)
    for span in spans:
        exclusive = span.get("exclusive_ns")
        if not isinstance(exclusive, int):
            continue
        seam = str(span.get("seam"))
        point = {
            "action_decision": "candidate_selection",
            "candidate_selection": "candidate_selection",
            "induction_and_generation": "induction_and_generation",
            "world_model_verification": "world_model_verification",
            "supervisor": "supervisor",
            "planner": "planner",
            "environment_step": "environment",
        }.get(seam)
        if point is not None:
            totals[point] += exclusive
    episode_span = next((row for row in spans if row.get("seam") == "episode"), {})
    start = episode_span.get("start_monotonic_ns")
    end = episode_span.get("end_monotonic_ns")
    wall_ns = end - start if isinstance(start, int) and isinstance(end, int) else None
    usage_rows = episode.get("backend_usage_rows") or []
    backend_tokens = sum(
        int(row.get("total_tokens") or 0) for row in usage_rows if isinstance(row, Mapping)
    )
    request_ids = {str(row.get("request_id")) for row in usage_rows if isinstance(row, Mapping)}
    generation_requests = {
        str(row.get("request_id"))
        for row in spans
        if row.get("generation_transport") is True and row.get("request_id")
    }
    token_join_passed = generation_requests == request_ids
    reconciliation = reconcile_spans(spans)
    return {
        "episode_id": episode.get("episode_id"),
        "game": episode.get("game"),
        "complete": episode.get("disposition") == "complete",
        "current_model": True,
        "numeric_eligible": episode.get("disposition") == "complete",
        "temperature": "cold" if episode.get("execution_order") == 0 else "warm",
        "episode_wall_s": None if wall_ns is None else wall_ns / 1_000_000_000,
        "backend_usage_tokens": backend_tokens,
        "phases": [
            {
                "decision_point": point,
                "wall_s": totals[point] / 1_000_000_000,
                "tokens": backend_tokens if point == "induction_and_generation" else 0,
                "concurrent": False,
            }
            for point in DECISION_POINTS
        ],
        "reconciliation_passed": reconciliation["passed"],
        "token_join_passed": token_join_passed,
        "attributed_wall_fraction": (
            None if not isinstance(wall_ns, int) or wall_ns <= 0 else sum(totals.values()) / wall_ns
        ),
        "cohort": "experiment_7491_timed",
    }


class EpisodeTimeout(Exception):
    """Stop one bounded episode."""


def _run_policy_episode(  # pragma: no cover - live model and public simulator.
    schedule: Mapping[str, Any], proposer: Any, capture: Any, event_path: Path, action_path: Path
) -> JsonDict:
    """Run one existing E3 policy episode with passive spans."""

    from arcengine import GameAction
    from carnot.agentic import arc_executable_world_model as e3
    from carnot.agentic import arc_solver_kit as kit
    from carnot.agentic.arc_competition_agent import make_carnot_agent

    episode_id = str(schedule["episode_id"])
    game = str(schedule["game"])
    seed = int(schedule["seed"])
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed % (2**32 - 1))
    except ImportError:
        pass
    os.environ["CARNOT_ARC_RANDOM_SEED"] = str(seed)
    os.environ["CARNOT_ARC_GENERATOR_SEED"] = str(seed)
    os.environ["CARNOT_ARC_TRAJECTORY_SUPERVISOR_WINDOW"] = str(exp7471.SUPERVISOR_THRESHOLD)
    os.environ.pop("CARNOT_ARC_TRAJECTORY_SUPERVISOR", None)
    os.environ.pop("CARNOT_ARC_SUPERVISOR_TOOL_ARM", None)

    environment_dir = os.environ.get("CARNOT_ARC_PUBLIC_ENV_DIR")
    if environment_dir:
        kit.ENV_DIR = Path(environment_dir)
    episode_dir = event_path.parent / "episodes" / episode_id.replace(":", "__")
    episode_dir.mkdir(parents=True, exist_ok=True)
    span_path = episode_dir / "exclusive_spans.jsonl"
    old_e3_dir = e3.E3_DIR
    e3.E3_DIR = episode_dir / "fresh_e3"
    capture.begin_episode(episode_id)
    budget = exp7471.live_support.DurableEpisodeRequestBudget(
        episode_id,
        limit=REQUEST_LIMIT,
        deadline_s=EPISODE_LIMIT_S,
        event_path=event_path,
    )
    attach_request_budget(proposer, budget)

    class LocalAgentBase:
        def __init__(self, game_id: str) -> None:
            self.game_id = game_id

    originals = exp7471._disable_cross_game_loaders()
    previous_telemetry_episode = os.environ.get(TELEMETRY_EPISODE_ENV)
    os.environ[TELEMETRY_EPISODE_ENV] = episode_id
    try:
        agent_type = make_carnot_agent(LocalAgentBase, cascade=True, proposer=proposer)
        agent = agent_type(game_id=game)
    finally:
        if previous_telemetry_episode is None:
            os.environ.pop(TELEMETRY_EPISODE_ENV, None)
        else:
            os.environ[TELEMETRY_EPISODE_ENV] = previous_telemetry_episode
        exp7471._restore_cross_game_loaders(originals)
    policy = agent._policy
    observer = E6TimedObserver(episode_id, span_path, enabled=True)
    observer.install(policy)
    arcade = kit.offline_arcade()
    env = arcade.make(game, scorecard_id=arcade.open_scorecard())
    frames: list[Any] = []
    latest: Any = None
    action_rows: list[JsonDict] = []
    trace: list[JsonDict] = []
    entered = time.monotonic()
    start_level: int | None = None
    peak_level = 0
    terminal_level = 0
    disposition = "complete"
    error: str | None = None
    episode_context = observer.episode("complete")
    episode_context.__enter__()

    def alarm_handler(_signum: int, _frame: Any) -> None:
        raise EpisodeTimeout(f"episode exceeded {EPISODE_LIMIT_S}s")

    previous_alarm = signal.signal(signal.SIGALRM, alarm_handler)
    signal.setitimer(signal.ITIMER_REAL, EPISODE_LIMIT_S)
    episode_exception: BaseException | None = None
    try:
        for action_index in range(ACTION_LIMIT):
            if agent.is_done(frames, latest):
                break
            with observer.span("action_decision"):
                action_started = time.monotonic_ns()
                action = agent.choose_action(frames, latest)
                name = str(getattr(action, "name", action))
                data_value = getattr(action, "action_data", None)
                data = data_value.model_dump() if hasattr(data_value, "model_dump") else None
                if isinstance(data, Mapping):
                    data = {key: value for key, value in data.items() if key != "game_id"}
                with observer.span(
                    "environment_step", operation="reset" if name == "RESET" else "step"
                ):
                    latest = (
                        env.reset()
                        if name == "RESET"
                        else env.step(
                            action if isinstance(action, GameAction) else action, data=data
                        )
                    )
                level = exp7471.live_support._level(latest)
                start_level = level if start_level is None else start_level
                peak_level = max(peak_level, level)
                terminal_level = level
                trace.append({"action": name, "data": deepcopy(data)})
                row = {
                    "episode_id": episode_id,
                    "action_index": action_index + 1,
                    "action": name,
                    "data": deepcopy(data),
                    "interval_start_monotonic_ns": action_started,
                    "interval_end_monotonic_ns": time.monotonic_ns(),
                    "level": level,
                    "state_sha256": exp7471.canonical_hash(str(latest)),
                    "later_progress": False,
                }
                action_rows.append(row)
                exp7471._append_jsonl(action_path, {"event": "action_end", **row})
                frames.append(latest)
    except EpisodeTimeout as exc:
        episode_exception = exc
        disposition = "censored_timeout"
        error = f"{type(exc).__name__}: {exc}"
    except BaseException as exc:
        episode_exception = exc
        disposition = "complete_error" if action_rows else "censored_no_first_action"
        error = f"{type(exc).__name__}: {exc}"[:500]
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_alarm)
        budget.cancel("episode_terminal")
        e3.E3_DIR = old_e3_dir
        episode_context.__exit__(
            type(episode_exception) if episode_exception is not None else None,
            episode_exception,
            episode_exception.__traceback__ if episode_exception is not None else None,
        )

    transport_rows = exp7471.live_support._transport_rows(
        exp7471.live_support._read_jsonl(event_path), episode_id
    )
    budget_receipt = budget.receipt()
    usage_rows = add_request_spans(observer, budget_receipt, transport_rows)
    policy.finish_decision_telemetry(
        level_end=terminal_level if start_level is not None else None,
        actions_used=len(action_rows),
    )
    observer.finish(disposition)
    spans = read_span_rows(span_path)
    for index, row in enumerate(action_rows):
        row["later_progress"] = any(
            int(later.get("level") or 0) > int(row.get("level") or 0)
            for later in action_rows[index + 1 :]
        )
    reached = start_level is not None and peak_level > start_level
    reproduction = (
        exp7471._reproduce_trace(game, trace, peak_level)
        if reached
        else {"attempted": False, "passed": False, "reason": "no_level_reached"}
    )
    row = {
        **deepcopy(dict(schedule)),
        "disposition": disposition,
        "policy_entry": {
            "factory": "make_carnot_agent",
            "policy_class": type(policy).__name__,
            "choose_action_path": True,
            "is_done_path": True,
            "instrumentation_default_off": True,
            "withheld_inputs": list(WITHHELD_INPUTS),
        },
        "action_count": len(action_rows),
        "start_level": start_level,
        "peak_level": peak_level if start_level is not None else None,
        "terminal_level": terminal_level if start_level is not None else None,
        "action_rows": action_rows,
        "request_budget_receipt": budget_receipt,
        "server_request_rows": transport_rows,
        "backend_usage_rows": usage_rows,
        "elapsed_s": round(time.monotonic() - entered, 6),
        "solve_provenance": "live_agent_self_discovery",
        "trace_reproduction": reproduction,
        "new_level_credit": 0,
        "span_path": str(span_path),
        "span_reconciliation": reconcile_spans(spans),
        "recorder_error_count": int(
            getattr(policy._decision_telemetry, "error_count", observer.error_count) or 0
        ),
        "decision_telemetry_path": os.environ.get(TELEMETRY_PATH_ENV),
        "error": error,
    }
    row["normalized_cost"] = normalize_episode_cost(row, spans)
    return row


def _unstarted_row(schedule: Mapping[str, Any], disposition: str = "unstarted") -> JsonDict:
    return {
        **deepcopy(dict(schedule)),
        "disposition": disposition,
        "action_count": 0,
        "start_level": None,
        "peak_level": None,
        "terminal_level": None,
        "action_rows": [],
        "backend_usage_rows": [],
        "elapsed_s": 0.0,
        "solve_provenance": "live_agent_self_discovery",
        "trace_reproduction": {"attempted": False, "passed": False},
        "new_level_credit": 0,
        "span_reconciliation": {"passed": False, "reason": disposition},
        "recorder_error_count": 0,
        "error": None,
        "normalized_cost": None,
    }


def _configure_live_driver() -> None:
    """Point the qualified 7471 process supervisor at Experiment 7491."""

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
        "AGGREGATE_LIVE_LIMIT_S": TOTAL_LIVE_LIMIT_S,
    }
    for name, value in values.items():
        setattr(exp7471, name, value)


def run_live_session(args: argparse.Namespace) -> int:  # pragma: no cover - owned child.
    """Load one owned server and run all frozen episodes on GPU 1."""

    started = time.monotonic()
    deadline_ns = int(
        os.environ.get(
            "CARNOT_E6_DEADLINE_MONOTONIC_NS",
            str(time.monotonic_ns() + int(TOTAL_LIVE_LIMIT_S * 1_000_000_000)),
        )
    )
    raw_dir = Path(args.raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    event_path = raw_dir / RUNTIME_EVENT_PATH.name
    action_path = raw_dir / ACTION_PATH.name
    schedule = load_json(Path(args.schedule_path)).get("rows") or []
    capture = exp7471.live_support.DurableRequestCapture(raw_dir, event_path)
    proposer: Any = None
    rows: list[JsonDict] = []
    session: JsonDict = {
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
        owned_vram = _owned_process_vram_mb(server_pid)
        offload_ok = isinstance(owned_vram, int) and OFFLOAD_MIN_MB <= owned_vram <= OFFLOAD_MAX_MB
        session["runtime_receipt"] = {
            "child_pid": os.getpid(),
            "server_pid": server_pid,
            "physical_gpu_index": GPU_INDEX,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "owned_server_vram_mb_after_load": owned_vram,
            "owned_server_vram_mb_before_load": 0,
            "owned_server_vram_delta_mb": owned_vram,
            "offload_expected_range_mb": [OFFLOAD_MIN_MB, OFFLOAD_MAX_MB],
            "offload_real": offload_ok,
            "native_binary": proposer.last_launch_argv[0] if proposer.last_launch_argv else None,
            "server_command": list(proposer.last_launch_argv),
            "requested_n_gpu_layers": 999,
            "n_ctx": 49_152,
            "kv_quantization": "q8_0",
            "embedded_tokenizer": True,
            "use_chat_template": True,
            "mtp": False,
            "max_new_tokens": MAX_NEW_TOKENS,
            "request_limit_per_episode": REQUEST_LIMIT,
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
            offload_real=offload_ok,
        )
        if not offload_ok:
            raise RuntimeError("owned server did not show the required near-18-GB CUDA offload")
        for index, sealed in enumerate(schedule):
            remaining_s = (deadline_ns - time.monotonic_ns()) / 1_000_000_000
            if remaining_s <= EPISODE_LIMIT_S:
                rows.extend(_unstarted_row(row, "unstarted") for row in schedule[index:])
                break
            progress(
                started,
                "episode",
                "before",
                episode_id=sealed["episode_id"],
                completed_units=index,
            )
            row = _run_policy_episode(sealed, proposer, capture, event_path, action_path)
            rows.append(row)
            current_work_receipt.atomic_json(raw_dir / "episode_rows.json", {"rows": rows})
            current_work_receipt.atomic_json(
                Path(args.checkpoint_path),
                {
                    "stage": "episodes",
                    "model_loaded": True,
                    "completed_units": len(rows),
                    "total_units": len(schedule),
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
            minimum_opportunities = int(
                os.environ.get("CARNOT_B2_MIN_GATE_OPPORTUNITIES", "0") or 0
            )
            minimum_attempts = int(os.environ.get("CARNOT_B2_MIN_INDUCTION_ATTEMPTS", "0") or 0)
            telemetry_path = os.environ.get(TELEMETRY_PATH_ENV)
            if minimum_opportunities and minimum_attempts and telemetry_path:
                telemetry_rows = load_telemetry(telemetry_path)
                opportunity_count = sum(
                    row.get("record_type") == "decision"
                    and row.get("seam") == "induction_timing"
                    and row.get("gate_decision") is not None
                    for row in telemetry_rows
                )
                attempt_count = sum(
                    row.get("record_type") == "induction_attempt" for row in telemetry_rows
                )
                progress(
                    started,
                    "sample_floor",
                    "observed",
                    gate_opportunities=opportunity_count,
                    induction_attempts=attempt_count,
                )
                if opportunity_count >= minimum_opportunities and attempt_count >= minimum_attempts:
                    rows.extend(_unstarted_row(item, "unstarted") for item in schedule[index + 1 :])
                    break
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


def _artifact_checksum(artifact: Mapping[str, Any]) -> str:
    copied = deepcopy(dict(artifact))
    copied["reproducibility_checksum"] = ""
    return canonical_hash(copied)


def build_blocked_artifact(
    *,
    failed_check: Mapping[str, Any],
    duration_s: float,
    cited_artifacts: Sequence[Mapping[str, Any]],
    preconditions_checked: Sequence[Mapping[str, Any]] | None = None,
    schedule: Sequence[Mapping[str, Any]] = (),
    model_specs: Sequence[Mapping[str, Any]] = (),
    inference_substrate: str = "no_model_load",
    inference_substrate_class: str = "no_model_load",
    model_invoked: bool = False,
    invocation_counts: Mapping[str, Any] | None = None,
) -> JsonDict:
    """Build one schema-complete blocker without dropping scheduled units."""

    rows = [_unstarted_row(row) for row in schedule]
    dispositions = Counter(str(row.get("disposition")) for row in rows)

    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "run_date": RUN_DATE,
        "status": "blocked_precondition",
        "honest_verdict": f"blocked_{failed_check.get('check', 'precondition')}",
        "verdict_class": "blocked",
        "inference_substrate": inference_substrate,
        "inference_substrate_class": inference_substrate_class,
        "execution_venue": EXECUTION_VENUE,
        "model_invoked": bool(model_invoked),
        "invocation_counts": deepcopy(dict(invocation_counts or {})),
        "MODEL_SPECS": deepcopy(list(model_specs)),
        "model_specs": deepcopy(list(model_specs)),
        "random_seed": {"episodes": list(EPISODE_SEEDS), "ordering": 7_491},
        "duration_s": max(0.000001, float(duration_s)),
        "preconditions_checked": deepcopy(list(preconditions_checked or [dict(failed_check)])),
        "protocol_schedule": deepcopy(list(schedule)),
        "rows": rows,
        "per_game_results": [
            {
                "game": game,
                "episodes": [row for row in rows if row.get("game") == game],
            }
            for game in PANEL_GAMES
            if any(row.get("game") == game for row in rows)
        ],
        "sample_size_budget": {
            "planned_units": len(schedule),
            "complete_units": dispositions["complete"],
            "failed_units": dispositions["failed"] + dispositions["complete_error"],
            "censored_units": sum(
                count for name, count in dispositions.items() if name.startswith("censored_")
            ),
            "unstarted_units": dispositions["unstarted"],
        },
        "cited_artifacts": deepcopy(list(cited_artifacts)),
        "solve_provenance": "live_agent_self_discovery",
        "hidden_game_efficacy_claim": False,
        "new_level_credit": 0,
        "flagged_adversarial": False,
        "reproducibility_checksum": "",
    }
    artifact["reproducibility_checksum"] = _artifact_checksum(artifact)
    return artifact


def build_terminal_artifact(
    *,
    started_at_utc: str,
    duration_s: float,
    preconditions: Sequence[Mapping[str, Any]],
    schedule: Sequence[Mapping[str, Any]],
    session: Mapping[str, Any],
    model_specs: Sequence[Mapping[str, Any]],
    cited_artifacts: Sequence[Mapping[str, Any]],
    boundary_events: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Build the timed terminal artifact from current raw rows."""

    observed = {
        str(row.get("episode_id")): dict(row)
        for row in session.get("episodes", [])
        if isinstance(row, Mapping) and row.get("episode_id")
    }
    rows = [
        {**deepcopy(dict(sealed)), **deepcopy(observed[str(sealed["episode_id"])])}
        if str(sealed["episode_id"]) in observed
        else _unstarted_row(sealed)
        for sealed in schedule
    ]
    dispositions = Counter(str(row.get("disposition")) for row in rows)
    runtime = dict(session.get("runtime_receipt") or {})
    invocation = exp7471._invocation_reduction(boundary_events, child_terminal=True)
    span_shards: list[JsonDict] = []
    for row in rows:
        raw = row.get("span_path")
        if not isinstance(raw, str):
            continue
        path = Path(raw)
        if path.is_file():
            try:
                relative = path.relative_to(REPO_ROOT).as_posix()
            except ValueError:
                relative = str(path)
            span_shards.append(
                {
                    "episode_id": row.get("episode_id"),
                    "path": relative,
                    "sha256": sha256_file(path),
                    "bytes": path.stat().st_size,
                    "row_count": len(read_span_rows(path)),
                }
            )
    normalized = [
        deepcopy(dict(row["normalized_cost"]))
        for row in rows
        if isinstance(row.get("normalized_cost"), Mapping)
    ]
    offload_real = runtime.get("offload_real") is True
    terminal_valid = len(rows) == 36 and all(
        row.get("disposition") in TERMINAL_DISPOSITIONS for row in rows
    )
    recorder_errors = sum(int(row.get("recorder_error_count") or 0) for row in rows)
    status = (
        "complete_timed_live_profile"
        if offload_real and terminal_valid and recorder_errors == 0
        else "complete_timed_live_profile_with_incomplete_instrumentation"
    )
    per_game = [
        {
            "game": game,
            "episodes": [
                {
                    key: row.get(key)
                    for key in (
                        "episode_id",
                        "seed",
                        "disposition",
                        "action_count",
                        "elapsed_s",
                        "start_level",
                        "peak_level",
                        "terminal_level",
                        "solve_provenance",
                        "recorder_error_count",
                    )
                }
                for row in rows
                if row.get("game") == game
            ],
        }
        for game in PANEL_GAMES
    ]
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "experiment_name": EXPERIMENT_NAME,
        "run_date": RUN_DATE,
        "started_at_utc": started_at_utc,
        "ended_at_utc": utc_now(),
        "status": status,
        "honest_verdict": status,
        "verdict_class": "null",
        "inference_substrate": invocation["inference_substrate"],
        "inference_substrate_class": invocation["inference_substrate_class"],
        "execution_venue": EXECUTION_VENUE,
        "execution_venue_details": runtime,
        "model_invoked": invocation["model_invoked"],
        "invocation_counts": invocation["invocation_counts"],
        "MODEL_SPECS": deepcopy(list(model_specs)),
        "model_specs": deepcopy(list(model_specs)),
        "random_seed": {
            "selection": 7_491,
            "episodes": list(EPISODE_SEEDS),
            "ordering": 7_491,
        },
        "duration_s": max(0.000001, float(duration_s)),
        "preconditions_checked": deepcopy(list(preconditions)),
        "protocol_schedule": deepcopy(list(schedule)),
        "rows": rows,
        "normalized_cost_rows": normalized,
        "per_game_results": per_game,
        "sample_size_budget": {
            "planned_units": 36,
            "complete_units": dispositions["complete"],
            "failed_units": dispositions["failed"] + dispositions["complete_error"],
            "censored_units": sum(
                count for name, count in dispositions.items() if name.startswith("censored_")
            ),
            "unstarted_units": dispositions["unstarted"],
            "independent_game_clusters": 12,
        },
        "span_shards": span_shards,
        "recorder_error_count": recorder_errors,
        "cited_artifacts": deepcopy(list(cited_artifacts)),
        "solve_provenance": "live_agent_self_discovery",
        "generalization_scope": "public_adapter_withheld_development_proxy",
        "hidden_game_efficacy_claim": False,
        "speed_claim": None,
        "new_level_credit": 0,
        "remote_submission": False,
        "submission_kernel_changed": False,
        "flagged_adversarial": False,
        "reproducibility_checksum": "",
    }
    artifact["reproducibility_checksum"] = _artifact_checksum(artifact)
    return artifact


def run_experiment(root: Path) -> JsonDict:  # pragma: no cover - orchestration.
    """Check exact prerequisites, run the owned child, and publish once."""

    started = time.monotonic()
    started_at = utc_now()
    progress(started, "1", "static_preconditions_before")
    static_checks, cited, environment_dir = collect_static_preconditions(root)
    progress(
        started,
        "1",
        "static_preconditions_after",
        passed=all(row["passed"] for row in static_checks),
    )
    if not all(row["passed"] for row in static_checks):
        failure = next(row for row in static_checks if row["passed"] is not True)
        artifact = build_blocked_artifact(
            failed_check=failure,
            duration_s=time.monotonic() - started,
            cited_artifacts=cited,
        )
        write_json(root / RESULT_PATH, artifact)
        return artifact
    frozen = load_json(root / FROZEN_PANEL_PATH)
    schedule = protocol_schedule(frozen["public_survey_games"])
    write_json(
        root / SCHEDULE_PATH, {"rows": schedule, "frozen_panel": FROZEN_PANEL_PATH.as_posix()}
    )

    progress(started, "2", "runtime_preconditions_before")
    runtime_checks, runtime_cited, resources = collect_runtime_preconditions(root, started)
    cited.extend(runtime_cited)
    all_checks = [*static_checks, *runtime_checks]
    progress(
        started,
        "2",
        "runtime_preconditions_after",
        passed=all(row.get("passed") is True for row in runtime_checks),
    )
    if not all(row.get("passed") is True for row in runtime_checks):
        failure = next(row for row in runtime_checks if row.get("passed") is not True)
        artifact = build_blocked_artifact(
            failed_check=failure,
            duration_s=time.monotonic() - started,
            cited_artifacts=cited,
            preconditions_checked=all_checks,
            schedule=schedule,
        )
        write_json(root / RESULT_PATH, artifact)
        return artifact
    if resources.get("gpu") is None or environment_dir is None:
        raise RuntimeError("validated runtime resources disappeared")

    _configure_live_driver()
    for relative in (
        BOUNDARY_PATH,
        RUNTIME_EVENT_PATH,
        ACTION_PATH,
        SESSION_PATH,
        CHECKPOINT_PATH,
        RAW_DIR / "episode_rows.json",
    ):
        (root / relative).unlink(missing_ok=True)
    os.environ["CARNOT_ARC_PUBLIC_ENV_DIR"] = str(environment_dir)
    remaining_budget_s = max(0.0, TOTAL_LIVE_LIMIT_S - (time.monotonic() - started))
    if remaining_budget_s <= 0:
        failure = _check(
            "aggregate_budget_before_model_load",
            ">0s",
            remaining_budget_s,
            passed=False,
            path="time.monotonic",
        )
        artifact = build_blocked_artifact(
            failed_check=failure,
            duration_s=time.monotonic() - started,
            cited_artifacts=cited,
            preconditions_checked=[*all_checks, failure],
            schedule=schedule,
        )
        write_json(root / RESULT_PATH, artifact)
        return artifact
    exp7471.AGGREGATE_LIVE_LIMIT_S = remaining_budget_s
    os.environ["CARNOT_E6_DEADLINE_MONOTONIC_NS"] = str(
        time.monotonic_ns() + int(remaining_budget_s * 1_000_000_000)
    )
    progress(started, "3", "live_run_before", planned_units=36, gpu_index=GPU_INDEX)
    session = exp7471.run_child_with_lease(
        resources=resources,
        schedule_path=root / SCHEDULE_PATH,
        started=started,
    )
    progress(
        started,
        "3",
        "live_run_after",
        observed_units=len(session.get("episodes") or []),
    )
    boundary_events = InvocationBoundaryLedger(root / BOUNDARY_PATH).read_events()
    for relative, role in (
        (SCHEDULE_PATH, "frozen_protocol"),
        (SESSION_PATH, "live_session"),
        (BOUNDARY_PATH, "current_invocation_ledger"),
        (RUNTIME_EVENT_PATH, "request_events"),
        (ACTION_PATH, "action_events"),
    ):
        path = root / relative
        if path.is_file():
            cited.append(
                {
                    "path": relative.as_posix(),
                    "sha256": sha256_file(path),
                    "bytes": path.stat().st_size,
                    "role": role,
                }
            )
    runtime = dict(session.get("runtime_receipt") or {})
    offload_check = _check(
        "owned_server_cuda_offload_near_18gb",
        {"minimum_mb": OFFLOAD_MIN_MB, "maximum_mb": OFFLOAD_MAX_MB},
        runtime.get("owned_server_vram_mb_after_load"),
        passed=runtime.get("offload_real") is True,
        path="nvidia-smi_compute_process_used_gpu_memory",
    )
    all_checks.append(offload_check)
    if not offload_check["passed"]:
        invocation = exp7471._invocation_reduction(boundary_events, child_terminal=True)
        artifact = build_blocked_artifact(
            failed_check=offload_check,
            duration_s=time.monotonic() - started,
            cited_artifacts=cited,
            preconditions_checked=all_checks,
            schedule=schedule,
            model_specs=[resources["model_spec"]],
            inference_substrate=invocation["inference_substrate"],
            inference_substrate_class=invocation["inference_substrate_class"],
            model_invoked=invocation["model_invoked"],
            invocation_counts=invocation["invocation_counts"],
        )
        write_json(root / RESULT_PATH, artifact)
        return artifact
    progress(started, "4", "terminal_build_before")
    artifact = build_terminal_artifact(
        started_at_utc=started_at,
        duration_s=time.monotonic() - started,
        preconditions=all_checks,
        schedule=schedule,
        session=session,
        model_specs=[resources["model_spec"]],
        cited_artifacts=cited,
        boundary_events=boundary_events,
    )
    write_json(root / RESULT_PATH, artifact)
    progress(started, "4", "terminal_build_after", verdict=artifact["honest_verdict"])
    return artifact


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", choices=[RUN_DATE])
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
    args = parser.parse_args(argv)
    if args.date is None:
        parser.error("--date is required")
    return args


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - thin CLI.
    args = parse_args(argv)
    if args.role == "live-session":
        _configure_live_driver()
        return run_live_session(args)
    artifact = run_experiment(REPO_ROOT)
    return 0 if str(artifact.get("honest_verdict", "")).startswith(("complete_", "blocked_")) else 1


__all__ = [
    "E4_GAMES",
    "EPISODE_SEEDS",
    "E6TimedObserver",
    "FROZEN_PANEL_PATH",
    "NOOP_RECORDER",
    "PANEL_GAMES",
    "REPO_ROOT",
    "add_request_spans",
    "build_blocked_artifact",
    "build_frozen_schedule",
    "build_terminal_artifact",
    "main",
    "normalize_episode_cost",
    "read_span_rows",
    "reconcile_spans",
    "select_gpu_one",
    "write_json",
]
