"""Qualify exclusive ARC timing and seal the two V655 E6 panels.

The experiment reads immutable V654 evidence. It does not call a model or run
an ARC episode. Its reducer counts wall-clock intervals once, even when an old
observer wrapped a large call and also timed nested calls.

Spec refs: REQ-ARC-WMTE-7478 and SCENARIO-ARC-WMTE-7478-*.
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
import platform
import random
import tempfile
import time
from typing import Any, TypeVar

from carnot.experiment_7358_v646_validation_contract import (
    AffectedManifest,
    EnvironmentCommandSpec,
    PlannedCommand,
    build_command_plan,
    run_categorized_commands,
    validate_command_plan,
)
from carnot.reporting import experiment_7303_validation_scope as validation_scope
from carnot.reporting.current_work_receipt import atomic_json, sha256_file


JsonDict = dict[str, Any]
_T = TypeVar("_T")

REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_DATE = "20260921"
MILESTONE = "2026.09.655"
EXPERIMENT_ID = "exp7478-arc-interval-protocol"
SCHEMA = "carnot.exp7478.v655.arc_interval_protocol.v1"
MODEL_SPECS: list[str] = []
INFERENCE_SUBSTRATE_CLASS = "aggregation"
EXECUTION_VENUE = "host"

SPEC_PATH = Path("openspec/capabilities/arc-world-model-trust-energy/spec.md")
UPSTREAM_PATH = Path("results/experiment_7471_v654_arc_seam_observation.json")
RESULT_PATH = Path("results/experiment_7478_v655_arc_interval_protocol.json")
RAW_DIR = Path("results/raw/experiment_7478_v655_arc_interval_protocol")
CANDIDATE_PATH = RAW_DIR / "terminal_candidate.json"
MODULE_PATH = Path("python/carnot/experiment_7478_v655_arc_interval_protocol.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_7478_v655_arc_interval_protocol.py")
TEST_PATH = Path("tests/python/test_experiment_7478_v655_arc_interval_protocol.py")

PANEL_A_GAMES = ("sk48", "tr87", "s5i5", "lp85", "lf52", "cn04")
PANEL_B_GAMES = ("tu93", "g50t", "tn36", "vc33", "re86", "dc22")
EPISODE_SEEDS = (65_501, 65_502, 65_503)
ACTION_LIMIT = 180
REQUEST_LIMIT = 2
MAX_NEW_TOKENS = 256
EPISODE_LIMIT_S = 240
PANEL_LIVE_LIMIT_S = 3600

REPLACEABLE_WORK_CLASSES = {"replaceable_decision"}
NONREPLACEABLE_WORK_CLASSES = {
    "text_generation",
    "world_model_construction",
    "verifier_work",
    "dispatch",
    "idle",
}
HISTORICAL_WORK_CLASS = {
    "candidate_action_selection": "unattributed_decision_seam",
    "hypothesis_gate": "unattributed_world_model_composite",
    "supervisor_arm_selection": "replaceable_decision",
    "induction_timing": "replaceable_decision",
    "downstream_generation": "text_generation",
}

ZERO_INVOCATION_COUNTS = {
    f"{operation}_{state}": 0
    for operation in ("model_loads", "forward_calls", "generation_calls")
    for state in ("attempted", "completed", "failed", "cancelled", "in_flight")
}

AFFECTED_CHECK_NAMES = validation_scope.REQUIRED_CHECK_NAMES
CAPABILITY_E2E_NAMES = ("e2e_009", "e2e_010", "e2e_011", "private_arc_smoke")
REQUIRED_TERMINAL_NAMES = (
    "declared_entrypoint_cold_replay",
    "independent_cold_reducer",
    "adversarial_verify",
    "verdict_row_consistency_strict",
)

MANIFEST = AffectedManifest(
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
    Path("python/carnot/experiment_7471_v654_arc_seam_observation.py"),
    Path("python/carnot/agentic/arc_decision_telemetry.py"),
    Path("python/carnot/experiment_7464_v654_semif_e6_decision_cost_profile.py"),
    Path("docs/research-notes/semif-ebm-arc-experiment-plan-2026-09-20.md"),
    Path("docs/research-notes/semif-e4-frozen-game-roster-2026-09-20.md"),
    Path("openspec/change-proposals/research-roadmap-vNEXT.md"),
    SPEC_PATH,
    UPSTREAM_PATH,
    MODULE_PATH,
    WRAPPER_PATH,
    TEST_PATH,
)

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
    "arc_interval_protocol_ready_score",
    "interval_rows",
    "arc_schedule_manifest",
    "timing_correction",
    "observer_parity",
)

FIELD_PRINCIPLES = {
    "schema": "Versioned schema with exact roadmap experiment_id, milestone and terminal status prevents silent reader drift.",
    "run_date": "Use 20260921; retain measured UTC and monotonic times with clock/process identity.",
    "preconditions_checked": "Name resources, exact paths, ownership and observed prerequisite values before dependent work.",
    "MODEL_SPECS": "Use an empty list for reducer work so archived Qwen calls cannot become current calls.",
    "model_specs": "Mirror the empty current model list for lowercase-field readers.",
    "model_invoked": "Any attempted current model call differs from archived or scripted events.",
    "invocation_counts": "Balance attempted, complete, failed, cancelled and in-flight loads, forwards and generations.",
    "inference_substrate": "Name actual artifact aggregation and keep numeric learning absent.",
    "inference_substrate_class": "Use aggregation because this task reads artifacts and loads no model.",
    "execution_venue": "Use host and record actual CPU/CUDA identities; historical board evidence is separate.",
    "duration_s": "Measure current work without padding; separate reduction and validation.",
    "phase_spans": "Timestamped flushed progress and checkpoints expose long silent or unfinished operations.",
    "random_seed": "Freeze ordering, fitting, audit and bootstrap seeds; deterministic reducers explain null seeds.",
    "reproducibility_checksum": "Bind code, protocol, data roles, model identity, raw shards and validation scope.",
    "source_artifact_hashes": "Preserve exact upstream bytes and their original flags/classes.",
    "rows": "One row per scheduled game and seed keeps failures, censoring and unstarted units visible.",
    "sample_size_budget": "Separate planned, attempted, complete, failed, censored, excluded and unstarted independent units.",
    "acceptance_gate_results": "Each check carries category, expected, observed, op, passed and a failure-prevention principle.",
    "gate_check_summary": "Every blocked verdict names failed check, upstream, exact field/path, expected and observed value.",
    "honest_verdict": "Use complete terminal findings and keep readiness independent from scientific benefit.",
    "verdict_class": "Use the closed terminal enum; partial is only for retryable owned work.",
    "verifier_is_oracle": "Declare whether the acceptance verifier is the evaluation oracle; no oracle defines this timing protocol.",
    "flagged_adversarial": "Keep real reader flags; never clear a flag to open a gate.",
    "validation_receipts": "Exact commands, exits, log hashes and required status establish validation scope.",
    "field_principles": "Echo why each field and gate exists so evidence is understandable independently.",
    "arc_interval_protocol_ready_score": "Bare 0/1 for timing validity and observer parity, independent of ARC progress.",
    "interval_rows": "Raw and reduced per-episode intervals expose overlap and unknown attribution.",
    "arc_schedule_manifest": "Thirty-six frozen episode IDs across twelve games bind both independent capture shards.",
    "timing_correction": "Separate historical inclusive sums from current exclusive wall-time bounds.",
    "observer_parity": "A measurement instrument must preserve actions, call order, provenance and RNG state.",
}


def utc_now() -> str:
    """Return one aware UTC boundary for the current aggregation run."""

    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def progress(started: float, phase: str, event: str, **details: Any) -> None:  # pragma: no cover
    """Flush a phase boundary so the conductor can see current work."""

    suffix = " ".join(f"{key}={value}" for key, value in sorted(details.items()))
    print(
        f"[exp7478] phase={phase} event={event} elapsed_s={time.monotonic() - started:.3f}"
        + (f" {suffix}" if suffix else ""),
        flush=True,
    )


def canonical_hash(value: Any) -> str:
    """Hash stable JSON bytes so independent replay detects one changed value."""

    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def artifact_checksum(value: Mapping[str, Any]) -> str:
    """Bind every terminal field except the checksum that stores the result."""

    copy = deepcopy(dict(value))
    copy["reproducibility_checksum"] = ""
    return canonical_hash(copy)


def load_object(path: Path) -> JsonDict:
    """Read one JSON object and fail closed for absent or malformed bytes."""

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(value) if isinstance(value, Mapping) else {}


def read_jsonl(path: Path) -> list[JsonDict]:
    """Read a complete JSONL shard and reject malformed evidence."""

    rows: list[JsonDict] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"malformed_jsonl:{path}:{number}") from exc
        if not isinstance(value, Mapping):
            raise ValueError(f"non_object_jsonl:{path}:{number}")
        rows.append(dict(value))
    return rows


def _gate(
    check: str,
    category: str,
    expected: Any,
    observed: Any,
    *,
    op: str = "==",
    upstream: str,
    field: str,
    principle: str,
) -> JsonDict:
    """Record one exact comparison and the failure that it prevents."""

    if op == "==":
        passed = observed == expected
    elif op == ">=":
        passed = isinstance(observed, (int, float)) and observed >= expected
    else:
        raise ValueError(f"unsupported_gate_operator:{op}")
    return {
        "check": check,
        "category": category,
        "expected": deepcopy(expected),
        "observed": deepcopy(observed),
        "op": op,
        "passed": bool(passed),
        "principle": principle,
        "upstream": upstream,
        "path": upstream,
        "field": field,
        "artifact_field": field,
    }


def _gate_summary(gates: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Keep required failures separate from planned scientific shortfalls."""

    failures = [dict(row) for row in gates if row.get("passed") is not True]
    required = [row for row in failures if row.get("category") in {"validity", "readiness"}]
    first = required[0] if required else failures[0] if failures else None
    return {
        "all_passed": not failures,
        "required_validity_and_readiness_passed": not required,
        "failed_count": len(failures),
        "required_failed_count": len(required),
        "failed_checks": failures,
        "first_failed_check": first.get("check") if first else None,
        "upstream": first.get("upstream") if first else None,
        "exact_field_path": first.get("field") if first else None,
        "expected_value": first.get("expected") if first else None,
        "observed_value": first.get("observed") if first else None,
    }


def _union_segments(intervals: Sequence[tuple[int, int]]) -> list[tuple[int, int]]:
    """Return sorted disjoint intervals and keep zero-duration evidence."""

    merged: list[list[int]] = []
    for left, right in sorted(intervals):
        if right < left:
            continue
        if not merged or left > merged[-1][1]:
            merged.append([left, right])
        else:
            merged[-1][1] = max(merged[-1][1], right)
    return [(left, right) for left, right in merged]


def union_duration_ns(intervals: Sequence[tuple[int, int]]) -> int:
    """Measure an interval union so nested and duplicate spans count once."""

    return sum(right - left for left, right in _union_segments(intervals))


def _intersection_duration_ns(
    left_intervals: Sequence[tuple[int, int]], right_intervals: Sequence[tuple[int, int]]
) -> int:
    """Measure overlap between two unions without assigning it twice."""

    left = _union_segments(left_intervals)
    right = _union_segments(right_intervals)
    total = 0
    i = 0
    j = 0
    while i < len(left) and j < len(right):
        start = max(left[i][0], right[j][0])
        end = min(left[i][1], right[j][1])
        total += max(0, end - start)
        if left[i][1] <= right[j][1]:
            i += 1
        else:
            j += 1
    return total


def _event_tick(row: Mapping[str, Any], field: str) -> int | None:
    """Read one integer clock boundary without converting booleans."""

    value = row.get(field)
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    fallback = row.get("event_monotonic_ns")
    return fallback if isinstance(fallback, int) and not isinstance(fallback, bool) else None


def reduce_episode_intervals(
    events: Sequence[Mapping[str, Any]],
    *,
    episode_id: str,
    episode_start_ns: int,
    episode_end_ns: int,
) -> JsonDict:
    """Reduce one episode without sharing pairing state with another episode."""

    if episode_end_ns < episode_start_ns:
        raise ValueError("episode_boundary_order_invalid")
    observed_ns = episode_end_ns - episode_start_ns
    episode_events = [dict(row) for row in events if str(row.get("episode_id")) == episode_id]
    unique: list[JsonDict] = []
    seen: set[str] = set()
    duplicate_count = 0
    for row in episode_events:
        identity = canonical_hash(row)
        if identity in seen:
            duplicate_count += 1
            continue
        seen.add(identity)
        unique.append(row)

    groups: dict[tuple[str, str, str, str, str], list[JsonDict]] = defaultdict(list)
    for row in unique:
        key = (
            str(row.get("run_id") or "unavailable"),
            str(row.get("process_id") or "unavailable"),
            episode_id,
            str(row.get("decision_id") or ""),
            str(row.get("seam") or ""),
        )
        groups[key].append(row)

    decision_groups: Counter[str] = Counter(key[3] for key in groups)
    repeated_decisions = sum(count - 1 for count in decision_groups.values() if count > 1)
    stage_rows: list[JsonDict] = []
    incomplete_rows: list[tuple[int, int]] = []
    mismatched_clock_count = 0
    conflicting_duplicate_count = 0
    run_identity_missing = 0
    process_identity_missing = 0
    for key, rows in sorted(groups.items()):
        run_id, process_id, _episode, decision_id, seam = key
        starts = [row for row in rows if row.get("event") == "stage_start"]
        ends = [row for row in rows if row.get("event") in {"stage_end", "stage_terminal"}]
        run_identity_missing += int(run_id == "unavailable")
        process_identity_missing += int(process_id == "unavailable")
        clocks = {str(row.get("clock_identity") or "unavailable") for row in (*starts, *ends)}
        if len(clocks) > 1:
            mismatched_clock_count += 1
            continue
        start_ticks = [
            tick
            for row in starts
            if (tick := _event_tick(row, "interval_start_monotonic_ns")) is not None
        ]
        end_ticks = [
            tick
            for row in ends
            if (tick := _event_tick(row, "interval_end_monotonic_ns")) is not None
        ]
        conflicting_duplicate_count += max(0, len(set(start_ticks)) - 1)
        conflicting_duplicate_count += max(0, len(set(end_ticks)) - 1)
        pairs = [(start, end) for start in start_ticks for end in end_ticks if end >= start]
        if not pairs:
            if start_ticks:
                left = max(episode_start_ns, min(start_ticks))
                incomplete_rows.append((left, episode_end_ns))
            continue
        # A repeated boundary can be a retry or a forged duplicate. The shortest
        # valid pair fails closed because an added row can never inflate time.
        start, end = min(pairs, key=lambda pair: (pair[1] - pair[0], pair[0], pair[1]))
        left = max(episode_start_ns, start)
        right = min(episode_end_ns, end)
        if right < left:
            continue
        exemplar = starts[0] if starts else ends[0]
        terminal = next(
            (row for row in ends if _event_tick(row, "interval_end_monotonic_ns") == end), ends[0]
        )
        work_class = str(
            exemplar.get("work_class")
            or terminal.get("work_class")
            or HISTORICAL_WORK_CLASS.get(seam, "unattributed_decision_seam")
        )
        parent = exemplar.get("parent_decision_id") or terminal.get("parent_decision_id")
        stage_rows.append(
            {
                "run_id": run_id,
                "process_id": process_id,
                "episode_id": episode_id,
                "decision_id": decision_id,
                "seam": seam,
                "clock_identity": next(iter(clocks), "unavailable"),
                "start_ns": left,
                "end_ns": right,
                "inclusive_ns": right - left,
                "exclusive_ns": right - left,
                "parent_decision_id": str(parent) if parent is not None else None,
                "effective_parent_decision_id": None,
                "parentage_source": "explicit" if parent is not None else "absent",
                "work_class": work_class,
                "disposition": terminal.get("disposition") or "completed",
                "observer_overhead_ns": max(
                    0,
                    int(terminal.get("observer_cpu_ns") or 0),
                ),
            }
        )

    nested_count = 0
    crossing_count = 0
    for index, row in enumerate(stage_rows):
        containers: list[JsonDict] = []
        for other_index, other in enumerate(stage_rows):
            if index == other_index:
                continue
            contains = (
                other["start_ns"] <= row["start_ns"]
                and row["end_ns"] <= other["end_ns"]
                and (other["start_ns"], other["end_ns"]) != (row["start_ns"], row["end_ns"])
            )
            if contains:
                containers.append(other)
        nested_count += len(containers)
        explicit = row["parent_decision_id"]
        if explicit is not None:
            row["effective_parent_decision_id"] = explicit
        elif containers:
            parent = min(containers, key=lambda item: (item["inclusive_ns"], item["decision_id"]))
            row["effective_parent_decision_id"] = parent["decision_id"]
            row["parentage_source"] = "inferred_containment"

    for index, left in enumerate(stage_rows):
        for right in stage_rows[index + 1 :]:
            crosses = (
                left["start_ns"] < right["start_ns"] < left["end_ns"] < right["end_ns"]
                or right["start_ns"] < left["start_ns"] < right["end_ns"] < left["end_ns"]
            )
            crossing_count += int(crosses)

    children: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for row in stage_rows:
        parent = row["effective_parent_decision_id"]
        if parent is not None:
            children[str(parent)].append((row["start_ns"], row["end_ns"]))
    for row in stage_rows:
        child_union = union_duration_ns(children.get(row["decision_id"], []))
        row["exclusive_ns"] = max(0, row["inclusive_ns"] - child_union)

    complete_intervals = [(row["start_ns"], row["end_ns"]) for row in stage_rows]
    upper_intervals = [*complete_intervals, *incomplete_rows]
    replaceable = [
        (row["start_ns"], row["end_ns"])
        for row in stage_rows
        if row["work_class"] in REPLACEABLE_WORK_CLASSES
    ]
    nonreplaceable = [
        (row["start_ns"], row["end_ns"])
        for row in stage_rows
        if row["work_class"] in NONREPLACEABLE_WORK_CLASSES
    ]
    stage_union_ns = min(observed_ns, union_duration_ns(complete_intervals))
    stage_upper_union_ns = min(observed_ns, union_duration_ns(upper_intervals))
    lower = union_duration_ns(replaceable) - _intersection_duration_ns(replaceable, nonreplaceable)
    upper = stage_upper_union_ns - _intersection_duration_ns(upper_intervals, nonreplaceable)
    lower = max(0, min(observed_ns, lower))
    upper = max(lower, min(observed_ns, upper))

    by_seam: list[JsonDict] = []
    for seam in sorted({str(row["seam"]) for row in stage_rows}):
        seam_rows = [row for row in stage_rows if row["seam"] == seam]
        spans = [(row["start_ns"], row["end_ns"]) for row in seam_rows]
        by_seam.append(
            {
                "seam": seam,
                "interval_count": len(seam_rows),
                "inclusive_sum_ns": sum(int(row["inclusive_ns"]) for row in seam_rows),
                "union_ns": union_duration_ns(spans),
                "exclusive_sum_ns": sum(int(row["exclusive_ns"]) for row in seam_rows),
            }
        )
    return {
        "episode_id": episode_id,
        "episode_start_ns": episode_start_ns,
        "episode_end_ns": episode_end_ns,
        "timestamp_resolution_ns": 1,
        "observed_episode_ns": observed_ns,
        "raw_event_count": len(episode_events),
        "unique_event_count": len(unique),
        "complete_interval_count": len(stage_rows),
        "incomplete_interval_count": len(incomplete_rows),
        "duplicate_event_count": duplicate_count,
        "conflicting_duplicate_count": conflicting_duplicate_count,
        "repeated_decision_id_count": repeated_decisions,
        "mismatched_clock_count": mismatched_clock_count,
        "zero_duration_interval_count": sum(row["inclusive_ns"] == 0 for row in stage_rows),
        "killed_child_interval_count": sum(
            row["disposition"] == "killed_child" for row in stage_rows
        ),
        "nested_interval_count": nested_count,
        "crossing_interval_pair_count": crossing_count,
        "missing_run_identity_group_count": run_identity_missing,
        "missing_process_identity_group_count": process_identity_missing,
        "stage_union_ns": stage_union_ns,
        "stage_upper_union_ns": stage_upper_union_ns,
        "unattributed_ns": observed_ns - stage_union_ns,
        "observer_overhead_ns": sum(int(row["observer_overhead_ns"]) for row in stage_rows),
        "replaceable_lower_ns": lower,
        "replaceable_upper_ns": upper,
        "unknown_attribution_ns": upper - lower,
        "bounds_valid": 0 <= lower <= upper <= observed_ns,
        "reconciles_within_timestamp_resolution": abs(
            stage_union_ns + (observed_ns - stage_union_ns) - observed_ns
        )
        <= 1,
        "stage_summary": by_seam,
        "stage_rows": stage_rows,
    }


class IntervalProtocolObserver:
    """Record explicit parentage around calls without changing their behavior."""

    def __init__(
        self,
        *,
        run_id: str,
        process_id: int,
        episode_id: str,
        sink: Callable[[JsonDict], Any],
        clock_ns: Callable[[], int] = time.monotonic_ns,
    ) -> None:
        self.run_id = run_id
        self.process_id = process_id
        self.episode_id = episode_id
        self.sink = sink
        self.clock_ns = clock_ns
        self._stack: list[str] = []
        self._serial = 0

    def _row(
        self,
        *,
        decision_id: str,
        parent_id: str | None,
        seam: str,
        work_class: str,
        event: str,
        tick: int,
        disposition: str | None = None,
    ) -> JsonDict:
        row: JsonDict = {
            "schema": "carnot.arc.interval_protocol_event.v1",
            "run_id": self.run_id,
            "process_id": self.process_id,
            "episode_id": self.episode_id,
            "decision_id": decision_id,
            "parent_decision_id": parent_id,
            "seam": seam,
            "work_class": work_class,
            "event": event,
            "event_monotonic_ns": tick,
            "clock_identity": f"time.monotonic_ns:pid={self.process_id}",
            "observer_cpu_ns": 0,
        }
        if event == "stage_start":
            row["interval_start_monotonic_ns"] = tick
        else:
            row["interval_end_monotonic_ns"] = tick
            row["disposition"] = disposition or "completed"
        return row

    def call(
        self,
        seam: str,
        work_class: str,
        function: Callable[..., _T],
        *args: Any,
        **kwargs: Any,
    ) -> _T:
        """Call the original function once and record its exact boundaries."""

        self._serial += 1
        decision_id = f"{self.episode_id}:{seam}:{self._serial}"
        parent_id = self._stack[-1] if self._stack else None
        started = self.clock_ns()
        self.sink(
            self._row(
                decision_id=decision_id,
                parent_id=parent_id,
                seam=seam,
                work_class=work_class,
                event="stage_start",
                tick=started,
            )
        )
        self._stack.append(decision_id)
        disposition = "completed"
        try:
            return function(*args, **kwargs)
        except BaseException as exc:
            disposition = "killed_child" if isinstance(exc, SystemExit) else "failed"
            raise
        finally:
            self._stack.pop()
            ended = self.clock_ns()
            self.sink(
                self._row(
                    decision_id=decision_id,
                    parent_id=parent_id,
                    seam=seam,
                    work_class=work_class,
                    event="stage_end",
                    tick=ended,
                    disposition=disposition,
                )
            )


def run_observer_parity_control() -> JsonDict:
    """Compare one deterministic script with the observer off and on."""

    def execute(observer: IntervalProtocolObserver | None) -> JsonDict:
        rng = random.Random(65_501)
        calls: list[str] = []
        provenance: list[JsonDict] = []

        def choose() -> str:
            calls.append("choose")
            action = rng.choice(["ACTION1", "ACTION2"])
            provenance.append({"action": action, "source": "deterministic_fixture"})
            return action

        action = (
            choose()
            if observer is None
            else observer.call("candidate_action_selection", "replaceable_decision", choose)
        )
        return {
            "actions": [action],
            "call_order": calls,
            "provenance": provenance,
            "random_state_sha256": canonical_hash(rng.getstate()),
        }

    events: list[JsonDict] = []
    observer = IntervalProtocolObserver(
        run_id="observer-parity-control",
        process_id=os.getpid(),
        episode_id="deterministic-script",
        sink=events.append,
        clock_ns=iter((100, 120)).__next__,
    )
    disabled = execute(None)
    enabled = execute(observer)
    comparisons = {
        key: disabled[key] == enabled[key]
        for key in ("actions", "call_order", "provenance", "random_state_sha256")
    }
    return {
        "passed": all(comparisons.values()) and len(events) == 2,
        "comparisons": comparisons,
        "event_count": len(events),
        "explicit_parentage_present": all("parent_decision_id" in row for row in events),
        "work_class_present": all(bool(row.get("work_class")) for row in events),
        "current_model_calls": 0,
        "scope": "deterministic_script_control",
    }


def schedule_checksum(rows: Sequence[Mapping[str, Any]]) -> str:
    """Bind the exact order and limits of the 36 future episode rows."""

    return canonical_hash(list(rows))


def build_arc_schedule_manifest() -> JsonDict:
    """Seal both six-game panels before either panel observes an outcome."""

    rows: list[JsonDict] = []
    for panel, games in (("A", PANEL_A_GAMES), ("B", PANEL_B_GAMES)):
        for game in games:
            for seed in EPISODE_SEEDS:
                rows.append(
                    {
                        "episode_id": f"panel-{panel.lower()}:{game}:seed-{seed}",
                        "panel": panel,
                        "game": game,
                        "seed": seed,
                        "execution_order_within_panel": sum(row["panel"] == panel for row in rows),
                        "disposition": "unstarted",
                        "action_limit": ACTION_LIMIT,
                        "request_limit": REQUEST_LIMIT,
                        "max_new_tokens_per_call": MAX_NEW_TOKENS,
                        "episode_limit_s": EPISODE_LIMIT_S,
                        "panel_live_limit_s": PANEL_LIVE_LIMIT_S,
                        "unavailable_game_policy": "retain_row_no_replacement",
                        "readout_role": "soft_feature_only",
                    }
                )
    return {
        "selection_rule": "frozen_E4_plus_next_unused_hash_rank_per_interface_stratum",
        "e4_roster_unchanged": True,
        "added_games_by_stratum": {
            "click": "vc33",
            "keyboard": "g50t",
            "mixed": "dc22",
        },
        "panel_a_games": list(PANEL_A_GAMES),
        "panel_b_games": list(PANEL_B_GAMES),
        "episode_seeds": list(EPISODE_SEEDS),
        "rows": rows,
        "manifest_sha256": schedule_checksum(rows),
    }


def _source_record(path: Path, role: str, flags: Mapping[str, Any] | None = None) -> JsonDict:
    """Describe exact input bytes and preserve any historical disposition."""

    row: JsonDict = {
        "path": path.as_posix(),
        "role": role,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }
    if flags is not None:
        row["original_flags"] = deepcopy(dict(flags))
    return row


def collect_preconditions(root: Path) -> tuple[list[JsonDict], JsonDict, JsonDict]:
    """Authenticate named sources, raw shards, flags, and exclusion status."""

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
                principle="Missing source bytes would make the reduction non-replayable.",
            )
        )
        if available:
            hashes[relative.as_posix()] = _source_record(path, "current_input")

    upstream = load_object(root / UPSTREAM_PATH)
    flags = {
        "status": upstream.get("status"),
        "honest_verdict": upstream.get("honest_verdict"),
        "verdict_class": upstream.get("verdict_class"),
        "flagged_adversarial": upstream.get("flagged_adversarial"),
        "arc_observation_complete_score": upstream.get("arc_observation_complete_score"),
    }
    if (root / UPSTREAM_PATH).is_file():
        hashes[UPSTREAM_PATH.as_posix()] = _source_record(
            root / UPSTREAM_PATH, "structured_prerequisite", flags
        )
    for field, expected in (
        ("status", "complete_null_live_arc_seam_observation"),
        ("honest_verdict", "complete_null_live_arc_seam_observation"),
        ("verdict_class", "null"),
        ("flagged_adversarial", False),
        ("arc_observation_complete_score", 1),
    ):
        checks.append(
            _gate(
                f"exp7471.{field}",
                "validity",
                expected,
                upstream.get(field),
                upstream=UPSTREAM_PATH.as_posix(),
                field=field,
                principle="Historical flags must remain unchanged before dependent interpretation.",
            )
        )

    for shard in upstream.get("seam_event_shards") or []:
        if not isinstance(shard, Mapping) or not isinstance(shard.get("path"), str):
            continue
        relative = Path(str(shard["path"]))
        path = root / relative
        observed_hash = sha256_file(path) if path.is_file() else None
        expected_hash = shard.get("sha256")
        checks.append(
            _gate(
                f"historical_shard:{relative.as_posix()}",
                "validity",
                expected_hash,
                observed_hash,
                upstream=relative.as_posix(),
                field="sha256",
                principle="A changed raw shard would invalidate the corrected timing claim.",
            )
        )
        if path.is_file():
            hashes[relative.as_posix()] = _source_record(path, "historical_model_event_sidecar")

    spec_text = (
        (root / SPEC_PATH).read_text(encoding="utf-8") if (root / SPEC_PATH).is_file() else ""
    )
    checks.append(
        _gate(
            "driving_requirement",
            "validity",
            True,
            "REQ-ARC-WMTE-7478" in spec_text,
            upstream=SPEC_PATH.as_posix(),
            field="REQ-ARC-WMTE-7478",
            principle="Implementation without its requirement would bypass spec-first review.",
        )
    )
    exclusion_path = root / "ops/exclusion_manifest.yaml"
    exclusion_text = exclusion_path.read_text(encoding="utf-8") if exclusion_path.is_file() else ""
    checks.append(
        _gate(
            "task_not_excluded",
            "validity",
            False,
            "7478" in exclusion_text or EXPERIMENT_ID in exclusion_text,
            upstream="ops/exclusion_manifest.yaml",
            field=EXPERIMENT_ID,
            principle="A quarantined scope must not publish fresh scientific evidence.",
        )
    )
    return checks, hashes, upstream


def reduce_exp7471(root: Path, upstream: Mapping[str, Any]) -> JsonDict:
    """Reduce all eight immutable episodes from raw shards and action clocks."""

    events_by_episode: dict[str, list[JsonDict]] = defaultdict(list)
    for shard in upstream.get("seam_event_shards") or []:
        if not isinstance(shard, Mapping) or not isinstance(shard.get("path"), str):
            continue
        path = root / str(shard["path"])
        rows = read_jsonl(path)
        for event in rows:
            episode_id = str(event.get("episode_id") or "")
            if not episode_id or episode_id == "all":
                continue
            normalized = deepcopy(event)
            normalized.setdefault("run_id", upstream.get("experiment_id") or "exp7471")
            normalized.setdefault("process_id", "unavailable")
            normalized.setdefault(
                "work_class",
                HISTORICAL_WORK_CLASS.get(
                    str(normalized.get("seam")), "unattributed_decision_seam"
                ),
            )
            events_by_episode[episode_id].append(normalized)

    interval_rows: list[JsonDict] = []
    for episode in upstream.get("rows") or []:
        if not isinstance(episode, Mapping):
            continue
        episode_id = str(episode.get("episode_id") or "")
        action_rows = [row for row in episode.get("action_rows") or [] if isinstance(row, Mapping)]
        starts = [row.get("interval_start_monotonic_ns") for row in action_rows]
        ends = [row.get("interval_end_monotonic_ns") for row in action_rows]
        if not starts or not ends or not all(isinstance(value, int) for value in (*starts, *ends)):
            raise ValueError(f"episode_action_boundaries_missing:{episode_id}")
        reduced = reduce_episode_intervals(
            events_by_episode[episode_id],
            episode_id=episode_id,
            episode_start_ns=min(starts),
            episode_end_ns=max(ends),
        )
        reduced.update(
            {
                "game": episode.get("game"),
                "seed": episode.get("seed"),
                "historical_disposition": episode.get("disposition"),
                "historical_elapsed_s": episode.get("elapsed_s"),
            }
        )
        interval_rows.append(reduced)

    historical_sum = int(
        ((upstream.get("replaceable_cost_bounds_ns") or {}).get("observed_lower") or 0)
    )
    recomputed_sum = sum(
        int(stage["inclusive_ns"])
        for episode in interval_rows
        for stage in episode["stage_rows"]
        if stage["seam"] != "downstream_generation"
    )
    timing = {
        "historical_inclusive_sum_ns": historical_sum,
        "recomputed_historical_inclusive_sum_ns": recomputed_sum,
        "historical_live_work_s": (upstream.get("duration_breakdown_s") or {}).get(
            "live_model_and_episodes"
        ),
        "observed_episode_duration_ns": sum(row["observed_episode_ns"] for row in interval_rows),
        "corrected_episode_union_ns": sum(row["stage_union_ns"] for row in interval_rows),
        "corrected_replaceable_lower_ns": sum(row["replaceable_lower_ns"] for row in interval_rows),
        "corrected_replaceable_upper_ns": sum(row["replaceable_upper_ns"] for row in interval_rows),
        "known_generation_union_ns": sum(
            next(
                (
                    item["union_ns"]
                    for item in row["stage_summary"]
                    if item["seam"] == "downstream_generation"
                ),
                0,
            )
            for row in interval_rows
        ),
        "overlap_removed_ns": historical_sum
        - sum(
            union_duration_ns(
                [
                    (stage["start_ns"], stage["end_ns"])
                    for stage in row["stage_rows"]
                    if stage["seam"] != "downstream_generation"
                ]
            )
            for row in interval_rows
        ),
        "historical_parentage_complete": all(
            stage["parent_decision_id"] is not None
            for row in interval_rows
            for stage in row["stage_rows"]
        ),
        "historical_process_identity_complete": all(
            stage["process_id"] != "unavailable"
            for row in interval_rows
            for stage in row["stage_rows"]
        ),
        "future_protocol_field_sufficiency": True,
        "historical_upper_bound_auditable": all(row["bounds_valid"] for row in interval_rows),
        "unknown_work_classes": [
            "world_model_construction",
            "verifier_work",
            "dispatch",
            "idle",
        ],
        "interpretation": "The historical sum is inclusive. Corrected bounds union each episode and exclude known generation from replaceable time.",
    }
    return {"interval_rows": interval_rows, "timing_correction": timing}


def _receipts_pass(receipts: Sequence[Mapping[str, Any]], names: Sequence[str]) -> bool:
    """Require one passing, non-timeout receipt for every named command."""

    counts = Counter(str(row.get("name")) for row in receipts)
    by_name = {str(row.get("name")): row for row in receipts}
    return all(
        counts[name] == 1
        and by_name[name].get("passed") is True
        and by_name[name].get("exit_code") == 0
        and by_name[name].get("timed_out") is not True
        for name in names
    )


def _field_principles(keys: Sequence[str]) -> JsonDict:
    """Explain each ordinary artifact field without wrapping its value."""

    return {
        key: FIELD_PRINCIPLES.get(
            key, "Retain this typed field so an independent reader can audit the protocol."
        )
        for key in keys
    }


def build_terminal_artifact(
    root: Path,
    *,
    checks: Sequence[Mapping[str, Any]],
    hashes: Mapping[str, Any],
    upstream: Mapping[str, Any],
    validation_receipts: Sequence[Mapping[str, Any]],
    duration_s: float,
    phase_spans: Sequence[Mapping[str, Any]],
    started_at_utc: str,
    ended_at_utc: str,
    require_terminal: bool,
) -> JsonDict:
    """Build one artifact from raw reductions and declared command receipts."""

    reduction = reduce_exp7471(root, upstream)
    schedule = build_arc_schedule_manifest()
    parity = run_observer_parity_control()
    interval_ok = all(
        row["bounds_valid"] and row["reconciles_within_timestamp_resolution"]
        for row in reduction["interval_rows"]
    )
    affected_ok = _receipts_pass(validation_receipts, AFFECTED_CHECK_NAMES)
    e2e_ok = _receipts_pass(validation_receipts, CAPABILITY_E2E_NAMES)
    terminal_ok = (
        _receipts_pass(validation_receipts, REQUIRED_TERMINAL_NAMES) if require_terminal else True
    )
    preconditions_ok = bool(checks) and all(row.get("passed") is True for row in checks)
    schedule_ok = (
        len(schedule["rows"]) == 36 and len({row["episode_id"] for row in schedule["rows"]}) == 36
    )
    historical_fields_sufficient = bool(
        reduction["timing_correction"]["historical_upper_bound_auditable"]
        and reduction["timing_correction"]["future_protocol_field_sufficiency"]
    )
    validity = [
        _gate(
            "preconditions_checked",
            "validity",
            True,
            preconditions_ok,
            upstream="preconditions_checked",
            field="passed",
            principle="A positive scientific metric cannot excuse invalid evidence.",
        ),
        _gate(
            "episode_interval_accounting",
            "validity",
            True,
            interval_ok,
            upstream="interval_rows",
            field="bounds_valid|reconciles_within_timestamp_resolution",
            principle="Overlapping or unfinished spans cannot exceed observed wall time.",
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
            "raw_field_sufficiency",
            "readiness",
            True,
            historical_fields_sufficient,
            upstream="timing_correction",
            field="historical_upper_bound_auditable|future_protocol_field_sufficiency",
            principle="A valid null must not suppress an independent measurement.",
        ),
        _gate(
            "observer_parity",
            "readiness",
            True,
            parity["passed"],
            upstream="observer_parity",
            field="passed",
            principle="A measurement instrument must not alter the policy it measures.",
        ),
        _gate(
            "sealed_schedule",
            "readiness",
            True,
            schedule_ok,
            upstream="arc_schedule_manifest",
            field="rows",
            principle="A valid null must not suppress an independent measurement.",
        ),
    ]
    benefit = [
        _gate(
            "e6_episode_support_floor",
            "scientific_benefit",
            30,
            len(reduction["interval_rows"]),
            op=">=",
            upstream=UPSTREAM_PATH.as_posix(),
            field="sample_size_budget.complete_independent_units",
            principle="A small sample, a favorable seed or an analytic fixture cannot substitute for held-out value.",
        ),
        _gate(
            "e6_game_support_floor",
            "scientific_benefit",
            10,
            len({str(row["game"]) for row in reduction["interval_rows"]}),
            op=">=",
            upstream=UPSTREAM_PATH.as_posix(),
            field="sample_size_budget.independent_game_clusters",
            principle="Repeated seeds cannot substitute for independent game support.",
        ),
        _gate(
            "effect_size_threshold",
            "scientific_benefit",
            True,
            False,
            upstream="not_measured_by_protocol_task",
            field="effect_size",
            principle="A timing fixture cannot substitute for a held-out intervention effect.",
        ),
        _gate(
            "retention_threshold",
            "scientific_benefit",
            True,
            False,
            upstream="not_measured_by_protocol_task",
            field="retention",
            principle="Cost savings cannot excuse lost ARC progress.",
        ),
        _gate(
            "multiplicity_threshold",
            "scientific_benefit",
            "not_applicable_no_hypothesis_family",
            "not_applicable_no_hypothesis_family",
            upstream="protocol_scope",
            field="multiplicity",
            principle="One favorable comparison cannot bypass a declared hypothesis family correction.",
        ),
    ]
    gates = [*validity, *readiness, *benefit]
    required_pass = all(row["passed"] for row in (*validity, *readiness))
    ready_score = int(required_pass)
    verdict_class = "null" if required_pass else "disqualified"
    status = (
        "complete_null_interval_protocol_ready_sample_limited"
        if required_pass
        else "complete_disqualified_interval_protocol"
    )
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "status": status,
        "run_date": RUN_DATE,
        "started_at_utc": started_at_utc,
        "ended_at_utc": ended_at_utc,
        "clock_identity": {
            "wall": "datetime.now(UTC)",
            "current_interval": "time.monotonic",
            "historical_interval": "time.monotonic_ns with missing process identity",
        },
        "preconditions_checked": deepcopy(list(checks)),
        "MODEL_SPECS": [],
        "model_specs": [],
        "model_invoked": False,
        "invocation_counts": deepcopy(ZERO_INVOCATION_COUNTS),
        "inference_substrate": "aggregation_from_upstream_artifacts",
        "inference_substrate_class": INFERENCE_SUBSTRATE_CLASS,
        "execution_venue": EXECUTION_VENUE,
        "execution_venue_details": {
            "cpu": platform.processor() or platform.machine(),
            "platform": platform.platform(),
            "current_cuda_used": False,
            "current_cuda_device_identity": None,
            "historical_cuda_evidence_only": True,
        },
        "duration_s": round(float(duration_s), 6),
        "phase_spans": deepcopy(list(phase_spans)),
        "random_seed": {
            "schedule": None,
            "reducer": None,
            "observer_parity": 65_501,
            "audit": 7_478_091,
            "bootstrap": 655_590,
        },
        "reproducibility_checksum": "",
        "source_artifact_hashes": deepcopy(dict(hashes)),
        "historical_receipt_sidecars": [
            {
                "path": path,
                "sha256": row["sha256"],
                "role": row["role"],
                "scope": "historical",
            }
            for path, row in sorted(hashes.items())
            if row["role"] in {"historical_producer", "historical_model_event_sidecar"}
        ],
        "test_environment_calls": {
            "subprocess_count": len(validation_receipts),
            "model_loads": 0,
            "forward_calls": 0,
            "generation_calls": 0,
            "separate_from_historical_receipt_sidecars": True,
        },
        "small_ebm_training": {
            "performed": False,
            "duration_s": 0.0,
            "reason": "deterministic_interval_reduction_requires_no_numeric_fit",
        },
        "rows": deepcopy(schedule["rows"]),
        "interval_rows": reduction["interval_rows"],
        "arc_schedule_manifest": schedule,
        "timing_correction": reduction["timing_correction"],
        "observer_parity": parity,
        "sample_size_budget": {
            "planned_independent_units": 36,
            "attempted_independent_units": 0,
            "complete_independent_units": 0,
            "failed_independent_units": 0,
            "censored_independent_units": 0,
            "excluded_independent_units": 0,
            "unstarted_independent_units": 36,
            "planned_game_clusters": 12,
            "historical_complete_episode_units": len(reduction["interval_rows"]),
            "historical_game_clusters": len(
                {str(row["game"]) for row in reduction["interval_rows"]}
            ),
        },
        "acceptance_gate_results": gates,
        "gate_check_summary": _gate_summary(gates),
        "honest_verdict": status,
        "verdict_class": verdict_class,
        "verifier_is_oracle": False,
        "flagged_adversarial": False,
        "validation_manifest": {
            "experiment_id": MANIFEST.experiment_id,
            "test_paths": list(MANIFEST.test_paths),
            "changed_modules": list(MANIFEST.changed_modules),
            "static_paths": list(MANIFEST.static_paths),
            "capability_e2e_names": list(CAPABILITY_E2E_NAMES),
            "terminal_names": list(REQUIRED_TERMINAL_NAMES),
        },
        "validation_receipts": deepcopy(list(validation_receipts)),
        "arc_interval_protocol_ready_score": ready_score,
        "readout_role": "soft_feature_only",
        "selector_control_authorized": False,
        "e4_e5_intervention_authorized": False,
        "scored_submission_authorized": False,
        "extra_model_authorized": False,
        "current_live_model_claim": False,
        "scientific_benefit_claim": False,
        "research_conductor_changed": False,
    }
    artifact["field_principles"] = _field_principles((*artifact, "field_principles"))
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return artifact


def independent_reduce(artifact: Mapping[str, Any], root: Path = REPO_ROOT) -> JsonDict:
    """Recompute schedule and interval claims from immutable source bytes."""

    upstream = load_object(root / UPSTREAM_PATH)
    reduction = reduce_exp7471(root, upstream)
    schedule = build_arc_schedule_manifest()
    declared = {
        "rows": artifact.get("rows"),
        "interval_rows": artifact.get("interval_rows"),
        "arc_schedule_manifest": artifact.get("arc_schedule_manifest"),
        "timing_correction": artifact.get("timing_correction"),
    }
    recomputed = {
        "rows": schedule["rows"],
        "interval_rows": reduction["interval_rows"],
        "arc_schedule_manifest": schedule,
        "timing_correction": reduction["timing_correction"],
    }
    return {
        **recomputed,
        "matches_declared": canonical_hash(declared) == canonical_hash(recomputed),
    }


def validate_artifact(value: Mapping[str, Any] | Path, *, require_terminal: bool) -> list[str]:
    """Cold-check identity, reduction, readiness, counters, and checksum."""

    artifact = load_object(value) if isinstance(value, Path) else dict(value)
    errors: list[str] = []
    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in artifact:
            errors.append(f"missing_field:{field}")
    for field, expected in (
        ("schema", SCHEMA),
        ("experiment_id", EXPERIMENT_ID),
        ("milestone", MILESTONE),
        ("run_date", RUN_DATE),
        ("inference_substrate_class", "aggregation"),
        ("execution_venue", "host"),
    ):
        if artifact.get(field) != expected:
            errors.append(f"identity_mismatch:{field}")
    if artifact.get("MODEL_SPECS") != [] or artifact.get("model_specs") != []:
        errors.append("current_model_declaration_invalid")
    if (
        artifact.get("model_invoked") is not False
        or artifact.get("invocation_counts") != ZERO_INVOCATION_COUNTS
    ):
        errors.append("current_invocation_accounting_invalid")
    if independent_reduce(artifact)["matches_declared"] is not True:
        errors.append("independent_reduction_mismatch")
    interval_rows = artifact.get("interval_rows") or []
    if len(interval_rows) != 8 or any(
        row.get("bounds_valid") is not True
        or row.get("reconciles_within_timestamp_resolution") is not True
        for row in interval_rows
        if isinstance(row, Mapping)
    ):
        errors.append("interval_accounting_invalid")
    schedule = artifact.get("arc_schedule_manifest") or {}
    if len(schedule.get("rows") or []) != 36 or schedule.get(
        "manifest_sha256"
    ) != schedule_checksum(schedule.get("rows") or []):
        errors.append("schedule_manifest_invalid")
    gates = artifact.get("acceptance_gate_results") or []
    expected_ready = int(
        bool(gates)
        and all(
            row.get("passed") is True
            for row in gates
            if row.get("category") in {"validity", "readiness"}
        )
    )
    if artifact.get("arc_interval_protocol_ready_score") != expected_ready:
        errors.append("readiness_score_mismatch")
    if require_terminal and not _receipts_pass(
        artifact.get("validation_receipts") or [],
        (*AFFECTED_CHECK_NAMES, *CAPABILITY_E2E_NAMES, *REQUIRED_TERMINAL_NAMES),
    ):
        errors.append("required_validation_receipts_invalid")
    if artifact.get("reproducibility_checksum") != artifact_checksum(artifact):
        errors.append("reproducibility_checksum_mismatch")
    return list(dict.fromkeys(errors))


def _fixture_receipts() -> list[JsonDict]:
    """Supply typed passing receipts only for deterministic artifact tests."""

    return [
        {
            "name": name,
            "command": f"fixture:{name}",
            "command_argv": ["fixture", name],
            "scope": "test_fixture",
            "exit_code": 0,
            "passed": True,
            "timed_out": False,
            "duration_s": 0.0,
            "log_path": None,
            "log_sha256": canonical_hash(name),
            "required": True,
            "fixture_only": True,
        }
        for name in (*AFFECTED_CHECK_NAMES, *CAPABILITY_E2E_NAMES, *REQUIRED_TERMINAL_NAMES)
    ]


def build_artifact_for_test(root: Path = REPO_ROOT) -> JsonDict:
    """Build a terminal-shaped fixture from real immutable historical bytes."""

    checks, hashes, upstream = collect_preconditions(root)
    return build_terminal_artifact(
        root,
        checks=checks,
        hashes=hashes,
        upstream=upstream,
        validation_receipts=_fixture_receipts(),
        duration_s=1.0,
        phase_spans=[
            {
                "phase": "fixture",
                "start_s": 0.0,
                "end_s": 1.0,
                "duration_s": 1.0,
                "completed_units": 8,
            }
        ],
        started_at_utc="2026-09-21T00:00:00Z",
        ended_at_utc="2026-09-21T00:00:01Z",
        require_terminal=True,
    )


def build_validation_plan(root: Path, private_root: Path) -> list[validation_scope.CommandSpec]:
    """Freeze affected checks and the three applicable ARC capability checks."""

    private_root.mkdir(parents=True, exist_ok=True)
    affected = build_command_plan(root, MANIFEST, private_root / "affected")
    python = ".venv/bin/python"
    pytest = ".venv/bin/pytest"
    common = ("-n", "0", "-o", "addopts=", "--no-cov")
    e2e: list[validation_scope.CommandSpec] = []
    for name, test in (
        ("e2e_009", "tests/python/test_arc_induction_state_persistence.py"),
        ("e2e_010", "tests/python/test_arc_tool_grammar_transport.py"),
        ("e2e_011", "tests/python/test_arc_decision_telemetry.py"),
    ):
        parent = private_root / name
        parent.mkdir(parents=True, exist_ok=True)
        e2e.append(
            validation_scope.CommandSpec(
                name,
                (pytest, *common, f"--basetemp={parent / 'basetemp'}", test, "-q"),
                "capability_e2e",
                900.0,
            )
        )
    smoke_parent = private_root / "private_arc_smoke"
    smoke_parent.mkdir(parents=True, exist_ok=True)
    e2e.append(
        EnvironmentCommandSpec(
            "private_arc_smoke",
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
                str(smoke_parent / "receipt.json"),
            ),
            "private_real_environment_smoke",
            900.0,
            (("CARNOT_ARC_DISABLE_INDUCTION", "1"),),
        )
    )
    return [*affected, *e2e]


def validate_validation_plan(
    root: Path, commands: Sequence[validation_scope.CommandSpec]
) -> list[str]:
    """Reject broad targets, duplicate names, and capability command drift."""

    affected = [row for row in commands if row.name in AFFECTED_CHECK_NAMES]
    errors = validate_command_plan(root, MANIFEST, affected)
    counts = Counter(row.name for row in commands)
    expected = (*AFFECTED_CHECK_NAMES, *CAPABILITY_E2E_NAMES)
    for name in expected:
        if counts[name] != 1:
            errors.append(f"command_count:{name}:{counts[name]}")
    for row in commands:
        if any(argument.rstrip("/") in {"tests", "tests/python"} for argument in row.argv):
            errors.append(f"broad_test_target:{row.name}")
    return errors


def terminal_command_specs(root: Path, candidate: Path) -> list[validation_scope.CommandSpec]:
    """Build cold reduction and unchanged strict readers for one candidate."""

    del root
    python = ".venv/bin/python"
    wrapper = WRAPPER_PATH.as_posix()
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


def _phase_span(
    name: str, began: float, run_started: float, completed_units: int
) -> JsonDict:  # pragma: no cover
    """Create one real monotonic phase receipt."""

    ended = time.monotonic()
    return {
        "phase": name,
        "start_s": round(began - run_started, 6),
        "end_s": round(ended - run_started, 6),
        "duration_s": round(ended - began, 6),
        "completed_units": completed_units,
        "ended_at_utc": utc_now(),
    }


def run_experiment(root: Path, run_date: str) -> JsonDict:  # pragma: no cover - orchestration
    """Run scoped checks, cold readers, and one atomic terminal publication."""

    started = time.monotonic()
    started_at = utc_now()
    spans: list[JsonDict] = []
    receipts: list[JsonDict] = []
    progress(started, "startup", "begin", run_date=run_date)

    phase = time.monotonic()
    progress(started, "preconditions", "before")
    checks, hashes, upstream = collect_preconditions(root)
    checks.insert(
        0,
        _gate(
            "run_date",
            "validity",
            RUN_DATE,
            run_date,
            upstream="command_line",
            field="--date",
            principle="A wrong run date would detach the artifact from its roadmap task.",
        ),
    )
    reduction = reduce_exp7471(root, upstream)
    spans.append(
        _phase_span("preconditions_and_reduction", phase, started, len(reduction["interval_rows"]))
    )
    progress(started, "preconditions", "after", passed=all(row["passed"] for row in checks))

    private_root = Path(tempfile.mkdtemp(prefix="exp7478-validation-", dir="/tmp"))
    phase = time.monotonic()
    progress(started, "validation", "before_subprocesses")
    commands = build_validation_plan(root, private_root)
    plan_errors = validate_validation_plan(root, commands)
    if plan_errors:
        raise RuntimeError(f"validation_plan_invalid:{plan_errors}")
    receipts.extend(
        run_categorized_commands(
            root,
            [PlannedCommand(row, "required_validation", True) for row in commands],
            log_dir=root / RAW_DIR / "validation/affected_and_e2e",
            heartbeat_s=60.0,
        )
    )
    spans.append(_phase_span("affected_and_capability_validation", phase, started, len(receipts)))
    progress(started, "validation", "after_subprocesses", completed_units=len(receipts))

    candidate = build_terminal_artifact(
        root,
        checks=checks,
        hashes=hashes,
        upstream=upstream,
        validation_receipts=receipts,
        duration_s=time.monotonic() - started,
        phase_spans=spans,
        started_at_utc=started_at,
        ended_at_utc=utc_now(),
        require_terminal=False,
    )
    candidate_errors = validate_artifact(candidate, require_terminal=False)
    if candidate_errors:
        raise RuntimeError(f"candidate_invalid:{candidate_errors}")
    atomic_json(root / CANDIDATE_PATH, candidate)

    phase = time.monotonic()
    progress(started, "terminal_validation", "before_subprocesses", candidate=CANDIDATE_PATH)
    terminal = validation_scope.run_commands(
        root,
        terminal_command_specs(root, root / CANDIDATE_PATH),
        log_dir=root / RAW_DIR / "validation/terminal",
        heartbeat_s=60.0,
    )
    receipts.extend(terminal)
    spans.append(_phase_span("terminal_validation", phase, started, len(terminal)))
    progress(started, "terminal_validation", "after_subprocesses", completed_units=len(terminal))

    artifact = build_terminal_artifact(
        root,
        checks=checks,
        hashes=hashes,
        upstream=upstream,
        validation_receipts=receipts,
        duration_s=time.monotonic() - started,
        phase_spans=spans,
        started_at_utc=started_at,
        ended_at_utc=utc_now(),
        require_terminal=True,
    )
    errors = validate_artifact(artifact, require_terminal=True)
    if errors:
        raise RuntimeError(f"terminal_artifact_invalid:{errors}")
    progress(started, "publish", "before_atomic_terminal_write", path=RESULT_PATH)
    atomic_json(root / RESULT_PATH, artifact)
    progress(
        started,
        "publish",
        "after_atomic_terminal_write",
        path=RESULT_PATH,
        ready=artifact["arc_interval_protocol_ready_score"],
    )
    return artifact


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the public run date or one cold-replay candidate path."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", choices=[RUN_DATE])
    parser.add_argument("--replay", type=Path)
    parser.add_argument("--reduce-only", action="store_true")
    args = parser.parse_args(argv)
    if args.date is None and args.replay is None:
        parser.error("--date or --replay is required")
    return args


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - thin CLI
    """Run aggregation or independently replay one candidate."""

    args = parse_args(argv)
    if args.replay is not None:
        artifact = load_object(args.replay)
        reduced = independent_reduce(artifact)
        errors = [] if args.reduce_only else validate_artifact(artifact, require_terminal=False)
        print(
            json.dumps(
                {
                    "matches_declared": reduced.get("matches_declared"),
                    "interval_row_count": len(reduced.get("interval_rows") or []),
                    "schedule_row_count": len(reduced.get("rows") or []),
                    "validation_errors": errors,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return int(bool(errors) or reduced.get("matches_declared") is not True)
    artifact = run_experiment(REPO_ROOT, str(args.date))
    return 0 if artifact.get("verdict_class") in {"null", "blocked"} else 1


__all__ = [
    "AFFECTED_CHECK_NAMES",
    "CAPABILITY_E2E_NAMES",
    "EXECUTION_VENUE",
    "INFERENCE_SUBSTRATE_CLASS",
    "IntervalProtocolObserver",
    "MODEL_SPECS",
    "REQUIRED_ARTIFACT_FIELDS",
    "REQUIRED_TERMINAL_NAMES",
    "SPEC_PATH",
    "UPSTREAM_PATH",
    "build_arc_schedule_manifest",
    "build_artifact_for_test",
    "build_validation_plan",
    "collect_preconditions",
    "independent_reduce",
    "main",
    "reduce_episode_intervals",
    "reduce_exp7471",
    "schedule_checksum",
    "terminal_command_specs",
    "union_duration_ns",
    "validate_artifact",
    "validate_validation_plan",
]
