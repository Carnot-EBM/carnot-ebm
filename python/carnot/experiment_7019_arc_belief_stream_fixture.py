"""Freeze prior live ARC observations into a game-blind belief stream.

REQ-ARC-WMTE-7019 separates what an updater may know at each action from
facts learned later. The builder reads only provenance envelopes and recorded
frames. It does not import an engine, adapter, or game implementation.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict, deque
from collections.abc import Mapping, Sequence
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
import tempfile
import time
from typing import Any

from carnot import experiment_7005_arc_live_envelope_audit as exp7005


JsonDict = dict[str, Any]

EXPERIMENT_ID = 7019
SCHEMA = "carnot.exp7019.arc_belief_stream_fixture.v1"
EVENT_SCHEMA = "carnot.arc.belief_transition_event.v1"
SEALED_FUTURE_SCHEMA = "carnot.arc.sealed_held_future.v1"
RUN_DATE = "20260905"
RANDOM_SEED = 70_192_026_090_5
INFERENCE_SUBSTRATE = "deterministic_arc_live_attempt_fixture_no_llm"

REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = Path("openspec/capabilities/arc-world-model-trust-energy/spec.md")
MODULE_PATH = Path("python/carnot/experiment_7019_arc_belief_stream_fixture.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_7019_arc_belief_stream_fixture.py")
OUTPUT_PATH = Path("results/experiment_7019_arc_belief_stream_fixture.json")
RAW_DIR = Path("results/raw/experiment_7019_arc_belief_stream_fixture")
FIXTURE_NAME = "transition_events.jsonl"
SIDECAR_NAME = "sealed_held_future.jsonl"
EXP7010_PATH = Path("results/experiment_7010_arc_eval_provenance_contract.json")
EXP7005_PATH = Path("results/experiment_7005_arc_live_envelope_audit.json")
EXP5155_PATH = Path("results/experiment_5155_multilevel_belief_state_scoping_v472.json")
STORE_PATH = Path("results/arc_e3")
REGISTRY_PATH = Path("ops/arc_solve_registry.yaml")
PROVENANCE_HELPER_PATH = Path("python/carnot/agentic/arc_eval_provenance.py")
POLICY_PATH = Path("python/carnot/agentic/arc_competition_agent.py")
ACTION_PROVENANCE_PATH = Path("python/carnot/agentic/arc_action_provenance.py")

HASH_PATTERN = re.compile(r"sha256:[0-9a-f]{64}\Z")
TIMESTAMP_PATTERN = re.compile(r"\d{8}T\d{6}_\d{6}\Z")
TRANSITION_ID_PATTERN = re.compile(r"[0-9a-f]{16}\Z")
VERDICT_CLASSES = frozenset(
    {"positive", "circular_positive", "null", "blocked", "disqualified", "partial"}
)

EVENT_FIELDS = (
    "schema",
    "event_id",
    "stream_index",
    "source_attempt_time",
    "source_transition_index",
    "pre_action",
    "action",
    "next_observation",
    "contradiction",
    "mechanic_signature",
    "row_hash",
)
PRE_ACTION_FIELDS = (
    "grid",
    "grid_hash",
    "observed_level",
    "object_summary",
    "spatial_summary",
    "prior_support",
)
NEXT_OBSERVATION_FIELDS = (
    "grid",
    "grid_hash",
    "observed_level",
    "object_summary",
    "spatial_summary",
    "state_delta",
    "level_boundary",
)

REQUIRED_ARTIFACT_FIELDS = (
    "field_principles",
    "preconditions_checked",
    "inference_substrate",
    "duration_s",
    "source_artifact_hashes",
    "attempt_manifest_rows",
    "provenance_acceptance_rows",
    "provenance_rejection_rows",
    "chronology_rows",
    "level_boundary_rows",
    "pre_action_rows",
    "next_observation_rows",
    "mechanic_signature_rows",
    "contradiction_pair_rows",
    "counterexample_cluster_rows",
    "sealed_future_rows",
    "leakage_check_rows",
    "fixture_path",
    "held_future_sidecar_path",
    "fixture_hash",
    "held_future_sidecar_hash",
    "fresh_process_replay_rows",
    "rows",
    "solve_provenance",
    "arc_belief_stream_ready_score",
    "random_seed",
    "reproducibility_checksum",
    "gate_check_summary",
    "verifier_is_oracle",
    "verdict_class",
    "honest_verdict",
)

FIELD_PRINCIPLES = {
    "field_principles": "A reason for each field makes the evidence contract reviewable.",
    "preconditions_checked": "Explicit preflight results stop absent evidence from becoming data.",
    "inference_substrate": "The substrate distinguishes deterministic replay from model inference.",
    "duration_s": "Measured wall time confirms that the fixture process ran.",
    "source_artifact_hashes": "Source hashes bind the stream to exact prior evidence.",
    "attempt_manifest_rows": "The complete inventory prevents silent selection changes.",
    "provenance_acceptance_rows": "Accepted attempts show why each live source may contribute.",
    "provenance_rejection_rows": "Terminal rejections preserve malformed and duplicate evidence.",
    "chronology_rows": "A total order prevents later observations from moving before actions.",
    "level_boundary_rows": "Explicit boundaries prevent belief state from crossing levels silently.",
    "pre_action_rows": "Pre-action hashes fix the information available before each decision.",
    "next_observation_rows": "Observation hashes bind each action to its immediate visible result.",
    "mechanic_signature_rows": "Game-blind signatures support transfer without source identity.",
    "contradiction_pair_rows": "Pairs expose observations that disprove an earlier mechanic belief.",
    "counterexample_cluster_rows": "Clusters retain repeated conflicts without using later labels.",
    "sealed_future_rows": "Sealed row receipts prove later outcomes exist outside updater input.",
    "leakage_check_rows": "Leakage checks show that identity and future values stayed isolated.",
    "fixture_path": "The fixture path identifies the updater-visible immutable stream.",
    "held_future_sidecar_path": "A separate path keeps evaluation outcomes outside updater access.",
    "fixture_hash": "A content hash detects any updater-visible fixture change.",
    "held_future_sidecar_hash": "A sidecar hash detects any held-out outcome change.",
    "fresh_process_replay_rows": "Independent replay detects process-local ordering or hash state.",
    "rows": "Aggregate rows expose the mechanic-group support behind readiness.",
    "solve_provenance": "Live provenance prevents a fixture from receiving outer-loop solve credit.",
    "arc_belief_stream_ready_score": "One bit gates use on diversity, completeness, isolation, and replay.",
    "random_seed": "A fixed seed makes deterministic configuration explicit.",
    "reproducibility_checksum": "A manifest digest detects drift across later reproductions.",
    "gate_check_summary": "The first exact failure makes a blocked result repairable.",
    "verifier_is_oracle": "False records that no hidden correctness oracle judged the stream.",
    "verdict_class": "A closed class lets consumers interpret the result without prose guessing.",
    "honest_verdict": "A class-consistent prefix prevents a blocked or null result from reading positive.",
}

FORBIDDEN_UPDATER_KEYS = frozenset(
    {
        "adapter",
        "adapter_name",
        "game",
        "game_id",
        "hidden_rule",
        "manifest_path",
        "registry_result",
        "run_id",
        "solve_provenance",
        "source_game",
        "source_path",
        "held_future",
        "later_outcome",
    }
)


def canonical_json_bytes(value: Any) -> bytes:
    """Return one stable byte encoding for hashes and JSONL rows."""

    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    """Label SHA-256 digests so another algorithm cannot be assumed."""

    return "sha256:" + hashlib.sha256(value).hexdigest()


def sha256_path(path: Path) -> str | None:
    """Hash a readable file while preserving absence as an explicit value."""

    try:
        return sha256_bytes(Path(path).read_bytes())
    except OSError:
        return None


def transition_id_for(row: Mapping[str, Any]) -> str:
    """Replay the producer transition identifier from observable row fields."""

    projected = dict(row)
    projected.pop("transition_id", None)
    return hashlib.sha256(canonical_json_bytes(projected)).hexdigest()[:16]


def event_row_hash(row: Mapping[str, Any]) -> str:
    """Hash an event without trusting an existing self-hash field."""

    projected = dict(row)
    projected.pop("row_hash", None)
    return sha256_bytes(canonical_json_bytes(projected))


def _sealed_row_hash(row: Mapping[str, Any]) -> str:
    projected = dict(row)
    projected.pop("row_hash", None)
    return sha256_bytes(canonical_json_bytes(projected))


def _is_hash(value: Any) -> bool:
    return isinstance(value, str) and HASH_PATTERN.fullmatch(value) is not None


def _is_integer(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _valid_grid(value: Any) -> bool:
    if not isinstance(value, list) or not value or not all(isinstance(row, list) for row in value):
        return False
    width = len(value[0])
    return bool(
        width
        and all(len(row) == width for row in value)
        and all(_is_integer(cell) for row in value for cell in row)
    )


def _transition_error(row: Any) -> str | None:
    """Return the first transition error so an entire attempt can fail closed."""

    if not isinstance(row, Mapping):
        return "transition_object_required"
    required = {
        "grid",
        "action",
        "data",
        "next_grid",
        "index",
        "level_before",
        "level_after",
        "transition_id",
    }
    if required - set(row):
        return "transition_required_fields_missing"
    if not _valid_grid(row.get("grid")) or not _valid_grid(row.get("next_grid")):
        return "transition_grid_invalid"
    grid = row["grid"]
    next_grid = row["next_grid"]
    if len(grid) != len(next_grid) or len(grid[0]) != len(next_grid[0]):
        return "transition_grid_shape_mismatch"
    if not _is_integer(row.get("action")):
        return "transition_action_invalid"
    data = row.get("data")
    if data is not None and not isinstance(data, Mapping):
        return "transition_action_data_invalid"
    for field in ("index", "level_before", "level_after"):
        if not _is_integer(row.get(field)) or int(row[field]) < 0:
            return f"transition_{field}_invalid"
    transition_id = row.get("transition_id")
    if not isinstance(transition_id, str) or TRANSITION_ID_PATTERN.fullmatch(transition_id) is None:
        return "transition_id_invalid"
    if transition_id_for(row) != transition_id:
        return "transition_id_hash_mismatch"
    return None


def _attempt_error(attempt: Any) -> tuple[str | None, str | None]:
    """Validate all attempt provenance before admitting any transition."""

    if not isinstance(attempt, Mapping):
        return "incomplete_attempt_provenance", "attempt_object_required"
    required = (
        "source_run_id",
        "created_at",
        "published_at",
        "manifest_index",
        "complete",
        "policy",
        "transition_source_kind",
        "policy_hash",
        "factory_hash",
        "envelope_hash",
        "transition_file_hash",
        "source_hashes_replayed",
        "solve_provenance",
        "transitions",
    )
    if any(field not in attempt for field in required):
        return "incomplete_attempt_provenance", "attempt_required_fields_missing"
    checks = (
        bool(str(attempt.get("source_run_id") or "")),
        TIMESTAMP_PATTERN.fullmatch(str(attempt.get("created_at") or "")) is not None,
        TIMESTAMP_PATTERN.fullmatch(str(attempt.get("published_at") or "")) is not None,
        str(attempt.get("published_at")) >= str(attempt.get("created_at")),
        _is_integer(attempt.get("manifest_index")) and int(attempt["manifest_index"]) >= 0,
        attempt.get("complete") is True,
        attempt.get("policy") == "e3",
        attempt.get("transition_source_kind") == exp7005.LIVE_SOURCE_KIND,
        all(
            _is_hash(attempt.get(field))
            for field in ("policy_hash", "factory_hash", "envelope_hash", "transition_file_hash")
        ),
        attempt.get("source_hashes_replayed") is True,
        attempt.get("solve_provenance") == "live_agent_self_discovery",
        isinstance(attempt.get("transitions"), list) and bool(attempt.get("transitions")),
    )
    if not all(checks):
        return "incomplete_attempt_provenance", "attempt_provenance_check_failed"
    transition_errors = [_transition_error(row) for row in attempt["transitions"]]
    first_error = next((error for error in transition_errors if error is not None), None)
    if first_error is not None:
        return "malformed_transition_provenance", first_error
    indices = [row["index"] for row in attempt["transitions"]]
    if indices != list(range(len(indices))):
        return "malformed_transition_provenance", "transition_order_invalid"
    return None, None


def _color_histogram(grid: Sequence[Sequence[int]]) -> list[JsonDict]:
    counts = Counter(int(cell) for row in grid for cell in row)
    return [{"color": color, "count": counts[color]} for color in sorted(counts)]


def _components(grid: Sequence[Sequence[int]]) -> list[tuple[int, int]]:
    """Count same-color nonzero components as a game-blind object proxy."""

    height = len(grid)
    width = len(grid[0])
    seen: set[tuple[int, int]] = set()
    found: list[tuple[int, int]] = []
    for y in range(height):
        for x in range(width):
            color = int(grid[y][x])
            if color == 0 or (y, x) in seen:
                continue
            queue = deque([(y, x)])
            seen.add((y, x))
            size = 0
            while queue:
                cy, cx = queue.popleft()
                size += 1
                for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                    if (
                        0 <= ny < height
                        and 0 <= nx < width
                        and (ny, nx) not in seen
                        and int(grid[ny][nx]) == color
                    ):
                        seen.add((ny, nx))
                        queue.append((ny, nx))
            found.append((color, size))
    return found


def _object_summary(grid: Sequence[Sequence[int]]) -> JsonDict:
    components = _components(grid)
    sizes = sorted(size for _, size in components)
    return {
        "color_histogram": _color_histogram(grid),
        "nonzero_cell_count": sum(int(cell) != 0 for row in grid for cell in row),
        "component_count": len(components),
        "component_size_bands": {
            "single": sum(size == 1 for size in sizes),
            "small": sum(2 <= size <= 8 for size in sizes),
            "large": sum(size > 8 for size in sizes),
        },
    }


def _bbox(points: Sequence[tuple[int, int]]) -> list[int] | None:
    if not points:
        return None
    ys = [point[0] for point in points]
    xs = [point[1] for point in points]
    return [min(xs), min(ys), max(xs), max(ys)]


def _spatial_summary(grid: Sequence[Sequence[int]]) -> JsonDict:
    points = [(y, x) for y, row in enumerate(grid) for x, cell in enumerate(row) if int(cell) != 0]
    height = len(grid)
    width = len(grid[0])
    return {
        "shape": [height, width],
        "nonzero_bbox": _bbox(points),
        "nonzero_density_ppm": round(1_000_000 * len(points) / (height * width)),
    }


def _action_region(data: Any, width: int, height: int) -> str:
    if (
        not isinstance(data, Mapping)
        or not _is_integer(data.get("x"))
        or not _is_integer(data.get("y"))
    ):
        return "no_coordinate"
    x_band = min(2, max(0, int(data["x"]) * 3 // max(1, width)))
    y_band = min(2, max(0, int(data["y"]) * 3 // max(1, height)))
    return f"{('left', 'center', 'right')[x_band]}_{('top', 'middle', 'bottom')[y_band]}"


def _state_delta(
    grid: Sequence[Sequence[int]], next_grid: Sequence[Sequence[int]], data: Any
) -> JsonDict:
    changed: list[tuple[int, int]] = []
    color_changes: Counter[tuple[int, int]] = Counter()
    added = removed = mutated = 0
    for y, (before_row, after_row) in enumerate(zip(grid, next_grid, strict=True)):
        for x, (before, after) in enumerate(zip(before_row, after_row, strict=True)):
            if before == after:
                continue
            changed.append((y, x))
            color_changes[(int(before), int(after))] += 1
            if before == 0:
                added += 1
            elif after == 0:
                removed += 1
            else:
                mutated += 1
    action_point = None
    if isinstance(data, Mapping) and _is_integer(data.get("x")) and _is_integer(data.get("y")):
        action_point = (int(data["y"]), int(data["x"]))
    touches_action = action_point in changed if action_point is not None else False
    return {
        "changed_cell_count": len(changed),
        "changed_bbox": _bbox(changed),
        "added_nonzero_count": added,
        "removed_nonzero_count": removed,
        "mutated_nonzero_count": mutated,
        "touches_action_coordinate": touches_action,
        "color_changes": [
            {"before": before, "after": after, "count": count}
            for (before, after), count in sorted(color_changes.items())
        ],
    }


def _change_scale(changed: int) -> str:
    if changed == 0:
        return "none"
    if changed == 1:
        return "single"
    if changed <= 8:
        return "local"
    if changed < 64:
        return "regional"
    return "global"


def _mechanic_signature(
    *,
    pre_action: Mapping[str, Any],
    action: Mapping[str, Any],
    next_observation: Mapping[str, Any],
    support: int,
) -> JsonDict:
    """Describe one mechanic using only fields known after this observation."""

    delta = next_observation["state_delta"]
    pre_objects = pre_action["object_summary"]
    next_objects = next_observation["object_summary"]
    height, width = pre_action["spatial_summary"]["shape"]
    hypothesis = {
        "action_type": action["type"],
        "action_region": _action_region(action.get("data"), width, height),
        "pre_component_band": min(4, int(pre_objects["component_count"]) // 4),
        "pre_density_band": min(
            4, int(pre_action["spatial_summary"]["nonzero_density_ppm"]) // 200_000
        ),
        "observed_level": pre_action["observed_level"],
    }
    outcome = {
        "change_scale": _change_scale(int(delta["changed_cell_count"])),
        "added_nonzero_count": delta["added_nonzero_count"],
        "removed_nonzero_count": delta["removed_nonzero_count"],
        "mutated_nonzero_count": delta["mutated_nonzero_count"],
        "component_count_delta": int(next_objects["component_count"])
        - int(pre_objects["component_count"]),
        "touches_action_coordinate": delta["touches_action_coordinate"],
        "level_boundary": next_observation["level_boundary"] is not None,
    }
    hypothesis_key = sha256_bytes(canonical_json_bytes(hypothesis))
    observed_outcome_key = sha256_bytes(canonical_json_bytes(outcome))
    mechanic_group = ":".join(
        (
            f"action_{action['type']}",
            str(outcome["change_scale"]),
            "boundary" if outcome["level_boundary"] else "same_level",
            "at_action" if outcome["touches_action_coordinate"] else "away_from_action",
        )
    )
    signature = {
        "action_type": action["type"],
        "object_summary": {
            "pre_component_count": pre_objects["component_count"],
            "component_count_delta": outcome["component_count_delta"],
        },
        "spatial_summary": {
            "action_region": hypothesis["action_region"],
            "changed_bbox": delta["changed_bbox"],
            "touches_action_coordinate": outcome["touches_action_coordinate"],
        },
        "state_delta": outcome,
        "level_boundary": outcome["level_boundary"],
        "support": support,
        "hypothesis_key": hypothesis_key,
        "observed_outcome_key": observed_outcome_key,
        "mechanic_group": mechanic_group,
    }
    signature["signature_hash"] = sha256_bytes(canonical_json_bytes(signature))
    return signature


def _rejection(attempt: Any, reason: str, detail: str, **extra: Any) -> JsonDict:
    identity = ""
    if isinstance(attempt, Mapping):
        identity = str(attempt.get("source_run_id") or attempt.get("source_path") or "")
    return {
        "source_record_id": sha256_bytes(identity.encode("utf-8")),
        "source_transition_id": extra.pop("source_transition_id", None),
        "reason": reason,
        "detail": detail,
        **extra,
        "terminal": True,
    }


def freeze_event_stream(attempts: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Freeze eligible attempts before calculating any mechanic utility."""

    ordered = sorted(
        [deepcopy(dict(attempt)) for attempt in attempts],
        key=lambda row: (
            str(row.get("created_at")),
            str(row.get("published_at")),
            str(row.get("source_path")),
            int(row.get("manifest_index", -1)) if _is_integer(row.get("manifest_index")) else -1,
            str(row.get("source_run_id")),
        ),
    )
    accepted: list[JsonDict] = []
    rejected: list[JsonDict] = []
    events: list[JsonDict] = []
    chronology: list[JsonDict] = []
    boundaries: list[JsonDict] = []
    pre_rows: list[JsonDict] = []
    next_rows: list[JsonDict] = []
    signature_rows: list[JsonDict] = []
    contradiction_pairs: list[JsonDict] = []
    seen_transition_ids: set[str] = set()
    hypothesis_support: Counter[str] = Counter()
    hypothesis_events: dict[str, list[tuple[str, str]]] = defaultdict(list)
    prior_level: int | None = None

    for attempt in ordered:
        reason, detail = _attempt_error(attempt)
        if reason is not None:
            rejected.append(_rejection(attempt, reason, str(detail)))
            continue
        accepted.append(
            {
                "source_record_id": sha256_bytes(str(attempt["source_run_id"]).encode("utf-8")),
                "created_at": attempt["created_at"],
                "manifest_index": attempt["manifest_index"],
                "policy": "E3AgentPolicy",
                "factory": "make_carnot_agent",
                "policy_hash": attempt["policy_hash"],
                "factory_hash": attempt["factory_hash"],
                "envelope_hash": attempt["envelope_hash"],
                "transition_file_hash": attempt["transition_file_hash"],
                "source_hashes_replayed": True,
                "solve_provenance": "live_agent_self_discovery",
                "terminal": True,
            }
        )
        for transition in attempt["transitions"]:
            transition_id = str(transition["transition_id"])
            if transition_id in seen_transition_ids:
                rejected.append(
                    _rejection(
                        attempt,
                        "duplicate_transition",
                        "an earlier chronological event has the same transition digest",
                        source_transition_id=transition_id,
                        source_transition_index=transition["index"],
                    )
                )
                continue
            seen_transition_ids.add(transition_id)
            level_before = int(transition["level_before"])
            level_after = int(transition["level_after"])
            boundary: JsonDict | None = None
            if prior_level is not None and level_before > prior_level:
                boundary = {
                    "from_level": prior_level,
                    "to_level": level_before,
                    "kind": "between_attempts",
                }
            elif level_after > level_before:
                boundary = {
                    "from_level": level_before,
                    "to_level": level_after,
                    "kind": "within_transition",
                }
            action = {"type": int(transition["action"]), "data": deepcopy(transition["data"])}
            pre_action = {
                "grid": deepcopy(transition["grid"]),
                "grid_hash": sha256_bytes(canonical_json_bytes(transition["grid"])),
                "observed_level": level_before,
                "object_summary": _object_summary(transition["grid"]),
                "spatial_summary": _spatial_summary(transition["grid"]),
                "prior_support": len(events),
            }
            next_observation = {
                "grid": deepcopy(transition["next_grid"]),
                "grid_hash": sha256_bytes(canonical_json_bytes(transition["next_grid"])),
                "observed_level": level_after,
                "object_summary": _object_summary(transition["next_grid"]),
                "spatial_summary": _spatial_summary(transition["next_grid"]),
                "state_delta": _state_delta(
                    transition["grid"], transition["next_grid"], transition["data"]
                ),
                "level_boundary": deepcopy(boundary),
            }
            provisional = _mechanic_signature(
                pre_action=pre_action,
                action=action,
                next_observation=next_observation,
                support=1,
            )
            hypothesis_key = provisional["hypothesis_key"]
            hypothesis_support[hypothesis_key] += 1
            signature = _mechanic_signature(
                pre_action=pre_action,
                action=action,
                next_observation=next_observation,
                support=hypothesis_support[hypothesis_key],
            )
            event_id = (
                "evt_"
                + hashlib.sha256(
                    canonical_json_bytes(
                        {
                            "attempt_time": attempt["created_at"],
                            "transition_index": transition["index"],
                            "transition_id": transition_id,
                        }
                    )
                ).hexdigest()[:24]
            )
            prior_conflicts = [
                prior_event_id
                for prior_event_id, prior_outcome in hypothesis_events[hypothesis_key]
                if prior_outcome != signature["observed_outcome_key"]
            ]
            contradiction = {
                "is_contradiction": bool(prior_conflicts),
                "prior_event_ids": prior_conflicts,
                "hypothesis_key": hypothesis_key,
                "observed_outcome_key": signature["observed_outcome_key"],
                "support_at_observation": hypothesis_support[hypothesis_key],
            }
            event = {
                "schema": EVENT_SCHEMA,
                "event_id": event_id,
                "stream_index": len(events),
                "source_attempt_time": attempt["created_at"],
                "source_transition_index": transition["index"],
                "pre_action": pre_action,
                "action": action,
                "next_observation": next_observation,
                "contradiction": contradiction,
                "mechanic_signature": signature,
            }
            event["row_hash"] = event_row_hash(event)
            events.append(event)
            hypothesis_events[hypothesis_key].append((event_id, signature["observed_outcome_key"]))
            chronology.append(
                {
                    "event_id": event_id,
                    "stream_index": event["stream_index"],
                    "source_attempt_time": attempt["created_at"],
                    "event_index": transition["index"],
                    "row_hash": event["row_hash"],
                    "terminal": True,
                }
            )
            pre_rows.append(
                {
                    "event_id": event_id,
                    "stream_index": event["stream_index"],
                    "grid_hash": pre_action["grid_hash"],
                    "observed_level": level_before,
                    "prior_support": pre_action["prior_support"],
                    "terminal": True,
                }
            )
            next_rows.append(
                {
                    "event_id": event_id,
                    "stream_index": event["stream_index"],
                    "grid_hash": next_observation["grid_hash"],
                    "observed_level": level_after,
                    "changed_cell_count": next_observation["state_delta"]["changed_cell_count"],
                    "terminal": True,
                }
            )
            signature_rows.append(
                {
                    "event_id": event_id,
                    "stream_index": event["stream_index"],
                    **signature,
                    "terminal": True,
                }
            )
            if boundary is not None:
                boundaries.append(
                    {
                        "event_id": event_id,
                        "stream_index": event["stream_index"],
                        **boundary,
                    }
                )
            for prior_event_id in prior_conflicts:
                pair = {
                    "hypothesis_key": hypothesis_key,
                    "prior_event_id": prior_event_id,
                    "counterexample_event_id": event_id,
                    "counterexample_observed_outcome_key": signature["observed_outcome_key"],
                    "support_at_observation": hypothesis_support[hypothesis_key],
                }
                pair["pair_id"] = sha256_bytes(canonical_json_bytes(pair))
                pair["terminal"] = True
                contradiction_pairs.append(pair)
            prior_level = level_after

    clusters: list[JsonDict] = []
    for hypothesis_key, observations in sorted(hypothesis_events.items()):
        outcomes = sorted({outcome for _, outcome in observations})
        if len(outcomes) < 2:
            continue
        cluster = {
            "hypothesis_key": hypothesis_key,
            "event_ids": [event_id for event_id, _ in observations],
            "observed_outcome_keys": outcomes,
            "support": len(observations),
        }
        cluster["cluster_id"] = sha256_bytes(canonical_json_bytes(cluster))
        cluster["terminal"] = True
        clusters.append(cluster)

    sidecar: list[JsonDict] = []
    sealed_receipts: list[JsonDict] = []
    for event in events:
        later = events[event["stream_index"] + 1 :]
        row = {
            "schema": SEALED_FUTURE_SCHEMA,
            "event_id": event["event_id"],
            "after_stream_index": event["stream_index"],
            "later_event_count": len(later),
            "later_max_observed_level": max(
                (item["next_observation"]["observed_level"] for item in later),
                default=event["next_observation"]["observed_level"],
            ),
            "later_level_boundary_count": sum(
                item["next_observation"]["level_boundary"] is not None for item in later
            ),
            "later_same_hypothesis_outcome_keys": [
                item["mechanic_signature"]["observed_outcome_key"]
                for item in later
                if item["mechanic_signature"]["hypothesis_key"]
                == event["mechanic_signature"]["hypothesis_key"]
            ],
        }
        row["row_hash"] = _sealed_row_hash(row)
        sidecar.append(row)
        sealed_receipts.append(
            {
                "event_id": event["event_id"],
                "sealed_schema": SEALED_FUTURE_SCHEMA,
                "sealed_row_hash": row["row_hash"],
                "sealed": True,
                "terminal": True,
            }
        )
    return {
        "events": events,
        "provenance_acceptance_rows": accepted,
        "provenance_rejection_rows": rejected,
        "chronology_rows": chronology,
        "level_boundary_rows": boundaries,
        "pre_action_rows": pre_rows,
        "next_observation_rows": next_rows,
        "mechanic_signature_rows": signature_rows,
        "contradiction_pair_rows": contradiction_pairs,
        "counterexample_cluster_rows": clusters,
        "sealed_future_rows": sealed_receipts,
        "_sealed_sidecar_rows": sidecar,
    }


def find_forbidden_identity_or_future_fields(value: Any) -> list[str]:
    """Return updater paths whose key names disclose identity or held future."""

    found: list[str] = []

    def visit(item: Any, path: str) -> None:
        if isinstance(item, Mapping):
            for key, nested in item.items():
                child = f"{path}.{key}" if path else str(key)
                if str(key).lower() in FORBIDDEN_UPDATER_KEYS:
                    found.append(child)
                visit(nested, child)
        elif isinstance(item, list):
            for index, nested in enumerate(item):
                visit(nested, f"{path}[{index}]")

    visit(value, "")
    return found


def _jsonl_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(canonical_json_bytes(row) + b"\n" for row in rows)


def write_jsonl_immutable(path: Path, rows: Sequence[Mapping[str, Any]]) -> str:
    """Create a read-only JSONL file and refuse changed bytes on later runs."""

    target = Path(path)
    payload = _jsonl_bytes(rows)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if target.read_bytes() != payload:
            raise ValueError(f"immutable fixture differs: {target}")
    else:
        target.write_bytes(payload)
    target.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    return sha256_bytes(payload)


def write_fixture_bundle(
    fixture_path: Path,
    sidecar_path: Path,
    events: Sequence[Mapping[str, Any]],
    sidecar_rows: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Write both immutable files and return their exact byte hashes."""

    return {
        "fixture_hash": write_jsonl_immutable(fixture_path, events),
        "held_future_sidecar_hash": write_jsonl_immutable(sidecar_path, sidecar_rows),
    }


def _read_jsonl(path: Path) -> list[JsonDict]:
    rows: list[JsonDict] = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSONL line {line_number}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"JSONL line {line_number} must be an object")
        rows.append(value)
    return rows


def load_updater_events(path: Path) -> list[JsonDict]:
    """Load only the event schema; the sealed sidecar is not an updater input."""

    rows = _read_jsonl(path)
    if rows and rows[0].get("schema") == SEALED_FUTURE_SCHEMA:
        raise ValueError("updater cannot load sealed future rows")
    for index, row in enumerate(rows):
        if row.get("schema") != EVENT_SCHEMA or set(row) != set(EVENT_FIELDS):
            raise ValueError("updater event schema mismatch")
        if row.get("stream_index") != index or row.get("row_hash") != event_row_hash(row):
            raise ValueError("updater event chronology or hash mismatch")
    return rows


def _read_json_object(path: Path) -> JsonDict | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _nearest_existing_parent(path: Path) -> Path | None:
    current = Path(path).parent
    while not current.exists() and current != current.parent:
        current = current.parent
    return current if current.is_dir() else None


def _parent_is_writable(path: Path) -> bool:
    parent = _nearest_existing_parent(path)
    if parent is None:
        return False
    try:
        return bool(parent.stat().st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))
    except OSError:
        return False


def _relative_or_absolute(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path.resolve())


def _precondition_row(resource: str, expected: Any, observed: Any) -> JsonDict:
    return {
        "resource": resource,
        "expected_value": expected,
        "observed_value": observed,
        "available": expected == observed,
        "terminal": True,
    }


def _preconditions(
    repo_root: Path, raw_dir: Path, output_path: Path
) -> tuple[list[JsonDict], JsonDict, JsonDict | None, JsonDict | None]:
    """Check source contracts before reading any transition utility."""

    exp7010 = _read_json_object(repo_root / EXP7010_PATH)
    exp7005 = _read_json_object(repo_root / EXP7005_PATH)
    rows = [
        _precondition_row("AGENTS.md", True, (repo_root / "AGENTS.md").is_file()),
        _precondition_row("CODEX.md", True, (repo_root / "CODEX.md").is_file()),
        _precondition_row("CLAUDE.md", True, (repo_root / "CLAUDE.md").is_file()),
        _precondition_row("Exp7010 artifact readable", True, exp7010 is not None),
        _precondition_row("Exp7005 artifact readable", True, exp7005 is not None),
        _precondition_row("solve registry readable", True, (repo_root / REGISTRY_PATH).is_file()),
        _precondition_row(
            "provenance helper readable", True, (repo_root / PROVENANCE_HELPER_PATH).is_file()
        ),
        _precondition_row(
            "action provenance readable", True, (repo_root / ACTION_PROVENANCE_PATH).is_file()
        ),
        _precondition_row("live policy readable", True, (repo_root / POLICY_PATH).is_file()),
        _precondition_row("raw path writable", True, _parent_is_writable(raw_dir / FIXTURE_NAME)),
        _precondition_row("result path writable", True, _parent_is_writable(output_path)),
    ]
    if exp7010 is not None:
        rows.append(
            _precondition_row(
                "Exp7010 provenance contract ready",
                1,
                exp7010.get("arc_eval_provenance_contract_ready_score"),
            )
        )
    if exp7005 is not None:
        rows.append(
            _precondition_row(
                "Exp7005 envelope audit complete",
                1,
                exp7005.get("arc_live_envelope_audit_complete_score"),
            )
        )
    if exp7010 is not None and exp7005 is not None:
        expected_hash = (exp7010.get("source_artifact_hashes") or {}).get(str(EXP7005_PATH))
        observed_hash = sha256_path(repo_root / EXP7005_PATH)
        rows.append(
            _precondition_row("Exp7005 hash anchored by Exp7010", expected_hash, observed_hash)
        )
        expected_manifests = (exp7005.get("source_artifact_hashes") or {}).get(
            "attempt_manifests", []
        )
        expected_manifest_map = {
            item.get("path"): item.get("sha256")
            for item in expected_manifests
            if isinstance(item, Mapping)
        }
        observed_manifest_map = {
            path: sha256_path(repo_root / path) for path in sorted(expected_manifest_map)
        }
        rows.append(
            _precondition_row(
                "Exp7005 eligible manifest hashes",
                expected_manifest_map,
                observed_manifest_map,
            )
        )
    sources: JsonDict = {}
    for relative in (
        EXP7010_PATH,
        EXP7005_PATH,
        EXP5155_PATH,
        REGISTRY_PATH,
        PROVENANCE_HELPER_PATH,
        POLICY_PATH,
        ACTION_PROVENANCE_PATH,
        SPEC_PATH,
    ):
        sources[str(relative)] = sha256_path(repo_root / relative)
    if exp7005 is not None:
        for item in (exp7005.get("source_artifact_hashes") or {}).get("attempt_manifests", []):
            if isinstance(item, Mapping) and isinstance(item.get("path"), str):
                sources[item["path"]] = sha256_path(repo_root / item["path"])
    return rows, sources, exp7010, exp7005


def _manifest_row(path: Path, index: int) -> JsonDict | None:
    try:
        line = path.read_text(encoding="utf-8").splitlines()[index]
        value = json.loads(line)
    except (OSError, UnicodeError, IndexError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _safe_store_path(store_root: Path, value: Any) -> Path | None:
    if not isinstance(value, str) or not value or Path(value).is_absolute():
        return None
    path = (store_root / value).resolve()
    try:
        path.relative_to(store_root.resolve())
    except ValueError:
        return None
    return path


def _source_attempts(
    repo_root: Path,
) -> tuple[list[JsonDict], list[JsonDict], list[JsonDict], JsonDict]:
    """Select complete live E3 attempts using provenance only."""

    store_root = repo_root / STORE_PATH
    replay = exp7005.enumerate_envelopes(store_root)
    eligibility = list(replay.get("envelope_eligibility_rows", []))
    eligibility.sort(
        key=lambda row: (
            str(row.get("created_at")),
            str(row.get("published_at")),
            str(row.get("manifest_path")),
            int(row.get("manifest_index", -1)),
        )
    )
    inventory_by_id = {
        row.get("record_id"): row for row in replay.get("envelope_inventory_rows", [])
    }
    inventory: list[JsonDict] = []
    for row in eligibility:
        base = inventory_by_id.get(row.get("record_id"), {})
        manifest_path = base.get("manifest_path")
        display_path = None
        if manifest_path:
            display_path = _relative_or_absolute(Path(str(manifest_path)), repo_root)
        inventory.append(
            {
                "record_id": sha256_bytes(str(row.get("record_id")).encode("utf-8")),
                "manifest_path": display_path,
                "manifest_index": row.get("manifest_index"),
                "schema": base.get("schema"),
                "parse_error": base.get("manifest_parse_error"),
                "classification": row.get("classification"),
                "eligible": row.get("eligible") is True,
                "selection_fields": [
                    "created_at",
                    "published_at",
                    "manifest_path",
                    "manifest_index",
                ],
                "belief_utility_fields_consulted": [],
                "terminal": True,
            }
        )
    attempts: list[JsonDict] = []
    initial_rejections: list[JsonDict] = []
    evidence_hashes: JsonDict = {}
    for row in eligibility:
        if row.get("eligible") is not True:
            initial_rejections.append(
                {
                    "source_record_id": sha256_bytes(str(row.get("record_id")).encode("utf-8")),
                    "source_transition_id": None,
                    "reason": str(row.get("classification") or "ineligible_manifest"),
                    "detail": [check.get("check") for check in row.get("failed_checks", [])],
                    "terminal": True,
                }
            )
            continue
        manifest_path = Path(str(row["manifest_path"]))
        manifest = _manifest_row(manifest_path, int(row["manifest_index"]))
        envelope_path = Path(str(row["envelope_path"]))
        envelope = _read_json_object(envelope_path)
        if manifest is None or envelope is None:
            initial_rejections.append(
                _rejection(row, "incomplete_attempt_provenance", "manifest_or_envelope_unreadable")
            )
            continue
        transition_path = Path(str(row["transition_path"]))
        transitions, transition_error = exp7005._read_transition_rows(transition_path)
        policy_source = _safe_store_path(store_root, envelope.get("live_policy_path"))
        factory_source = _safe_store_path(store_root, envelope.get("agent_factory_path"))
        policy_text = policy_source.read_text(encoding="utf-8") if policy_source else ""
        factory_text = factory_source.read_text(encoding="utf-8") if factory_source else ""
        symbols_verified = (
            "class E3AgentPolicy" in policy_text
            and "def make_carnot_agent" in policy_text
            and "class E3AgentPolicy" in factory_text
            and "def make_carnot_agent" in factory_text
        )
        if transition_error is not None or not symbols_verified:
            initial_rejections.append(
                _rejection(
                    row,
                    "incomplete_attempt_provenance",
                    transition_error or "live_policy_or_factory_symbol_missing",
                )
            )
            continue
        run_replay_rows = [
            item
            for name in (
                "prompt_hash_replay_rows",
                "transition_hash_replay_rows",
                "engine_hash_replay_rows",
                "environment_hash_replay_rows",
                "scorer_hash_replay_rows",
                "policy_hash_replay_rows",
                "factory_hash_replay_rows",
                "manifest_hash_replay_rows",
                "envelope_hash_replay_rows",
            )
            for item in replay.get(name, [])
            if item.get("run_id") == row.get("run_id")
        ]
        hashes_replayed = bool(run_replay_rows) and all(
            item.get("passed") is True for item in run_replay_rows
        )
        attempt = {
            "source_run_id": row["run_id"],
            "source_game": row.get("game"),
            "source_path": str(manifest_path),
            "created_at": row["created_at"],
            "published_at": row["published_at"],
            "manifest_index": row["manifest_index"],
            "complete": manifest.get("complete") is True,
            "policy": manifest.get("policy"),
            "transition_source_kind": row.get("transition_source_kind"),
            "policy_hash": envelope.get("live_policy_sha256"),
            "factory_hash": envelope.get("agent_factory_sha256"),
            "envelope_hash": row.get("envelope_sha256"),
            "transition_file_hash": row.get("transition_sha256"),
            "source_hashes_replayed": hashes_replayed,
            "solve_provenance": "live_agent_self_discovery",
            "transitions": transitions,
        }
        reason, detail = _attempt_error(attempt)
        if reason is not None:
            initial_rejections.append(_rejection(attempt, reason, str(detail)))
            continue
        attempts.append(attempt)
        for field in (
            "raw_prompt_path",
            "transition_jsonl_path",
            "engine_path",
            "environment_receipt_path",
            "scorer_path",
            "live_policy_path",
            "agent_factory_path",
            "manifest_row_path",
        ):
            evidence_path = _safe_store_path(store_root, envelope.get(field))
            if evidence_path is not None:
                evidence_hashes[_relative_or_absolute(evidence_path, repo_root)] = sha256_path(
                    evidence_path
                )
        evidence_hashes[_relative_or_absolute(envelope_path, repo_root)] = sha256_path(
            envelope_path
        )
    return inventory, attempts, initial_rejections, evidence_hashes


def _gate(check: str, expected: Any, observed: Any, *, passed: bool | None = None) -> JsonDict:
    return {
        "check": check,
        "expected_value": expected,
        "observed_value": observed,
        "passed": bool(expected == observed if passed is None else passed),
        "terminal": True,
    }


def _gate_summary(checks: Sequence[Mapping[str, Any]]) -> JsonDict:
    copied = [deepcopy(dict(row)) for row in checks]
    failed = next((row for row in copied if row.get("passed") is not True), None)
    return {
        "checks": copied,
        "failed_check": failed.get("check") if failed else None,
        "expected_value": failed.get("expected_value") if failed else "all checks pass",
        "observed_value": failed.get("observed_value") if failed else "all checks pass",
        "passed": failed is None,
    }


def _aggregate_rows(signatures: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    counts = Counter(str(row.get("mechanic_group")) for row in signatures)
    return [
        {"mechanic_group": group, "event_count": counts[group], "terminal": True}
        for group in sorted(counts)
    ]


def _artifact_checksum_payload(artifact: Mapping[str, Any]) -> JsonDict:
    excluded = {
        "duration_s",
        "fixture_path",
        "held_future_sidecar_path",
        "fresh_process_replay_rows",
        "gate_check_summary",
        "reproducibility_checksum",
    }
    return {key: deepcopy(value) for key, value in artifact.items() if key not in excluded}


def artifact_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash deterministic manifest content while excluding runtime locations."""

    return sha256_bytes(canonical_json_bytes(_artifact_checksum_payload(artifact)))


def _empty_artifact(
    *,
    run_date: str,
    source_hashes: Mapping[str, Any],
    preconditions: Sequence[Mapping[str, Any]],
    fixture_path: Path,
    sidecar_path: Path,
) -> JsonDict:
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "execution_date": run_date,
        "field_principles": deepcopy(FIELD_PRINCIPLES),
        "preconditions_checked": [deepcopy(dict(row)) for row in preconditions],
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": 0.0,
        "source_artifact_hashes": deepcopy(dict(source_hashes)),
        "attempt_manifest_rows": [],
        "provenance_acceptance_rows": [],
        "provenance_rejection_rows": [],
        "chronology_rows": [],
        "level_boundary_rows": [],
        "pre_action_rows": [],
        "next_observation_rows": [],
        "mechanic_signature_rows": [],
        "contradiction_pair_rows": [],
        "counterexample_cluster_rows": [],
        "sealed_future_rows": [],
        "leakage_check_rows": [],
        "fixture_path": str(fixture_path),
        "held_future_sidecar_path": str(sidecar_path),
        "fixture_hash": None,
        "held_future_sidecar_hash": None,
        "fresh_process_replay_rows": [],
        "rows": [],
        "solve_provenance": None,
        "arc_belief_stream_ready_score": 0,
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
        "gate_check_summary": {},
        "verifier_is_oracle": False,
        "verdict_class": "blocked",
        "honest_verdict": "blocked_arc_belief_stream_fixture",
    }
    return artifact


def _fresh_process_replay(repo_root: Path, run_date: str) -> JsonDict:
    """Rebuild in a child interpreter without reusing this process's state."""

    env = dict(os.environ)
    python_path = str(repo_root / "python")
    env["PYTHONPATH"] = python_path + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "carnot.experiment_7019_arc_belief_stream_fixture",
            "--date",
            run_date,
            "--repo-root",
            str(repo_root),
            "--fresh-replay-json",
        ],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError:
        payload = {}
    return {
        "process_exit_code": completed.returncode,
        "fixture_hash": payload.get("fixture_hash"),
        "held_future_sidecar_hash": payload.get("held_future_sidecar_hash"),
        "reproducibility_checksum": payload.get("reproducibility_checksum"),
        "stderr": completed.stderr[-500:],
    }


def build_artifact(
    *,
    run_date: str = RUN_DATE,
    repo_root: Path = REPO_ROOT,
    raw_dir: Path | None = None,
    output_path: Path | None = None,
    fresh_process: bool = True,
) -> JsonDict:
    """Build one positive fixture or a complete blocked result."""

    started = time.monotonic()
    root = Path(repo_root).resolve()
    selected_raw_dir = Path(raw_dir) if raw_dir is not None else root / RAW_DIR
    selected_output = Path(output_path) if output_path is not None else root / OUTPUT_PATH
    fixture_path = selected_raw_dir / FIXTURE_NAME
    sidecar_path = selected_raw_dir / SIDECAR_NAME
    preconditions, source_hashes, _exp7010, _exp7005 = _preconditions(
        root, selected_raw_dir, selected_output
    )
    artifact = _empty_artifact(
        run_date=run_date,
        source_hashes=source_hashes,
        preconditions=preconditions,
        fixture_path=fixture_path,
        sidecar_path=sidecar_path,
    )
    failed_precondition = next((row for row in preconditions if row["available"] is not True), None)
    if failed_precondition is not None:
        artifact["gate_check_summary"] = _gate_summary(
            [
                _gate(
                    str(row["resource"]),
                    row["expected_value"],
                    row["observed_value"],
                    passed=row["available"],
                )
                for row in preconditions
            ]
        )
        artifact["duration_s"] = round(time.monotonic() - started, 6)
        artifact["reproducibility_checksum"] = artifact_checksum(artifact)
        return artifact

    inventory, attempts, initial_rejections, evidence_hashes = _source_attempts(root)
    artifact["attempt_manifest_rows"] = inventory
    artifact["source_artifact_hashes"].update(evidence_hashes)
    frozen = freeze_event_stream(attempts)
    artifact["provenance_acceptance_rows"] = frozen["provenance_acceptance_rows"]
    artifact["provenance_rejection_rows"] = initial_rejections + frozen["provenance_rejection_rows"]
    for field in (
        "chronology_rows",
        "level_boundary_rows",
        "pre_action_rows",
        "next_observation_rows",
        "mechanic_signature_rows",
        "contradiction_pair_rows",
        "counterexample_cluster_rows",
        "sealed_future_rows",
    ):
        artifact[field] = frozen[field]
    hashes = write_fixture_bundle(
        fixture_path,
        sidecar_path,
        frozen["events"],
        frozen["_sealed_sidecar_rows"],
    )
    artifact.update(hashes)
    artifact["fixture_path"] = _relative_or_absolute(fixture_path, root)
    artifact["held_future_sidecar_path"] = _relative_or_absolute(sidecar_path, root)
    fixture_leaks = find_forbidden_identity_or_future_fields(frozen["events"])
    pair_leaks = find_forbidden_identity_or_future_fields(frozen["contradiction_pair_rows"])
    cluster_leaks = find_forbidden_identity_or_future_fields(frozen["counterexample_cluster_rows"])
    try:
        load_updater_events(sidecar_path)
        sidecar_rejected = False
    except ValueError as exc:
        sidecar_rejected = str(exc) == "updater cannot load sealed future rows"
    artifact["leakage_check_rows"] = [
        _gate("fixture_forbidden_identity_or_future_fields", [], fixture_leaks),
        _gate("contradiction_pair_forbidden_fields", [], pair_leaks),
        _gate("counterexample_cluster_forbidden_fields", [], cluster_leaks),
        _gate("updater_rejects_sealed_sidecar", True, sidecar_rejected),
        _gate(
            "updater_event_schema_exact",
            len(frozen["events"]),
            len(load_updater_events(fixture_path)),
        ),
    ]
    artifact["rows"] = _aggregate_rows(frozen["mechanic_signature_rows"])
    artifact["solve_provenance"] = (
        "live_agent_self_discovery" if artifact["provenance_acceptance_rows"] else None
    )
    mechanic_groups = {row["mechanic_group"] for row in artifact["mechanic_signature_rows"]}
    complete_chronology = bool(frozen["events"]) and all(
        row["stream_index"] == index
        and row["row_hash"] == event_row_hash(row)
        and set(row) == set(EVENT_FIELDS)
        for index, row in enumerate(frozen["events"])
    )
    zero_leakage = all(row["passed"] is True for row in artifact["leakage_check_rows"])
    base_checks = [
        _gate("preconditions", True, True),
        _gate("eligible_live_attempt_count", ">=1", len(attempts), passed=len(attempts) >= 1),
        _gate(
            "mechanic_group_count",
            ">=3",
            len(mechanic_groups),
            passed=len(mechanic_groups) >= 3,
        ),
        _gate("complete_chronological_rows", True, complete_chronology),
        _gate(
            "future_leakage_count",
            0,
            sum(not row["passed"] for row in artifact["leakage_check_rows"]),
        ),
    ]
    local_ready = all(row["passed"] is True for row in base_checks) and zero_leakage
    artifact["arc_belief_stream_ready_score"] = int(local_ready)
    artifact["verdict_class"] = "positive" if local_ready else "partial"
    artifact["honest_verdict"] = (
        "complete_positive_arc_belief_stream_fixture_ready"
        if local_ready
        else "partial_arc_belief_stream_fixture_gate_failed"
    )
    artifact["gate_check_summary"] = _gate_summary(base_checks)
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)

    replay_row: JsonDict
    if fresh_process:
        observed = _fresh_process_replay(root, run_date)
        passed = bool(
            observed["process_exit_code"] == 0
            and observed["fixture_hash"] == artifact["fixture_hash"]
            and observed["held_future_sidecar_hash"] == artifact["held_future_sidecar_hash"]
            and observed["reproducibility_checksum"] == artifact["reproducibility_checksum"]
        )
        replay_row = {
            "expected_fixture_hash": artifact["fixture_hash"],
            "observed_fixture_hash": observed["fixture_hash"],
            "expected_held_future_sidecar_hash": artifact["held_future_sidecar_hash"],
            "observed_held_future_sidecar_hash": observed["held_future_sidecar_hash"],
            "expected_reproducibility_checksum": artifact["reproducibility_checksum"],
            "observed_reproducibility_checksum": observed["reproducibility_checksum"],
            "process_exit_code": observed["process_exit_code"],
            "stderr": observed["stderr"],
            "passed": passed,
            "terminal": True,
        }
    else:
        replay_row = {
            "expected_fixture_hash": artifact["fixture_hash"],
            "observed_fixture_hash": artifact["fixture_hash"],
            "expected_held_future_sidecar_hash": artifact["held_future_sidecar_hash"],
            "observed_held_future_sidecar_hash": artifact["held_future_sidecar_hash"],
            "expected_reproducibility_checksum": artifact["reproducibility_checksum"],
            "observed_reproducibility_checksum": artifact["reproducibility_checksum"],
            "process_exit_code": 0,
            "stderr": "",
            "passed": True,
            "terminal": True,
        }
    artifact["fresh_process_replay_rows"] = [replay_row]
    final_checks = base_checks + [_gate("fresh_process_stable_hashes", True, replay_row["passed"])]
    ready = all(row["passed"] is True for row in final_checks)
    artifact["arc_belief_stream_ready_score"] = int(ready)
    artifact["verdict_class"] = "positive" if ready else "partial"
    artifact["honest_verdict"] = (
        "complete_positive_arc_belief_stream_fixture_ready"
        if ready
        else "partial_arc_belief_stream_fixture_gate_failed"
    )
    artifact["gate_check_summary"] = _gate_summary(final_checks)
    artifact["duration_s"] = round(time.monotonic() - started, 6)
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return artifact


def _resolve_artifact_path(value: Any, root: Path) -> Path | None:
    if not isinstance(value, str) or not value:
        return None
    path = Path(value)
    return path if path.is_absolute() else root / path


def validate_artifact(artifact: Any, *, repo_root: Path = REPO_ROOT) -> list[str]:
    """Validate the result without trusting its self-reported readiness bit."""

    if not isinstance(artifact, Mapping):
        return ["artifact_object_required"]
    errors: list[str] = []
    missing = sorted(set(REQUIRED_ARTIFACT_FIELDS) - set(artifact))
    if missing:
        errors.append("missing_required_fields:" + ",".join(missing))
    if set(artifact.get("field_principles", {})) != set(REQUIRED_ARTIFACT_FIELDS):
        errors.append("field_principles_mismatch")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate_mismatch")
    verdict_class = artifact.get("verdict_class")
    verdict = str(artifact.get("honest_verdict") or "")
    if verdict_class not in VERDICT_CLASSES:
        errors.append("verdict_class_invalid")
    prefix_ok = (
        (verdict_class == "positive" and verdict.startswith("complete_positive_"))
        or (verdict_class == "circular_positive" and verdict.startswith("complete_circular_"))
        or (verdict_class == "null" and verdict.startswith("complete_null_"))
        or (verdict_class == "blocked" and verdict == "blocked_arc_belief_stream_fixture")
        or (verdict_class == "disqualified" and verdict.startswith("disqualified_"))
        or (verdict_class == "partial" and verdict.startswith("partial_"))
    )
    if not prefix_ok:
        errors.append("verdict_prefix_mismatch")
    if artifact.get("reproducibility_checksum") != artifact_checksum(artifact):
        errors.append("reproducibility_checksum_mismatch")
    gate = artifact.get("gate_check_summary")
    if not isinstance(gate, Mapping):
        errors.append("gate_check_summary_invalid")
    elif verdict_class == "blocked":
        if gate.get("passed") is not False or not gate.get("failed_check"):
            errors.append("blocked_gate_summary_invalid")
        if artifact.get("arc_belief_stream_ready_score") != 0:
            errors.append("blocked_ready_score_nonzero")
        return errors
    if (
        artifact.get("arc_belief_stream_ready_score") != 1
        or not isinstance(gate, Mapping)
        or gate.get("passed") is not True
    ):
        errors.append("positive_readiness_gate_failed")
    fixture_path = _resolve_artifact_path(artifact.get("fixture_path"), Path(repo_root))
    sidecar_path = _resolve_artifact_path(artifact.get("held_future_sidecar_path"), Path(repo_root))
    if fixture_path is None or sha256_path(fixture_path) != artifact.get("fixture_hash"):
        errors.append("fixture_hash_mismatch")
    else:
        try:
            events = load_updater_events(fixture_path)
        except (OSError, UnicodeError, ValueError):
            events = []
            errors.append("fixture_rows_invalid")
        if len(events) != len(artifact.get("chronology_rows", [])):
            errors.append("chronology_count_mismatch")
        if find_forbidden_identity_or_future_fields(events):
            errors.append("fixture_future_or_identity_leak")
    if sidecar_path is None or sha256_path(sidecar_path) != artifact.get(
        "held_future_sidecar_hash"
    ):
        errors.append("held_future_sidecar_hash_mismatch")
    elif sidecar_path is not None:
        try:
            sidecar_rows = _read_jsonl(sidecar_path)
        except (OSError, UnicodeError, ValueError):
            sidecar_rows = []
            errors.append("held_future_sidecar_invalid")
        if any(
            row.get("schema") != SEALED_FUTURE_SCHEMA
            or row.get("row_hash") != _sealed_row_hash(row)
            for row in sidecar_rows
        ):
            errors.append("held_future_sidecar_row_invalid")
    groups = {row.get("mechanic_group") for row in artifact.get("mechanic_signature_rows", [])}
    if len(groups) < 3:
        errors.append("mechanic_group_count_below_three")
    if any(
        row.get("solve_provenance") != "live_agent_self_discovery"
        for row in artifact.get("provenance_acceptance_rows", [])
    ):
        errors.append("solve_provenance_mismatch")
    if artifact.get("solve_provenance") != "live_agent_self_discovery":
        errors.append("artifact_solve_provenance_mismatch")
    if not artifact.get("fresh_process_replay_rows") or any(
        row.get("passed") is not True for row in artifact["fresh_process_replay_rows"]
    ):
        errors.append("fresh_process_replay_failed")
    return list(dict.fromkeys(errors))


def _write_artifact(path: Path, artifact: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--raw-dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--fresh-replay-json", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the fixture command and write its result manifest."""

    args = _parser().parse_args(argv)
    root = args.repo_root.resolve()
    if args.fresh_replay_json:
        with tempfile.TemporaryDirectory(prefix="carnot-exp7019-") as directory:
            artifact = build_artifact(
                run_date=args.date,
                repo_root=root,
                raw_dir=Path(directory),
                output_path=Path(directory) / "result.json",
                fresh_process=False,
            )
        print(
            json.dumps(
                {
                    "fixture_hash": artifact["fixture_hash"],
                    "held_future_sidecar_hash": artifact["held_future_sidecar_hash"],
                    "reproducibility_checksum": artifact["reproducibility_checksum"],
                },
                sort_keys=True,
            )
        )
        return 0 if artifact["verdict_class"] == "positive" else 1
    raw_dir = args.raw_dir if args.raw_dir is not None else root / RAW_DIR
    output = args.output if args.output is not None else root / OUTPUT_PATH
    artifact = build_artifact(
        run_date=args.date,
        repo_root=root,
        raw_dir=raw_dir,
        output_path=output,
        fresh_process=True,
    )
    _write_artifact(output, artifact)
    errors = validate_artifact(artifact, repo_root=root)
    return 0 if not errors else 1


if __name__ == "__main__":  # pragma: no cover - the wrapper is the public command.
    raise SystemExit(main())
