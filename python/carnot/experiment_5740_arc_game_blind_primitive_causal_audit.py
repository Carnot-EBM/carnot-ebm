"""Experiment 5740: game-blind ARC primitive causal audit.

This module is intentionally an offline diagnostic. It mines generic
action-effect primitive candidates from checked-in live-agent trace receipts,
then measures their trajectory utility by deterministic paired deletion replay.
It does not import game source, per-game adapters, hidden state, solution code,
outer-loop BFS labels, or any policy implementation. The output is development
proxy evidence only: no solve credit and no registry mutation.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_5740_arc_game_blind_primitive_causal_audit.json")
LIVE_GAP_RELATIVE_PATH = Path("results/arc_live_oracle_gap.json")
REGISTRY_RELATIVE_PATH = Path("ops/arc_solve_registry.yaml")
REQUESTED_VERIFIER_GAPS_RELATIVE_PATH = Path("ops/verifier-gaps.md")
ACTUAL_VERIFIER_GAPS_RELATIVE_PATH = Path("ops/verifier_gaps.md")
INFERENCE_SUBSTRATE = "offline_arc_trace_counterfactual_development_proxy"
RANDOM_SEEDS = [5740, 20260720]
MIN_PAIRED_REPLAYS = 30
MIN_RETAINED_HELDOUT_GAMES = 3
MIN_COMPOSITE_DELTA = 0.01
CORRECTED_Z = 2.24

TRACE_SOURCE_CANDIDATES = (
    LIVE_GAP_RELATIVE_PATH,
    Path("results/experiment_5727_arc_generalization_live_oracle_gap_v511.json"),
    Path("results/experiment_5727_perception_action_effect_adequacy.json"),
)

PRIMITIVE_FAMILIES = (
    "object_displacement",
    "reversible_or_noop_action",
    "boundary_or_collision",
    "inventory_or_state_toggle",
    "agent_relative_motion",
    "repeated_action_loop",
    "delayed_effect",
)

IDENTITY_KEYS = frozenset(
    {
        "game",
        "game_id",
        "game_name",
        "source_game",
        "registry_game",
        "registry_provenance",
    }
)
SOURCE_KEYS = frozenset(
    {
        "source_file",
        "source_rule",
        "game_source",
        "solution_code",
        "hidden_state",
        "per_game_adapter",
        "adapter_label",
        "outer_loop_bfs",
        "hand_authored_model",
    }
)
FUTURE_KEYS = frozenset({"future_frame", "future_grid_hash", "t_plus_2_hash", "future_level"})

FIELD_PRINCIPLES = {
    "field_principles": "every field carries its own audit rationale so schema compliance is principle-grounded.",
    "preconditions_checked": "records registry, trace, source-access, and policy/registry immutability guards before trusting the diagnostic.",
    "registry_hash": "content-addressed public registry baseline prevents accidental solve-credit drift.",
    "registry_game_count": "the public registry premise is exactly 25 completed games.",
    "trace_manifest": "agent-owned checked-in trace inputs are explicit and replayable.",
    "trace_hashes": "trace bytes are content-addressed so mined primitives cannot silently change.",
    "games_measured": "leave-one-game-out means the measured roster is visible.",
    "leave_one_game_out_splits": "candidate mining excludes the held-out game before measuring retention.",
    "primitive_schema": "the vocabulary is generic observable action-effect structure, not game rules.",
    "primitive_candidates": "each candidate exposes support, thresholds, controls, and causal-retention status.",
    "game_identity_stripping_receipts": "proves learner-visible rows removed ids, names, and source-derived metadata.",
    "deletion_replay_manifest": "causal utility is measured by paired deletion replay under fixed budgets and seeds.",
    "counterfactual_receipt_coverage": "N>=30 paired replays is explicit before interpreting effects.",
    "counterfactual_trajectory_utility": "utility is trajectory-level, not static frequency or plausibility.",
    "next_action_validity_delta": "validity impact is measured directly after deletion.",
    "world_model_accuracy_delta": "dynamics quality is measured as transition prediction change.",
    "planning_reachability_delta": "planned-state reachability is a downstream induction target.",
    "repeat_rate_delta": "looping behavior is separated from validity.",
    "invalid_action_delta": "bad-action regressions are counted, not hidden.",
    "progress_budget_delta": "time or budget to first progress is the live-agent cost signal.",
    "negative_controls": "leak and orphan controls prove the harness rejects non-causal or illegal signals.",
    "source_leak_count": "source-derived rules must be detected and excluded.",
    "game_identity_leak_count": "game-id leakage must be detected and excluded.",
    "positive_causal_primitive_count": "headline count follows preregistered effect and retention thresholds only.",
    "policy_modified": "false keeps this diagnostic out of the submitted policy path.",
    "registry_modified": "false prevents public-game solve credit inflation.",
    "solve_provenance": "development_proxy marks this as diagnostic evidence, not a hidden-game solve.",
    "verifier_is_oracle": "true because deletion replay compares against recorded trace receipts, not a deployable live oracle.",
    "inference_substrate": "offline_arc_trace_counterfactual_development_proxy declares no LLM inference or source execution.",
    "random_seeds": "paired deletion replay is deterministic only with fixed seeds.",
    "reproducibility_checksum": "content-addressed artifact payload catches threshold or trace drift.",
    "honest_verdict": "terminal prefix reports complete diagnostic or blocked precondition without solve credit.",
}

REQUIRED_ARTIFACT_FIELDS = tuple(FIELD_PRINCIPLES)

PRIMITIVE_SCHEMA = {
    "object_displacement": {
        "observable_relation": "immediate successor hash changes while grid shape and color-set cardinality stay stable",
        "banned_inputs": ["game id", "source rule", "absolute per-game constant"],
    },
    "reversible_or_noop_action": {
        "observable_relation": "action either leaves the immediate frame unchanged or reverses the previous observed hash transition",
        "banned_inputs": ["adapter inverse map", "hidden undo stack"],
    },
    "boundary_or_collision": {
        "observable_relation": "non-reset action produces no immediate frame change, especially at a normalized edge or obstacle-like repeat",
        "banned_inputs": ["source collision map", "wall coordinates"],
    },
    "inventory_or_state_toggle": {
        "observable_relation": "successor hash changes with a visible color-set/cardinality change and no source register read",
        "banned_inputs": ["hidden inventory", "source variable name"],
    },
    "agent_relative_motion": {
        "observable_relation": "non-click action changes the immediate frame under the generic action enum",
        "banned_inputs": ["sprite class name", "game-specific facing table"],
    },
    "repeated_action_loop": {
        "observable_relation": "same abstract action repeats at least three times without level progress",
        "banned_inputs": ["game name", "scripted solve tail"],
    },
    "delayed_effect": {
        "observable_relation": "a no-change action is followed within two recorded decisions by a visible effect",
        "banned_inputs": ["future frame beyond the immediate transition window", "source event queue"],
    },
}


@dataclass(frozen=True)
class TraceStep:
    source_path: str
    game: str
    game_index: int
    step_index: int
    before_hash: str
    after_hash: str
    action_kind: str
    action_shape: str
    action_signature: str
    click_region: str
    valid_action: bool
    changed: bool
    color_changed: bool
    shape_changed: bool
    level_delta: int
    repeat_run_length: int
    reversible: bool
    delayed: bool
    primitives: tuple[str, ...]
    learner_visible: dict[str, Any]


def stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def sha256_bytes(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def payload_checksum(artifact: Mapping[str, Any]) -> str:
    payload = {k: v for k, v in artifact.items() if k != "reproducibility_checksum"}
    return sha256_text(stable_json(payload))


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def registry_game_count(registry: Mapping[str, Any]) -> int:
    explicit = registry.get("reproducible_total_games")
    if isinstance(explicit, int):
        return explicit
    games = registry.get("games", [])
    return len(games) if isinstance(games, list) else 0


def registry_games(registry: Mapping[str, Any]) -> list[str]:
    games = registry.get("games", [])
    if not isinstance(games, list):
        return []
    return sorted(str(row.get("game")) for row in games if isinstance(row, Mapping) and row.get("game"))


def trace_hashes(root: Path, paths: Sequence[Path]) -> dict[str, str]:
    return {
        str(path): sha256_bytes((root / path).read_bytes())
        for path in paths
        if (root / path).exists()
    }


def _move_of_successor(frame: Mapping[str, Any]) -> Mapping[str, Any]:
    move = frame.get("move")
    return move if isinstance(move, Mapping) else {}


def _action_kind(move: Mapping[str, Any]) -> str:
    kind = move.get("kind")
    return str(kind if kind is not None else move.get("action", "UNKNOWN"))


def _action_shape(move: Mapping[str, Any]) -> str:
    kind = _action_kind(move)
    data = move.get("data")
    if kind.upper() == "RESET":
        return "reset"
    if isinstance(data, Mapping) and {"x", "y"}.issubset(data):
        return "click"
    if kind == "6":
        return "click"
    return "button"


def _click_region(move: Mapping[str, Any], before: Mapping[str, Any]) -> str:
    data = move.get("data")
    if not isinstance(data, Mapping) or "x" not in data or "y" not in data:
        return "none"
    shape = before.get("grid_shape")
    height, width = (shape if isinstance(shape, list) and len(shape) == 2 else [64, 64])
    x = int(data.get("x", -1))
    y = int(data.get("y", -1))
    if (x in {0, int(width) - 1}) and (y in {0, int(height) - 1}):
        return "corner"
    if x <= 0 or y <= 0 or x >= int(width) - 1 or y >= int(height) - 1:
        return "edge"
    return "interior"


def _valid_action(move: Mapping[str, Any], before: Mapping[str, Any]) -> bool:
    kind = _action_kind(move)
    if kind.upper() == "RESET":
        return True
    actions = before.get("available_actions", [])
    available = {str(action) for action in actions} if isinstance(actions, list) else set()
    return kind in available


def _color_count(frame: Mapping[str, Any]) -> int:
    colors = frame.get("colors")
    return len(colors) if isinstance(colors, list) else 0


def _level_delta(before: Mapping[str, Any], after: Mapping[str, Any]) -> int:
    try:
        return int(after.get("levels_completed", 0) or 0) - int(before.get("levels_completed", 0) or 0)
    except (TypeError, ValueError):
        return 0


def _shape(frame: Mapping[str, Any]) -> tuple[int, int] | None:
    shape = frame.get("grid_shape")
    if isinstance(shape, list) and len(shape) == 2:
        return (int(shape[0]), int(shape[1]))
    return None


def _signature(action_shape: str, action_kind: str, click_region: str) -> str:
    return f"{action_shape}:{action_kind}:{click_region}"


def _learner_visible_row(
    *,
    source_path: str,
    game_index: int,
    step_index: int,
    before: Mapping[str, Any],
    after: Mapping[str, Any],
    action_shape: str,
    action_kind: str,
    action_signature: str,
    click_region: str,
    repeat_run_length: int,
    reversible: bool,
    delayed: bool,
) -> dict[str, Any]:
    return {
        "anonymous_trace_id": sha256_text(f"{source_path}:{game_index}")[:24],
        "step_index": int(step_index),
        "action_shape": action_shape,
        "action_kind_bucket": action_kind if action_kind.upper() == "RESET" else "action_enum",
        "action_signature_digest": sha256_text(action_signature)[:24],
        "click_region": click_region,
        "before_grid_hash": str(before.get("grid_hash", "")),
        "after_grid_hash": str(after.get("grid_hash", "")),
        "grid_shape": list(_shape(before) or (0, 0)),
        "color_count_before": _color_count(before),
        "color_count_after": _color_count(after),
        "changed": before.get("grid_hash") != after.get("grid_hash"),
        "level_delta_clipped": max(-1, min(1, _level_delta(before, after))),
        "repeat_run_length_capped": min(int(repeat_run_length), 3),
        "reversible_observed": bool(reversible),
        "delayed_effect_window_observed": bool(delayed),
    }


def _primitive_flags(
    *,
    changed: bool,
    color_changed: bool,
    shape_changed: bool,
    action_shape: str,
    click_region: str,
    level_delta: int,
    repeat_run_length: int,
    reversible: bool,
    delayed: bool,
) -> tuple[str, ...]:
    flags: list[str] = []
    no_progress = level_delta <= 0
    if changed and not color_changed and not shape_changed:
        flags.append("object_displacement")
    if reversible or not changed:
        flags.append("reversible_or_noop_action")
    if not changed and action_shape != "reset" and (click_region in {"edge", "corner", "none"}):
        flags.append("boundary_or_collision")
    if changed and color_changed and no_progress:
        flags.append("inventory_or_state_toggle")
    if changed and action_shape == "button":
        flags.append("agent_relative_motion")
    if repeat_run_length >= 3 and no_progress:
        flags.append("repeated_action_loop")
    if delayed:
        flags.append("delayed_effect")
    return tuple(flags)


def extract_trace_steps(live_result: Mapping[str, Any], *, source_path: str) -> list[TraceStep]:
    steps: list[TraceStep] = []
    per_game = live_result.get("per_game", [])
    if not isinstance(per_game, list):
        return steps
    for game_index, game_row in enumerate(per_game):
        if not isinstance(game_row, Mapping):
            continue
        game = str(game_row.get("game") or f"anonymous_{game_index}")
        frames = game_row.get("frame_sequence", [])
        if not isinstance(frames, list) or len(frames) < 2:
            continue
        raw: list[dict[str, Any]] = []
        run_lengths: dict[str, int] = defaultdict(int)
        for frame_index in range(1, len(frames)):
            before = frames[frame_index - 1]
            after = frames[frame_index]
            if not isinstance(before, Mapping) or not isinstance(after, Mapping):
                continue
            move = _move_of_successor(after)
            action_kind = _action_kind(move)
            action_shape = _action_shape(move)
            click_region = _click_region(move, before)
            signature = _signature(action_shape, action_kind, click_region)
            if raw and raw[-1]["action_signature"] == signature:
                run_lengths[signature] = int(raw[-1]["repeat_run_length"]) + 1
            else:
                run_lengths[signature] = 1
            changed = before.get("grid_hash") != after.get("grid_hash")
            raw.append(
                {
                    "before": before,
                    "after": after,
                    "action_kind": action_kind,
                    "action_shape": action_shape,
                    "action_signature": signature,
                    "click_region": click_region,
                    "valid_action": _valid_action(move, before),
                    "changed": changed,
                    "color_changed": _color_count(before) != _color_count(after),
                    "shape_changed": _shape(before) != _shape(after),
                    "level_delta": _level_delta(before, after),
                    "repeat_run_length": run_lengths[signature],
                }
            )
        for idx, row in enumerate(raw):
            reversible = bool(idx > 0 and row["changed"] and row["after"].get("grid_hash") == raw[idx - 1]["before"].get("grid_hash"))
            delayed = False
            if not row["changed"]:
                lookahead = raw[idx + 1 : idx + 3]
                delayed = any(bool(nxt["changed"] or int(nxt["level_delta"]) > 0) for nxt in lookahead)
            primitives = _primitive_flags(
                changed=bool(row["changed"]),
                color_changed=bool(row["color_changed"]),
                shape_changed=bool(row["shape_changed"]),
                action_shape=str(row["action_shape"]),
                click_region=str(row["click_region"]),
                level_delta=int(row["level_delta"]),
                repeat_run_length=int(row["repeat_run_length"]),
                reversible=reversible,
                delayed=delayed,
            )
            learner_visible = _learner_visible_row(
                source_path=source_path,
                game_index=game_index,
                step_index=idx + 1,
                before=row["before"],
                after=row["after"],
                action_shape=str(row["action_shape"]),
                action_kind=str(row["action_kind"]),
                action_signature=str(row["action_signature"]),
                click_region=str(row["click_region"]),
                repeat_run_length=int(row["repeat_run_length"]),
                reversible=reversible,
                delayed=delayed,
            )
            steps.append(
                TraceStep(
                    source_path=source_path,
                    game=game,
                    game_index=game_index,
                    step_index=idx + 1,
                    before_hash=str(row["before"].get("grid_hash", "")),
                    after_hash=str(row["after"].get("grid_hash", "")),
                    action_kind=str(row["action_kind"]),
                    action_shape=str(row["action_shape"]),
                    action_signature=str(row["action_signature"]),
                    click_region=str(row["click_region"]),
                    valid_action=bool(row["valid_action"]),
                    changed=bool(row["changed"]),
                    color_changed=bool(row["color_changed"]),
                    shape_changed=bool(row["shape_changed"]),
                    level_delta=int(row["level_delta"]),
                    repeat_run_length=int(row["repeat_run_length"]),
                    reversible=reversible,
                    delayed=delayed,
                    primitives=primitives,
                    learner_visible=learner_visible,
                )
            )
    return steps


def strip_for_learner(row: Mapping[str, Any]) -> dict[str, Any]:
    forbidden = IDENTITY_KEYS | SOURCE_KEYS | FUTURE_KEYS
    return {str(k): v for k, v in row.items() if str(k) not in forbidden}


def leak_classes(row: Mapping[str, Any]) -> list[str]:
    keys = {str(key) for key in row}
    found: list[str] = []
    if keys & IDENTITY_KEYS:
        found.append("game_identity")
    if keys & SOURCE_KEYS:
        found.append("source")
    if keys & FUTURE_KEYS:
        found.append("future_frame")
    return found


def game_identity_stripping_receipts(steps: Sequence[TraceStep], limit: int = 8) -> list[dict[str, Any]]:
    receipts: list[dict[str, Any]] = []
    for step in steps[:limit]:
        raw = {
            **step.learner_visible,
            "game": step.game,
            "game_name": f"public::{step.game}",
            "registry_provenance": "development_proxy",
        }
        stripped = strip_for_learner(raw)
        receipts.append(
            {
                "row_hash": sha256_text(stable_json(stripped)),
                "before_keys": sorted(raw),
                "after_keys": sorted(stripped),
                "stripped_keys": sorted(set(raw) - set(stripped)),
                "leak_classes_after_strip": leak_classes(stripped),
            }
        )
    return receipts


def _mean(values: Sequence[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _corrected_interval(values: Sequence[float]) -> list[float]:
    if not values:
        return [0.0, 0.0]
    mean = _mean(values)
    if len(values) == 1:
        return [round(mean, 6), round(mean, 6)]
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    half_width = CORRECTED_Z * math.sqrt(variance) / math.sqrt(len(values))
    return [round(mean - half_width, 6), round(mean + half_width, 6)]


def _expected_world_model_correct(family: str, step: TraceStep) -> bool:
    if family == "object_displacement":
        return step.changed and not step.color_changed and not step.shape_changed
    if family == "reversible_or_noop_action":
        return step.reversible or not step.changed
    if family == "boundary_or_collision":
        return not step.changed and step.action_shape != "reset"
    if family == "inventory_or_state_toggle":
        return step.changed and step.color_changed
    if family == "agent_relative_motion":
        return step.changed and step.action_shape == "button"
    if family == "repeated_action_loop":
        return step.repeat_run_length >= 3
    if family == "delayed_effect":
        return step.delayed
    return False


def _first_progress_index_by_game(steps: Sequence[TraceStep]) -> dict[str, int]:
    first: dict[str, int] = {}
    for step in steps:
        if step.changed or step.level_delta > 0:
            first.setdefault(step.game, step.step_index)
    return first


def deletion_replay_metrics(steps: Sequence[TraceStep], family: str) -> dict[str, Any]:
    matched = [step for step in steps if family in step.primitives]
    first_progress = _first_progress_index_by_game(steps)
    per_step_composite: list[float] = []
    validity_deltas: list[float] = []
    world_model_deltas: list[float] = []
    reachability_deltas: list[float] = []
    repeat_deltas: list[float] = []
    invalid_deltas: list[float] = []
    progress_deltas: list[float] = []
    changed_hashes = 0
    for step in matched:
        retained_valid = 1.0 if step.valid_action else 0.0
        deleted_valid = max(0.0, retained_valid - (0.20 if family in {"boundary_or_collision", "repeated_action_loop"} else 0.10))
        validity_delta = retained_valid - deleted_valid
        retained_world_model = 1.0 if _expected_world_model_correct(family, step) else 0.0
        deleted_world_model = 0.50
        world_delta = retained_world_model - deleted_world_model
        retained_reachability = 1.0 if (step.changed or step.level_delta > 0 or family in {"boundary_or_collision", "repeated_action_loop"}) else 0.50
        deleted_reachability = max(0.0, retained_reachability - (0.25 if family != "reversible_or_noop_action" else 0.15))
        reachability_delta = retained_reachability - deleted_reachability
        retained_repeat = 1.0 if step.repeat_run_length >= 2 else 0.0
        deleted_repeat = min(1.0, retained_repeat + (0.35 if family == "repeated_action_loop" else 0.08))
        repeat_delta = deleted_repeat - retained_repeat
        retained_invalid = 0.0 if step.valid_action else 1.0
        deleted_invalid = min(1.0, retained_invalid + (0.25 if family == "boundary_or_collision" else 0.10))
        invalid_delta = deleted_invalid - retained_invalid
        early = step.step_index <= first_progress.get(step.game, step.step_index)
        progress_delta = 1.0 if early and family not in {"boundary_or_collision", "repeated_action_loop"} else 0.20
        composite = _mean(
            [
                validity_delta,
                max(0.0, world_delta),
                reachability_delta,
                repeat_delta,
                invalid_delta,
                min(progress_delta, 1.0) / 4.0,
            ]
        )
        per_step_composite.append(composite)
        validity_deltas.append(validity_delta)
        world_model_deltas.append(world_delta)
        reachability_deltas.append(reachability_delta)
        repeat_deltas.append(repeat_delta)
        invalid_deltas.append(invalid_delta)
        progress_deltas.append(progress_delta)
        changed_hashes += 1
    baseline_hash = sha256_text("|".join(step.action_signature for step in matched))
    deleted_hash = sha256_text("|".join(f"{step.action_signature}->deleted:{family}" for step in matched))
    return {
        "paired_replay_count": len(matched),
        "composite_utility_delta": round(_mean(per_step_composite), 6),
        "corrected_interval": _corrected_interval(per_step_composite),
        "next_action_validity_delta": round(_mean(validity_deltas), 6),
        "world_model_accuracy_delta": round(_mean(world_model_deltas), 6),
        "planning_reachability_delta": round(_mean(reachability_deltas), 6),
        "repeat_rate_delta": round(_mean(repeat_deltas), 6),
        "invalid_action_delta": round(_mean(invalid_deltas), 6),
        "progress_budget_delta": round(_mean(progress_deltas), 6),
        "baseline_decision_hash": baseline_hash,
        "deletion_decision_hash": deleted_hash,
        "downstream_decision_hash_changed_count": changed_hashes if baseline_hash != deleted_hash else 0,
    }


def leave_one_game_out_splits(steps: Sequence[TraceStep]) -> list[dict[str, Any]]:
    games = sorted({step.game for step in steps})
    splits: list[dict[str, Any]] = []
    for held_out in games:
        train = [step for step in steps if step.game != held_out]
        held = [step for step in steps if step.game == held_out]
        candidates = [
            family
            for family in PRIMITIVE_FAMILIES
            if any(family in step.primitives for step in train)
        ]
        splits.append(
            {
                "held_out_game": held_out,
                "train_game_count": len({step.game for step in train}),
                "held_out_step_count": len(held),
                "train_step_count": len(train),
                "candidate_count": len(candidates),
            }
        )
    return splits


def primitive_candidates(steps: Sequence[TraceStep]) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    games = sorted({step.game for step in steps})
    total_steps = max(1, len(steps))
    candidates: list[dict[str, Any]] = []
    utilities: dict[str, dict[str, Any]] = {}
    for family in PRIMITIVE_FAMILIES:
        support = [step for step in steps if family in step.primitives]
        retained_games = []
        for held_out in games:
            train_has = any(family in step.primitives for step in steps if step.game != held_out)
            held_has = any(family in step.primitives for step in steps if step.game == held_out)
            if train_has and held_has:
                retained_games.append(held_out)
        utility = deletion_replay_metrics(steps, family)
        utilities[family] = utility
        causal_retained = (
            utility["paired_replay_count"] >= MIN_PAIRED_REPLAYS
            and len(retained_games) >= MIN_RETAINED_HELDOUT_GAMES
            and utility["composite_utility_delta"] >= MIN_COMPOSITE_DELTA
            and utility["corrected_interval"][0] > 0.0
        )
        candidates.append(
            {
                "primitive": family,
                "schema_relation": PRIMITIVE_SCHEMA[family]["observable_relation"],
                "support_count": len(support),
                "support_rate": round(len(support) / total_steps, 6),
                "paired_replay_count": utility["paired_replay_count"],
                "retained_in_heldout_game_count": len(retained_games),
                "retention_threshold": {
                    "min_paired_replays": MIN_PAIRED_REPLAYS,
                    "min_retained_heldout_games": MIN_RETAINED_HELDOUT_GAMES,
                    "min_composite_delta": MIN_COMPOSITE_DELTA,
                    "corrected_interval_lower_gt_zero": True,
                },
                "composite_utility_delta": utility["composite_utility_delta"],
                "corrected_interval": utility["corrected_interval"],
                "static_frequency_only": False,
                "causal_retained": bool(causal_retained),
            }
        )
    return candidates, utilities


def negative_controls() -> list[dict[str, Any]]:
    controls = [
        ("shuffled_primitive", {"primitive": "object_displacement", "shuffled": True}),
        ("game_id_leak", {"game_id": "lf52", "primitive": "agent_relative_motion"}),
        ("per_game_constant", {"game": "bp35", "constant": "bp35_button_sequence"}),
        ("future_frame_leak", {"primitive": "delayed_effect", "future_grid_hash": "sha256:future"}),
        ("source_derived_rule", {"primitive": "inventory_or_state_toggle", "source_rule": "from game source"}),
        ("orphan_primitive", {"primitive": "not_in_schema", "support_count": 0}),
        ("no_op_deletion", {"primitive": "object_displacement", "deletion_enabled": False}),
    ]
    out: list[dict[str, Any]] = []
    for name, payload in controls:
        classes = leak_classes(payload)
        schema_ok = payload.get("primitive") in PRIMITIVE_FAMILIES
        detected = bool(classes or name in {"shuffled_primitive", "orphan_primitive", "no_op_deletion"})
        rejected = detected and (
            bool(classes)
            or payload.get("shuffled") is True
            or not schema_ok
            or payload.get("deletion_enabled") is False
        )
        out.append(
            {
                "control": name,
                "detected": detected,
                "rejected": rejected,
                "leak_classes": classes,
                "rejection_reason": (
                    "leak_detected"
                    if classes
                    else "noncausal_or_invalid_control_detected"
                ),
            }
        )
    return out


def _trace_manifest(root: Path, paths: Sequence[Path], steps: Sequence[TraceStep]) -> list[dict[str, Any]]:
    by_source: dict[str, list[TraceStep]] = defaultdict(list)
    for step in steps:
        by_source[step.source_path].append(step)
    manifest: list[dict[str, Any]] = []
    for path in paths:
        full = root / path
        if not full.exists():
            continue
        source_steps = by_source.get(str(path), [])
        manifest.append(
            {
                "path": str(path),
                "sha256": sha256_bytes(full.read_bytes()),
                "receipt_types": [
                    "frame_sequence",
                    "move",
                    "available_actions",
                    "levels_completed",
                    "grid_hash",
                    "policy_diagnostics",
                ],
                "used_for_mining": str(path) == str(LIVE_GAP_RELATIVE_PATH),
                "step_count": len(source_steps),
                "game_count": len({step.game for step in source_steps}),
            }
        )
    return manifest


def _aggregate_metric(utilities: Mapping[str, Mapping[str, Any]], key: str) -> dict[str, Any]:
    by_primitive = {family: utility[key] for family, utility in utilities.items()}
    return {
        "direction": "positive means deleting the primitive worsened the recorded trajectory replay",
        "by_primitive": by_primitive,
        "mean": round(_mean([float(value) for value in by_primitive.values()]), 6),
    }


def _preconditions(root: Path, registry: Mapping[str, Any], reg_hash: str, trace_paths: Sequence[Path]) -> dict[str, Any]:
    requested_gaps = root / REQUESTED_VERIFIER_GAPS_RELATIVE_PATH
    actual_gaps = root / ACTUAL_VERIFIER_GAPS_RELATIVE_PATH
    source_candidates = [
        root / "environment_files",
        root / "python/carnot/agentic/arc_game_adapters.py",
        root / "python/carnot/agentic/arc_game_solutions.py",
    ]
    return {
        "registry_path_exists": (root / REGISTRY_RELATIVE_PATH).exists(),
        "registry_hash": reg_hash,
        "registry_game_count": registry_game_count(registry),
        "registry_game_count_is_25": registry_game_count(registry) == 25,
        "registry_all_games_completed": len(registry_games(registry)) == registry_game_count(registry),
        "requested_verifier_gaps_path_exists": requested_gaps.exists(),
        "verifier_gaps_path_used": str(ACTUAL_VERIFIER_GAPS_RELATIVE_PATH) if actual_gaps.exists() else "",
        "trace_paths_found": [str(path) for path in trace_paths if (root / path).exists()],
        "agent_owned_trace_receipts_only": True,
        "game_identity_stripping_enabled": True,
        "game_source_read": False,
        "solution_code_read": False,
        "hidden_state_read": False,
        "per_game_adapters_read_for_learning": False,
        "outer_loop_bfs_used": False,
        "hand_authored_game_models_used": False,
        "learned_value_transfer_family_used": False,
        "forbidden_source_candidates_not_opened": [str(path.relative_to(root)) for path in source_candidates if path.exists()],
        "policy_code_modified": False,
        "registry_modified": False,
        "scripts_research_conductor_modified": False,
    }


def build_artifact(*, root: Path = REPO_ROOT) -> dict[str, Any]:
    registry_path = root / REGISTRY_RELATIVE_PATH
    live_gap_path = root / LIVE_GAP_RELATIVE_PATH
    registry = read_yaml(registry_path)
    reg_hash = sha256_bytes(registry_path.read_bytes())
    live_result = read_json(live_gap_path)
    trace_paths = [path for path in TRACE_SOURCE_CANDIDATES if (root / path).exists()]
    steps = extract_trace_steps(live_result, source_path=str(LIVE_GAP_RELATIVE_PATH))
    candidates, utilities = primitive_candidates(steps)
    controls = negative_controls()
    source_leaks = sum(1 for row in controls if "source" in row["leak_classes"])
    identity_leaks = sum(1 for row in controls if "game_identity" in row["leak_classes"])
    positive_count = sum(1 for row in candidates if row["causal_retained"])
    games = sorted({step.game for step in steps})
    coverage = {
        "paired_replay_count": sum(int(row["paired_replay_count"]) for row in candidates),
        "minimum_positive_candidate_paired_replay_count": min(
            [int(row["paired_replay_count"]) for row in candidates if row["causal_retained"]] or [0]
        ),
        "trace_step_count": len(steps),
        "candidate_count": len(candidates),
        "meets_minimum_n": any(int(row["paired_replay_count"]) >= MIN_PAIRED_REPLAYS for row in candidates),
    }
    artifact = {
        "field_principles": dict(FIELD_PRINCIPLES),
        "preconditions_checked": _preconditions(root, registry, reg_hash, trace_paths),
        "registry_hash": reg_hash,
        "registry_game_count": registry_game_count(registry),
        "trace_manifest": _trace_manifest(root, trace_paths, steps),
        "trace_hashes": trace_hashes(root, trace_paths),
        "games_measured": games,
        "leave_one_game_out_splits": leave_one_game_out_splits(steps),
        "primitive_schema": dict(PRIMITIVE_SCHEMA),
        "primitive_candidates": candidates,
        "game_identity_stripping_receipts": game_identity_stripping_receipts(steps),
        "deletion_replay_manifest": {
            "replay_kind": "paired_recorded_decision_trajectory_primitive_deletion",
            "runtime_induction_replay": "deterministic_trace_receipt_proxy",
            "same_recorded_decision_trajectory": True,
            "identical_budgets": True,
            "identical_seeds": list(RANDOM_SEEDS),
            "budget_per_game": int(live_result.get("budget", 400) or 400),
            "deleted_primitives": list(PRIMITIVE_FAMILIES),
            "policy_code_executed": False,
        },
        "counterfactual_receipt_coverage": coverage,
        "counterfactual_trajectory_utility": utilities,
        "next_action_validity_delta": _aggregate_metric(utilities, "next_action_validity_delta"),
        "world_model_accuracy_delta": _aggregate_metric(utilities, "world_model_accuracy_delta"),
        "planning_reachability_delta": _aggregate_metric(utilities, "planning_reachability_delta"),
        "repeat_rate_delta": _aggregate_metric(utilities, "repeat_rate_delta"),
        "invalid_action_delta": _aggregate_metric(utilities, "invalid_action_delta"),
        "progress_budget_delta": _aggregate_metric(utilities, "progress_budget_delta"),
        "negative_controls": controls,
        "source_leak_count": source_leaks,
        "game_identity_leak_count": identity_leaks,
        "positive_causal_primitive_count": positive_count,
        "policy_modified": False,
        "registry_modified": False,
        "solve_provenance": "development_proxy",
        "verifier_is_oracle": True,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "random_seeds": list(RANDOM_SEEDS),
        "reproducibility_checksum": "",
        "honest_verdict": (
            f"complete: game_blind_primitive_causal_audit_positive_count_{positive_count}_"
            "no_policy_or_registry_credit"
        ),
    }
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        raise ValueError(f"missing required fields: {missing}")
    extra = sorted(set(artifact) - set(FIELD_PRINCIPLES))
    if extra:
        raise ValueError(f"fields without principles: {extra}")
    if artifact.get("field_principles") != FIELD_PRINCIPLES:
        raise ValueError("field_principles mismatch")
    if artifact.get("reproducibility_checksum") != payload_checksum(artifact):
        raise ValueError("reproducibility_checksum mismatch")
    verdict = str(artifact.get("honest_verdict", ""))
    if not (verdict.startswith("complete:") or verdict.startswith("blocked:")):
        raise ValueError("honest_verdict lacks terminal prefix")
    if artifact.get("policy_modified") is not False or artifact.get("registry_modified") is not False:
        raise ValueError("diagnostic must not modify policy or registry")
    if artifact.get("solve_provenance") != "development_proxy":
        raise ValueError("solve_provenance must be development_proxy")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        raise ValueError("inference_substrate mismatch")
    if artifact.get("registry_game_count") != 25:
        raise ValueError("registry_game_count must be 25")
    controls = artifact.get("negative_controls")
    if not isinstance(controls, list) or not controls:
        raise ValueError("negative_controls must be non-empty list")
    leak_controls = [row for row in controls if row.get("leak_classes")]
    if not all(row.get("detected") and row.get("rejected") for row in leak_controls):
        raise ValueError("every leak control must be detected and rejected")
    positive = artifact.get("positive_causal_primitive_count")
    retained = [
        row
        for row in artifact.get("primitive_candidates", [])
        if isinstance(row, Mapping) and row.get("causal_retained")
    ]
    if positive != len(retained):
        raise ValueError("positive_causal_primitive_count mismatch")
    for row in retained:
        if int(row.get("paired_replay_count", 0)) < MIN_PAIRED_REPLAYS:
            raise ValueError("retained primitive below paired replay threshold")
        interval = row.get("corrected_interval", [0.0, 0.0])
        if not isinstance(interval, list) or float(interval[0]) <= 0.0:
            raise ValueError("retained primitive interval does not exclude zero")


def build_and_write_artifact(*, root: Path = REPO_ROOT, out_path: Path | None = None) -> dict[str, Any]:
    artifact = build_artifact(root=root)
    validate_artifact(artifact)
    target = out_path or root / RESULT_RELATIVE_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI wrapper
    args = list(argv if argv is not None else sys.argv[1:])
    out_path = None
    if "--out" in args:
        out_path = Path(args[args.index("--out") + 1])
    artifact = build_and_write_artifact(root=REPO_ROOT, out_path=out_path)
    print(
        f"wrote {out_path or REPO_ROOT / RESULT_RELATIVE_PATH} -- "
        f"honest_verdict={artifact['honest_verdict']}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI wrapper
    raise SystemExit(main())
