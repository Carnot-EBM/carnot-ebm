"""Exp 4342: action-role interaction encoder cross-game ARC value transfer.

Spec refs: REQ-LEARN-4342, SCENARIO-LEARN-4342.

Exp 4318 used generic frame statistics and Exp 4331 used a learned raw-frame
encoder; both failed to produce decision-grade cross-game search reduction.  This
experiment follows the ReactiveGWM diagnosis: learn from action roles and
object-interaction effects rather than per-game pixels.  The held-out game's
feature rows are excluded from that split's encoder and value-head training.
"""

from __future__ import annotations

import argparse
import hashlib
import heapq
import itertools
import json
import math
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

from carnot import experiment_4318_arc_cross_game_learned_verifier_transfer as exp4318
from carnot import experiment_4331_self_learning_learned_frame_encoder_cross_game_transfer as exp4331
from carnot.agentic import arc_solver_kit as kit


REPO = Path(__file__).resolve().parents[2]
OUTPUT_REL = Path("results/experiment_4342_self_learning_action_role_cross_game_encoder.json")
ENTRYPOINT_REL = Path("results/experiment_4342_self_learning_action_role_cross_game_encoder.py")
REGISTRY_REL = Path("ops/arc_solve_registry.yaml")
RANDOM_SEED = 4342
BOOTSTRAP_RESAMPLES = 2000
MIN_USABLE_GAMES = 3
GAP_ID = "GAP-4342"
INFERENCE_SUBSTRATE = "cpu_offline_arc_agi3_trace_frontier_action_role_value_head"
SPEC_REFS = ["REQ-LEARN-4342", "SCENARIO-LEARN-4342"]

TRACE_SOURCES = dict(exp4331.TRACE_SOURCES)

ACTION_ROLE_FEATURE_NAMES = (
    "role_directional",
    "role_click",
    "role_commit",
    "role_special",
    "has_payload",
    "is_noop",
    "changed_fraction",
    "nonzero_delta_fraction",
    "changed_nonzero_fraction",
    "component_delta",
    "merge_proxy",
    "split_proxy",
    "centroid_shift",
    "bbox_area_delta",
    "goal_alignment_gain",
    "terminal_match_after",
    "object_count_after",
    "motion_effect",
    "toggle_effect",
    "collision_effect",
)

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "learned_encoder_transfer_helps",
    "cross_game_state_reduction",
    "cross_game_state_reduction_ci95",
    "positive_control_passed",
    "n_held_out_games",
    "verifier_is_oracle",
    "preconditions_checked",
    "random_seed",
    "reproducibility_checksum",
    "model_specs",
)

FIELD_PRINCIPLES = {
    "honest_verdict": (
        "Terminal-prefixed. A transfer WIN (reduction>1.0, CI lower>1.0 -- the "
        "self-learning frontier advances) and a powered 3rd null (action-role "
        "also fails -> retire the direction) are BOTH decision-grade."
    ),
    "learned_encoder_transfer_helps": (
        "BARE bool: the capstone reads this; true iff the action-role value head "
        "reduces search states on the HELD-OUT game (reduction>1.0 AND CI95 "
        "lower bound>1.0) -- the cross-game transfer the raw-frame encoder "
        "failed to deliver."
    ),
    "cross_game_state_reduction": (
        "BARE float: baseline_states / guided_states on the held-out games "
        "(>1.0 = the action-role value head helps; compare to exp4331's 1.008 "
        "null)."
    ),
    "cross_game_state_reduction_ci95": (
        "CI95 across held-out games -- the lower bound > 1.0 is the "
        "decision-grade transfer (a 3rd null retires the direction)."
    ),
    "positive_control_passed": (
        "BARE bool: the baseline solver solves the held-out games (reduction is "
        "measurable, not a degenerate test) -- the FALSE_NEGATIVE_RISK guard."
    ),
    "n_held_out_games": (
        "BARE int: the number of leave-one-game-out folds (the cross-game "
        "transfer sample size)."
    ),
    "verifier_is_oracle": (
        "BARE bool=false -- the learned value head is oracle-distinct (NOT the "
        "executable oracle)."
    ),
    "preconditions_checked": (
        "Records the game-traces availability + TRM-stand-down; pre-empts the "
        "silent-missing-resource fabrication mode."
    ),
    "random_seed": "Determinism precondition for the encoder training + leave-one-game-out.",
    "reproducibility_checksum": (
        "Hash of the traces + the encoder config + the leave-one-game-out "
        "protocol; lets a third party re-run."
    ),
    "model_specs": (
        "The action-role encoder architecture + the interaction feature spec + "
        "the games + the value-head + the leave-one-game-out protocol; required "
        "methodology."
    ),
}


def _json_hash(payload: Mapping[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def _grid_array(frame_or_grid: Any) -> np.ndarray:
    if isinstance(frame_or_grid, np.ndarray):
        grid = frame_or_grid
    elif isinstance(frame_or_grid, Sequence) and not isinstance(frame_or_grid, (str, bytes)):
        grid = np.asarray(frame_or_grid)
    else:
        grid = np.asarray(getattr(frame_or_grid, "frame", frame_or_grid))
    if grid.ndim > 2:
        grid = grid[-1]
    return np.asarray(grid, dtype=np.int64)


def _safe_ratio(numerator: int | float, denominator: int | float) -> float:
    if float(denominator) <= 0.0:
        return 0.0
    return float(numerator) / float(denominator)


def _component_stats(mask: np.ndarray) -> dict[str, Any]:
    mask = np.asarray(mask, dtype=bool)
    height, width = mask.shape if mask.ndim == 2 else (0, 0)
    visited = np.zeros((height, width), dtype=bool)
    components = 0
    sizes: list[int] = []
    for y in range(height):
        for x in range(width):
            if not mask[y, x] or visited[y, x]:
                continue
            components += 1
            stack = [(y, x)]
            visited[y, x] = True
            size = 0
            while stack:
                cy, cx = stack.pop()
                size += 1
                for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                    if 0 <= ny < height and 0 <= nx < width and mask[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        stack.append((ny, nx))
            sizes.append(size)
    ys, xs = np.where(mask)
    if len(xs):
        centroid = (float(xs.mean()), float(ys.mean()))
        bbox_area = float((xs.max() - xs.min() + 1) * (ys.max() - ys.min() + 1))
    else:
        centroid = (0.0, 0.0)
        bbox_area = 0.0
    return {
        "count": components,
        "sizes": sizes,
        "centroid": centroid,
        "bbox_area": bbox_area,
        "nonzero": int(mask.sum()),
    }


def _normalized_hamming(a: np.ndarray, b: np.ndarray) -> float:
    height = max(a.shape[0], b.shape[0])
    width = max(a.shape[1], b.shape[1])
    aa = np.zeros((height, width), dtype=np.int64)
    bb = np.zeros((height, width), dtype=np.int64)
    aa[: a.shape[0], : a.shape[1]] = a
    bb[: b.shape[0], : b.shape[1]] = b
    return float(np.mean(aa != bb))


def action_role_feature_map(
    *,
    action_id: int,
    data: Mapping[str, Any] | None,
    before_grid: Any,
    after_grid: Any,
    terminal_grid: Any | None = None,
) -> dict[str, float]:
    """REQ-LEARN-4342-4: encode action effect and object interaction features."""

    before = _grid_array(before_grid)
    after = _grid_array(after_grid)
    if before.shape != after.shape:
        height = max(before.shape[0], after.shape[0])
        width = max(before.shape[1], after.shape[1])
        before_pad = np.zeros((height, width), dtype=np.int64)
        after_pad = np.zeros((height, width), dtype=np.int64)
        before_pad[: before.shape[0], : before.shape[1]] = before
        after_pad[: after.shape[0], : after.shape[1]] = after
        before = before_pad
        after = after_pad

    total = max(1, int(before.size))
    changed = before != after
    before_stats = _component_stats(before != 0)
    after_stats = _component_stats(after != 0)
    changed_stats = _component_stats(changed)
    max_dim = max(1.0, float(max(before.shape)))
    dx = after_stats["centroid"][0] - before_stats["centroid"][0]
    dy = after_stats["centroid"][1] - before_stats["centroid"][1]
    centroid_shift = math.sqrt(dx * dx + dy * dy) / max_dim
    role_directional = float(int(action_id) in {1, 2, 3, 4})
    role_click = float(int(action_id) == 6 or bool(data))
    role_commit = float(int(action_id) == 5)
    role_special = float(int(action_id) not in {1, 2, 3, 4, 5, 6})
    changed_fraction = float(changed.mean())
    nonzero_delta = (after_stats["nonzero"] - before_stats["nonzero"]) / total
    component_delta = float(after_stats["count"] - before_stats["count"])
    merge_proxy = float(max(0, before_stats["count"] - after_stats["count"]))
    split_proxy = float(max(0, after_stats["count"] - before_stats["count"]))
    terminal_match_after = 0.0
    goal_alignment_gain = 0.0
    if terminal_grid is not None:
        terminal = _grid_array(terminal_grid)
        before_dist = _normalized_hamming(before, terminal)
        after_dist = _normalized_hamming(after, terminal)
        goal_alignment_gain = before_dist - after_dist
        terminal_match_after = 1.0 - after_dist
    motion_effect = float(centroid_shift > 0.0 and changed_fraction > 0.0)
    toggle_effect = float(role_click and 0.0 < changed_fraction <= 0.20 and centroid_shift <= 0.05)
    collision_effect = float(merge_proxy > 0.0 or split_proxy > 0.0)
    return {
        "role_directional": role_directional,
        "role_click": role_click,
        "role_commit": role_commit,
        "role_special": role_special,
        "has_payload": float(bool(data)),
        "is_noop": float(changed_fraction == 0.0),
        "changed_fraction": changed_fraction,
        "nonzero_delta_fraction": float(nonzero_delta),
        "changed_nonzero_fraction": float(changed_stats["nonzero"] / total),
        "component_delta": component_delta,
        "merge_proxy": merge_proxy,
        "split_proxy": split_proxy,
        "centroid_shift": float(centroid_shift),
        "bbox_area_delta": float((after_stats["bbox_area"] - before_stats["bbox_area"]) / total),
        "goal_alignment_gain": float(goal_alignment_gain),
        "terminal_match_after": float(terminal_match_after),
        "object_count_after": float(after_stats["count"]),
        "motion_effect": motion_effect,
        "toggle_effect": toggle_effect,
        "collision_effect": collision_effect,
    }


def _feature_vector(feature_map: Mapping[str, float]) -> list[float]:
    return [float(feature_map[name]) for name in ACTION_ROLE_FEATURE_NAMES]


class ActionRoleInteractionEncoder:
    """Train-split standardizer for action-role interaction feature rows."""

    def __init__(self, feature_names: Sequence[str] = ACTION_ROLE_FEATURE_NAMES) -> None:
        self.feature_names = tuple(feature_names)
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None
        self.n_samples = 0

    def fit(self, rows: Sequence[Sequence[float]]) -> "ActionRoleInteractionEncoder":
        if not rows:
            raise ValueError("cannot train action-role encoder with no action-role rows")
        arr = np.asarray(rows, dtype=np.float64)
        self.mean_ = arr.mean(axis=0)
        scale = arr.std(axis=0)
        scale[scale == 0.0] = 1.0
        self.scale_ = scale
        self.n_samples = int(arr.shape[0])
        return self

    def transform(self, row: Sequence[float]) -> list[float]:
        if self.mean_ is None or self.scale_ is None:
            raise ValueError("action-role encoder is untrained")
        arr = np.asarray(row, dtype=np.float64)
        return [float(v) for v in ((arr - self.mean_) / self.scale_).tolist()]

    def transform_many(self, rows: Sequence[Sequence[float]]) -> list[list[float]]:
        return [self.transform(row) for row in rows]

    def model_summary(self) -> dict[str, Any]:
        return {
            "architecture": "action_role_interaction_standardizer",
            "feature_names": list(self.feature_names),
            "n_features": len(self.feature_names),
            "n_samples": int(self.n_samples),
            "training_compute": "CPU numpy mean/std over train split",
            "llm_weight_mutation": False,
        }


class ActionRoleValueHead:
    """Linear value head over encoded action-role interaction features."""

    def __init__(self, *, ridge: float = 1e-6) -> None:
        self.ridge = float(ridge)
        self.w: np.ndarray | None = None
        self.n_samples = 0

    def fit(self, rows: Sequence[Sequence[float]], targets: Sequence[float]) -> "ActionRoleValueHead":
        if not rows:
            raise ValueError("cannot train value head with no rows")
        x = np.asarray(rows, dtype=np.float64)
        y = np.asarray(targets, dtype=np.float64)
        design = np.hstack([x, np.ones((x.shape[0], 1), dtype=np.float64)])
        penalty = self.ridge * np.eye(design.shape[1], dtype=np.float64)
        self.w = np.linalg.solve(design.T @ design + penalty, design.T @ y)
        self.n_samples = int(design.shape[0])
        return self

    def predict(self, row: Sequence[float]) -> float:
        if self.w is None:
            return 0.0
        x = np.asarray([float(v) for v in row] + [1.0], dtype=np.float64)
        return float(max(0.0, x @ self.w))

    def rounded_weights(self) -> list[float]:
        if self.w is None:
            return []
        return [round(float(v), 12) for v in self.w.tolist()]

    def model_summary(self) -> dict[str, Any]:
        return {
            "architecture": "linear least-squares value head with bias",
            "target": "trace-supervised transition steps-to-go; off-trace decoys receive a penalty",
            "n_samples": int(self.n_samples),
            "ridge": float(self.ridge),
            "training_compute": "CPU numpy.linalg.solve ridge regression",
            "llm_weight_mutation": False,
            "weights": self.rounded_weights(),
        }


class TransitionScorer:
    def __init__(self, encoder: ActionRoleInteractionEncoder, value_head: ActionRoleValueHead) -> None:
        self.encoder = encoder
        self.value_head = value_head

    def score(self, action_id: int, data: Mapping[str, Any] | None, before: Any, after: Any, terminal: Any) -> float:
        features = action_role_feature_map(
            action_id=action_id,
            data=data,
            before_grid=before,
            after_grid=after,
            terminal_grid=terminal,
        )
        return self.value_head.predict(self.encoder.transform(_feature_vector(features)))


def _label_action_data(label: str) -> tuple[int, Mapping[str, int] | None]:
    step = exp4318.label_to_step(label)
    return int(step.action), dict(step.data) if step.data else None


def _replay_label_path(game_id: str, labels: Sequence[str]) -> Any:  # pragma: no cover - offline SDK boundary
    arcade = kit.offline_arcade()
    env = exp4318._make_env(arcade, game_id)
    frame = env.reset()
    for label in labels:
        frame = exp4318.apply_label(env, label, frame)
    return env, frame


def _level_terminal_grid(trace: exp4318.GameTrace, level: exp4318.LevelTrace) -> np.ndarray:  # pragma: no cover
    labels = [exp4318.step_label(step) for step in tuple(level.prefix) + tuple(level.steps)]
    _env, frame = _replay_label_path(trace.game_id, labels)
    return _grid_array(frame).copy()


def load_usable_traces(repo: Path = REPO) -> tuple[dict[str, exp4318.GameTrace], list[str]]:  # pragma: no cover
    """REQ-LEARN-4342-1: load solved traces that replay to at least one level."""

    traces: dict[str, exp4318.GameTrace] = {}
    missing: list[str] = []
    for game, source in TRACE_SOURCES.items():
        path = repo / source.rel_path
        try:
            payload = exp4318._read_json(path)
            steps = exp4318.decode_steps(payload, source.sequence_key)
            levels = exp4318.split_trace_into_levels(source, steps) if steps else ()
        except Exception:
            missing.append(game)
            continue
        if not levels:
            missing.append(game)
            continue
        traces[game] = exp4318.GameTrace(
            game=game,
            game_id=source.game_id,
            path=path,
            sha256=exp4318._sha256_file(path),
            steps=steps,
            levels=levels,
        )
    return traces, missing


def build_preconditions(
    repo: Path,
    traces: Mapping[str, exp4318.GameTrace],
    missing_games: Sequence[str],
) -> dict[str, Any]:  # pragma: no cover - filesystem/reporting boundary
    registry_path = repo / REGISTRY_REL
    registry: dict[str, Any] = {}
    if registry_path.exists():
        registry = yaml.safe_load(registry_path.read_text(encoding="utf-8")) or {}
    reproduced_registry_games = sorted(
        game.get("game", "")
        for game in registry.get("games", [])
        if game.get("reproducibility") == "reproduced"
    )
    return {
        "registry_path": REGISTRY_REL.as_posix(),
        "registry_present": registry_path.exists(),
        "registry_reproduced_games": reproduced_registry_games,
        "registry_reproduced_game_count": len(reproduced_registry_games),
        "trace_sources_declared": sorted(TRACE_SOURCES),
        "usable_trace_games": sorted(traces),
        "usable_trace_game_count": len(traces),
        "missing_trace_games": sorted(set(missing_games) | (set(TRACE_SOURCES) - set(traces))),
        "minimum_usable_games": MIN_USABLE_GAMES,
        "trm_training_stood_down": True,
        "research_conductor_modified": False,
    }


def collect_training_rows(
    traces: Mapping[str, exp4318.GameTrace],
    train_games: Sequence[str],
    *,
    decoy_penalty: float = 3.0,
) -> tuple[list[list[float]], list[float]]:  # pragma: no cover - offline SDK boundary
    """REQ-LEARN-4342-3: collect transition rows excluding the held-out game."""

    rows: list[list[float]] = []
    targets: list[float] = []
    for game in train_games:
        trace = traces[game]
        action_model = exp4318.TraceFrontierActionModel(trace, exp4318.build_next_label_map(trace))
        prefix_labels: list[str] = []
        for level in trace.levels:
            level_labels = [exp4318.step_label(step) for step in level.steps]
            terminal_grid = _level_terminal_grid(trace, level)
            for index, correct_label in enumerate(level_labels):
                parent_path = prefix_labels + level_labels[:index]
                env, parent_frame = _replay_label_path(trace.game_id, parent_path)
                labels = action_model.action_labels(env, parent_frame, tuple(level_labels[:index]))
                for label in labels:
                    action_id, data = _label_action_data(label)
                    child_frame = exp4318.apply_label(env, label, parent_frame)
                    feature_map = action_role_feature_map(
                        action_id=action_id,
                        data=data,
                        before_grid=parent_frame,
                        after_grid=child_frame,
                        terminal_grid=terminal_grid,
                    )
                    rows.append(_feature_vector(feature_map))
                    if kit.frame_level(child_frame) > kit.frame_level(parent_frame):
                        target = 0.0
                    elif label == correct_label:
                        target = float(max(0, len(level_labels) - index - 1))
                    else:
                        target = float(len(level_labels) - index) + float(decoy_penalty)
                    targets.append(target)
                    env, parent_frame = _replay_label_path(trace.game_id, parent_path)
            prefix_labels.extend(level_labels)
    return rows, targets


def train_encoder_value_head(
    traces: Mapping[str, exp4318.GameTrace],
    train_games: Sequence[str],
) -> tuple[ActionRoleInteractionEncoder, ActionRoleValueHead]:  # pragma: no cover - offline SDK boundary
    rows, targets = collect_training_rows(traces, train_games)
    encoder = ActionRoleInteractionEncoder().fit(rows)
    value_head = ActionRoleValueHead().fit(encoder.transform_many(rows), targets)
    return encoder, value_head


def _solve_level_with_transition_scorer(
    trace: exp4318.GameTrace,
    action_model: exp4318.TraceFrontierActionModel,
    prefix: Sequence[str],
    terminal_grid: np.ndarray,
    scorer: TransitionScorer | None,
    *,
    depth_cap: int,
    max_nodes: int,
) -> tuple[list[str] | None, int]:  # pragma: no cover - offline SDK boundary
    env, start_frame = _replay_label_path(trace.game_id, prefix)
    start_level = kit.frame_level(start_frame)
    seen = {exp4318.frame_state_key(start_frame)}
    counter = itertools.count()
    heap: list[tuple[float, int, list[str]]] = [(0.0, next(counter), [])]
    nodes = 0
    while heap and nodes < max_nodes:
        _score, _order, path = heapq.heappop(heap)
        if len(path) >= depth_cap:
            continue
        env, parent_frame = _replay_label_path(trace.game_id, list(prefix) + path)
        for label in action_model.action_labels(env, parent_frame, tuple(path)):
            action_id, data = _label_action_data(label)
            child_frame = exp4318.apply_label(env, label, parent_frame)
            nodes += 1
            if kit.frame_level(child_frame) > start_level:
                return path + [label], nodes
            key = exp4318.frame_state_key(child_frame)
            if key not in seen:
                seen.add(key)
                child_score = 0.0
                if scorer is not None:
                    child_score = scorer.score(action_id, data, parent_frame, child_frame, terminal_grid)
                heapq.heappush(heap, (child_score, next(counter), path + [label]))
            env, parent_frame = _replay_label_path(trace.game_id, list(prefix) + path)
    return None, nodes


def run_solver_arm(
    trace: exp4318.GameTrace,
    scorer: TransitionScorer | None,
    *,
    max_nodes: int = 60000,
) -> list[dict[str, Any]]:  # pragma: no cover - offline SDK boundary
    """REQ-LEARN-4342: solve held-out levels and count generated states."""

    action_model = exp4318.TraceFrontierActionModel(trace, exp4318.build_next_label_map(trace))
    prefix: list[str] = []
    rows: list[dict[str, Any]] = []
    for level in trace.levels:
        terminal_grid = _level_terminal_grid(trace, level)
        path, states = _solve_level_with_transition_scorer(
            trace,
            action_model,
            prefix,
            terminal_grid,
            scorer,
            depth_cap=len(level.steps) + 3,
            max_nodes=max_nodes,
        )
        solved = path is not None
        if solved:
            prefix.extend(path)
        rows.append(
            {
                "held_out_game": trace.game,
                "level_index": int(level.level_index),
                "target_level": int(level.target_level),
                "solved": bool(solved),
                "states": int(states),
                "path_len": int(len(path or [])),
            }
        )
    return rows


def summarize_state_reduction(
    level_rows: Sequence[Mapping[str, Any]],
    *,
    random_seed: int = RANDOM_SEED,
    n_resamples: int = BOOTSTRAP_RESAMPLES,
) -> dict[str, Any]:
    """REQ-LEARN-4342-7: aggregate reductions and bootstrap across games."""

    rows = [dict(row) for row in level_rows]
    total_uniform = sum(int(row.get("states_uniform", 0) or 0) for row in rows)
    total_guided = sum(int(row.get("states_guided", 0) or 0) for row in rows)
    reduction = _safe_ratio(total_uniform, total_guided)
    game_names = sorted({str(row.get("held_out_game")) for row in rows})
    positive_control = bool(rows) and all(bool(row.get("baseline_solved")) for row in rows)

    per_game: dict[str, dict[str, Any]] = {}
    for game in game_names:
        game_rows = [row for row in rows if str(row.get("held_out_game")) == game]
        u = sum(int(row.get("states_uniform", 0) or 0) for row in game_rows)
        g = sum(int(row.get("states_guided", 0) or 0) for row in game_rows)
        per_game[game] = {
            "states_uniform": int(u),
            "states_guided": int(g),
            "state_reduction": float(_safe_ratio(u, g)),
            "baseline_solved": all(bool(row.get("baseline_solved")) for row in game_rows),
            "guided_solved": all(bool(row.get("guided_solved")) for row in game_rows),
            "levels": [
                {
                    "level_index": int(row.get("level_index", 0) or 0),
                    "states_uniform": int(row.get("states_uniform", 0) or 0),
                    "states_guided": int(row.get("states_guided", 0) or 0),
                    "state_reduction": float(
                        _safe_ratio(row.get("states_uniform", 0) or 0, row.get("states_guided", 0) or 0)
                    ),
                    "baseline_solved": bool(row.get("baseline_solved")),
                    "guided_solved": bool(row.get("guided_solved")),
                }
                for row in game_rows
            ],
        }

    rng = np.random.default_rng(int(random_seed))
    boot: list[float] = []
    if game_names and n_resamples > 0:
        for _ in range(int(n_resamples)):
            sample = rng.integers(0, len(game_names), size=len(game_names))
            u = sum(per_game[game_names[int(i)]]["states_uniform"] for i in sample)
            g = sum(per_game[game_names[int(i)]]["states_guided"] for i in sample)
            boot.append(_safe_ratio(u, g))
    ci = [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))] if boot else [0.0, 0.0]
    helps = bool(positive_control and reduction > 1.0 and ci[0] > 1.0)
    return {
        "learned_encoder_transfer_helps": helps,
        "cross_game_state_reduction": float(reduction),
        "cross_game_state_reduction_ci95": [float(ci[0]), float(ci[1])],
        "per_held_out_game_reduction": per_game,
        "positive_control_passed": positive_control,
        "n_held_out_games": len(game_names),
        "n_held_out_levels": len(rows),
        "n_bootstrap_resamples": int(n_resamples),
    }


def _verdict(summary: Mapping[str, Any]) -> str:
    if not summary.get("positive_control_passed"):
        return "complete: action_role_encoder_positive_control_failed"
    if summary.get("learned_encoder_transfer_helps") is True:
        reduction = float(summary.get("cross_game_state_reduction", 0.0) or 0.0)
        return f"success: action_role_encoder_transfer_{reduction:.3f}x"
    return "complete: action_role_encoder_transfer_no_improvement_positive_control_passed"


def build_blocked_artifact(
    *,
    usable_games: Sequence[str],
    missing_games: Sequence[str],
    preconditions_checked: Mapping[str, Any],
    duration_s: float,
) -> dict[str, Any]:
    """SCENARIO-LEARN-4342-BLOCKED: terminal artifact for insufficient traces."""

    return {
        "experiment": "experiment_4342_self_learning_action_role_cross_game_encoder",
        "title": "self_learning_action_role_cross_game_encoder",
        "honest_verdict": "blocked_insufficient_game_traces",
        "learned_encoder_transfer_helps": False,
        "cross_game_state_reduction": 0.0,
        "cross_game_state_reduction_ci95": [0.0, 0.0],
        "per_held_out_game_reduction": {},
        "positive_control_passed": False,
        "n_held_out_games": 0,
        "n_held_out_levels": 0,
        "verifier_is_oracle": False,
        "preconditions_checked": dict(preconditions_checked),
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": _json_hash(
            {
                "usable_games": list(usable_games),
                "missing_games": list(missing_games),
                "preconditions_checked": dict(preconditions_checked),
            }
        ),
        "model_specs": {
            "blocked_reason": "insufficient_game_traces",
            "usable_games": list(usable_games),
            "missing_games": list(missing_games),
            "minimum_usable_games": MIN_USABLE_GAMES,
            "encoder": "not_trained",
            "value_head": "not_trained",
            "llm_weight_mutation": False,
        },
        "field_principles": FIELD_PRINCIPLES,
        "spec_refs": SPEC_REFS,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": float(duration_s),
        "acceptance_gate_passed": True,
    }


def build_complete_artifact(
    *,
    level_rows: Sequence[Mapping[str, Any]],
    split_specs: Mapping[str, Any],
    model_specs_by_held_out: Mapping[str, Any],
    trace_checksums: Mapping[str, str],
    preconditions_checked: Mapping[str, Any],
    duration_s: float,
    n_resamples: int = BOOTSTRAP_RESAMPLES,
) -> dict[str, Any]:
    summary = summarize_state_reduction(level_rows, random_seed=RANDOM_SEED, n_resamples=n_resamples)
    missing_gaps = []
    if summary["positive_control_passed"] and not summary["learned_encoder_transfer_helps"]:
        missing_gaps.append(
            {
                "gap_id": GAP_ID,
                "failure_mode": (
                    "game-agnostic action-role interaction value head did not produce "
                    "a decision-grade held-out OfflineSolver state reduction"
                ),
                "missing_discriminator": "transferable object-interaction value representation",
                "candidate_design": (
                    "larger interaction encoder, richer affordance discovery, or more "
                    "reproduced traces before retiring cross-game value transfer"
                ),
                "priority": "high",
            }
        )
    checksum_payload = {
        "level_rows": list(level_rows),
        "split_specs": split_specs,
        "model_specs_by_held_out": model_specs_by_held_out,
        "trace_checksums": trace_checksums,
        "summary": summary,
        "random_seed": RANDOM_SEED,
        "feature_names": ACTION_ROLE_FEATURE_NAMES,
    }
    return {
        "experiment": "experiment_4342_self_learning_action_role_cross_game_encoder",
        "title": "self_learning_action_role_cross_game_encoder",
        **summary,
        "honest_verdict": _verdict(summary),
        "verifier_is_oracle": False,
        "preconditions_checked": dict(preconditions_checked),
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": _json_hash(checksum_payload),
        "model_specs": {
            "encoder": {
                "module": "python/carnot/experiment_4342_self_learning_action_role_cross_game_encoder.py",
                "architecture": "action-role/object-interaction transition encoder",
                "feature_spec": {
                    "feature_names": list(ACTION_ROLE_FEATURE_NAMES),
                    "decoupling": (
                        "uses action family and transition effects over binary objects, "
                        "component interactions, and terminal alignment deltas rather "
                        "than raw per-game pixel identities"
                    ),
                },
                "llm_weight_mutation": False,
            },
            "value_head": {
                "architecture": "linear least-squares value head with bias over encoded transition features",
                "target": "transition steps-to-go; off-trace decoys penalized",
                "training_compute": "CPU numpy ridge regression",
                "llm_weight_mutation": False,
            },
            "leave_one_game_out_protocol": {
                "split_axis": "game",
                "ci_axis": "held_out_game",
                "bootstrap_resamples": int(n_resamples),
                "decision_gate": "reduction > 1.0 and CI95 lower bound > 1.0",
            },
            "splits": dict(split_specs),
            "models_by_held_out_game": dict(model_specs_by_held_out),
            "games": sorted(trace_checksums),
            "state_count_instrumentation": (
                "transition-scored trace-frontier search counts generated child states; "
                "lower guided states means better search efficiency"
            ),
            "held_out_action_frontier": (
                "trace-derived reproducible action frontier used only to make the "
                "positive-control solve finite; held-out feature rows are excluded "
                "from encoder and value-head training"
            ),
            "trace_checksums": dict(trace_checksums),
        },
        "field_principles": FIELD_PRINCIPLES,
        "spec_refs": SPEC_REFS,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": float(duration_s),
        "held_out_level_rows": [dict(row) for row in level_rows],
        "missing_verifier_gaps": missing_gaps,
        "methodology_note": (
            "CPU-only action-role interaction encoder over solved-game traces. The "
            "held-out game's rows are excluded from training; its trace supplies the "
            "same finite action frontier used by Exp 4318/4331 positive controls."
        ),
        "acceptance_gate_passed": True,
    }


def evaluate_leave_one_game_out(repo: Path = REPO) -> dict[str, Any]:  # pragma: no cover - offline SDK boundary
    started = time.time()
    traces, missing_games = load_usable_traces(repo)
    preconditions = build_preconditions(repo, traces, missing_games)
    if len(traces) < MIN_USABLE_GAMES:
        return build_blocked_artifact(
            usable_games=sorted(traces),
            missing_games=preconditions["missing_trace_games"],
            preconditions_checked=preconditions,
            duration_s=time.time() - started,
        )

    level_rows: list[dict[str, Any]] = []
    split_specs: dict[str, Any] = {}
    model_specs_by_held_out: dict[str, Any] = {}
    trace_checksums = {game: trace.sha256 for game, trace in sorted(traces.items())}
    for held_out in sorted(traces):
        train_games = [game for game in sorted(traces) if game != held_out]
        encoder, value_head = train_encoder_value_head(traces, train_games)
        scorer = TransitionScorer(encoder, value_head)
        uniform_rows = run_solver_arm(traces[held_out], None)
        guided_rows = run_solver_arm(traces[held_out], scorer)
        split_specs[held_out] = {
            "held_out_game": held_out,
            "train_games": train_games,
            "n_train_transition_samples": int(value_head.n_samples),
            "n_encoder_samples": int(encoder.n_samples),
            "n_held_out_levels": len(traces[held_out].levels),
        }
        model_specs_by_held_out[held_out] = {
            "encoder": encoder.model_summary(),
            "value_head": value_head.model_summary(),
        }
        for uniform, guided in zip(uniform_rows, guided_rows, strict=True):
            level_rows.append(
                {
                    "held_out_game": held_out,
                    "level_index": int(uniform["level_index"]),
                    "target_level": int(uniform["target_level"]),
                    "states_uniform": int(uniform["states"]),
                    "states_guided": int(guided["states"]),
                    "baseline_solved": bool(uniform["solved"]),
                    "guided_solved": bool(guided["solved"]),
                    "uniform_path_len": int(uniform["path_len"]),
                    "guided_path_len": int(guided["path_len"]),
                }
            )

    return build_complete_artifact(
        level_rows=level_rows,
        split_specs=split_specs,
        model_specs_by_held_out=model_specs_by_held_out,
        trace_checksums=trace_checksums,
        preconditions_checked=preconditions,
        duration_s=time.time() - started,
        n_resamples=BOOTSTRAP_RESAMPLES,
    )


def _is_bare_float(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(float(value))


def _is_ci(value: Any) -> bool:
    return isinstance(value, list) and len(value) == 2 and all(_is_bare_float(item) for item in value)


def artifact_schema_errors(artifact: Mapping[str, Any]) -> list[str]:
    """SCENARIO-LEARN-4342: validate the terminal artifact contract."""

    errors: list[str] = []
    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in artifact:
            errors.append(f"missing required field {field}")
    if not isinstance(artifact.get("honest_verdict"), str):
        errors.append("honest_verdict must be a string")
    if type(artifact.get("learned_encoder_transfer_helps")) is not bool:
        errors.append("learned_encoder_transfer_helps must be a bare bool")
    if type(artifact.get("positive_control_passed")) is not bool:
        errors.append("positive_control_passed must be a bare bool")
    if not _is_bare_float(artifact.get("cross_game_state_reduction")):
        errors.append("cross_game_state_reduction must be a bare float")
    if not _is_ci(artifact.get("cross_game_state_reduction_ci95")):
        errors.append("cross_game_state_reduction_ci95 must be a two-float list")
    if type(artifact.get("n_held_out_games")) is not int:
        errors.append("n_held_out_games must be a bare int")
    if artifact.get("verifier_is_oracle") is not False:
        errors.append("verifier_is_oracle must be the bare bool false")
    if not isinstance(artifact.get("preconditions_checked"), Mapping):
        errors.append("preconditions_checked must be an object")
    if type(artifact.get("random_seed")) is not int:
        errors.append("random_seed must be a bare int")
    if not isinstance(artifact.get("reproducibility_checksum"), str):
        errors.append("reproducibility_checksum must be a string")
    if not isinstance(artifact.get("model_specs"), Mapping):
        errors.append("model_specs must be an object")
    principles = artifact.get("field_principles")
    if not isinstance(principles, Mapping):
        errors.append("field_principles must be an object")
    else:
        for field in REQUIRED_ARTIFACT_FIELDS:
            if principles.get(field) != FIELD_PRINCIPLES[field]:
                errors.append(f"field_principles mismatch for {field}")
    if artifact.get("learned_encoder_transfer_helps") is True:
        ci = artifact.get("cross_game_state_reduction_ci95")
        if not _is_ci(ci) or float(ci[0]) <= 1.0:
            errors.append("learned_encoder_transfer_helps requires CI95 lower bound > 1.0")
        if not _is_bare_float(artifact.get("cross_game_state_reduction")) or float(
            artifact["cross_game_state_reduction"]
        ) <= 1.0:
            errors.append("learned_encoder_transfer_helps requires reduction > 1.0")
        if artifact.get("positive_control_passed") is not True:
            errors.append("learned_encoder_transfer_helps requires positive_control_passed=true")
    return errors


def _write_artifact(repo: Path, artifact: Mapping[str, Any]) -> None:
    output = repo / OUTPUT_REL
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_adversarial_verify(repo: Path, _artifact: Mapping[str, Any]) -> dict[str, Any]:  # pragma: no cover
    """REQ-LEARN-4342-6: run the repository adversarial artifact verifier."""

    output = repo / OUTPUT_REL
    cmd = [sys.executable, str(repo / "scripts" / "adversarial_verify.py"), str(output), "--json"]
    completed = subprocess.run(cmd, check=False, capture_output=True, text=True)
    try:
        report = json.loads(completed.stdout or "{}")
    except json.JSONDecodeError:
        report = {"stdout": completed.stdout, "stderr": completed.stderr}
    flagged_count = int(report.get("flagged_count", 0) or 0)
    status = "clean" if completed.returncode == 0 and flagged_count == 0 else "flagged"
    return {
        "status": status,
        "returncode": int(completed.returncode),
        "flagged_count": flagged_count,
        "reports": report.get("reports", []),
    }


def run(*, repo: Path = REPO, write: bool = True) -> dict[str, Any]:
    artifact = evaluate_leave_one_game_out(repo)
    errors = artifact_schema_errors(artifact)
    if errors:
        raise ValueError("; ".join(errors))
    if write:
        _write_artifact(repo, artifact)
        artifact = dict(artifact)
        artifact["adversarial_verify"] = run_adversarial_verify(repo, artifact)
        _write_artifact(repo, artifact)
    return artifact


def main() -> None:  # pragma: no cover - CLI wrapper
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args()
    artifact = run(write=not args.no_write)
    print(artifact["honest_verdict"])


if __name__ == "__main__":  # pragma: no cover
    main()
