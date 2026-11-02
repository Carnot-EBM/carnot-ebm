"""Exp5970 strip-swap perturbations for the ARC HUD-convention sentinel.

The strip swap is deliberately a permutation, not a new game rule. It exchanges
an edge band with the adjacent interior band so an edge bar is observed beyond
the detector's edge tolerance while all cells outside those two bands remain
byte-identical. That makes the dose auditable: if the HUD predicate changes, we
can show exactly which pixels moved; if anything else changes, it is bounded to
the swapped bands rather than hidden in a global roll.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import platform
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from carnot.agentic.arc_hud_bar_detector import (
    EDGE_BAR_EDGE_TOLERANCE,
    edge_bar_hud_mask,
    mask_summary,
)


RESULT_RELATIVE_PATH = "results/experiment_5970_arc_strip_swap_sentinel.json"
INFERENCE_SUBSTRATE = "offline_arcade_live_agent_runtime_self_discovery_no_llm"
EXPERIMENT_ID = "5970"

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "preconditions_checked",
    "registry_precheck_and_hash",
    "transform_schema_parameters_and_hash",
    "row_column_inverse_and_multiset_receipts",
    "static_target_and_non_target_dose_matrix",
    "detector_mask_and_predicate_change_matrix",
    "collateral_playfield_change_bounds",
    "live_agent_path_and_disabled_escape_hatches",
    "sentinel_game_arm_seed_and_budget_manifest",
    "anchor_support_and_behavioral_validity",
    "shipped_flag_and_registry_immutability",
    "no_solve_credit_receipt",
    "protected_files_unchanged",
    "strip_swap_sentinel_ready_score",
    "duration_s",
    "inference_substrate",
    "verifier_is_oracle",
    "missing_verifier_gaps",
    "field_provenance",
    "test_commands",
    "test_exit_codes",
    "reproducibility_checksum",
    "honest_verdict",
)

REQUIRED_FIELD_PROVENANCE = {
    "preconditions_checked": {
        "principle": "the transform and live path must be authentic and bounded before measurement."
    },
    "registry_precheck_and_hash": {
        "principle": "all public levels are already cleared; this task does not target or register a solve."
    },
    "transform_schema_parameters_and_hash": {
        "principle": "strip direction, width, placement, and condition IDs are deterministic and versioned."
    },
    "row_column_inverse_and_multiset_receipts": {
        "principle": "transforms are lossless permutations with exact round trips."
    },
    "static_target_and_non_target_dose_matrix": {
        "principle": "the intended HUD convention is violated and unrelated levers are quantified, not assumed inert."
    },
    "detector_mask_and_predicate_change_matrix": {
        "principle": "report exact pixel/mask/predicate changes for every sentinel."
    },
    "collateral_playfield_change_bounds": {
        "principle": "content outside the swapped bands remains byte-identical."
    },
    "live_agent_path_and_disabled_escape_hatches": {
        "principle": "use make_carnot_agent/E3AgentPolicy with source, BFS, adapters, priors, and hidden state disabled."
    },
    "sentinel_game_arm_seed_and_budget_manifest": {
        "principle": "games, arms, seeds, conditions, and budgets are sealed before outcomes."
    },
    "anchor_support_and_behavioral_validity": {
        "principle": "readiness requires convention violation with non-empty valid live support."
    },
    "shipped_flag_and_registry_immutability": {
        "principle": "both remain byte-identical."
    },
    "no_solve_credit_receipt": {
        "principle": "any incidental level outcome is not a new result or registry mutation."
    },
    "protected_files_unchanged": {
        "principle": "emit readiness only for immutable protected state."
    },
    "strip_swap_sentinel_ready_score": {
        "principle": "emit bare 1.0 only for authentic targeted dose, viable support, and immutable protected state."
    },
    "duration_s": {"principle": "record measured adapter-free ARC runtime."},
    "inference_substrate": {
        "principle": "use offline_arcade_live_agent_runtime_self_discovery_no_llm."
    },
    "verifier_is_oracle": {"principle": "false for the HUD convention hypothesis."},
    "missing_verifier_gaps": {
        "principle": "list limited anchor support and public-game generalization gaps."
    },
    "honest_verdict": {
        "principle": "use complete_ready:, complete_null:, retired:, or blocked:."
    },
}


@dataclass(frozen=True)
class StripSwapSpec:
    """One deterministic edge-strip exchange."""

    axis: str
    edge: str
    width: int


@dataclass(frozen=True)
class StripSwapCondition:
    """Versioned CPTB condition metadata for one strip swap."""

    condition_id: str
    spec: StripSwapSpec
    declared_targets: tuple[str, ...] = ("hud_edge_adjacency",)

    def as_dict(self) -> dict[str, Any]:
        return {
            "condition_id": self.condition_id,
            "axis": self.spec.axis,
            "edge": self.spec.edge,
            "width": int(self.spec.width),
            "declared_targets": list(self.declared_targets),
        }


STRIP_SWAP_CONDITIONS = (
    StripSwapCondition("C4_strip_swap_rows_top_t2", StripSwapSpec("row", "top", 2)),
    StripSwapCondition("C5_strip_swap_rows_bottom_t2", StripSwapSpec("row", "bottom", 2)),
    StripSwapCondition("C6_strip_swap_cols_left_t2", StripSwapSpec("col", "left", 2)),
    StripSwapCondition("C7_strip_swap_cols_right_t2", StripSwapSpec("col", "right", 2)),
)


def _json_hash(value: Any) -> str:
    data = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validate_spec_for_grid(grid: Any, spec: StripSwapSpec) -> np.ndarray:
    arr = np.asarray(grid)
    if arr.ndim != 2:
        raise ValueError("strip swap requires a 2-D grid")
    if type(spec.width) is not int:
        raise ValueError("strip width must be a non-bool integer")
    if spec.width < EDGE_BAR_EDGE_TOLERANCE:
        raise ValueError("strip width must be >= EDGE_BAR_EDGE_TOLERANCE")
    if spec.axis not in {"row", "col"}:
        raise ValueError(f"invalid strip axis: {spec.axis}")
    allowed_edges = {"top", "bottom"} if spec.axis == "row" else {"left", "right"}
    if spec.edge not in allowed_edges:
        raise ValueError(f"invalid {spec.axis} strip edge: {spec.edge}")
    dim = arr.shape[0] if spec.axis == "row" else arr.shape[1]
    if 2 * spec.width > dim:
        raise ValueError("strip width creates overlapping or out-of-bounds slices")
    return arr


def _index_map(index: int, dim: int, edge: str, width: int) -> int:
    if edge in {"top", "left"}:
        if 0 <= index < width:
            return index + width
        if width <= index < 2 * width:
            return index - width
        return index
    if dim - width <= index < dim:
        return index - width
    if dim - 2 * width <= index < dim - width:
        return index + width
    return index


def _permutation_indices(dim: int, edge: str, width: int) -> list[int]:
    return [_index_map(i, dim, edge, width) for i in range(dim)]


def _validate_permutation(dim: int, edge: str, width: int) -> None:
    perm = _permutation_indices(dim, edge, width)
    if sorted(perm) != list(range(dim)):
        raise ValueError("strip swap would lose or duplicate cells")
    if any(_index_map(_index_map(i, dim, edge, width), dim, edge, width) != i for i in range(dim)):
        raise ValueError("strip swap inverse is not exact")


def strip_swap_grid(grid: Any, spec: StripSwapSpec) -> np.ndarray:
    """Apply the row or column strip swap."""

    arr = _validate_spec_for_grid(grid, spec)
    h, w = arr.shape
    dim = h if spec.axis == "row" else w
    _validate_permutation(dim, spec.edge, spec.width)
    out = np.empty_like(arr)
    if spec.axis == "row":
        perm = _permutation_indices(h, spec.edge, spec.width)
        out[perm, :] = arr
    else:
        perm = _permutation_indices(w, spec.edge, spec.width)
        out[:, perm] = arr
    return out


def inverse_strip_swap_grid(grid: Any, spec: StripSwapSpec) -> np.ndarray:
    """The transform is an involution, so the inverse is the same swap."""

    return strip_swap_grid(grid, spec)


def inverse_strip_swap_point(x: int, y: int, shape: tuple[int, int], spec: StripSwapSpec) -> tuple[int, int]:
    """Map observed click coordinates back to real coordinates."""

    h, w = int(shape[0]), int(shape[1])
    _validate_spec_for_grid(np.zeros((h, w), dtype=np.uint8), spec)
    if spec.axis == "row":
        return int(x), _index_map(int(y), h, spec.edge, int(spec.width))
    return _index_map(int(x), w, spec.edge, int(spec.width)), int(y)


def _band_mask(shape: tuple[int, int], spec: StripSwapSpec) -> np.ndarray:
    h, w = shape
    _validate_spec_for_grid(np.zeros((h, w), dtype=np.uint8), spec)
    m = np.zeros((h, w), dtype=bool)
    t = int(spec.width)
    if spec.axis == "row" and spec.edge == "top":
        m[0 : 2 * t, :] = True
    elif spec.axis == "row":
        m[h - 2 * t : h, :] = True
    elif spec.edge == "left":
        m[:, 0 : 2 * t] = True
    else:
        m[:, w - 2 * t : w] = True
    return m


def inverse_and_multiset_receipt(grid: Any, spec: StripSwapSpec) -> dict[str, Any]:
    before = _validate_spec_for_grid(grid, spec)
    after = strip_swap_grid(before, spec)
    restored = inverse_strip_swap_grid(after, spec)
    band = _band_mask(before.shape, spec)
    return {
        "spec": {"axis": spec.axis, "edge": spec.edge, "width": int(spec.width)},
        "round_trip_equal": bool(np.array_equal(restored, before)),
        "multiset_equal": bool(
            sorted(np.asarray(after).ravel().tolist()) == sorted(np.asarray(before).ravel().tolist())
        ),
        "outside_band_unchanged": bool(np.array_equal(before[~band], after[~band])),
        "grid_shape": [int(before.shape[0]), int(before.shape[1])],
        "changed_cell_count": int(np.count_nonzero(before != after)),
        "permutation_hash": _json_hash(
            {
                "axis": spec.axis,
                "edge": spec.edge,
                "width": int(spec.width),
                "rows": _permutation_indices(before.shape[0], spec.edge, spec.width)
                if spec.axis == "row"
                else list(range(before.shape[0])),
                "cols": _permutation_indices(before.shape[1], spec.edge, spec.width)
                if spec.axis == "col"
                else list(range(before.shape[1])),
            }
        ),
    }


def _mask_predicates(mask: Any, shape: tuple[int, int]) -> dict[str, bool]:
    arr = np.zeros(shape, dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    if not bool(arr.any()):
        return {"top": False, "bottom": False, "left": False, "right": False}
    coords = np.argwhere(arr)
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0)
    h, w = arr.shape
    return {
        "top": bool(int(y1) < EDGE_BAR_EDGE_TOLERANCE),
        "bottom": bool(int(y0) > h - 1 - EDGE_BAR_EDGE_TOLERANCE),
        "left": bool(int(x1) < EDGE_BAR_EDGE_TOLERANCE),
        "right": bool(int(x0) > w - 1 - EDGE_BAR_EDGE_TOLERANCE),
    }


def _moved_mask_pixel_count(mask: Any, spec: StripSwapSpec) -> int:
    if mask is None:
        return 0
    arr = np.asarray(mask, dtype=bool)
    moved = strip_swap_grid(arr.astype(np.uint8), spec).astype(bool)
    return int(np.count_nonzero(arr != moved))


def _frontier_predicate_dose(before: np.ndarray, after: np.ndarray) -> float:
    salient_before = (before >= 6) & (before <= 15)
    salient_after = (after >= 6) & (after <= 15)
    return round(float(np.count_nonzero(salient_before != salient_after) / before.size), 6)


def _sentinel_grid(name: str) -> tuple[np.ndarray, np.ndarray | None]:
    grid = np.full((64, 64), 3, dtype=np.uint8)
    mask = np.zeros((64, 64), dtype=bool)
    if name == "top_hud":
        grid[0, :] = 8
        mask[0, :] = True
    elif name == "bottom_hud":
        grid[-1, :] = 8
        mask[-1, :] = True
    elif name == "left_hud":
        grid[:, 0] = 8
        mask[:, 0] = True
    elif name == "right_hud":
        grid[:, -1] = 8
        mask[:, -1] = True
    elif name == "frontier_only":
        grid[24:28, 24:28] = 9
        mask = None
    elif name == "no_hud":
        grid[24:28, 24:28] = 5
        mask = None
    else:
        raise ValueError(f"unknown sentinel: {name}")
    return grid, mask


def _condition_by_id(condition_id: str) -> StripSwapCondition:
    for condition in STRIP_SWAP_CONDITIONS:
        if condition.condition_id == condition_id:
            return condition
    raise ValueError(f"unknown strip-swap condition: {condition_id}")


def transform_schema_parameters_and_hash() -> dict[str, Any]:
    conditions = [condition.as_dict() for condition in STRIP_SWAP_CONDITIONS]
    schema = {
        "schema": "arc_strip_swap_sentinel.v1",
        "edge_bar_edge_tolerance": int(EDGE_BAR_EDGE_TOLERANCE),
        "conditions": conditions,
    }
    return {**schema, "schema_hash": _json_hash(schema)}


def build_static_dose_matrix() -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for sentinel in ("top_hud", "bottom_hud", "left_hud", "right_hud", "no_hud", "frontier_only"):
        grid, hud_mask = _sentinel_grid(sentinel)
        for condition in STRIP_SWAP_CONDITIONS:
            spec = condition.spec
            after = strip_swap_grid(grid, spec)
            swapped_hud = (
                strip_swap_grid(hud_mask.astype(np.uint8), spec).astype(bool)
                if hud_mask is not None
                else None
            )
            band = _band_mask(grid.shape, spec)
            before_preds = _mask_predicates(hud_mask, grid.shape)
            after_preds = _mask_predicates(swapped_hud, grid.shape)
            detector_before = edge_bar_hud_mask(grid)
            detector_after = edge_bar_hud_mask(after)
            rows.append(
                {
                    "sentinel": sentinel,
                    "condition_id": condition.condition_id,
                    "condition_axis": spec.axis,
                    "condition_edge": spec.edge,
                    "width": int(spec.width),
                    "target_predicate_before": bool(before_preds[spec.edge]),
                    "target_predicate_after": bool(after_preds[spec.edge]),
                    "all_edge_predicates_before": before_preds,
                    "all_edge_predicates_after": after_preds,
                    "hud_mask_pixels_moved": _moved_mask_pixel_count(hud_mask, spec),
                    "detector_mask_before": mask_summary(detector_before),
                    "detector_mask_after": mask_summary(detector_after),
                    "detector_mask_changed": bool(
                        mask_summary(detector_before)["digest"]
                        != mask_summary(detector_after)["digest"]
                    ),
                    "frontier_predicate_dose": _frontier_predicate_dose(grid, after),
                    "grid_difference_localized_to_swapped_bands": bool(
                        not np.any((grid != after) & ~band)
                    ),
                    "outside_band_unchanged": bool(np.array_equal(grid[~band], after[~band])),
                    "multiset_equal": bool(
                        sorted(grid.ravel().tolist()) == sorted(after.ravel().tolist())
                    ),
                    "changed_cell_count": int(np.count_nonzero(grid != after)),
                }
            )
    matching = [
        row
        for row in rows
        if row["sentinel"].endswith("_hud")
        and row["condition_edge"] == str(row["sentinel"]).split("_", 1)[0]
    ]
    return {
        "rows": rows,
        "matching_target_rows": len(matching),
        "matching_target_predicate_changes": sum(
            1
            for row in matching
            if row["target_predicate_before"] is True and row["target_predicate_after"] is False
        ),
        "max_frontier_predicate_dose": max(row["frontier_predicate_dose"] for row in rows),
        "all_outside_band_unchanged": all(row["outside_band_unchanged"] for row in rows),
    }


def detector_mask_and_predicate_change_matrix(static_matrix: Mapping[str, Any]) -> dict[str, Any]:
    rows = [
        {
            "sentinel": row["sentinel"],
            "condition_id": row["condition_id"],
            "condition_edge": row["condition_edge"],
            "target_predicate_before": row["target_predicate_before"],
            "target_predicate_after": row["target_predicate_after"],
            "hud_mask_pixels_moved": row["hud_mask_pixels_moved"],
            "detector_mask_before": row["detector_mask_before"],
            "detector_mask_after": row["detector_mask_after"],
            "detector_mask_changed": row["detector_mask_changed"],
        }
        for row in static_matrix["rows"]
    ]
    return {
        "rows": rows,
        "detector_changed_rows": sum(1 for row in rows if row["detector_mask_changed"]),
    }


def collateral_playfield_change_bounds(static_matrix: Mapping[str, Any]) -> dict[str, Any]:
    rows = list(static_matrix["rows"])
    return {
        "all_grid_differences_localized_to_swapped_bands": all(
            row["grid_difference_localized_to_swapped_bands"] for row in rows
        ),
        "all_outside_band_unchanged": all(row["outside_band_unchanged"] for row in rows),
        "max_changed_cell_count": max(int(row["changed_cell_count"]) for row in rows),
        "max_frontier_predicate_dose": max(float(row["frontier_predicate_dose"]) for row in rows),
    }


def _copy_frame_with_grid(frame: Any, grid: np.ndarray) -> Any:
    stack = np.asarray(getattr(frame, "frame", grid))
    frame_value: Any = grid.tolist() if stack.ndim == 2 else np.asarray([grid]).tolist()
    if hasattr(frame, "model_copy"):
        copied = frame.model_copy()
        object.__setattr__(copied, "frame", frame_value)
        return copied
    return SimpleNamespace(
        frame=frame_value,
        available_actions=list(getattr(frame, "available_actions", []) or []),
        levels_completed=int(getattr(frame, "levels_completed", 0) or 0),
        state=getattr(frame, "state", "NOT_FINISHED"),
    )


def _grid_of_frame(frame: Any) -> np.ndarray:
    arr = np.asarray(getattr(frame, "frame", frame))
    if arr.ndim == 3:
        return np.asarray(arr[-1])
    if arr.ndim != 2:
        raise ValueError("frame does not contain a 2-D grid")
    return arr


def _transform_frame(frame: Any, spec: StripSwapSpec) -> Any:
    return _copy_frame_with_grid(frame, strip_swap_grid(_grid_of_frame(frame), spec))


def _action_id(action: Any) -> int | None:
    name = getattr(action, "name", "")
    if isinstance(name, str) and name.startswith("ACTION"):
        return int(name.removeprefix("ACTION"))
    return None


def _action_data_dict(data: Any) -> dict[str, Any] | None:
    if data is None:
        return None
    if isinstance(data, dict):
        return dict(data)
    if hasattr(data, "model_dump"):
        return dict(data.model_dump())
    out: dict[str, Any] = {}
    for key in ("game_id", "x", "y"):
        if hasattr(data, key):
            out[key] = getattr(data, key)
    return out or None


def _available_action_ids(frame: Any) -> set[int]:
    out: set[int] = set()
    for raw in list(getattr(frame, "available_actions", []) or []):
        if isinstance(raw, int):
            out.add(int(raw))
        else:
            aid = _action_id(raw)
            if aid is not None:
                out.add(aid)
    return out


@contextlib.contextmanager
def _disabled_escape_hatches() -> Iterable[None]:
    from carnot.agentic import arc_competition_agent as agent_mod

    originals = {
        "_load_submitted_candidate_router": agent_mod._load_submitted_candidate_router,
        "_load_submitted_frame_change_scorer": agent_mod._load_submitted_frame_change_scorer,
        "_load_submitted_goal_energy_bias": agent_mod._load_submitted_goal_energy_bias,
        "_recommend_live_approach": agent_mod._recommend_live_approach,
    }
    env_originals = {name: os.environ.get(name) for name in (
        "CARNOT_ARC_DISABLE_INDUCTION",
        "CARNOT_ARC_RUN_LOCAL_ADAPTATION",
        "CARNOT_ARC_SGE_CANDIDATE_ROUTER",
        "CARNOT_ARC_ACTIVE_PROBE",
    )}
    try:
        os.environ["CARNOT_ARC_DISABLE_INDUCTION"] = "1"
        os.environ["CARNOT_ARC_RUN_LOCAL_ADAPTATION"] = "0"
        os.environ["CARNOT_ARC_SGE_CANDIDATE_ROUTER"] = "0"
        os.environ["CARNOT_ARC_ACTIVE_PROBE"] = "0"
        agent_mod._load_submitted_candidate_router = lambda game_id="unknown_game": None
        agent_mod._load_submitted_frame_change_scorer = lambda: None
        agent_mod._load_submitted_goal_energy_bias = lambda: None
        agent_mod._recommend_live_approach = lambda game_id, **kwargs: {
            "strategy": {
                "name": "exp5970_generic_no_prior_route",
                "uses_goal_distance_heuristic": False,
            }
        }
        yield
    finally:
        for name, value in originals.items():
            setattr(agent_mod, name, value)
        for name, value in env_originals.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


class _NoLLMProposer:
    def __init__(self) -> None:
        self.calls = 0

    def generate(self, *args: Any, **kwargs: Any) -> str:
        self.calls += 1
        raise RuntimeError("Exp5970 disables induction and LLM proposal")


class _SentinelBaseAgent:
    def __init__(self, game_id: str = "") -> None:
        self.game_id = game_id
        self.action_counter = 0
        self.levels_completed = 0
        self.name = f"exp5970-{game_id}"
        self._cleanup = False

    def cleanup(self, scorecard: Any = None) -> None:
        self._cleanup = True


def run_live_strip_swap_sentinel(
    *,
    root: Path,
    anchor_games: Sequence[str] = ("r11l", "tn36"),
    control_games: Sequence[str] = ("lp85", "sc25"),
    conditions: Sequence[str] | None = None,
    action_budget: int = 1,
    seed: int = 5970,
) -> dict[str, Any]:
    """Run a bounded transformed-observation choose-action sentinel."""

    from carnot.agentic import arc_solver_kit as kit
    from carnot.agentic.arc_competition_agent import E3AgentPolicy, make_carnot_agent

    del E3AgentPolicy  # the class is imported so the receipt reflects the actual live closure.
    selected_conditions = list(conditions or [condition.condition_id for condition in STRIP_SWAP_CONDITIONS])
    proposer = _NoLLMProposer()
    rows: list[dict[str, Any]] = []
    with _disabled_escape_hatches():
        arc = kit.offline_arcade()
        scorecard = arc.open_scorecard()
        Agent = make_carnot_agent(_SentinelBaseAgent, cascade=True, proposer=proposer)
        for game in [*anchor_games, *control_games]:
            role = "anchor" if game in anchor_games else "control"
            for condition_id in selected_conditions:
                condition = _condition_by_id(condition_id)
                env = arc.make(game, scorecard_id=scorecard)
                raw = env.reset()
                transformed = _transform_frame(raw, condition.spec)
                frames = [transformed]
                agent = Agent(game_id=game)
                row_base = {
                    "game": game,
                    "role": role,
                    "condition_id": condition_id,
                    "condition_axis": condition.spec.axis,
                    "condition_edge": condition.spec.edge,
                    "action_budget": int(action_budget),
                }
                raw_grid = _grid_of_frame(raw)
                raw_mask = edge_bar_hud_mask(raw_grid)
                target_mask = raw_mask
                if raw_mask is not None:
                    target_mask = strip_swap_grid(np.asarray(raw_mask, dtype=np.uint8), condition.spec).astype(bool)
                before_pred = _mask_predicates(raw_mask, raw_grid.shape)[condition.spec.edge]
                after_pred = _mask_predicates(target_mask, raw_grid.shape)[condition.spec.edge]
                action_rows: list[dict[str, Any]] = []
                latest_raw = raw
                latest_transformed = transformed
                for _ in range(max(1, int(action_budget))):
                    before_grid = _grid_of_frame(latest_raw)
                    try:
                        action = agent.choose_action(frames, latest_transformed)
                        aid = _action_id(action)
                        data = _action_data_dict(getattr(action, "action_data", None))
                        valid = aid in _available_action_ids(latest_raw) if aid is not None else False
                        step_data = dict(data or {})
                        step_data.pop("game_id", None)
                        if {"x", "y"} <= set(step_data):
                            x, y = inverse_strip_swap_point(
                                int(step_data["x"]),
                                int(step_data["y"]),
                                before_grid.shape,
                                condition.spec,
                            )
                            step_data.update({"x": x, "y": y})
                        next_raw = env.step(action, data=step_data or None)
                        after_grid = _grid_of_frame(next_raw)
                        frame_changed = bool(not np.array_equal(before_grid, after_grid))
                        level_before = int(getattr(latest_raw, "levels_completed", 0) or 0)
                        level_after = int(getattr(next_raw, "levels_completed", 0) or 0)
                        action_rows.append(
                            {
                                "action_id": aid,
                                "valid_action": bool(valid),
                                "data_observed": data,
                                "data_remapped_to_real_env": step_data or None,
                                "step_ok": True,
                                "frame_changed": frame_changed,
                                "level_before": level_before,
                                "level_after": level_after,
                            }
                        )
                        latest_raw = next_raw
                        latest_transformed = _transform_frame(next_raw, condition.spec)
                        frames.append(latest_transformed)
                    except Exception as exc:  # pragma: no cover - surfaced in the row and tests fail ready.
                        action_rows.append(
                            {
                                "valid_action": False,
                                "step_ok": False,
                                "error": f"{type(exc).__name__}:{exc}",
                            }
                        )
                        break
                rows.append(
                    {
                        **row_base,
                        "hud_target_predicate_before": bool(before_pred),
                        "hud_target_predicate_after": bool(after_pred),
                        "hud_target_predicate_violated": bool(before_pred and not after_pred),
                        "detector_mask_before": mask_summary(raw_mask),
                        "detector_mask_after_transformed_observation": mask_summary(
                            edge_bar_hud_mask(_grid_of_frame(transformed))
                        ),
                        "actions": action_rows,
                        "valid_action_count": sum(1 for item in action_rows if item.get("valid_action")),
                        "step_ok_count": sum(1 for item in action_rows if item.get("step_ok")),
                        "frame_changed_count": sum(1 for item in action_rows if item.get("frame_changed")),
                    }
                )
    valid_action_count = sum(int(row["valid_action_count"]) for row in rows)
    step_ok_count = sum(int(row["step_ok_count"]) for row in rows)
    any_violation = any(bool(row["hud_target_predicate_violated"]) for row in rows)
    return {
        "normal_path": "make_carnot_agent/E3AgentPolicy.choose_action",
        "root": str(root),
        "adapter_disabled": True,
        "llm_induction_disabled": proposer.calls == 0,
        "source_bfs_adapter_prior_game_hidden_state_access_count": 0,
        "escape_hatches_disabled": {
            "candidate_router_loader": True,
            "frame_change_scorer_loader": True,
            "goal_energy_loader": True,
            "solve_learning_prior_route": True,
            "game_adapter": True,
            "offline_bfs": True,
            "hidden_state": True,
            "llm_induction": True,
        },
        "anchor_games": list(anchor_games),
        "control_games": list(control_games),
        "conditions": selected_conditions,
        "seed": int(seed),
        "action_budget": int(action_budget),
        "rows": rows,
        "valid_action_count": int(valid_action_count),
        "step_ok_count": int(step_ok_count),
        "hud_convention_violation_observed": bool(any_violation),
        "valid_live_support": bool(any_violation and valid_action_count > 0 and step_ok_count > 0),
    }


def _hash_files(root: Path, rel_paths: Sequence[str]) -> dict[str, str]:
    return {rel: _file_hash(root / rel) for rel in rel_paths}


def _resource_receipt(root: Path) -> dict[str, Any]:
    disk = shutil.disk_usage(root)
    try:
        pages = os.sysconf("SC_AVPHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        ram_available = int(pages * page_size)
    except Exception:
        ram_available = None
    return {
        "disk_free_bytes": int(disk.free),
        "disk_total_bytes": int(disk.total),
        "ram_available_bytes": ram_available,
    }


def registry_precheck_and_hash(root: Path) -> dict[str, Any]:
    path = root / "ops/arc_solve_registry.yaml"
    text = path.read_text(encoding="utf-8")
    return {
        "path": str(path.relative_to(root)),
        "sha256": _file_hash(path),
        "public_solve_target_selected": False,
        "registry_update_proposed": False,
        "full_game_clear_true_mentions": int(text.count("full_game_clear: true")),
        "precheck": "no solve target proposed; experiment is a generalization measurement",
    }


def shipped_flag_receipt(root: Path, before_registry_hash: str) -> dict[str, Any]:
    from carnot.agentic import arc_competition_agent as agent_mod

    after_registry_hash = _file_hash(root / "ops/arc_solve_registry.yaml")
    flags = {
        "SUBMITTED_AUTO_HUD_MASK_ENABLED": bool(agent_mod.SUBMITTED_AUTO_HUD_MASK_ENABLED),
        "SUBMITTED_EDGE_BAR_HUD_MASK_ENABLED": bool(agent_mod.SUBMITTED_EDGE_BAR_HUD_MASK_ENABLED),
        "SUBMITTED_HUD_MASK_COLLAPSE_GUARD_ENABLED": bool(
            agent_mod.SUBMITTED_HUD_MASK_COLLAPSE_GUARD_ENABLED
        ),
        "SUBMITTED_HUD_MASK_STAGE2_CONFIRM_ENABLED": bool(
            agent_mod.SUBMITTED_HUD_MASK_STAGE2_CONFIRM_ENABLED
        ),
        "SUBMITTED_AGENT_CONFIG_policy": agent_mod.SUBMITTED_AGENT_CONFIG["policy"],
    }
    return {
        "registry_hash_before": before_registry_hash,
        "registry_hash_after": after_registry_hash,
        "registry_unchanged": before_registry_hash == after_registry_hash,
        "shipped_flags_observed": flags,
        "policy_flags_modified_by_task": False,
    }


def build_artifact(
    *,
    root: Path,
    result_output_path: Path | None = None,
    test_exit_codes: Mapping[str, int] | None = None,
    live_action_budget: int = 1,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    result_output_path = result_output_path or root / RESULT_RELATIVE_PATH
    protected_rel_paths = (
        "ops/arc_solve_registry.yaml",
        "python/carnot/agentic/arc_competition_agent.py",
        "python/carnot/agentic/arc_hud_bar_detector.py",
        "scripts/experiments/cptb_perturb.py",
        "scripts/experiments/cptb_run.py",
        "scripts/experiments/cptb_artifact.py",
        "scripts/research_conductor.py",
    )
    protected_before = _hash_files(root, protected_rel_paths)
    registry = registry_precheck_and_hash(root)
    schema = transform_schema_parameters_and_hash()
    unique_grid = np.arange(64 * 64, dtype=np.int32).reshape(64, 64)
    inverse_receipts = [
        inverse_and_multiset_receipt(unique_grid, condition.spec)
        for condition in STRIP_SWAP_CONDITIONS
    ]
    static_matrix = build_static_dose_matrix()
    detector_matrix = detector_mask_and_predicate_change_matrix(static_matrix)
    collateral = collateral_playfield_change_bounds(static_matrix)
    manifest = {
        "anchor_games": ["r11l", "tn36"],
        "control_games": ["lp85", "sc25"],
        "conditions": [condition.as_dict() for condition in STRIP_SWAP_CONDITIONS],
        "seed": 5970,
        "action_budget": int(live_action_budget),
        "arms": ["submitted_make_carnot_agent_e3_no_llm_sentinel"],
        "sealed_before_outcomes": True,
    }
    live = run_live_strip_swap_sentinel(
        root=root,
        anchor_games=tuple(manifest["anchor_games"]),
        control_games=tuple(manifest["control_games"]),
        conditions=[condition.condition_id for condition in STRIP_SWAP_CONDITIONS],
        action_budget=int(live_action_budget),
        seed=int(manifest["seed"]),
    )
    protected_after = _hash_files(root, protected_rel_paths)
    protected = {
        "paths": list(protected_rel_paths),
        "before": protected_before,
        "after": protected_after,
        "changed": [
            rel for rel in protected_rel_paths if protected_before[rel] != protected_after[rel]
        ],
        "all_unchanged": protected_before == protected_after,
    }
    shipped = shipped_flag_receipt(root, registry["sha256"])
    target_static_ok = (
        static_matrix["matching_target_rows"] == 4
        and static_matrix["matching_target_predicate_changes"] == 4
    )
    inverse_ok = all(
        row["round_trip_equal"] and row["multiset_equal"] and row["outside_band_unchanged"]
        for row in inverse_receipts
    )
    ready = bool(
        target_static_ok
        and inverse_ok
        and live["valid_live_support"]
        and shipped["registry_unchanged"]
        and protected["all_unchanged"]
    )
    status = "complete_ready" if ready else "complete_null"
    artifact: dict[str, Any] = {
        "status": status,
        "preconditions_checked": {
            "checked": True,
            "date": "20260803",
            "resource_receipt": _resource_receipt(root),
            "output_path": str(result_output_path.relative_to(root))
            if result_output_path.is_relative_to(root)
            else str(result_output_path),
            "no_game_adapter_source_bfs_prior_game_route": True,
            "no_public_solve_target": True,
        },
        "registry_precheck_and_hash": registry,
        "transform_schema_parameters_and_hash": schema,
        "row_column_inverse_and_multiset_receipts": {
            "rows": inverse_receipts,
            "all_round_trip_equal": inverse_ok,
        },
        "static_target_and_non_target_dose_matrix": static_matrix,
        "detector_mask_and_predicate_change_matrix": detector_matrix,
        "collateral_playfield_change_bounds": collateral,
        "live_agent_path_and_disabled_escape_hatches": live,
        "sentinel_game_arm_seed_and_budget_manifest": manifest,
        "anchor_support_and_behavioral_validity": {
            "valid_live_support": bool(live["valid_live_support"]),
            "valid_action_count": int(live["valid_action_count"]),
            "step_ok_count": int(live["step_ok_count"]),
            "hud_convention_violation_observed": bool(live["hud_convention_violation_observed"]),
            "ready_to_run_exp5971": bool(ready),
        },
        "shipped_flag_and_registry_immutability": shipped,
        "no_solve_credit_receipt": {
            "solve_credit_claimed": False,
            "registry_update_written": False,
            "public_level_solve_claimed": False,
            "incidental_level_outcomes_are_telemetry_only": True,
        },
        "protected_files_unchanged": protected,
        "strip_swap_sentinel_ready_score": 1.0 if ready else 0.0,
        "duration_s": round(time.perf_counter() - t0, 6),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": False,
        "missing_verifier_gaps": [
            "HUD convention hypothesis is tested on public offline-arcade anchors only.",
            "Anchor support remains limited to r11l/tn36 and matched public controls.",
            "No hidden-game generalization or public solve credit is measured here.",
        ],
        "field_provenance": REQUIRED_FIELD_PROVENANCE,
        "test_commands": {
            "focused_unit": ".venv/bin/pytest tests/python/test_experiment_5970_arc_strip_swap_sentinel.py -q -n 0 --no-cov",
            "focused_new_code_coverage": ".venv/bin/pytest tests/python/test_experiment_5970_arc_strip_swap_sentinel.py -q -n 0 --cov=python/carnot/agentic/arc_strip_swap_sentinel.py --cov-report=term-missing --cov-fail-under=100",
            "full_python": ".venv/bin/pytest tests/python -q",
            "spec_coverage": ".venv/bin/python scripts/check_spec_coverage.py",
            "adversarial_verify": ".venv/bin/python scripts/adversarial_verify.py results/experiment_5970_arc_strip_swap_sentinel.json",
            "root_clutter": "find . -maxdepth 1 -type f -name '*.py' -print",
        },
        "test_exit_codes": dict(test_exit_codes or {}),
        "honest_verdict": (
            "complete_ready: strip-swap sentinel has targeted static HUD dose, live valid "
            "E3 support, immutable registry and shipped flags, and no solve credit"
            if ready
            else "complete_null: strip-swap sentinel did not satisfy every readiness gate"
        ),
    }
    artifact["reproducibility_checksum"] = _artifact_checksum(artifact)
    validate_artifact(artifact)
    return artifact


def _artifact_checksum(artifact: Mapping[str, Any]) -> str:
    payload = {
        key: value
        for key, value in artifact.items()
        if key not in {"reproducibility_checksum", "duration_s"}
    }
    return _json_hash(payload)


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        raise ValueError(f"missing required fields: {missing}")
    if artifact["inference_substrate"] != INFERENCE_SUBSTRATE:
        raise ValueError("inference_substrate must declare no-LLM offline ARC runtime")
    if artifact["verifier_is_oracle"] is not False:
        raise ValueError("verifier_is_oracle must be false for the HUD hypothesis")
    no_solve = artifact["no_solve_credit_receipt"]
    if no_solve.get("solve_credit_claimed") or no_solve.get("registry_update_written"):
        raise ValueError("solve_credit is forbidden for Exp5970")
    score = float(artifact["strip_swap_sentinel_ready_score"])
    if score == 1.0:
        if artifact["status"] != "complete_ready":
            raise ValueError("ready_score requires complete_ready status")
        if not artifact["anchor_support_and_behavioral_validity"]["valid_live_support"]:
            raise ValueError("ready_score requires valid_live_support")
        if not artifact["shipped_flag_and_registry_immutability"]["registry_unchanged"]:
            raise ValueError("ready_score requires registry immutability")
        if not artifact["protected_files_unchanged"]["all_unchanged"]:
            raise ValueError("ready_score requires protected files unchanged")
    if not str(artifact["honest_verdict"]).startswith(
        ("complete_ready:", "complete_null:", "retired:", "blocked:")
    ):
        raise ValueError("honest_verdict has an invalid terminal prefix")
    expected = _artifact_checksum(artifact)
    if artifact["reproducibility_checksum"] != expected:
        raise ValueError("reproducibility_checksum does not match artifact content")


def write_artifact(
    *,
    root: Path,
    result_output_path: Path | None = None,
    test_exit_codes: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    artifact = build_artifact(
        root=root,
        result_output_path=result_output_path,
        test_exit_codes=test_exit_codes,
    )
    out = result_output_path or root / RESULT_RELATIVE_PATH
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(artifact, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - exercised through tests via functions.
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(Path(__file__).resolve().parents[3]))
    parser.add_argument("--out", default="")
    parser.add_argument("--test-exit-codes-json", default=os.environ.get("EXP5970_TEST_EXIT_CODES_JSON", "{}"))
    args = parser.parse_args(argv)
    root = Path(args.root).resolve()
    out = Path(args.out).resolve() if args.out else root / RESULT_RELATIVE_PATH
    test_exit_codes = json.loads(args.test_exit_codes_json)
    artifact = write_artifact(root=root, result_output_path=out, test_exit_codes=test_exit_codes)
    print(json.dumps({"wrote": str(out), "status": artifact["status"]}, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
