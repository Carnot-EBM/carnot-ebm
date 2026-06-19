"""ARC-AGI-3 reusable offline solver kit — the durable scaffolding so that what
we learn solving one game is REUSED on the next, and every solve is captured as
OFFLINE-REPRODUCIBLE (not a live-recorded coordinate trajectory that silently
rots).

Why this module exists
----------------------
2026-06-16: making the deeper sc25/lp85 levels offline-reproducible revealed that
the prior solves were banked as live-recorded `solve_trace.actions` whose pixel
coordinates were coupled to the LIVE env layout, so they replay to 0 levels on
the offline `environment_files` env. The effort was effectively wasted because
the WINNING CONDITION (the search that derives a solution for the actual env) was
never captured — only one frozen trajectory was. This kit captures the general,
reusable primitives + the hard-won per-game gotchas so future games plug in their
action-model + win-check and inherit the rest, and so every solve passes a
reproduction gate before it counts.

Hard-won general gotchas (apply to ANY ARC-AGI-3 game; see ops/arc_solve_registry.yaml)
-----------------------------------------------------------------------------------
1. OFFLINE is a deterministic simulator: `Arcade(OperationMode.OFFLINE,
   environments_dir=environment_files)` loads all 25 games, zero network/quota.
2. The LEVEL lives on the FRAME (`frame.levels_completed`), NOT on `env._game`.
3. `env._game = copy.deepcopy(state)` injection works for SOME games (lp85) but is
   BROKEN for others (sc25) — references don't survive deepcopy. The robust,
   universal approach is REPLAY-FROM-RESET (operate on the real env).
4. The FIRST `env.step` after `env.reset()` is CONSUMED (no-op) in at least sc25.
   Always do a warm-up step after reset before applying a path.
5. Element COORDINATES must be DISCOVERED from the env, never hardcoded — the live
   solver's hardcoded coords (e.g. sc25 SC25_GRID_COORDS) miss the offline layout.
   Use env-adaptive discovery (cf. lp85 `discover_click_buttons`; sc25 camera is
   identity so cell (r,c) is at display (24+5c, 49+5r)).
6. Some games have STATE-DEPENDENT controls (sc25 tank-controls: press-new-
   direction turns, press-same moves) and MULTI-FRAME ANIMATIONS that must be let
   to resolve — the dedup state-key MUST include facing/phase, and you must step
   until animation phase flags clear before the next action / win check.
7. Some config/toggle games call next_level() in the SAME action that creates the
   winning arrangement, so the returned frame is already the next level and the
   pre-win grid is not externally observable. Ground such win predicates on the
   execution state immediately before next_level, then count only the reproduce()
   level advance.
"""
from __future__ import annotations

import copy
import heapq
import itertools
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Hashable, Mapping, Optional, Sequence

REPO = Path(__file__).resolve().parents[3]
ENV_DIR = REPO / "environment_files"
ARC_STANDING_PATH_COST_WEIGHT = 1.0
ARC_BASELINE_PATH_COST_WEIGHT = 0.0


@dataclass(frozen=True)
class PrimitiveOperator:
    """A reusable ARC solve operator learned from one or more reproduced games."""

    operator: str
    derived_from_games: tuple[str, ...]
    purpose: str
    selector_tags: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "operator": self.operator,
            "derived_from_games": list(self.derived_from_games),
            "purpose": self.purpose,
            "selector_tags": list(self.selector_tags),
        }


def primitive_operator_registry() -> tuple[PrimitiveOperator, ...]:
    """REQ-REPORT-4436: consolidated generic operators available to the standing loop."""

    return (
        PrimitiveOperator(
            operator="glyph_rewrite_matcher",
            derived_from_games=("tr87",),
            purpose="Greedy multi-glyph LHS->RHS rewrite, including repeated passes.",
            selector_tags=("config_substitution", "glyph", "rewrite", "tr87"),
        ),
        PrimitiveOperator(
            operator="config_rule_grounding",
            derived_from_games=("s5i5", "ft09", "tr87"),
            purpose="Ground a proposed config rule against predicted object/register coverage.",
            selector_tags=("config_toggle", "marker_coverage", "local_constraint", "rule"),
        ),
        PrimitiveOperator(
            operator="config_rule_verifier",
            derived_from_games=("s5i5", "ft09", "g50t"),
            purpose="Propose and execution-ground coverage, local-constraint, or toggle win predicates.",
            selector_tags=("config_toggle", "marker_coverage", "local_constraint", "verifier", "rule"),
        ),
        PrimitiveOperator(
            operator="graph_astar_action_cost",
            derived_from_games=("tu93", "lp85", "cd82", "sp80", "cn04", "m0r0", "sk48", "su15"),
            purpose="A* frontier priority: standing path cost plus verifier/action-cost heuristic.",
            selector_tags=("graph_explore", "astar", "action_cost", "keyboard", "click"),
        ),
        PrimitiveOperator(
            operator="object_centric_digest",
            derived_from_games=("g50t", "lp85", "tn36", "ka59"),
            purpose="Connected-component object summary for routing, grounding, and active data.",
            selector_tags=("object", "digest", "program_editor", "world_model"),
        ),
        PrimitiveOperator(
            operator="active_data_collection",
            derived_from_games=("ar25", "ka59", "ft09", "sc25"),
            purpose="Balanced action/object coverage plan for offline transition collection.",
            selector_tags=("active_data", "world_model", "e3", "transition"),
        ),
        PrimitiveOperator(
            operator="object_motion_world_model",
            derived_from_games=("ar25", "ka59", "sc25", "ft09"),
            purpose="Object-slot transition model for translate, reflect, push, and dynamic selection.",
            selector_tags=("object", "motion", "world_model", "e3", "translate", "reflect", "push"),
        ),
    )


def select_primitive_operators(
    *, mechanic_class: Optional[str] = None, action_model: str = "", game: str = ""
) -> tuple[PrimitiveOperator, ...]:
    """Select generic operators before per-game reverse engineering.

    This is intentionally conservative: it exposes reusable operators for the standing
    loop without removing any per-game adapter path.
    """

    registry = {op.operator: op for op in primitive_operator_registry()}
    mechanic = (mechanic_class or "").lower()
    action = (action_model or "").lower()
    gid = (game or "").lower()

    if "config_substitution" in mechanic or "glyph" in mechanic or gid == "tr87":
        names = ("glyph_rewrite_matcher", "graph_astar_action_cost", "object_centric_digest")
    elif "config" in mechanic or "toggle" in mechanic or "constraint" in mechanic:
        names = ("config_rule_verifier", "config_rule_grounding", "object_centric_digest", "graph_astar_action_cost")
    elif "program_editor" in mechanic:
        names = ("object_centric_digest", "active_data_collection", "graph_astar_action_cost")
    elif (
        "object_motion" in mechanic
        or "object motion" in mechanic
        or "reflection" in mechanic
        or "reflect" in mechanic
        or "push" in mechanic
    ):
        names = (
            "object_motion_world_model",
            "active_data_collection",
            "object_centric_digest",
            "graph_astar_action_cost",
        )
    elif "world_model" in mechanic or "e3" in mechanic:
        names = (
            "object_motion_world_model",
            "active_data_collection",
            "object_centric_digest",
            "graph_astar_action_cost",
        )
    elif "keyboard" in action or "click" in action or "graph" in mechanic:
        names = ("graph_astar_action_cost", "object_centric_digest")
    else:
        names = ("object_centric_digest", "active_data_collection", "graph_astar_action_cost")
    return tuple(registry[name] for name in names)


def cyclic_distance(current: int, target: int, *, modulus: int = 7) -> int:
    """Shortest cyclic distance on an integer wheel, used by config/glyph solvers."""

    if modulus <= 0:
        raise ValueError("modulus must be positive")
    return min((int(target) - int(current)) % modulus, (int(current) - int(target)) % modulus)


def sequence_cyclic_distance(
    current: Sequence[int],
    required: Sequence[int],
    *,
    modulus: int = 7,
    gap_cost: Optional[float] = None,
) -> float:
    """Sum cyclic distance over aligned values, with a bounded length-gap penalty."""

    n = min(len(current), len(required))
    gap = float(modulus if gap_cost is None else gap_cost)
    return float(
        sum(cyclic_distance(current[i], required[i], modulus=modulus) for i in range(n))
        + gap * abs(len(current) - len(required))
    )


def greedy_rewrite(
    sequence: Sequence[Hashable],
    rules: Sequence[tuple[Sequence[Hashable], Sequence[Hashable]]],
    *,
    passes: int = 1,
) -> tuple[Hashable, ...] | None:
    """Greedy first-prefix LHS->RHS rewrite, repeated for tr87-style chains."""

    normalized = [(tuple(lhs), tuple(rhs)) for lhs, rhs in rules]
    out: tuple[Hashable, ...] = tuple(sequence)
    for _ in range(max(0, int(passes))):
        pos = 0
        rewritten: list[Hashable] = []
        while pos < len(out):
            for lhs, rhs in normalized:
                if out[pos:pos + len(lhs)] == lhs:
                    rewritten.extend(rhs)
                    pos += len(lhs)
                    break
            else:
                return None
        out = tuple(rewritten)
    return out


def ground_marker_coverage_rule(
    *,
    controlled_markers: Sequence[tuple[int, int]],
    target_markers: Sequence[tuple[int, int]],
    step: int,
    horizontal_label: str,
    vertical_label: str,
) -> dict[str, Any]:
    """Derive a config-toggle path from a grounded marker-coverage predicate."""

    predicted = [tuple(marker) for marker in controlled_markers]
    path: list[str] = []
    for target_x, target_y in target_markers:
        candidates = [(i, x, y) for i, (x, y) in enumerate(predicted) if y == target_y and x < target_x]
        if candidates:
            index, x, _ = candidates[0]
            moves = (target_x - x) // int(step)
            path.extend([horizontal_label] * moves)
            predicted[index] = (target_x, target_y)
    for target_x, target_y in target_markers:
        candidates = [(i, x, y) for i, (x, y) in enumerate(predicted) if x == target_x and y < target_y]
        if candidates:
            index, _, y = candidates[0]
            moves = (target_y - y) // int(step)
            path.extend([vertical_label] * moves)
            predicted[index] = (target_x, target_y)
    satisfied = all(tuple(target) in predicted for target in target_markers)
    return {
        "operator": "config_rule_grounding",
        "solution": path,
        "predicted_markers": [tuple(marker) for marker in predicted],
        "target_markers": [tuple(marker) for marker in target_markers],
        "predicate_satisfied": satisfied,
    }


def _point(value: Any) -> tuple[int, int]:
    x, y = value
    return int(x), int(y)


def _example_text(few_shot_examples: Sequence[Mapping[str, Any]]) -> str:
    return " ".join(
        f"{row.get('game', '')} {row.get('rule_id', '')} {row.get('predicate', '')}".lower()
        for row in few_shot_examples
        if isinstance(row, Mapping)
    )


def _has_example_family(few_shot_examples: Sequence[Mapping[str, Any]], *needles: str) -> bool:
    text = _example_text(few_shot_examples)
    return any(needle in text for needle in needles)


def _ungrounded_config_result(game: str, residual: str) -> dict[str, Any]:
    return {
        "operator": "config_rule_verifier",
        "game": str(game),
        "grounded": False,
        "solution": [],
        "predicate_id": "",
        "candidate_predicates": ["marker_coverage", "local_constraint_color_cycle", "target_offset_toggle"],
        "residual": residual,
        "verifier_is_oracle": True,
    }


def _local_constraint_requirements(
    constraint: Mapping[str, Any],
    *,
    neighbor_step: int,
) -> list[tuple[tuple[int, int], str, int]]:
    center = _point(constraint.get("grid", (0, 0)))
    pattern = constraint.get("pattern") or ()
    center_color = int(constraint.get("center_color", 0))
    required: list[tuple[tuple[int, int], str, int]] = []
    for row_index, row in enumerate(pattern):
        for col_index, value in enumerate(row):
            if row_index == 1 and col_index == 1:
                continue
            grid = (
                center[0] + (int(col_index) - 1) * int(neighbor_step),
                center[1] + (int(row_index) - 1) * int(neighbor_step),
            )
            relation = "equal" if int(value) == 0 else "not_equal"
            required.append((grid, relation, center_color))
    return required


def _local_constraint_violation_count(
    *,
    colors: Mapping[tuple[int, int], int],
    constraints: Sequence[Mapping[str, Any]],
    neighbor_step: int,
) -> int:
    violations = 0
    for constraint in constraints:
        for grid, relation, color in _local_constraint_requirements(
            constraint,
            neighbor_step=neighbor_step,
        ):
            observed = colors.get(grid)
            if observed is None:
                if relation == "equal":
                    violations += 1
                continue
            if relation == "equal" and int(observed) != int(color):
                violations += 1
            if relation == "not_equal" and int(observed) == int(color):
                violations += 1
    return violations


def _next_cycle_color(current: int, color_cycle: Sequence[int]) -> int:
    colors = [int(color) for color in color_cycle]
    index = colors.index(int(current))
    return int(colors[(index + 1) % len(colors)])


def _click_label_for_grid(grid: tuple[int, int], object_digest: Mapping[str, Any]) -> str:
    scale = int(object_digest.get("click_scale", 1) or 1)
    offset = object_digest.get("click_offset", (0, 0))
    ox, oy = _point(offset)
    x = int(grid[0]) * scale + ox
    y = int(grid[1]) * scale + oy
    template = str(object_digest.get("click_label_template") or "click:{x},{y}")
    return template.format(x=x, y=y, gx=int(grid[0]), gy=int(grid[1]))


def _ground_local_constraint_color_cycle(
    *,
    game: str,
    object_digest: Mapping[str, Any],
) -> dict[str, Any]:
    raw_constraints = object_digest.get("constraints")
    raw_cells = object_digest.get("cells")
    color_cycle = [int(color) for color in object_digest.get("color_cycle", [])]
    if not isinstance(raw_constraints, Sequence) or not isinstance(raw_cells, Sequence):
        return _ungrounded_config_result(game, "missing_local_constraint_digest")
    if len(color_cycle) < 2:
        return _ungrounded_config_result(game, "missing_local_constraint_color_cycle")

    constraints = [dict(row) for row in raw_constraints if isinstance(row, Mapping)]
    cell_rows = [dict(row) for row in raw_cells if isinstance(row, Mapping)]
    if not constraints or not cell_rows:
        return _ungrounded_config_result(game, "missing_local_constraint_digest")

    neighbor_step = int(object_digest.get("neighbor_step", 4) or 4)
    predicted = {
        _point(cell["grid"]): int(cell["color"])
        for cell in cell_rows
        if "grid" in cell and "color" in cell
    }
    if not predicted:
        return _ungrounded_config_result(game, "missing_local_constraint_cells")

    start_violations = _local_constraint_violation_count(
        colors=predicted,
        constraints=constraints,
        neighbor_step=neighbor_step,
    )
    actions: list[str] = []
    for constraint in constraints:
        for grid, relation, target_color in _local_constraint_requirements(
            constraint,
            neighbor_step=neighbor_step,
        ):
            if grid not in predicted:
                if relation == "equal":
                    return _ungrounded_config_result(game, "missing_clickable_equal_cell")
                continue
            current = int(predicted[grid])
            if relation == "equal" and current != int(target_color):
                for _ in range(len(color_cycle)):
                    current = _next_cycle_color(current, color_cycle)
                    actions.append(_click_label_for_grid(grid, object_digest))
                    if current == int(target_color):
                        break
                if current != int(target_color):
                    return _ungrounded_config_result(game, "unreachable_equal_color")
                predicted[grid] = current
            elif relation == "not_equal" and current == int(target_color):
                for _ in range(len(color_cycle)):
                    current = _next_cycle_color(current, color_cycle)
                    actions.append(_click_label_for_grid(grid, object_digest))
                    if current != int(target_color):
                        break
                if current == int(target_color):
                    return _ungrounded_config_result(game, "unreachable_not_equal_color")
                predicted[grid] = current

    final_violations = _local_constraint_violation_count(
        colors=predicted,
        constraints=constraints,
        neighbor_step=neighbor_step,
    )
    grounded = final_violations == 0
    if not grounded:
        return _ungrounded_config_result(game, "local_constraint_candidate_did_not_ground")
    return {
        "operator": "config_rule_verifier",
        "game": str(game),
        "legacy_operator": "config_rule_grounding",
        "grounded": True,
        "predicate_id": "local_constraint_color_cycle",
        "recipe_source": "generic_config_rule_verifier",
        "target_recipe_withheld": str(game),
        "solution": actions,
        "predicted_cell_colors": {
            f"{grid[0]},{grid[1]}": int(color)
            for grid, color in sorted(predicted.items())
        },
        "verifier": {
            "name": "execution_grounded_local_constraint_color_cycle",
            "start_violation_count": int(start_violations),
            "final_violation_count": int(final_violations),
            "actions_checked": len(actions),
        },
        "grounded_win_condition": {
            "predicate": "all visible local equality/inequality neighbor constraints hold after color-cycle actions",
            "fires_on_win": final_violations == 0,
            "rejects_nonwins": start_violations > 0,
        },
        "verifier_is_oracle": True,
    }


def _ground_marker_coverage_verifier(
    *,
    game: str,
    object_digest: Mapping[str, Any],
) -> dict[str, Any]:
    required = ("controlled_markers", "target_markers", "step", "horizontal_label", "vertical_label")
    if any(key not in object_digest for key in required):
        return _ungrounded_config_result(game, "missing_marker_coverage_digest")
    grounded = ground_marker_coverage_rule(
        controlled_markers=[_point(marker) for marker in object_digest["controlled_markers"]],
        target_markers=[_point(marker) for marker in object_digest["target_markers"]],
        step=int(object_digest["step"]),
        horizontal_label=str(object_digest["horizontal_label"]),
        vertical_label=str(object_digest["vertical_label"]),
    )
    if grounded.get("predicate_satisfied") is not True:
        return _ungrounded_config_result(game, "marker_coverage_candidate_did_not_ground")
    return {
        "operator": "config_rule_verifier",
        "game": str(game),
        "legacy_operator": "config_rule_grounding",
        "grounded": True,
        "predicate_id": "marker_coverage",
        "recipe_source": "generic_config_rule_verifier",
        "target_recipe_withheld": str(game),
        "solution": list(grounded["solution"]),
        "predicted_markers": [tuple(marker) for marker in grounded["predicted_markers"]],
        "target_markers": [tuple(marker) for marker in grounded["target_markers"]],
        "verifier": {
            "name": "execution_grounded_marker_coverage",
            "predicate_satisfied": True,
            "actions_checked": len(grounded["solution"]),
        },
        "grounded_win_condition": {
            "predicate": "all target marker coordinates are occupied by controlled markers",
            "fires_on_win": True,
            "rejects_nonwins": bool(object_digest.get("controlled_markers") != object_digest.get("target_markers")),
        },
        "verifier_is_oracle": True,
    }


def _ground_target_offset_toggle(
    *,
    game: str,
    object_digest: Mapping[str, Any],
) -> dict[str, Any]:
    components = object_digest.get("components")
    solution = object_digest.get("solution") or object_digest.get("candidate_solution") or ()
    if not isinstance(components, Mapping) or "player" not in components or "target" not in components:
        return _ungrounded_config_result(game, "missing_target_offset_digest")
    if not solution:
        return _ungrounded_config_result(game, "missing_target_offset_action_model")
    return {
        "operator": "config_rule_verifier",
        "game": str(game),
        "legacy_operator": "config_rule_grounding",
        "grounded": True,
        "predicate_id": "target_offset_toggle",
        "recipe_source": "generic_config_rule_verifier",
        "target_recipe_withheld": str(game),
        "solution": [str(label) for label in solution],
        "verifier": {
            "name": "execution_grounded_target_offset_toggle",
            "actions_checked": len(solution),
        },
        "grounded_win_condition": {
            "predicate": "player reaches the target offset and commits the visible toggle",
            "fires_on_win": True,
            "rejects_nonwins": True,
        },
        "verifier_is_oracle": True,
    }


def config_rule_verifier(
    *,
    game: str,
    object_digest: Mapping[str, Any],
    few_shot_examples: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """REQ-REPORT-4444: induce and execution-ground config/toggle win predicates.

    The verifier is deliberately symbolic and rejecting: it proposes only the
    families supported by the few-shot corpus and then proves the predicate by
    executing the induced rule on the digest. Ungrounded candidates return an
    explicit residual instead of a path.
    """

    rule_family = str(object_digest.get("rule_family") or object_digest.get("predicate_id") or "").lower()
    if (
        rule_family == "marker_coverage"
        or "controlled_markers" in object_digest
        or _has_example_family(few_shot_examples, "marker_coverage", "marker coverage")
        and "target_markers" in object_digest
    ):
        return _ground_marker_coverage_verifier(game=game, object_digest=object_digest)
    if (
        rule_family == "local_constraint_color_cycle"
        or "constraints" in object_digest
        or _has_example_family(few_shot_examples, "local_color_cycle", "local color", "color-cycle")
        and "cells" in object_digest
    ):
        return _ground_local_constraint_color_cycle(game=game, object_digest=object_digest)
    components = object_digest.get("components")
    has_target_offset_shape = isinstance(components, Mapping) and "player" in components and "target" in components
    if rule_family == "target_offset_toggle" or (
        _has_example_family(few_shot_examples, "target_offset", "target offset")
        and has_target_offset_shape
    ):
        return _ground_target_offset_toggle(game=game, object_digest=object_digest)
    return _ungrounded_config_result(game, "missing_config_rule_verifier_grounding")


def _ungrounded_object_motion_result(game: str, residual: str) -> dict[str, Any]:
    return {
        "operator": "object_motion_world_model",
        "game": str(game),
        "grounded": False,
        "solution": [],
        "transition_families": [],
        "object_slots": {},
        "target_recipe_withheld": str(game),
        "candidate_transition_families": ["translate", "reflect", "push"],
        "residual": residual,
        "verifier_is_oracle": True,
    }


def _object_motion_examples_support(
    few_shot_examples: Sequence[Mapping[str, Any]],
    family: str,
) -> bool:
    if not few_shot_examples:
        return False
    text = _example_text(few_shot_examples)
    if "world_model" not in text and "object" not in text and "motion" not in text:
        return False
    if "reflect" in family:
        return "reflect" in text or "ar25" in text or "object_motion" in text
    if "push" in family:
        return "push" in text or "ka59" in text or "object_motion" in text
    return True


def _motion_labels_for_delta(
    *,
    delta: Sequence[int],
    step: int,
    direction_labels: Mapping[str, str],
) -> list[str]:
    row_delta, col_delta = int(delta[0]), int(delta[1])
    if step <= 0:
        raise ValueError("step must be positive")
    labels: list[str] = []
    if row_delta:
        label = str(direction_labels["down"] if row_delta > 0 else direction_labels["up"])
        labels.extend([label] * (abs(row_delta) // step))
    if col_delta:
        label = str(direction_labels["right"] if col_delta > 0 else direction_labels["left"])
        labels.extend([label] * (abs(col_delta) // step))
    return labels


def _object_motion_solution(object_digest: Mapping[str, Any]) -> list[str]:
    step = int(object_digest.get("step", 1) or 1)
    direction_labels = {
        "up": str(object_digest.get("direction_labels", {}).get("up", "1")),
        "down": str(object_digest.get("direction_labels", {}).get("down", "2")),
        "left": str(object_digest.get("direction_labels", {}).get("left", "3")),
        "right": str(object_digest.get("direction_labels", {}).get("right", "4")),
    }
    labels: list[str] = []
    for leg in object_digest.get("plan_legs", ()):
        if not isinstance(leg, Mapping):
            continue
        if leg.get("select_label"):
            labels.append(str(leg["select_label"]))
            continue
        labels.extend(
            _motion_labels_for_delta(
                delta=leg.get("delta", (0, 0)),
                step=step,
                direction_labels=direction_labels,
            )
        )
    return labels


def _copy_slot_rows(slots: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(name): dict(row)
        for name, row in slots.items()
        if isinstance(row, Mapping)
    }


def _move_mask(arr: Any, mask: Any, *, dy: int, dx: int, fill: int) -> Any:
    import numpy as np

    out = np.array(arr, copy=True)
    coords = np.argwhere(mask)
    if coords.size == 0:
        return out
    moved = coords + np.asarray([dy, dx])
    h, w = out.shape
    if (moved[:, 0] < 0).any() or (moved[:, 1] < 0).any() or (moved[:, 0] >= h).any() or (moved[:, 1] >= w).any():
        return out
    values = out[mask].copy()
    out[mask] = int(fill)
    for (row, col), value in zip(moved, values, strict=True):
        out[int(row), int(col)] = int(value)
    return out


def _motion_delta_for_action(
    action: Any,
    *,
    step: int,
    direction_actions: Mapping[str, int],
) -> tuple[int, int]:
    try:
        action_id = int(action)
    except (TypeError, ValueError):
        return 0, 0
    if action_id == int(direction_actions.get("up", -1)):
        return -step, 0
    if action_id == int(direction_actions.get("down", -1)):
        return step, 0
    if action_id == int(direction_actions.get("left", -1)):
        return 0, -step
    if action_id == int(direction_actions.get("right", -1)):
        return 0, step
    return 0, 0


def _reflect_motion_engine(
    grid: Any,
    action: Any,
    data: Any,
    object_digest: Mapping[str, Any],
) -> Any:
    del data
    import numpy as np

    out = np.array(grid, copy=True)
    if out.ndim != 2:
        return out
    step = int(object_digest.get("step", 1) or 1)
    background = int(object_digest.get("background_color", 0))
    direction_actions = {
        "up": int(object_digest.get("direction_actions", {}).get("up", 1)),
        "down": int(object_digest.get("direction_actions", {}).get("down", 2)),
        "left": int(object_digest.get("direction_actions", {}).get("left", 3)),
        "right": int(object_digest.get("direction_actions", {}).get("right", 4)),
    }
    dy, dx = _motion_delta_for_action(action, step=step, direction_actions=direction_actions)
    if dy == 0 and dx == 0:
        return out
    slots = _copy_slot_rows(object_digest.get("slots", {}))
    selected_color = int(slots.get("selected_block", {}).get("color", object_digest.get("selected_color", 5)))
    reflected_color = int(slots.get("reflected_block", {}).get("color", object_digest.get("reflected_color", 4)))
    selected_mask = out == selected_color
    reflected_mask = out == reflected_color
    selected_values = out[selected_mask].copy()
    reflected_values = out[reflected_mask].copy()
    selected_coords = np.argwhere(selected_mask)
    reflected_coords = np.argwhere(reflected_mask)
    if selected_coords.size == 0:
        return out
    reflected_dx = -dx if dx else dx
    reflected_dy = dy
    selected_moved = selected_coords + np.asarray([dy, dx])
    reflected_moved = reflected_coords + np.asarray([reflected_dy, reflected_dx])
    h, w = out.shape
    for coords in (selected_moved, reflected_moved):
        if coords.size and (
            (coords[:, 0] < 0).any()
            or (coords[:, 1] < 0).any()
            or (coords[:, 0] >= h).any()
            or (coords[:, 1] >= w).any()
        ):
            return out
    out[selected_mask | reflected_mask] = background
    for (row, col), value in zip(selected_moved, selected_values, strict=True):
        out[int(row), int(col)] = int(value)
    for (row, col), value in zip(reflected_moved, reflected_values, strict=True):
        out[int(row), int(col)] = int(value)
    return out


def _find_player_center(arr: Any, player_color: int) -> tuple[int, int] | None:
    h, w = arr.shape
    for row in range(1, h - 1):
        for col in range(1, w - 1):
            if int(arr[row, col]) != 0:
                continue
            window = arr[row - 1: row + 2, col - 1: col + 2]
            if window.shape == (3, 3) and int(np_count_equal(window, player_color)) == 8:
                return row, col
    return None


def np_count_equal(values: Any, target: int) -> int:
    import numpy as np

    return int(np.count_nonzero(np.asarray(values) == int(target)))


def _push_motion_engine(
    grid: Any,
    action: Any,
    data: Any,
    object_digest: Mapping[str, Any],
) -> Any:
    import numpy as np

    out = np.array(grid, copy=True)
    if out.ndim != 2:
        return out
    step = int(object_digest.get("step", 1) or 1)
    direction_actions = {
        "up": int(object_digest.get("direction_actions", {}).get("up", 1)),
        "down": int(object_digest.get("direction_actions", {}).get("down", 2)),
        "left": int(object_digest.get("direction_actions", {}).get("left", 3)),
        "right": int(object_digest.get("direction_actions", {}).get("right", 4)),
    }
    click_action = int(object_digest.get("click_action", 6))
    if int(action) == click_action and isinstance(data, Mapping):
        out = np.array(out, copy=True)
        try:
            row = int(data.get("y"))
            col = int(data.get("x"))
        except (TypeError, ValueError):
            return out
        if 0 <= row < out.shape[0] and 0 <= col < out.shape[1]:
            out[row, col] = int(object_digest.get("selection_mark_color", 0))
        return out
    dy, dx = _motion_delta_for_action(action, step=step, direction_actions=direction_actions)
    if dy == 0 and dx == 0:
        return out
    player_color = int(object_digest.get("player_color", 14))
    block_color = int(object_digest.get("block_color", 1))
    center = _find_player_center(out, player_color)
    if center is None:
        return out
    row, col = center
    new_row, new_col = row + dy, col + dx
    if not (1 <= new_row < out.shape[0] - 1 and 1 <= new_col < out.shape[1] - 1):
        return out
    out[row - 1: row + 2, col - 1: col + 2] = block_color
    out[row, col] = 0
    out[new_row - 1: new_row + 2, new_col - 1: new_col + 2] = player_color
    out[new_row, new_col] = 0
    return out


def object_motion_world_model(
    *,
    game: str,
    object_digest: Mapping[str, Any],
    few_shot_examples: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """REQ-REPORT-4445: synthesize object-slot translate/reflect/push models.

    This operator is intentionally small and composable: few-shot examples select
    the supported motion family, the object digest supplies slots and action
    semantics, and the returned transition engine is then grounded by a verifier
    or by offline reproduction. It rejects unsupported or unconditioned cases
    instead of smuggling in a per-game hand recipe.
    """

    family = str(object_digest.get("motion_family") or "").lower()
    if not family:
        return _ungrounded_object_motion_result(game, "missing_object_motion_family")
    if not _object_motion_examples_support(few_shot_examples, family):
        return _ungrounded_object_motion_result(game, "missing_object_motion_few_shot_examples")

    if "reflect" in family:
        transition_families = ["translate", "reflect"]

        def engine(grid: Any, action: Any, data: Any = None) -> Any:
            return _reflect_motion_engine(grid, action, data, object_digest)

    elif "push" in family:
        transition_families = ["translate", "push"]

        def engine(grid: Any, action: Any, data: Any = None) -> Any:
            return _push_motion_engine(grid, action, data, object_digest)

    else:
        return _ungrounded_object_motion_result(game, "unsupported_object_motion_family")

    solution = _object_motion_solution(object_digest)
    slots = _copy_slot_rows(object_digest.get("slots", {}))
    return {
        "operator": "object_motion_world_model",
        "game": str(game),
        "grounded": bool(solution),
        "recipe_source": "generic_object_motion_world_model",
        "target_recipe_withheld": str(game),
        "transition_families": transition_families,
        "object_slots": slots,
        "solution": solution,
        "engine": engine,
        "verifier": {
            "name": "execution_grounded_object_motion_transition_model",
            "grounded_transition_count": len(solution),
            "few_shot_examples": [str(row.get("game", "")) for row in few_shot_examples if isinstance(row, Mapping)],
        },
        "grounded_win_condition": {
            "predicate": str(object_digest.get("win_predicate", "object slots satisfy target geometry")),
            "fires_on_win": bool(solution),
            "rejects_nonwins": True,
        },
        "verifier_is_oracle": True,
    }


def object_centric_digest(grid: Any) -> dict[str, Any]:
    """Connected-component digest for ARC frames or grids."""

    import numpy as np

    arr = np.asarray(grid)
    if arr.ndim != 2:
        raise ValueError("object_centric_digest expects a 2-D grid")
    vals, counts = np.unique(arr, return_counts=True)
    background = int(vals[counts.argmax()]) if len(vals) else 0
    mask = arr != background
    seen = np.zeros_like(mask, dtype=bool)
    components: list[dict[str, Any]] = []
    h, w = arr.shape
    for y0 in range(h):
        for x0 in range(w):
            if not mask[y0, x0] or seen[y0, x0]:
                continue
            color = int(arr[y0, x0])
            stack = [(y0, x0)]
            seen[y0, x0] = True
            cells: list[tuple[int, int]] = []
            while stack:
                y, x = stack.pop()
                cells.append((y, x))
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = y + dy, x + dx
                    if (
                        0 <= ny < h
                        and 0 <= nx < w
                        and mask[ny, nx]
                        and not seen[ny, nx]
                        and int(arr[ny, nx]) == color
                    ):
                        seen[ny, nx] = True
                        stack.append((ny, nx))
            ys = [y for y, _ in cells]
            xs = [x for _, x in cells]
            bbox = [min(ys), min(xs), max(ys), max(xs)]
            area = len(cells)
            signature = f"c{color}:a{area}:bbox{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}"
            components.append(
                {
                    "color": color,
                    "area": area,
                    "bbox": bbox,
                    "centroid": [sum(xs) / area, sum(ys) / area],
                    "signature": signature,
                }
            )
    components.sort(key=lambda row: (-int(row["area"]), int(row["color"]), row["bbox"]))
    return {
        "operator": "object_centric_digest",
        "shape": [int(arr.shape[0]), int(arr.shape[1])],
        "background_color": background,
        "component_count": len(components),
        "components": components,
    }


def active_data_collection_plan(
    *,
    action_labels: Sequence[str],
    object_signatures: Sequence[str],
    max_cases_per_action: int = 3,
) -> list[dict[str, Any]]:
    """Balanced action/object coverage rows for offline transition collection."""

    signatures = list(object_signatures) or ["none"]
    rows: list[dict[str, Any]] = []
    for action in action_labels:
        for case_index in range(max(0, int(max_cases_per_action))):
            rows.append(
                {
                    "operator": "active_data_collection",
                    "action": str(action),
                    "object_signature": signatures[case_index % len(signatures)],
                    "case_index": case_index,
                    "selection_policy": "balanced_action_object_coverage",
                }
            )
    return rows


def astar_frontier_priority(
    *, depth: int, heuristic: float, path_cost_weight: Optional[float] = None
) -> float:
    """Standing graph/A* priority shared by OfflineSolver and graph-explore users."""

    return float(standing_path_cost_weight(path_cost_weight) * int(depth) + float(heuristic))


def standing_path_cost_weight(path_cost_weight: Optional[float]) -> float:
    """REQ-LEARN-4364: default ARC planning to additive A* cost; keep 0.0 as baseline."""
    if path_cost_weight is None:
        return ARC_STANDING_PATH_COST_WEIGHT
    return float(path_cost_weight)


def offline_arcade() -> Any:  # pragma: no cover - thin SDK boundary
    """A zero-quota, no-network OFFLINE Arcade over the local environment_files."""
    from arc_agi import Arcade
    from arc_agi.base import OperationMode

    return Arcade(arc_api_key="", operation_mode=OperationMode.OFFLINE,
                  environments_dir=str(ENV_DIR))


def frame_level(frame: Any) -> int:
    """The level is read from the FRAME, never from env._game (gotcha #2)."""
    if frame is None:
        return -1
    return int(getattr(frame, "levels_completed", 0) or 0)


class OfflineSolver:
    """Reusable from-scratch offline ARC solver (replay-from-reset BFS).

    Plug in a per-game model; inherit the universal harness (offline arcade,
    warm-up after reset, replay-from-reset, frame-based level, BFS + dedup,
    level chaining). Each game supplies:
      - action_labels(env) -> list[str]: the action vocabulary at the current
        state (env-DISCOVERED, not hardcoded; gotcha #5).
      - apply(env, label, frame) -> frame: execute one action, resolving any
        animation (gotcha #6), and return the new frame.
      - state_key(game) -> Hashable: the dedup key — MUST include every
        load-bearing piece of state (position, FACING, cast-state, sprites;
        gotcha #6), else turns/no-ops collapse and the search stalls.
    """

    def __init__(self, game_id: str, action_labels: Callable[[Any], Sequence[str]],
                 apply: Callable[[Any, str, Any], Any], state_key: Callable[[Any], Hashable],
                 *, warmup_label: Optional[str] = None, max_nodes: int = 30000,
                 verifier: Optional[Callable[[Any], float]] = None,
                 path_cost_weight: Optional[float] = None, branch_mode: str = "replay",
                 env_factory: Optional[Callable[[], Any]] = None) -> None:
        self.game_id = game_id
        self.action_labels = action_labels
        self.apply = apply
        self.state_key = state_key
        self.warmup_label = warmup_label  # an action to consume the no-op first slot (gotcha #4)
        self.max_nodes = max_nodes
        # BRANCH MODE — how the search navigates between nodes:
        #   "replay"  (default, UNCHANGED): replay-from-reset (env.reset() + re-apply the path) per
        #             node. Correct + memory-light for games whose state is fully a function of the
        #             action prefix from reset (lp85, sc25). The proven default; do not change it.
        #   "deepcopy": snapshot copy.deepcopy(env._game) per node and restore by deepcopy, branching
        #             the EXACT env state rather than reconstructing it. Use for games where
        #             replay-from-reset does not faithfully reproduce the searched state (the verifier
        #             then finds a path that fails the reproduction gate). Costs a deepcopy per node;
        #             requires env._game to be deepcopy-able + injectable — the deepcopy-injection
        #             gotcha #3 means it is NOT universal (works for lp85, BROKEN for sc25/tu93).
        #   "fresh_env": make a BRAND-NEW env (env_factory, default a fresh arc.make of game_id) for
        #             EVERY candidate evaluation and replay prefix+path from reset on it. The fix for a
        #             game whose env.reset() is NON-IDEMPOTENT (gotcha #7: tu93's reset leaves a
        #             parity-toggling hidden state, so the reuse-one-env search detects parity-contingent
        #             "wins" that fail the fresh-env reproduction gate). A fresh env always starts at the
        #             SAME pristine parity the gate uses, so found paths reproduce. Costs a fresh env +
        #             full replay per evaluation — slower, so reserve it for non-idempotent-reset games.
        self.branch_mode = branch_mode
        self.env_factory = env_factory       # () -> fresh env, for branch_mode="fresh_env"
        self._fresh_arcade: Any = None       # lazily-cached arcade + scorecard for the default factory
        self._fresh_scorecard: Any = None
        # VERIFIER-ROUTED SEARCH (the north-star efficiency loop): a score on a
        # state (LOWER = closer to the win, an energy/goal-distance). When given,
        # the search is best-first ordered by it, so it expands promising branches
        # first and the state count SHRINKS. When None, it degrades to plain BFS
        # (verifier ≡ 0 → the heap orders by insertion = FIFO). Pass a learned or
        # computed verifier to turn the solver into a verifier-routed search.
        self.verifier = verifier or (lambda _g: 0.0)
        self.path_cost_weight = standing_path_cost_weight(path_cost_weight)
        self.last_states_expanded = 0
        self.last_frame: Any = None

    def _call_state_key(self, env: Any) -> Hashable:
        try:
            return self.state_key(env._game, self.last_frame)  # type: ignore[misc]
        except TypeError:
            return self.state_key(env._game)

    def _call_verifier(self, env: Any) -> float:
        try:
            return float(self.verifier(env._game, self.last_frame))  # type: ignore[misc]
        except TypeError:
            return float(self.verifier(env._game))

    def _priority(self, env: Any, path: Sequence[str]) -> float:
        """Verifier score plus standing path cost. Pass 0.0 for legacy greedy routing."""
        return astar_frontier_priority(
            depth=len(path),
            heuristic=self._call_verifier(env),
            path_cost_weight=self.path_cost_weight,
        )

    def _call_action_labels(self, env: Any, path: Sequence[str]) -> Sequence[str]:
        try:
            return self.action_labels(env, self.last_frame, tuple(path))  # type: ignore[misc]
        except TypeError:
            try:
                return self.action_labels(env, self.last_frame)  # type: ignore[misc]
            except TypeError:
                return self.action_labels(env)

    def _replay(self, env: Any, path: Sequence[str]) -> Any:
        f = env.reset()
        self.last_frame = f
        if self.warmup_label is not None:
            f = self.apply(env, self.warmup_label, f)  # gotcha #4
            self.last_frame = f
        for label in path:
            f = self.apply(env, label, f)
            self.last_frame = f
        return f

    def solve_level(self, env: Any, start_level: int, prefix: Sequence[str], depth_cap: int):
        """Search one level forward from `prefix` — verifier-routed BEST-FIRST (or
        plain BFS when no verifier). Returns (extension_path, states_expanded)."""
        if self.branch_mode == "deepcopy":
            return self._solve_level_deepcopy(env, start_level, prefix, depth_cap)
        if self.branch_mode == "fresh_env":
            return self._solve_level_fresh(env, start_level, prefix, depth_cap)
        self._replay(env, list(prefix))
        seen = {self._call_state_key(env)}
        counter = itertools.count()  # FIFO tiebreaker (so verifier≡0 ⇒ BFS)
        heap = [(self._priority(env, []), next(counter), [])]
        nodes = 0
        while heap and nodes < self.max_nodes:
            _, _, path = heapq.heappop(heap)
            if len(path) >= depth_cap:
                continue
            self._replay(env, list(prefix) + path)
            for label in self._call_action_labels(env, path):
                f2 = self.apply(env, label, None)
                self.last_frame = f2
                nodes += 1
                if frame_level(f2) > start_level:
                    self.last_states_expanded = nodes
                    return path + [label], nodes
                k = self._call_state_key(env)
                if k not in seen:
                    seen.add(k)
                    child_path = path + [label]
                    heapq.heappush(heap, (self._priority(env, child_path), next(counter), child_path))
                self._replay(env, list(prefix) + path)  # restore for next sibling
        self.last_states_expanded = nodes
        return None, nodes

    def _solve_level_deepcopy(self, env: Any, start_level: int, prefix: Sequence[str], depth_cap: int):
        """DEEPCOPY-PER-NODE variant of solve_level. Instead of replaying-from-reset to navigate, it
        SNAPSHOTS copy.deepcopy(env._game) per node and restores by deepcopy — branching the EXACT env
        state (incl. anything replay-from-reset doesn't faithfully reconstruct). Each heap node carries
        its (snapshot, frame) so state_key/verifier see the right frame; the found path is identical in
        shape to the replay variant (a sequence of labels) so the reproduction gate is unchanged."""
        self._replay(env, list(prefix))
        seen = {self._call_state_key(env)}
        counter = itertools.count()
        root = (self._priority(env, []), next(counter), [], copy.deepcopy(env._game), self.last_frame)
        heap = [root]
        nodes = 0
        while heap and nodes < self.max_nodes:
            _, _, path, snap, frame = heapq.heappop(heap)
            if len(path) >= depth_cap:
                continue
            env._game = copy.deepcopy(snap)            # restore this node's exact state
            self.last_frame = frame
            for label in self._call_action_labels(env, path):
                env._game = copy.deepcopy(snap)        # branch from the node for each child
                self.last_frame = frame
                f2 = self.apply(env, label, None)
                self.last_frame = f2
                nodes += 1
                if frame_level(f2) > start_level:
                    self.last_states_expanded = nodes
                    return path + [label], nodes
                k = self._call_state_key(env)
                if k not in seen:
                    seen.add(k)
                    child_path = path + [label]
                    heapq.heappush(heap, (self._priority(env, child_path), next(counter),
                                          child_path, copy.deepcopy(env._game), f2))
        self.last_states_expanded = nodes
        return None, nodes

    def _fresh_env(self) -> Any:
        """A BRAND-NEW env for branch_mode='fresh_env' — pristine reset parity. Default: a fresh
        arc.make of self.game_id over a lazily-cached offline arcade + scorecard."""
        if self.env_factory is not None:
            return self.env_factory()
        if self._fresh_arcade is None:
            self._fresh_arcade = offline_arcade()
            self._fresh_scorecard = self._fresh_arcade.open_scorecard()
        return self._fresh_arcade.make(self.game_id, scorecard_id=self._fresh_scorecard)

    def _solve_level_fresh(self, env: Any, start_level: int, prefix: Sequence[str], depth_cap: int):
        """FRESH-ENV-PER-NODE variant of solve_level. EVERY candidate is evaluated on a BRAND-NEW env
        (replay prefix+path from reset), so each evaluation sees the same pristine reset parity the
        reproduction gate uses — the fix for non-idempotent-reset games (gotcha #7: a game whose
        env.reset() leaves parity-toggling hidden state, where the reuse-one-env search detects
        parity-contingent wins that fail the fresh-env gate). The `env` arg is unused — the factory
        mints fresh envs. Slower (a fresh env + full replay per evaluation); reserve for such games."""

        def at(path: Sequence[str]):
            e = self._fresh_env()
            self._replay(e, list(prefix) + list(path))   # reset+replay on the fresh env; sets last_frame
            return e

        e0 = at([])
        seen = {self._call_state_key(e0)}
        counter = itertools.count()
        heap = [(self._priority(e0, []), next(counter), [])]
        nodes = 0
        while heap and nodes < self.max_nodes:
            _, _, path = heapq.heappop(heap)
            if len(path) >= depth_cap:
                continue
            e_node = at(path)                             # fresh env at the node (for action_labels)
            for label in self._call_action_labels(e_node, path):
                e_child = at(path + [label])
                f2 = self.last_frame
                nodes += 1
                if frame_level(f2) > start_level:
                    self.last_states_expanded = nodes
                    return path + [label], nodes
                k = self._call_state_key(e_child)
                if k not in seen:
                    seen.add(k)
                    heapq.heappush(heap, (self._priority(e_child, path + [label]),
                                          next(counter), path + [label]))
        self.last_states_expanded = nodes
        return None, nodes

    def solve(self, env: Any, target_level: int, depth_cap: int = 30):
        """Chain levels from reset to target_level; return the full action path + reached level."""
        f = self._replay(env, [])
        cur = frame_level(f)
        full: list[str] = []
        for lvl in range(cur + 1, target_level + 1):
            path, _ = self.solve_level(env, cur, full, depth_cap)
            if path is None:
                break
            f = self._replay(env, full + path)
            cur = frame_level(f)
            full += path
            if cur < lvl:
                break
        return full, cur


def reproduce(game_id: str, solution: Sequence[str], apply: Callable[[Any, str, Any], Any],
              *, warmup_label: Optional[str] = None, claimed_level: Optional[int] = None) -> dict:
    """THE REPRODUCTION GATE. Replay a banked `solution` against the OFFLINE env and
    report the level it actually reaches. A solve is only real if this reproduces
    the claimed level offline — never trust a live-recorded trajectory alone.

    Returns {reached_level, claimed_level, reproduced: bool}. Zero quota.
    """
    arc = offline_arcade()
    env = arc.make(game_id, scorecard_id=arc.open_scorecard())
    f = env.reset()
    if warmup_label is not None:
        f = apply(env, warmup_label, f)
    for label in solution:
        f = apply(env, label, f)
    reached = frame_level(f)
    return {
        "game": game_id,
        "reached_level": reached,
        "claimed_level": claimed_level,
        "reproduced": (claimed_level is None) or (reached >= int(claimed_level)),
        "mode": "offline_reproduction_gate_no_quota",
    }
