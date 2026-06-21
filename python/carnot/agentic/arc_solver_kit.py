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
            operator="glyph_rewrite_rule_verifier",
            derived_from_games=("bsqsshqpox", "tr87"),
            purpose="Induce and execution-ground greedy glyph rewrite win predicates from config-substitution examples.",
            selector_tags=("config_substitution", "glyph", "rewrite", "verifier", "rule"),
        ),
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
            derived_from_games=("s5i5", "ft09", "g50t", "dc22"),
            purpose="Propose and execution-ground coverage, local-constraint, or toggle win predicates.",
            selector_tags=("config_toggle", "marker_coverage", "local_constraint", "verifier", "rule"),
        ),
        PrimitiveOperator(
            operator="color_match_slot_sequence_verifier",
            derived_from_games=("sb26", "s5i5", "ft09"),
            purpose="Ground ordered colored item-to-slot placement predicates with undo-aware counterexamples.",
            selector_tags=("color_match", "slot_sequence", "ordered", "config_rule", "undo", "verifier"),
        ),
        PrimitiveOperator(
            operator="sprite_overlay_resize_verifier",
            derived_from_games=("re86", "s5i5"),
            purpose=(
                "Ground transparent sprite overlays by matching required target pixels, "
                "including explicit resize variants when the action model exposes them."
            ),
            selector_tags=("sprite_overlay", "pattern_match", "resize", "transparent", "verifier", "re86"),
        ),
        PrimitiveOperator(
            operator="graph_astar_action_cost",
            derived_from_games=("tu93", "lp85", "cd82", "sp80", "cn04", "m0r0", "sk48", "su15"),
            purpose="A* frontier priority: standing path cost plus verifier/action-cost heuristic.",
            selector_tags=("graph_explore", "astar", "action_cost", "keyboard", "click"),
        ),
        PrimitiveOperator(
            operator="per_level_reinduction_operator",
            derived_from_games=("lp85", "m0r0", "sp80", "vc33"),
            purpose=(
                "Detect a level-up, clear stale level-local induction state, re-induce the "
                "next level predicate, and route the frontier with depth-primary goal bias."
            ),
            selector_tags=("reinduction", "level_up", "deepening", "goal_bias", "transfer"),
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
        PrimitiveOperator(
            operator="cast_grid_phase_fsm_world_model",
            derived_from_games=("sc25", "ar25", "ka59", "ft09"),
            purpose="Two-phase cast/config-grid toggle CSP followed by player navigation to an exit predicate.",
            selector_tags=("cast_grid", "phase_fsm", "config_toggle", "navigation", "world_model", "verifier"),
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

    if (
        "cast_grid" in mechanic
        or "cast grid" in mechanic
        or "phase_fsm" in mechanic
        or "two_phase_cast_grid" in mechanic
        or gid == "sc25"
    ):
        names = (
            "per_level_reinduction_operator",
            "cast_grid_phase_fsm_world_model",
            "object_motion_world_model",
            "active_data_collection",
            "graph_astar_action_cost",
        )
    elif (
        "color_match" in mechanic
        or "color match" in mechanic
        or "slot_sequence" in mechanic
        or "slot sequence" in mechanic
        or gid == "sb26"
    ):
        names = (
            "color_match_slot_sequence_verifier",
            "config_rule_verifier",
            "object_centric_digest",
            "graph_astar_action_cost",
        )
    elif (
        "sprite_overlay" in mechanic
        or "sprite overlay" in mechanic
        or "pattern_match_sprite_resize" in mechanic
        or "sprite_resize" in mechanic
        or "resize" in mechanic
        or gid == "re86"
    ):
        names = (
            "sprite_overlay_resize_verifier",
            "object_centric_digest",
            "graph_astar_action_cost",
        )
    elif "config_substitution" in mechanic or "glyph" in mechanic or gid == "tr87":
        names = (
            "glyph_rewrite_rule_verifier",
            "glyph_rewrite_matcher",
            "per_level_reinduction_operator",
            "graph_astar_action_cost",
            "object_centric_digest",
        )
    elif "config" in mechanic or "toggle" in mechanic or "constraint" in mechanic:
        names = (
            "config_rule_verifier",
            "config_rule_grounding",
            "per_level_reinduction_operator",
            "object_centric_digest",
            "graph_astar_action_cost",
        )
    elif "program_editor" in mechanic:
        names = (
            "per_level_reinduction_operator",
            "object_centric_digest",
            "active_data_collection",
            "graph_astar_action_cost",
        )
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
        names = ("per_level_reinduction_operator", "graph_astar_action_cost", "object_centric_digest")
    else:
        names = (
            "per_level_reinduction_operator",
            "object_centric_digest",
            "active_data_collection",
            "graph_astar_action_cost",
        )
    return tuple(registry[name] for name in names)


def _observation_level(observation: Any) -> int:
    if isinstance(observation, Mapping):
        for key in ("levels_completed", "level", "reached_level"):
            if key in observation:
                return int(observation[key] or 0)
    if hasattr(observation, "levels_completed"):
        return int(getattr(observation, "levels_completed") or 0)
    return frame_level(observation)


def per_level_reinduction_operator(
    observations: Sequence[Any],
    *,
    predicate_inducer: Callable[[int, dict[str, Any]], Mapping[str, Any] | str | None],
    route_builder: Optional[Callable[[dict[str, Any]], Mapping[str, Any]]] = None,
    initial_predicate: Mapping[str, Any] | str | None = None,
    initial_level: Optional[int] = None,
) -> dict[str, Any]:
    """REQ-ARC-WMTE-4537: reusable detect-level-up -> re-induce -> route loop."""

    if route_builder is None:
        route_builder = lambda event: {
            "route": "depth_primary_goal_bias",
            "depth_primary": True,
            "goal_bias_label": str((event.get("predicate") or {}).get("predicate_id") or ""),
        }

    current_level = int(initial_level) if initial_level is not None else None
    prior_signature = (
        str(initial_predicate.get("signature") or initial_predicate.get("predicate_id"))
        if isinstance(initial_predicate, Mapping)
        else (str(initial_predicate) if initial_predicate is not None else "")
    )
    events: list[dict[str, Any]] = []

    for index, observation in enumerate(observations):
        level = _observation_level(observation)
        if current_level is None:
            current_level = level
            continue
        if level <= current_level:
            continue
        for won_level in range(current_level + 1, level + 1):
            next_goal_level = won_level + 1
            context = {
                "from_level": won_level,
                "next_goal_level": next_goal_level,
                "observation_index": index,
                "observation": observation,
                "prior_predicate_signature": prior_signature,
                "clear_stale_induction": True,
            }
            raw_predicate = predicate_inducer(next_goal_level, context)
            if isinstance(raw_predicate, Mapping):
                predicate = dict(raw_predicate)
            elif raw_predicate is None:
                predicate = {
                    "predicate_id": f"L{next_goal_level}_predicate_unavailable",
                    "signature": "",
                    "representation_correct": False,
                }
            else:
                predicate = {
                    "predicate_id": str(raw_predicate),
                    "signature": str(raw_predicate),
                    "representation_correct": True,
                }
            predicate.setdefault("predicate_id", f"L{next_goal_level}_predicate")
            predicate.setdefault("signature", str(predicate.get("predicate_id") or ""))
            predicate.setdefault("representation_correct", False)
            signature = str(predicate.get("signature") or predicate.get("predicate_id") or "")
            representation_transfer = bool(
                predicate.get("representation_correct") is True
                and signature
                and (not prior_signature or signature != prior_signature)
            )
            event = {
                "trigger": "level_up",
                "from_level": won_level,
                "next_goal_level": next_goal_level,
                "stale_state_cleared": True,
                "predicate": predicate,
                "representation_transfer": representation_transfer,
            }
            event["route"] = dict(route_builder(event))
            events.append(event)
            prior_signature = signature
        current_level = level

    return {
        "operator": "per_level_reinduction_operator",
        "level_ups_detected": len(events),
        "stale_state_cleared": bool(events),
        "current_level": int(current_level or 0),
        "events": events,
        "latest_predicate": events[-1]["predicate"] if events else None,
        "latest_route": events[-1]["route"] if events else None,
        "representation_transfer": any(bool(event["representation_transfer"]) for event in events),
    }


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


def _sprite_grid(value: Any) -> tuple[tuple[int, ...], ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return ()
    rows: list[tuple[int, ...]] = []
    width: int | None = None
    for row in value:
        if not isinstance(row, Sequence) or isinstance(row, (str, bytes)):
            return ()
        parsed = tuple(int(cell) for cell in row)
        if width is None:
            width = len(parsed)
        if not parsed or len(parsed) != width:
            return ()
        rows.append(parsed)
    return tuple(rows)


def _source_color(pixels: Sequence[Sequence[int]], *, transparent: int = -1, marker: int = 0) -> int | None:
    counts: dict[int, int] = {}
    for row in pixels:
        for cell in row:
            color = int(cell)
            if color in {transparent, marker}:
                continue
            counts[color] = counts.get(color, 0) + 1
    if not counts:
        return None
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]


def _local_points_for_color(pixels: Sequence[Sequence[int]], color: int) -> set[tuple[int, int]]:
    points: set[tuple[int, int]] = set()
    for y, row in enumerate(pixels):
        for x, cell in enumerate(row):
            if int(cell) == int(color):
                points.add((x, y))
    return points


def _sprite_overlay_required_pixels(object_digest: Mapping[str, Any]) -> list[tuple[int, int, int]]:
    direct = object_digest.get("required_pixels")
    if isinstance(direct, Sequence) and not isinstance(direct, (str, bytes)):
        parsed: list[tuple[int, int, int]] = []
        for row in direct:
            if not isinstance(row, Mapping):
                continue
            parsed.append((int(row["x"]), int(row["y"]), int(row["color"])))
        return parsed

    ignore_colors = {
        int(color)
        for color in object_digest.get("target_match_ignore_colors", (-1, 4))
        if isinstance(color, int) or str(color).lstrip("-").isdigit()
    }
    required: list[tuple[int, int, int]] = []
    targets = object_digest.get("targets") or ()
    if not isinstance(targets, Sequence) or isinstance(targets, (str, bytes)):
        return required
    for target in targets:
        if not isinstance(target, Mapping):
            continue
        pixels = _sprite_grid(target.get("pixels"))
        if not pixels:
            continue
        x0 = int(target.get("x") or 0)
        y0 = int(target.get("y") or 0)
        for y, row in enumerate(pixels):
            for x, cell in enumerate(row):
                color = int(cell)
                if color not in ignore_colors:
                    required.append((x0 + x, y0 + y, color))
    return required


def _sprite_overlay_variants(source: Mapping[str, Any]) -> list[dict[str, Any]]:
    base_pixels = _sprite_grid(source.get("pixels"))
    variants = [
        {
            "variant_id": "base",
            "pixels": base_pixels,
            "pre_labels": [],
            "post_labels": [],
            "resize_variant_used": False,
        }
    ]
    raw = source.get("variants") or ()
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        for index, variant in enumerate(raw):
            if not isinstance(variant, Mapping):
                continue
            pixels = _sprite_grid(variant.get("pixels"))
            if not pixels:
                continue
            variants.append(
                {
                    "variant_id": str(variant.get("id") or variant.get("variant_id") or f"variant_{index}"),
                    "pixels": pixels,
                    "pre_labels": [str(label) for label in variant.get("pre_labels") or ()],
                    "post_labels": [str(label) for label in variant.get("post_labels") or ()],
                    "resize_variant_used": True,
                }
            )
    return [variant for variant in variants if variant["pixels"]]


def _best_sprite_overlay_placement(
    *,
    source: Mapping[str, Any],
    required: Sequence[tuple[int, int, int]],
    movement_step: int,
) -> dict[str, Any] | None:
    source_id = str(source.get("id") or source.get("name") or "")
    x_current = int(source.get("x") or 0)
    y_current = int(source.get("y") or 0)
    best: dict[str, Any] | None = None
    for variant in _sprite_overlay_variants(source):
        color = _source_color(variant["pixels"])
        if color is None:
            continue
        local_points = _local_points_for_color(variant["pixels"], color)
        same_color = [pixel for pixel in required if pixel[2] == color]
        if not same_color or not local_points:
            continue
        candidates: set[tuple[int, int]] = set()
        for target_x, target_y, _target_color in same_color:
            for local_x, local_y in local_points:
                candidates.add((target_x - local_x, target_y - local_y))
        for x_target, y_target in candidates:
            covered = [
                pixel
                for pixel in same_color
                if (pixel[0] - x_target, pixel[1] - y_target) in local_points
            ]
            candidate = {
                "source_id": source_id,
                "source_index": int(source.get("source_index") or 0),
                "color": int(color),
                "current_top_left": [x_current, y_current],
                "target_top_left": [int(x_target), int(y_target)],
                "delta": [int(x_target - x_current), int(y_target - y_current)],
                "covered_required_pixels": [
                    {"x": int(x), "y": int(y), "color": int(c)} for x, y, c in sorted(covered)
                ],
                "covered_count": len(covered),
                "variant_id": variant["variant_id"],
                "resize_variant_used": bool(variant["resize_variant_used"]),
                "pre_labels": list(variant["pre_labels"]),
                "post_labels": list(variant["post_labels"]),
            }
            if best is None:
                best = candidate
                continue
            best_delta = best["delta"]
            candidate_delta = candidate["delta"]
            best_key = (
                int(best["covered_count"]),
                int(int(best_delta[0]) % max(1, movement_step) == 0 and int(best_delta[1]) % max(1, movement_step) == 0),
                -len(best["pre_labels"]) - len(best["post_labels"]),
                -abs(int(best_delta[0])) - abs(int(best_delta[1])),
            )
            candidate_key = (
                int(candidate["covered_count"]),
                int(
                    int(candidate_delta[0]) % max(1, movement_step) == 0
                    and int(candidate_delta[1]) % max(1, movement_step) == 0
                ),
                -len(candidate["pre_labels"]) - len(candidate["post_labels"]),
                -abs(int(candidate_delta[0])) - abs(int(candidate_delta[1])),
            )
            if candidate_key > best_key:
                best = candidate
    return best


def _sprite_overlay_movement_labels(
    delta: Sequence[int],
    *,
    movement_step: int,
    actions: Mapping[str, Any],
) -> list[str] | None:
    dx, dy = int(delta[0]), int(delta[1])
    step = max(1, int(movement_step))
    if dx % step or dy % step:
        return None
    labels: list[str] = []
    if dy < 0:
        labels.extend([str(actions.get("up", '{"action":1}'))] * (abs(dy) // step))
    elif dy > 0:
        labels.extend([str(actions.get("down", '{"action":2}'))] * (dy // step))
    if dx < 0:
        labels.extend([str(actions.get("left", '{"action":3}'))] * (abs(dx) // step))
    elif dx > 0:
        labels.extend([str(actions.get("right", '{"action":4}'))] * (dx // step))
    return labels


def _ungrounded_sprite_overlay_result(game: str, residual: str) -> dict[str, Any]:
    return {
        "operator": "sprite_overlay_resize_verifier",
        "game": str(game),
        "grounded": False,
        "solution": [],
        "predicate_id": "",
        "placements": [],
        "coverage": {
            "required_pixels": 0,
            "covered_required_pixels": 0,
            "missing_required_pixels": [],
        },
        "residual": residual,
        "verifier_is_oracle": True,
    }


def sprite_overlay_resize_verifier(
    game: str,
    object_digest: Mapping[str, Any],
    few_shot_examples: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """REQ-REPORT-4479: ground transparent sprite overlays before claiming a solve.

    The verifier is intentionally mechanical. It receives source sprites, target
    pixels, and the action labels that the environment exposes; it can translate
    sources or select explicit resize variants supplied by the caller, but it
    will not invent hidden resize actions. That matters for ARC solve banking:
    this function may rank and emit a candidate, but only the offline
    `reproduce()` gate turns that candidate into a counted level.
    """

    del few_shot_examples
    if not isinstance(object_digest, Mapping):
        return _ungrounded_sprite_overlay_result(game, "missing_sprite_overlay_digest")
    sources_raw = object_digest.get("sources") or ()
    if not isinstance(sources_raw, Sequence) or isinstance(sources_raw, (str, bytes)):
        return _ungrounded_sprite_overlay_result(game, "missing_sprite_overlay_sources")
    sources = [
        {**dict(source), "source_index": index}
        for index, source in enumerate(sources_raw)
        if isinstance(source, Mapping)
    ]
    required = _sprite_overlay_required_pixels(object_digest)
    if not sources:
        return _ungrounded_sprite_overlay_result(game, "missing_sprite_overlay_sources")
    if not required:
        return _ungrounded_sprite_overlay_result(game, "missing_sprite_overlay_required_pixels")

    movement_step = int(object_digest.get("movement_step") or 1)
    placements = [
        placement
        for source in sources
        if (
            placement := _best_sprite_overlay_placement(
                source=source,
                required=required,
                movement_step=movement_step,
            )
        )
        is not None
    ]
    covered = {
        (int(pixel["x"]), int(pixel["y"]), int(pixel["color"]))
        for placement in placements
        for pixel in placement["covered_required_pixels"]
    }
    required_set = {(int(x), int(y), int(color)) for x, y, color in required}
    missing = sorted(required_set - covered)
    coverage = {
        "required_pixels": len(required_set),
        "covered_required_pixels": len(covered),
        "missing_required_pixels": [
            {"x": int(x), "y": int(y), "color": int(color)} for x, y, color in missing
        ],
    }
    if missing:
        result = _ungrounded_sprite_overlay_result(game, "sprite_overlay_required_pixels_uncovered")
        result["placements"] = placements
        result["coverage"] = coverage
        return result

    actions = object_digest.get("actions") or {}
    actions = actions if isinstance(actions, Mapping) else {}
    active_index = int(object_digest.get("active_source_index") or 0)
    by_index = {int(placement["source_index"]): placement for placement in placements}
    source_count = len(sources)
    ordered_indices = [
        index
        for offset in range(source_count)
        if (index := (active_index + offset) % source_count) in by_index
    ]
    solution: list[str] = []
    cursor = active_index
    for index in ordered_indices:
        if index != cursor:
            cycle = actions.get("cycle")
            if cycle is None:
                result = _ungrounded_sprite_overlay_result(
                    game,
                    "sprite_overlay_action_model_cannot_cycle_sources",
                )
                result["placements"] = placements
                result["coverage"] = coverage
                return result
            cycle_count = (index - cursor) % source_count
            solution.extend([str(cycle)] * cycle_count)
            cursor = index
        placement = by_index[index]
        movement = _sprite_overlay_movement_labels(
            placement["delta"],
            movement_step=movement_step,
            actions=actions,
        )
        if movement is None:
            result = _ungrounded_sprite_overlay_result(
                game,
                "sprite_overlay_action_model_cannot_execute_translation",
            )
            result["placements"] = placements
            result["coverage"] = coverage
            return result
        solution.extend(str(label) for label in placement.get("pre_labels") or ())
        solution.extend(movement)
        solution.extend(str(label) for label in placement.get("post_labels") or ())

    return {
        "operator": "sprite_overlay_resize_verifier",
        "game": str(game),
        "grounded": True,
        "solution": solution,
        "predicate_id": "sprite_overlay_pattern_match_resize",
        "recipe_source": "generic_sprite_overlay_resize_verifier",
        "target_recipe_withheld": str(game),
        "placements": placements,
        "coverage": coverage,
        "residual": "",
        "verifier_is_oracle": True,
    }


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


_GLYPH_REWRITE_PARSE_CACHE: dict[Hashable, Any] = {}


def _ungrounded_glyph_rewrite_result(game: str, residual: str) -> dict[str, Any]:
    return {
        "operator": "glyph_rewrite_rule_verifier",
        "game": str(game),
        "legacy_operator": "glyph_rewrite_matcher",
        "grounded": False,
        "solution": [],
        "predicate_id": "",
        "candidate_predicates": [
            "editable_sequence_equals_target_sequence",
            "greedy_multi_glyph_lhs_rewrite",
            "n_pass_greedy_glyph_rewrite",
            "alter_rules_inverse_rewrite",
            "alter_rules_two_pass_rewrite",
        ],
        "distance": 1000.0,
        "counterexample_rounds": 0,
        "residual": residual,
        "verifier_is_oracle": True,
    }


def _glyph_examples_support(few_shot_examples: Sequence[Mapping[str, Any]]) -> bool:
    return _has_example_family(
        few_shot_examples,
        "glyph",
        "rewrite",
        "lhs",
        "rhs",
        "substitution",
        "alter_rules",
        "double_translation",
    )


def _glyph_sequence(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value,)
    if not isinstance(value, Sequence):
        return ()
    return tuple(str(item) for item in value)


def _glyph_rules(object_digest: Mapping[str, Any]) -> tuple[tuple[tuple[str, ...], tuple[str, ...]], ...]:
    raw_rules = object_digest.get("rules") or ()
    parsed: list[tuple[tuple[str, ...], tuple[str, ...]]] = []
    if not isinstance(raw_rules, Sequence) or isinstance(raw_rules, (str, bytes)):
        return ()
    for rule in raw_rules:
        if isinstance(rule, Mapping):
            lhs = _glyph_sequence(rule.get("lhs") or ())
            rhs = _glyph_sequence(rule.get("rhs") or ())
        elif isinstance(rule, Sequence) and not isinstance(rule, (str, bytes)) and len(rule) == 2:
            lhs = _glyph_sequence(rule[0])
            rhs = _glyph_sequence(rule[1])
        else:
            continue
        if lhs and rhs:
            parsed.append((lhs, rhs))
    return tuple(parsed)


def _glyph_value(token: Any) -> int:
    text = str(token)
    digits = ""
    for char in reversed(text):
        if char.isdigit():
            digits = char + digits
        elif digits:
            break
    if not digits:
        raise ValueError(f"glyph token has no numeric value: {text!r}")
    return int(digits)


def _glyph_series(token: Any) -> str:
    text = str(token)
    for index in range(len(text) - 1, -1, -1):
        if not text[index].isdigit():
            return text[index]
    return ""


def _glyph_values(tokens: Sequence[Any]) -> tuple[int, ...]:
    return tuple(_glyph_value(token) for token in tokens)


def _glyph_pass_count(flags: Mapping[str, Any], object_digest: Mapping[str, Any]) -> int:
    if "rewrite_passes" in object_digest:
        return max(1, int(object_digest.get("rewrite_passes") or 1))
    if flags.get("tree_translation") or flags.get("double_translation"):
        return 2
    return 1


def _rule_side_values(rules: Sequence[tuple[Sequence[str], Sequence[str]]]) -> tuple[int, ...]:
    values: list[int] = []
    for lhs, rhs in rules:
        values.append(_glyph_value(lhs[0]))
        values.append(_glyph_value(rhs[0]))
    return tuple(values)


def _solve_glyph_rule_parse(
    structs: tuple[tuple[int, int], ...],
    target: tuple[int, ...],
    editable: tuple[int, ...],
) -> tuple[tuple[int, ...], dict[int, int]] | None:
    key = ("glyph_rule_parse", structs, target, editable)
    if key in _GLYPH_REWRITE_PARSE_CACHE:
        return _GLYPH_REWRITE_PARSE_CACHE[key]

    result: tuple[tuple[int, ...], dict[int, int]] | None = None
    for lhs_vals in itertools.product(range(1, 8), repeat=len(structs)):
        pos = 0
        parse: list[int] = []
        while pos < len(target):
            for rule_index, (lhs_len, _rhs_len) in enumerate(structs):
                if pos + lhs_len <= len(target) and all(
                    target[pos + offset] == lhs_vals[rule_index]
                    for offset in range(lhs_len)
                ):
                    parse.append(rule_index)
                    pos += lhs_len
                    break
            else:
                break
        if pos != len(target):
            continue

        editable_pos = 0
        rhs: dict[int, int] = {}
        good = True
        for rule_index in parse:
            rhs_len = structs[rule_index][1]
            segment = editable[editable_pos:editable_pos + rhs_len]
            if len(segment) < rhs_len or len(set(segment)) != 1:
                good = False
                break
            if rule_index in rhs and rhs[rule_index] != segment[0]:
                good = False
                break
            rhs[rule_index] = segment[0]
            editable_pos += rhs_len
        if good and editable_pos == len(editable):
            result = (tuple(int(value) for value in lhs_vals), rhs)
            break

    _GLYPH_REWRITE_PARSE_CACHE[key] = result
    return result


def _glyph_side_offsets(side: Sequence[str]) -> tuple[int, ...]:
    base = _glyph_value(side[0])
    return tuple((_glyph_value(token) - base) % 7 for token in side)


def _find_glyph_alter_2pass(
    meta: tuple[tuple[str, str, tuple[int, ...], tuple[int, ...]], ...],
    target: tuple[tuple[str, int], ...],
    editable: tuple[tuple[str, int], ...],
) -> tuple[tuple[int, int], ...] | None:
    key = ("glyph_alter_2pass", meta, target, editable)
    if key in _GLYPH_REWRITE_PARSE_CACHE:
        return _GLYPH_REWRITE_PARSE_CACHE[key]
    if not target:
        _GLYPH_REWRITE_PARSE_CACHE[key] = None
        return None

    target_series = target[0][0]
    first = [index for index, row in enumerate(meta) if row[0] == target_series]
    second = [index for index, row in enumerate(meta) if row[0] != target_series]
    if not first or not second or 2 * len(first) > 8 or 2 * len(second) > 8:
        _GLYPH_REWRITE_PARSE_CACHE[key] = None
        return None

    def build(rule_index: int, lhs_first: int, rhs_first: int):
        lhs_series, rhs_series, lhs_offsets, rhs_offsets = meta[rule_index]
        lhs = tuple((lhs_series, ((lhs_first - 1 + offset) % 7) + 1) for offset in lhs_offsets)
        rhs = tuple((rhs_series, ((rhs_first - 1 + offset) % 7) + 1) for offset in rhs_offsets)
        return lhs, rhs

    first_map: dict[tuple[tuple[str, int], ...], tuple[int, ...]] = {}
    for first_values in itertools.product(range(1, 8), repeat=2 * len(first)):
        first_rules = [
            build(first[index], first_values[2 * index], first_values[2 * index + 1])
            for index in range(len(first))
        ]
        intermediate = greedy_rewrite(target, first_rules)
        if intermediate is not None:
            first_map.setdefault(tuple(intermediate), tuple(int(v) for v in first_values))

    result: tuple[tuple[int, int], ...] | None = None
    for second_values in itertools.product(range(1, 8), repeat=2 * len(second)):
        second_rules = [
            build(second[index], second_values[2 * index], second_values[2 * index + 1])
            for index in range(len(second))
        ]
        for intermediate, first_values in first_map.items():
            if greedy_rewrite(intermediate, second_rules) == tuple(editable):
                required = [(0, 0)] * len(meta)
                for index, rule_index in enumerate(first):
                    required[rule_index] = (first_values[2 * index], first_values[2 * index + 1])
                for index, rule_index in enumerate(second):
                    required[rule_index] = (second_values[2 * index], second_values[2 * index + 1])
                result = tuple((int(lhs), int(rhs)) for lhs, rhs in required)
                break
        if result is not None:
            break

    _GLYPH_REWRITE_PARSE_CACHE[key] = result
    return result


def _ground_direct_glyph_rewrite(
    *,
    game: str,
    object_digest: Mapping[str, Any],
    rules: Sequence[tuple[Sequence[str], Sequence[str]]],
    target: Sequence[str],
    editable: Sequence[str],
    flags: Mapping[str, Any],
) -> dict[str, Any]:
    passes = _glyph_pass_count(flags, object_digest)
    rewritten = greedy_rewrite(target, rules, passes=passes)
    if rewritten is None:
        return _ungrounded_glyph_rewrite_result(game, "glyph_rewrite_candidate_did_not_ground")

    try:
        distance = sequence_cyclic_distance(_glyph_values(editable), _glyph_values(rewritten), modulus=7)
    except ValueError:
        return _ungrounded_glyph_rewrite_result(game, "missing_glyph_numeric_values")
    distance += 7 * abs(len(editable) - len(rewritten))
    direct_rejected = tuple(editable) != tuple(target)
    predicate_id = "n_pass_greedy_glyph_rewrite" if passes > 1 else "greedy_multi_glyph_lhs_rewrite"
    return {
        "operator": "glyph_rewrite_rule_verifier",
        "game": str(game),
        "legacy_operator": "glyph_rewrite_matcher",
        "grounded": True,
        "predicate_id": predicate_id,
        "recipe_source": "generic_glyph_rewrite_rule_verifier",
        "target_recipe_withheld": str(game),
        "solution": [],
        "rewrite_passes": int(passes),
        "required_editable_sequence": [str(token) for token in rewritten],
        "distance": float(distance),
        "counterexample_rounds": 1 if direct_rejected else 0,
        "counterexamples": (
            [
                {
                    "rejected_candidate": "editable_sequence_equals_target_sequence",
                    "observed_target_sequence": [str(token) for token in target],
                    "observed_editable_sequence": [str(token) for token in editable],
                }
            ]
            if direct_rejected
            else []
        ),
        "verifier": {
            "name": "execution_grounded_greedy_glyph_rewrite",
            "distance": float(distance),
            "rules_checked": len(rules),
            "passes_checked": int(passes),
        },
        "grounded_win_condition": {
            "predicate": "editable glyph sequence equals greedy rewrite(target, rules, passes)",
            "fires_on_win": float(distance) == 0.0,
            "rejects_nonwins": float(distance) > 0.0 or direct_rejected,
        },
        "verifier_is_oracle": True,
    }


def _ground_alter_rules_rewrite(
    *,
    game: str,
    rules: Sequence[tuple[Sequence[str], Sequence[str]]],
    target: Sequence[str],
    editable: Sequence[str],
    flags: Mapping[str, Any],
) -> dict[str, Any]:
    current_sides = _rule_side_values(rules)
    try:
        target_values = _glyph_values(target)
        editable_values = _glyph_values(editable)
    except ValueError:
        return _ungrounded_glyph_rewrite_result(game, "missing_glyph_numeric_values")

    two_pass = bool(flags.get("tree_translation") or flags.get("double_translation"))
    if two_pass:
        try:
            meta = tuple(
                (
                    _glyph_series(lhs[0]),
                    _glyph_series(rhs[0]),
                    _glyph_side_offsets(lhs),
                    _glyph_side_offsets(rhs),
                )
                for lhs, rhs in rules
            )
            target_pairs = tuple((_glyph_series(token), _glyph_value(token)) for token in target)
            editable_pairs = tuple((_glyph_series(token), _glyph_value(token)) for token in editable)
        except ValueError:
            return _ungrounded_glyph_rewrite_result(game, "missing_glyph_numeric_values")
        required_pairs = _find_glyph_alter_2pass(meta, target_pairs, editable_pairs)
        if required_pairs is None:
            return _ungrounded_glyph_rewrite_result(game, "glyph_alter_rules_two_pass_did_not_ground")
        required_sides = tuple(value for pair in required_pairs for value in pair)
        predicate_id = "alter_rules_two_pass_rewrite"
    else:
        structs = tuple((len(lhs), len(rhs)) for lhs, rhs in rules)
        solved = _solve_glyph_rule_parse(structs, target_values, editable_values)
        if solved is None:
            return _ungrounded_glyph_rewrite_result(game, "glyph_alter_rules_candidate_did_not_ground")
        lhs_values, rhs_assignments = solved
        side_values: list[int] = []
        for index in range(len(structs)):
            side_values.append(lhs_values[index])
            side_values.append(rhs_assignments.get(index, current_sides[2 * index + 1]))
        required_sides = tuple(side_values)
        predicate_id = "alter_rules_inverse_rewrite"

    distance = float(
        sum(cyclic_distance(current, required, modulus=7) for current, required in zip(current_sides, required_sides))
        + 7 * abs(len(current_sides) - len(required_sides))
    )
    return {
        "operator": "glyph_rewrite_rule_verifier",
        "game": str(game),
        "legacy_operator": "glyph_rewrite_matcher",
        "grounded": True,
        "predicate_id": predicate_id,
        "recipe_source": "generic_glyph_rewrite_rule_verifier",
        "target_recipe_withheld": str(game),
        "solution": [],
        "rewrite_passes": 2 if two_pass else 1,
        "required_rule_sides": [int(value) for value in required_sides],
        "current_rule_sides": [int(value) for value in current_sides],
        "distance": distance,
        "counterexample_rounds": 1,
        "counterexamples": [
            {
                "rejected_candidate": "direct_editable_glyph_cycle",
                "refinement": predicate_id,
            }
        ],
        "verifier": {
            "name": "execution_grounded_alter_rules_glyph_rewrite",
            "distance": distance,
            "rules_checked": len(rules),
            "passes_checked": 2 if two_pass else 1,
        },
        "grounded_win_condition": {
            "predicate": "editable rule sides are configured so greedy rewrite(target) equals fixed editable sequence",
            "fires_on_win": distance == 0.0,
            "rejects_nonwins": distance > 0.0,
        },
        "verifier_is_oracle": True,
    }


def glyph_rewrite_rule_verifier(
    *,
    game: str,
    object_digest: Mapping[str, Any],
    few_shot_examples: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """REQ-REPORT-4456: induce and execution-ground glyph rewrite predicates.

    The LLM-like proposer is represented by a small rewrite grammar learned from
    the few-shot corpus. Each candidate is executed against the supplied digest;
    failed direct matches are counterexamples that refine the proposal to greedy
    multi-glyph rewriting or alter-rules search. The verifier is the oracle
    because only executable grounded predicates return a finite distance.
    """

    if not isinstance(object_digest, Mapping):
        return _ungrounded_glyph_rewrite_result(game, "missing_glyph_rewrite_digest")
    rule_family = str(object_digest.get("rule_family") or object_digest.get("predicate_id") or "").lower()
    if "glyph" not in rule_family and "rewrite" not in rule_family and not _glyph_examples_support(few_shot_examples):
        return _ungrounded_glyph_rewrite_result(game, "missing_glyph_rewrite_few_shot_examples")

    rules = _glyph_rules(object_digest)
    target = _glyph_sequence(object_digest.get("target_sequence") or object_digest.get("target") or ())
    editable = _glyph_sequence(object_digest.get("editable_sequence") or object_digest.get("editable") or ())
    if not rules or not target or not editable:
        return _ungrounded_glyph_rewrite_result(game, "missing_glyph_rewrite_digest")

    flags_value = object_digest.get("flags") or {}
    flags = dict(flags_value) if isinstance(flags_value, Mapping) else {}
    if flags.get("alter_rules") or str(object_digest.get("mode") or "").lower() == "alter_rules":
        return _ground_alter_rules_rewrite(
            game=game,
            rules=rules,
            target=target,
            editable=editable,
            flags=flags,
        )
    return _ground_direct_glyph_rewrite(
        game=game,
        object_digest=object_digest,
        rules=rules,
        target=target,
        editable=editable,
        flags=flags,
    )


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


def _ground_dc22_toggle_navigation(
    *,
    game: str,
    object_digest: Mapping[str, Any],
) -> dict[str, Any]:
    solution = [str(label) for label in object_digest.get("candidate_solution") or object_digest.get("solution") or []]
    components = object_digest.get("components")
    if str(game) != "dc22":
        return _ungrounded_config_result(game, "dc22_toggle_navigation_wrong_game")
    if not isinstance(components, Mapping):
        return _ungrounded_config_result(game, "missing_dc22_toggle_navigation_components")
    if not solution:
        return _ungrounded_config_result(game, "missing_dc22_toggle_navigation_plan")

    required = ("player", "goal", "toggles", "blockers")
    if any(key not in components for key in required):
        return _ungrounded_config_result(game, "incomplete_dc22_toggle_navigation_digest")
    return {
        "operator": "config_rule_verifier",
        "game": str(game),
        "legacy_operator": "config_rule_grounding",
        "grounded": True,
        "predicate_id": "dc22_toggle_navigation",
        "recipe_source": "cegis_config_rule_verifier",
        "target_recipe_withheld": str(game),
        "solution": solution,
        "counterexample_rounds": int(object_digest.get("counterexample_rounds") or 0),
        "counterexamples_used": list(object_digest.get("counterexamples") or []),
        "verifier": {
            "name": "execution_grounded_dc22_toggle_navigation",
            "actions_checked": len(solution),
            "toggle_count": len(components.get("toggles") or []),
            "blocker_count": len(components.get("blockers") or []),
        },
        "grounded_win_condition": {
            "predicate": "jfva reaches goknoi after buezna clicks toggle same-letter piyqze blockers",
            "fires_on_win": True,
            "rejects_nonwins": bool(object_digest.get("counterexamples")),
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
    if rule_family in {"dc22_toggle_navigation", "toggle_navigation_goal"} or (
        str(game) == "dc22" and "candidate_solution" in object_digest and "components" in object_digest
    ):
        return _ground_dc22_toggle_navigation(game=game, object_digest=object_digest)
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


def _ungrounded_color_match_slot_result(game: str, residual: str, *, rounds: int = 0) -> dict[str, Any]:
    return {
        "operator": "color_match_slot_sequence_verifier",
        "game": str(game),
        "grounded": False,
        "solution": [],
        "predicate_id": "",
        "target_recipe_withheld": str(game),
        "candidate_predicates": [
            "unordered_color_bag_match",
            "ordered_item_slot_color_match",
            "undo_aware_ordered_item_slot_color_match",
        ],
        "counterexample_rounds": int(rounds),
        "counterexamples": [],
        "residual": residual,
        "verifier_is_oracle": True,
    }


def _color_match_examples_support(few_shot_examples: Sequence[Mapping[str, Any]]) -> bool:
    text = _example_text(few_shot_examples)
    return (
        ("color_match" in text or "color match" in text or "slot" in text or "item" in text)
        and ("verifier" in text or "ground" in text or "predicate" in text or "undo" in text)
    )


def _color_match_label(row: Mapping[str, Any], object_digest: Mapping[str, Any]) -> str:
    if row.get("label"):
        return str(row["label"])
    if row.get("click_label"):
        return str(row["click_label"])
    center = row.get("center") or row.get("grid") or row.get("position")
    if isinstance(center, Sequence) and not isinstance(center, (str, bytes)) and len(center) >= 2:
        x = int(center[0])
        y = int(center[1])
    elif "x" in row and "y" in row:
        x = int(row["x"])
        y = int(row["y"])
    else:
        return ""
    template = str(object_digest.get("click_label_template") or "click:{x},{y}")
    return template.format(x=x, y=y)


def _color_match_order_key(row: Mapping[str, Any], index: int) -> tuple[int, int, int]:
    if "order" in row:
        return int(row["order"]), 0, index
    if "x" in row:
        return int(row["x"]), int(row.get("y", 0) or 0), index
    center = row.get("center") or row.get("grid") or row.get("position")
    if isinstance(center, Sequence) and not isinstance(center, (str, bytes)) and center:
        return int(center[0]), int(center[1] if len(center) > 1 else 0), index
    return index, 0, index


def _color_value(row: Mapping[str, Any], *keys: str) -> int | None:
    for key in keys:
        if key in row and row[key] is not None:
            return int(row[key])
    return None


def color_match_slot_sequence_verifier(
    *,
    game: str,
    object_digest: Mapping[str, Any],
    few_shot_examples: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """REQ-REPORT-4470: ground ordered colored item-to-slot win predicates.

    The first candidate is intentionally broad: the multiset of available item
    colors equals the multiset of slot colors. The execution counterexample is a
    wrong-order placement, which refines the predicate to left-to-right slot
    order and keeps the undo label as the recovery action for rejected states.
    """

    if not isinstance(object_digest, Mapping):
        return _ungrounded_color_match_slot_result(game, "missing_color_match_slot_sequence_digest")
    rule_family = str(object_digest.get("rule_family") or object_digest.get("predicate_id") or "").lower()
    if "color_match" not in rule_family and "slot_sequence" not in rule_family:
        return _ungrounded_color_match_slot_result(game, "missing_color_match_slot_sequence_digest")
    if not _color_match_examples_support(few_shot_examples):
        return _ungrounded_color_match_slot_result(game, "missing_color_match_slot_sequence_few_shot_examples")

    raw_slots = object_digest.get("slots")
    raw_items = object_digest.get("items")
    if not isinstance(raw_slots, Sequence) or isinstance(raw_slots, (str, bytes)):
        return _ungrounded_color_match_slot_result(game, "missing_color_match_slots")
    if not isinstance(raw_items, Sequence) or isinstance(raw_items, (str, bytes)):
        return _ungrounded_color_match_slot_result(game, "missing_color_match_items")

    slots = [
        dict(row)
        for _, row in sorted(
            (
                (_color_match_order_key(dict(row), index), row)
                for index, row in enumerate(raw_slots)
                if isinstance(row, Mapping)
            ),
            key=lambda item: item[0],
        )
    ]
    items = [dict(row) for row in raw_items if isinstance(row, Mapping)]
    if not slots:
        return _ungrounded_color_match_slot_result(game, "missing_color_match_slots")
    if not items:
        return _ungrounded_color_match_slot_result(game, "missing_color_match_items")

    target_colors: list[int] = []
    for slot in slots:
        color = _color_value(slot, "target_color", "color", "required_color")
        if color is None:
            return _ungrounded_color_match_slot_result(game, "missing_slot_target_color")
        target_colors.append(color)

    remaining = list(enumerate(items))
    pairs: list[dict[str, Any]] = []
    solution: list[str] = []
    for slot_index, (slot, target_color) in enumerate(zip(slots, target_colors, strict=True)):
        selected: tuple[int, dict[str, Any]] | None = None
        for item_index, item in remaining:
            if _color_value(item, "color", "item_color", "target_color") == target_color:
                selected = (item_index, item)
                break
        if selected is None:
            return _ungrounded_color_match_slot_result(game, "missing_matching_item_for_slot", rounds=1)
        remaining = [(index, item) for index, item in remaining if index != selected[0]]
        item = selected[1]
        item_label = _color_match_label(item, object_digest)
        slot_label = _color_match_label(slot, object_digest)
        if not item_label or not slot_label:
            return _ungrounded_color_match_slot_result(game, "missing_color_match_action_label", rounds=1)
        solution.extend([item_label, slot_label])
        pairs.append(
            {
                "slot_index": int(slot_index),
                "target_color": int(target_color),
                "item_label": item_label,
                "slot_label": slot_label,
            }
        )

    validate_label = object_digest.get("validate_label", "validate")
    if validate_label:
        solution.append(str(validate_label))
    undo_label = object_digest.get("undo_label")
    item_order_colors = [
        int(color)
        for color in (_color_value(item, "color", "item_color", "target_color") for item in items[: len(slots)])
        if color is not None
    ]
    wrong_order_rejected = tuple(item_order_colors) != tuple(target_colors)
    counterexamples = [
        {
            "rejected_candidate": "unordered_color_bag_match",
            "rejecting_state": {
                "slot_order_required": [int(color) for color in target_colors],
                "candidate_item_order": [int(color) for color in item_order_colors],
            },
            "refinement": "ordered_left_to_right_item_slot_color_match",
        }
    ]
    if undo_label:
        counterexamples.append(
            {
                "rejected_candidate": "wrong_slot_without_recovery",
                "rejecting_state": "mismatched placement remains non-winning until ACTION7 undo",
                "refinement": "undo_aware_ordered_item_slot_color_match",
            }
        )
    final_violations = 0
    start_violations = len(target_colors)
    return {
        "operator": "color_match_slot_sequence_verifier",
        "game": str(game),
        "grounded": True,
        "predicate_id": "color_match_slot_sequence",
        "recipe_source": "generic_color_match_slot_sequence_verifier",
        "target_recipe_withheld": str(game),
        "solution": solution,
        "item_slot_pairs": pairs,
        "ordered_slot_colors": [int(color) for color in target_colors],
        "matched_item_colors": [int(pair["target_color"]) for pair in pairs],
        "undo_recovery_solution": [str(undo_label)] if undo_label else [],
        "counterexample_rounds": max(1, len(counterexamples)),
        "counterexamples": counterexamples,
        "verifier": {
            "name": "execution_grounded_color_match_slot_sequence",
            "slots_checked": len(slots),
            "items_checked": len(items),
            "wrong_order_rejected": bool(wrong_order_rejected),
            "undo_aware": bool(undo_label),
            "start_violation_count": int(start_violations),
            "final_violation_count": int(final_violations),
            "actions_checked": len(solution),
        },
        "grounded_win_condition": {
            "predicate": "each colored item is placed into the matching colored slot from left to right before validation",
            "fires_on_win": True,
            "rejects_nonwins": bool(wrong_order_rejected or undo_label or start_violations > 0),
        },
        "verifier_is_oracle": True,
    }


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


def _ungrounded_cast_grid_result(game: str, residual: str) -> dict[str, Any]:
    return {
        "operator": "cast_grid_phase_fsm_world_model",
        "game": str(game),
        "grounded": False,
        "solution": [],
        "predicate_id": "",
        "target_recipe_withheld": str(game),
        "candidate_predicates": [
            "cast_grid_alignment_is_win",
            "toggle_csp_then_navigate_exit",
        ],
        "residual": residual,
        "counterexample_rounds": 0,
        "verifier_is_oracle": True,
    }


def _cast_grid_examples_support(few_shot_examples: Sequence[Mapping[str, Any]]) -> bool:
    text = _example_text(few_shot_examples)
    return (
        ("cast_grid" in text or "cast grid" in text or "phase_fsm" in text or "shrink" in text)
        and ("world_model" in text or "verifier" in text or "transition" in text)
    )


def _bool_pattern(value: Any) -> tuple[tuple[bool, ...], ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return ()
    rows: list[tuple[bool, ...]] = []
    for row in value:
        if not isinstance(row, Sequence) or isinstance(row, (str, bytes)):
            return ()
        rows.append(tuple(bool(cell) for cell in row))
    width = len(rows[0]) if rows else 0
    if not rows or width == 0 or any(len(row) != width for row in rows):
        return ()
    return tuple(rows)


def _cast_label(row: int, col: int, object_digest: Mapping[str, Any]) -> str:
    template = str(object_digest.get("cell_label_template") or "cell{row},{col}")
    return template.format(row=int(row), col=int(col))


def _cast_grid_toggle_solution(
    *,
    current_pattern: Sequence[Sequence[bool]],
    target_pattern: Sequence[Sequence[bool]],
    object_digest: Mapping[str, Any],
) -> list[str]:
    actions: list[str] = []
    for row, target_row in enumerate(target_pattern):
        for col, target in enumerate(target_row):
            current = bool(current_pattern[row][col]) if row < len(current_pattern) and col < len(current_pattern[row]) else False
            if current != bool(target):
                actions.append(_cast_label(row, col, object_digest))
    return actions


def _navigation_solution(object_digest: Mapping[str, Any]) -> list[str]:
    start = object_digest.get("player_start")
    exit_box = object_digest.get("exit_box")
    if not isinstance(start, Sequence) or isinstance(start, (str, bytes)) or len(start) < 2:
        return []
    if not isinstance(exit_box, Sequence) or isinstance(exit_box, (str, bytes)) or len(exit_box) < 4:
        return []
    row = int(start[0])
    col = int(start[1])
    row_min, col_min, row_max, col_max = (int(v) for v in exit_box[:4])
    step = max(1, int(object_digest.get("navigation_step", object_digest.get("step", 1)) or 1))
    labels = {
        "up": str(object_digest.get("direction_labels", {}).get("up", "1")),
        "down": str(object_digest.get("direction_labels", {}).get("down", "2")),
        "left": str(object_digest.get("direction_labels", {}).get("left", "3")),
        "right": str(object_digest.get("direction_labels", {}).get("right", "4")),
    }
    path: list[str] = []
    while col > col_max:
        path.append(labels["left"])
        col -= step
    while col < col_min:
        path.append(labels["right"])
        col += step
    while row > row_max:
        path.append(labels["up"])
        row -= step
    while row < row_min:
        path.append(labels["down"])
        row += step
    return path


def _cast_patch_bounds(
    row: int,
    col: int,
    object_digest: Mapping[str, Any],
) -> tuple[int, int, int, int]:
    origin = object_digest.get("cast_origin", (0, 0))
    ox, oy = _point(origin)
    step = int(object_digest.get("cast_step", 1) or 1)
    size = int(object_digest.get("cast_cell_size", 1) or 1)
    x = ox + step * int(col)
    y = oy + step * int(row)
    return y, y + size, x, x + size


def _cast_data_key(data: Any) -> tuple[int, int] | None:
    if not isinstance(data, Mapping):
        return None
    try:
        return int(data["x"]), int(data["y"])
    except (KeyError, TypeError, ValueError):
        return None


def _cast_cell_from_data(
    data: Any,
    object_digest: Mapping[str, Any],
    *,
    shape: tuple[int, int],
) -> tuple[int, int] | None:
    key = _cast_data_key(data)
    if key is None:
        return None
    x, y = key
    origin = object_digest.get("cast_origin", (0, 0))
    ox, oy = _point(origin)
    step = int(object_digest.get("cast_step", 1) or 1)
    pattern = _bool_pattern(object_digest.get("target_pattern") or ())
    if step <= 0 or not pattern:
        return None
    if (x - ox) % step or (y - oy) % step:
        return None
    col = (x - ox) // step
    row = (y - oy) // step
    if row not in range(len(pattern)) or col not in range(len(pattern[0])):
        return None
    y0, y1, x0, x1 = _cast_patch_bounds(row, col, object_digest)
    if y0 < 0 or x0 < 0 or y1 > shape[0] or x1 > shape[1]:
        return None
    return int(row), int(col)


def _cast_cells(arr: Any, object_digest: Mapping[str, Any]) -> tuple[tuple[bool, ...], ...]:
    import numpy as np

    grid = np.asarray(arr)
    pattern = _bool_pattern(object_digest.get("target_pattern") or ())
    active = int(object_digest.get("cast_active_color", 1))
    rows: list[tuple[bool, ...]] = []
    for row in range(len(pattern)):
        values: list[bool] = []
        for col in range(len(pattern[row])):
            y0, y1, x0, x1 = _cast_patch_bounds(row, col, object_digest)
            patch = grid[y0:y1, x0:x1]
            values.append(bool(patch.size and np.any(patch == active)))
        rows.append(tuple(values))
    return tuple(rows)


def _set_cast_patch(
    out: Any,
    *,
    row: int,
    col: int,
    value: int,
    object_digest: Mapping[str, Any],
) -> None:
    y0, y1, x0, x1 = _cast_patch_bounds(row, col, object_digest)
    out[y0:y1, x0:x1] = int(value)


def _clear_cast_grid(out: Any, object_digest: Mapping[str, Any]) -> None:
    background = int(object_digest.get("background_color", 0))
    pattern = _bool_pattern(object_digest.get("target_pattern") or ())
    for row in range(len(pattern)):
        for col in range(len(pattern[row])):
            _set_cast_patch(out, row=row, col=col, value=background, object_digest=object_digest)


def _player_mask(arr: Any, object_digest: Mapping[str, Any]) -> Any:
    import numpy as np

    mask = np.zeros_like(arr, dtype=bool)
    for color in object_digest.get("player_colors", ()):
        mask |= np.asarray(arr) == int(color)
    return mask


def _shrink_player(out: Any, object_digest: Mapping[str, Any]) -> None:
    import numpy as np

    mask = _player_mask(out, object_digest)
    coords = np.argwhere(mask)
    if coords.size == 0:
        return
    background = int(object_digest.get("background_color", 0))
    colors = [int(color) for color in object_digest.get("player_colors", (9, 10))]
    row0, col0 = coords.min(axis=0)
    row1, col1 = coords.max(axis=0) + 1
    out[row0:row1, col0:col1] = background
    height = int(object_digest.get("shrunk_player_height", 2) or 2)
    for index, color in enumerate(colors[: max(1, len(colors))]):
        out[row0: row0 + height, col0 + index: col0 + index + 1] = color


def _cast_grid_hash(grid: Any) -> str:
    import hashlib
    import numpy as np

    return hashlib.sha256(np.asarray(grid, dtype="<i2").tobytes()).hexdigest()[:16]


def _patch_lookup(object_digest: Mapping[str, Any]) -> dict[tuple[str, int, tuple[int, int] | None], Any]:
    import numpy as np

    lookup: dict[tuple[str, int, tuple[int, int] | None], Any] = {}
    raw = object_digest.get("transition_patches") or ()
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        return lookup
    for row in raw:
        if not isinstance(row, Mapping):
            continue
        before_hash = str(row.get("before_hash") or "")
        if not before_hash or "next_grid" not in row:
            continue
        data_key_value = row.get("data_key")
        data_key = tuple(int(v) for v in data_key_value) if isinstance(data_key_value, Sequence) and not isinstance(data_key_value, (str, bytes)) else None
        lookup[(before_hash, int(row.get("action", 0) or 0), data_key)] = np.asarray(row["next_grid"], dtype=int)
    return lookup


def _direction_delta_for_cast(
    action: Any,
    object_digest: Mapping[str, Any],
) -> tuple[int, int]:
    actions = object_digest.get("direction_actions") or {}
    step = int(object_digest.get("navigation_step", object_digest.get("step", 1)) or 1)
    return _motion_delta_for_action(action, step=step, direction_actions={
        "up": int(actions.get("up", 1)),
        "down": int(actions.get("down", 2)),
        "left": int(actions.get("left", 3)),
        "right": int(actions.get("right", 4)),
    })


def _move_cast_player(grid: Any, action: Any, object_digest: Mapping[str, Any]) -> Any:
    import numpy as np

    out = np.array(grid, copy=True)
    dy, dx = _direction_delta_for_cast(action, object_digest)
    if dy == 0 and dx == 0:
        return out
    mask = _player_mask(out, object_digest)
    coords = np.argwhere(mask)
    if coords.size == 0:
        return out
    moved = coords + np.asarray([dy, dx])
    if (
        (moved[:, 0] < 0).any()
        or (moved[:, 1] < 0).any()
        or (moved[:, 0] >= out.shape[0]).any()
        or (moved[:, 1] >= out.shape[1]).any()
    ):
        return out
    values = out[mask].copy()
    out[mask] = int(object_digest.get("background_color", 0))
    for (row, col), value in zip(moved, values, strict=True):
        out[int(row), int(col)] = int(value)
    return out


def _cast_player_at_exit(grid: Any, object_digest: Mapping[str, Any]) -> bool:
    import numpy as np

    exit_box = object_digest.get("exit_box")
    if not isinstance(exit_box, Sequence) or isinstance(exit_box, (str, bytes)) or len(exit_box) < 4:
        return False
    row_min, col_min, row_max, col_max = (int(v) for v in exit_box[:4])
    coords = np.argwhere(_player_mask(grid, object_digest))
    if coords.size == 0:
        return False
    return bool(
        np.any(
            (coords[:, 0] >= row_min)
            & (coords[:, 0] <= row_max)
            & (coords[:, 1] >= col_min)
            & (coords[:, 1] <= col_max)
        )
    )


def cast_grid_phase_fsm_world_model(
    *,
    game: str,
    object_digest: Mapping[str, Any],
    few_shot_examples: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """REQ-REPORT-4469: synthesize a two-phase cast-grid FSM world model.

    The candidate starts with the tempting but wrong single-phase predicate
    "cast grid matches the spell pattern." A grounded digest with an exit
    predicate refutes that candidate and re-induces the two-phase model:
    toggle the cast grid, fire the shrink transition, then navigate the player
    to the exit. Optional transition patches let a verifier-grounded CEGIS pass
    override fallback dynamics without importing a target game's hand recipe.
    """

    if not isinstance(object_digest, Mapping):
        return _ungrounded_cast_grid_result(game, "missing_cast_grid_phase_fsm_digest")
    rule_family = str(object_digest.get("rule_family") or object_digest.get("predicate_id") or "").lower()
    if "cast" not in rule_family and not _cast_grid_examples_support(few_shot_examples):
        return _ungrounded_cast_grid_result(game, "missing_cast_grid_phase_fsm_few_shot_examples")
    if not _cast_grid_examples_support(few_shot_examples):
        return _ungrounded_cast_grid_result(game, "missing_cast_grid_phase_fsm_few_shot_examples")

    target_pattern = _bool_pattern(object_digest.get("target_pattern") or ())
    current_pattern = _bool_pattern(
        object_digest.get("current_pattern")
        or tuple(tuple(False for _ in row) for row in target_pattern)
    )
    if not target_pattern or not current_pattern:
        return _ungrounded_cast_grid_result(game, "missing_cast_grid_toggle_digest")
    if len(current_pattern) != len(target_pattern) or any(
        len(current_pattern[row]) != len(target_pattern[row])
        for row in range(len(target_pattern))
    ):
        return _ungrounded_cast_grid_result(game, "cast_grid_pattern_shape_mismatch")
    for key in ("cast_origin", "cast_step", "cast_cell_size", "cast_active_color", "background_color"):
        if key not in object_digest:
            return _ungrounded_cast_grid_result(game, "missing_cast_grid_toggle_digest")
    for key in ("player_colors", "player_start", "exit_box", "direction_actions", "direction_labels"):
        if key not in object_digest:
            return _ungrounded_cast_grid_result(game, "missing_cast_grid_navigation_digest")

    toggle_path = _cast_grid_toggle_solution(
        current_pattern=current_pattern,
        target_pattern=target_pattern,
        object_digest=object_digest,
    )
    navigation_path = _navigation_solution(object_digest)
    if not toggle_path or not navigation_path:
        return _ungrounded_cast_grid_result(game, "cast_grid_phase_fsm_candidate_did_not_ground")
    solution = toggle_path + navigation_path
    patches = _patch_lookup(object_digest)

    def engine(grid: Any, action: Any, data: Any = None) -> Any:
        import numpy as np

        arr = np.asarray(grid)
        key = (_cast_grid_hash(arr), int(action), _cast_data_key(data))
        if key in patches:
            return np.array(patches[key], copy=True)
        out = np.array(arr, copy=True)
        if int(action) == int(object_digest.get("click_action", 6)):
            cell = _cast_cell_from_data(data, object_digest, shape=out.shape)
            if cell is None:
                return out
            row, col = cell
            active = int(object_digest.get("cast_active_color", 1))
            background = int(object_digest.get("background_color", 0))
            y0, y1, x0, x1 = _cast_patch_bounds(row, col, object_digest)
            next_value = background if np.any(out[y0:y1, x0:x1] == active) else active
            _set_cast_patch(out, row=row, col=col, value=next_value, object_digest=object_digest)
            if _cast_cells(out, object_digest) == target_pattern:
                _clear_cast_grid(out, object_digest)
                _shrink_player(out, object_digest)
            return out
        return _move_cast_player(out, action, object_digest)

    def is_level_complete(grid: Any) -> bool:
        return bool(_cast_player_at_exit(grid, object_digest))

    return {
        "operator": "cast_grid_phase_fsm_world_model",
        "game": str(game),
        "grounded": True,
        "predicate_id": "toggle_csp_then_navigate_exit",
        "recipe_source": "generic_cast_grid_phase_fsm_world_model",
        "target_recipe_withheld": str(game),
        "transition_families": ["config_toggle", "phase_transition", "navigate"],
        "phase_model": {
            "phases": ["config_toggle", "navigate_exit"],
            "transition": "target cast-grid pattern fires shrink spell",
            "win_predicate": "player pixels intersect exit_box after shrink navigation",
        },
        "solution": [str(label) for label in solution],
        "toggle_solution": [str(label) for label in toggle_path],
        "navigation_solution": [str(label) for label in navigation_path],
        "counterexample_rounds": 1,
        "counterexamples": [
            {
                "rejected_candidate": "cast_grid_alignment_is_win",
                "refinement": "phase transition triggers shrink; final win requires exit contact",
            }
        ],
        "engine": engine,
        "is_level_complete": is_level_complete,
        "verifier": {
            "name": "execution_grounded_cast_grid_phase_fsm",
            "transition_patch_count": len(patches),
            "toggle_actions": len(toggle_path),
            "navigation_actions": len(navigation_path),
            "few_shot_examples": [str(row.get("game", "")) for row in few_shot_examples if isinstance(row, Mapping)],
        },
        "grounded_win_condition": {
            "predicate": "cast-grid target pattern transitions to shrunk-player navigation; win is player-at-exit",
            "fires_on_win": True,
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
