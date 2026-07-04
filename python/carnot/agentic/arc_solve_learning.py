"""ARC-AGI-3 solve LEARNING loop — turn the static reuse-substrate (arc_solver_kit
+ ops/arc_solve_registry.yaml) into something that actively SPEEDS UP the next
game by routing it to the closest SOLVED game's recipe, and that surfaces prior
DEAD-ENDS so we don't repeat them.

Why this exists
---------------
2026-06-16: the kit + registry capture what we learned, but a new game's solver
still started from zero (sc25 took ~10 reverse-engineering layers). The operator
asked: "are we learning from our successes and failures as part of our harness so
as to speed up progress?" This module is the success/failure feedback loop:
`recommend_approach(game)` reads the survey features + the registry, ranks the
solved games by similarity, and hands back the most-applicable proven recipe
(solver module, win-condition, action-model, reusable gotchas) PLUS the relevant
dead-ends to avoid. The agent/planner calls it BEFORE reverse-engineering a new
game, so each solve compounds onto the last instead of restarting.

This is the routing layer; the deeper search-acceleration (a verifier/value head
that prunes the BFS — the north-star verifier-routed-efficiency, cf. exp4071) is
the next loop on top of it.
"""

from __future__ import annotations

from collections.abc import Mapping
import json
from pathlib import Path
import statistics
from typing import Any, Optional

import yaml

from .arc_typed_memory_provenance_guard import typed_memory_provenance_guard

REPO = Path(__file__).resolve().parents[3]
SURVEY = REPO / "results" / "arc3_win_condition_survey.json"
REGISTRY = REPO / "ops" / "arc_solve_registry.yaml"

_WIN_KEYWORDS = (
    "align",
    "goal",
    "+1",
    "reflect",
    "pattern",
    "template",
    "rotate",
    "exit",
    "spell",
    "cast",
    "drag",
    "click",
    "move",
    "position",
)


def _action_type(s: str) -> str:
    s = (s or "").lower()
    has_click = "click" in s or "[6]" in s or "6]" in s
    has_kbd = "keyboard" in s or "action1" in s or "1-4" in s or "1-5" in s or "1-6" in s
    if has_click and has_kbd:
        return "mixed"
    if has_click:
        return "click"
    if has_kbd:
        return "keyboard"
    return "unknown"


def _features(entry: dict) -> dict:
    wc = str(entry.get("win_condition_summary", "")).lower()
    return {
        "game": entry.get("game", ""),
        "action_type": _action_type(entry.get("available_actions", "")),
        "spatial": bool(entry.get("is_spatial_planning")),
        "difficulty": str(entry.get("win_difficulty", "")),
        "win_kw": {k for k in _WIN_KEYWORDS if k in wc},
    }


def _survey_features() -> dict[str, dict]:
    d = json.load(open(SURVEY))
    pgs = d["per_game_surveys"]
    entries = list(pgs.values()) if isinstance(pgs, dict) else pgs
    return {e.get("game", ""): _features(e) for e in entries}


def _registry() -> dict:
    return yaml.safe_load(open(REGISTRY))


def _solved_games(reg: dict) -> list[dict]:
    """Games with a usable recipe (reproduced, or provisional-with-mechanics)."""
    return [
        g
        for g in reg.get("games", [])
        if g.get("reproducibility") in ("reproduced", "provisional") and g.get("solver")
    ]


def _similarity(a: dict, b: dict) -> float:
    score = 0.0
    if a["action_type"] == b["action_type"] and a["action_type"] != "unknown":
        score += 3.0
    elif "mixed" in (a["action_type"], b["action_type"]):
        score += 1.0  # mixed partially overlaps either
    if a["spatial"] == b["spatial"]:
        score += 1.5
    if a["difficulty"] and a["difficulty"] == b["difficulty"]:
        score += 0.5
    score += 1.0 * len(a["win_kw"] & b["win_kw"])  # shared win-condition vocabulary
    return score


# FinAcumen (arXiv:2606.17642) selective-retrieval lesson: an IRRELEVANT retrieved example actively
# DEGRADES reasoning -- precision beats recall. So below a confidence bar we must NOT few-shot the
# top recipe (it would mislead the small local proposer); fall back to cold induction / the strategy
# solver / graph-explore instead. On _similarity's scale the dominant signal is an action-type match
# (+3): a top match that does not even share the action model is a risky transfer. Require >= that.
_CONFIDENT_TRANSFER_MIN_SIM = 3.0

FEATURE_ROUTER_APPROACHES: dict[str, dict[str, Any]] = {
    "default_graph_explore": {
        "entrypoint": "arc_graph_explore.graph_explore_solve_v2",
        "toolkit": "default no-regression fallback",
        "uses_llm": False,
    },
    "systematic_bfs": {
        "entrypoint": "arc_graph_explore.graph_explore_solve_v2",
        "toolkit": "systematic BFS over the frame graph",
        "uses_llm": False,
    },
    "diversity_graph_explore": {
        "entrypoint": "arc_graph_explore.graph_explore_solve_v2 + CARNOT_ARC_EXPLORE_DIVERSITY",
        "toolkit": "diversity-on-stall graph exploration",
        "uses_llm": False,
    },
    "goal_distance_astar": {
        "entrypoint": "arc_goal_distance.goal_distance_solve",
        "toolkit": "avatar+goal calibration with A*-ordered graph exploration",
        "uses_llm": False,
    },
    "llm_reasoner": {
        "entrypoint": "arc_llm_guided_solve.llm_guided_solve",
        "toolkit": "residual hard-tail reasoner selected only as an approach, not by an LLM router",
        "uses_llm": True,
    },
}

_FEATURE_ROUTER_DEFAULTS: dict[str, str] = {
    "avatar_navigation": "goal_distance_astar",
    "click_connect": "systematic_bfs",
    "config_toggle": "diversity_graph_explore",
    "hidden_carry_state": "llm_reasoner",
    "keyboard_graph": "systematic_bfs",
    "click_graph": "diversity_graph_explore",
    "unknown": "default_graph_explore",
}


def _grid_tuple(value: Any) -> tuple[tuple[Any, ...], ...]:
    """Return a hashable visible grid snapshot for early-play probes.

    The probe records only visible frame differences. It accepts raw list grids,
    numpy-like arrays, ARC frame objects, and small dictionaries from tests or
    cached traces; unreadable frames become an empty tuple instead of failing the
    router.
    """

    if value is None:
        return ()
    if isinstance(value, dict):
        value = value.get("grid", value.get("frame", value.get("cells", value)))
    elif hasattr(value, "frame"):
        value = getattr(value, "frame")
    try:
        rows = value.tolist() if hasattr(value, "tolist") else value
        return tuple(tuple(row) for row in rows)
    except Exception:
        return ()


def _changed_cells(
    before: tuple[tuple[Any, ...], ...], after: tuple[tuple[Any, ...], ...]
) -> list[tuple[int, int, Any, Any]]:
    if not before or not after or len(before) != len(after):
        return []
    changed: list[tuple[int, int, Any, Any]] = []
    for y, row in enumerate(before):
        if y >= len(after) or len(row) != len(after[y]):
            return []
        for x, value in enumerate(row):
            new_value = after[y][x]
            if value != new_value:
                changed.append((y, x, value, new_value))
    return changed


def _coords_by_value(grid: tuple[tuple[Any, ...], ...]) -> dict[Any, list[tuple[int, int]]]:
    coords: dict[Any, list[tuple[int, int]]] = {}
    for y, row in enumerate(grid):
        for x, value in enumerate(row):
            coords.setdefault(value, []).append((y, x))
    return coords


def _background_value(grid: tuple[tuple[Any, ...], ...]) -> Any:
    counts: dict[Any, int] = {}
    for row in grid:
        for value in row:
            counts[value] = counts.get(value, 0) + 1
    return max(counts, key=counts.get) if counts else None


def _translated_visible_object(
    before: tuple[tuple[Any, ...], ...], after: tuple[tuple[Any, ...], ...]
) -> bool:
    if not before or not after:
        return False
    bg = _background_value(before)
    before_coords = _coords_by_value(before)
    after_coords = _coords_by_value(after)
    for value, coords0 in before_coords.items():
        if value == bg or value not in after_coords:
            continue
        coords1 = after_coords[value]
        if not coords0 or len(coords0) != len(coords1):
            continue
        dy = coords1[0][0] - coords0[0][0]
        dx = coords1[0][1] - coords0[0][1]
        if dy == 0 and dx == 0:
            continue
        translated = sorted((y + dy, x + dx) for y, x in coords0)
        if translated == sorted(coords1):
            return True
    return False


def _is_click_action(action_id: Any, data: Any) -> bool:
    try:
        aid = int(action_id)
    except Exception:
        aid = -1
    return aid == 6 or isinstance(data, dict) and {"x", "y"} <= set(data)


def _is_keyboard_action(action_id: Any, data: Any) -> bool:
    try:
        aid = int(action_id)
    except Exception:
        return False
    return 1 <= aid <= 5 and not _is_click_action(action_id, data)


def _cell_connect_effect(changed: list[tuple[int, int, Any, Any]]) -> bool:
    if len(changed) < 2:
        return False
    new_values = {new for _y, _x, _old, new in changed}
    if len(new_values) != 1:
        return False
    cells = [(y, x) for y, x, _old, _new in changed]
    for y1, x1 in cells:
        for y2, x2 in cells:
            if (y1, x1) != (y2, x2) and abs(y1 - y2) + abs(x1 - x2) == 1:
                return True
    return False


def extract_early_play_signature(
    transitions: Any, *, k: int = 8
) -> dict[str, Any]:
    """REQ-CAPSTONE-4582: summarize the first K action effects for mechanic routing.

    The signature is intentionally behavioral: it counts which input families
    visibly changed the frame and records coarse effects such as translated
    avatar motion, click-connected cells, hidden carry-state, and reversible
    config toggles. It does not use the executable win-check and does not call
    an LLM.
    """

    if isinstance(transitions, dict) and "transitions" in transitions:
        transitions = transitions["transitions"]
    rows = list(transitions or [])[: max(0, int(k))]
    keyboard_effects = 0
    click_effects = 0
    avatar_motion = False
    cell_connect = False
    hidden_carry = False
    visible_pairs: list[tuple[Any, Any, tuple[tuple[Any, ...], ...], tuple[tuple[Any, ...], ...]]] = []
    seen_effects: dict[
        tuple[Any, Any, tuple[tuple[Any, ...], ...]],
        tuple[tuple[Any, ...], ...],
    ] = {}
    changed_counts: list[int] = []

    for row in rows:
        action_id = row.get("action_id", row.get("action", row.get("a"))) if isinstance(row, dict) else None
        data = row.get("data") if isinstance(row, dict) else None
        before = _grid_tuple(row.get("before", row.get("prev"))) if isinstance(row, dict) else ()
        after = _grid_tuple(row.get("after", row.get("cur", row.get("frame")))) if isinstance(row, dict) else ()
        changed = _changed_cells(before, after)
        changed_counts.append(len(changed))
        if changed:
            if _is_keyboard_action(action_id, data):
                keyboard_effects += 1
            if _is_click_action(action_id, data):
                click_effects += 1
            if _translated_visible_object(before, after):
                avatar_motion = True
            if _is_click_action(action_id, data) and _cell_connect_effect(changed):
                cell_connect = True
        if isinstance(row, dict) and (
            row.get("hidden_carry_state") is True
            or row.get("hidden_state_changed") is True
            or row.get("carry_state_changed") is True
        ):
            hidden_carry = True
        key = (action_id, json.dumps(data, sort_keys=True, default=str), before)
        previous_after = seen_effects.get(key)
        if previous_after is not None and previous_after != after:
            hidden_carry = True
        seen_effects[key] = after
        visible_pairs.append((action_id, json.dumps(data, sort_keys=True, default=str), before, after))

    config_toggle = False
    for index, (_aid, data_key, before, after) in enumerate(visible_pairs):
        if not before or not after or before == after:
            continue
        for other_aid, other_data_key, other_before, other_after in visible_pairs[index + 1 :]:
            if data_key == other_data_key and other_before == after and other_after == before:
                config_toggle = True
                break
        if config_toggle:
            break

    return {
        "probe_count": len(rows),
        "keyboard_effect_count": keyboard_effects,
        "click_effect_count": click_effects,
        "avatar_motion_present": avatar_motion,
        "cell_connect": cell_connect,
        "hidden_carry_state": hidden_carry,
        "config_toggle": config_toggle,
        "changed_cell_counts": changed_counts,
        "llm_used": False,
    }


def classify_early_play_mechanic(signature: Mapping[str, Any]) -> str:
    """REQ-CAPSTONE-4582: map early-play signatures to coarse mechanic classes."""

    if signature.get("hidden_carry_state") is True:
        return "hidden_carry_state"
    if signature.get("avatar_motion_present") is True and int(signature.get("keyboard_effect_count") or 0) > 0:
        return "avatar_navigation"
    if signature.get("config_toggle") is True:
        return "config_toggle"
    if signature.get("cell_connect") is True and int(signature.get("click_effect_count") or 0) > 0:
        return "click_connect"
    if int(signature.get("keyboard_effect_count") or 0) > 0:
        return "keyboard_graph"
    if int(signature.get("click_effect_count") or 0) > 0:
        return "click_graph"
    return "unknown"


def _coarse_mechanic_class(raw: Any, features: Mapping[str, Any] | None = None) -> str:
    text = str(raw or "").lower()
    if text in _FEATURE_ROUTER_DEFAULTS:
        return text
    if any(token in text for token in ("hidden", "carry", "checkpoint", "timed_trap")):
        return "hidden_carry_state"
    if any(token in text for token in ("config", "toggle", "program", "palette")):
        return "config_toggle"
    if any(token in text for token in ("connect", "slot", "fill", "merge", "drag")):
        return "click_connect"
    if any(token in text for token in ("navigation", "goal", "avatar", "graph_explore", "motion", "reflect")):
        return "avatar_navigation"
    features = features or {}
    action_type = str(features.get("action_type") or "").lower()
    if action_type == "keyboard":
        return "avatar_navigation"
    if action_type == "click":
        return "click_connect"
    return "unknown"


def _approach_from_trace(row: Mapping[str, Any]) -> str:
    explicit = str(row.get("approach") or row.get("route") or row.get("selected_approach") or "")
    if explicit in FEATURE_ROUTER_APPROACHES:
        return explicit
    winner = str(row.get("winner") or "").lower()
    if winner in {"cell_count", "region_count", "goal_distance", "goal_distance_astar"}:
        return "goal_distance_astar"
    if winner == "bfs":
        return "systematic_bfs"
    solver = str(row.get("solver") or row.get("executor") or row.get("closed_by_operator") or "").lower()
    if "goal_distance" in solver or "a_star" in solver or "astar" in solver:
        return "goal_distance_astar"
    if "llm" in solver or "reasoner" in solver:
        return "llm_reasoner"
    if "diversity" in solver or "go-explore" in solver:
        return "diversity_graph_explore"
    if "bfs" in solver or "graph" in solver or "explore" in solver:
        return "systematic_bfs"
    return ""


def _trace_solved(row: Mapping[str, Any]) -> bool:
    for key in ("solved", "reproduced", "offline_reproduced", "winner_generated"):
        if row.get(key) is True:
            return True
    outcome = row.get("outcome")
    if isinstance(outcome, Mapping):
        return any(outcome.get(key) is True for key in ("solved", "reproduced"))
    return str(row.get("result") or "").lower() in {"solved", "reproduced", "win", "won"}


def _positive_action_count(row: Mapping[str, Any]) -> float | None:
    for key in ("actions_to_first_levelup", "first_levelup_actions", "actions"):
        value = row.get(key)
        if isinstance(value, int | float) and not isinstance(value, bool) and value > 0:
            return float(value)
    return None


def _load_feature_router_trace_rows(root: Path | str = REPO) -> list[dict[str, Any]]:
    """Load local self-play/registry traces in a forgiving common row format."""

    root_path = Path(root)
    rows: list[dict[str, Any]] = []
    registry_path = root_path / "ops" / "arc_solve_registry.yaml"
    if registry_path.exists():
        try:
            reg = yaml.safe_load(registry_path.read_text(encoding="utf-8")) or {}
            for game in reg.get("games", []) or []:
                if not isinstance(game, Mapping):
                    continue
                rows.append(
                    {
                        "game": game.get("game"),
                        "mechanic_class": game.get("mechanic_class"),
                        "solver": game.get("solver"),
                        "solved": game.get("reproducibility") in {"reproduced", "provisional"},
                        "actions_to_first_levelup": game.get("actions_to_first_levelup"),
                    }
                )
        except Exception:
            pass
    ledger_path = root_path / "ops" / "arc_router_ledger.json"
    if ledger_path.exists():
        try:
            ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
            for entry in ledger.get("entries", []) or []:
                if not isinstance(entry, Mapping):
                    continue
                features = entry.get("features") if isinstance(entry.get("features"), Mapping) else {}
                outcomes = entry.get("outcomes") if isinstance(entry.get("outcomes"), Mapping) else {}
                for approach, outcome in outcomes.items():
                    if isinstance(outcome, Mapping):
                        rows.append(
                            {
                                "game": entry.get("game"),
                                "mechanic_class": _coarse_mechanic_class("", features),
                                "approach": "goal_distance_astar"
                                if approach in {"cell_count", "region_count"}
                                else "systematic_bfs",
                                "solved": outcome.get("reproduced") is True,
                                "actions": outcome.get("actions"),
                            }
                        )
        except Exception:
            pass
    for path in sorted((root_path / "results").glob("arc_loop_solve_*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(payload, Mapping):
            continue
        rows.append(
            {
                "game": payload.get("game") or path.stem.replace("arc_loop_solve_", ""),
                "mechanic_class": payload.get("mechanic_class"),
                "approach": payload.get("executor") or payload.get("solver"),
                "solved": payload.get("offline_reproduced") is True
                or int(payload.get("reached_level") or 0) > 0,
                "actions": payload.get("moves") or payload.get("actions"),
            }
        )
    return rows


def learn_feature_router_policy(trace_rows: Any | None = None) -> dict[str, Any]:
    """REQ-CAPSTONE-4582: learn mechanic-class to approach routes from traces.

    Positive rows add evidence for an approach; negative rows count against it.
    Ties keep lower median action cost first, then deterministic approach name.
    Missing trace evidence falls back to the conservative default map so the
    current shipped route remains available.
    """

    rows = list(trace_rows) if trace_rows is not None else _load_feature_router_trace_rows()
    stats: dict[str, dict[str, dict[str, Any]]] = {}
    usable_rows = 0
    for raw in rows:
        if not isinstance(raw, Mapping):
            continue
        features = raw.get("features") if isinstance(raw.get("features"), Mapping) else None
        mechanic = _coarse_mechanic_class(
            raw.get("mechanic_class", raw.get("mechanic")),
            features=features,
        )
        approach = _approach_from_trace(raw)
        if not approach:
            approach = _FEATURE_ROUTER_DEFAULTS.get(mechanic, "default_graph_explore")
        bucket = stats.setdefault(mechanic, {}).setdefault(
            approach,
            {"wins": 0, "losses": 0, "attempts": 0, "actions": []},
        )
        solved = _trace_solved(raw)
        bucket["attempts"] += 1
        bucket["wins" if solved else "losses"] += 1
        action_count = _positive_action_count(raw)
        if action_count is not None:
            bucket["actions"].append(action_count)
        usable_rows += 1

    routes: dict[str, dict[str, Any]] = {}
    for mechanic, default_approach in _FEATURE_ROUTER_DEFAULTS.items():
        candidates = stats.get(mechanic, {})
        if not candidates:
            routes[mechanic] = {
                "mechanic_class": mechanic,
                "approach": default_approach,
                "confidence": 0.2 if mechanic != "unknown" else 0.0,
                "wins": 0,
                "losses": 0,
                "attempts": 0,
                "median_actions": None,
                "source": "default_prior",
            }
            continue

        def rank(item: tuple[str, dict[str, Any]]) -> tuple[float, int, float, str]:
            approach, bucket = item
            actions = bucket["actions"]
            median_actions = statistics.median(actions) if actions else 1e9
            return (
                float(bucket["wins"]) - 0.5 * float(bucket["losses"]),
                int(bucket["wins"]),
                -float(median_actions),
                approach,
            )

        approach, bucket = max(candidates.items(), key=rank)
        attempts = int(bucket["attempts"])
        wins = int(bucket["wins"])
        confidence = round(float(wins) / float(attempts), 4) if attempts else 0.0
        actions = bucket["actions"]
        routes[mechanic] = {
            "mechanic_class": mechanic,
            "approach": approach,
            "confidence": confidence,
            "wins": wins,
            "losses": int(bucket["losses"]),
            "attempts": attempts,
            "median_actions": float(statistics.median(actions)) if actions else None,
            "source": "self_play_trace_rows",
        }
    return {
        "schema": "arc_feature_router_policy_v1",
        "routes": routes,
        "default_approach": "default_graph_explore",
        "trace_rows": usable_rows,
        "approaches": FEATURE_ROUTER_APPROACHES,
        "verifier_is_oracle": False,
    }


def route_feature_approach(
    early_play_signature: Mapping[str, Any] | list[Mapping[str, Any]],
    *,
    policy: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Classify an early-play signature and return the learned toolkit route."""

    signature = (
        extract_early_play_signature(early_play_signature)
        if isinstance(early_play_signature, list)
        else dict(early_play_signature)
    )
    mechanic = classify_early_play_mechanic(signature)
    policy_map = dict(policy or learn_feature_router_policy())
    routes = policy_map.get("routes") if isinstance(policy_map.get("routes"), Mapping) else {}
    route = dict(routes.get(mechanic) or routes.get("unknown") or {})
    approach = str(route.get("approach") or policy_map.get("default_approach") or "default_graph_explore")
    descriptor = FEATURE_ROUTER_APPROACHES.get(approach, FEATURE_ROUTER_APPROACHES["default_graph_explore"])
    return {
        "enabled": True,
        "mechanic_class": mechanic,
        "approach": approach,
        "confidence": float(route.get("confidence") or 0.0),
        "policy_source": route.get("source", "unknown"),
        "policy_trace_rows": int(policy_map.get("trace_rows") or 0),
        "signature": signature,
        "approach_descriptor": dict(descriptor),
        "verifier_is_oracle": False,
    }


def probe_early_play_signature(env: Any, *, k: int = 8) -> dict[str, Any]:  # pragma: no cover - ARC boundary
    """Probe the first few actions in a throwaway env and extract a feature signature."""

    from arcengine import GameAction

    try:
        from carnot.agentic.arc_agi3_world_model import grid_of
    except Exception:
        grid_of = lambda frame: getattr(frame, "frame", frame)  # noqa: E731

    transitions: list[dict[str, Any]] = []
    try:
        frame = env.reset()
    except Exception:
        return extract_early_play_signature([])
    action_ids = [1, 2, 3, 4, 5, 6, 1, 6][: max(0, int(k))]
    for aid in action_ids:
        before_grid = _grid_tuple(grid_of(frame))
        data = {"x": 32, "y": 32} if aid == 6 else None
        try:
            next_frame = env.step(getattr(GameAction, f"ACTION{aid}"), data=data)
        except TypeError:
            try:
                next_frame = env.step(getattr(GameAction, f"ACTION{aid}"))
                data = None
            except Exception:
                continue
        except Exception:
            continue
        transitions.append(
            {
                "action_id": aid,
                "data": data,
                "before": before_grid,
                "after": _grid_tuple(grid_of(next_frame)),
            }
        )
        if next_frame is not None:
            frame = next_frame
    return extract_early_play_signature(transitions)


def _cautions_from(ranked: list[dict], reg: dict) -> list[str]:
    """Aggregate failure-derived CAUTIONS (FinAcumen's Cautions, the complement of the success
    Findings) from the top matched games' recorded dead-ends + the registry general gotchas, so the
    runtime induction prompt can be told what NOT to do, not just what worked. Deduplicated, order-
    preserving."""
    out: list[str] = []
    seen: set[str] = set()

    def _add(items: Any) -> None:
        for it in items or []:
            s = it if isinstance(it, str) else (it.get("note") or it.get("dead_end") or str(it))
            s = str(s).strip()
            if s and s not in seen:
                seen.add(s)
                out.append(s)

    by_game = {g.get("game"): g for g in reg.get("games", [])}
    for r in ranked[:3]:  # only the games we'd actually transfer from
        g = by_game.get(r.get("game"), {})
        _add(g.get("dead_ends"))
        _add(g.get("dead_ends_recorded"))
    _add(reg.get("general_gotchas"))
    return out[:8]  # cap (FinAcumen k_max-style): a few high-signal cautions, not a flood


def _feature_router_payload(
    *,
    early_play_signature: Mapping[str, Any] | list[Mapping[str, Any]] | None,
    feature_router_policy: Mapping[str, Any] | None,
    feature_router_traces: Any | None,
    fallback_strategy: Mapping[str, Any],
) -> dict[str, Any]:
    fallback = str(fallback_strategy.get("solver") or "arc_graph_explore.graph_explore_solve_v2")
    if early_play_signature is None:
        return {
            "enabled": False,
            "reason": "early_play_signature_not_supplied",
            "approach": "default_graph_explore",
            "mechanic_class": "",
            "confidence": 0.0,
            "no_regression_fallback": fallback,
            "verifier_is_oracle": False,
        }
    policy = feature_router_policy or learn_feature_router_policy(feature_router_traces)
    routed = route_feature_approach(early_play_signature, policy=policy)
    routed["no_regression_fallback"] = fallback
    routed["reason"] = (
        f"{routed['mechanic_class']} -> {routed['approach']} from early-play feature signature"
    )
    return routed


def recommend_approach(
    target_game: str,
    *,
    mechanic: Optional[str] = None,
    early_play_signature: Mapping[str, Any] | list[Mapping[str, Any]] | None = None,
    feature_router_policy: Mapping[str, Any] | None = None,
    feature_router_traces: Any | None = None,
) -> dict:
    """Route a NEW game to the closest proven recipe. Returns the ranked solved
    games with their registry recipe (solver, win-condition, action-model,
    reusable gotchas) + the general gotchas + the matched games' dead-ends.

    The FIRST routing decision is the STRATEGY CLASS (arc_strategy_router): a
    program-editor game routes to the frame-only program-editor model and SKIPS
    the goal-distance heuristic portfolio (which only applies to graph-explore);
    a graph-explore game gets the heuristic policy as before. For an unseen live
    game, pass the frame-only-detected class via `mechanic=` (else it is read
    from the registry's structured `mechanic_class`, defaulting to graph_explore).

    Call this BEFORE reverse-engineering a new game (CLAUDE.md ARC Solve
    Reproducibility + Solver-Reuse Discipline)."""
    from . import arc_strategy_router as strat

    feats = _survey_features()
    reg = _registry()
    strategy = strat.route_for_game(target_game, mechanic=mechanic, reg=reg)
    typed_memory_guard = typed_memory_provenance_guard()
    feature_router = _feature_router_payload(
        early_play_signature=early_play_signature,
        feature_router_policy=feature_router_policy,
        feature_router_traces=feature_router_traces,
        fallback_strategy=strategy,
    )
    if target_game not in feats:
        from . import arc_solver_kit as kit
        from . import arc_primitive_library as primitive_library

        # Unseen LIVE game (not in the public survey): no feature-based similarity transfer is
        # possible, so route COLD via the strategy + generic operators + cautions (FinAcumen: do not
        # fabricate a confident transfer when there is no relevant match).
        primitive_digest = {
            "game": target_game,
            "mechanic_class": strategy.get("routed_mechanic", ""),
            "action_model": "",
        }
        return {
            "error": f"{target_game} not in survey",
            "strategy": strategy,
            "typed_memory_provenance_guard": typed_memory_guard,
            "feature_router": feature_router,
            "retrieved_primitives": primitive_library.retrieve_primitives(primitive_digest),
            "selected_generic_operators": [
                op.as_dict()
                for op in kit.select_primitive_operators(
                    mechanic_class=strategy.get("routed_mechanic", ""), game=target_game
                )
            ],
            "confident_transfer": False,
            "routing_confidence": 0.0,
            "top_similarity": 0.0,
            "cautions": _cautions_from([], reg),
            "general_gotchas": reg.get("general_gotchas", []),
        }
    tf = feats[target_game]
    from . import arc_solver_kit as kit
    from . import arc_primitive_library as primitive_library

    by_game = {g["game"]: g for g in reg.get("games", [])}
    if target_game in by_game:
        primitive_digest = primitive_library.game_digest(by_game[target_game])
    else:
        primitive_digest = {
            "game": target_game,
            "mechanic_class": strategy.get("routed_mechanic", ""),
            "action_model": str(tf.get("action_type", "")),
            "target_features": {**tf, "win_kw": sorted(tf["win_kw"])},
        }
    retrieved_primitives = primitive_library.retrieve_primitives(
        primitive_digest,
        exclude_games=(target_game,),
    )
    selected_generic_operators = [
        op.as_dict()
        for op in kit.select_primitive_operators(
            mechanic_class=strategy.get("routed_mechanic", ""),
            action_model=str(tf.get("action_type", "")),
            game=target_game,
        )
    ]
    ranked = []
    for solved in _solved_games(reg):
        gid = solved["game"]
        if gid == target_game or gid not in feats:
            continue
        sim = _similarity(tf, feats[gid])
        ranked.append(
            {
                "game": gid,
                "similarity": round(sim, 2),
                "reproducibility": solved.get("reproducibility"),
                "solver": solved.get("solver"),
                "win_condition": solved.get("win_condition"),
                "action_model": solved.get("action_model"),
                "reusable_gotchas": solved.get("gotchas", []),
            }
        )
    ranked.sort(key=lambda r: r["similarity"], reverse=True)
    # The goal-distance heuristic portfolio only applies to the graph-explore class. For a
    # program-editor (or other non-graph-explore) game it is a category error — surface the strategy's
    # own solver instead, so the agent does not waste a portfolio run that can never win.
    if strategy.get("uses_goal_distance_heuristic"):
        policy = _heuristic_policy()
    else:
        policy = {
            "not_applicable": (
                f"goal-distance heuristics do not apply to the "
                f"{strategy['routed_mechanic']} class; use the strategy solver"
            ),
            "strategy_solver": strategy.get("solver"),
            "search_engine": strategy.get("search_engine"),
            "needs": strategy.get("needs"),
        }
    top_sim = ranked[0]["similarity"] if ranked else 0.0
    confident_transfer = top_sim >= _CONFIDENT_TRANSFER_MIN_SIM
    # routing_confidence: top_sim normalized to [0,1] against the confident bar (clamped) -- a
    # monotone, interpretable proxy, NOT a probability.
    routing_confidence = round(min(1.0, top_sim / (_CONFIDENT_TRANSFER_MIN_SIM * 2)), 2)
    return {
        "target_game": target_game,
        "target_features": {**tf, "win_kw": sorted(tf["win_kw"])},
        "strategy": strategy,
        "typed_memory_provenance_guard": typed_memory_guard,
        "feature_router": feature_router,
        "retrieved_primitives": retrieved_primitives,
        "selected_generic_operators": selected_generic_operators,
        "recommended": ranked[:3],
        # FinAcumen (arXiv:2606.17642) selective-activation: only few-shot the top recipe when the
        # match clears the confidence bar; below it, an irrelevant example MISLEADS the small proposer.
        "confident_transfer": confident_transfer,
        "routing_confidence": routing_confidence,
        "top_similarity": round(top_sim, 2),
        # Failure-derived CAUTIONS (what NOT to do), the complement of the success recipe, for the
        # runtime induction prompt.
        "cautions": _cautions_from(ranked, reg),
        "heuristic_policy": policy,
        "general_gotchas": reg.get("general_gotchas", []),
        "guidance": (
            ("CONFIDENT transfer (top match cleared the bar): start from the top-ranked solved game's "
             "solver + reuse its action-model and gotchas, only reverse-engineering the DELTA; few-shot "
             "the runtime proposer with that recipe AND the `cautions`. "
             if confident_transfer else
             "LOW-confidence routing (no solved game cleared the transfer bar): do NOT blind-copy the "
             "top recipe -- an irrelevant example misleads the proposer (FinAcumen arXiv:2606.17642). "
             "Induce COLD via the routed strategy.solver / graph-explore, using only the `cautions` + "
             "general_gotchas as guardrails. ")
            + "FIRST follow the routed STRATEGY (strategy.solver). Import arc_solver_kit; run the "
            "reproduction gate; append new mechanics/dead-ends to the registry."
        ),
    }


def _heuristic_policy() -> dict:
    """The learned WHEN-to-use-which-heuristic policy (arc_heuristic_select). The choice is
    DATA-DRIVEN per game and cannot be decided from survey features alone (it needs the per-
    action cell-impact, measured from a few transitions) — so the policy is: run the portfolio
    selector ONCE a win-state is available; for a first-ever solve (no target) use pure BFS."""
    return {
        "trained_router": "arc_router.route(features, arc_router.train()) — learned decision tree "
        "(thresholds from ops/arc_router_ledger.json; 8/8 leave-one-out); "
        "predicts approach + explore/exploit by novelty",
        "selector": "arc_heuristic_select.select_and_learn(game, win, transitions, mask_hud=...) "
        "— runs the portfolio, banks the winner, records to the router ledger (online)",
        "feature": "per-action cell impact + bfs_expansions headroom probe",
        "rule": {
            "no_win_state_yet (first solve)": "pure BFS — a goal-distance heuristic needs a target",
            "low search headroom (BFS solves cheaply)": "BFS — no room for a heuristic to help",
            "high cell-impact (>= learned ~36 cells/action)": "misplaced_region_distance (8-conn)",
            "low cell-impact (< learned threshold)": "cell_count_distance (Hamming)",
        },
        "default_if_unmeasured": (
            "region_count — it NEVER regressed across the 8-game validation "
            "and wins the high-impact games; cell_count only when low-impact"
        ),
        "captured": "reuse gap_fills/<game>_goal_distance.py first (no recompute) when present",
    }


if __name__ == "__main__":  # pragma: no cover - manual probe
    import sys

    print(json.dumps(recommend_approach(sys.argv[1] if len(sys.argv) > 1 else "vc33"), indent=2))
