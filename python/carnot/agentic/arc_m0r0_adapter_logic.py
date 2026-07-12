"""m0r0 visible-state adapter logic for graph-explore L1->L2 deepening.

Spec refs: REQ-ARC-WMTE-4515, SCENARIO-ARC-WMTE-4515.
"""

from __future__ import annotations

from collections import Counter, deque
import json
from typing import Any

from arcengine import GameAction

PLAYER_NAMES = (
    "pikgci-boweok-leklkn",
    "pikgci-boweok-rivmdg",
    "pikgci-toljda-leklkn",
    "pikgci-toljda-rivmdg",
)
MOVEMENT_ACTIONS = (1, 2, 3, 4)
ACTION_DELTAS = {
    1: (0, -1),
    2: (0, 1),
    3: (-1, 0),
    4: (1, 0),
}


def m0r0_label(action: int) -> str:
    return json.dumps({"action": int(action)}, sort_keys=True, separators=(",", ":"))


def _cells_for_sprite(sprite: Any) -> tuple[tuple[int, int], ...]:
    pixels = getattr(sprite, "pixels", [[0]])
    rotation = int(getattr(sprite, "rotation", 0) or 0) % 360
    rows = [list(row) for row in pixels]
    for _ in range((rotation // 90) % 4):
        rows = [list(row) for row in zip(*rows[::-1])]
    cells: list[tuple[int, int]] = []
    for dy, row in enumerate(rows):
        for dx, value in enumerate(row):
            if int(value) != -1:
                cells.append((int(getattr(sprite, "x", 0)) + dx, int(getattr(sprite, "y", 0)) + dy))
    return tuple(cells)


def _level(game: Any) -> Any:
    return getattr(game, "current_level", None)


def _sprites_by_name(game: Any, name: str) -> list[Any]:
    level = _level(game)
    if level is None:
        return []
    try:
        return list(level.get_sprites_by_name(name))
    except Exception:
        return []


def _sprites_by_tag(game: Any, tag: str) -> list[Any]:
    level = _level(game)
    if level is None:
        return []
    try:
        return list(level.get_sprites_by_tag(tag))
    except Exception:
        return []


def _is_inactive(sprite: Any) -> bool:
    interaction = str(getattr(sprite, "interaction", "")).upper()
    return "INTANGIBLE" in interaction or "REMOVED" in interaction


def m0r0_active_players(game: Any) -> tuple[tuple[str, int, int], ...]:
    retired = set(getattr(game, "okpvcjupabr", set()) or set())
    players: list[tuple[str, int, int]] = []
    for name in PLAYER_NAMES:
        if name in retired:
            continue
        for sprite in _sprites_by_name(game, name):
            if not _is_inactive(sprite):
                players.append((name, int(sprite.x), int(sprite.y)))
                break
    return tuple(sorted(players))


def m0r0_bad_cells(game: Any) -> tuple[tuple[int, int], ...]:
    cells: set[tuple[int, int]] = set()
    for sprite in _sprites_by_tag(game, "spswjz"):
        cells.update(_cells_for_sprite(sprite))
    return tuple(sorted(cells))


def m0r0_visible_walls(game: Any) -> tuple[tuple[int, int], ...]:
    cells: set[tuple[int, int]] = set()
    for tag in ("wahtyt", "xbso"):
        for sprite in _sprites_by_tag(game, tag):
            cells.update(_cells_for_sprite(sprite))
    return tuple(sorted(cells))


def _grid_size(game: Any, cells: set[tuple[int, int]]) -> tuple[int, int]:
    size = getattr(_level(game), "grid_size", None)
    if size:
        return int(size[0]), int(size[1])
    if cells:
        return max(x for x, _ in cells) + 1, max(y for _, y in cells) + 1
    return 64, 64


def _delta_for_player(name: str, action: int) -> tuple[int, int]:
    dx, dy = ACTION_DELTAS[int(action)]
    if "rivmdg" in name:
        dx = -dx
    if "boweok" in name:
        dy = -dy
    return dx, dy


def _move_cell(
    cell: tuple[int, int],
    delta: tuple[int, int],
    *,
    grid_size: tuple[int, int],
    walls: set[tuple[int, int]],
) -> tuple[int, int]:
    nx = cell[0] + delta[0]
    ny = cell[1] + delta[1]
    width, height = grid_size
    if nx < 0 or nx >= width or ny < 0 or ny >= height or (nx, ny) in walls:
        return cell
    return nx, ny


def _apply_pair_crossing(
    before: tuple[tuple[str, int, int], ...],
    after: list[tuple[str, int, int]],
) -> tuple[tuple[str, int, int], ...]:
    previous = {name: (x, y) for name, x, y in before}
    current = {name: (x, y) for name, x, y in after}
    names = [name for name, _x, _y in after]
    for index, left_name in enumerate(names):
        for right_name in names[index + 1 :]:
            left_prev = previous[left_name]
            right_prev = previous[right_name]
            if abs(left_prev[0] - right_prev[0]) != 1 or left_prev[1] != right_prev[1]:
                continue
            left_now = current[left_name]
            right_now = current[right_name]
            if left_now == right_prev or right_now == left_prev:
                midpoint = ((left_now[0] + right_now[0]) // 2, (left_now[1] + right_now[1]) // 2)
                current[left_name] = midpoint
                current[right_name] = midpoint
    return tuple(sorted((name, xy[0], xy[1]) for name, xy in current.items()))


def _all_paired(players: tuple[tuple[str, int, int], ...]) -> bool:
    if len(players) < 2:
        return True
    counts = Counter((x, y) for _name, x, y in players)
    return all(count >= 2 for count in counts.values())


def _advance_state(
    players: tuple[tuple[str, int, int], ...],
    action: int,
    *,
    grid_size: tuple[int, int],
    walls: set[tuple[int, int]],
    bad_cells: set[tuple[int, int]],
) -> tuple[tuple[str, int, int], ...] | None:
    moved: list[tuple[str, int, int]] = []
    for name, x, y in players:
        nx, ny = _move_cell(
            (x, y), _delta_for_player(name, action), grid_size=grid_size, walls=walls
        )
        if (nx, ny) in bad_cells:
            return None
        moved.append((name, nx, ny))
    return _apply_pair_crossing(players, moved)


def m0r0_visible_plan_actions(game: Any, *, depth_cap: int = 80) -> list[int] | None:
    players = m0r0_active_players(game)
    if not players:
        return None
    if _all_paired(players):
        return []
    walls = set(m0r0_visible_walls(game))
    bad_cells = set(m0r0_bad_cells(game))
    grid_size = _grid_size(game, walls | bad_cells | {(x, y) for _name, x, y in players})
    queue: deque[tuple[tuple[tuple[str, int, int], ...], list[int]]] = deque([(players, [])])
    seen = {players}
    while queue:
        state, path = queue.popleft()
        if len(path) >= depth_cap:
            continue
        for action in MOVEMENT_ACTIONS:
            child = _advance_state(
                state,
                action,
                grid_size=grid_size,
                walls=walls,
                bad_cells=bad_cells,
            )
            if child is None or child in seen:
                continue
            next_path = path + [action]
            if _all_paired(child):
                return next_path
            seen.add(child)
            queue.append((child, next_path))
    return None


def m0r0_action_labels(env: Any, frame: Any = None, path: tuple[str, ...] = ()) -> list[str]:
    del path
    game = getattr(env, "_game", env)
    plan = m0r0_visible_plan_actions(game)
    if plan:
        return [m0r0_label(plan[0])]
    actions = list(
        getattr(frame, "available_actions", []) or getattr(game, "available_actions", []) or []
    )
    moves = [int(action) for action in actions if int(action) in MOVEMENT_ACTIONS] or list(
        MOVEMENT_ACTIONS
    )
    return [m0r0_label(action) for action in moves]


def m0r0_apply(env: Any, label: str, frame: Any) -> Any:
    del frame
    from carnot.agentic.arc_agi3_live_adapter import _game_action

    parsed = json.loads(label)
    return env.step(_game_action(GameAction, int(parsed["action"])), data=parsed.get("data"))


def m0r0_state_key(game: Any, frame: Any = None) -> tuple[tuple[str, Any], ...]:
    level = int(getattr(frame, "levels_completed", 0) or 0) if frame is not None else 0
    return (
        ("level", level),
        ("players", m0r0_active_players(game)),
        ("walls", m0r0_visible_walls(game)),
        ("bad", m0r0_bad_cells(game)),
        ("retired", tuple(sorted(getattr(game, "okpvcjupabr", set()) or set()))),
        ("lose_animation", int(getattr(game, "ukempikfmtm", -1) or -1)),
    )


def m0r0_hand_verifier(game: Any, frame: Any = None) -> float:
    if frame is not None and int(getattr(frame, "levels_completed", 0) or 0) > 1:
        return 0.0
    plan = m0r0_visible_plan_actions(game)
    if plan is not None:
        return float(len(plan))
    players = m0r0_active_players(game)
    if len(players) < 2:
        return 1000.0
    xs = [x for _name, x, _y in players]
    ys = [y for _name, _x, y in players]
    return float((max(xs) - min(xs)) + (max(ys) - min(ys)) + 100)
