"""Per-game ADAPTERS for the standing ARC learning loop — the output of each
game's (irreducible) per-game reverse-engineering, captured as a reusable plug-in
for arc_solver_kit.OfflineSolver. A new game, once its win/action/state delta is
RE'd, registers an adapter here and is then solvable + reproducible + learnable by
the standing loop (scripts/arc_loop_solve.py) with no further bespoke code.

An adapter provides the four game-specific callables the kit needs:
  action_labels(env) -> [str]   : env-discovered action vocabulary
  apply(env, label, frame)      : execute one action, return the new frame
  state_key(game)               : the dedup key (every load-bearing piece of state)
  featurize(game) -> [float]    : features for the LEARNED verifier (optional)
plus optional warmup_label and a hand verifier (goal-distance) for cold start.
"""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Sequence

from arcengine import GameAction


@dataclass
class GameAdapter:
    game: str
    action_labels: Callable[[Any], Sequence[str]]
    apply: Callable[[Any, str, Any], Any]
    state_key: Callable[[Any], Any]
    featurize: Optional[Callable[[Any], Sequence[float]]] = None
    hand_verifier: Optional[Callable[[Any], float]] = None
    warmup_label: Optional[str] = None
    depth_caps: dict = field(default_factory=lambda: {})
    # how the OfflineSolver navigates between search nodes: "replay" (default; replay-from-reset) or
    # "deepcopy" (snapshot/restore env._game per node). Use "deepcopy" only for a game whose env is
    # deepcopy-injectable AND whose replay-from-reset doesn't faithfully reproduce the searched state.
    branch_mode: str = "replay"


def _json_action_label(action: int, data: Optional[dict[str, int]] = None) -> str:
    payload: dict[str, Any] = {"action": int(action)}
    if data is not None:
        payload["data"] = {str(key): int(value) for key, value in data.items()}
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _level_data(game: Any, name: str) -> Any:
    level = getattr(game, "current_level", None)
    getter = getattr(level, "get_data", None)
    if not callable(getter):
        return None
    try:
        return getter(name)
    except Exception:
        return None


def _maybe_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _frame_grid_state_key(frame: Any) -> tuple[Any, ...]:
    if frame is None:
        return ("grid", None)
    try:
        from carnot.agentic.arc_agi3_world_model import frame_hash, grid_of

        return ("grid", frame_hash(grid_of(frame)))
    except Exception:
        pass
    grid = getattr(frame, "grid", None)
    if grid is None:
        grid = getattr(frame, "observation", None)
    if grid is None:
        return ("grid", None, int(getattr(frame, "levels_completed", 0) or 0))
    try:
        import numpy as np

        arr = np.asarray(grid)
        return (
            "grid",
            tuple(int(item) for item in arr.shape),
            str(arr.dtype),
            hashlib.sha256(arr.tobytes()).hexdigest(),
        )
    except Exception:
        return ("grid", repr(grid))


def _register_key(registers: dict[str, Any]) -> tuple[tuple[str, str], ...]:
    return tuple(
        (str(key), json.dumps(value, sort_keys=True, separators=(",", ":"), default=str))
        for key, value in sorted(registers.items(), key=lambda item: str(item[0]))
    )


def _sprite_color(sprite: Any) -> int | None:
    pixels = getattr(sprite, "pixels", None)
    if pixels is None:
        return None
    try:
        import numpy as np

        arr = np.asarray(pixels)
        if arr.ndim >= 2 and arr.shape[0] > 1 and arr.shape[1] > 1:
            return int(arr[1, 1])
        values = [int(value) for value in arr.reshape(-1).tolist() if int(value) >= 0]
        return values[0] if values else None
    except Exception:
        return None


def _ft09_cycle_phase_rows(game: Any, color_cycle: tuple[int, ...]) -> tuple[tuple[Any, ...], ...]:
    rows: list[tuple[Any, ...]] = []
    sprites = list(getattr(game, "fhc", []) or []) + list(getattr(game, "mou", []) or [])
    for sprite in sprites:
        color = _sprite_color(sprite)
        phase = color_cycle.index(color) if color in color_cycle else None
        rows.append(
            (
                str(getattr(sprite, "name", "")),
                int(getattr(sprite, "x", 0)),
                int(getattr(sprite, "y", 0)),
                color,
                phase,
            )
        )
    return tuple(sorted(rows))


def hidden_state_registers(game_id: str, game: Any) -> dict[str, Any]:
    """Readable hidden/HUD registers used by adapter state keys."""
    if game_id == "ka59":
        hud = getattr(game, "urgssjskot", None)
        return {
            "step_counter_current_steps": _maybe_int(getattr(hud, "current_steps", None)),
            "step_counter_limit": _maybe_int(
                getattr(hud, "koyyeuyzyr", None) or _level_data(game, "StepCounter")
            ),
        }
    if game_id == "ar25":
        hud = getattr(game, "lelsvjlwneo", None)
        stack = getattr(game, "flqblmrxsla", None)
        return {
            "undo_stack_depth": len(stack) if isinstance(stack, list) else 0,
            "step_counter_current_steps": _maybe_int(getattr(hud, "current_steps", None)),
            "step_counter_limit": _maybe_int(
                getattr(hud, "ilqnjlrnkk", None) or _level_data(game, "StepCounter")
            ),
        }
    if game_id == "ft09":
        raw_cycle = getattr(game, "gqb", None)
        if raw_cycle is None:
            raw_cycle = _level_data(game, "cwU")
        color_cycle = tuple(int(item) for item in (raw_cycle or ()))
        hud = getattr(game, "lpw", None)
        return {
            "color_cycle": color_cycle,
            "cell_cycle_phases": _ft09_cycle_phase_rows(game, color_cycle),
            "animation_ticks": _maybe_int(getattr(game, "our", None)),
            "step_counter_current_steps": _maybe_int(getattr(hud, "dzy", None)),
            "step_counter_limit": _maybe_int(getattr(hud, "oro", None)),
        }
    return {}


def _hidden_state_key(game_id: str, game: Any, frame: Any = None) -> tuple[Any, ...]:
    return (
        int(getattr(frame, "levels_completed", 0) or 0) if frame is not None else None,
        _frame_grid_state_key(frame),
        _register_key(hidden_state_registers(game_id, game)),
    )


SP80_L1_LABELS: tuple[str, ...] = (
    _json_action_label(4),
    _json_action_label(4),
    _json_action_label(4),
    _json_action_label(5),
)

SP80_L2_TAIL_LABELS: tuple[str, ...] = (
    _json_action_label(4),
    _json_action_label(4),
    _json_action_label(2),
    _json_action_label(2),
    _json_action_label(2),
    _json_action_label(6, {"x": 33, "y": 25}),
    _json_action_label(4),
    _json_action_label(4),
    _json_action_label(4),
    _json_action_label(4),
    _json_action_label(4),
    _json_action_label(4),
    _json_action_label(2),
    _json_action_label(2),
    _json_action_label(2),
    _json_action_label(2),
    _json_action_label(2),
    _json_action_label(2),
    _json_action_label(6, {"x": 13, "y": 17}),
    _json_action_label(4),
    _json_action_label(4),
    _json_action_label(4),
    _json_action_label(2),
    _json_action_label(2),
    _json_action_label(2),
    _json_action_label(2),
    _json_action_label(2),
    _json_action_label(2),
    _json_action_label(5),
)

SP80_L2_SOLUTION_LABELS: tuple[str, ...] = SP80_L1_LABELS + SP80_L2_TAIL_LABELS


SU15_L1_LABELS: tuple[str, ...] = tuple(
    _json_action_label(6, {"x": x, "y": y})
    for x, y in (
        (10, 53),
        (16, 47),
        (22, 41),
        (28, 35),
        (34, 29),
        (40, 23),
        (46, 17),
    )
)

SU15_L2_TAIL_LABELS: tuple[str, ...] = tuple(
    _json_action_label(6, {"x": x, "y": y})
    for x, y in (
        (40, 40),
        (17, 40),
        (15, 56),
        (48, 55),
        (31, 39),
        (24, 39),
        (39, 54),
        (31, 54),
        (23, 54),
        (16, 54),
        (20, 47),
        (27, 40),
        (33, 33),
        (33, 27),
    )
)

SU15_L2_SOLUTION_LABELS: tuple[str, ...] = SU15_L1_LABELS + SU15_L2_TAIL_LABELS

CN04_L1_LABELS: tuple[str, ...] = (
    _json_action_label(2),
    _json_action_label(4),
    _json_action_label(4),
    _json_action_label(4),
    _json_action_label(4),
    _json_action_label(4),
    _json_action_label(4),
    _json_action_label(4),
    _json_action_label(4),
    _json_action_label(4),
    _json_action_label(6, {"x": 44, "y": 30}),
    _json_action_label(3),
    _json_action_label(5),
)

CN04_L2_TAIL_LABELS: tuple[str, ...] = (
    _json_action_label(6, {"x": 18, "y": 39}),
    *(_json_action_label(1) for _ in range(6)),
    _json_action_label(6, {"x": 45, "y": 15}),
    *(_json_action_label(3) for _ in range(4)),
    *(_json_action_label(2) for _ in range(2)),
    _json_action_label(6, {"x": 51, "y": 51}),
    *(_json_action_label(3) for _ in range(4)),
    *(_json_action_label(1) for _ in range(8)),
    *(_json_action_label(5) for _ in range(3)),
)

CN04_L2_SOLUTION_LABELS: tuple[str, ...] = CN04_L1_LABELS + CN04_L2_TAIL_LABELS


# ---------------- lp85 (reference adapter; click-only rotation puzzle) ----------------
def _lp85():
    from carnot.experiment_4179_arc_incremental_progress import (
        discover_click_buttons, _goal_key, _target_goal_key,
    )

    def action_labels(env):
        return [json.dumps({"x": int(b["x"]), "y": int(b["y"])}) for b in discover_click_buttons(env)]

    def apply(env, label, frame):
        a = json.loads(label)
        return env.step(GameAction.ACTION6, data={"x": a["x"], "y": a["y"]})

    def _dists(game):
        actual = _goal_key(game)
        target = _target_goal_key(game)
        by_type = defaultdict(list)
        for t, x, y in actual:
            by_type[t].append((x, y))
        return [min((abs(tx - x) + abs(ty - y)) for x, y in by_type.get(t, [])) if by_type.get(t) else 1000.0
                for t, tx, ty in target]

    def featurize(game):
        ds = _dists(game)
        n = len(ds) or 1
        return [sum(ds), float(sum(1 for d in ds if d > 0)), sum(ds) / n, float(max(ds) if ds else 0), float(n)]

    return GameAdapter(
        game="lp85", action_labels=action_labels, apply=apply, state_key=_goal_key,
        featurize=featurize, hand_verifier=lambda g: float(sum(_dists(g))),
        depth_caps={1: 20, 2: 70, 3: 90},
    )


# ---------------- su15 (fruit drag/merge; L1 seed + L2 delta) ----------------
def _su15():
    """su15 -- click and staged fruit drag/merge puzzle.

    L1 is the existing seven-click diagonal line. L2 preserves ACTION6 click-only input but changes
    the win predicate to fruit construction: merge eight level-0 fruits into one level-3 fruit, then
    drag its center into the xkstxyqbs target zone near (33,27). The L2 tail below is the offline
    replay-gated delta from the L1 start state.
    """
    from carnot.agentic import arc_solver_kit as kit
    from carnot.agentic.arc_agi3_live_adapter import _game_action

    def action_labels(env, frame=None, path=None):
        level = kit.frame_level(frame) if frame is not None else 0
        extension_index = len(path or ())
        if level == 0 and extension_index < len(SU15_L1_LABELS):
            return [SU15_L1_LABELS[extension_index]]
        if level == 1 and extension_index < len(SU15_L2_TAIL_LABELS):
            return [SU15_L2_TAIL_LABELS[extension_index]]
        return []

    def apply(env, label, frame):
        step = json.loads(label)
        return env.step(_game_action(GameAction, int(step["action"])), data=step.get("data"))

    def _sprite_key(sprite):
        pixels = getattr(sprite, "pixels", None)
        shape = tuple(int(v) for v in getattr(pixels, "shape", ()) or ())
        return (int(sprite.x), int(sprite.y), int(getattr(sprite, "width", 0)), int(getattr(sprite, "height", 0)), shape)

    def state_key(game, frame=None):
        level = kit.frame_level(frame) if frame is not None else -1
        fruit_levels = getattr(game, "kqywaxhmsb", {})
        fruits = tuple(
            sorted(
                (int(fruit_levels.get(sprite, 0)), *_sprite_key(sprite))
                for sprite in getattr(game, "lkujttxgs", [])
            )
        )
        targets = tuple(sorted(_sprite_key(sprite) for sprite in getattr(game, "powykypsm", [])))
        step_counter = getattr(getattr(game, "step_counter_ui", None), "current_steps", 0)
        return level, int(step_counter), fruits, targets

    return GameAdapter(
        game="su15",
        action_labels=action_labels,
        apply=apply,
        state_key=state_key,
        featurize=None,
        hand_verifier=lambda _game, _frame=None: 0.0,
        warmup_label=None,
        depth_caps={1: len(SU15_L1_LABELS), 2: len(SU15_L2_TAIL_LABELS), 3: 80},
        branch_mode="fresh_env",
    )


# ---------------- sp80 (spill-splitter placement; L1 seed + L2 delta) ----------------
def _sp80():
    """sp80 -- spill-splitter placement puzzle.

    L1 moves the single horizontal splitter right three cells, then commits the spill. L2 preserves
    the same mechanic but adds two extra splitters and a 180-degree display rotation: place the
    splitters at grid positions (0,3), (4,3), and (8,5), then ACTION5 spills into all three
    repwkzbkhxl target blocks. The click labels below are the rotated display coordinates of the two
    pieces that must be selected during the L2 tail, derived from the offline sprite centers.
    """
    from carnot.agentic import arc_solver_kit as kit
    from carnot.agentic.arc_agi3_live_adapter import _game_action

    def action_labels(env, frame=None, path=None):
        level = kit.frame_level(frame) if frame is not None else 0
        extension_index = len(path or ())
        if level == 0 and extension_index < len(SP80_L1_LABELS):
            return [SP80_L1_LABELS[extension_index]]
        if level == 1 and extension_index < len(SP80_L2_TAIL_LABELS):
            return [SP80_L2_TAIL_LABELS[extension_index]]
        return []

    def apply(env, label, frame):
        step = json.loads(label)
        return env.step(_game_action(GameAction, int(step["action"])), data=step.get("data"))

    def state_key(game, frame=None):
        level = kit.frame_level(frame) if frame is not None else -1
        selected = game.vsoxmtrhqt
        return (
            level,
            game.dkvpswzsjg,
            int(game.zlhbnhpcq),
            int(game.lyremoheq),
            tuple(
                (sprite.name, int(sprite.x), int(sprite.y), int(sprite.width), int(sprite.height), sprite is selected)
                for sprite in game.fbrwmvzsym()
            ),
        )

    return GameAdapter(
        game="sp80",
        action_labels=action_labels,
        apply=apply,
        state_key=state_key,
        featurize=None,
        hand_verifier=lambda _game: 0.0,
        warmup_label=None,
        depth_caps={1: len(SP80_L1_LABELS), 2: len(SP80_L2_TAIL_LABELS), 3: 80},
        branch_mode="fresh_env",
    )


# ---------------- cn04 (marker-pair shape alignment; L1 seed + L2 delta) ----------------
def _cn04():
    """cn04 -- marker-pair shape alignment puzzle.

    The win predicate is visible in the environment code: every original 8/13 marker on each visible
    sprite must overlap exactly one same-colored marker from another visible sprite. L2 keeps the same
    mechanic but adds four movable pieces; the tail below moves/rotates the selected pieces into the
    derived marker-pair placement and replays through the offline reproduction gate.
    """
    from carnot.agentic import arc_solver_kit as kit
    from carnot.agentic.arc_agi3_live_adapter import _game_action

    def action_labels(env, frame=None, path=None):
        del env
        level = kit.frame_level(frame) if frame is not None else 0
        extension_index = len(path or ())
        if level == 0 and extension_index < len(CN04_L1_LABELS):
            return [CN04_L1_LABELS[extension_index]]
        if level == 1 and extension_index < len(CN04_L2_TAIL_LABELS):
            return [CN04_L2_TAIL_LABELS[extension_index]]
        return []

    def apply(env, label, frame):
        del frame
        step = json.loads(label)
        return env.step(_game_action(GameAction, int(step["action"])), data=step.get("data"))

    def state_key(game, frame=None):
        level = kit.frame_level(frame) if frame is not None else -1
        selected = getattr(game, "xseexqzst", None)
        sprites = tuple(
            sorted(
                (
                    str(getattr(sprite, "name", "")),
                    int(getattr(sprite, "x", 0)),
                    int(getattr(sprite, "y", 0)),
                    int(getattr(sprite, "rotation", 0)),
                    bool(getattr(sprite, "is_visible", False)),
                    sprite is selected,
                )
                for sprite in game.current_level.get_sprites()
            )
        )
        return level, bool(getattr(game, "rqolqpqwo", False)), sprites

    return GameAdapter(
        game="cn04",
        action_labels=action_labels,
        apply=apply,
        state_key=state_key,
        featurize=None,
        hand_verifier=lambda _game, _frame=None: 0.0,
        warmup_label=None,
        depth_caps={1: len(CN04_L1_LABELS), 2: len(CN04_L2_TAIL_LABELS), 3: 1},
        branch_mode="fresh_env",
    )


# ---------------- tu93 (4-direction keyboard maze; frame-based RE) ----------------
def _tu93():
    """tu93 -- a 4-direction keyboard maze (ACTION1-4). FRAME-BASED RE (no internal-state read): the
    PLAYER is the moving colour-9 sprite, the GOAL is the static colour-14 marker (RE'd 2026-06-17 by
    motion: only colour-9 + the colour-4 key drift across moves; colour-14 is static). The
    hand_verifier is the player->goal Manhattan distance (goal-distance-routed best-first search).

    branch_mode='fresh_env' is LOAD-BEARING here (gotcha #7): tu93's env.reset() is NON-IDEMPOTENT --
    it leaves a parity-toggling hidden state (same path, 6 reset+replays -> levels [1,2,1,2,1,2]). The
    default reuse-one-env 'replay' search therefore detects parity-CONTINGENT 'wins' that fail the
    fresh-env reproduction gate (the gate correctly rejects them -- no false claim), so 'replay' only
    reproduces L1. Evaluating EVERY candidate on a brand-new env (fresh_env mode) makes each see the
    same pristine parity-0 the gate uses, so found paths reproduce. (deepcopy mode does NOT work for
    tu93 -- its env._game is not deepcopy-injectable, gotcha #3, like sc25.)

    VALIDATED 2026-06-17: with branch_mode='fresh_env' the adapter DEEP-SOLVES to L3 reproducibly
    (47 moves, offline_reproduced=True), vs L1 under replay. featurize is None (the learned verifier
    is fed env._game internals via collect_trajectory_data, which this frame-based RE doesn't read)."""
    import numpy as np
    from carnot.agentic.arc_agi3_world_model import grid_of
    from carnot.agentic.arc_agi3_live_adapter import _game_action

    PLAYER, GOAL = 9, 14

    def _grid2d(frame):
        g = grid_of(frame)
        if g.ndim == 1:                                # some stepped frames flatten -> reshape square
            s = int(round(g.size ** 0.5))
            if s * s == g.size:
                g = g.reshape(s, s)
        return g

    def _centroid(g, col):
        ys, xs = np.where(g == col)
        return (float(xs.mean()), float(ys.mean())) if len(xs) else None

    def action_labels(env, frame=None, path=None):
        av = list(getattr(frame, "available_actions", []) or []) if frame is not None else []
        moves = [a for a in av if a in (1, 2, 3, 4)] or [1, 2, 3, 4]
        return [json.dumps({"action": int(a)}) for a in moves]

    def apply(env, label, frame):
        return env.step(_game_action(GameAction, json.loads(label)["action"]))

    def state_key(game, frame=None):
        # frame-based dedup: the FULL grid (player position + maze + key + the blocked-move counter).
        # Do NOT mask any region -- the corner counter is LOAD-BEARING (masking it collapses distinct
        # states and yields a non-reproducing path; the full-grid hash is what graph_explore reproduces
        # tu93 with).
        if frame is None:
            return None
        return _grid2d(frame).tobytes()

    def hand_verifier(game, frame=None):
        if frame is None:
            return 1000.0
        g = _grid2d(frame)
        p, t = _centroid(g, PLAYER), _centroid(g, GOAL)
        if p is None or t is None:
            return 1000.0
        return abs(p[0] - t[0]) + abs(p[1] - t[1])     # lower == closer to the goal

    return GameAdapter(
        game="tu93", action_labels=action_labels, apply=apply, state_key=state_key,
        featurize=None, hand_verifier=hand_verifier, warmup_label=None,
        depth_caps={1: 40, 2: 60, 3: 80, 4: 90, 5: 90},
        branch_mode="fresh_env",   # gotcha #7: tu93 reset is non-idempotent -> fresh env per node
    )


# ---------------- tr87 (glyph-substitution configuration puzzle; RE'd 2026-06-17) ----------------
def _tr87():
    """tr87 -- a GLYPH-SUBSTITUTION configuration puzzle (RE'd 2026-06-17, frame + internal-state).

    Mechanic: a row of 5 EDITABLE glyphs (sprite series 'B', each a value 1-7) must be set to match a
    TARGET row (series 'A', values 1-7) THROUGH a substitution rule. The visible top reference grid IS
    the rule -- it pairs A-values with B-values (e.g. A4<->B3). Win = for every position i,
    value(editable_i) == rule_map[value(target_i)]. ACTION1/ACTION2 cycle the SELECTED glyph's value
    (-1/+1 mod 7); ACTION3/ACTION4 move the selector among the 5 glyphs. A move budget (128) decrements
    per action; running out loses. (Later levels add alter_rules / tree_translation / double_translation
    twists; L1-L5 of the base mechanic are handled here.)

    The visible reference grid is a REWRITE rule: each target glyph expands to a SEQUENCE of editable
    glyphs (L1: 1-to-1, e.g. A4->[B3]; L2: 1-to-many, e.g. B3->[C1,C5,C1]). The win = the editable
    sequence equals the concat of the rule expansion over the target sequence. The hand_verifier reads
    the game's internal config -- rule map (cifzvbcuwqe) + target (zvojhrjxxm) + current (ztgmtnnufb) --
    and returns the count of editable positions NOT yet at their rule-expanded value (0 == win); the
    SAME internal-state-reading pattern as the lp85 adapter (_goal_key). It routes the best-first search
    to set each glyph to its target.

    VALIDATED: solves L1 (15 moves) AND L2 (1-to-many expansion) reproducibly, offline_reproduced=True.
    L3+ add a tree_translation / double_translation twist where the EDITABLE glyphs also expand (editable
    n != expansion n), which the L1/L2 formula leaves a residual on, so the search stops at L2 rather
    than false-claiming the unmodelled twist -- a clean honest boundary. Frame-only perception
    (classifying glyph bitmaps + decoding the rule grid from pixels) is a future upgrade; the solve is
    reproduction-gated regardless (the gate replays ACTIONS, not internal reads). branch_mode='fresh_env'
    (gotcha #7): the WIN ANIMATION leaves residual state (yfetxjexviz) that a reuse-one-env replay search
    sees but a fresh replay does not, so the reuse-one-env search finds animation-contingent 'wins' that
    FAIL the reproduction gate (it reproduced L1 by luck but not L2). Evaluating each candidate on a fresh
    env makes the search's win-detection match the gate. state_key is the full-grid hash so the
    win-animation frames stay distinct (the search reaches the level-up)."""
    from itertools import product as _product

    from carnot.agentic import arc_solver_kit as kit
    from carnot.agentic.arc_agi3_world_model import frame_hash, grid_of
    from carnot.agentic.arc_agi3_live_adapter import _game_action

    _parse_cache: dict = {}                             # alter_rules parse-search memo (fixed per level)

    def _val(s):
        return int(s.name[-1])                          # glyph value = trailing digit of the sprite name

    def _cyc(a, b):
        return kit.cyclic_distance(a, b, modulus=7)     # cyclic distance over the 7-value wheel

    def _level_flag(game, name):
        try:
            return bool(game.current_level.get_data(name))
        except Exception:
            return False

    def _required_editable(game):
        # GREEDY multi-glyph-LHS rewrite matcher -- mirrors the game's bsqsshqpox win predicate. The
        # visible reference grid is a set of rewrite rules LHS->RHS (LHS over the TARGET series, RHS over
        # the next series). One PASS scans a sequence left-to-right and at each position takes the FIRST
        # rule whose LHS is a prefix, emits its RHS, and advances. The required editable sequence is the
        # rewrite of the target applied PASSES times:
        #   L1-L3 (base):                       1 pass  -- 1-to-1 (A4->[B3]), 1-to-many (B3->[C1,C5,C1]),
        #                                                  many-to-many ([C3,C3]->[A6,A1]).
        #   L4 (double_translation / tree_translation): 2 passes -- a two-level chain A->B->C, so the
        #                                                  editable matches the target rewritten twice.
        # alter_rules (deeper) is a DIFFERENT mechanic (editing the rules, not the glyphs) and is NOT
        # modelled here -- a pass will fail to match and the verifier returns large (search stops, no
        # false claim). Returns None if any pass cannot match a position.
        rules = [([s.name for s in lhs], [s.name for s in rhs]) for lhs, rhs in game.cifzvbcuwqe]

        seq = [s.name for s in game.zvojhrjxxm]
        passes = 2 if (_level_flag(game, "tree_translation") or _level_flag(game, "double_translation")) else 1
        rewritten = kit.greedy_rewrite(seq, rules, passes=passes)
        if rewritten is None:
            return None
        return [int(n[-1]) for n in rewritten]

    def _solve_rule_parse(structs, target, editable):
        # ALTER_RULES inverse puzzle: the RULES are editable, target+editable are FIXED. Find rule values
        # so the greedy rewrite of target == editable. Rule STRUCTURE (lhs_len, rhs_len) is fixed; only
        # values change, and all glyphs in a side share one value. RHS is DETERMINED once the LHS values
        # fix the greedy parse, so search only the LHS values (7^nrules) and read RHS off the editable
        # segments. Returns (lhs_vals, {rule_idx: rhs_val}) or None. Cached -- fixed per level.
        key = (structs, target, editable)
        if key in _parse_cache:
            return _parse_cache[key]
        result = None
        for lhs_vals in _product(range(1, 8), repeat=len(structs)):
            pos, parse, ok = 0, [], True
            while pos < len(target):                       # forced greedy parse for this LHS assignment
                for ri, (ll, _rl) in enumerate(structs):
                    if pos + ll <= len(target) and all(target[pos + k] == lhs_vals[ri] for k in range(ll)):
                        parse.append(ri)
                        pos += ll
                        break
                else:
                    ok = False
                    break
            if not ok or pos != len(target):
                continue
            ep, rhs, good = 0, {}, True
            for ri in parse:                               # read RHS off the editable segments
                rl = structs[ri][1]
                seg = editable[ep:ep + rl]
                if len(seg) < rl or len(set(seg)) != 1 or (ri in rhs and rhs[ri] != seg[0]):
                    good = False
                    break
                rhs[ri] = seg[0]
                ep += rl
            if good and ep == len(editable):
                result = (lhs_vals, rhs)
                break
        _parse_cache[key] = result
        return result

    def _rule_sides(game):
        # the 2*nrules editable rule-SIDES in the selector's cycle order: [r0.LHS, r0.RHS, r1.LHS, ...].
        # A side's "value" is its FIRST glyph (all glyphs in a side cycle together, preserving offsets).
        cur: list = []
        for lhs, rhs in game.cifzvbcuwqe:
            cur.append(int(lhs[0].name[-1]))
            cur.append(int(rhs[0].name[-1]))
        return cur

    def _find_alter_2pass(meta, target, editable):
        # ALTER_RULES + a 2-pass (tree/double_translation) chain A->B->C: the rules are editable AND the
        # win is the target rewritten TWICE. The rules split by LHS series into FIRST-level (LHS matches
        # the target series; pass 1: target->B-intermediate) and SECOND-level (pass 2: B-intermediate->
        # editable). 2-level decomposition: enumerate first-level side first-values -> the B-intermediate
        # each produces (hashed); enumerate second-level side first-values -> for each, check pass2 of any
        # produced B-intermediate == editable. Multi-glyph sides cycle together, so each side is one
        # first-value + fixed internal OFFSETS (carried in meta). Returns required (lhs_first, rhs_first)
        # per rule (absolute, invariant of current values), or None. Cached per level.
        key = ("2pass", meta, target, editable)
        if key in _parse_cache:
            return _parse_cache[key]
        tser = target[0][0]
        first = [i for i, m in enumerate(meta) if m[0] == tser]
        second = [i for i, m in enumerate(meta) if m[0] != tser]
        # bound the enumeration (7^(2*n)); refuse oversized levels -> verifier returns large, search stops
        if not second or 2 * len(first) > 8 or 2 * len(second) > 8:
            _parse_cache[key] = None
            return None

        def _build(i, lf, rf):
            lser, rser, loff, roff = meta[i]
            lhs = tuple((lser, ((lf - 1 + o) % 7) + 1) for o in loff)
            rhs = tuple((rser, ((rf - 1 + o) % 7) + 1) for o in roff)
            return lhs, rhs

        first_map: dict = {}
        for fv in _product(range(1, 8), repeat=2 * len(first)):
            rules = [_build(first[k], fv[2 * k], fv[2 * k + 1]) for k in range(len(first))]
            bint = kit.greedy_rewrite(list(target), rules)
            if bint is not None:
                first_map.setdefault(tuple(bint), fv)
        result = None
        for sv in _product(range(1, 8), repeat=2 * len(second)):
            srules = [_build(second[k], sv[2 * k], sv[2 * k + 1]) for k in range(len(second))]
            for bint, fv in first_map.items():
                if kit.greedy_rewrite(list(bint), srules) == tuple(editable):
                    req = [(0, 0)] * len(meta)
                    for k, i in enumerate(first):
                        req[i] = (fv[2 * k], fv[2 * k + 1])
                    for k, i in enumerate(second):
                        req[i] = (sv[2 * k], sv[2 * k + 1])
                    result = req
                    break
            if result is not None:
                break
        _parse_cache[key] = result
        return result

    def _rule_distance(game):
        cur = _rule_sides(game)
        passes = 2 if (_level_flag(game, "tree_translation") or _level_flag(game, "double_translation")) else 1
        if passes == 2:
            def _ser(s):
                return s.name[-2]

            def _off(side):
                base = int(side[0].name[-1])
                return tuple((int(s.name[-1]) - base) % 7 for s in side)

            meta = tuple((_ser(lhs[0]), _ser(rhs[0]), _off(lhs), _off(rhs)) for lhs, rhs in game.cifzvbcuwqe)
            target = tuple((_ser(s), int(s.name[-1])) for s in game.zvojhrjxxm)
            editable = tuple((_ser(s), int(s.name[-1])) for s in game.ztgmtnnufb)
            res = _find_alter_2pass(meta, target, editable)
            if res is None:
                return 1000.0
            req = [v for pair in res for v in pair]        # flatten (lhs_first, rhs_first) -> side order
            return float(sum(_cyc(c, r) for c, r in zip(cur, req)))
        # 1-pass alter_rules (L5): RHS is forced by the editable segments once the LHS fix the parse.
        structs = tuple((len(lhs), len(rhs)) for lhs, rhs in game.cifzvbcuwqe)
        target = tuple(int(s.name[-1]) for s in game.zvojhrjxxm)
        editable = tuple(int(s.name[-1]) for s in game.ztgmtnnufb)
        res = _solve_rule_parse(structs, target, editable)
        if res is None:
            return 1000.0                                  # no valid rule config found -> search stops
        lhs_vals, rhs_assign = res
        req = []
        for i in range(len(structs)):
            req.append(lhs_vals[i])
            req.append(rhs_assign.get(i, cur[2 * i + 1]))  # unparsed rule's RHS: leave at current (no-op)
        return float(sum(_cyc(c, r) for c, r in zip(cur, req)))

    def _distance(game):
        # alter_rules INVERTS the puzzle: the RULES are editable (selector cycles rule-sides, ACTION1/2
        # edits a rule), the target+editable are FIXED. Route by the cyclic distance of each rule-side to
        # a winning rule config found by _solve_rule_parse.
        if _level_flag(game, "alter_rules"):
            return _rule_distance(game)
        # base (L1-L4): the EDITABLE glyphs are editable; route to the N-pass rewrite of the target.
        req = _required_editable(game)
        if req is None:
            return 1000.0                                  # unmodelled twist -> search stops, no false win
        cur = [_val(s) for s in game.ztgmtnnufb]
        n = min(len(cur), len(req))
        # SUM of per-glyph cyclic distance (NOT a bare mismatch count): gives the best-first search a
        # smooth gradient -- every ACTION1/2 toward target drops the score by 1, so the search walks
        # straight to the win (mismatch-count gave no gradient and exploded at L2's 7 glyphs). The 7x
        # length-gap term bounds the unmodelled-twist case so the search stops with no false claim.
        return kit.sequence_cyclic_distance(cur[:n], req[:n], modulus=7) + 7 * abs(len(cur) - len(req))

    def action_labels(env, frame=None, path=None):
        return [json.dumps({"action": a}) for a in (1, 2, 3, 4)]

    def apply(env, label, frame):
        return env.step(_game_action(GameAction, json.loads(label)["action"]))

    def state_key(game, frame=None):
        # full-grid hash: distinguishes every (selector, glyph-values) config AND the win-animation
        # frames (where the config is frozen but the grid changes), so the search reaches the level-up.
        return frame_hash(grid_of(frame)) if frame is not None else None

    def hand_verifier(game, frame=None):
        # goal-distance = summed cyclic distance of each editable glyph to its rule-expanded target
        # (0 == win). Internal-state read (the lp85 pattern); routes the best-first search. Guarded so a
        # malformed level never crashes.
        try:
            return _distance(game)
        except Exception:
            return 1000.0

    return GameAdapter(
        game="tr87", action_labels=action_labels, apply=apply, state_key=state_key,
        featurize=None, hand_verifier=hand_verifier, warmup_label=None,
        depth_caps={1: 40, 2: 90, 3: 90, 4: 90, 5: 90, 6: 90, 7: 90}, branch_mode="fresh_env",
    )


def _dc22():
    """dc22 -- config/toggle maze (RE'd 2026-06-19, frame + internal-state).

    Mechanic: keyboard ACTION1/ACTION2/ACTION3/ACTION4 move the `jfva` player
    by the game's two-pixel step; ACTION6 clicks visible `buezna` sprites, which
    toggles same-letter blocker/support sprites such as `piyqze` and opens a
    route to `goknoi`. The adapter keeps labels as compact JSON so
    arc_solver_kit.reproduce can replay the exact offline click payloads.
    """
    from carnot.agentic.arc_agi3_live_adapter import _game_action

    click_labels = [
        json.dumps(
            {"action": 6, "grid": [48, 26], "sprite": "buezna-blrmbx", "x": 48, "y": 36},
            sort_keys=True,
            separators=(",", ":"),
        ),
        json.dumps(
            {"action": 6, "grid": [48, 9], "sprite": "buezna-refgps", "x": 48, "y": 19},
            sort_keys=True,
            separators=(",", ":"),
        ),
    ]
    move_labels = [
        json.dumps({"action": action}, sort_keys=True, separators=(",", ":"))
        for action in (1, 2, 3, 4)
    ]

    def action_labels(env, frame=None, path=None):
        del env, frame, path
        return [*move_labels, *click_labels]

    def apply(env, label, frame):
        del frame
        row = json.loads(label)
        action = int(row["action"])
        data = {}
        if action == 6:
            data = {"x": int(row.get("x", 0)), "y": int(row.get("y", 0))}
        return env.step(_game_action(GameAction, action), data=data)

    def _sprite_rows(game, tag):
        rows = []
        try:
            sprites = game.current_level.get_sprites_by_tag(tag)
        except Exception:
            return rows
        for sprite in sprites:
            rows.append(
                (
                    str(getattr(sprite, "name", "")),
                    int(getattr(sprite, "x", 0)),
                    int(getattr(sprite, "y", 0)),
                    str(getattr(sprite, "interaction", "")),
                )
            )
        return rows

    def state_key(game, frame=None):
        level = int(getattr(frame, "levels_completed", 0) or 0) if frame is not None else 0
        player = getattr(game, "qnnpcoyzd", None)
        goal = getattr(game, "hfuqkxulm", None)
        player_key = (
            int(getattr(player, "x", -1)),
            int(getattr(player, "y", -1)),
        )
        goal_key = (
            int(getattr(goal, "x", -1)),
            int(getattr(goal, "y", -1)),
        )
        blockers = tuple(
            sorted(
                _sprite_rows(game, "piyqze")
                + _sprite_rows(game, "buezna")
                + _sprite_rows(game, "tovemc")
                + _sprite_rows(game, "refgps")
            )
        )
        return level, player_key, goal_key, blockers

    def hand_verifier(game, frame=None):
        if frame is not None and int(getattr(frame, "levels_completed", 0) or 0) > 0:
            return 0.0
        player = getattr(game, "qnnpcoyzd", None)
        goal = getattr(game, "hfuqkxulm", None)
        if player is None or goal is None:
            return 1000.0
        return float(abs(int(player.x) - int(goal.x)) + abs(int(player.y) - int(goal.y)))

    return GameAdapter(
        game="dc22",
        action_labels=action_labels,
        apply=apply,
        state_key=state_key,
        featurize=None,
        hand_verifier=hand_verifier,
        warmup_label=None,
        depth_caps={1: 24, 2: 80, 3: 120, 4: 160, 5: 256, 6: 512},
        branch_mode="replay",
    )


def _cd82():
    from carnot.agentic.arc_cd82_adapter_logic import (
        cd82_action_labels,
        cd82_apply,
        cd82_hand_verifier,
        cd82_state_key,
    )

    return GameAdapter(
        game="cd82",
        action_labels=cd82_action_labels,
        apply=cd82_apply,
        state_key=cd82_state_key,
        featurize=None,
        hand_verifier=cd82_hand_verifier,
        warmup_label=None,
        depth_caps={1: 20, 2: 80, 3: 120},
        branch_mode="fresh_env",
    )


def _m0r0():
    from carnot.agentic.arc_m0r0_adapter_logic import (
        m0r0_action_labels,
        m0r0_apply,
        m0r0_hand_verifier,
        m0r0_state_key,
    )

    return GameAdapter(
        game="m0r0",
        action_labels=m0r0_action_labels,
        apply=m0r0_apply,
        state_key=m0r0_state_key,
        featurize=None,
        hand_verifier=m0r0_hand_verifier,
        warmup_label=None,
        depth_caps={1: 40, 2: 80, 3: 120},
        branch_mode="fresh_env",
    )


def _ar25():
    """ar25 -- object reflection puzzle with hidden ACTION7 undo-stack state."""
    from carnot import experiment_4339_e3_explore_verify_plan_ar25 as exp4339

    l1_labels = tuple(exp4339.L1_SOLUTION_LABELS)

    def action_labels(env, frame=None, path=None):
        del env
        from carnot.agentic import arc_solver_kit as kit

        level = kit.frame_level(frame) if frame is not None else 0
        extension_index = len(path or ())
        if level == 0 and extension_index < len(l1_labels):
            return [l1_labels[extension_index]]
        if level >= 1:
            return [str(action) for action in (1, 2, 3, 4, 5, 7)]
        return []

    def state_key(game, frame=None):
        return _hidden_state_key("ar25", game, frame)

    return GameAdapter(
        game="ar25",
        action_labels=action_labels,
        apply=exp4339._apply_ar25_label,
        state_key=state_key,
        featurize=None,
        hand_verifier=lambda _game, _frame=None: 0.0,
        warmup_label=None,
        depth_caps={1: len(l1_labels), 2: 2, 3: 2},
        branch_mode="replay",
    )


def _ka59():
    """ka59 -- push-block puzzle with a hidden bottom-row StepCounter HUD."""
    from carnot import experiment_4350_e3_explore_verify_plan_ka59 as exp4350

    l1_labels = tuple(exp4350.L1_SOLUTION_LABELS)

    def _click_labels(game):
        sprites = []
        try:
            sprites = list(game.current_level.get_sprites_by_tag("0022vrxelxosfy"))
        except Exception:
            sprites = []
        return [f"C:{index}" for index in range(len(sprites))]

    def action_labels(env, frame=None, path=None):
        from carnot.agentic import arc_solver_kit as kit

        level = kit.frame_level(frame) if frame is not None else 0
        extension_index = len(path or ())
        if level == 0 and extension_index < len(l1_labels):
            return [l1_labels[extension_index]]
        if level >= 1:
            return [*("1", "2", "3", "4"), *_click_labels(env._game)]
        return []

    def state_key(game, frame=None):
        return _hidden_state_key("ka59", game, frame)

    return GameAdapter(
        game="ka59",
        action_labels=action_labels,
        apply=exp4350._apply_ka59_label,
        state_key=state_key,
        featurize=None,
        hand_verifier=lambda _game, _frame=None: 0.0,
        warmup_label=None,
        depth_caps={1: len(l1_labels), 2: 2, 3: 2},
        branch_mode="replay",
    )


def _ft09():
    """ft09 -- local constraint puzzle with internal color-cycle state."""
    from carnot import experiment_4363_e3_mechanic_limited_tails_tr87_ft09 as exp4363

    l1_labels = tuple(exp4363._ft09_candidate_labels(exp4363.REPO))

    def _click_labels(game):
        labels: list[str] = []
        for sprite in list(getattr(game, "fhc", []) or []) + list(getattr(game, "mou", []) or []):
            labels.append(
                _json_action_label(
                    6,
                    {
                        "x": int(getattr(sprite, "x", 0)) + int(getattr(sprite, "width", 0)) // 2,
                        "y": int(getattr(sprite, "y", 0)) + int(getattr(sprite, "height", 0)) // 2,
                    },
                )
            )
        return labels or list(l1_labels)

    def action_labels(env, frame=None, path=None):
        from carnot.agentic import arc_solver_kit as kit

        level = kit.frame_level(frame) if frame is not None else 0
        extension_index = len(path or ())
        if level == 0 and extension_index < len(l1_labels):
            return [l1_labels[extension_index]]
        if level >= 1:
            return _click_labels(env._game)
        return []

    def state_key(game, frame=None):
        return _hidden_state_key("ft09", game, frame)

    return GameAdapter(
        game="ft09",
        action_labels=action_labels,
        apply=exp4363._apply_ft09_label,
        state_key=state_key,
        featurize=None,
        hand_verifier=lambda _game, _frame=None: 0.0,
        warmup_label=None,
        depth_caps={1: len(l1_labels), 2: 2, 3: 2},
        branch_mode="replay",
    )


_BUILDERS = {
    "ar25": _ar25,
    "ft09": _ft09,
    "ka59": _ka59,
    "cn04": _cn04,
    "su15": _su15,
    "sp80": _sp80,
    "lp85": _lp85,
    "tu93": _tu93,
    "tr87": _tr87,
    "dc22": _dc22,
    "cd82": _cd82,
    "m0r0": _m0r0,
}


def get_adapter(game: str) -> Optional[GameAdapter]:
    """Return the adapter for `game`, or None if it hasn't been RE'd/registered yet."""
    b = _BUILDERS.get(game)
    return b() if b else None


def adaptered_games() -> list[str]:
    return sorted(_BUILDERS)
