"""Per-game ADAPTERS for the standing ARC learning loop — the output of each
game's (irreducible) per-game reverse-engineering, captured as a reusable plug-in
for arc_solver_kit.OfflineSolver. A new game, once a level's win/action/state delta
is RE'd, registers an adapter here and that level is then solvable + reproducible +
learnable by the standing loop (scripts/arc_loop_solve.py) with no further bespoke
code.

An adapter provides the four game-specific callables the kit needs:
  action_labels(env) -> [str]   : env-discovered action vocabulary
  apply(env, label, frame)      : execute one action, return the new frame
  state_key(game)               : the dedup key (every load-bearing piece of state)
  featurize(game) -> [float]    : features for the LEARNED verifier (optional)
plus optional warmup_label and a hand verifier (goal-distance) for cold start.

SCOPE: AN ADAPTER COVERS A SHALLOW LEVEL PREFIX, NOT THE WHOLE GAME (corrected
2026-07-31). This paragraph exists because the sentence above used to read "once
its win/action/state delta is RE'd" — singular, per GAME — which invites the
reasonable conclusion that a registered adapter solves every level. It does not,
and 22 of the 24 registered adapters do not.

    The delta is PER LEVEL, not per game. ARC-AGI-3 levels are not difficulty
    scaling; each level generally introduces NEW MECHANICS, so a level is closer
    to a new game than to a harder instance of the same one. lf52 L2 adds
    rail-carried peg-jump removal; sk48 L3 adds a second, dark-headed vertical
    manipulator; wa30 L2 adds a helper-robot NPC, L3 a fence-socket relay, L4
    three side-affiliated robots; tn36 re-lays-out its program slots per level
    (L1: 5 slots, L2: 4, different x/y). An adapter's `action_labels` and
    `state_key` encode ONE level's mechanics, so the next level is usually not
    expressible in them at all. That is why the registry's failure vocabulary is
    `no_grounded_lN_delta` / `<mechanic>_L2_delta_not_adaptered_this_run` rather
    than "adapter broken" — see the `dead_ends` field on nearly every game in
    ops/arc_solve_registry.yaml.

    THIS SHALLOWNESS IS DELIBERATE, NOT DEBT. If these adapters covered all 183
    registered levels, that would mean 183 hand-RE'd level-deltas — precisely the
    `outer_loop_re` anti-pattern CLAUDE.md names, where the outer loop solves
    games by hand and the LIVE agent learns nothing. The live agent is the
    deliverable and it faces games that can never have an adapter. So an adapter
    is added when generic search has demonstrably failed on a level worth banking
    (e.g. `Exp4884 retired the prior g50t adapter-free L2 bounded-search dead end
    by registering _g50t`), and deeper levels are left to whatever generic method
    can earn them (sk48 L3-L4 were won by probe campaigns, not by extending the
    adapter).

    Depth is therefore a MEASURED property, not a promised one. The registry's
    per-game `solver` prose ("L1-L2: GameAdapter _lf52 ... L3: <other mechanism>")
    is the intent; `scripts/arc_adapter_depth_probe.py` measures what each adapter
    actually reaches, and ops/arc_adapter_depth_baseline.json records it so a
    silent regression is visible.

    NOT ON THE SCORED PATH. `arc_competition_agent.E3AgentPolicy` makes ZERO calls
    to get_adapter / solve_adaptered / adaptered_games — verified by import-closure
    analysis over its 43 carnot modules. A hidden game can never have an adapter,
    so adapter depth cannot affect a submission; it bounds offline development
    work only (which levels can be searched, and which games can produce an
    induction window via build_progress_window).
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
    # How a solution LABEL maps to the (action_id, data) pair that a Transition records.
    # Most games label actions with a plain integer, so the default `int(label)` in the
    # consumers is right and this stays None. It exists for games whose labels carry a
    # payload -- ka59 emits "C:<sprite_index>" for a click, which is action 6 plus {x,y}
    # data resolved against the live env. Without this hook `build_window` did
    # `int("C:1")` and raised, so ka59 had NO induction window at all and was silently
    # absent from every A/B corpus built that way (found 2026-07-31).
    # Does this adapter REPLAY a stored solution, or SEARCH a real action space?
    #
    # Measured 2026-07-31: 18 of 25 replay. Their `action_labels` hand back the next label of
    # a banked plan (or, for s5i5, one fixed click), so `solve_adaptered`'s verifier-routed
    # search, hazard pruning, learned-verifier warm start and state dedup are all unexercised
    # -- the search is a straight line through a solution that was already known. Removing the
    # plan does not reveal a search space underneath: 17 of the 18 return NO labels at all at
    # L0 once the plan is exhausted. The plan IS the L0 behaviour.
    #
    # This flag exists because the distinction was previously invisible without reading each
    # adapter's source, and because it makes downstream numbers honest: a replay adapter
    # "reaching L1" in ops/arc_adapter_depth_baseline.json measures the STORED PLAN, not any
    # capability of the adapter. Branching factor alone cannot separate the classes -- tn36
    # also returns a single label at L0, but COMPUTES it from live state, so it survives a
    # layout change where a stored plan would not.
    replay: bool = False
    # Where a replay adapter's plan comes from. Required when replay=True so the claim is
    # traceable to an artifact or constant rather than asserted.
    replay_source: Optional[str] = None
    label_to_action_data: Optional[Callable[[Any, str], tuple]] = None
    warmup_label: Optional[str] = None
    depth_caps: dict = field(default_factory=lambda: {})
    level_tails: dict[int, Sequence[str]] = field(default_factory=dict)
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


def _default_json_apply(env: Any, label: str, _frame: Any = None) -> Any:
    """Apply a `_json_action_label` string: {"action": N, "data": {...}}.

    Several adapters defined this inline (sp80's is the original). Hoisted so tn36, whose
    labels are generated rather than listed, can share one implementation instead of a
    fourth copy that could drift from the others.
    """
    from arcengine import GameAction

    from carnot.agentic.arc_agi3_live_adapter import _game_action

    step = json.loads(label)
    return env.step(_game_action(GameAction, int(step["action"])), data=step.get("data"))


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


S5I5_L2_TAIL_LABELS: tuple[str, ...] = tuple(
    _json_action_label(6, {"x": x, "y": y})
    for x, y in (
        (15, 57),
        (15, 57),
        (15, 57),
        (15, 57),
        (15, 57),
        (15, 57),
        (15, 57),
        (15, 57),
        (30, 57),
        (30, 57),
        (30, 57),
        (30, 57),
        (30, 57),
        (30, 57),
        (30, 57),
        (30, 57),
        (15, 57),
        (45, 57),
        (45, 57),
        (45, 57),
        (60, 57),
        (60, 57),
        (60, 57),
        (60, 57),
        (60, 57),
        (60, 57),
    )
)


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

AR25_L1_LABELS: tuple[str, ...] = tuple(["3"] * 5 + ["2"] * 10)

AR25_L2_TAIL_LABELS: tuple[str, ...] = tuple(["3", "3", "5"] + ["2"] * 8)

AR25_L2_SOLUTION_LABELS: tuple[str, ...] = AR25_L1_LABELS + AR25_L2_TAIL_LABELS

AR25_L3_TAIL_LABELS: tuple[str, ...] = tuple(
    ["1"] * 7 + ["5"] + ["4"] * 7 + ["2"] * 7 + ["5"] + ["3"] * 12 + ["2"] * 5
)

AR25_L3_SOLUTION_LABELS: tuple[str, ...] = AR25_L2_SOLUTION_LABELS + AR25_L3_TAIL_LABELS

FT09_L1_LABELS: tuple[str, ...] = (
    _json_action_label(6, {"x": 36, "y": 36}),
    _json_action_label(6, {"x": 36, "y": 44}),
    _json_action_label(6, {"x": 52, "y": 44}),
    _json_action_label(6, {"x": 36, "y": 52}),
)

FT09_L2_TAIL_LABELS: tuple[str, ...] = (
    _json_action_label(6, {"x": 22, "y": 16}),
    _json_action_label(6, {"x": 22, "y": 24}),
    _json_action_label(6, {"x": 38, "y": 24}),
    _json_action_label(6, {"x": 22, "y": 32}),
    _json_action_label(6, {"x": 38, "y": 32}),
    _json_action_label(6, {"x": 30, "y": 48}),
    _json_action_label(6, {"x": 22, "y": 48}),
)

FT09_L2_SOLUTION_LABELS: tuple[str, ...] = FT09_L1_LABELS + FT09_L2_TAIL_LABELS

FT09_L3_TAIL_LABELS: tuple[str, ...] = (
    _json_action_label(6, {"x": 28, "y": 20}),
    _json_action_label(6, {"x": 44, "y": 28}),
    _json_action_label(6, {"x": 28, "y": 36}),
    _json_action_label(6, {"x": 44, "y": 36}),
    _json_action_label(6, {"x": 20, "y": 44}),
    _json_action_label(6, {"x": 20, "y": 52}),
    _json_action_label(6, {"x": 28, "y": 52}),
    _json_action_label(6, {"x": 36, "y": 52}),
    _json_action_label(6, {"x": 20, "y": 4}),
    _json_action_label(6, {"x": 28, "y": 4}),
    _json_action_label(6, {"x": 36, "y": 4}),
    _json_action_label(6, {"x": 20, "y": 12}),
    _json_action_label(6, {"x": 12, "y": 20}),
    _json_action_label(6, {"x": 12, "y": 28}),
)

FT09_L3_SOLUTION_LABELS: tuple[str, ...] = FT09_L2_SOLUTION_LABELS + FT09_L3_TAIL_LABELS

VC33_L1_LABELS: tuple[str, ...] = tuple(_json_action_label(6, {"x": 61, "y": 33}) for _ in range(3))

VC33_L2_TAIL_LABELS: tuple[str, ...] = (
    _json_action_label(6, {"x": 1, "y": 25}),
    _json_action_label(6, {"x": 1, "y": 25}),
    _json_action_label(6, {"x": 1, "y": 45}),
    _json_action_label(6, {"x": 1, "y": 45}),
    _json_action_label(6, {"x": 1, "y": 45}),
    _json_action_label(6, {"x": 1, "y": 45}),
    _json_action_label(6, {"x": 1, "y": 45}),
)

VC33_L2_SOLUTION_LABELS: tuple[str, ...] = VC33_L1_LABELS + VC33_L2_TAIL_LABELS

SK48_L1_LABELS: tuple[str, ...] = tuple(
    _json_action_label(action) for action in (1, 1, 1, 4, 4, 4, 4, 3, 2, 2, 4, 3, 1, 4)
)

SK48_L2_TAIL_LABELS: tuple[str, ...] = tuple(
    _json_action_label(action)
    for action in (
        1,
        1,
        4,
        4,
        4,
        4,
        4,
        1,
        4,
        3,
        3,
        1,
        4,
        4,
        3,
        3,
        3,
        1,
        4,
        4,
        4,
        3,
        3,
        3,
        3,
        1,
        4,
        4,
        4,
        4,
    )
)

SK48_L2_SOLUTION_LABELS: tuple[str, ...] = SK48_L1_LABELS + SK48_L2_TAIL_LABELS

LS20_L1_LABELS: tuple[str, ...] = tuple(
    _json_action_label(action) for action in (3, 3, 3, 1, 1, 1, 1, 4, 4, 4, 1, 1, 1)
)

LS20_L2_TAIL_LABELS: tuple[str, ...] = tuple(
    _json_action_label(action)
    for action in (
        1,
        4,
        1,
        1,
        1,
        1,
        1,
        4,
        4,
        2,
        4,
        2,
        2,
        2,
        2,
        2,
        2,
        3,
        4,
        2,
        3,
        3,
        4,
        1,
        4,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        3,
        3,
        3,
        3,
        3,
        3,
        2,
        3,
        2,
        2,
        2,
        2,
        2,
    )
)

LS20_L2_SOLUTION_LABELS: tuple[str, ...] = LS20_L1_LABELS + LS20_L2_TAIL_LABELS

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

CN04_L3_TAIL_LABELS: tuple[str, ...] = (
    _json_action_label(4),
    _json_action_label(4),
    _json_action_label(4),
    _json_action_label(6, {"x": 44, "y": 42}),
    _json_action_label(5),
    _json_action_label(5),
    *(_json_action_label(1) for _ in range(6)),
    _json_action_label(6, {"x": 36, "y": 15}),
    _json_action_label(3),
    *(_json_action_label(2) for _ in range(6)),
)

CN04_L3_SOLUTION_LABELS: tuple[str, ...] = CN04_L2_SOLUTION_LABELS + CN04_L3_TAIL_LABELS

SB26_L1_LABELS: tuple[str, ...] = (
    _json_action_label(6, {"x": 36, "y": 59}),
    _json_action_label(6, {"x": 23, "y": 30}),
    _json_action_label(6, {"x": 20, "y": 59}),
    _json_action_label(6, {"x": 29, "y": 30}),
    _json_action_label(6, {"x": 44, "y": 59}),
    _json_action_label(6, {"x": 35, "y": 30}),
    _json_action_label(6, {"x": 28, "y": 59}),
    _json_action_label(6, {"x": 41, "y": 30}),
    _json_action_label(5),
)

SB26_L2_TAIL_LABELS: tuple[str, ...] = (
    _json_action_label(6, {"x": 32, "y": 59}),
    _json_action_label(6, {"x": 23, "y": 23}),
    _json_action_label(6, {"x": 18, "y": 59}),
    _json_action_label(6, {"x": 29, "y": 23}),
    _json_action_label(6, {"x": 11, "y": 59}),
    _json_action_label(6, {"x": 23, "y": 37}),
    _json_action_label(6, {"x": 46, "y": 59}),
    _json_action_label(6, {"x": 29, "y": 37}),
    _json_action_label(6, {"x": 25, "y": 59}),
    _json_action_label(6, {"x": 35, "y": 37}),
    _json_action_label(6, {"x": 53, "y": 59}),
    _json_action_label(6, {"x": 41, "y": 37}),
    _json_action_label(6, {"x": 39, "y": 59}),
    _json_action_label(6, {"x": 41, "y": 23}),
    _json_action_label(5),
)

SB26_L2_SOLUTION_LABELS: tuple[str, ...] = SB26_L1_LABELS + SB26_L2_TAIL_LABELS

LF52_L1_LABELS: tuple[str, ...] = tuple(
    _json_action_label(6, {"x": x, "y": y})
    for x, y in (
        (18, 19),
        (30, 19),
        (30, 19),
        (42, 19),
        (42, 19),
        (42, 31),
        (42, 31),
        (42, 43),
    )
)

LF52_L2_TAIL_LABELS: tuple[str, ...] = (
    _json_action_label(6, {"x": 14, "y": 16}),
    _json_action_label(6, {"x": 26, "y": 16}),
    _json_action_label(6, {"x": 26, "y": 16}),
    _json_action_label(6, {"x": 38, "y": 16}),
    _json_action_label(4),
    _json_action_label(4),
    _json_action_label(4),
    _json_action_label(4),
    _json_action_label(1),
    _json_action_label(1),
    _json_action_label(1),
    _json_action_label(3),
    _json_action_label(6, {"x": 38, "y": 16}),
    _json_action_label(6, {"x": 50, "y": 16}),
    _json_action_label(4),
    _json_action_label(2),
    _json_action_label(2),
    _json_action_label(2),
    _json_action_label(3),
    _json_action_label(3),
    _json_action_label(3),
    _json_action_label(3),
    _json_action_label(3),
    _json_action_label(3),
    _json_action_label(3),
    _json_action_label(2),
    _json_action_label(2),
    _json_action_label(2),
    _json_action_label(4),
    _json_action_label(4),
    _json_action_label(4),
    _json_action_label(4),
    _json_action_label(6, {"x": 38, "y": 52}),
    _json_action_label(6, {"x": 50, "y": 52}),
)

LF52_L2_SOLUTION_LABELS: tuple[str, ...] = LF52_L1_LABELS + LF52_L2_TAIL_LABELS

R11L_L1_LABELS: tuple[str, ...] = tuple(
    _json_action_label(6, {"x": x, "y": y})
    for x, y in (
        (38, 18),
        (27, 59),
        (34, 31),
    )
)

R11L_L2_TAIL_LABELS: tuple[str, ...] = tuple(
    _json_action_label(6, {"x": x, "y": y})
    for x, y in (
        (20, 3),
        (47, 27),
        (47, 33),
        (51, 29),
        (45, 23),
        (45, 15),
        (53, 7),
        (49, 3),
        (54, 48),
        (55, 23),
        (49, 3),
        (55, 15),
        (55, 19),
        (8, 21),
        (20, 21),
        (32, 33),
        (44, 45),
        (44, 57),
        (44, 63),
        (49, 9),
        (32, 57),
    )
)

R11L_L2_SOLUTION_LABELS: tuple[str, ...] = R11L_L1_LABELS + R11L_L2_TAIL_LABELS

RE86_L1_LABELS: tuple[str, ...] = (
    *(_json_action_label(1) for _ in range(7)),
    *(_json_action_label(4) for _ in range(4)),
    _json_action_label(5),
    *(_json_action_label(1) for _ in range(6)),
    *(_json_action_label(3) for _ in range(2)),
)

RE86_L2_TAIL_LABELS: tuple[str, ...] = (
    *(_json_action_label(2) for _ in range(10)),
    *(_json_action_label(3) for _ in range(3)),
    _json_action_label(5),
    *(_json_action_label(1) for _ in range(6)),
    *(_json_action_label(3) for _ in range(6)),
    _json_action_label(5),
    *(_json_action_label(2) for _ in range(2)),
    *(_json_action_label(3) for _ in range(7)),
)

RE86_L2_SOLUTION_LABELS: tuple[str, ...] = RE86_L1_LABELS + RE86_L2_TAIL_LABELS

BP35_L1_LABELS: tuple[str, ...] = (
    _json_action_label(4),
    _json_action_label(4),
    _json_action_label(4),
    _json_action_label(4),
    _json_action_label(6, {"x": 42, "y": 30}),
    _json_action_label(3),
    _json_action_label(3),
    _json_action_label(6, {"x": 24, "y": 36}),
    _json_action_label(3),
    _json_action_label(6, {"x": 18, "y": 36}),
    _json_action_label(3),
    _json_action_label(6, {"x": 18, "y": 30}),
    _json_action_label(4),
    _json_action_label(4),
    _json_action_label(6, {"x": 30, "y": 30}),
    _json_action_label(3),
    _json_action_label(3),
)

BP35_L2_TAIL_LABELS: tuple[str, ...] = (
    _json_action_label(4),
    _json_action_label(4),
    _json_action_label(4),
    _json_action_label(6, {"x": 36, "y": 30}),
    _json_action_label(6, {"x": 36, "y": 30}),
    _json_action_label(6, {"x": 30, "y": 36}),
    _json_action_label(3),
    _json_action_label(6, {"x": 24, "y": 36}),
    _json_action_label(3),
    _json_action_label(6, {"x": 18, "y": 36}),
    _json_action_label(3),
    _json_action_label(6, {"x": 12, "y": 36}),
    _json_action_label(3),
    _json_action_label(6, {"x": 12, "y": 30}),
    _json_action_label(4),
    _json_action_label(4),
    _json_action_label(4),
    _json_action_label(6, {"x": 30, "y": 30}),
    _json_action_label(6, {"x": 30, "y": 30}),
    _json_action_label(3),
    _json_action_label(3),
    _json_action_label(6, {"x": 18, "y": 30}),
    _json_action_label(6, {"x": 18, "y": 30}),
    _json_action_label(6, {"x": 18, "y": 30}),
    _json_action_label(6, {"x": 24, "y": 36}),
    _json_action_label(4),
    _json_action_label(6, {"x": 30, "y": 36}),
    _json_action_label(4),
    _json_action_label(6, {"x": 36, "y": 36}),
    _json_action_label(4),
    _json_action_label(6, {"x": 42, "y": 36}),
    _json_action_label(4),
    _json_action_label(6, {"x": 48, "y": 36}),
    _json_action_label(4),
    _json_action_label(6, {"x": 48, "y": 30}),
    _json_action_label(6, {"x": 48, "y": 30}),
    _json_action_label(3),
    _json_action_label(3),
    _json_action_label(3),
    _json_action_label(6, {"x": 30, "y": 30}),
)

BP35_L2_SOLUTION_LABELS: tuple[str, ...] = BP35_L1_LABELS + BP35_L2_TAIL_LABELS


G50T_L1_LABELS: tuple[str, ...] = tuple(
    _json_action_label(action) for action in (4, 4, 4, 4, 5, 2, 2, 2, 2, 2, 2, 2, 4, 4, 4, 4, 4)
)

G50T_L2_TAIL_LABELS: tuple[str, ...] = tuple(
    _json_action_label(action)
    for action in (
        3,
        3,
        5,
        2,
        2,
        2,
        2,
        3,
        3,
        3,
        3,
        1,
        1,
        3,
        3,
        5,
        1,
        1,
        1,
        3,
        3,
        3,
        3,
        3,
        3,
        3,
        2,
        2,
        4,
        4,
        4,
    )
)

G50T_L2_SOLUTION_LABELS: tuple[str, ...] = G50T_L1_LABELS + G50T_L2_TAIL_LABELS


def _g50t():
    """g50t -- target-offset toggle puzzle with clone-held plate routing."""
    from carnot.agentic import arc_solver_kit as kit
    from carnot.agentic.arc_agi3_live_adapter import _game_action

    def action_labels(env, frame=None, path=None):
        del env
        level = kit.frame_level(frame) if frame is not None else 0
        extension_index = len(path or ())
        if level == 0 and extension_index < len(G50T_L1_LABELS):
            return [G50T_L1_LABELS[extension_index]]
        if level == 1 and extension_index < len(G50T_L2_TAIL_LABELS):
            return [G50T_L2_TAIL_LABELS[extension_index]]
        return []

    def apply(env, label, frame):
        del frame
        step = json.loads(label)
        return env.step(_game_action(GameAction, int(step["action"])), data=step.get("data"))

    def _object_row(obj: Any) -> tuple[Any, ...]:
        return (
            type(obj).__name__,
            int(getattr(obj, "x", 0)),
            int(getattr(obj, "y", 0)),
            int(getattr(obj, "xjzbfpegay", getattr(obj, "x", 0))),
            int(getattr(obj, "yzwsikkoyp", getattr(obj, "y", 0))),
            bool(getattr(obj, "dijhfchobv", False)),
            int(getattr(obj, "ecvzipvalj", 0) or 0),
            int(getattr(obj, "lzwacefckd", 0) or 0),
            int(getattr(obj, "rotation", 0) or 0),
            bool(getattr(obj, "is_visible", True)),
            bool(getattr(obj, "pddqxjztas", False)),
            bool(getattr(obj, "amjlzzsesf", False)),
        )

    def state_key(game, frame=None):
        state = game.vgwycxsxjz
        return (
            kit.frame_level(frame) if frame is not None else -1,
            int(getattr(game, "ucorwtereb", 0) or 0),
            bool(getattr(game, "qgzorkgosv", False)),
            bool(getattr(game, "hctlyapjnq", False)),
            _object_row(state.dzxunlkwxt),
            _object_row(state.whftgckbcu),
            tuple(_object_row(obj) for obj in state.uwxkstolmf),
            tuple(
                _object_row(obj) + (len(getattr(obj, "vbqvjbxkfm", ())),)
                for obj in state.hamayflsib
            ),
            tuple(
                _object_row(obj)
                + (int(getattr(obj, "wzgvpxcawd", 0) or 0), len(moves), tuple(moves))
                for obj, moves in state.kgvnkyaimw.items()
            ),
            tuple(
                _object_row(obj) + (len(moves), tuple(moves))
                for obj, moves in state.rloltuowth.items()
            ),
            tuple(state.areahjypvy),
            int(state.rlazdofsxb),
            bool(state.dofntsemri),
            bool(state.pohkooyzds),
            len(state.hjvvibklzv),
            int(getattr(getattr(game, "twyixucrqi", None), "x", 0) or 0),
        )

    def hand_verifier(game, _frame=None):
        state = game.vgwycxsxjz
        player = state.dzxunlkwxt
        target = state.whftgckbcu
        if getattr(game, "hctlyapjnq", False) or state.zvuxrhnlcb or game.ptayayjhqx():
            return 10000.0
        return float(abs(player.x - (target.x + 1)) + abs(player.y - (target.y + 1)))

    return GameAdapter(
        game="g50t",
        replay=True,
        replay_source="banked L1 plan; registered by Exp4884 after adapter-free L2 search dead-ended",
        action_labels=action_labels,
        apply=apply,
        state_key=state_key,
        featurize=None,
        hand_verifier=hand_verifier,
        warmup_label=None,
        depth_caps={1: len(G50T_L1_LABELS), 2: len(G50T_L2_TAIL_LABELS), 3: 2},
        level_tails={1: G50T_L1_LABELS, 2: G50T_L2_TAIL_LABELS},
        branch_mode="replay",
    )


def _re86_center_color(sprite: Any) -> int:
    pixels = getattr(sprite, "pixels", None)
    try:
        return int(pixels[sprite.height // 2, sprite.width // 2])
    except Exception:
        return -1


def _re86():
    """re86 -- sprite-overlay pattern match with verifier-derived L2 resize tail."""
    from carnot.agentic import arc_solver_kit as kit
    from carnot.agentic.arc_agi3_live_adapter import _game_action

    def action_labels(env, frame=None, path=None):
        del env
        level = kit.frame_level(frame) if frame is not None else 0
        extension_index = len(path or ())
        if level == 0 and extension_index < len(RE86_L1_LABELS):
            return [RE86_L1_LABELS[extension_index]]
        if level == 1 and extension_index < len(RE86_L2_TAIL_LABELS):
            return [RE86_L2_TAIL_LABELS[extension_index]]
        return []

    def apply(env, label, frame):
        del frame
        row = json.loads(label)
        data = row.get("data") if isinstance(row.get("data"), dict) else None
        return env.step(_game_action(GameAction, int(row["action"])), data=data)

    def _sprite_row(sprite):
        return (
            str(getattr(sprite, "name", "")),
            int(getattr(sprite, "x", 0)),
            int(getattr(sprite, "y", 0)),
            int(getattr(sprite, "width", 0)),
            int(getattr(sprite, "height", 0)),
            int(getattr(sprite, "rotation", 0) or 0),
            _re86_center_color(sprite),
        )

    def _sources(game):
        return list(game.current_level.get_sprites_by_tag("0031cppcuvqlbi"))

    def _targets(game):
        return list(game.current_level.get_sprites_by_tag("0054xnsuqceejm"))

    def _active_index(game):
        for index, sprite in enumerate(_sources(game)):
            if _re86_center_color(sprite) == 0:
                return index
        return 0

    def state_key(game, frame=None):
        level = kit.frame_level(frame) if frame is not None else -1
        return (
            int(level),
            int(_active_index(game)),
            tuple(sorted(_sprite_row(sprite) for sprite in _sources(game))),
            tuple(sorted(_sprite_row(sprite) for sprite in _targets(game))),
        )

    def featurize(game):
        sources = _sources(game)
        targets = _targets(game)
        target = targets[0] if targets else None
        active = _active_index(game)
        source = sources[active] if 0 <= active < len(sources) else None
        sx = float(getattr(source, "x", 0) if source is not None else 0)
        sy = float(getattr(source, "y", 0) if source is not None else 0)
        tx = float(getattr(target, "x", 0) if target is not None else 0)
        ty = float(getattr(target, "y", 0) if target is not None else 0)
        sw = float(getattr(source, "width", 0) if source is not None else 0)
        sh = float(getattr(source, "height", 0) if source is not None else 0)
        return [
            float(getattr(game, "level_index", 0) or 0),
            float(active),
            float(len(sources)),
            float(len(targets)),
            sx,
            sy,
            sw,
            sh,
            tx,
            ty,
            abs(sx - tx) + abs(sy - ty),
        ]

    def hand_verifier(game, _frame=None):
        features = featurize(game)
        return float(features[-1])

    return GameAdapter(
        game="re86",
        replay=True,
        replay_source="banked L1 plan (module constant)",
        action_labels=action_labels,
        apply=apply,
        state_key=state_key,
        featurize=featurize,
        hand_verifier=hand_verifier,
        warmup_label=None,
        depth_caps={1: len(RE86_L1_LABELS), 2: len(RE86_L2_TAIL_LABELS), 3: 2},
        level_tails={1: RE86_L1_LABELS, 2: RE86_L2_TAIL_LABELS},
        branch_mode="fresh_env",
    )


def _bp35():
    """bp35 -- upward-gravity platformer with support clearing and same-row blocker deltas."""
    from carnot.agentic import arc_solver_kit as kit
    from carnot.agentic.arc_agi3_live_adapter import _game_action

    def action_labels(env, frame=None, path=None):
        del env
        level = kit.frame_level(frame) if frame is not None else 0
        extension_index = len(path or ())
        if level == 0 and extension_index < len(BP35_L1_LABELS):
            return [BP35_L1_LABELS[extension_index]]
        if level == 1 and extension_index < len(BP35_L2_TAIL_LABELS):
            return [BP35_L2_TAIL_LABELS[extension_index]]
        return []

    def apply(env, label, frame):
        del frame
        row = json.loads(str(label))
        data = row.get("data") if isinstance(row.get("data"), dict) else None
        return env.step(_game_action(GameAction, int(row["action"])), data=data)

    def _world(game):
        return getattr(game, "oztjzzyqoek", None)

    def _player_position(game) -> tuple[int, int]:
        world = _world(game)
        player = getattr(world, "twdpowducb", None)
        pos = getattr(player, "qumspquyus", (0, 0))
        return int(pos[0]), int(pos[1])

    def _camera_y(game) -> int:
        world = _world(game)
        camera = getattr(world, "camera", None)
        offset = getattr(camera, "rczgvgfsfb", (0, 0))
        return int(offset[1])

    def _goal_position(game) -> tuple[int, int]:
        world = _world(game)
        board = getattr(world, "hdnrlfmyrj", None)
        if board is None:
            return 0, 0
        goals = board.wwkbcxznzg("fjlzdjxhant")
        if not goals:
            return 0, 0
        pos = goals[0].qumspquyus
        return int(pos[0]), int(pos[1])

    def _object_rows(game) -> tuple[tuple[str, int, int], ...]:
        world = _world(game)
        board = getattr(world, "hdnrlfmyrj", None)
        rows: list[tuple[str, int, int]] = []
        names = {
            "qclfkhjnaac",
            "etlsaqqtjvn",
            "yuuqpmlxorv",
            "oonshderxef",
            "lrpkmzabbfa",
            "aknlbboysnc",
            "jcyhkseuorf",
            "ubhhgljbnpu",
            "hzusueifitk",
            "fjlzdjxhant",
        }
        for obj in getattr(board, "ugywcmguyv", []) or []:
            name = str(getattr(obj, "name", ""))
            if name not in names:
                continue
            pos = getattr(obj, "qumspquyus", (0, 0))
            rows.append((name, int(pos[0]), int(pos[1])))
        return tuple(sorted(rows))

    def state_key(game, frame=None):
        world = _world(game)
        px, py = _player_position(game)
        return (
            kit.frame_level(frame) if frame is not None else int(getattr(game, "_score", 0) or 0),
            str(getattr(game, "_state", "")),
            px,
            py,
            bool(getattr(world, "ybmkdxbdko", True)),
            bool(getattr(world, "vivnprldht", True)),
            int(getattr(world, "wjidupyeoa", 0) or 0),
            _camera_y(game),
            _object_rows(game),
        )

    def featurize(game):
        px, py = _player_position(game)
        gx, gy = _goal_position(game)
        blockers = sum(1 for name, _x, _y in _object_rows(game) if name == "qclfkhjnaac")
        world = _world(game)
        return [
            float(getattr(game, "_score", 0) or 0),
            float(px),
            float(py),
            float(gx),
            float(gy),
            float(abs(px - gx)),
            float(max(0, py - gy)),
            float(_camera_y(game)),
            float(getattr(world, "wjidupyeoa", 0) or 0),
            float(blockers),
            float("GAME_OVER" in str(getattr(game, "_state", ""))),
        ]

    def hand_verifier(game, _frame=None):
        px, py = _player_position(game)
        gx, gy = _goal_position(game)
        penalty = 1000.0 if "GAME_OVER" in str(getattr(game, "_state", "")) else 0.0
        return float(abs(px - gx) + max(0, py - gy) + penalty)

    return GameAdapter(
        game="bp35",
        replay=True,
        replay_source="banked L1 plan (module constant)",
        action_labels=action_labels,
        apply=apply,
        state_key=state_key,
        featurize=featurize,
        hand_verifier=hand_verifier,
        warmup_label=None,
        depth_caps={1: len(BP35_L1_LABELS), 2: len(BP35_L2_TAIL_LABELS), 3: 2},
        level_tails={1: BP35_L1_LABELS, 2: BP35_L2_TAIL_LABELS},
        branch_mode="replay",
    )


# ---------------- lp85 (reference adapter; click-only rotation puzzle) ----------------
def _lp85():
    from carnot.experiment_4179_arc_incremental_progress import (
        discover_click_buttons,
        _goal_key,
        _target_goal_key,
    )

    def action_labels(env):
        return [
            json.dumps({"x": int(b["x"]), "y": int(b["y"])}) for b in discover_click_buttons(env)
        ]

    def apply(env, label, frame):
        a = json.loads(label)
        return env.step(GameAction.ACTION6, data={"x": a["x"], "y": a["y"]})

    def _dists(game):
        actual = _goal_key(game)
        target = _target_goal_key(game)
        by_type = defaultdict(list)
        for t, x, y in actual:
            by_type[t].append((x, y))
        return [
            min((abs(tx - x) + abs(ty - y)) for x, y in by_type.get(t, []))
            if by_type.get(t)
            else 1000.0
            for t, tx, ty in target
        ]

    def featurize(game):
        ds = _dists(game)
        n = len(ds) or 1
        return [
            sum(ds),
            float(sum(1 for d in ds if d > 0)),
            sum(ds) / n,
            float(max(ds) if ds else 0),
            float(n),
        ]

    return GameAdapter(
        game="lp85",
        action_labels=action_labels,
        apply=apply,
        state_key=_goal_key,
        featurize=featurize,
        hand_verifier=lambda g: float(sum(_dists(g))),
        depth_caps={1: 20, 2: 70, 3: 90},
    )


def _s5i5():
    """s5i5 -- dynamic marker-control coverage across L1/L2."""
    from carnot.agentic import arc_solver_kit as kit
    from carnot.agentic.arc_agi3_live_adapter import _game_action
    from carnot.agentic.arc_exp4101_eleventh_game_explore_first import (
        build_s5i5_l1_plan,
        observe_s5i5_state_from_env,
    )

    def _plan_labels(env, frame=None) -> list[str]:
        try:
            observed = observe_s5i5_state_from_env(
                env,
                level_completed=kit.frame_level(frame),
            )
            plan = build_s5i5_l1_plan(observed)
        except (AttributeError, TypeError, ValueError):
            return []
        return [_json_action_label(6, {"x": action.x, "y": action.y}) for action in plan.actions]

    def action_labels(env, frame=None, path=None):
        level = kit.frame_level(frame) if frame is not None else 0
        extension_index = len(path or ())
        if level == 1 and extension_index < len(S5I5_L2_TAIL_LABELS):
            return [S5I5_L2_TAIL_LABELS[extension_index]]
        labels = _plan_labels(env, frame)
        if not labels:
            return []
        return [labels[0]]

    def apply(env, label, frame):
        del frame
        row = json.loads(label)
        data = row.get("data") if isinstance(row.get("data"), dict) else None
        return env.step(_game_action(GameAction, int(row["action"])), data=data)

    def _sprite_rows(game, tag: str) -> tuple[tuple[str, int, int, int, int], ...]:
        level = getattr(game, "current_level", None)
        getter = getattr(level, "get_sprites_by_tag", None)
        if not callable(getter):
            return ()
        try:
            sprites = list(getter(tag))
        except Exception:
            return ()
        return tuple(
            sorted(
                (
                    str(getattr(sprite, "name", "")),
                    int(getattr(sprite, "x", 0)),
                    int(getattr(sprite, "y", 0)),
                    int(getattr(sprite, "width", 0)),
                    int(getattr(sprite, "height", 0)),
                )
                for sprite in sprites
            )
        )

    def _control_rows(game) -> tuple[tuple[str, int, int, int, int], ...]:
        controls = getattr(game, "pigtralzpb", {})
        try:
            keys = list(controls.keys())
        except Exception:
            keys = []
        return tuple(
            sorted(
                (
                    str(getattr(control, "name", "")),
                    int(getattr(control, "x", 0)),
                    int(getattr(control, "y", 0)),
                    int(getattr(control, "width", 0)),
                    int(getattr(control, "height", 0)),
                )
                for control in keys
            )
        )

    def _linked_rows(game) -> tuple[tuple[str, str, int, int, int, int], ...]:
        rows: list[tuple[str, str, int, int, int, int]] = []
        for mapping_name in ("pigtralzpb", "uricqfoplr"):
            mapping = getattr(game, mapping_name, {})
            try:
                items = list(mapping.items())
            except Exception:
                items = []
            for key, values in items:
                rows.append(
                    (
                        mapping_name,
                        str(getattr(key, "name", "")),
                        int(getattr(key, "x", 0)),
                        int(getattr(key, "y", 0)),
                        int(getattr(key, "width", 0)),
                        int(getattr(key, "height", 0)),
                    )
                )
                try:
                    value_sprites = list(values)
                except Exception:
                    value_sprites = []
                for sprite in value_sprites:
                    rows.append(
                        (
                            mapping_name,
                            str(getattr(sprite, "name", "")),
                            int(getattr(sprite, "x", 0)),
                            int(getattr(sprite, "y", 0)),
                            int(getattr(sprite, "width", 0)),
                            int(getattr(sprite, "height", 0)),
                        )
                    )
        return tuple(sorted(rows))

    def state_key(game, frame=None):
        return (
            kit.frame_level(frame) if frame is not None else -1,
            _frame_grid_state_key(frame),
            _sprite_rows(game, "0064ocqkuqacti"),
            _sprite_rows(game, "0087vvmblxkzdi"),
            _control_rows(game),
            _linked_rows(game),
        )

    def hand_verifier(game, frame=None):
        fake_env = type("EnvProxy", (), {"_game": game})()
        try:
            observed = observe_s5i5_state_from_env(
                fake_env,
                level_completed=kit.frame_level(frame),
            )
        except (AttributeError, TypeError, ValueError):
            return 1000.0
        return float(sum(max(0, int(item.clicks_needed)) for item in observed.items))

    return GameAdapter(
        game="s5i5",
        replay=True,
        replay_source="single fixed ACTION6 click at (48,21) -- constant, not a plan index; registered by Exp4873",
        action_labels=action_labels,
        apply=apply,
        state_key=state_key,
        featurize=None,
        hand_verifier=hand_verifier,
        warmup_label=None,
        depth_caps={1: 24, 2: 32, 3: 2},
        level_tails={2: S5I5_L2_TAIL_LABELS},
        branch_mode="replay",
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
        return (
            int(sprite.x),
            int(sprite.y),
            int(getattr(sprite, "width", 0)),
            int(getattr(sprite, "height", 0)),
            shape,
        )

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
        replay=True,
        replay_source="banked L1 plan (module constant)",
        action_labels=action_labels,
        apply=apply,
        state_key=state_key,
        featurize=None,
        hand_verifier=lambda _game, _frame=None: 0.0,
        warmup_label=None,
        depth_caps={1: len(SU15_L1_LABELS), 2: len(SU15_L2_TAIL_LABELS), 3: 80},
        branch_mode="fresh_env",
    )


# ---------------- sb26 (ordered color-match slot sequence; L1 seed + L2 nested-frame delta) ----------------
def _sb26():
    """sb26 -- ordered color-to-slot matching with a nested-frame L2 branch.

    L1 is the existing flat color-match solve. L2 keeps the same click/validate
    mechanic but adds a pre-placed `vgszefyyyp` branch in the root frame: the
    root slots consume colors 12 and 15, the branch descends to the second frame
    for 8, 9, 14, and 11, then the root tail consumes 6. The labels are the
    offline-reproduced display centers for those item/slot placements.
    """
    from carnot.agentic import arc_solver_kit as kit
    from carnot.agentic.arc_agi3_live_adapter import _game_action

    def action_labels(env, frame=None, path=None):
        del env
        level = kit.frame_level(frame) if frame is not None else 0
        extension_index = len(path or ())
        if level == 0 and extension_index < len(SB26_L1_LABELS):
            return [SB26_L1_LABELS[extension_index]]
        if level == 1 and extension_index < len(SB26_L2_TAIL_LABELS):
            return [SB26_L2_TAIL_LABELS[extension_index]]
        return []

    def apply(env, label, frame):
        del frame
        row = json.loads(label)
        data = row.get("data") if isinstance(row.get("data"), dict) else None
        return env.step(_game_action(GameAction, int(row["action"])), data=data)

    def _clickable_rows(game):
        rows = []
        for sprite in list(getattr(game, "dkouqqads", []) or []) + list(
            getattr(game, "dewwplfix", []) or []
        ):
            rows.append(
                (
                    str(getattr(sprite, "name", "")),
                    int(getattr(sprite, "x", 0)),
                    int(getattr(sprite, "y", 0)),
                    _sprite_color(sprite),
                    bool(getattr(sprite, "is_visible", True)),
                )
            )
        return tuple(sorted(rows))

    def _target_fill_count(game):
        filled = 0
        targets = list(getattr(game, "wcfyiodrx", []) or [])
        for sprite in targets:
            pixels = getattr(sprite, "pixels", None)
            try:
                filled += int(pixels[1, 1] != -1)
            except Exception:
                pass
        return filled, len(targets)

    def state_key(game, frame=None):
        level = (
            kit.frame_level(frame)
            if frame is not None
            else int(getattr(game, "level_index", 0) or 0)
        )
        selected = getattr(game, "lqcskynzr", None)
        filled, total = _target_fill_count(game)
        return (
            int(level),
            int(getattr(game, "pmygakdvy", 0) or 0),
            len(getattr(game, "buvfjfmpp", []) or []),
            int(getattr(game, "modqnpqfi", 0) or 0),
            int(getattr(game, "sjcuorclg", 0) or 0),
            bool(selected),
            filled,
            total,
            _clickable_rows(game),
        )

    def featurize(game):
        filled, total = _target_fill_count(game)
        selected = getattr(game, "lqcskynzr", None)
        placed = sum(
            1 for _name, _x, y, _color, visible in _clickable_rows(game) if visible and int(y) < 53
        )
        return [
            float(getattr(game, "level_index", 0) or 0),
            float(getattr(game, "pmygakdvy", 0) or 0),
            float(filled),
            float(total),
            float(placed),
            float(bool(selected)),
            float(getattr(game, "sjcuorclg", 0) or 0),
        ]

    return GameAdapter(
        game="sb26",
        replay=True,
        replay_source="banked L1 plan (module constant)",
        action_labels=action_labels,
        apply=apply,
        state_key=state_key,
        featurize=featurize,
        hand_verifier=lambda game, _frame=None: float(
            max(0, _target_fill_count(game)[1] - _target_fill_count(game)[0])
        ),
        warmup_label=None,
        depth_caps={1: len(SB26_L1_LABELS), 2: len(SB26_L2_TAIL_LABELS), 3: 2},
        level_tails={1: SB26_L1_LABELS, 2: SB26_L2_TAIL_LABELS},
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
                (
                    sprite.name,
                    int(sprite.x),
                    int(sprite.y),
                    int(sprite.width),
                    int(sprite.height),
                    sprite is selected,
                )
                for sprite in game.fbrwmvzsym()
            ),
        )

    return GameAdapter(
        game="sp80",
        replay=True,
        replay_source="SP80_L1_LABELS (module constant)",
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
        if level == 2 and extension_index < len(CN04_L3_TAIL_LABELS):
            return [CN04_L3_TAIL_LABELS[extension_index]]
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
        replay=True,
        replay_source="banked L1 plan (module constant)",
        action_labels=action_labels,
        apply=apply,
        state_key=state_key,
        featurize=None,
        hand_verifier=lambda _game, _frame=None: 0.0,
        warmup_label=None,
        depth_caps={
            1: len(CN04_L1_LABELS),
            2: len(CN04_L2_TAIL_LABELS),
            3: len(CN04_L3_TAIL_LABELS),
        },
        level_tails={3: CN04_L3_TAIL_LABELS},
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
        if g.ndim == 1:  # some stepped frames flatten -> reshape square
            s = int(round(g.size**0.5))
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
        return abs(p[0] - t[0]) + abs(p[1] - t[1])  # lower == closer to the goal

    return GameAdapter(
        game="tu93",
        action_labels=action_labels,
        apply=apply,
        state_key=state_key,
        featurize=None,
        hand_verifier=hand_verifier,
        warmup_label=None,
        depth_caps={1: 40, 2: 60, 3: 80, 4: 90, 5: 90},
        branch_mode="fresh_env",  # gotcha #7: tu93 reset is non-idempotent -> fresh env per node
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

    _parse_cache: dict = {}  # alter_rules parse-search memo (fixed per level)

    def _val(s):
        return int(s.name[-1])  # glyph value = trailing digit of the sprite name

    def _cyc(a, b):
        return kit.cyclic_distance(a, b, modulus=7)  # cyclic distance over the 7-value wheel

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
        passes = (
            2
            if (_level_flag(game, "tree_translation") or _level_flag(game, "double_translation"))
            else 1
        )
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
            while pos < len(target):  # forced greedy parse for this LHS assignment
                for ri, (ll, _rl) in enumerate(structs):
                    if pos + ll <= len(target) and all(
                        target[pos + k] == lhs_vals[ri] for k in range(ll)
                    ):
                        parse.append(ri)
                        pos += ll
                        break
                else:
                    ok = False
                    break
            if not ok or pos != len(target):
                continue
            ep, rhs, good = 0, {}, True
            for ri in parse:  # read RHS off the editable segments
                rl = structs[ri][1]
                seg = editable[ep : ep + rl]
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
        passes = (
            2
            if (_level_flag(game, "tree_translation") or _level_flag(game, "double_translation"))
            else 1
        )
        if passes == 2:

            def _ser(s):
                return s.name[-2]

            def _off(side):
                base = int(side[0].name[-1])
                return tuple((int(s.name[-1]) - base) % 7 for s in side)

            meta = tuple(
                (_ser(lhs[0]), _ser(rhs[0]), _off(lhs), _off(rhs)) for lhs, rhs in game.cifzvbcuwqe
            )
            target = tuple((_ser(s), int(s.name[-1])) for s in game.zvojhrjxxm)
            editable = tuple((_ser(s), int(s.name[-1])) for s in game.ztgmtnnufb)
            res = _find_alter_2pass(meta, target, editable)
            if res is None:
                return 1000.0
            req = [v for pair in res for v in pair]  # flatten (lhs_first, rhs_first) -> side order
            return float(sum(_cyc(c, r) for c, r in zip(cur, req)))
        # 1-pass alter_rules (L5): RHS is forced by the editable segments once the LHS fix the parse.
        structs = tuple((len(lhs), len(rhs)) for lhs, rhs in game.cifzvbcuwqe)
        target = tuple(int(s.name[-1]) for s in game.zvojhrjxxm)
        editable = tuple(int(s.name[-1]) for s in game.ztgmtnnufb)
        res = _solve_rule_parse(structs, target, editable)
        if res is None:
            return 1000.0  # no valid rule config found -> search stops
        lhs_vals, rhs_assign = res
        req = []
        for i in range(len(structs)):
            req.append(lhs_vals[i])
            req.append(
                rhs_assign.get(i, cur[2 * i + 1])
            )  # unparsed rule's RHS: leave at current (no-op)
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
            return 1000.0  # unmodelled twist -> search stops, no false win
        cur = [_val(s) for s in game.ztgmtnnufb]
        n = min(len(cur), len(req))
        # SUM of per-glyph cyclic distance (NOT a bare mismatch count): gives the best-first search a
        # smooth gradient -- every ACTION1/2 toward target drops the score by 1, so the search walks
        # straight to the win (mismatch-count gave no gradient and exploded at L2's 7 glyphs). The 7x
        # length-gap term bounds the unmodelled-twist case so the search stops with no false claim.
        return kit.sequence_cyclic_distance(cur[:n], req[:n], modulus=7) + 7 * abs(
            len(cur) - len(req)
        )

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
        game="tr87",
        action_labels=action_labels,
        apply=apply,
        state_key=state_key,
        featurize=None,
        hand_verifier=hand_verifier,
        warmup_label=None,
        depth_caps={1: 40, 2: 90, 3: 90, 4: 90, 5: 90, 6: 90, 7: 90},
        branch_mode="fresh_env",
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

    move_labels = [
        json.dumps({"action": action}, sort_keys=True, separators=(",", ":"))
        for action in (1, 2, 3, 4)
    ]

    def _click_label(game, sprite):
        grid_w, grid_h = getattr(getattr(game, "current_level", None), "grid_size", None) or (
            64,
            64,
        )
        grid_x = int(getattr(sprite, "x", 0)) + int(getattr(sprite, "width", 1)) // 2
        grid_y = int(getattr(sprite, "y", 0)) + int(getattr(sprite, "height", 1)) // 2
        display_x = grid_x + max(0, (64 - int(grid_w)) // 2)
        display_y = grid_y + max(0, (64 - int(grid_h)) // 2)
        return json.dumps(
            {
                "action": 6,
                "grid": [grid_x, grid_y],
                "sprite": str(getattr(sprite, "name", "")),
                "x": display_x,
                "y": display_y,
            },
            sort_keys=True,
            separators=(",", ":"),
        )

    def _current_click_labels(game):
        level = getattr(game, "current_level", None)
        getter = getattr(level, "get_sprites_by_tag", None)
        if not callable(getter):
            return []
        labels = []
        for sprite in sorted(
            getter("buezna"),
            key=lambda item: (
                str(getattr(item, "name", "")),
                int(getattr(item, "x", 0)),
                int(getattr(item, "y", 0)),
            ),
        ):
            if "sys_click" not in getattr(sprite, "tags", []):
                continue
            labels.append(_click_label(game, sprite))
        return labels

    def action_labels(env, frame=None, path=None):
        del frame, path
        game = getattr(env, "_game", None)
        return [*move_labels, *_current_click_labels(game)]

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
        player = getattr(game, "qnnpcoyzd", None)
        goal = getattr(game, "hfuqkxulm", None)
        if player is None or goal is None:
            return 1000.0
        return float(abs(int(player.x) - int(goal.x)) + abs(int(player.y) - int(goal.y)))

    def featurize(game):
        player = getattr(game, "qnnpcoyzd", None)
        goal = getattr(game, "hfuqkxulm", None)
        px = float(getattr(player, "x", 0))
        py = float(getattr(player, "y", 0))
        gx = float(getattr(goal, "x", 0))
        gy = float(getattr(goal, "y", 0))
        distance = abs(px - gx) + abs(py - gy)
        steps = float(getattr(getattr(game, "ujotjblwn", None), "current_steps", 0) or 0)
        return [distance, px, py, gx, gy, float(len(_current_click_labels(game))), steps]

    return GameAdapter(
        game="dc22",
        action_labels=action_labels,
        apply=apply,
        state_key=state_key,
        featurize=featurize,
        hand_verifier=hand_verifier,
        warmup_label=None,
        depth_caps={1: 24, 2: 80, 3: 120, 4: 160, 5: 256, 6: 512},
        branch_mode="fresh_env",
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

    def action_labels(env, frame=None, path=None):
        del env
        from carnot.agentic import arc_solver_kit as kit

        level = kit.frame_level(frame) if frame is not None else 0
        extension_index = len(path or ())
        if level == 0 and extension_index < len(AR25_L1_LABELS):
            return [AR25_L1_LABELS[extension_index]]
        if level == 1 and extension_index < len(AR25_L2_TAIL_LABELS):
            return [AR25_L2_TAIL_LABELS[extension_index]]
        if level == 2 and extension_index < len(AR25_L3_TAIL_LABELS):
            return [AR25_L3_TAIL_LABELS[extension_index]]
        return []

    def state_key(game, frame=None):
        return _hidden_state_key("ar25", game, frame)

    def _mirror_object_rows(game):
        mirror = getattr(game, "iabnfcqotmd", None)
        selected = getattr(game, "yvifanjrcyu", None)
        movable = [
            sprite
            for sprite in getattr(game, "ouurgkpbbjj", []) or []
            if "0056icpryeujyf" not in getattr(sprite, "tags", [])
        ]
        primary = movable[0] if movable else None
        return mirror, primary, selected

    def featurize(game):
        mirror, primary, selected = _mirror_object_rows(game)
        mirror_x = float(getattr(mirror, "x", 0))
        object_x = float(getattr(primary, "x", 0))
        object_y = float(getattr(primary, "y", 0))
        selected_is_object = float(primary is not None and selected is primary)
        selected_is_mirror = float(mirror is not None and selected is mirror)
        hidden = hidden_state_registers("ar25", game)
        return [
            float(getattr(game, "level_index", 0)),
            mirror_x,
            object_x,
            object_y,
            abs(mirror_x - 10.0),
            abs(object_x - 15.0) + abs(object_y - 14.0),
            selected_is_object,
            selected_is_mirror,
            float(hidden.get("undo_stack_depth") or 0),
            float(hidden.get("step_counter_current_steps") or 0),
        ]

    def hand_verifier(game, frame=None):
        del frame
        mirror, primary, selected = _mirror_object_rows(game)
        if mirror is None or primary is None:
            return 1000.0
        distance = (
            abs(float(mirror.x) - 10.0)
            + abs(float(primary.x) - 15.0)
            + abs(float(primary.y) - 14.0)
        )
        if mirror.x != 10 and selected is not mirror:
            distance += 4.0
        if mirror.x == 10 and selected is not primary:
            distance += 2.0
        return float(distance)

    return GameAdapter(
        game="ar25",
        replay=True,
        replay_source="AR25_L1_LABELS (module constant)",
        action_labels=action_labels,
        apply=exp4339._apply_ar25_label,
        state_key=state_key,
        featurize=featurize,
        hand_verifier=hand_verifier,
        warmup_label=None,
        depth_caps={
            1: len(AR25_L1_LABELS),
            2: len(AR25_L2_TAIL_LABELS),
            3: len(AR25_L3_TAIL_LABELS),
        },
        branch_mode="fresh_env",
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
        replay=True,
        replay_source="exp4350.L1_SOLUTION_LABELS",
        action_labels=action_labels,
        apply=exp4350._apply_ka59_label,
        label_to_action_data=exp4350._label_to_action_data,
        state_key=state_key,
        featurize=None,
        hand_verifier=lambda _game, _frame=None: 0.0,
        warmup_label=None,
        depth_caps={1: len(l1_labels), 2: 2, 3: 2},
        branch_mode="fresh_env",
    )


def _ft09():
    """ft09 -- local constraint puzzle with internal color-cycle state."""
    from carnot import experiment_4363_e3_mechanic_limited_tails_tr87_ft09 as exp4363

    def action_labels(env, frame=None, path=None):
        from carnot.agentic import arc_solver_kit as kit

        del env
        level = kit.frame_level(frame) if frame is not None else 0
        extension_index = len(path or ())
        if level == 0 and extension_index < len(FT09_L1_LABELS):
            return [FT09_L1_LABELS[extension_index]]
        if level == 1 and extension_index < len(FT09_L2_TAIL_LABELS):
            return [FT09_L2_TAIL_LABELS[extension_index]]
        if level == 2 and extension_index < len(FT09_L3_TAIL_LABELS):
            return [FT09_L3_TAIL_LABELS[extension_index]]
        return []

    def state_key(game, frame=None):
        return _hidden_state_key("ft09", game, frame)

    def _constraint_violations(game):
        violations = 0
        constrained = 0
        for sprite in getattr(game, "gig", []) or []:
            center = int(sprite.pixels[1][1])
            for j, dy in enumerate((-4, 0, 4)):
                for i, dx in enumerate((-4, 0, 4)):
                    if dx == 0 and dy == 0:
                        continue
                    target = game.current_level.get_sprite_at(sprite.x + dx, sprite.y + dy, "Hkx")
                    if not target:
                        target = game.current_level.get_sprite_at(
                            sprite.x + dx, sprite.y + dy, "NTi"
                        )
                    if not target:
                        continue
                    constrained += 1
                    should_equal = int(sprite.pixels[j][i]) == 0
                    equal = int(target.pixels[1][1]) == center
                    if equal != should_equal:
                        violations += 1
        return violations, constrained

    def featurize(game):
        violations, _constrained = _constraint_violations(game)
        cells = list(getattr(game, "fhc", []) or []) + list(getattr(game, "mou", []) or [])
        cycle = tuple(getattr(game, "gqb", []) or ())
        non_initial = 0
        if cycle:
            first = int(cycle[0])
            non_initial = sum(1 for sprite in cells if int(sprite.pixels[1][1]) != first)
        return [
            float(violations),
            float(non_initial),
            float(getattr(getattr(game, "lpw", None), "dzy", 0) or 0),
        ]

    return GameAdapter(
        game="ft09",
        replay=True,
        replay_source="banked L1 plan (module constant)",
        action_labels=action_labels,
        apply=exp4363._apply_ft09_label,
        state_key=state_key,
        featurize=featurize,
        hand_verifier=lambda game, _frame=None: float(_constraint_violations(game)[0]),
        warmup_label=None,
        depth_caps={
            1: len(FT09_L1_LABELS),
            2: len(FT09_L2_TAIL_LABELS),
            3: len(FT09_L3_TAIL_LABELS),
        },
        level_tails={
            1: FT09_L1_LABELS,
            2: FT09_L2_TAIL_LABELS,
            3: FT09_L3_TAIL_LABELS,
        },
        branch_mode="replay",
    )


def _vc33():
    """vc33 -- support-clearance config puzzle, deepened from L1 to L2."""
    from carnot.agentic import arc_solver_kit as kit
    from carnot.agentic.arc_agi3_live_adapter import _game_action

    def action_labels(env, frame=None, path=None):
        del env
        level = kit.frame_level(frame) if frame is not None else 0
        extension_index = len(path or ())
        if level == 0 and extension_index < len(VC33_L1_LABELS):
            return [VC33_L1_LABELS[extension_index]]
        if level == 1 and extension_index < len(VC33_L2_TAIL_LABELS):
            return [VC33_L2_TAIL_LABELS[extension_index]]
        return []

    def apply(env, label, frame):
        del frame
        row = json.loads(label)
        data = row.get("data") if isinstance(row.get("data"), dict) else None
        return env.step(_game_action(GameAction, int(row["action"])), data=data)

    def _sprite_rows(game, tag):
        try:
            sprites = game.current_level.get_sprites_by_tag(tag)
        except Exception:
            return ()
        rows = []
        for sprite in sprites:
            rows.append(
                (
                    str(getattr(sprite, "name", "")),
                    int(getattr(sprite, "x", 0)),
                    int(getattr(sprite, "y", 0)),
                    int(getattr(sprite, "width", 0)),
                    int(getattr(sprite, "height", 0)),
                    _sprite_color(sprite),
                )
            )
        return tuple(sorted(rows))

    def _support_distance(game):
        # `ielczunthe` is the executable support-clearance predicate. The wall
        # width residual gives the learned verifier a smooth route before it flips.
        try:
            if bool(game.ielczunthe()):
                return 0.0
        except Exception:
            pass
        walls = _sprite_rows(game, "0043nzrtobajqi")
        width_residual = sum(max(0, int(row[3]) - 2) for row in walls)
        return float(max(1, width_residual))

    def state_key(game, frame=None):
        level = (
            kit.frame_level(frame) if frame is not None else int(getattr(game, "level_index", 0))
        )
        return (
            int(level),
            _frame_grid_state_key(frame),
            _sprite_rows(game, "0022jvmlspyigc"),
            _sprite_rows(game, "0043nzrtobajqi"),
            _sprite_rows(game, "0016uciqlhjlom"),
            _sprite_rows(game, "0010gnulkywfpz"),
        )

    def featurize(game):
        supports = _sprite_rows(game, "0022jvmlspyigc")
        walls = _sprite_rows(game, "0043nzrtobajqi")
        support_sum = sum(row[1] + row[2] for row in supports)
        wall_area = sum(row[3] * row[4] for row in walls)
        return [
            float(getattr(game, "level_index", 0) or 0),
            _support_distance(game),
            float(len(supports)),
            float(wall_area),
            float(support_sum),
        ]

    return GameAdapter(
        game="vc33",
        replay=True,
        replay_source="banked L1 plan (module constant)",
        action_labels=action_labels,
        apply=apply,
        state_key=state_key,
        featurize=featurize,
        hand_verifier=lambda game, _frame=None: _support_distance(game),
        warmup_label=None,
        depth_caps={1: len(VC33_L1_LABELS), 2: len(VC33_L2_TAIL_LABELS), 3: 2},
        level_tails={1: VC33_L1_LABELS, 2: VC33_L2_TAIL_LABELS},
        branch_mode="replay",
    )


def _ls20():
    """ls20 -- clean navigation plus visible shape/color/rotation target matching.

    The L1 seed is the adapter-free graph-explore trajectory already in the
    registry. L2 keeps the same four keyboard moves, but adds a step-counter
    constraint: route to the rotation trigger three times, collect both visible
    reset pickups before the counter expires, then enter the target with the
    required shape/color/rotation tuple. The replay labels below are the
    derived next-level delta and are still gated by arc_solver_kit.reproduce().
    """
    from carnot.agentic import arc_solver_kit as kit
    from carnot.agentic.arc_agi3_live_adapter import _game_action

    def action_labels(env, frame=None, path=None):
        del env
        level = kit.frame_level(frame) if frame is not None else 0
        extension_index = len(path or ())
        if level == 0 and extension_index < len(LS20_L1_LABELS):
            return [LS20_L1_LABELS[extension_index]]
        if level == 1 and extension_index < len(LS20_L2_TAIL_LABELS):
            return [LS20_L2_TAIL_LABELS[extension_index]]
        return []

    def apply(env, label, frame):
        del frame
        row = json.loads(label)
        return env.step(_game_action(GameAction, int(row["action"])))

    def _target_rows(game):
        rows = []
        targets = list(getattr(game, "plrpelhym", []) or [])
        shapes = list(getattr(game, "ldxlnycps", []) or [])
        colors = list(getattr(game, "yjdexjsoa", []) or [])
        rotations = list(getattr(game, "ehwheiwsk", []) or [])
        consumed = list(getattr(game, "lvrnuajbl", []) or [])
        for index, sprite in enumerate(targets):
            rows.append(
                (
                    int(getattr(sprite, "x", 0)),
                    int(getattr(sprite, "y", 0)),
                    int(shapes[index]) if index < len(shapes) else -1,
                    int(colors[index]) if index < len(colors) else -1,
                    int(rotations[index]) if index < len(rotations) else -1,
                    bool(consumed[index]) if index < len(consumed) else False,
                )
            )
        return tuple(rows)

    def state_key(game, frame=None):
        level = (
            kit.frame_level(frame)
            if frame is not None
            else int(getattr(game, "level_index", 0) or 0)
        )
        player = getattr(game, "gudziatsk", None)
        step_counter = getattr(getattr(game, "_step_counter_ui", None), "current_steps", 0)

        def _visible(sprite):
            value = getattr(sprite, "is_visible", None)
            if callable(value):
                return bool(value())
            if value is not None:
                return bool(value)
            return bool(getattr(sprite, "visible", True))

        reset_pickups = tuple(
            sorted(
                (
                    str(getattr(sprite, "name", "")),
                    int(getattr(sprite, "x", 0)),
                    int(getattr(sprite, "y", 0)),
                    _visible(sprite),
                )
                for sprite in getattr(game, "ofoahudlo", [])
            )
        )
        return (
            int(level),
            int(getattr(player, "x", -1)),
            int(getattr(player, "y", -1)),
            int(getattr(game, "fwckfzsyc", 0) or 0),
            int(getattr(game, "hiaauhahz", 0) or 0),
            int(getattr(game, "cklxociuu", 0) or 0),
            _target_rows(game),
            int(step_counter or 0),
            int(getattr(game, "aqygnziho", 0) or 0),
            int(getattr(game, "ebfuxzbvn", 0) or 0),
            int(getattr(game, "akoadfsur", 0) or 0),
            len(getattr(game, "euemavvxz", []) or []),
            reset_pickups,
        )

    def hand_verifier(game, frame=None):
        del frame
        player = getattr(game, "gudziatsk", None)
        if player is None:
            return 1000.0
        best = 1000.0
        for x, y, shape, color, rotation, consumed in _target_rows(game):
            if consumed:
                continue
            distance = abs(int(player.x) - x) + abs(int(player.y) - y)
            distance += 20 * (int(getattr(game, "fwckfzsyc", 0) or 0) != shape)
            distance += 20 * (int(getattr(game, "hiaauhahz", 0) or 0) != color)
            current_rotation = int(getattr(game, "cklxociuu", 0) or 0)
            distance += 8 * min(
                (current_rotation - rotation) % 4,
                (rotation - current_rotation) % 4,
            )
            best = min(best, float(distance))
        return 0.0 if best == 1000.0 else best

    def featurize(game):
        player = getattr(game, "gudziatsk", None)
        step_counter = getattr(getattr(game, "_step_counter_ui", None), "current_steps", 0)
        verifier = hand_verifier(game)
        unconsumed = sum(1 for row in _target_rows(game) if not row[-1])
        return [
            verifier,
            float(getattr(player, "x", 0)),
            float(getattr(player, "y", 0)),
            float(getattr(game, "cklxociuu", 0) or 0),
            float(step_counter or 0),
            float(unconsumed),
        ]

    return GameAdapter(
        game="ls20",
        replay=True,
        replay_source="banked L1 plan (module constant)",
        action_labels=action_labels,
        apply=apply,
        state_key=state_key,
        featurize=featurize,
        hand_verifier=hand_verifier,
        warmup_label=None,
        depth_caps={1: len(LS20_L1_LABELS), 2: len(LS20_L2_TAIL_LABELS), 3: 2},
        branch_mode="fresh_env",
    )


def _sk48():
    """sk48 -- chain-color reorder L1->L2 delta registered from the offline env."""
    from carnot.agentic.arc_agi3_live_adapter import _game_action
    from carnot.agentic import arc_solver_kit as kit

    def action_labels(env, frame=None, path=None):
        del env
        level = kit.frame_level(frame) if frame is not None else 0
        extension_index = len(path or ())
        if level == 0 and extension_index < len(SK48_L1_LABELS):
            return [SK48_L1_LABELS[extension_index]]
        if level == 1 and extension_index < len(SK48_L2_TAIL_LABELS):
            return [SK48_L2_TAIL_LABELS[extension_index]]
        return []

    def apply(env, label, frame):
        del frame
        row = json.loads(label)
        action = int(row["action"])
        data = row.get("data") if isinstance(row.get("data"), dict) else None
        return env.step(_game_action(GameAction, action), data=data)

    def _sprite_row(sprite):
        return (
            str(getattr(sprite, "name", "")),
            int(getattr(sprite, "x", 0)),
            int(getattr(sprite, "y", 0)),
            int(getattr(sprite, "rotation", 0)),
            _sprite_color(sprite),
            bool(getattr(sprite, "visible", True)),
        )

    def _chain_rows(game):
        rows = []
        for head, chain in getattr(game, "mwfajkguqx", {}).items():
            rows.append(
                (
                    _sprite_row(head),
                    tuple(_sprite_row(segment) for segment in chain),
                    _sprite_row(getattr(game, "xpmcmtbcv", {}).get(head)),
                )
            )
        return tuple(sorted(rows))

    def _match_counts(game):
        try:
            game.gvtmoopqgy()
            pairs = getattr(game, "xpmcmtbcv", {})
            seen = getattr(game, "vjfbwggsd", {})
            guides = getattr(game, "jdojcthkf", {})
            missing = 0
            mismatched = 0
            matched = 0
            for active, paired in pairs.items():
                active_seen = list(seen.get(active, []) or [])
                paired_seen = list(seen.get(paired, []) or [])
                required = max(len(guides.get(paired, []) or []), len(paired_seen))
                for index in range(required):
                    if index >= len(active_seen) or index >= len(paired_seen):
                        missing += 1
                    elif _sprite_color(active_seen[index]) == _sprite_color(paired_seen[index]):
                        matched += 1
                    else:
                        mismatched += 1
            return matched, mismatched, missing
        except Exception:
            return 0, 1000, 0

    def state_key(game, frame=None):
        level = (
            kit.frame_level(frame)
            if frame is not None
            else int(getattr(game, "level_index", 0) or 0)
        )
        active = getattr(game, "vzvypfsnt", None)
        return (
            int(level),
            _sprite_row(active),
            _chain_rows(game),
            int(getattr(game, "qiercdohl", 0) or 0),
            int(getattr(game, "lgdrixfno", -1) or -1),
        )

    def hand_verifier(game, frame=None):
        del frame
        _matched, mismatched, missing = _match_counts(game)
        return float(mismatched + missing)

    def featurize(game):
        matched, mismatched, missing = _match_counts(game)
        active = getattr(game, "vzvypfsnt", None)
        return [
            float(mismatched),
            float(missing),
            float(matched),
            float(len(getattr(game, "mwfajkguqx", {}) or {})),
            float(len((getattr(game, "mwfajkguqx", {}) or {}).get(active, []) or [])),
            float(getattr(game, "qiercdohl", 0) or 0),
        ]

    return GameAdapter(
        game="sk48",
        replay=True,
        replay_source="banked L1 plan (module constant)",
        action_labels=action_labels,
        apply=apply,
        state_key=state_key,
        featurize=featurize,
        hand_verifier=hand_verifier,
        warmup_label=None,
        depth_caps={1: len(SK48_L1_LABELS), 2: len(SK48_L2_TAIL_LABELS), 3: 2},
        branch_mode="fresh_env",
    )


def _r11l():
    """r11l -- click-template handle averaging with an offline-gated L2 tail."""
    import numpy as np

    from carnot.agentic import arc_solver_kit as kit
    from carnot.agentic.arc_agi3_live_adapter import _game_action

    def action_labels(env, frame=None, path=None):
        del env
        level = kit.frame_level(frame) if frame is not None else 0
        extension_index = len(path or ())
        if level == 0 and extension_index < len(R11L_L1_LABELS):
            return [R11L_L1_LABELS[extension_index]]
        if level == 1 and extension_index < len(R11L_L2_TAIL_LABELS):
            return [R11L_L2_TAIL_LABELS[extension_index]]
        return []

    def apply(env, label, frame):
        del frame
        step = json.loads(str(label))
        data = step.get("data") if isinstance(step.get("data"), dict) else step
        return env.step(_game_action(GameAction, int(step.get("action", 6))), data=data)

    def _sprite_row(sprite):
        return (
            str(getattr(sprite, "name", "")),
            int(getattr(sprite, "x", 0)),
            int(getattr(sprite, "y", 0)),
            int(getattr(sprite, "width", 0)),
            int(getattr(sprite, "height", 0)),
            bool(getattr(sprite, "is_visible", True)),
        )

    def _handles(game):
        return tuple(
            sorted(_sprite_row(sprite) for sprite in getattr(game, "bbijaigbknc", []) or [])
        )

    def _pieces(game):
        rows = []
        for name, sprites in (getattr(game, "bulmhgivatv", {}) or {}).items():
            if "dirwzt" in str(name):
                continue
            rows.append((str(name), tuple(sorted(_sprite_row(sprite) for sprite in sprites or []))))
        return tuple(sorted(rows))

    def _templates(game):
        rows = []
        for name, sprites in (getattr(game, "pebahdr", {}) or {}).items():
            if "dirwzt" in str(name):
                continue
            rows.append((str(name), tuple(sorted(_sprite_row(sprite) for sprite in sprites or []))))
        return tuple(sorted(rows))

    def state_key(game, frame=None):
        return (
            kit.frame_level(frame)
            if frame is not None
            else int(getattr(game, "level_index", 0) or 0),
            int(getattr(game, "holbcmkehyf", -1) or -1),
            _handles(game),
            _pieces(game),
            _templates(game),
        )

    def _rect_distance(source, target) -> int:
        return abs(int(source[1]) - int(target[1])) + abs(int(source[2]) - int(target[2]))

    def hand_verifier(game, _frame=None):
        by_target = {name: sprites for name, sprites in _templates(game)}
        distance = 0
        for name, sprites in _pieces(game):
            targets = by_target.get(name, ())
            for index, sprite in enumerate(sprites):
                if index < len(targets):
                    distance += _rect_distance(sprite, targets[index])
                else:
                    distance += 100
        return float(distance)

    def featurize(game):
        pixels = np.asarray(game.get_pixels(0, 0, 64, 64))
        if pixels.ndim == 3:
            pixels = pixels[0]
        flat = [int(value) for value in pixels.reshape(-1).tolist()]
        nonzero = [value for value in flat if value != 0]
        return [
            float(len(nonzero)),
            float(len(set(nonzero))),
            float(pixels.size),
        ]

    return GameAdapter(
        game="r11l",
        replay=True,
        replay_source="banked L1 plan (module constant)",
        action_labels=action_labels,
        apply=apply,
        state_key=state_key,
        featurize=featurize,
        hand_verifier=hand_verifier,
        warmup_label=None,
        depth_caps={1: 0, 2: 0, 3: 0},
        level_tails={1: R11L_L1_LABELS, 2: R11L_L2_TAIL_LABELS},
        branch_mode="fresh_env",
    )


# ---------------- lf52 (rail-carried peg-jump; L1 seed + L2 delta) ----------------
def _lf52():
    """lf52 -- peg-jump removal with a rail-carried landing cell."""
    from carnot.agentic import arc_solver_kit as kit
    from carnot.agentic.arc_agi3_live_adapter import _game_action

    def action_labels(env, frame=None, path=None):
        del env
        level = kit.frame_level(frame) if frame is not None else 0
        extension_index = len(path or ())
        if level == 0 and extension_index < len(LF52_L1_LABELS):
            return [LF52_L1_LABELS[extension_index]]
        if level == 1 and extension_index < len(LF52_L2_TAIL_LABELS):
            return [LF52_L2_TAIL_LABELS[extension_index]]
        return []

    def apply(env, label, frame):
        del frame
        step = json.loads(label)
        return env.step(_game_action(GameAction, int(step["action"])), data=step.get("data"))

    def _objects(grid, names):
        rows = []
        getter = getattr(grid, "ndtvadsrqf", None)
        for name in names:
            for obj in getter(name) if callable(getter) else []:
                cdpcbbnfdp = getattr(obj, "cdpcbbnfdp", (0, 0))
                rows.append(
                    (
                        str(getattr(obj, "name", name)),
                        int(getattr(obj, "grid_x", 0)),
                        int(getattr(obj, "grid_y", 0)),
                        int(cdpcbbnfdp[0]),
                        int(cdpcbbnfdp[1]),
                    )
                )
        return tuple(sorted(rows))

    def _carrier_cells(grid):
        rows = []
        getter = getattr(grid, "whdmasyorl", None)
        for obj in getter("hupkpseyuim2") if callable(getter) else []:
            cdpcbbnfdp = getattr(obj, "cdpcbbnfdp", (0, 0))
            rows.append(
                (
                    int(getattr(obj, "grid_x", 0)),
                    int(getattr(obj, "grid_y", 0)),
                    int(cdpcbbnfdp[0]),
                    int(cdpcbbnfdp[1]),
                )
            )
        return tuple(sorted(rows))

    def _selected_row(core):
        selected = getattr(getattr(core, "wpwvsglgmb", None), "qoifrofmiu", None)
        if selected is None:
            return None
        cdpcbbnfdp = getattr(selected, "cdpcbbnfdp", (0, 0))
        return (
            str(getattr(selected, "name", "")),
            int(getattr(selected, "grid_x", 0)),
            int(getattr(selected, "grid_y", 0)),
            int(cdpcbbnfdp[0]),
            int(cdpcbbnfdp[1]),
        )

    def state_key(game, frame=None):
        core = game.ikhhdzfmarl
        grid = core.hncnfaqaddg
        return (
            kit.frame_level(frame) if frame is not None else -1,
            int(getattr(core, "whtqurkphir", 0) or 0),
            _objects(grid, ("fozwvlovdui", "fozwvlovdui_red", "fozwvlovdui_blue")),
            _carrier_cells(grid),
            _selected_row(core),
            tuple(int(v) for v in getattr(grid, "cdpcbbnfdp", (0, 0))),
            bool(getattr(core, "zvcnglshzcx", False)),
            bool(getattr(core, "yxhdgwykzi", False)),
            bool(getattr(core, "iajuzrgttrv", False)),
            bool(getattr(core, "evxflhofing", False)),
        )

    def featurize(game):
        core = game.ikhhdzfmarl
        grid = core.hncnfaqaddg
        pegs = _objects(grid, ("fozwvlovdui", "fozwvlovdui_red", "fozwvlovdui_blue"))
        carriers = _carrier_cells(grid)
        selected = _selected_row(core)
        offset = tuple(int(v) for v in getattr(grid, "cdpcbbnfdp", (0, 0)))
        return [
            float(len(pegs)),
            float(getattr(core, "asqvqzpfdi", 0) or 0),
            float(len(carriers)),
            float(selected is not None),
            float(max(0, int(getattr(core, "whtqurkphir", 1) or 1) - 1)),
            float(offset[0] if offset else 0),
            float(offset[1] if len(offset) > 1 else 0),
        ]

    def hand_verifier(game, _frame=None):
        core = game.ikhhdzfmarl
        pegs = len(
            _objects(core.hncnfaqaddg, ("fozwvlovdui", "fozwvlovdui_red", "fozwvlovdui_blue"))
        )
        return float(max(0, pegs - 1) * 10 + (0 if _selected_row(core) else 1))

    return GameAdapter(
        game="lf52",
        replay=True,
        replay_source="banked L1 plan (module constant)",
        action_labels=action_labels,
        apply=apply,
        state_key=state_key,
        featurize=featurize,
        hand_verifier=hand_verifier,
        warmup_label=None,
        depth_caps={1: len(LF52_L1_LABELS), 2: len(LF52_L2_TAIL_LABELS), 3: 2},
        level_tails={1: LF52_L1_LABELS, 2: LF52_L2_TAIL_LABELS},
        branch_mode="fresh_env",
    )


def _sc25():
    """sc25 -- cast-grid spell puzzle. L1 replays exp4468's banked plan.

    The RE was already done and captured (registry: win_condition + action_model); this
    adapter only exposes it through the GameAdapter interface so `solve_adaptered` can reach
    L1 and `build_progress_window` can cut an induction window. Before this, sc25 had no
    adapter at all, so `_live_verifier_for_adapter` hit `None.hand_verifier` and the game was
    silently absent from every offline A/B corpus.
    """
    from carnot import experiment_4468_bank_sc25_provisional_levels as exp4468

    l1_labels = tuple(exp4468.SC25_PLANS_BY_LEVEL[1])

    def action_labels(env, frame=None, path=None):
        from carnot.agentic import arc_solver_kit as kit

        level = kit.frame_level(frame) if frame is not None else 0
        i = len(path or ())
        # Force the banked L1 plan while at level 0. Same shape as _ka59: the search becomes
        # a straight line to the level-up, which is all a window builder needs. Past L1 we
        # hand back the real action space rather than pretending the plan continues.
        if level == 0 and i < len(l1_labels):
            return [l1_labels[i]]
        if level >= 1:
            return [f"move{d}" for d in (1, 2, 3, 4)]
        return []

    def state_key(game, frame=None):
        return _hidden_state_key("sc25", game, frame)

    return GameAdapter(
        game="sc25",
        replay=True,
        replay_source="exp4468.SC25_PLANS_BY_LEVEL[1]",
        action_labels=action_labels,
        apply=exp4468.apply_sc25_label,
        label_to_action_data=lambda _env, label: exp4468.sc25_label_to_action_data(label),
        state_key=state_key,
        featurize=None,
        hand_verifier=lambda _game, _frame=None: 0.0,
        warmup_label="warmup",
        branch_mode="replay",
    )


def _tn36():
    """tn36 -- PROGRAM-EDITOR mechanic. Drives scripts/arc3_tn36_offline_solver.py's logic.

    NOT a replayed label list, deliberately. The registry's own RE note says the slot layout
    "RE-LAYS-OUT per level -> DISCOVER, don't hardcode", and the solver's `clk` re-issues a
    click while an animation flag is set, so a fixed sequence would desynchronise. This
    adapter is STATE-DRIVEN instead: on each call it compares the live program to the target
    program and emits the next single bit-toggle still needed, then the run button. A click
    that did not register leaves the state unchanged and is simply re-emitted, which is the
    same convergence the solver's retry loop gets.
    """
    import importlib.util
    import sys as _sys
    from pathlib import Path as _Path

    from carnot.paths import repo_root

    # The solver lives in scripts/, which is not an importable package.
    _spec = importlib.util.spec_from_file_location(
        "_arc3_tn36_offline_solver", _Path(repo_root()) / "scripts" / "arc3_tn36_offline_solver.py"
    )
    _tn = importlib.util.module_from_spec(_spec)
    _sys.modules[_spec.name] = _tn
    _spec.loader.exec_module(_tn)

    def _target_program(env):
        """The slot program that nets the current obj->target delta, or None if unreachable."""
        obj, tgt = _tn._obj_tgt(env)
        moves, why = _tn.compute_moves(obj, tgt)
        if not moves:
            return None
        n = len(_tn._slot_tops(env))
        orders = _tn._orderings(moves, n)
        return list(orders[0]) if orders else None

    def action_labels(env, frame=None, path=None):
        from carnot.agentic import arc_solver_kit as kit

        level = kit.frame_level(frame) if frame is not None else 0
        if level >= 1:
            return []  # L1 is all the window builder needs; do not invent a deeper action set
        target = _target_program(env)
        if target is None:
            return []
        cur, tops = _tn._program(env), _tn._slot_tops(env)
        for i, top in enumerate(tops):
            if i >= len(target):
                break
            diff = int(cur[i]) ^ int(target[i])
            if diff:
                b = (diff & -diff).bit_length() - 1  # lowest set bit still wrong
                return [
                    _json_action_label(6, {"x": int(top[0]), "y": int(top[1]) + _tn.BIT_DY * b})
                ]
        rx, ry = _tn._run_xy(env)
        return [_json_action_label(6, {"x": int(rx), "y": int(ry)})]

    def state_key(game, frame=None):
        return _hidden_state_key("tn36", game, frame)

    return GameAdapter(
        game="tn36",
        action_labels=action_labels,
        apply=_default_json_apply,
        state_key=state_key,
        featurize=None,
        hand_verifier=lambda _game, _frame=None: 0.0,
        warmup_label=None,
        branch_mode="replay",
    )


def _wa30():
    """wa30 -- block-delivery puzzle with helper-robot NPCs. Replays the verified L1 prefix.

    THE LAST OF THE 25 PUBLIC GAMES TO GET AN ADAPTER (2026-07-31). wa30 was solved by a
    16-round OUTER-LOOP agent probe campaign (gpt-5.6-sol v1-v8 stalled at L1; claude-fable-5
    v9-v16 cleared it to L9), never by the standing loop -- so no adapter was ever produced as
    a by-product, and `_live_verifier_for_adapter` raised `AttributeError` on the None, which
    read as "unsolvable" rather than "unregistered". That is why wa30 sat outside every
    offline induction corpus.

    L1 SOURCE, and why it is a replay rather than a computed solver: unlike tn36 -- whose
    registry entry records a full mechanic (bit-toggle program editor) that a solver can
    compute from live state -- wa30's captured RE is a per-level ACTION SEQUENCE plus prose
    about NPC behaviour. There is no per-level parameterisation to compute from, so the
    honest thing is to replay the banked prefix and say so, not to dress a fixed list up as
    a general solver.

    PROVENANCE: the 33 labels below are the first 33 of the 670-action route in
    `results/outer_loop_fable5_wa30_probe_l9.json`, whose full sequence was gate-verified to
    L9 on 2026-07-31 (`results/outer_loop_arc_wa30_reproduction_gate_20260731.json`). 33 is
    the MINIMAL L1 prefix -- measured by replaying prefixes 20..45, not taken from the prose.
    All actions are plain ACTION1-5 keyboard steps; the route contains zero ACTION6 clicks,
    so there is no live out-of-bounds risk.

    SCOPE: L1 only, like the majority of adapters here -- see this module's docstring on why
    the delta is per LEVEL. wa30's own levels bear that out sharply: L2 adds a passive
    helper-robot NPC, L3 a fence-socket relay, L4 three side-affiliated robots. None of that
    is expressible in an L1 action vocabulary, and hand-adaptering each would be the
    outer_loop_re treadmill rather than progress.
    """
    import json as _json
    from pathlib import Path as _Path

    from carnot.paths import repo_root

    _probe = _Path(repo_root()) / "results" / "outer_loop_fable5_wa30_probe_l9.json"
    _seq = _json.loads(_probe.read_text())["action_sequence"]
    # Read from the artifact rather than pasting 33 literals: if the banked route is ever
    # re-derived, the adapter follows it instead of silently diverging from the gated one.
    L1_PREFIX = 33
    l1_labels = tuple(_json_action_label(int(a["action"])) for a in _seq[:L1_PREFIX])

    def action_labels(env, frame=None, path=None):
        from carnot.agentic import arc_solver_kit as kit

        level = kit.frame_level(frame) if frame is not None else 0
        i = len(path or ())
        if level == 0 and i < len(l1_labels):
            return [l1_labels[i]]
        if level >= 1:
            # Past L1 the banked route continues, but its mechanics (helper robots, fence
            # relay) are not modelled here. Hand back the bare keyboard vocabulary rather
            # than implying a capability this adapter does not have.
            return [_json_action_label(a) for a in (1, 2, 3, 4, 5)]
        return []

    def state_key(game, frame=None):
        return _hidden_state_key("wa30", game, frame)

    return GameAdapter(
        game="wa30",
        replay=True,
        replay_source="first 33 actions of results/outer_loop_fable5_wa30_probe_l9.json (gate-verified to L9)",
        action_labels=action_labels,
        apply=_default_json_apply,
        state_key=state_key,
        featurize=None,
        hand_verifier=lambda _game, _frame=None: 0.0,
        warmup_label=None,
        branch_mode="replay",
    )


_BUILDERS = {
    "ar25": _ar25,
    "bp35": _bp35,
    "ft09": _ft09,
    "g50t": _g50t,
    "sk48": _sk48,
    "lf52": _lf52,
    "r11l": _r11l,
    "re86": _re86,
    "s5i5": _s5i5,
    "ka59": _ka59,
    "cn04": _cn04,
    "su15": _su15,
    "sb26": _sb26,
    "sp80": _sp80,
    "ls20": _ls20,
    "lp85": _lp85,
    "tu93": _tu93,
    "tr87": _tr87,
    "dc22": _dc22,
    "cd82": _cd82,
    "m0r0": _m0r0,
    "vc33": _vc33,
    "sc25": _sc25,
    "tn36": _tn36,
    "wa30": _wa30,
}


def get_adapter(game: str) -> Optional[GameAdapter]:
    """Return the adapter for `game`, or None if it hasn't been RE'd/registered yet."""
    b = _BUILDERS.get(game)
    return b() if b else None


def adaptered_games() -> list[str]:
    return sorted(_BUILDERS)
