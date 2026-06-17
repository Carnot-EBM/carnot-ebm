"""Reusable ARC-AGI-3 MAZE planners — the solving strategies for two discovered mechanic classes:

  - checkpoint_multirun: a run that ENDS on a checkpoint advances the object's base, so a path that
    exceeds one program's slot budget is staged across checkpoints (waypoint BFS, each edge a
    <=n-move collision-free leg). tn36 L6 is the exemplar.
  - timed_trap_aware: the same staged routing PLUS blinking spike-traps that toggle visibility on a
    fixed move cadence and kill the object on contact while visible (and via a residual hidden hitbox
    while invisible). The crossing of the hazard band must happen during the invisible window, and
    the object must be clear of the band by the toggle. tn36 L7 is the exemplar.

These were extracted from the tn36 per-game RE (`scripts/arc3_tn36_offline_solver.py`) into
game-agnostic functions so the STRATEGY router (`arc_strategy_router`) can dispatch them to ANY game
whose mechanic is detected as checkpoint_multirun / timed_trap_aware. They operate on a generic
`MazeModel` (object box, walls, checkpoints, hazard boxes, the move-code mapping) — NOT on internal
game state. For a KNOWN game the model is read from internal state; for an unseen LIVE game the model
must come from frame-only induction (the remaining build, same honest pattern as the program-editor
strategy). The planners are pure + unit-tested on synthetic models; the game remains the final oracle
(a produced plan is validated against the real environment before it counts).

Timing model (tn36 cadence, parameterised by `invisible_slots`): within an n-slot run the spikes are
INVISIBLE during slots [0, invisible_slots) and VISIBLE during the rest, with a death-check on the
toggle at the end of slot (invisible_slots-1). A losing/checkpoint run fires an even number of toggles
(mid-run + end-of-run), so every run starts with the band invisible again — the schedule holds per
leg. Other games may toggle on a different cadence; `invisible_slots` makes that explicit.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Optional

Cell = tuple[int, int]
Box = tuple[int, int, int, int]


@dataclass
class MazeModel:
    """A game-agnostic maze the planners route over. Coordinates + sizes are in grid units."""

    object_wh: tuple[int, int]
    start: Cell
    target: Cell
    walls: list[Box]                       # collision REVERTS the move (not lethal)
    checkpoints: list[Cell]                # ending a run here advances the base
    move_codes: list[tuple[Cell, int]]     # [((dx,dy), command_code), ...] the 4 steps
    settle_code: int                       # the no-op padding code
    n_slots: int                           # program length per run
    bounds: int = 64
    spikes_visible: list[Box] = field(default_factory=list)   # lethal when visible (full band)
    spikes_hidden: list[Box] = field(default_factory=list)    # lethal when invisible (residual)
    invisible_slots: int = 3               # slots [0, this) are invisible; this.. are visible


def _overlap(ax: int, ay: int, aw: int, ah: int, b: Box) -> bool:
    bx, by, bw, bh = b
    return ax < bx + bw and ax + aw > bx and ay < by + bh and ay + ah > by


def _collide(model: MazeModel, x: int, y: int) -> bool:
    w, h = model.object_wh
    if x < 0 or y < 0 or x + w > model.bounds or y + h > model.bounds:
        return True
    return any(_overlap(x, y, w, h, wall) for wall in model.walls)


def _spike_death(model: MazeModel, x: int, y: int, visible: bool) -> bool:
    w, h = model.object_wh
    boxes = model.spikes_visible if visible else model.spikes_hidden
    return any(_overlap(x, y, w, h, b) for b in boxes)


def _leg(model: MazeModel, src: Cell, dst: Cell, timed: bool) -> Optional[list[int]]:
    """Shortest collision-free (and, when `timed`, spike-safe) leg src->dst as command codes, padded
    to n_slots with the settle code; None if unreachable. BFS over (pos, slot-index) so the spike
    visibility schedule is respected per slot."""
    n = model.n_slots
    steps = model.move_codes + [((0, 0), model.settle_code)]
    q: deque = deque([(src, 0, [])])
    seen = {(src, 0)}
    while q:
        (x, y), idx, codes = q.popleft()
        if (x, y) == dst and (not timed or (
                not _spike_death(model, x, y, True) and not _spike_death(model, x, y, False))):
            return codes + [model.settle_code] * (n - len(codes))
        if idx >= n:
            continue
        for (dx, dy), code in steps:
            nx, ny = x + dx, y + dy
            if _collide(model, nx, ny):
                nx, ny = x, y                          # wall collision reverts the move
            if timed:
                visible = idx >= model.invisible_slots
                if _spike_death(model, nx, ny, visible):
                    continue                            # dies during this slot's move-check
                if idx == model.invisible_slots - 1 and _spike_death(model, nx, ny, True):
                    continue                            # dies on the visibility toggle after this slot
            st = ((nx, ny), idx + 1)
            if st in seen:
                continue
            seen.add(st)
            q.append(((nx, ny), idx + 1, codes + [code]))
    return None


def _waypoint_plan(model: MazeModel, timed: bool) -> Optional[list[list[int]]]:
    """Waypoint BFS start -> (checkpoints) -> target; each edge a (timed) leg. Returns the list of
    leg-programs or None. An already-at-target model returns None (no plan needed)."""
    if model.start == model.target:
        return None
    waypoints = list(model.checkpoints) + [model.target]
    q: deque = deque([(model.start, [])])
    seen = {model.start}
    while q:
        node, legs = q.popleft()
        if node == model.target:
            return legs or None
        for nxt in waypoints:
            if nxt == node or nxt in seen:
                continue
            lp = _leg(model, node, nxt, timed)
            if lp is not None:
                seen.add(nxt)
                q.append((nxt, legs + [lp]))
    return None


def checkpoint_multirun_plan(model: MazeModel) -> Optional[list[list[int]]]:
    """Stage a path across checkpoints when it exceeds one run's slot budget (no hazards). Returns a
    list of leg-programs (each padded to n_slots) or None."""
    return _waypoint_plan(model, timed=False)


def timed_trap_plan(model: MazeModel) -> Optional[list[list[int]]]:
    """Stage a path across checkpoints while avoiding blinking spike-traps (cross the hazard band only
    during the invisible window, clear of residual hidden hitboxes, out of the band by the toggle).
    Requires `spikes_visible`; returns a list of leg-programs or None."""
    if not model.spikes_visible:
        return None
    return _waypoint_plan(model, timed=True)
