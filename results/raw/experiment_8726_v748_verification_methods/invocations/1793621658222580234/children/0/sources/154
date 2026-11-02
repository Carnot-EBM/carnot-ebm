"""OFFLINE transition model for the PROGRAM-EDITOR mechanic class — the durable unlock the frame-only
findings pointed at.

Why this exists
---------------
The program-editor mechanic (tn36) runs an N-slot move-program that transforms an object; the win is
the object matching a target on five attributes (x, y, scale, rotation, property). Three frame-only
findings (design note arc-live-generalization-gap-2026-06-17.md) showed the LIVE frame stream cannot
supply a planning signal for this class: the run is ATOMIC (no per-move motion, no graded feedback —
only a binary win bit), so winner-discovery degenerates to BLIND program-space search that does not
scale. The unlock is an OFFLINE model of the transition `(object_attrs, program) -> final_attrs`,
learned/encoded once from the reverse-engineered dynamics, that SCORES a candidate program by its
predicted distance-to-target — the gradient the frames withhold. The agent then PLANS (model-guided
best-first over programs) instead of blind-searching, and the real environment remains the final
oracle (a produced plan is validated by running it).

This is NOT a "moat" / oracle-distinct claim (CLAUDE.md Circularity Discipline): the model PREDICTS
the executable oracle's outcome to make search cheap (amortized planning); it is execution-grounded
(the winning plan is env-confirmed). Its value is measured the honest way — (1) WIN-BIT AGREEMENT with
the real env over many programs (does the model correctly predict which programs win?), and (2) the
search EFFICIENCY it buys (model-guided search finds the winner in far fewer env evaluations than
blind search).

The semantics are the tn36 `okllwtboml` dispatch (tn36.py:2171), reverse-engineered 2026-06-17:
  move 1/34=left, 2=right, 3=down, 33=up (±STEP); 10/11=right, 12/13=left (±2·STEP); a move REVERTS on
  a wall collision (the object box scales with `scale`). rotate 5=+90, 6=-90, 7=+180, 16=+270 (mod
  360). scale 8=+1, 9=-1 (clamped to >=1). property (absolute set) 14->9, 15->8, 63->15. 0=settle.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

STEP = 4

# code -> (dx, dy) in grid units (a move; reverts on wall collision)
MOVE_CODES: dict[int, tuple[int, int]] = {
    1: (-STEP, 0),
    34: (-STEP, 0),
    2: (STEP, 0),
    3: (0, STEP),
    33: (0, -STEP),
    10: (2 * STEP, 0),
    11: (2 * STEP, 0),
    12: (-2 * STEP, 0),
    13: (-2 * STEP, 0),
}
ROT_CODES: dict[int, int] = {5: 90, 6: -90, 7: 180, 16: 270}  # delta degrees
SCALE_CODES: dict[int, int] = {8: 1, 9: -1}  # delta, clamped to >=1
PROP_CODES: dict[int, int] = {14: 9, 15: 8, 63: 15}  # absolute set
SETTLE = 0


@dataclass(frozen=True)
class EditorState:
    """The object's five win-relevant attributes — the model's state."""

    x: int
    y: int
    scale: int
    rotation: int
    prop: int

    def matches(self, other: "EditorState") -> bool:
        return (self.x, self.y, self.scale, self.rotation, self.prop) == (
            other.x,
            other.y,
            other.scale,
            other.rotation,
            other.prop,
        )


@dataclass
class EditorGeometry:
    """The static layout the transition needs: the object's BASE box (at scale 1), the walls a move
    reverts against, and the bounds. Walls/bounds may be empty for open levels."""

    object_wh: tuple[int, int] = (4, 4)
    walls: tuple[tuple[int, int, int, int], ...] = ()
    bounds: int = 64


def _box_wh(geom: EditorGeometry, scale: int) -> tuple[int, int]:
    # the object box scales with `scale` (an L4 finding: scaling changes collision footprint)
    w0, h0 = geom.object_wh
    return w0 * scale, h0 * scale


def _collides(geom: EditorGeometry, x: int, y: int, scale: int) -> bool:
    w, h = _box_wh(geom, scale)
    if x < 0 or y < 0 or x + w > geom.bounds or y + h > geom.bounds:
        return True
    for wx, wy, ww, wh in geom.walls:
        if x < wx + ww and x + w > wx and y < wy + wh and y + h > wy:
            return True
    return False


def apply_code(state: EditorState, code: int, geom: EditorGeometry) -> EditorState:
    """Apply ONE command code's transform to the object state (the per-slot transition)."""
    if code in MOVE_CODES:
        dx, dy = MOVE_CODES[code]
        nx, ny = state.x + dx, state.y + dy
        if _collides(geom, nx, ny, state.scale):
            return state  # wall collision reverts the move
        return EditorState(nx, ny, state.scale, state.rotation, state.prop)
    if code in ROT_CODES:
        return EditorState(
            state.x, state.y, state.scale, (state.rotation + ROT_CODES[code]) % 360, state.prop
        )
    if code in SCALE_CODES:
        return EditorState(
            state.x, state.y, max(1, state.scale + SCALE_CODES[code]), state.rotation, state.prop
        )
    if code in PROP_CODES:
        return EditorState(state.x, state.y, state.scale, state.rotation, PROP_CODES[code])
    return state  # settle / unknown -> no-op


def simulate(state: EditorState, program: list[int], geom: EditorGeometry) -> EditorState:
    """Predict the object's final attributes after running the whole program (one run)."""
    for code in program:
        state = apply_code(state, code, geom)
    return state


def attribute_distance(state: EditorState, target: EditorState) -> int:
    """A non-negative distance-to-target (0 == win) — the PLANNING GRADIENT the frames withhold.
    Counts the remaining moves/steps along each attribute: position in STEP units, scale in unit
    deltas, rotation/property as a 0/1 mismatch."""
    return (
        abs(state.x - target.x) // STEP
        + abs(state.y - target.y) // STEP
        + abs(state.scale - target.scale)
        + int(state.rotation != target.rotation)
        + int(state.prop != target.prop)
    )


def predict_win(
    state: EditorState, target: EditorState, program: list[int], geom: EditorGeometry
) -> bool:
    """The model's prediction of whether `program` wins (final attributes match the target)."""
    return simulate(state, program, geom).matches(target)


def plan_program(
    state: EditorState,
    target: EditorState,
    geom: EditorGeometry,
    n_slots: int,
    alphabet: Optional[list[int]] = None,
    *,
    max_expansions: int = 20000,
) -> Optional[list[int]]:
    """MODEL-GUIDED program search: best-first over programs ranked by predicted attribute_distance —
    the gradient that turns blind program-space search into directed search. Returns the first program
    (length n_slots) the model predicts wins, or None within the expansion budget. The caller validates
    the winner against the real env (the oracle)."""
    import heapq
    import itertools

    alphabet = (
        alphabet
        if alphabet is not None
        else ([SETTLE] + list(MOVE_CODES) + list(ROT_CODES) + list(SCALE_CODES) + list(PROP_CODES))
    )
    counter = itertools.count()  # unique tie-breaker (states/lists aren't orderable)
    heap = [(attribute_distance(state, target), next(counter), 0, [], state)]
    seen = {(state, 0)}
    expansions = 0
    while heap and expansions < max_expansions:
        _, _, depth, prog, st = heapq.heappop(heap)
        expansions += 1
        if st.matches(target):
            return prog + [SETTLE] * (n_slots - len(prog))
        if depth >= n_slots:
            continue
        for code in alphabet:
            ns = apply_code(st, code, geom)
            key = (ns, depth + 1)
            if key in seen:
                continue
            seen.add(key)
            heapq.heappush(
                heap,
                (
                    attribute_distance(ns, target) + depth + 1,
                    next(counter),
                    depth + 1,
                    prog + [code],
                    ns,
                ),
            )
    return None
