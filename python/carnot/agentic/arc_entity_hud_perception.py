#!/usr/bin/env python3
"""Entity + HUD perception detectors for the live ARC agent (REQ-ARC-WMTE-5833).

WHY THIS EXISTS -- the evidence chain that mandated it.
    - REQ-ARC-WMTE-5831 (source-grounded diagnosis): across 8 public games, a gemma-4-31B agent with the
      full leaderboard-winner recipe NEVER hypothesized the true win condition. Root cause 5/8 = PERCEPTION:
      the object/segmentation front-end fixated on the every-action HUD/budget-bar (a move counter, a mana
      meter, a lose-timer) and never even REPRESENTED the load-bearing entities -- the player token, the
      exit tile, the target blocks.
    - REQ-ARC-WMTE-5832 (oracle-perception counterfactual): when we HANDED the same model the correct
      entities (entity presence/role only, never the goal), goal induction flipped from 0/8 correct to 7/8.
      The goal-hypothesis generator is NOT the bottleneck; upstream perception is. bp35 further showed BOTH
      halves matter: adding the player/gem entities was not enough -- you also had to identify the row-63 bar
      as a counter to be ignored.

So the perception fix is two detectors, and CRUCIALLY both must run off the agent's OWN observed transitions
(a hidden-game agent may not read source and gets no oracle) -- oracle-distinct, sovereignty-safe:

    1. detect_hud_registers -- find edge bands whose non-background cell count changes MONOTONICALLY on most
       frame-changing actions (a fill/deplete counter), independent of where/what you acted. That monotone,
       act-on-everything signature is what distinguishes a status counter from the interactive board (whose
       changes are localized to the click target or a small moving component and oscillate, not accumulate).
    2. detect_mover -- find the color whose blob CENTROID translates in the DIRECTION of directional actions
       (1=up 2=down 3=left 4=right). A component that consistently slides with your move keys is the player.

perceive_entities composes them: segment the current frame, drop/label the HUD bands, label the mover as the
PLAYER, and emit a corrected object view -- the detector-produced analogue of REQ-ARC-WMTE-5832's hand-authored
oracle facts. The build gate (separate script) re-runs the 5832 goal measurement with THIS output substituted
for the hand-authored facts, to confirm it recovers the ~7/8 goal-correctness from frames alone.

verifier_is_oracle: N/A (perception, not a verifier). All signals come from the agent's own (frame, action,
frame) transitions; nothing here reads game source or a win predicate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from carnot.agentic.arc_color_blob_salience import ColorBlob, _as_grid, connected_color_blobs

# Action -> unit (drow, dcol). Matches the agent's convention (arc_greedy_direct_agent prompts):
# 1=up 2=down 3=left 4=right. 6=click carries (x=col, y=row); 5/7 are non-spatial and ignored here.
_DIR: dict[int, tuple[int, int]] = {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}


@dataclass(frozen=True)
class Transition:
    """One observed (before, action, after) the LIVE agent already records. `before`/`after` are HxW
    logical grids (or anything `_as_grid` accepts). `x`/`y` are the click COLUMN/ROW for action 6."""

    before: Any
    action: int
    after: Any
    x: Optional[int] = None
    y: Optional[int] = None


@dataclass(frozen=True)
class HudBand:
    """A detected status/counter band to be excluded from goal-relevant perception."""

    axis: str  # "row" | "col"
    index: int
    direction: str  # "fill" (count rises) | "deplete" (count falls)
    changed_fraction: float
    monotone_ratio: float


@dataclass(frozen=True)
class Mover:
    """The detected player entity: the color that translates with directional actions."""

    color: int
    alignment: float  # mean directional alignment in [-1, 1]; ~1 = moves exactly with the keys
    evidence: int  # number of directional transitions supporting it


@dataclass
class PerceptionResult:
    text: str
    objects: list[dict] = field(default_factory=list)
    hud_bands: list[HudBand] = field(default_factory=list)
    mover: Optional[Mover] = None


def _background_color(grid: np.ndarray) -> int:
    """The modal color -- the large uniform fill the segmentation already suppresses."""
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[int(np.argmax(counts))])


def _frame_changed(before: np.ndarray, after: np.ndarray) -> bool:
    return before.shape == after.shape and bool(np.any(before != after))


def _band_cells(grid: np.ndarray, axis: str, index: int) -> np.ndarray:
    return grid[index, :] if axis == "row" else grid[:, index]


def detect_hud_registers(
    transitions: list[Transition],
    *,
    edge_margin: int = 2,
    changed_fraction_min: float = 0.85,
    pos_independent_min: float = 0.7,
    mixed_changed_min: float = 0.5,
    min_frame_changes: int = 3,
) -> list[HudBand]:
    """Find edge rows/cols that behave like a status counter: they change on NEARLY EVERY action, and the
    change is DECOUPLED from where/what you acted. That is the signature of a move counter / mana meter /
    lose-timer bar -- the thing the winner recipe mistook for the game board on 5/8 games.

    The discriminating statistic is NOT the non-background cell COUNT: real ARC HUD counters change cell
    VALUES (a value-6 step-bar depletes, cells recolor to 15) while the non-background count stays flat, so
    a count-monotonicity test misses them entirely (measured on tu93/bp35: row63 changed on 100% of actions
    yet net count change was 0). Instead:
      - `changed_fraction` = share of frame-changing transitions in which the band's VALUES changed at all.
        A counter advances on nearly every action; a board region only changes when you act on/near it.
      - `pos_independent` = among CLICK actions whose click cell is OUTSIDE the band, the fraction where the
        band changed anyway. A counter changes no matter where you click (position-independent); a board
        region changes only where clicked. This rescues a counter that only advances on ~half the actions
        (e.g. a meter that ticks on moves but not clicks) from being confused with a busy board column.
    A band is flagged if it changes on nearly every action, OR it changes on at least half the actions AND
    is strongly position-independent. `monotone_ratio` (on non-bg count) is still reported best-effort to
    label fill vs deplete when the count does move.
    """
    changing = [t for t in transitions if _frame_changed(_as_grid(t.before), _as_grid(t.after))]
    if len(changing) < int(min_frame_changes):
        return []
    g0 = _as_grid(changing[0].before)
    h, w = g0.shape
    bg = _background_color(g0)

    candidates: list[tuple[str, int]] = []
    for r in range(h):
        if r < edge_margin or r >= h - edge_margin:
            candidates.append(("row", r))
    for c in range(w):
        if c < edge_margin or c >= w - edge_margin:
            candidates.append(("col", c))

    bands: list[HudBand] = []
    for axis, index in candidates:
        deltas: list[int] = []
        n_changed = 0
        click_outside = 0
        click_outside_changed = 0
        for t in changing:
            before = _as_grid(t.before)
            after = _as_grid(t.after)
            if before.shape != after.shape or index >= (before.shape[0] if axis == "row" else before.shape[1]):
                continue
            b = _band_cells(before, axis, index)
            a = _band_cells(after, axis, index)
            band_changed = not np.array_equal(b, a)
            if band_changed:
                n_changed += 1
                deltas.append(int((a != bg).sum()) - int((b != bg).sum()))
            # position-independence probe: a click whose target is NOT in this band
            if int(t.action) == 6 and t.x is not None and t.y is not None:
                click_in_band = (axis == "row" and int(t.y) == index) or (
                    axis == "col" and int(t.x) == index
                )
                if not click_in_band:
                    click_outside += 1
                    if band_changed:
                        click_outside_changed += 1
        if n_changed < int(min_frame_changes):
            continue
        changed_fraction = n_changed / len(changing)
        pos_independent = (click_outside_changed / click_outside) if click_outside > 0 else None
        is_hud = changed_fraction >= changed_fraction_min or (
            pos_independent is not None
            and pos_independent >= pos_independent_min
            and changed_fraction >= mixed_changed_min
        )
        if not is_hud:
            continue
        abs_sum = sum(abs(d) for d in deltas)
        net = sum(deltas)
        monotone_ratio = (abs(net) / abs_sum) if abs_sum > 0 else 0.0
        direction = "fill" if net > 0 else ("deplete" if net < 0 else "counter")
        bands.append(
            HudBand(
                axis=axis,
                index=index,
                direction=direction,
                changed_fraction=round(changed_fraction, 3),
                monotone_ratio=round(monotone_ratio, 3),
            )
        )
    return bands


def _color_centroid(grid: np.ndarray, color: int) -> Optional[tuple[float, float]]:
    ys, xs = np.nonzero(grid == color)
    if ys.size == 0:
        return None
    return (float(ys.mean()), float(xs.mean()))


def detect_mover(
    transitions: list[Transition],
    *,
    min_evidence: int = 2,
    align_min: float = 0.5,
    max_shift: float = 4.0,
    max_area_fraction: float = 0.15,
) -> Optional[Mover]:
    """Find the PLAYER: the color whose centroid consistently translates in the direction of the
    directional action (1-4) used. For each directional transition and each non-background color present
    in both frames, we score how well the color's centroid shift aligns with the action's unit vector
    (dot product of the unit shift with the unit direction), requiring a small, real shift (a rigid
    slide of ~1 cell, not a whole-board recolor). The color with the strongest, best-supported alignment
    is the player.

    A player is a SMALL, COMPACT entity, so a color occupying more than `max_area_fraction` of the grid is
    excluded -- this is what stops a large VOID/background region (not the modal color, so not caught by the
    background exclusion) from being mistaken for the player. Measured need: cn04's color-0 void was
    selected as the mover with alignment 1.0 before this guard, because its centroid drifts as the board
    changes even though it is not an entity."""
    scores: dict[int, list[float]] = {}
    for t in transitions:
        d = _DIR.get(int(t.action))
        if d is None:
            continue
        before = _as_grid(t.before)
        after = _as_grid(t.after)
        if before.shape != after.shape or not _frame_changed(before, after):
            continue
        bg = _background_color(before)
        area = before.size
        dir_vec = np.asarray(d, dtype=float)
        dir_unit = dir_vec / np.linalg.norm(dir_vec)
        for color in np.unique(before):
            color = int(color)
            if color == bg:
                continue
            if int((before == color).sum()) > max_area_fraction * area:
                continue  # too large to be a compact player entity (a void/background-like region)
            cb = _color_centroid(before, color)
            ca = _color_centroid(after, color)
            if cb is None or ca is None:
                continue
            shift = np.asarray(ca) - np.asarray(cb)
            mag = float(np.linalg.norm(shift))
            if mag < 0.3 or mag > max_shift:
                continue
            align = float(np.dot(shift / mag, dir_unit))
            scores.setdefault(color, []).append(align)
    best: Optional[Mover] = None
    for color, aligns in scores.items():
        positive = [a for a in aligns if a > 0]
        if len(positive) < int(min_evidence):
            continue
        mean_align = float(np.mean(aligns))
        if mean_align < align_min:
            continue
        if best is None or (mean_align, len(positive)) > (best.alignment, best.evidence):
            best = Mover(color=color, alignment=round(mean_align, 3), evidence=len(positive))
    return best


def _blob_in_bands(blob: ColorBlob, hud_rows: set[int], hud_cols: set[int]) -> bool:
    """A blob belongs to the HUD only if EVERY one of its cells lies in a HUD row or column -- so a game
    object that merely clips a HUD edge is not misclassified as status."""
    return all((r in hud_rows) or (c in hud_cols) for (r, c) in blob.cells)


def perceive_entities(
    current_frame: Any,
    transitions: list[Transition],
    *,
    max_objects: int = 12,
) -> PerceptionResult:
    """Produce the corrected object view: segment the current frame, drop/label detected HUD bands, label
    the detected mover as the PLAYER, and emit a natural-language perception string usable as the goal
    prompt's perception block. This is the detector-produced analogue of REQ-ARC-WMTE-5832's hand-authored
    oracle facts -- entity presence/role ONLY, never the goal."""
    grid = _as_grid(current_frame)
    hud_bands = detect_hud_registers(transitions)
    mover = detect_mover(transitions)
    hud_rows = {b.index for b in hud_bands if b.axis == "row"}
    hud_cols = {b.index for b in hud_bands if b.axis == "col"}

    blobs = connected_color_blobs(grid)
    objects: list[dict] = []
    player_seen = False
    for i, b in enumerate(blobs):
        is_hud = _blob_in_bands(b, hud_rows, hud_cols)
        is_player = mover is not None and int(b.color) == mover.color and not is_hud
        if is_player:
            player_seen = True
        objects.append(
            {
                "id": i,
                "color": int(b.color),
                "row": int(round(b.centroid[0])),
                "col": int(round(b.centroid[1])),
                "size": int(b.pixel_count),
                "role": "player" if is_player else ("status_bar" if is_hud else "object"),
            }
        )

    # Render: player first (if any), then non-HUD objects, then the HUD bands as an explicit "ignore" note.
    lines: list[str] = []
    if mover is not None and player_seen:
        p = next(o for o in objects if o["role"] == "player")
        lines.append(
            f"PLAYER token (color {p['color']}) at row={p['row']} col={p['col']} -- it MOVES when you use "
            f"actions 1/2/3/4."
        )
    elif mover is not None:
        lines.append(
            f"A PLAYER entity (color {mover.color}) moves with actions 1/2/3/4 (not visible as a discrete "
            f"object in this frame)."
        )
    game_objs = [o for o in objects if o["role"] == "object"][:max_objects]
    if game_objs:
        lines.append("Other distinct objects (candidate targets):")
        for o in game_objs:
            lines.append(f"  #{o['id']} color={o['color']} at row={o['row']} col={o['col']} size={o['size']}")
    if hud_bands:
        desc = ", ".join(f"{b.axis} {b.index} ({b.direction} counter)" for b in hud_bands)
        lines.append(
            f"STATUS/COUNTER bands to IGNORE for the goal (they change on every action; not the game "
            f"board): {desc}."
        )
    if not lines:
        lines.append("(no entities or HUD bands detected yet -- explore more to gather transitions)")
    return PerceptionResult(
        text="\n".join(lines), objects=objects, hud_bands=hud_bands, mover=mover
    )
