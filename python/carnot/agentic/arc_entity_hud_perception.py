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

import re
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


def _bounded_color_centroid(
    grid: np.ndarray, color: int, *, max_area_fraction: float
) -> Optional[tuple[float, float]]:
    """Centroid of a color's cells if the color occupies at most `max_area_fraction` of the grid (an
    entity, not a background-like fill). Global centroid -- not a rigid same-color blob match -- so a
    MORPHING avatar (ls20 changes color as it moves) stays trackable on the transitions where the color is
    present in both frames."""
    ys, xs = np.nonzero(grid == color)
    n = int(ys.size)
    if n == 0 or n > max_area_fraction * grid.size:
        return None
    return (float(ys.mean()), float(xs.mean()))


def detect_mover(
    transitions: list[Transition],
    *,
    min_evidence: int = 4,
    align_min: float = 0.7,
    max_shift_frac: float = 0.25,
    max_area_fraction: float = 0.15,
) -> Optional[Mover]:
    """Find the PLAYER: the color whose centroid consistently translates in the direction of the directional
    action (1-4) used. For each directional transition and each entity-sized non-background color present in
    both frames, we score how well its centroid shift aligns with the action's unit vector (dot of the unit
    shift with the unit direction), requiring a real shift up to a bounded distance (a rigid slide, not a
    whole-board recolor). The color with the strongest, best-supported alignment is the player.

    Robustness comes from the SIGNAL STRENGTH bar, not a compactness heuristic: a real avatar slides with the
    keys on nearly every directional action (measured align ~0.9-1.0, evidence 24-43 across sc25/ls20),
    whereas a scattered non-entity color (cn04's color-0 void) only drifts incidentally (align ~0.55, low
    evidence). Requiring `align_min`>=0.7 over `min_evidence`>=4 directional transitions keeps the avatars
    and drops the void, without a density heuristic that mis-rejected real avatars whose color is shared with
    other cells. Global centroid keeps a morphing avatar trackable.

    `max_shift` is RELATIVE to the grid (`max_shift_frac * min(H, W)`, floored at 4.0): a player can jump
    MULTIPLE cells per action on a coarse logical grid (tu93's token jumps ~6 cells/action on a 64-grid), so
    a fixed 4.0-cell cap wrongly rejected such players and made them undetectable (the REQ-ARC-WMTE-5836
    isolation found tu93 -- the one clean navigation game -- was skipped for exactly this reason). The
    relative cap still rejects a near-whole-board recolor (centroid shift ~half the grid)."""
    scores: dict[int, list[float]] = {}
    for t in transitions:
        d = _DIR.get(int(t.action))
        if d is None:
            continue
        before = _as_grid(t.before)
        after = _as_grid(t.after)
        if before.shape != after.shape or not _frame_changed(before, after):
            continue
        max_shift = max(4.0, max_shift_frac * float(min(before.shape)))
        bg = _background_color(before)
        dir_vec = np.asarray(d, dtype=float)
        dir_unit = dir_vec / np.linalg.norm(dir_vec)
        for color in np.unique(before):
            color = int(color)
            if color == bg:
                continue
            cb = _bounded_color_centroid(before, color, max_area_fraction=max_area_fraction)
            ca = _bounded_color_centroid(after, color, max_area_fraction=max_area_fraction)
            if cb is None or ca is None:
                continue
            shift = np.asarray(ca) - np.asarray(cb)
            mag = float(np.linalg.norm(shift))
            if mag < 0.3 or mag > max_shift:
                continue
            scores.setdefault(color, []).append(float(np.dot(shift / mag, dir_unit)))
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


# --- Re-authoring (REQ-ARC-WMTE-5834 fix) ---------------------------------------------------------------
# The end-to-end gate (REQ-ARC-WMTE-5834) found that emitting detector perception ALONGSIDE the model's own
# WRONG learned rules does NOT flip goal induction (0 correct, 2 partial vs the oracle's 7/8): the model
# follows a rule it already believes ("action 6 changes (63,x) to 15") even when the detector correctly flags
# that band as a counter to ignore. The oracle got 7/8 because it REPLACED the framing. So re-authoring must
# (a) RETRACT any learned rule that references a detected HUD band, (b) NAME the mover's nearest object as the
# candidate target, and (c) state that the goal involves the player + a target, NOT the counter.

_CHANGE_WORDS = ("change", "set", "fill", "toggle", "clear", "recolor", "turn", "->", "consume", "deplete")


def _line_references_hud(line: str, bands: list[HudBand]) -> bool:
    """True if a rule LINE is about a detected HUD band: it describes a cell change AND references the band's
    index as a coordinate (a `(R, ...)`/`(..., R)` tuple) or via a `row R`/`col R` keyword. This is what
    lets us retract exactly the HUD-fixation rules (bp35 'changes (63,x) to 15', lf52 'change cells in row 0')
    without touching genuine game rules."""
    low = line.lower()
    if not any(w in low for w in _CHANGE_WORDS):
        return False
    nums = {int(n) for n in re.findall(r"\d+", line)}
    for b in bands:
        if re.search(rf"\b(?:row|column|col)\s*{b.index}\b", low):
            return True
        if b.index in nums and re.search(
            rf"\(\s*{b.index}\s*,|,\s*{b.index}\s*\)", line
        ):
            return True
    return False


def _candidate_targets(percept: PerceptionResult, *, k: int = 2) -> tuple[list[dict], Optional[dict]]:
    """The player object (if any) and the k non-HUD objects nearest to it (or the k largest if no player) --
    the detector's best guess at what the player should be routed toward."""
    player = next((o for o in percept.objects if o.get("role") == "player"), None)
    objs = [o for o in percept.objects if o.get("role") == "object"]
    if player is not None and objs:
        px, py = player["col"], player["row"]
        objs = sorted(objs, key=lambda o: (o["col"] - px) ** 2 + (o["row"] - py) ** 2)
    else:
        objs = sorted(objs, key=lambda o: -int(o.get("size", 0)))
    return objs[:k], player


def reauthor_framing(rules_text: str, percept: PerceptionResult) -> tuple[str, str]:
    """Re-author the model's framing from the detector output (REQ-ARC-WMTE-5834 fix). Returns
    (corrected_rules, perception_block): the rules with HUD-band-referencing lines REMOVED, and a strong
    perception block that OVERRIDES the counter-fixation -- naming the player, its candidate target(s), the
    counters to ignore, and the retracted rules. Feed corrected_rules as the goal prompt's RULES and
    perception_block as its PERCEPTION so the correction replaces, not augments, the wrong framing."""
    bands = percept.hud_bands
    kept: list[str] = []
    retracted: list[str] = []
    for line in rules_text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if bands and _line_references_hud(stripped, bands):
            retracted.append(stripped)
        else:
            kept.append(stripped)
    corrected_rules = "\n".join(kept) if kept else "(no reliable rules yet)"

    targets, player = _candidate_targets(percept)
    lines = ["PERCEPTION (corrected -- this OVERRIDES any rule about changing counter cells):"]
    if player is not None:
        lines.append(
            f"- PLAYER: color {player['color']} at row {player['row']} col {player['col']}; "
            f"it MOVES with actions 1/2/3/4."
        )
    elif percept.mover is not None:
        lines.append(f"- A PLAYER entity (color {percept.mover.color}) moves with actions 1/2/3/4.")
    for o in targets:
        lines.append(
            f"- candidate TARGET: object color {o['color']} at row {o['row']} col {o['col']} "
            f"(the goal likely requires the player to reach/act on it)."
        )
    if bands:
        desc = ", ".join(f"{b.axis} {b.index}" for b in bands)
        lines.append(
            f"- IGNORE these STATUS COUNTERS ({desc}): they advance on EVERY action and are a score/"
            f"budget readout, NOT the game board. Filling or changing them is NOT the objective."
        )
    if retracted:
        lines.append(
            "- DISREGARD these earlier notes -- they describe a counter advancing as a side-effect, not the "
            "objective: " + " | ".join(retracted)
        )
    # Closing directive. When a player/mover WAS detected, push the navigate frame (it is almost certainly a
    # move-the-player game). When NONE was detected, do NOT force navigation -- that OVERCORRECTED the
    # non-navigate games (lf52 block-merge, ft09 hidden-CSP) into a wrong navigate frame. Instead offer the
    # options so the model infers from the objects: move-to-target, arrange/match/merge, or satisfy a pattern.
    if player is not None or percept.mover is not None:
        lines.append(
            "- The GOAL almost certainly involves moving the PLAYER onto one of the TARGET objects above, "
            "NOT filling/changing a counter band. (If the objects clearly need arranging/matching instead, "
            "say that.)"
        )
    else:
        lines.append(
            "- The GOAL involves the GAME OBJECTS above, NOT the counter band. Infer which fits the objects "
            "and rules: (a) move an object onto a matching target, (b) arrange/match/merge similar objects "
            "until a target configuration is reached, or (c) satisfy a pattern/constraint over the cells. "
            "State the single most likely goal."
        )
    return corrected_rules, "\n".join(lines)
