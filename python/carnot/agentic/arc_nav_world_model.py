"""Auto-fitting grid->grid world model for the ARC-AGI-3 4-direction NAVIGATION family.

Why this exists. The Executable-World-Models lever (arXiv:2605.05138) deepens by inducing a
`transition + goal` model once and planning IN IMAGINATION. The program-generalization first swing
(docs/research-notes/program-generalization-first-swing-2026-06-22.md) showed the lever WORKS at L1 with a
HAND-written tu93 model, but that a model frozen at L1 does not deepen when the next level's MECHANIC
shifts -- the model must be RE-INDUCED per mechanic (the precise cost the leader pays). Re-induction needs
the model to be FITTED FROM TRANSITIONS automatically, not hand-written. This module is that auto-inducer:
`InducedNavWorldModel.fit(transitions)` learns, purely from observed `(grid, action, next_grid)` data:

  * per-action DISPLACEMENT (which way + how far the avatar translates for each keyboard action),
  * the AVATAR object (the small, constant-count, co-translating colour set),
  * the FLOOR colour the avatar leaves behind when it moves,
  * the swept-path WALL colours that BLOCK a move, and
  * the GOAL colour whose coverage completes the level.

It then exposes `engine(grid, action, data)->grid` + `is_level_complete(grid)->bool`, the exact interface
`carnot.agentic.arc_executable_world_model.plan_in_model` consumes -- so a re-induced model drops straight
into imagination planning. Re-induction = call `fit` again on transitions collected at the new level.

Scope + honesty. This is a GRID->GRID model: it can only capture mechanics whose outcome is a deterministic
function of the VISIBLE grid. Where a level adds HIDDEN state (a rotation phase / sprite buffer not in the
rendered grid), re-induction at this resolution is necessarily incomplete -- and that incompleteness is a
correctly-attributed finding (state-augmentation needed), not a bug in the fitter. The fitter reports a
`fit_quality` so the caller can tell a clean fit from a noisy/hidden-state one.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

# Actions 1-4 are the canonical ARC keyboard directional set; 5/6 (confirm/click) are not nav moves.
_NAV_ACTIONS = (1, 2, 3, 4)


def _bg_color(grids: list[np.ndarray]) -> int:
    """Background = the globally most common colour across the sampled grids."""
    c: Counter = Counter()
    for g in grids:
        c.update(int(v) for v in np.asarray(g).flatten().tolist())
    return c.most_common(1)[0][0]


def _color_cells(grid: np.ndarray, colors) -> np.ndarray:
    g = np.asarray(grid)
    mask = np.zeros(g.shape, dtype=bool)
    for c in colors:
        mask |= g == c
    return mask


def _bbox(mask: np.ndarray):
    ys, xs = np.where(mask)
    if ys.size == 0:
        return None
    return int(ys.min()), int(xs.min()), int(ys.max()), int(xs.max())


@dataclass
class InducedNavWorldModel:
    """A nav world model whose parameters are FITTED from transitions (see module docstring)."""

    displacement: dict  # action -> (dy, dx)
    avatar_colors: frozenset  # colours composing the moving avatar
    bg_color: int  # background colour
    floor_color: int  # colour left behind when the avatar leaves a cell
    wall_colors: frozenset  # colours that BLOCK a move when present in the swept mid-gap
    goal_color: Optional[
        int
    ]  # colour whose coverage by the avatar completes the level (None if unknown)
    door_color: Optional[int] = (
        None  # the passable colour the avatar moves THROUGH on a successful move
    )
    fit_quality: dict = field(
        default_factory=dict
    )  # diagnostics: n_moves fitted, displacement agreement, etc.

    # ---- fitting -------------------------------------------------------------------------------------

    @classmethod
    def fit(cls, transitions) -> "InducedNavWorldModel":
        """Fit from an iterable of objects with .grid/.action/.next_grid (+ optional .level_before/
        .level_after), or (grid, action, next_grid) / (grid, action, next_grid, lb, la) tuples."""
        rows = [_norm(t) for t in transitions]
        grids = [r[0] for r in rows] + [r[2] for r in rows]
        bg = _bg_color(grids)

        # 1) AVATAR colours by CO-TRANSLATION (robust + seed-stable). The avatar is a rigid object: its
        #    component colours (e.g. tu93's colour-9 body + colour-4 centre) ALWAYS shift TOGETHER by the
        #    SAME vector. A stationary goal marker never shifts; walls/background are not count-conserved.
        #    So: per transition, record each small count-conserved colour's centroid shift, then group
        #    colours that repeatedly share a common NONZERO shift. The avatar = the anchor (most-shifting
        #    colour) + everything that co-shifts with it.
        per_color_shift_obs: dict = {}  # color -> list of (transition_idx, (dy,dx)) for nonzero shifts
        n_shift: Counter = Counter()
        for ti, (g0, a, g1, *_rest) in enumerate(rows):
            g0a = np.asarray(g0)
            g1a = np.asarray(g1)
            if np.array_equal(g0a, g1a):
                continue
            colors = set(int(v) for v in g0a.flatten().tolist()) | set(
                int(v) for v in g1a.flatten().tolist()
            )
            for col in colors:
                if col == bg:
                    continue
                m0 = g0a == col
                m1 = g1a == col
                n0 = int(m0.sum())
                n1 = int(m1.sum())
                if n0 == 0 or n1 == 0 or n0 != n1 or n0 > 40:
                    continue
                c0 = (np.where(m0)[0].mean(), np.where(m0)[1].mean())
                c1 = (np.where(m1)[0].mean(), np.where(m1)[1].mean())
                shift = (round(float(c1[0] - c0[0])), round(float(c1[1] - c0[1])))
                if shift != (0, 0):
                    per_color_shift_obs.setdefault(col, []).append((ti, shift))
                    n_shift[col] += 1
        avatar_colors: set = set()
        if n_shift:
            anchor = n_shift.most_common(1)[0][0]
            anchor_by_t = {ti: sh for ti, sh in per_color_shift_obs[anchor]}
            avatar_colors = {anchor}
            for col, obs in per_color_shift_obs.items():
                if col == anchor:
                    continue
                # co-shifts with the anchor (same vector, same transition) in >= 2 transitions?
                agree = sum(1 for ti, sh in obs if anchor_by_t.get(ti) == sh)
                if agree >= 2:
                    avatar_colors.add(col)

        # 2) DISPLACEMENT per action: median avatar-bbox translation over that action's move transitions.
        disp: dict = {}
        move_hits = Counter()
        move_total = Counter()
        for g0, a, g1, *_ in rows:
            if a not in _NAV_ACTIONS:
                continue
            b0 = _bbox(_color_cells(g0, avatar_colors))
            b1 = _bbox(_color_cells(g1, avatar_colors))
            if b0 is None or b1 is None:
                continue
            move_total[a] += 1
            dy, dx = b1[0] - b0[0], b1[1] - b0[1]
            if (dy, dx) != (0, 0):
                move_hits[a] += 1
                disp.setdefault(a, []).append((dy, dx))
        displacement = {a: Counter(v).most_common(1)[0][0] for a, v in disp.items()}

        # 3) FLOOR colour: what a vacated avatar cell becomes. Over move transitions, the cells that were
        #    avatar in g0 but avatar-free in g1 take the floor colour; pick the mode.
        floor_votes: Counter = Counter()
        for g0, a, g1, *_ in rows:
            g0 = np.asarray(g0)
            g1 = np.asarray(g1)
            av0 = _color_cells(g0, avatar_colors)
            av1 = _color_cells(g1, avatar_colors)
            vacated = av0 & ~av1
            for v in g1[vacated].flatten().tolist():
                floor_votes[int(v)] += 1
        floor_color = floor_votes.most_common(1)[0][0] if floor_votes else bg

        # 4) WALL colours: by DISCRIMINATION, not strict set-difference (a few successful moves may catch a
        #    wall corner in the gap window, so "never in a moved gap" is too brittle). A wall colour is one
        #    that appears in the swept mid-gap of BLOCKED moves far more often than in MOVED moves.
        moved_gap: Counter = Counter()
        blocked_gap: Counter = Counter()
        n_moved = 0
        n_blocked = 0
        for g0, a, g1, *_ in rows:
            if a not in displacement:
                continue
            dy, dx = displacement[a]
            b0 = _bbox(_color_cells(g0, avatar_colors))
            if b0 is None:
                continue
            r, c = b0[0], b0[1]
            h = b0[2] - b0[0] + 1
            w = b0[3] - b0[1] + 1
            my, mx = r + dy // 2, c + dx // 2
            gap = np.asarray(g0)[max(0, my) : my + h, max(0, mx) : mx + w]
            gapset = set(int(v) for v in gap.flatten().tolist())
            b1 = _bbox(_color_cells(g1, avatar_colors))
            moved = b1 is not None and (b1[0] - b0[0], b1[1] - b0[1]) != (0, 0)
            if moved:
                n_moved += 1
                moved_gap.update(gapset)
            else:
                n_blocked += 1
                blocked_gap.update(gapset)
        wall_colors = set()
        for col in set(blocked_gap) | set(moved_gap):
            # NB: do NOT exclude the background colour -- in many mazes the wall IS the background colour
            # (e.g. tu93: colour 5 is both). The discrimination test below already rejects colours that
            # appear equally in moved gaps, so a true-background non-wall colour won't be selected.
            if col in avatar_colors:
                continue
            p_block = blocked_gap[col] / n_blocked if n_blocked else 0.0
            p_move = moved_gap[col] / n_moved if n_moved else 0.0
            if (
                p_block - p_move > 0.5
            ):  # strongly predicts a blocked move when present in the swept gap
                wall_colors.add(col)
        # DOOR colour: the passable colour the avatar moves THROUGH on a successful move (the dominant
        # moved-gap colour that is not a wall / bg / floor / avatar). Captured so hazard fitting can EXCLUDE
        # it -- doors are everywhere, and a naive hazard detector would otherwise flag them as lethal. A real
        # door is UBIQUITOUS (present in the swept gap of a MAJORITY of successful moves); requiring that
        # avoids mis-picking a colour merely caught in a few gaps (e.g. a charger the avatar passed near).
        door_color = None
        for col, cnt in moved_gap.most_common():
            if (
                col not in wall_colors
                and col != bg
                and col != floor_color
                and col not in avatar_colors
            ):
                if n_moved and cnt >= 0.5 * n_moved:
                    door_color = col
                break

        # 5) GOAL colour: prefer the colour the avatar COVERS at a level-up; else a small, STATIONARY,
        #    distinctive colour (not bg/floor/avatar/wall) -- the classic reach-the-marker goal.
        goal_color = None
        for g0, a, g1, lb, la in rows:
            if la is not None and lb is not None and la > lb:
                b0 = _bbox(_color_cells(g0, avatar_colors))
                if b0 and a in displacement:
                    dy, dx = displacement[a]
                    r, c = b0[0] + dy, b0[1] + dx
                    h = b0[2] - b0[0] + 1
                    w = b0[3] - b0[1] + 1
                    dest = np.asarray(g0)[max(0, r) : r + h, max(0, c) : c + w]
                    cand = [
                        int(v)
                        for v in dest.flatten().tolist()
                        if int(v) not in (bg, floor_color)
                        and int(v) not in avatar_colors
                        and int(v) not in wall_colors
                    ]
                    if cand:
                        goal_color = Counter(cand).most_common(1)[0][0]
                        break
        if goal_color is None:
            # stationary-distinctive fallback: a non-bg/floor/avatar/wall colour that never translates
            stationary = {}
            for g0, a, g1, *_ in rows:
                for col in set(int(v) for v in np.asarray(g0).flatten().tolist()):
                    if col in (bg, floor_color) or col in avatar_colors or col in wall_colors:
                        continue
                    stationary[col] = stationary.get(col, 0) + 1
            if stationary:
                # smallest-footprint distinctive colour is the likeliest goal marker
                sizes = {}
                g0 = np.asarray(rows[0][0])
                for col in stationary:
                    sizes[col] = int((g0 == col).sum()) or 10**9
                goal_color = min(sizes, key=sizes.get)

        fit_quality = {
            "n_transitions": len(rows),
            "n_move_transitions": int(sum(move_hits.values())),
            "displacement_agreement": {
                a: (move_hits[a] / move_total[a]) if move_total[a] else None for a in _NAV_ACTIONS
            },
            "avatar_colors": sorted(avatar_colors),
            "wall_colors": sorted(wall_colors),
            "floor_color": floor_color,
            "goal_color": goal_color,
            "door_color": door_color,
            "displacement": {a: list(d) for a, d in displacement.items()},
        }
        return cls(
            displacement=displacement,
            avatar_colors=frozenset(avatar_colors),
            bg_color=bg,
            floor_color=floor_color,
            wall_colors=frozenset(wall_colors),
            goal_color=goal_color,
            door_color=door_color,
            fit_quality=fit_quality,
        )

    # ---- the induced model: engine + win predicate (plan_in_model interface) -------------------------

    def _avatar_bbox(self, grid):
        return _bbox(_color_cells(grid, self.avatar_colors))

    def engine(self, grid, action, data=None):
        """Predict the next logical grid for one action (pure grid->grid; hidden env state, if any, is
        outside this model's scope by construction)."""
        g = np.asarray(grid).copy()
        a = int(action)
        if a not in self.displacement:
            return g
        bb = self._avatar_bbox(g)
        if bb is None:
            return g
        r0, c0, r1, c1 = bb
        h, w = r1 - r0 + 1, c1 - c0 + 1
        dy, dx = self.displacement[a]
        nr, nc = r0 + dy, c0 + dx
        H, W = g.shape
        if nr < 0 or nc < 0 or nr + h > H or nc + w > W:
            return g
        # swept mid-gap blocking check
        my, mx = r0 + dy // 2, c0 + dx // 2
        gap = g[max(0, my) : my + h, max(0, mx) : mx + w]
        if self.wall_colors and np.any(np.isin(gap, list(self.wall_colors))):
            return g
        stamp = g[r0 : r1 + 1, c0 : c1 + 1].copy()
        g[r0 : r1 + 1, c0 : c1 + 1] = self.floor_color  # leave floor behind
        g[nr : nr + h, nc : nc + w] = stamp  # draw avatar at destination (covers goal if present)
        return g

    def is_level_complete(self, grid):
        """Level complete when the avatar has covered the goal colour (goal cells gone, avatar present)."""
        if self.goal_color is None:
            return False
        g = np.asarray(grid)
        if self._avatar_bbox(g) is None:
            return False
        return not bool(np.any(g == self.goal_color))

    # convenience: bound methods are picklable-free closures for plan_in_model
    def as_callables(self):
        return self.engine, self.is_level_complete

    def is_confident_nav(self, *, min_directions: int = 3) -> bool:
        """A HIGH-CONFIDENCE navigation fit -- used to gate the live inducer so it does NOT fire on
        non-navigation games where the fit is spurious (REQ-ARC-WMTE-5844). Two tells separate the one real
        nav game (tu93: avatar={9,4}, all 4 directions, goal=14) from the source-verified NON-nav games the
        fitter still fit a model for (sk48 = two-snake sequence-match, avatar={0,1}, only [1,2]; wa30 =
        Sokoban crate-push, avatar={0,14}, win reads CRATE positions not the avatar):

          1. The avatar must NOT include colour 0. Colour 0 is the PADDING/letterbox/void in the ARC render
             (confirmed in wa30 source, PADDING_COLOR=0); a genuine avatar sprite is non-padding, so an
             avatar that captured colour 0 has latched onto static frame-edge padding -> a corrupt fit.
          2. The avatar must translate in at least `min_directions` (default 3) of the 4 directional actions.
             A free-moving grid avatar moves in most directions; a fit that only saw <3 (sk48's orientation-
             locked snake, only up/down) is not a free 4-direction navigator.

        Plus a real goal colour. This is a CONSERVATIVE gate: a missed real nav game costs at most a
        forgone level-up (no harm), whereas firing on a non-nav game installs a plan that cannot win and
        wastes real actions (the wa30 A/B cost). tu93 passes (no 0, 4 directions, goal=14); sk48/wa30 fail.
        """
        if self.goal_color is None:
            return False
        if 0 in set(self.avatar_colors):
            return False
        return len({a for a in self.displacement if a in _NAV_ACTIONS}) >= int(min_directions)


def _blobs(mask):
    """Connected components (4-neighbour) of a boolean mask -> list of (centre_row, centre_col, size)."""
    H, W = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    out = []
    for i in range(H):
        for j in range(W):
            if mask[i, j] and not seen[i, j]:
                stack = [(i, j)]
                seen[i, j] = True
                cells = []
                while stack:
                    y, x = stack.pop()
                    cells.append((y, x))
                    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < H and 0 <= nx < W and mask[ny, nx] and not seen[ny, nx]:
                            seen[ny, nx] = True
                            stack.append((ny, nx))
                ys = [c[0] for c in cells]
                xs = [c[1] for c in cells]
                out.append((sum(ys) / len(ys), sum(xs) / len(xs), len(cells)))
    return out


@dataclass
class HazardAwareNavWorldModel(InducedNavWorldModel):
    """Nav world model EXTENDED to represent a charging-HAZARD object (e.g. tu93 L2's charging enemy).

    Why. The pure-nav model only translates/blocks, so it is structurally blind to a level mechanic that
    REMOVES the avatar -- it plans straight into the hazard (the tu93 L2 deepening wall, see
    docs/research-notes/mechanic-conditioned-reinduction-trigger-2026-06-22.md). This subclass LEARNS, from
    the death transitions in a level's data, a LINE-CHARGER hazard: an object (e.g. tu93 colour-8 block +
    colour-15 centre) that sits still until the avatar approaches ALONG a shared line (row or column) within
    a charge range, then charges to intercept and removes the avatar. `engine` predicts avatar-REMOVAL for a
    lethal move (yielding an avatar-less, dead-end grid), so `plan_in_model` routes AROUND the hazard to the
    goal -- exactly the safe detour the nav-only planner could not find.

    Scope/honesty: this is a grid-expressible hazard (the charger object is rendered + its lethality is a
    function of the visible avatar/hazard geometry; tu93 L2 is grid-deterministic, 0.0 nondeterminism). It
    does not model hazards whose lethality depends on non-rendered state.
    """

    hazard_colors: frozenset = frozenset()
    hazard_axis: Optional[str] = None  # 'row' (horizontal charger) or 'col' (vertical charger)
    charge_range: int = 0  # max post-move along-line distance at which the charge intercepts
    hazard_center_color: Optional[int] = (
        None  # the charger's centre-marker colour; its OFFSET within the
    )
    #                                            block gives the per-charger FACING (used by 'omni')
    align_tol: int = 2  # how close to the hazard's line counts as "on the line"
    lethal_mode: str = "toward"  # escalation rungs (the loop tries them in order, first that
    #   'toward' : charge on a TOWARD move along the single learned axis (tu93 L2's horizontal charger).
    #   'omni'   : OMNIDIRECTIONAL -- charge on a toward move along EITHER axis (row or col) + a collision
    #              (ending on the charger). Calibrated against tu93 L3's BFS ground truth: its 3 chargers
    #              kill on row-approach, col-approach, AND collision; a perpendicular step-ONTO the line is
    #              SAFE (so 'enter'/step_onto was wrong and is removed).
    hazard_fit: dict = field(default_factory=dict)

    @classmethod
    def fit(cls, transitions, goal_color=None, lethal_mode="toward") -> "HazardAwareNavWorldModel":
        """Fit nav params + a line-charger hazard. `goal_color` (if given) OVERRIDES the auto-detected goal
        -- the goal colour is level-invariant in reach-goal games, so when re-inducing at a deeper level
        (whose data may contain no level-up to anchor goal detection) the caller passes the L1 goal."""
        rows = [_norm(t) for t in transitions]
        base = InducedNavWorldModel.fit(rows)
        if goal_color is not None:
            base.goal_color = goal_color
            base.fit_quality["goal_color"] = goal_color
        structural = (
            {base.bg_color, base.floor_color, base.goal_color}
            | set(base.avatar_colors)
            | set(base.wall_colors)
        )
        if base.door_color is not None:
            structural.add(base.door_color)  # doors are passable + everywhere -> never a hazard
        # death transitions: avatar present in g0, ABSENT in g1, and NOT a level-up (those also clear the avatar)
        deaths = []
        for g0, a, g1, lb, la in rows:
            if la is not None and lb is not None and la > lb:
                continue
            if (
                _bbox(_color_cells(g0, base.avatar_colors)) is not None
                and _bbox(_color_cells(g1, base.avatar_colors)) is None
            ):
                deaths.append((np.asarray(g0), a, np.asarray(g1)))
        hz_color_votes: Counter = Counter()
        axis_votes: Counter = Counter()
        post_dists: list = []
        for g0, a, g1 in deaths:
            ab = _bbox(_color_cells(g0, base.avatar_colors))
            if ab is None or a not in base.displacement:
                continue
            dy, dx = base.displacement[a]
            acy, acx = (
                (ab[0] + ab[2]) / 2 + dy,
                (ab[1] + ab[3]) / 2 + dx,
            )  # avatar centre AFTER its nav move
            # The HAZARD is the object that CHARGED (moved) at the instant of death -- find the non-structural
            # colour whose compact blob TRANSLATED between g0 and g1 (the static door does NOT move; the
            # charging enemy does). This disambiguates the charger from passable decoration.
            best = None
            for col in set(int(v) for v in g0.flatten().tolist()):
                if col in structural:
                    continue
                n0 = int((g0 == col).sum())
                n1 = int((g1 == col).sum())
                if not (3 <= n0 <= 30) or n1 == 0:
                    continue
                b0 = _blobs(g0 == col)
                b1 = _blobs(g1 == col)
                if not b0 or not b1:
                    continue
                c0 = min(b0, key=lambda b: abs(b[0] - acy) + abs(b[1] - acx))
                c1 = min(b1, key=lambda b: abs(b[0] - c0[0]) + abs(b[1] - c0[1]))
                moved = abs(c0[0] - c1[0]) + abs(c0[1] - c1[1])
                if moved < 1:  # static (e.g. door) -> not the charger
                    continue
                d = abs(c0[0] - acy) + abs(c0[1] - acx)
                if best is None or d < best[0]:
                    best = (d, col, c0[0], c0[1])
            if best is None:
                continue
            _, col, by, bx = best
            hz_color_votes[col] += 1
            if abs(by - acy) <= abs(
                bx - acx
            ):  # avatar ends aligned on the hazard's ROW (perp offset small)
                axis_votes["row"] += 1
                post_dists.append(abs(bx - acx))
            else:
                axis_votes["col"] += 1
                post_dists.append(abs(by - acy))
        hazard_colors = frozenset(c for c, _ in hz_color_votes.most_common(2))
        # include the centre colour(s) co-located inside the hazard blob (tu93: 15 inside the 8-ring)
        if deaths and hazard_colors:
            g0 = deaths[0][0]
            hb = _bbox(_color_cells(g0, hazard_colors))
            if hb:
                inside = set(
                    int(v) for v in g0[hb[0] : hb[2] + 1, hb[1] : hb[3] + 1].flatten().tolist()
                )
                hazard_colors = frozenset(
                    set(hazard_colors)
                    | {c for c in inside if c not in structural and c != base.bg_color}
                )
        hazard_axis = axis_votes.most_common(1)[0][0] if axis_votes else None
        # CONSERVATIVE charge range: an INTERCEPTING charger reaches at least its own one-move step (the
        # enemy is a mirror of the avatar and moves the same step). The observed death distances are a lower
        # bound (a sparse sample can under-estimate the true reach by ~1 cell), so floor the range at the
        # avatar's move step -- under-estimating the range is fatal (the planner walks a "safe" move into a
        # charge), whereas a slight over-estimate only routes a little wider.
        step = max((abs(dy) + abs(dx) for (dy, dx) in base.displacement.values()), default=0)
        charge_range = max(
            int(round(max(post_dists))) if post_dists else 0, step if post_dists else 0
        )
        # The charger's CENTRE marker (the least-common hazard colour, e.g. tu93's colour-15 inside the
        # colour-8 ring) is OFFSET within the block in the direction the charger FACES -- so per-charger
        # facing is readable from the grid (calibrated vs tu93 L3: 15 offset in col => faces horizontal /
        # charges along its row; offset in row => faces vertical / charges along its column).
        hazard_center_color = None
        if hazard_colors and deaths:
            cnt = Counter()
            for g0, _a, _g1 in deaths:
                for c in hazard_colors:
                    cnt[c] += int((g0 == c).sum())
            hazard_center_color = min(cnt, key=cnt.get) if cnt else None
        hazard_fit = {
            "n_death_transitions": len(deaths),
            "hazard_colors": sorted(hazard_colors),
            "hazard_axis": hazard_axis,
            "charge_range": charge_range,
            "move_step": step,
            "hazard_center_color": hazard_center_color,
            "axis_votes": dict(axis_votes),
            "post_move_distances_at_death": sorted(post_dists),
        }
        return cls(
            displacement=base.displacement,
            avatar_colors=base.avatar_colors,
            bg_color=base.bg_color,
            floor_color=base.floor_color,
            wall_colors=base.wall_colors,
            goal_color=base.goal_color,
            door_color=base.door_color,
            fit_quality=base.fit_quality,
            hazard_colors=hazard_colors,
            hazard_axis=hazard_axis,
            charge_range=charge_range,
            lethal_mode=lethal_mode,
            hazard_center_color=hazard_center_color,
            hazard_fit=hazard_fit,
        )

    def _hazard_blobs(self, grid):
        return _blobs(_color_cells(grid, self.hazard_colors)) if self.hazard_colors else []

    def _charger_facing(self, grid, hy, hx):
        """The charger's FACING DIRECTION as a unit (fdy, fdx), read from its centre-marker offset within the
        block: the marker is offset in the direction the charger faces (and charges). e.g. tu93 a marker
        offset +column => faces right (charges along its row, to the right); +row => faces down. Returns None
        if no marker found (then 'omni' falls back to all directions for that charger)."""
        if self.hazard_center_color is None:
            return None
        g = np.asarray(grid)
        ys, xs = np.where(g == self.hazard_center_color)
        near = [(int(y), int(x)) for y, x in zip(ys, xs) if abs(y - hy) <= 4 and abs(x - hx) <= 4]
        if not near:
            return None
        my, mx = near[0]
        ody, odx = my - hy, mx - hx
        if abs(odx) >= abs(ody):
            return (0, 1 if odx > 0 else -1)  # faces along the row (horizontal charger)
        return (1 if ody > 0 else -1, 0)  # faces along the column (vertical charger)

    def is_lethal(self, grid, action):
        """A nav move is lethal if it leaves the avatar ON a hazard's charge line (perpendicular offset within
        align_tol of the hazard's row/col) AND within charge_range along that line -> the charger intercepts
        and removes the avatar. The approach DIRECTION does not matter: the avatar can enter the lethal zone by
        moving along the line toward the hazard OR by stepping perpendicularly ONTO the line (tu93 L3's
        vertical chargers kill on a perpendicular step-on, which a 'must move toward' rule wrongly missed)."""
        if not self.hazard_colors or self.hazard_axis is None or self.charge_range <= 0:
            return False
        a = int(action)
        if a not in self.displacement:
            return False
        bb = self._avatar_bbox(grid)
        if bb is None:
            return False
        dy, dx = self.displacement[a]
        bcy, bcx = (bb[0] + bb[2]) / 2, (bb[1] + bb[3]) / 2  # avatar centre BEFORE the move
        acy, acx = bcy + dy, bcx + dx  # avatar centre AFTER the move
        g = np.asarray(grid)
        tol, rng = self.align_tol, self.charge_range
        for hy, hx, _sz in self._hazard_blobs(grid):
            on_row = abs(acy - hy) <= tol  # avatar ends ALIGNED on the charger's row
            on_col = abs(acx - hx) <= tol  # avatar ends ALIGNED on the charger's column
            if self.lethal_mode == "omni":
                # FACING-AWARE (calibrated vs tu93 L3's BFS ground truth): each charger kills only when the
                # avatar's DESTINATION is on the charger's facing line (aligned on the perpendicular axis),
                # on the SIDE it faces, within reach -- regardless of the avatar's own approach direction (a
                # perpendicular step ONTO the line is just as lethal; collision is the zero-distance case).
                # The facing DIRECTION (not just axis) is read from the centre-marker offset; using the
                # signed direction (a charger does not kill what is behind it) is what stops the over-pruning
                # that forbade the BFS win path.
                # NB: landing exactly ON a charger (distance 0) is NOT lethal in tu93 L3 (it defeats/passes
                # the charger) -- every observed death is a charge INTERCEPTION at distance 1..reach on the
                # charger's facing side, so the lethal zone is strictly 0 < dist <= reach.
                f = self._charger_facing(grid, hy, hx)
                if f is None:
                    lethal = (on_row and 0 < abs(acx - hx) <= rng) or (
                        on_col and 0 < abs(acy - hy) <= rng
                    )
                elif f[0] == 0:  # horizontal charger (faces left/right)
                    lethal = on_row and 0 < (acx - hx) * f[1] <= rng
                else:  # vertical charger (faces up/down)
                    lethal = on_col and 0 < (acy - hy) * f[0] <= rng
            else:
                # 'toward' (single learned axis): charge only on an along-axis approach (tu93 L2's horizontal
                # charger tolerates the perpendicular step-on / on-line-not-approaching cases).
                if self.hazard_axis == "row":
                    lethal = on_row and (hx - bcx) * dx > 0 and abs(acx - hx) <= rng
                else:
                    lethal = on_col and (hy - bcy) * dy > 0 and abs(acy - hy) <= rng
            if lethal and self._charge_unobstructed(g, hy, hx, acy, acx):
                return True
        return False

    def _charge_unobstructed(self, g, hy, hx, acy, acx):
        """The charger can only intercept if its straight charge along the axis to the avatar is NOT blocked
        by a wall (line-of-sight). Without this, the model forbids on-line-within-range cells that are
        actually SAFE because a wall shields them -- the over-pruning that left tu93 L3 with no plan."""
        if not self.wall_colors:
            return True
        H, W = g.shape
        if self.hazard_axis == "row":
            r = int(round((hy + acy) / 2))
            c0, c1 = sorted((int(round(hx)), int(round(acx))))
            seg = g[max(0, r), max(0, c0 + 1) : c1] if 0 <= r < H else np.array([])
        else:
            c = int(round((hx + acx) / 2))
            r0, r1 = sorted((int(round(hy)), int(round(acy))))
            seg = g[max(0, r0 + 1) : r1, max(0, c)] if 0 <= c < W else np.array([])
        return not bool(np.any(np.isin(seg, list(self.wall_colors))))

    def engine(self, grid, action, data=None):
        if self.is_lethal(grid, action):
            # the charger intercepts and REMOVES the avatar -> erase it (an avatar-less, dead-end grid that
            # plan_in_model will never route through to the goal). This is the capability the nav model lacks.
            g = np.asarray(grid).copy()
            for c in self.avatar_colors:
                g[g == c] = self.floor_color
            return g
        return super().engine(grid, action, data)


def _norm(t):
    """Normalize a transition into (grid, action, next_grid, level_before, level_after)."""
    if hasattr(t, "grid"):
        return (
            np.asarray(t.grid),
            int(t.action),
            np.asarray(t.next_grid),
            getattr(t, "level_before", None),
            getattr(t, "level_after", None),
        )
    if len(t) >= 5:
        return (np.asarray(t[0]), int(t[1]), np.asarray(t[2]), t[3], t[4])
    return (np.asarray(t[0]), int(t[1]), np.asarray(t[2]), None, None)
