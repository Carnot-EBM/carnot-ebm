"""Convention-perturbation transforms for the shipped-lever transfer measurement.

WHY THIS EXISTS
---------------
The two levers flipped ON on 2026-07-25 (frontier tier discipline; HUD edge-bar mask) are
GAME-AGNOSTIC mechanisms -- they contain no per-game table and never see a game id.  So the
way they can fail on a hidden game is NOT a mis-fitted per-game constant; it is a CONVENTION
they silently assume that the 25 public games happen to share and a hidden game need not.

Each lever has exactly one such convention, readable straight off its source:

  * FRONTIER TIER PREDICATE (arc_graph_explore._tier_ordered_click_points):
        salient = color in _TIER_SALIENT_COLORS   # frozenset(range(6, 16))
    i.e. "interesting objects are painted in the upper half of the palette".  This is an
    ABSOLUTE-COLOUR convention.  Nothing enforces it in the ARC spec.

  * HUD EDGE-BAR DETECTOR (arc_hud_bar_detector.is_edge_bar_like):
        on_top = y1 < tol ... on_right = x0 > width - 1 - tol   # tol = 2
    i.e. "the status readout hugs a frame edge".  This is an EDGE-ADJACENCY convention.

So we build one perturbation per convention and measure, per game, how much of the
convention each perturbation actually destroys (the DOSE).  A perturbation with zero dose on
a lever is INERT for that lever, and a flat result under an inert perturbation is not
evidence of transfer -- it is evidence of nothing.  Reporting the dose alongside the result
is what stops this from repeating the project's failure mode #2 (a metric that cannot
causally depend on the intervention).

Both transforms are MECHANIC-PRESERVING observation transforms in the sense
arc_variant_generator already established: the REAL env keeps the true win/level logic and
`levels_completed` passes through untouched, so a level banked under a perturbation is a
real level.  Only what the agent SEES changes.

IMPLEMENTATION NOTE (why monkeypatching and not a new env class)
----------------------------------------------------------------
`arc_leaderboard_eval.run_game` already wires `VariantEnv(env, game, variant, reflect=...)`,
and `VariantEnv` resolves `color_permutation` / `reflect_grid` /
`remap_click_for_reflection` as MODULE globals at call time.  Patching those three names
gives us new conditions through the existing, already-tested variant plumbing (frame stack
handling, gridless-terminal-frame passthrough, click remap) instead of a parallel
implementation that could diverge from it.  Sentinel codes keep the original behaviour for
every variant/reflect value the existing harness uses, so nothing already measured changes.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]

# Sentinel codes.  Chosen far outside anything the existing harness passes (it uses
# variant in {0, 1} and reflect in {None, 0}).
VARIANT_SALIENCE_INVERSION = 901
VARIANT_IDENTITY_COLOR = 902  # colour untouched; used by the geometry-only condition
REFLECT_DIAG_ROLL = 903

ROLL_K = 3  # > EDGE_BAR_EDGE_TOLERANCE (2), so an edge-hugging bar leaves the edge

# DOSE-PARAMETERISED roll sentinels, added 2026-07-25 after the recorded run measured that
# the k=3 roll is not a "mild" perturbation at all: it takes the pre-flip control from 7 wins
# to 1 and takes the number of games no arm can win from 11/25 to 18/25 (see the dose-ceiling
# fields in the analysis).  A perturbation that razes 86% of the control's capability
# auto-falsifies any lever whose support is a couple of games, independently of that lever's
# mechanism -- so the roll needs a DOSE AXIS, not just a single magnitude, before a flat
# result under it can be attributed to a broken convention.
#
# Encoding: sentinel 910 + k selects roll magnitude k (911 -> k=1, ... 919 -> k=9).  903 is
# left alone so every already-recorded C2 row stays byte-reproducible.
REFLECT_DIAG_ROLL_K_BASE = 910
ROLL_K_MIN, ROLL_K_MAX = 1, 9


def reflect_code_for_roll_k(k: int) -> int:
    """The `reflect` sentinel that selects a roll of magnitude k."""

    k = int(k)
    if not ROLL_K_MIN <= k <= ROLL_K_MAX:
        raise ValueError(f"roll magnitude {k} outside the sentinel range")
    return REFLECT_DIAG_ROLL_K_BASE + k


def _roll_k_of_code(code: int) -> int | None:
    """k if `code` is a dose-parameterised roll sentinel, else None."""

    c = int(code)
    if REFLECT_DIAG_ROLL_K_BASE + ROLL_K_MIN <= c <= REFLECT_DIAG_ROLL_K_BASE + ROLL_K_MAX:
        return c - REFLECT_DIAG_ROLL_K_BASE
    return None

_N_COLORS = 16
_SALIENT = frozenset(range(6, 16))  # must match arc_graph_explore._TIER_SALIENT_COLORS


# ---------------------------------------------------------------------------
# C1 -- salience inversion (targets the FRONTIER lever's absolute-colour convention)
# ---------------------------------------------------------------------------


def _palette_of(game_id: str, arc=None, scorecard_id=None) -> list[int]:
    """The distinct colours present in the game's reset frame.

    Read from the game's OWN observation -- this is dev-side diagnosis of the environment,
    not knowledge handed to the agent.  The agent never sees this map; it only ever sees the
    already-transformed frame, exactly as it would see a hidden game's native palette.
    """

    from carnot.agentic import arc_solver_kit as kit

    if arc is None:
        arc = kit.offline_arcade()
        scorecard_id = arc.open_scorecard()
    env = arc.make(game_id, scorecard_id=scorecard_id)
    frame = env.reset()
    stack = np.array(frame.frame)
    if stack.ndim == 2:
        stack = stack[None, ...]
    return sorted({int(c) for c in np.unique(stack)})


def salience_inverting_map(game_id: str, palette: list[int]) -> np.ndarray:
    """A deterministic INVOLUTION that flips as much of the game's palette across the
    salience boundary as the palette permits.

    Greedy, deterministic pairing: pair the game's in-frame salient colours (ascending) with
    its in-frame non-salient non-background colours (ascending) and swap each pair.  Every
    swapped pair moves BOTH its members across the {6..15} boundary, so the dose is maximal
    for a bijection restricted to the colours the game actually uses.

    A bijection can never invert salience completely when a game uses more salient colours
    than non-salient ones (m0r0 uses {5, 10, 11, 12} -- one non-salient colour for three
    salient ones), which is why the per-game dose is measured and reported rather than
    assumed.  Background 0 is held fixed so "emptiness" stays invariant, matching the
    existing colour-permutation transform's contract.
    """

    perm = np.arange(_N_COLORS, dtype=np.uint8)
    salient = [c for c in palette if c in _SALIENT]
    plain = [c for c in palette if c not in _SALIENT and c != 0]
    for a, b in zip(salient, plain):
        perm[a] = b
        perm[b] = a
    return perm


# ---------------------------------------------------------------------------
# C2 -- row roll (targets the HUD lever's edge-adjacency convention)
# ---------------------------------------------------------------------------


def roll_grid(grid: np.ndarray, k: int = ROLL_K) -> np.ndarray:
    """Cyclically shift the observed grid by k on BOTH axes.

    observed[r][c] == real[r - k][c - k].  Any bar hugging ANY edge moves off that edge:
    index 0,1 -> 3,4 and index 62,63 -> 1,2, and in both cases the Stage-1 predicate's
    `y1 < 2` / `y0 > 61` tests stop firing.  BOTH axes are rolled because a ROW-only roll is
    provably inert for a full-height VERTICAL bar -- and a first-pass row-only dose probe
    showed exactly that on r11l, sc25, lp85 and sb26, i.e. on the single game (r11l) whose
    win the HUD lever is credited with.  A perturbation that is inert on the game the lever
    wins would have produced an uninterpretable flat result.

    Clicks are inverse-mapped (below) so a click on what the agent sees lands on the
    intended real cell.  A roll -- unlike the existing reflection condition -- COMMUTES with
    direction, so an unremapped directional move action still moves the agent the way the
    agent intends; the only structural damage in the OBSERVATION is that objects straddling
    the wrap seam are seen as two pieces.

    2026-07-25 CORRECTION -- this docstring used to conclude from the paragraph above that the
    roll "makes it a milder mechanic perturbation than reflection".  The recorded run
    MEASURED the opposite and the claim is withdrawn: at k=3 the pre-flip control drops from
    7 wins to 1 (86% of its capability), and the number of games no arm can win rises from
    11/25 to 18/25.  Whatever the argument from commutativity suggests, the roll as
    instantiated is a corpus-razing perturbation, not a mild one.  Two consequences, both now
    handled rather than reasoned away: (i) any lever whose gain rests on a couple of games is
    auto-falsified under it for reasons that have nothing to do with that lever's convention,
    so the analysis stamps a DOSE_SATURATED marker instead of reporting a retention ratio;
    and (ii) `k` is now a measurable dose axis (see `reflect_code_for_roll_k`) so the
    magnitude at which the corpus dies can be separated from the magnitude at which the HUD
    predicate stops firing, rather than assumed to be far apart.
    """

    g = np.asarray(grid)
    return np.roll(np.roll(g, k, axis=0), k, axis=1).copy()


def unroll_click(x: int, y: int, width: int = 64, height: int = 64, k: int = ROLL_K):
    """Inverse of `roll_grid` for a click: observed (x, y) is real ((x-k)%W, (y-k)%H)."""

    return (int((int(x) - k) % int(width)), int((int(y) - k) % int(height)))


# ---------------------------------------------------------------------------
# Patch installation
# ---------------------------------------------------------------------------

_ORIG: dict = {}
_MAPS: dict = {}


def install(game_palettes: dict[str, list[int]]) -> None:
    """Install the sentinel-code handlers into arc_variant_generator.

    Every non-sentinel (variant, reflect) value is delegated to the original function
    untouched, so the existing `recolor_negative_control` / `reflect_axis0` conditions and
    any already-recorded row remain byte-reproducible.
    """

    from carnot.agentic import arc_variant_generator as vg

    if _ORIG:
        return
    _ORIG["color_permutation"] = vg.color_permutation
    _ORIG["reflect_grid"] = vg.reflect_grid
    _ORIG["remap_click_for_reflection"] = vg.remap_click_for_reflection

    for gid, pal in game_palettes.items():
        _MAPS[gid] = salience_inverting_map(gid, pal)

    def color_permutation(game_id: str, variant: int) -> np.ndarray:
        if int(variant) == VARIANT_SALIENCE_INVERSION:
            return _MAPS[game_id]
        if int(variant) == VARIANT_IDENTITY_COLOR:
            return np.arange(_N_COLORS, dtype=np.uint8)
        return _ORIG["color_permutation"](game_id, variant)

    def reflect_grid(grid, axis: int = 1):
        if int(axis) == REFLECT_DIAG_ROLL:
            return roll_grid(grid)
        k = _roll_k_of_code(axis)
        if k is not None:
            return roll_grid(grid, k=k)
        return _ORIG["reflect_grid"](grid, axis=axis)

    def remap_click_for_reflection(x: int, y: int, w: int, h: int, axis: int = 1):
        if int(axis) == REFLECT_DIAG_ROLL:
            return unroll_click(x, y, width=w, height=h)
        k = _roll_k_of_code(axis)
        if k is not None:
            return unroll_click(x, y, width=w, height=h, k=k)
        return _ORIG["remap_click_for_reflection"](x, y, w, h, axis=axis)

    vg.color_permutation = color_permutation
    vg.reflect_grid = reflect_grid
    vg.remap_click_for_reflection = remap_click_for_reflection
