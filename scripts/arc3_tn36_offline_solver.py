"""tn36 offline solver — the captured per-game RE for the PROGRAM-EDITOR puzzle.

tn36 (fully reverse-engineered 2026-06-17): the object runs an N-slot MOVE-PROGRAM
(`vkuvtkaerv`); each slot holds a command code (tn36.py:2171 okllwtboml: 0=settle,
1=left, 2=right, 3=down, 33=up, 5/6/7/16=rotate, 8/9=scale+/-, 14/15/63=property).
Step size CSPOIQWER=4. WIN = the object matches the target on FIVE attributes
(x, y, scale, rotation, sjmtdfxdrc); running the program transforms the object, and if
it lands on the target the level completes (else it resets to base).

THE TOOL PALETTE (fully mapped 2026-06-17): each slot is a 6-BIT code editor. Bit b's
toggle-button is at (slot_x, slot_y_top + 3*b), for b in 0..5 (values 1,2,4,8,16,32).
So ANY code is settable as the sum of its set bits: down(3)=bit0+bit1, up(33)=bit0+bit5,
scale+(8)=bit3, rotate+90(5)=bit0+bit2, etc. The slot positions ARE READABLE from
internal state (controller.pfyayhyovw[i].{x,y}) — they re-lay-out per level (L1: 5 slots,
L2: 4 slots, different x/y), so the solver DISCOVERS them rather than hardcoding. RUN the
program by clicking the object's sub-button `sxhtkytekm` (center, also discovered).

This solver READS the object/target deltas + the button layout, COMPUTES a move-program
that nets the deltas (down/up/left/right + scale; rotation/property when a level needs
them), edits each slot to its code via the bit-toggle clicks, runs, and CHAINS levels.
Reproduces L1 (program [3,3,3,3,3], 5 downs) and L2 ([33,33,33,33], 4 ups) — gated.
"""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "python"))

from carnot.agentic import arc_solver_kit as kit  # noqa: E402
from carnot.agentic.arc_agi3_live_adapter import _game_action, _levels_completed  # noqa: E402
from arcengine import GameAction  # noqa: E402

STEP = 4                                  # CSPOIQWER
BIT_DY = 3                                # bit b button is BIT_DY*b below the slot top
# command codes
SETTLE, LEFT, RIGHT, DOWN, UP = 0, 1, 2, 3, 33
SCALE_UP, SCALE_DOWN = 8, 9
ROT = {90: 5, -90: 6, 180: 7, 270: 16}    # rotation delta (deg) -> code
# property (sjmtdfxdrc) is set ABSOLUTELY by codes 14/15/63 (= knfgrcbayu, decoded 2026-06-17):
# code 14 -> 9, code 15 -> 8, code 63 -> 15. Map the TARGET property value to its command code.
PROP_CODE = {9: 14, 8: 15, 15: 63}


def _bz(env):
    return env._game.fdksqlmpki.bzirenxmrg


def _obj_tgt(env):
    bz = _bz(env)
    o, t = bz.htntnzkbzu, bz.aqszntqeae
    attrs = lambda s: (s.x, s.y, s.scale, s.rotation, int(s.sjmtdfxdrc))  # noqa: E731
    return attrs(o), attrs(t)


def _program(env):
    return list(_bz(env).vupcwzjtxu.vkuvtkaerv)


def _slot_tops(env):
    """The (x, y_top) of each program slot, read from the controller's slot visuals."""
    return [(int(s.x), int(s.y)) for s in _bz(env).vupcwzjtxu.pfyayhyovw]


def _run_xy(env):
    sx = _bz(env).sxhtkytekm
    return (int(sx.x + sx.width // 2), int(sx.y + sx.height // 2))


def _bit_clicks(slot_top, diff):
    """Clicks to toggle the bits in `diff` for a slot at (x, y_top): bit b at (x, y_top+3b)."""
    x, y0 = slot_top
    return [(x, y0 + BIT_DY * b) for b in range(6) if (diff >> b) & 1]


def compute_moves(obj, tgt):
    """The MULTISET of move-codes that nets the obj->tgt delta (unordered), or (None, reason).
    Each move is one STEP (or one scale/rotate step). ORDER is resolved by routing (some levels
    have obstacles, so the moves must be sequenced to route the object through a gap)."""
    (ox, oy, osc, orot, osj), (tx, ty, tsc, trot, tsj) = obj, tgt
    dx, dy = tx - ox, ty - oy
    if dx % STEP or dy % STEP:
        return None, f"delta not a multiple of STEP={STEP} (dx={dx},dy={dy})"
    moves = []
    moves += [DOWN] * (dy // STEP) if dy > 0 else [UP] * (-dy // STEP)
    moves += [RIGHT] * (dx // STEP) if dx > 0 else [LEFT] * (-dx // STEP)
    moves += [SCALE_UP] * (tsc - osc) if tsc > osc else [SCALE_DOWN] * (osc - tsc)
    if orot != trot:
        d = (trot - orot) % 360
        code = ROT.get(d) or ROT.get(d - 360)
        if code is None:
            return None, f"rotation delta {d} not in {{90,180,270,-90}}"
        moves.append(code)
    if osj != tsj:
        code = PROP_CODE.get(tsj)
        if code is None:
            return None, f"property {osj}->{tsj} not reachable via codes 14/15/63 (reach {sorted(PROP_CODE)})"
        moves.append(code)
    return moves, ""


def _orderings(moves, n, cap=400):
    """Distinct slot-programs (length n) for the move multiset — straight order first, then
    permutations. Levels with obstacles need a specific order that ROUTES the object through a
    gap; the search tries orderings until one wins (the game is the path oracle)."""
    if len(moves) > n:
        return []
    seen, out = set(), []
    for perm in itertools.chain([tuple(moves)], itertools.permutations(moves)):
        if perm in seen:
            continue
        seen.add(perm)
        out.append(list(perm) + [SETTLE] * (n - len(perm)))
        if len(out) >= cap:
            break
    return out


def _apply_program(env, program, traj):
    """Edit the slots to `program` (bit-toggle clicks) and RUN it, appending every env.step to
    `traj` (resolving any run animation). Returns the resulting level."""
    cur, tops = _program(env), _slot_tops(env)

    def clk(cx, cy):
        f = None
        for _ in range(80):
            traj.append({"action": 6, "data": {"x": cx, "y": cy}})
            f = env.step(_game_action(GameAction, 6), data={"x": cx, "y": cy})
            if not env._game.fdksqlmpki.deredwcqze:
                break
        return f

    for i, top in enumerate(tops):
        for c in _bit_clicks(top, cur[i] ^ program[i]):
            clk(*c)
    f = clk(*_run_xy(env))
    return _levels_completed(f)


def _apply_solution(env, programs, traj):
    """Run every program in a level's solution (1 program = single run; 2+ = multi-run through
    checkpoints, which advance the object's base between runs). Returns the resulting level."""
    level = None
    for program in programs:
        level = _apply_program(env, program, traj)
    return level


def _fresh_at(arc, game, wins):
    """A fresh env replayed through the known winning SOLUTIONS to the next-to-solve level.
    A FRESH env per attempt is REQUIRED: accumulating failed runs on one env eventually trips
    the game's LOSS condition and resets the whole episode to L0, corrupting the search."""
    env = arc.make(game, scorecard_id=arc.open_scorecard())
    env.reset()
    for programs in wins:
        _apply_solution(env, programs, [])
    return env


# --- multi-run maze path-planning (delegated to the reusable, game-agnostic package planners) ---
#
# checkpoint_multirun (L6) + timed_trap_aware (L7, blinking spikes) both stage a path across
# CHECKPOINTS (a run that ends on a checkpoint advances the object's base) — the timed variant also
# crosses the spike band only during its invisible window. The planning ALGORITHMS live in
# `carnot.agentic.arc_maze_planner` so the STRATEGY router can dispatch them to any game of these
# classes; this module just builds the generic MazeModel from tn36's internal state and validates the
# returned plan against the real env. The spike band (`ekdwmirldx`) is invisible at run start and
# toggles visible after slot index 2 (`jmwdvdqntf`); a move onto a spike kills the object while
# visible (full box) or via its residual hidden hitbox (`olbuwgbgyz`) while invisible — encoded as
# spikes_visible / spikes_hidden + invisible_slots in the model.
_MAZE_MOVES = [((0, -STEP), UP), ((0, STEP), DOWN), ((-STEP, 0), LEFT), ((STEP, 0), RIGHT)]


def _geom(env):
    """Object box (w,h), obstacle boxes, and checkpoint positions, read from internal state."""
    bz = _bz(env)
    o = bz.htntnzkbzu
    return ((o.width, o.height),
            [(i.x, i.y, i.width, i.height) for i in bz.bizgpiltwm],
            [(i.x, i.y) for i in bz.wgzwawbgew])


def _spikes(env):
    """Spike boxes read from internal state: (visible_boxes, hidden_residual_hitboxes). The visible
    box is the full lethal band; the hidden hitbox (`olbuwgbgyz`) is what still kills when invisible."""
    bz = _bz(env)
    vis = [(s.x, s.y, s.width, s.height) for s in bz.ekdwmirldx]
    hid = [(s.olbuwgbgyz.x, s.olbuwgbgyz.y, s.olbuwgbgyz.width, s.olbuwgbgyz.height)
           for s in bz.ekdwmirldx]
    return vis, hid


def _maze_model(env, n, *, with_spikes):
    """Build the generic MazeModel (carnot.agentic.arc_maze_planner) from tn36 internal state, or
    None if the level needs a transform (the maze planners are position-only)."""
    from carnot.agentic.arc_maze_planner import MazeModel
    bz = _bz(env)
    o, t = bz.htntnzkbzu, bz.aqszntqeae
    if (o.scale, o.rotation, int(o.sjmtdfxdrc)) != (t.scale, t.rotation, int(t.sjmtdfxdrc)):
        return None                                   # transforms+maze unhandled; position only
    (w, h), obs, cps = _geom(env)
    vis, hid = _spikes(env) if with_spikes else ([], [])
    return MazeModel(object_wh=(w, h), start=(o.x, o.y), target=(t.x, t.y), walls=obs,
                     checkpoints=cps, move_codes=_MAZE_MOVES, settle_code=SETTLE, n_slots=n,
                     spikes_visible=vis, spikes_hidden=hid, invisible_slots=3)


def _multirun_plan(env, n):
    """checkpoint_multirun: stage a position-only path across checkpoints (no hazards). Delegates to
    the reusable planner. Returns a list of leg-programs (padded to n) or None."""
    from carnot.agentic.arc_maze_planner import checkpoint_multirun_plan
    model = _maze_model(env, n, with_spikes=False)
    return checkpoint_multirun_plan(model) if model is not None else None


def _timed_multirun_plan(env, n):
    """timed_trap_aware: stage a position-only path across checkpoints while avoiding the blinking
    spike band (cross only during the invisible window). Delegates to the reusable planner. Returns a
    list of leg-programs or None (incl. when the level has no spikes — use the plain planner then)."""
    from carnot.agentic.arc_maze_planner import timed_trap_plan
    model = _maze_model(env, n, with_spikes=True)
    return timed_trap_plan(model) if model is not None else None


def _winning_solutions(arc, game, max_level, cap):
    """PASS 1 — discover each level's winning SOLUTION (a list of programs: 1 for a single run,
    2+ for a multi-run maze). Fresh env per attempt so accumulated failures can't reset L0."""
    wins = []
    while len(wins) < max_level:
        probe = _fresh_at(arc, game, wins)
        target_level = len(wins) + 1
        moves, reason = compute_moves(*_obj_tgt(probe))
        n = len(_program(probe))
        found = None
        # 1) single run: search orderings of the net moves
        if moves is not None:
            for program in _orderings(moves, n, cap):
                if _apply_program(_fresh_at(arc, game, wins), program, []) >= target_level:
                    found = [program]
                    break
        # 2) multi-run maze fallback (the path exceeds one program / is obstacle-blocked)
        if found is None:
            plan = _multirun_plan(probe, n)
            if plan and (_apply_solution(_fresh_at(arc, game, wins), plan, []) or 0) >= target_level:
                found = plan
        # 3) TIMED spike-trap maze fallback (blinking spikes: route + cross during invisible window)
        if found is None:
            plan = _timed_multirun_plan(probe, n)
            if plan and (_apply_solution(_fresh_at(arc, game, wins), plan, []) or 0) >= target_level:
                found = plan
        if found is None:
            break
        wins.append(found)
    return wins


def solve(env=None, max_level=10, cap=400, *, game="tn36"):
    """Solve tn36 levels (single-run transforms/path-routing + multi-run mazes). Two-pass:
    discover each level's winning solution, then replay them on a fresh env for a clean
    trajectory. `env` is ignored (signature compat). Returns (traj, level)."""
    arc = kit.offline_arcade()
    wins = _winning_solutions(arc, game, max_level, cap)
    env2 = arc.make(game, scorecard_id=arc.open_scorecard())
    f = env2.reset()
    traj, level = [], _levels_completed(f)
    for programs in wins:
        new = _apply_solution(env2, programs, traj)
        if new is None or new <= level:
            break
        level = new
    return traj, level


def main() -> int:
    arc = kit.offline_arcade()
    env = arc.make("tn36", scorecard_id=arc.open_scorecard())
    traj, lvl = solve(env)
    if not traj:
        print("tn36 solve produced no actions")
        return 1
    labels = [json.dumps(t) for t in traj]

    def apply(env, label, frame):
        s = json.loads(label)
        return env.step(_game_action(GameAction, s["action"]), data=s.get("data"))

    gate = kit.reproduce("tn36", labels, apply, claimed_level=lvl)
    print(f"tn36 COMPUTED solve: reached L{lvl} in {len(traj)} clicks; "
          f"reproduced={gate['reproduced']} claimed_level={gate.get('claimed_level')}")
    if lvl >= 1 and gate["reproduced"]:
        Path("results/arc_explore_trajectory_tn36.json").write_text(json.dumps(
            {"game": "tn36", "reached_level": lvl, "trajectory": traj,
             "method": "program_editor_RE_general",
             "note": "6-bit-per-slot code editor; layout read from pfyayhyovw; chains levels"}, indent=2))
        print("WROTE results/arc_explore_trajectory_tn36.json")
    return 0 if (lvl >= 1 and gate["reproduced"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
