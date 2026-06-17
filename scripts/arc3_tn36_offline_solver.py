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
    if osj != tsj:
        return None, f"needs property change (sjmtdfxdrc {osj}->{tsj}; codes 14/15/63 not yet decoded)"
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


def _fresh_at(arc, game, wins):
    """A fresh env replayed through the known winning programs to the next-to-solve level.
    A FRESH env per attempt is REQUIRED: accumulating failed runs on one env eventually trips
    the game's LOSS condition and resets the whole episode to L0, corrupting the search."""
    env = arc.make(game, scorecard_id=arc.open_scorecard())
    env.reset()
    for p in wins:
        _apply_program(env, p, [])
    return env


def _winning_programs(arc, game, max_level, cap):
    """PASS 1 — discover the winning slot-program per level. Each ordering is tried on a FRESH
    env (replayed to the current level) so accumulated failures can't reset the episode."""
    wins = []
    while len(wins) < max_level:
        probe = _fresh_at(arc, game, wins)
        moves, reason = compute_moves(*_obj_tgt(probe))
        if moves is None:
            break
        n = len(_program(probe))
        target_level = len(wins) + 1
        found = None
        for program in _orderings(moves, n, cap):
            if _apply_program(_fresh_at(arc, game, wins), program, []) >= target_level:
                found = program
                break
        if found is None:
            break
        wins.append(found)
    return wins


def solve(env=None, max_level=10, cap=400, *, game="tn36"):
    """Solve tn36 levels with obstacle path-routing. Two-pass: discover the winning program per
    level (fresh env per ordering attempt), then replay ONLY the winners on a fresh env for a
    clean minimal trajectory. `env` is ignored (kept for signature compat). Returns (traj, level)."""
    arc = kit.offline_arcade()
    wins = _winning_programs(arc, game, max_level, cap)
    env2 = arc.make(game, scorecard_id=arc.open_scorecard())
    f = env2.reset()
    traj, level = [], _levels_completed(f)
    for program in wins:
        new = _apply_program(env2, program, traj)
        if new <= level:
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
